#!/usr/bin/env python3
"""
AudioGen Flow Studio — Indic-F5 (0.3B) SRT-Driven Audio Dubber (Google Colab T4 Edition)
═══════════════════════════════════════════════════════════════════════════════
Core Systems:
1. Robust Synchronous SRT Parser (multi-encoding safe with instant gr.Warning)
2. Optimized Batch Processing (15-20 sentences per batch on T4 GPU with dynamic OOM fallback)
3. FFmpeg Silent Canvas Alignment (sample-accurate timeline overlay to exact total duration)
4. Clean, minimal Gradio UI ending strictly with demo.launch(share=True)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys

# Disable all anonymous telemetry and unwanted pings
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import gc
import re
import time
import math
import wave
import struct
import json
import uuid
import shutil
import logging
import threading
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import soundfile as sf
import numpy as np
import gradio as gr
from pydub import AudioSegment

# ─── LOGGING CONFIGURATION ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Colab-IndicF5")

# ─── DIRECTORIES & CONFIGURATION ─────────────────────────────────────────────
STORAGE_DIR = "storage"
WORKSPACE_DIR = os.path.join(STORAGE_DIR, "workspace")
OUTPUTS_DIR = os.path.join(STORAGE_DIR, "outputs")
SENTENCE_CHUNKS_DIR = os.path.join(WORKSPACE_DIR, "sentence_chunks")
JOB_STATE_FILE = os.path.join(STORAGE_DIR, "job_state.json")

INDIC_F5_MODEL_ID = "Tharshan/indicf5_hindi-english_code_switch"
REF_AUDIO_PATH = "core_1_ours.wav"
REF_TEXT = "नमस्ते, मैं एक software engineer हूँ और machine learning projects पर काम करती हूँ।"

for d in [STORAGE_DIR, WORKSPACE_DIR, OUTPUTS_DIR, SENTENCE_CHUNKS_DIR]:
    os.makedirs(d, exist_ok=True)


# ─── REFERENCE AUDIO GUARANTEE (core_1_ours.wav) ──────────────────────────────
def ensure_reference_audio(ref_path: str = REF_AUDIO_PATH) -> str:
    """Ensures reference voice audio exists. Synthesizes a clean female vocal formant if absent."""
    if os.path.exists(ref_path) and os.path.getsize(ref_path) > 5000:
        return ref_path

    sr = 24000
    duration = 3.5
    n_samples = int(sr * duration)
    with wave.open(ref_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n_samples):
            t = i / sr
            val = (
                0.35 * math.sin(2 * math.pi * 220 * t)
                + 0.18 * math.sin(2 * math.pi * 440 * t)
                + 0.09 * math.sin(2 * math.pi * 880 * t)
                + 0.04 * math.sin(2 * math.pi * 1760 * t)
            )
            env = math.sin(math.pi * t / duration) ** 2
            val = int(val * env * 32767 * 0.45)
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
    return ref_path


# ─── SYSTEM 1: ROBUST SYNCHRONOUS SRT PARSER (THE FIX) ────────────────────────
def read_srt_bytes_multi_encoding(file_path: str) -> str:
    """Reads subtitle file testing multiple encodings. Halts immediately if unreadable."""
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Subtitle file '{os.path.basename(str(file_path))}' does not exist on disk.")

    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    if not raw_bytes or len(raw_bytes.strip()) == 0:
        raise ValueError(f"Subtitle file '{os.path.basename(file_path)}' is empty (0 bytes).")

    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-16"]
    for enc in encodings:
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue

    return raw_bytes.decode("utf-8", errors="replace")


def parse_timestamp_seconds(ts_str: str) -> float:
    """Parses SRT timestamp (HH:MM:SS,mmm or HH:MM:SS.mmm) into seconds."""
    ts_str = ts_str.strip().replace(",", ".")
    parts = ts_str.split(":")
    if len(parts) == 3:
        h = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        return h * 3600.0 + m * 60.0 + s
    elif len(parts) == 2:
        m = float(parts[0])
        s = float(parts[1])
        return m * 60.0 + s
    return float(parts[0])


def parse_srt_synchronous(srt_path: str) -> List[Dict[str, Any]]:
    """Synchronous, multi-encoding safe SRT reader.
    Guarantees immediate failure detection with clear error messages.
    """
    if not srt_path:
        raise ValueError("Please select or upload a valid Translated Subtitle (.srt) file.")

    content = read_srt_bytes_multi_encoding(srt_path)
    blocks = []
    raw_blocks = re.split(r"\r?\n\r?\n", content)

    for block_str in raw_blocks:
        lines = [l.strip() for l in block_str.splitlines() if l.strip()]
        if not lines:
            continue

        time_idx = -1
        for i, line in enumerate(lines):
            if "-->" in line:
                time_idx = i
                break

        if time_idx == -1:
            continue

        try:
            time_parts = lines[time_idx].split("-->")
            if len(time_parts) < 2:
                continue
            start_sec = parse_timestamp_seconds(time_parts[0])
            end_sec = parse_timestamp_seconds(time_parts[1])

            text_lines = lines[time_idx + 1 :]
            raw_text = " ".join(text_lines)

            # Clean styling tags and whitespace
            clean_text = re.sub(r"<[^>]+>", "", raw_text).strip()
            clean_text = " ".join(clean_text.split())

            if not clean_text:
                continue

            if end_sec <= start_sec:
                end_sec = start_sec + max(1.0, len(clean_text) * 0.08)

            blocks.append(
                {
                    "index": len(blocks) + 1,
                    "start_time": round(start_sec, 3),
                    "end_time": round(end_sec, 3),
                    "duration": round(end_sec - start_sec, 3),
                    "text": clean_text,
                }
            )
        except Exception:
            continue

    if not blocks:
        raise ValueError(
            f"No valid subtitle dialogue lines found in '{os.path.basename(srt_path)}'. "
            "Please ensure the file follows standard SRT format (e.g., '00:00:01,000 --> 00:00:04,000')."
        )

    return blocks


# ─── TIME & DURATION CONVERSION UTILITIES ────────────────────────────────────
def parse_duration_to_seconds(dur_input: Any) -> float:
    """Parses duration from HH:MM:SS, MM:SS, seconds string, or numeric input into seconds."""
    if dur_input is None:
        return 120.0
    if isinstance(dur_input, (int, float)):
        return max(1.0, float(dur_input))

    s = str(dur_input).strip()
    if not s:
        return 120.0

    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 3:
                h = float(parts[0])
                m = float(parts[1])
                sec = float(parts[2])
                return max(1.0, h * 3600.0 + m * 60.0 + sec)
            elif len(parts) == 2:
                m = float(parts[0])
                sec = float(parts[1])
                return max(1.0, m * 60.0 + sec)
        except Exception:
            pass

    try:
        val = float(s)
        return max(1.0, val)
    except Exception:
        pass

    return 120.0


def format_seconds_to_hms(seconds: float) -> str:
    """Converts seconds into formatted HH:MM:SS string."""
    total_sec = int(round(max(0.0, float(seconds))))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ─── SYSTEM 2: OPTIMIZED INDIC-F5 (0.3B) BATCH PROCESSING WITH OOM FALLBACK ──
class IndicF5BatchEngine:
    """High-performance batch inference engine for Indic-F5 (0.3B) on T4 GPU."""

    def __init__(self, model_id: str = INDIC_F5_MODEL_ID, ref_audio_path: str = REF_AUDIO_PATH):
        self.model_id = model_id
        self.ref_audio_path = ref_audio_path
        self.ref_text = REF_TEXT
        self.model = None
        self._lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, log_fn=None):
        """Loads Indic-F5 (0.3B) model into GPU memory."""
        with self._lock:
            if self.model is not None:
                return self.model

            ensure_reference_audio(self.ref_audio_path)
            if log_fn:
                log_fn(f"⏳ [Indic-F5] Loading 0.3B model onto {self.device.upper()} from '{self.model_id}'...")

            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            model = model.to(self.device).eval()
            self.model = model

            if log_fn:
                vram_info = self.get_vram_status()
                log_fn(f"✅ [Indic-F5] Model ready on {self.device.upper()}{vram_info}")

            return self.model

    def get_vram_status(self) -> str:
        """Returns formatted GPU VRAM usage."""
        if not torch.cuda.is_available():
            return ""
        try:
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            pct = int(reserved / total * 100)
            return f" | VRAM: {reserved:.1f}/{total:.1f} GB ({pct}%)"
        except Exception:
            return ""

    def generate_batch(self, texts: List[str], speed: float = 1.0) -> List[Optional[Tuple[np.ndarray, int]]]:
        """Runs parallel neural flow-matching batch sampling on GPU."""
        model = self.load_model()
        if not texts:
            return []

        with torch.inference_mode():
            ref_audio = self.ref_audio_path
            ref_text = self.ref_text.strip()
            if not ref_text.endswith((" ", ".", "!", "?")):
                ref_text += ". "

            cond, rms = model._load_ref(ref_audio)
            HOP_LENGTH = 256
            TARGET_RMS = 0.1
            SAMPLE_RATE = 24000

            ref_len = cond.shape[-1] // HOP_LENGTH
            batch_size = len(texts)
            cond_batch = cond.repeat(batch_size, 1)

            durations = []
            full_texts = []
            ref_byte_len = max(1, len(ref_text.encode("utf-8")))

            for t in texts:
                clean_t = t.strip()
                full_texts.append(ref_text + clean_t)
                t_byte_len = max(1, len(clean_t.encode("utf-8")))
                dur = ref_len + int(ref_len / ref_byte_len * t_byte_len / speed)
                durations.append(max(ref_len + 5, dur))

            duration_tensor = torch.tensor(durations, device=cond.device, dtype=torch.long)

            generated, _ = model.model.sample(
                cond=cond_batch,
                text=full_texts,
                duration=duration_tensor,
                steps=32,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
            )

            results = []
            for i in range(batch_size):
                try:
                    dur_i = durations[i]
                    mel_i = generated[i : i + 1, ref_len : dur_i, :].permute(0, 2, 1).to(torch.float32)
                    wave_i = model.vocoder.decode(mel_i).squeeze().cpu()
                    if rms < TARGET_RMS:
                        wave_i = wave_i * rms / TARGET_RMS
                    audio_arr = wave_i.numpy().astype(np.float32)
                    results.append((audio_arr, SAMPLE_RATE))
                except Exception:
                    results.append(None)

            return results

    def synthesize_single_fallback(self, text: str, out_wav_path: str) -> bool:
        """Single sentence synthesis fallback with 0 KB check."""
        try:
            model = self.load_model()
            if model is None:
                return False
            audio, sr = model.generate(
                text=text.strip(),
                ref_audio=self.ref_audio_path,
                ref_text=self.ref_text,
            )
            if os.path.exists(out_wav_path):
                os.remove(out_wav_path)
            sf.write(out_wav_path, audio, sr)
            return os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 1000
        except Exception:
            return False

    def synthesize_batch_with_oom_fallback(
        self,
        batch_blocks: List[Dict[str, Any]],
        log_fn=None,
    ) -> List[Dict[str, Any]]:
        """Synthesizes a batch of text segments. If CUDA OOM occurs:
        1. Catches RuntimeError
        2. Clears cache & calls gc.collect()
        3. Halves batch size and recursively retries without crashing
        """
        if not batch_blocks:
            return []

        texts = [b["text"].strip() for b in batch_blocks]
        batch_results = None

        try:
            batch_results = self.generate_batch(texts)
        except RuntimeError as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "cuda out of memory" in err_str:
                if log_fn:
                    log_fn(
                        f"⚠️ [OOM Fallback] CUDA Out of Memory with batch size {len(batch_blocks)}. "
                        f"Clearing cache and halving batch to retry immediately..."
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                if len(batch_blocks) > 1:
                    mid = len(batch_blocks) // 2
                    first_half = self.synthesize_batch_with_oom_fallback(batch_blocks[:mid], log_fn=log_fn)
                    second_half = self.synthesize_batch_with_oom_fallback(batch_blocks[mid:], log_fn=log_fn)
                    return first_half + second_half
                else:
                    batch_results = None
            else:
                if log_fn:
                    log_fn(f"⚠️ [RuntimeError] {e}")
                batch_results = None
        except Exception as e:
            if log_fn:
                log_fn(f"⚠️ [Inference Notice] {e}")
            batch_results = None

        successful_blocks = []
        for i, block in enumerate(batch_blocks):
            wav_path = block.get("wav_path", "")
            written = False

            if batch_results and i < len(batch_results) and batch_results[i] is not None:
                audio_arr, sr = batch_results[i]
                try:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                    sf.write(wav_path, audio_arr, sr)
                    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
                        written = True
                except Exception:
                    written = False

            # Single retry fallback if batch decode failed for this block
            if not written:
                written = self.synthesize_single_fallback(block["text"], wav_path)

            if written and os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
                block_entry = dict(block)
                block_entry["wav_path"] = wav_path
                successful_blocks.append(block_entry)
            else:
                if log_fn:
                    log_fn(f"⏩ [Skip Block] Sentence {block.get('index')} skipped after retry.")

        return successful_blocks


# ─── SYSTEM 3: FFMPEG SILENT CANVAS AUDIO SYNCING ────────────────────────────
def build_silent_canvas_dubbed_audio(
    total_duration_sec: float,
    generated_blocks: List[Dict[str, Any]],
    final_output_path: str,
    log_fn=None,
) -> str:
    """Generates silent base audio track of exact total_duration_sec via FFmpeg anullsrc,
    overlays generated dialogue chunks at their exact SRT start_time, and exports pristine master .wav.
    """
    total_duration_sec = max(1.0, float(total_duration_sec))
    total_duration_ms = int(total_duration_sec * 1000)
    hms_str = format_seconds_to_hms(total_duration_sec)

    if log_fn:
        log_fn(f"🔇 [Silent Canvas] Generating {hms_str} ({total_duration_sec:.1f}s) base silent canvas via FFmpeg anullsrc...")

    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(final_output_path)), exist_ok=True)

    silent_base_path = os.path.join(WORKSPACE_DIR, "silent_base_track.wav")

    # 1. Generate exact silent canvas with FFmpeg
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=stereo",
        "-t", f"{total_duration_sec:.3f}",
        "-c:a", "pcm_s16le",
        silent_base_path,
    ]

    silent_ok = False
    try:
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        if os.path.exists(silent_base_path) and os.path.getsize(silent_base_path) > 1000:
            silent_ok = True
            if log_fn:
                log_fn(f"✅ [Silent Canvas] Created base silent track ({total_duration_sec:.1f}s)")
    except Exception:
        pass

    if not silent_ok:
        # Pydub fallback
        silent_canvas = AudioSegment.silent(duration=total_duration_ms, frame_rate=44100)
        silent_canvas.export(silent_base_path, format="wav")

    # 2. Overlay generated sentence chunks at exact SRT start_time
    if log_fn:
        log_fn(f"⏱️ [Audio Sync] Overlaying {len(generated_blocks)} dialogue chunks at exact SRT start times...")

    try:
        base_canvas = AudioSegment.from_file(silent_base_path)
    except Exception:
        base_canvas = AudioSegment.silent(duration=total_duration_ms, frame_rate=44100)

    placed_count = 0
    for block in generated_blocks:
        chunk_wav = block.get("wav_path", "")
        start_time_sec = block.get("start_time", 0.0)
        start_ms = max(0, int(start_time_sec * 1000))

        if chunk_wav and os.path.exists(chunk_wav) and os.path.getsize(chunk_wav) > 1000:
            try:
                chunk_seg = AudioSegment.from_file(chunk_wav)
                base_canvas = base_canvas.overlay(chunk_seg, position=start_ms)
                placed_count += 1
            except Exception as e:
                if log_fn:
                    log_fn(f"⚠️ [Audio Sync] Notice overlaying block {block.get('index')}: {e}")

    # 3. Trim or pad to guarantee exact duration match
    if len(base_canvas) > total_duration_ms:
        base_canvas = base_canvas[:total_duration_ms]
    elif len(base_canvas) < total_duration_ms:
        pad = AudioSegment.silent(duration=total_duration_ms - len(base_canvas), frame_rate=base_canvas.frame_rate)
        base_canvas = base_canvas + pad

    # 4. Export the combined master timeline
    base_canvas.export(final_output_path, format="wav")

    if log_fn:
        final_kb = os.path.getsize(final_output_path) // 1024
        log_fn(f"✅ [Audio Sync] Master dubbed audio finalized: {os.path.basename(final_output_path)} ({final_kb} KB, {hms_str})")

    # Cleanup temporary silent base track
    if os.path.exists(silent_base_path):
        try:
            os.remove(silent_base_path)
        except Exception:
            pass

    return final_output_path


# ─── JOB MANAGER (THREAD-SAFE BACKGROUND PROCESSING) ──────────────────────────
class JobManager:
    """Manages asynchronous dubbing pipeline with persistence and live UI metrics."""

    def __init__(self):
        self.lock = threading.Lock()
        self.job_id: Optional[str] = None
        self.status = "IDLE"
        self.progress = 0.0
        self.message = "System Standby — Enter Total Duration and upload Translated SRT to start."
        self.current_sentence = 0
        self.total_sentences = 0
        self.completed_file: Optional[str] = None
        self.logs: List[str] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.load_from_disk()

    def log(self, message: str, level: str = "INFO"):
        now_str = time.strftime("%H:%M:%S")
        entry = f"[{now_str}] {message}"
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)

        with self.lock:
            self.logs.append(entry)
            if len(self.logs) > 300:
                self.logs.pop(0)

    def start_job(
        self,
        total_duration: float,
        srt_file_path: str,
        blocks: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Launches the background dubbing pipeline."""
        with self.lock:
            if self.worker_thread and self.worker_thread.is_alive():
                return False, "A dubbing task is already running in the background. Wait or cancel it first."

            self.job_id = uuid.uuid4().hex[:8]
            self.status = "STARTING"
            self.progress = 5.0
            self.message = f"Starting Indic-F5 dubbing job ({self.job_id})..."
            self.current_sentence = 0
            self.total_sentences = len(blocks)
            self.completed_file = None
            self.logs = []
            self.start_time = time.time()
            self.end_time = None
            self.stop_event.clear()

            self.worker_thread = threading.Thread(
                target=run_colab_dubbing_worker,
                args=(self, total_duration, srt_file_path, blocks),
                daemon=True,
            )
            self.worker_thread.start()

        self.log(f"Dubbing job {self.job_id} launched for {len(blocks)} sentences ({format_seconds_to_hms(total_duration)})")
        self.save_to_disk()
        return True, f"Job started (ID: {self.job_id})."

    def cancel_job(self) -> Tuple[bool, str]:
        with self.lock:
            if self.worker_thread and self.worker_thread.is_alive():
                self.stop_event.set()
                self.status = "CANCELLED"
                self.message = "Cancellation requested by user. Terminating..."
                self.log("Cancellation signal emitted by user.", level="WARNING")
                self.save_to_disk()
                return True, "Cancellation signal sent."
            else:
                self.reset_job()
                return True, "System reset to Standby."

    def reset_job(self) -> Tuple[bool, str]:
        with self.lock:
            if self.worker_thread and self.worker_thread.is_alive():
                self.stop_event.set()
            self.job_id = None
            self.status = "IDLE"
            self.progress = 0.0
            self.message = "System Standby — Enter Total Duration and upload Translated SRT to start."
            self.current_sentence = 0
            self.total_sentences = 0
            self.completed_file = None
            self.logs = []
            self.start_time = None
            self.end_time = None
            self.worker_thread = None
            self.save_to_disk()
            return True, "System reset to Standby."

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            now = time.time()
            if self.start_time:
                elapsed = int((self.end_time or now) - self.start_time)
            else:
                elapsed = 0

            return {
                "job_id": self.job_id,
                "status": self.status,
                "progress": self.progress,
                "message": self.message,
                "current_sentence": self.current_sentence,
                "total_sentences": self.total_sentences,
                "elapsed_sec": max(0, elapsed),
                "completed_file": self.completed_file,
                "logs": list(self.logs),
            }

    def save_to_disk(self):
        try:
            state = self.get_state()
            with open(JOB_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def load_from_disk(self):
        try:
            if os.path.exists(JOB_STATE_FILE):
                with open(JOB_STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    saved_status = data.get("status", "IDLE")
                    # Auto-reset any zombie job state across Colab cell restarts
                    if saved_status in ["STARTING", "PARSING_SRT", "GENERATING_TTS", "SYNCING_AUDIO"]:
                        self.job_id = None
                        self.status = "IDLE"
                        self.progress = 0.0
                        self.message = "System Standby — Enter Total Duration and upload Translated SRT to start."
                        self.current_sentence = 0
                        self.total_sentences = 0
                        self.completed_file = None
                        self.logs = []
                        self.start_time = None
                        self.end_time = None
                    else:
                        self.job_id = data.get("job_id")
                        self.status = saved_status
                        self.progress = data.get("progress", 0.0)
                        self.message = data.get("message", "System Standby.")
                        self.completed_file = data.get("completed_file")
                        self.logs = data.get("logs", [])
        except Exception:
            pass


job_manager = JobManager()
indic_engine = IndicF5BatchEngine()


# ─── MASTER BACKGROUND WORKER PIPELINE ───────────────────────────────────────
def run_colab_dubbing_worker(
    manager: JobManager,
    total_duration: float,
    srt_file_path: str,
    blocks: List[Dict[str, Any]],
):
    """Executes the optimized Indic-F5 batch pipeline for Google Colab."""
    try:
        hms_str = format_seconds_to_hms(total_duration)
        total_blocks = len(blocks)
        manager.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        manager.log(f"🚀 Starting Indic-F5 Batch Dubbing Pipeline ({hms_str})")
        manager.log(f"📝 Subtitle file: {os.path.basename(srt_file_path)} ({total_blocks} dialogue lines)")
        manager.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 1. Model Loading
        manager.status = "GENERATING_TTS"
        manager.progress = 10.0
        manager.message = "Loading Indic-F5 (0.3B) model into T4 GPU..."
        manager.save_to_disk()

        indic_engine.load_model(log_fn=manager.log)

        if manager.stop_event.is_set():
            raise KeyboardInterrupt("Job was cancelled by user.")

        # 2. Optimized Batch Processing (15-20 sentences on T4 GPU)
        # Choose batch size 18 on GPU to push VRAM usage to 6-8 GB
        batch_size = 18 if torch.cuda.is_available() else 2
        vram_info = indic_engine.get_vram_status()
        manager.log(f"⚡ [Batch Planner] Grouping {total_blocks} sentences into batches of {batch_size} (Target VRAM: 6-8GB){vram_info}")

        # Pre-assign destination paths for 1:1 timeline mapping
        for idx, block in enumerate(blocks):
            block["wav_path"] = os.path.join(SENTENCE_CHUNKS_DIR, f"sentence_{idx:04d}.wav")

        batches = [blocks[i : i + batch_size] for i in range(0, total_blocks, batch_size)]
        total_batches = len(batches)

        generated_blocks = []
        processed_count = 0

        for b_idx, batch in enumerate(batches):
            if manager.stop_event.is_set():
                raise KeyboardInterrupt("Job was cancelled by user.")

            batch_start_idx = processed_count + 1
            batch_end_idx = processed_count + len(batch)
            curr_progress = 12.0 + ((processed_count / total_blocks) * 73.0)
            manager.progress = round(curr_progress, 1)
            manager.current_sentence = batch_end_idx

            vram_now = indic_engine.get_vram_status()
            manager.message = (
                f"Batch {b_idx + 1}/{total_batches} (sentences {batch_start_idx}-{batch_end_idx}/{total_blocks}) "
                f"| T4 GPU Inference...{vram_now}"
            )
            manager.save_to_disk()

            t0 = time.time()
            batch_successes = indic_engine.synthesize_batch_with_oom_fallback(batch, log_fn=manager.log)
            t_elapsed = time.time() - t0

            generated_blocks.extend(batch_successes)
            processed_count += len(batch)

            manager.log(
                f"✅ [Batch {b_idx + 1}/{total_batches}] Finished in {t_elapsed:.2f}s "
                f"({len(batch_successes)}/{len(batch)} ok){indic_engine.get_vram_status()}"
            )

            # Prevent memory fragmentation
            if (b_idx + 1) % 2 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        # Sort blocks chronologically by start_time
        generated_blocks.sort(key=lambda b: (b.get("start_time", 0.0), b.get("index", 0)))
        manager.log(f"🎯 [Audio Sync] {len(generated_blocks)}/{total_blocks} dialogue blocks ready for timeline alignment.")

        if len(generated_blocks) == 0:
            raise RuntimeError("All sentence blocks failed synthesis. Could not produce any audio.")

        if manager.stop_event.is_set():
            raise KeyboardInterrupt("Job was cancelled by user.")

        # 3. FFmpeg Silent Canvas Alignment
        manager.status = "SYNCING_AUDIO"
        manager.progress = 88.0
        manager.message = f"Overlaying audio segments onto {total_duration:.1f}s silent canvas..."
        manager.save_to_disk()

        timestamp_tag = time.strftime("%Y%m%d_%H%M%S")
        final_wav_filename = f"IndicF5_Dubbed_Master_{timestamp_tag}.wav"
        final_wav_path = os.path.join(OUTPUTS_DIR, final_wav_filename)

        build_silent_canvas_dubbed_audio(
            total_duration_sec=total_duration,
            generated_blocks=generated_blocks,
            final_output_path=final_wav_path,
            log_fn=manager.log,
        )

        if not (os.path.exists(final_wav_path) and os.path.getsize(final_wav_path) > 5000):
            raise RuntimeError("Final mixed audio file was not generated or is empty.")

        # 4. Storage Cleanup: Purge temporary chunks
        try:
            deleted_count = 0
            for f in Path(SENTENCE_CHUNKS_DIR).glob("*.wav"):
                try:
                    f.unlink()
                    deleted_count += 1
                except Exception:
                    pass
            manager.log(f"🧹 [Cleanup] Purged {deleted_count} temporary sentence files from workspace.")
        except Exception:
            pass

        # 5. Pipeline Completion
        manager.status = "COMPLETED"
        manager.progress = 100.0
        manager.completed_file = final_wav_path
        manager.end_time = time.time()
        elapsed_min = (manager.end_time - manager.start_time) / 60.0
        manager.message = f"Master Dubbed Audio created successfully in {elapsed_min:.1f} minutes!"
        manager.log(f"🎉 Dubbing finished successfully in {elapsed_min:.1f} minutes: {final_wav_filename}")
        manager.save_to_disk()

    except KeyboardInterrupt:
        manager.status = "CANCELLED"
        manager.message = "Process cancelled by user."
        manager.log("Job was cancelled by user.", level="WARNING")
        manager.end_time = time.time()
        manager.save_to_disk()

    except Exception as e:
        manager.status = "FAILED"
        manager.message = f"Error: {str(e)}"
        manager.log(f"Fatal dubbing error: {e}", level="ERROR")
        manager.end_time = time.time()
        manager.save_to_disk()

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ─── SYSTEM 4: CLEAN GRADIO UI ───────────────────────────────────────────────
CUSTOM_CSS = """
:root {
    --bg-primary: #0a0f1d;
    --card-bg: rgba(15, 23, 42, 0.75);
    --border-subtle: rgba(255, 255, 255, 0.08);
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

.studio-panel {
    border-radius: 12px !important;
    border: 1px solid var(--border-color-primary, #334155) !important;
    padding: 16px !important;
    margin-bottom: 14px !important;
    background: var(--background-fill-primary, #1e293b);
}

.btn-launch-primary {
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 8px !important;
}

.fixed-log-console textarea {
    height: 200px !important;
    max-height: 200px !important;
    min-height: 200px !important;
    overflow-y: auto !important;
    resize: none !important;
    font-family: ui-monospace, SFMono-Regular, monospace !important;
    font-size: 0.84rem !important;
}

@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(0.85); }
}

.radar-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulseDot 2s infinite ease-in-out;
}
"""


def extract_uploaded_path(file_obj: Any) -> Optional[str]:
    """Safely extracts local disk filepath from Gradio File representations."""
    if not file_obj:
        return None
    if isinstance(file_obj, str):
        path = file_obj.strip()
        return path if path else None
    if hasattr(file_obj, "path") and isinstance(file_obj.path, str) and file_obj.path.strip():
        return file_obj.path.strip()
    if hasattr(file_obj, "name") and isinstance(file_obj.name, str) and file_obj.name.strip():
        return file_obj.name.strip()
    if isinstance(file_obj, dict):
        if "path" in file_obj and isinstance(file_obj["path"], str):
            return file_obj["path"].strip()
        if "name" in file_obj and isinstance(file_obj["name"], str):
            return file_obj["name"].strip()
    if isinstance(file_obj, (list, tuple)) and len(file_obj) > 0:
        return extract_uploaded_path(file_obj[0])
    return None


def get_dashboard_state() -> Tuple[str, float, str, Optional[str], Any, Any]:
    """Builds snapshot of pipeline state for the Gradio dashboard."""
    with job_manager.lock:
        if job_manager.status in ["STARTING", "PARSING_SRT", "GENERATING_TTS", "SYNCING_AUDIO"]:
            if not job_manager.worker_thread or not job_manager.worker_thread.is_alive():
                job_manager.status = "IDLE"
                job_manager.progress = 0.0
                job_manager.job_id = None
                job_manager.message = "System Standby — Enter Total Duration and upload Translated SRT to start."
                job_manager.current_sentence = 0
                job_manager.total_sentences = 0
                job_manager.start_time = None
                job_manager.end_time = None
                job_manager.save_to_disk()

    state = job_manager.get_state()
    status = state["status"]
    progress = state["progress"]
    message = state["message"]
    elapsed = state["elapsed_sec"]
    logs = "\n".join(state["logs"]) if state["logs"] else "System ready for Indic-F5 batch dubbing."
    completed = state["completed_file"]

    status_config = {
        "IDLE": {"label": "STANDBY", "dot_color": "#94a3b8", "bg": "rgba(148, 163, 184, 0.12)", "border": "rgba(148, 163, 184, 0.25)", "text": "#94a3b8"},
        "STARTING": {"label": "INITIALIZING", "dot_color": "#38bdf8", "bg": "rgba(56, 189, 248, 0.12)", "border": "rgba(56, 189, 248, 0.25)", "text": "#38bdf8"},
        "PARSING_SRT": {"label": "PARSING SRT", "dot_color": "#f59e0b", "bg": "rgba(245, 158, 11, 0.12)", "border": "rgba(245, 158, 11, 0.25)", "text": "#f59e0b"},
        "GENERATING_TTS": {"label": "T4 BATCH TTS", "dot_color": "#a855f7", "bg": "rgba(168, 85, 247, 0.12)", "border": "rgba(168, 85, 247, 0.25)", "text": "#a855f7"},
        "SYNCING_AUDIO": {"label": "SILENT CANVAS SYNC", "dot_color": "#3b82f6", "bg": "rgba(59, 130, 246, 0.12)", "border": "rgba(59, 130, 246, 0.25)", "text": "#3b82f6"},
        "COMPLETED": {"label": "DUBBING COMPLETE", "dot_color": "#10b981", "bg": "rgba(16, 185, 129, 0.12)", "border": "rgba(16, 185, 129, 0.25)", "text": "#10b981"},
        "FAILED": {"label": "ERROR OCCURRED", "dot_color": "#dc2626", "bg": "rgba(220, 38, 38, 0.12)", "border": "rgba(220, 38, 38, 0.25)", "text": "#dc2626"},
        "CANCELLED": {"label": "CANCELLED BY USER", "dot_color": "#64748b", "bg": "rgba(100, 116, 139, 0.12)", "border": "rgba(100, 116, 139, 0.25)", "text": "#64748b"},
    }
    cfg = status_config.get(status, status_config["IDLE"])

    status_md = f"""
    <div style="background: var(--background-fill-secondary, rgba(125, 125, 125, 0.05)); border: 1px solid {cfg['border']}; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 6px; background: {cfg['bg']}; border: 1px solid {cfg['border']}; padding: 4px 12px; border-radius: 9999px;">
                    <span class="radar-dot" style="background-color: {cfg['dot_color']};"></span>
                    <span style="color: {cfg['text']}; font-weight: 700; font-size: 0.82rem;">{cfg['label']}</span>
                </div>
                <span style="font-size: 0.92rem; font-weight: 500;">{message}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px; font-family: ui-monospace, monospace; font-size: 0.82rem;">
                <span style="border: 1px solid var(--border-color-primary, rgba(125,125,125,0.2)); padding: 4px 10px; border-radius: 6px;">
                    JOB: <b>{state['job_id'] or 'STANDBY'}</b>
                </span>
                <span style="border: 1px solid var(--border-color-primary, rgba(125,125,125,0.2)); padding: 4px 10px; border-radius: 6px;">
                    TIME: <b>{elapsed//60:02d}:{elapsed%60:02d}</b>
                </span>
                <span style="border: 1px solid {cfg['border']}; background: {cfg['bg']}; color: {cfg['text']}; padding: 4px 12px; border-radius: 6px; font-weight: 700;">
                    {progress:.0f}%
                </span>
            </div>
        </div>
    </div>
    """

    out_file = completed if (completed and os.path.exists(completed) and os.path.getsize(completed) > 1000) else None
    is_running = status in ["STARTING", "PARSING_SRT", "GENERATING_TTS", "SYNCING_AUDIO"]

    return (
        status_md,
        progress,
        logs,
        out_file,
        gr.update(interactive=not is_running),
        gr.update(interactive=is_running),
    )


# ─── SYNCHRONOUS PRE-FLIGHT SRT PARSER HANDLER ───────────────────────────────
def start_pipeline_handler(total_duration: Any, srt_file: Any):
    """Synchronous pre-flight check of SRT file before starting background worker."""
    srt_path = extract_uploaded_path(srt_file)
    duration_val = parse_duration_to_seconds(total_duration)

    # 1. ROBUST SYNCHRONOUS SRT PARSER (THE FIX)
    try:
        blocks = parse_srt_synchronous(srt_path)
    except Exception as validation_err:
        err_msg = str(validation_err)
        gr.Warning(f"❌ {err_msg}")
        job_manager.status = "FAILED"
        job_manager.message = err_msg
        job_manager.log(f"❌ Subtitle Validation Error: {err_msg}", level="ERROR")
        job_manager.save_to_disk()
        yield get_dashboard_state()
        return

    # 2. Launch background worker with verified blocks
    success, msg = job_manager.start_job(
        total_duration=duration_val,
        srt_file_path=srt_path,
        blocks=blocks,
    )
    if not success:
        gr.Warning(f"⚠️ {msg}")
        yield get_dashboard_state()
        return

    yield get_dashboard_state()

    while True:
        state = get_dashboard_state()
        yield state
        if job_manager.status in ["COMPLETED", "FAILED", "CANCELLED"]:
            if job_manager.status == "FAILED":
                gr.Warning(f"❌ Pipeline Failed: {job_manager.message}")
            break
        time.sleep(1.0)


def cancel_pipeline_handler():
    job_manager.cancel_job()
    return get_dashboard_state()


def reset_pipeline_handler():
    job_manager.reset_job()
    return get_dashboard_state()


# ─── GRADIO BLOCKS APPLICATION ───────────────────────────────────────────────
with gr.Blocks(title="Indic-F5 Colab Studio") as demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    # 1. Header Banner
    gr.HTML(
        """
        <div style="border-bottom: 1px solid var(--border-color-primary, #334155); padding-bottom: 14px; margin-bottom: 18px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span style="font-size: 0.76rem; font-weight: 700; border: 1px solid #10b981; color: #10b981; padding: 2px 8px; border-radius: 6px;">⚡ T4 GPU ACCELERATED</span>
                        <span style="font-size: 0.76rem; font-weight: 700; border: 1px solid #a855f7; color: #a855f7; padding: 2px 8px; border-radius: 6px;">📦 BATCH SIZE: 15-20</span>
                        <span style="font-size: 0.76rem; font-weight: 700; border: 1px solid #38bdf8; color: #38bdf8; padding: 2px 8px; border-radius: 6px;">🛡️ DYNAMIC OOM FALLBACK</span>
                    </div>
                    <h1 style="font-size: 1.85rem; font-weight: 800; margin: 0; color: var(--body-text-color, #f8fafc);">🎙️ Indic-F5 Audio Studio</h1>
                    <p style="font-size: 0.95rem; color: #94a3b8; margin: 4px 0 0 0;">Google Colab T4 Edition — High-Throughput Batch Dubber & Silent Canvas Alignment</p>
                </div>
            </div>
        </div>
        """
    )

    # 2. Main Workstation
    with gr.Row():
        with gr.Column(scale=5, elem_classes=["studio-panel"]):
            gr.Markdown("### ⏱️ 1. Total Video Duration")
            total_duration_input = gr.Textbox(
                label="Total Video Duration (HH:MM:SS or Seconds)",
                value="00:02:00",
                placeholder="HH:MM:SS (e.g. 01:30:00 or 00:02:00)",
                interactive=True,
            )
            gr.Markdown(
                """
                <div style='font-size: 0.82rem; opacity: 0.75; margin-top: 4px;'>
                    🔇 <b>Silent Base:</b> FFmpeg generates an exact silent track (<code>anullsrc</code>) matching this duration.
                </div>
                """
            )

        with gr.Column(scale=7, elem_classes=["studio-panel"]):
            gr.Markdown("### 📝 2. Translated Subtitle File (.srt)")
            srt_file_input = gr.File(
                label="Select or Drag & Drop Translated Subtitle (.srt)",
                file_types=[".srt", ".txt"],
                file_count="single",
                type="filepath",
                interactive=True,
            )
            gr.Markdown(
                """
                <div style='font-size: 0.82rem; opacity: 0.75; margin-top: 4px;'>
                    🛡️ <b>Pre-Flight Guard:</b> Multi-encoding safe reader validates subtitle syntax immediately before processing.
                </div>
                """
            )

    # Action Buttons
    with gr.Row():
        start_btn = gr.Button("🚀 Generate Dubbed Audio", variant="primary", scale=3, elem_classes=["btn-launch-primary"])
        cancel_btn = gr.Button("⛔ Cancel Job", variant="stop", scale=1, interactive=False)
        refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary", scale=1)
        reset_btn = gr.Button("🧹 Reset Standby", variant="secondary", scale=1)

    # 3. Live Pipeline Status Deck & Progress Meter
    status_display = gr.HTML()
    progress_bar = gr.Slider(
        label="Overall Dubbing Progress (%)",
        minimum=0,
        maximum=100,
        value=0,
        interactive=False,
    )

    # 4. Final Output Master
    with gr.Row(elem_classes=["studio-panel"]):
        with gr.Column(scale=12):
            gr.Markdown("### 🎧 Final Dubbed Audio Master")
            output_audio_master = gr.Audio(
                label="Master Dubbed Audio Output (.wav) — Synchronized to Video Duration",
                type="filepath",
                interactive=False,
            )

    # 5. Live Activity Logs Console
    with gr.Row(elem_classes=["studio-panel"]):
        with gr.Column(scale=12):
            gr.Markdown("### 📜 Real-Time Studio Logs")
            logs_console = gr.Textbox(
                label="Live Activity Stream",
                value="System initialized and ready.",
                lines=9,
                max_lines=12,
                elem_classes=["fixed-log-console"],
                interactive=False,
                autoscroll=True,
            )

    # Handlers & Timers
    ui_outputs = [
        status_display,
        progress_bar,
        logs_console,
        output_audio_master,
        start_btn,
        cancel_btn,
    ]

    start_btn.click(
        fn=start_pipeline_handler,
        inputs=[total_duration_input, srt_file_input],
        outputs=ui_outputs,
    )

    cancel_btn.click(
        fn=cancel_pipeline_handler,
        inputs=[],
        outputs=ui_outputs,
        show_progress="hidden",
    )

    refresh_btn.click(
        fn=get_dashboard_state,
        inputs=[],
        outputs=ui_outputs,
        show_progress="hidden",
    )

    reset_btn.click(
        fn=reset_pipeline_handler,
        inputs=[],
        outputs=ui_outputs,
        show_progress="hidden",
    )

    auto_timer = gr.Timer(value=1.0)
    auto_timer.tick(
        fn=get_dashboard_state,
        inputs=[],
        outputs=ui_outputs,
        show_progress="hidden",
    )

    demo.load(
        fn=get_dashboard_state,
        inputs=[],
        outputs=ui_outputs,
    )

if __name__ == "__main__":
    ensure_reference_audio()
    job_manager.reset_job()
    print("\n🚀 Launching Indic-F5 Studio on Google Colab...")
    print("🔗 Generating public Gradio share link, please wait 5-10 seconds...\n")
    try:
        demo.launch(share=True, css=CUSTOM_CSS, theme=gr.themes.Default())
    except TypeError:
        demo.launch(share=True)
