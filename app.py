#!/usr/bin/env python3
"""
AudioGen Flow Studio — Neural Voice SRT-Driven Audio Dubber (Google Colab Edition)
═══════════════════════════════════════════════════════════════════════════════
Architecture:
1. Inputs:
   a) Total Video Duration (HH:MM:SS or numeric seconds, default 00:02:00 / 120s)
   b) Translated Subtitle File (.srt)
   c) Voice Selection (Hindi, Indian English, US English, and regional Indian voices)
   d) Speech Rate (+0% normal, +10%, +20%, etc.)
   Output: The final synchronized dubbed Audio file (.wav).
2. High-Performance Neural Voice Engine:
   - 100% Free of Hugging Face Dependencies (Zero Hugging Face Hub, Zero Transformers, Zero Tokenizers).
   - Powered by Microsoft Edge Neural Speech Synthesis (edge-tts).
   - Concurrent asynchronous batch synthesis with automatic retry & 0 KB crash checks.
3. FFmpeg Silent Canvas Alignment:
   - Generates a completely silent base canvas audio of exact total_duration seconds via anullsrc.
   - Overlays each generated sentence audio chunk at its exact SRT start_time offset.
   - Trims and pads timeline boundaries to guarantee an exact match with the target video duration.
   - Exports the combined master timeline as a single pristine .wav file.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import gc
import re
import time
import json
import uuid
import shutil
import logging
import asyncio
import threading
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import edge_tts
import gradio as gr
from pydub import AudioSegment

# ─── LOGGING CONFIGURATION ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Neural-Dubber")

# ─── DIRECTORIES & CONFIGURATION ─────────────────────────────────────────────
STORAGE_DIR = "storage"
WORKSPACE_DIR = os.path.join(STORAGE_DIR, "workspace")
OUTPUTS_DIR = os.path.join(STORAGE_DIR, "outputs")
SENTENCE_CHUNKS_DIR = os.path.join(WORKSPACE_DIR, "sentence_chunks")
JOB_STATE_FILE = os.path.join(STORAGE_DIR, "job_state.json")

for d in [STORAGE_DIR, WORKSPACE_DIR, OUTPUTS_DIR, SENTENCE_CHUNKS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── NEURAL VOICES CATALOG ───────────────────────────────────────────────────
SUPPORTED_VOICES: Dict[str, str] = {
    "Hindi - Swara (Female, Natural & Expressive)": "hi-IN-SwaraNeural",
    "Hindi - Madhur (Male, Deep & Storyteller)": "hi-IN-MadhurNeural",
    "English (India) - Neerja (Female, Clear Accent)": "en-IN-NeerjaNeural",
    "English (India) - Prabhat (Male, Clear Accent)": "en-IN-PrabhatNeural",
    "English (US) - Christopher (Male, Dynamic Narrator)": "en-US-ChristopherNeural",
    "English (US) - Jenny (Female, Conversational)": "en-US-JennyNeural",
    "Bengali (India) - Tanishaa (Female)": "bn-IN-TanishaaNeural",
    "Marathi (India) - Aarohi (Female)": "mr-IN-AarohiNeural",
    "Tamil (India) - Pallavi (Female)": "ta-IN-PallaviNeural",
    "Telugu (India) - Shruti (Female)": "te-IN-ShrutiNeural",
    "Urdu (Pakistan) - Uzma (Female)": "ur-PK-UzmaNeural",
}
DEFAULT_VOICE_LABEL = "Hindi - Swara (Female, Natural & Expressive)"

RATE_OPTIONS = [
    "+0% (Normal)",
    "+5% (Slightly Faster)",
    "+10% (Brisk)",
    "+15% (Fast)",
    "+20% (Very Fast)",
    "-5% (Slightly Slower)",
    "-10% (Slower)",
]


# ─── CORE SRT PARSING ENGINE ─────────────────────────────────────────────────
def read_srt_file_content(file_path: str) -> str:
    """Safely reads the contents of an SRT file, testing multiple common encodings."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SRT file does not exist: {file_path}")

    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]
    for enc in encodings:
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue

    return raw_bytes.decode("utf-8", errors="replace")


def parse_timestamp(ts_str: str) -> float:
    """Parses standard SRT timestamps (HH:MM:SS,mmm or HH:MM:SS.mmm) into seconds."""
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


def parse_srt(srt_path: str) -> List[Dict[str, Any]]:
    """Linear-time parser extracting timestamped dialogue blocks from an SRT file."""
    try:
        content = read_srt_file_content(srt_path)
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
                start_sec = parse_timestamp(time_parts[0])
                end_sec = parse_timestamp(time_parts[1])

                text_lines = lines[time_idx + 1 :]
                raw_text = " ".join(text_lines)

                # Strip HTML/styling tags (<i>, <b>, <font>, etc.)
                clean_text = re.sub(r"<[^>]+>", "", raw_text).strip()
                # Collapse internal whitespace
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
                "No valid subtitle blocks could be extracted. Please check that timestamps follow "
                "standard SRT format (e.g., '00:00:01,000 --> 00:00:04,000')."
            )

        return blocks

    except Exception:
        raise


# ─── NEURAL TTS GENERATION ENGINE (ZERO HUGGING FACE) ────────────────────────
class NeuralEdgeTTSGenerator:
    """Manages sentence-level high-fidelity speech synthesis with concurrency and 0 KB crash checks."""

    def __init__(self):
        pass

    async def _synthesize_single_sentence(
        self,
        text: str,
        out_path: str,
        voice: str,
        rate_str: str,
        semaphore: asyncio.Semaphore,
        manager: Optional[Any] = None,
    ) -> bool:
        """Synthesizes one sentence block using Edge-TTS with 1 automatic retry."""
        clean_text = text.strip()
        if not clean_text:
            return False

        async with semaphore:
            for attempt in range(2):
                try:
                    if os.path.exists(out_path):
                        try:
                            os.remove(out_path)
                        except Exception:
                            pass

                    communicate = edge_tts.Communicate(
                        text=clean_text,
                        voice=voice,
                        rate=rate_str,
                    )
                    await communicate.save(out_path)

                    # 0 KB Crash Protection
                    if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
                        return True
                    else:
                        if attempt == 0 and manager:
                            manager.log(f"⚠️ [Edge-TTS] File 0 KB for: '{clean_text[:25]}...'. Retrying 1 time...", level="WARNING")
                        await asyncio.sleep(0.4)
                except Exception as err:
                    if attempt == 0 and manager:
                        manager.log(f"⚠️ [Edge-TTS] Attempt 1 error for '{clean_text[:25]}...': {err}. Retrying...", level="WARNING")
                    await asyncio.sleep(0.4)

            return False

    def synthesize_blocks(
        self,
        blocks: List[Dict[str, Any]],
        voice: str = "hi-IN-SwaraNeural",
        rate_str: str = "+0%",
        manager: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Synthesizes all SRT subtitle blocks concurrently using asyncio."""
        if not blocks:
            return []

        total_blocks = len(blocks)
        successful_blocks: List[Dict[str, Any]] = []
        lock = threading.Lock()
        completed_count = 0

        async def orchestrate():
            nonlocal completed_count
            # Concurrency limit of 6 to prevent connection throttling while maximizing throughput
            semaphore = asyncio.Semaphore(6)

            async def handle_block(block: Dict[str, Any]):
                nonlocal completed_count
                if manager and manager.stop_event.is_set():
                    return

                out_path = block.get("wav_path", "")
                success = await self._synthesize_single_sentence(
                    text=block["text"],
                    out_path=out_path,
                    voice=voice,
                    rate_str=rate_str,
                    semaphore=semaphore,
                    manager=manager,
                )

                with lock:
                    completed_count += 1
                    if success and os.path.exists(out_path) and os.path.getsize(out_path) > 500:
                        b_copy = dict(block)
                        successful_blocks.append(b_copy)
                    else:
                        if manager:
                            manager.log(f"⏩ [Skip Block] Sentence {block.get('index')} skipped after retry.", level="WARNING")

                    if manager:
                        manager.current_sentence = completed_count
                        curr_prog = 10.0 + ((completed_count / total_blocks) * 75.0)
                        manager.progress = round(curr_prog, 1)
                        manager.message = f"Synthesizing sentence {completed_count}/{total_blocks} with {voice.split('-')[0]} voice..."
                        if completed_count % 5 == 0 or completed_count == total_blocks:
                            manager.save_to_disk()

            tasks = [asyncio.create_task(handle_block(b)) for b in blocks]
            await asyncio.gather(*tasks)

        # Run async event loop inside the worker thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(orchestrate())
        finally:
            loop.close()

        # Chronological sort by SRT start_time
        successful_blocks.sort(key=lambda b: (b.get("start_time", 0.0), b.get("index", 0)))
        return successful_blocks


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


# ─── FFMPEG SILENT CANVAS AUDIO SYNCING ───────────────────────────────────────
def build_silent_canvas_dubbed_audio(
    total_duration_sec: float,
    generated_blocks: List[Dict[str, Any]],
    final_output_path: str,
    manager: Optional[Any] = None,
) -> str:
    """Generates a silent base track of exact total_duration seconds via FFmpeg anullsrc,
    overlays generated dialogue chunks at their respective SRT start_time, and exports pristine master .wav.
    """
    total_duration_sec = max(1.0, float(total_duration_sec))
    total_duration_ms = int(total_duration_sec * 1000)
    hms_str = format_seconds_to_hms(total_duration_sec)

    if manager:
        manager.log(f"🔇 [Silent Canvas] Generating {hms_str} ({total_duration_sec:.1f}s) silent canvas base...")

    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(final_output_path)), exist_ok=True)

    silent_base_path = os.path.join(WORKSPACE_DIR, "silent_base_track.wav")

    # 1. Use FFmpeg to generate completely silent audio track of exact total_duration
    ffmpeg_silent_cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=stereo",
        "-t", f"{total_duration_sec:.3f}",
        "-c:a", "pcm_s16le",
        silent_base_path,
    ]

    silent_created = False
    try:
        res = subprocess.run(ffmpeg_silent_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if os.path.exists(silent_base_path) and os.path.getsize(silent_base_path) > 1000:
            silent_created = True
            if manager:
                manager.log(f"✅ [Silent Canvas] Created base silent track ({total_duration_sec:.1f}s)")
    except Exception as ff_err:
        if manager:
            manager.log(f"ℹ️ [Silent Canvas] FFmpeg binary notice: {ff_err}. Using pydub generator.", level="INFO")

    if not silent_created:
        # Pydub fallback for silent base canvas
        silent_canvas = AudioSegment.silent(duration=total_duration_ms, frame_rate=44100)
        silent_canvas.export(silent_base_path, format="wav")

    # 2. Overlay generated audio chunks at their exact SRT start_time
    if manager:
        manager.log(f"⏱️ [Audio Sync] Overlaying {len(generated_blocks)} dialogue chunks onto timeline...")

    try:
        base_canvas = AudioSegment.from_file(silent_base_path)
    except Exception:
        base_canvas = AudioSegment.silent(duration=total_duration_ms, frame_rate=44100)

    placed_count = 0
    for block in generated_blocks:
        chunk_path = block.get("wav_path", "")
        start_time_sec = block.get("start_time", 0.0)
        start_ms = max(0, int(start_time_sec * 1000))

        if start_ms >= total_duration_ms:
            if manager:
                manager.log(
                    f"⚠️ [Audio Sync] Block {block.get('index')} start_time ({start_time_sec:.1f}s) "
                    f"exceeds total duration ({total_duration_sec:.1f}s).",
                    level="WARNING",
                )

        if chunk_path and os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 500:
            try:
                chunk_seg = AudioSegment.from_file(chunk_path)
                base_canvas = base_canvas.overlay(chunk_seg, position=start_ms)
                placed_count += 1
            except Exception as place_err:
                if manager:
                    manager.log(f"⚠️ [Audio Sync] Notice placing block {block.get('index')}: {place_err}", level="WARNING")

    # 3. Trim or pad to guarantee exact duration match with requested total_duration
    if len(base_canvas) > total_duration_ms:
        base_canvas = base_canvas[:total_duration_ms]
    elif len(base_canvas) < total_duration_ms:
        pad = AudioSegment.silent(duration=total_duration_ms - len(base_canvas), frame_rate=base_canvas.frame_rate)
        base_canvas = base_canvas + pad

    # 4. Export the final combined audio as a pristine .wav file
    base_canvas.export(final_output_path, format="wav")

    if manager:
        final_size_kb = os.path.getsize(final_output_path) // 1024
        manager.log(f"✅ [Audio Sync] Master dubbed audio finalized: {os.path.basename(final_output_path)} ({final_size_kb} KB, {hms_str})")

    # Cleanup temporary silent base track
    if os.path.exists(silent_base_path):
        try:
            os.remove(silent_base_path)
        except Exception:
            pass

    return final_output_path


# ─── JOB MANAGER (THREAD-SAFE BACKGROUND PROCESSING) ──────────────────────────
class JobManager:
    """Manages asynchronous dubbing job with live status and persistence."""

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
        voice_id: str,
        rate_str: str,
    ) -> Tuple[bool, str]:
        """Launches the background dubbing job in a detached daemon thread."""
        with self.lock:
            if self.worker_thread and self.worker_thread.is_alive():
                return False, "A dubbing task is already running in the background. Wait or cancel it first."

            if total_duration <= 0:
                return False, "Please specify a positive Total Video Duration in seconds."

            if not srt_file_path or not os.path.exists(srt_file_path):
                err_msg = "Please upload a valid Translated Subtitle file (.srt)."
                self.log(f"❌ {err_msg}", level="ERROR")
                return False, err_msg

            self.job_id = uuid.uuid4().hex[:8]
            self.status = "STARTING"
            self.progress = 1.0
            self.message = f"Initializing neural dubbing job ({self.job_id})..."
            self.current_sentence = 0
            self.total_sentences = 0
            self.completed_file = None
            self.logs = []
            self.start_time = time.time()
            self.end_time = None
            self.stop_event.clear()

            self.worker_thread = threading.Thread(
                target=run_neural_dubbing_worker,
                args=(self, total_duration, srt_file_path, voice_id, rate_str),
                daemon=True,
            )
            self.worker_thread.start()

        self.log(f"Neural dubbing job started with ID: {self.job_id} ({format_seconds_to_hms(total_duration)})")
        self.save_to_disk()
        return True, f"Dubbing job started (ID: {self.job_id})."

    def cancel_job(self) -> Tuple[bool, str]:
        """Signals background worker to halt gracefully, or resets zombie state."""
        with self.lock:
            if self.worker_thread and self.worker_thread.is_alive():
                self.stop_event.set()
                self.status = "CANCELLED"
                self.message = "Cancellation requested by user. Terminating..."
                self.log("Cancellation signal emitted by user.", level="WARNING")
                self.save_to_disk()
                return True, "Cancellation signal sent."
            else:
                self.status = "IDLE"
                self.progress = 0.0
                self.job_id = None
                self.message = "System Standby — Enter Total Duration and upload Translated SRT to start."
                self.current_sentence = 0
                self.total_sentences = 0
                self.start_time = None
                self.end_time = None
                self.log("Job state reset to Standby.", level="INFO")
                self.save_to_disk()
                return True, "Job state reset to Standby."

    def reset_job(self) -> Tuple[bool, str]:
        """Force resets the entire job manager state to fresh clean IDLE standby."""
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
                if self.end_time:
                    elapsed = int(self.end_time - self.start_time)
                else:
                    elapsed = int(now - self.start_time)
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
tts_generator = NeuralEdgeTTSGenerator()


# ─── MASTER BACKGROUND WORKER PIPELINE ───────────────────────────────────────
def run_neural_dubbing_worker(
    manager: JobManager,
    total_duration: float,
    srt_file_path: str,
    voice_id: str,
    rate_str: str,
):
    """Executes the neural SRT-driven dubbing pipeline onto silent canvas."""
    try:
        hms_str = format_seconds_to_hms(total_duration)
        manager.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        manager.log(f"🚀 Starting Neural Audio Dubbing Pipeline ({hms_str})")
        manager.log(f"🎙️ Selected Voice: {voice_id} | Speed Rate: {rate_str}")
        manager.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 1. Parse SRT Subtitle File
        manager.status = "PARSING_SRT"
        manager.progress = 5.0
        manager.message = "Parsing translated subtitle file (.srt)..."
        manager.save_to_disk()

        try:
            if not srt_file_path or not os.path.exists(srt_file_path):
                raise FileNotFoundError(f"Subtitle file '{os.path.basename(srt_file_path)}' was not found on disk.")

            blocks = parse_srt(srt_file_path)
            total_blocks = len(blocks)

            if total_blocks == 0:
                raise ValueError(f"No valid dialogue lines found in '{os.path.basename(srt_file_path)}'.")

            manager.total_sentences = total_blocks
            manager.log(f"📝 [SRT Parser] Extracted {total_blocks} valid dialogue sentences.")
            manager.progress = 10.0
            manager.message = f"Parsed {total_blocks} dialogue sentences. Initializing neural synthesis..."
            manager.save_to_disk()

        except Exception as srt_err:
            err_msg = f"SRT Parsing Error: {srt_err}"
            manager.status = "FAILED"
            manager.message = err_msg
            manager.log(f"❌ {err_msg}", level="ERROR")
            manager.end_time = time.time()
            manager.save_to_disk()
            return

        if manager.stop_event.is_set():
            raise KeyboardInterrupt("Job was cancelled by user.")

        # 2. High-Performance Neural Voice Synthesis
        manager.status = "GENERATING_TTS"
        manager.progress = 12.0
        manager.message = f"Synthesizing {total_blocks} sentences with neural voice..."
        manager.save_to_disk()

        # Pre-assign individual target destination filepaths
        for idx, block in enumerate(blocks):
            block["wav_path"] = os.path.join(SENTENCE_CHUNKS_DIR, f"sentence_{idx:04d}.mp3")

        t0 = time.time()
        generated_blocks = tts_generator.synthesize_blocks(
            blocks=blocks,
            voice=voice_id,
            rate_str=rate_str,
            manager=manager,
        )
        t_elapsed = time.time() - t0

        if manager.stop_event.is_set():
            raise KeyboardInterrupt("Job was cancelled by user.")

        manager.log(
            f"✅ [Neural Synthesis] Finished {len(generated_blocks)}/{total_blocks} sentences in {t_elapsed:.2f}s "
            f"({len(generated_blocks)/max(1, t_elapsed):.1f} sent/sec)"
        )

        if len(generated_blocks) == 0:
            raise RuntimeError("All sentence blocks failed synthesis. Could not produce any audio.")

        # 3. FFmpeg Silent Canvas Audio Syncing
        manager.status = "SYNCING_AUDIO"
        manager.progress = 88.0
        manager.message = f"Overlaying audio chunks onto {total_duration:.1f}s silent canvas..."
        manager.save_to_disk()

        timestamp_tag = time.strftime("%Y%m%d_%H%M%S")
        final_wav_filename = f"Dubbed_Master_{timestamp_tag}.wav"
        final_wav_path = os.path.join(OUTPUTS_DIR, final_wav_filename)

        build_silent_canvas_dubbed_audio(
            total_duration_sec=total_duration,
            generated_blocks=generated_blocks,
            final_output_path=final_wav_path,
            manager=manager,
        )

        if not (os.path.exists(final_wav_path) and os.path.getsize(final_wav_path) > 5000):
            raise RuntimeError("Final mixed audio file was not generated or is empty.")

        # 4. Storage Cleanup: Clean temporary sentence chunk files
        try:
            deleted_count = 0
            for f in Path(SENTENCE_CHUNKS_DIR).glob("*.mp3"):
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
        gc.collect()


# ─── GRADIO UI INTERFACE ─────────────────────────────────────────────────────
CUSTOM_CSS = """
/* Modern Dark Glassmorphic Studio Theme */
:root {
    --bg-primary: #0a0f1d;
    --card-bg: rgba(15, 23, 42, 0.75);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --accent-blue: #38bdf8;
    --accent-emerald: #10b981;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

.studio-panel {
    border-radius: 14px !important;
    border: 1px solid var(--border-color-primary, #334155) !important;
    padding: 18px !important;
    margin-bottom: 16px !important;
    background: var(--background-fill-primary, #1e293b);
    backdrop-filter: blur(8px);
}

.btn-launch-primary {
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 8px !important;
}

.fixed-log-console textarea {
    height: 220px !important;
    max-height: 220px !important;
    min-height: 220px !important;
    overflow-y: auto !important;
    resize: none !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    font-size: 0.84rem !important;
    line-height: 1.45 !important;
}

.status-summary-card {
    min-height: 56px;
    box-sizing: border-box;
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
        if "path" in file_obj and isinstance(file_obj["path"], str) and file_obj["path"].strip():
            return file_obj["path"].strip()
        if "name" in file_obj and isinstance(file_obj["name"], str) and file_obj["name"].strip():
            return file_obj["name"].strip()
    if isinstance(file_obj, (list, tuple)) and len(file_obj) > 0:
        return extract_uploaded_path(file_obj[0])
    return None


def get_dashboard_state() -> Tuple[str, float, str, Optional[str], Any, Any]:
    """Builds snapshot of pipeline state for the Gradio dashboard."""
    with job_manager.lock:
        if job_manager.status in ["STARTING", "PARSING_SRT", "GENERATING_TTS", "SYNCING_AUDIO"]:
            if not job_manager.worker_thread or not job_manager.worker_thread.is_alive():
                # Zombie job from a past session / server reboot! Auto-reset to IDLE
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
    logs = "\n".join(state["logs"]) if state["logs"] else "System ready for Neural Voice dubbing."
    completed = state["completed_file"]

    status_config = {
        "IDLE": {"label": "STANDBY", "dot_color": "#94a3b8", "bg": "rgba(148, 163, 184, 0.12)", "border": "rgba(148, 163, 184, 0.25)", "text": "#94a3b8"},
        "STARTING": {"label": "INITIALIZING", "dot_color": "#38bdf8", "bg": "rgba(56, 189, 248, 0.12)", "border": "rgba(56, 189, 248, 0.25)", "text": "#38bdf8"},
        "PARSING_SRT": {"label": "PARSING SRT", "dot_color": "#f59e0b", "bg": "rgba(245, 158, 11, 0.12)", "border": "rgba(245, 158, 11, 0.25)", "text": "#f59e0b"},
        "GENERATING_TTS": {"label": "NEURAL SYNTHESIS", "dot_color": "#a855f7", "bg": "rgba(168, 85, 247, 0.12)", "border": "rgba(168, 85, 247, 0.25)", "text": "#a855f7"},
        "SYNCING_AUDIO": {"label": "SILENT CANVAS SYNC", "dot_color": "#3b82f6", "bg": "rgba(59, 130, 246, 0.12)", "border": "rgba(59, 130, 246, 0.25)", "text": "#3b82f6"},
        "COMPLETED": {"label": "DUBBING COMPLETE", "dot_color": "#10b981", "bg": "rgba(16, 185, 129, 0.12)", "border": "rgba(16, 185, 129, 0.25)", "text": "#10b981"},
        "FAILED": {"label": "ERROR OCCURRED", "dot_color": "#dc2626", "bg": "rgba(220, 38, 38, 0.12)", "border": "rgba(220, 38, 38, 0.25)", "text": "#dc2626"},
        "CANCELLED": {"label": "CANCELLED BY USER", "dot_color": "#64748b", "bg": "rgba(100, 116, 139, 0.12)", "border": "rgba(100, 116, 139, 0.25)", "text": "#64748b"},
    }
    cfg = status_config.get(status, status_config["IDLE"])

    status_md = f"""
    <div class="status-summary-card" style="background: var(--background-fill-secondary, rgba(125, 125, 125, 0.05)); border: 1px solid {cfg['border']}; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; min-height: 56px; box-sizing: border-box;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 6px; background: {cfg['bg']}; border: 1px solid {cfg['border']}; padding: 4px 12px; border-radius: 9999px;">
                    <span class="radar-dot" style="background-color: {cfg['dot_color']};"></span>
                    <span style="color: {cfg['text']}; font-weight: 700; font-size: 0.82rem; letter-spacing: 0.03em;">{cfg['label']}</span>
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


def start_pipeline_handler(total_duration: Any, srt_file: Any, voice_choice: str, rate_choice: str):
    """Initiates dubbing job and yields live status updates."""
    srt_path = extract_uploaded_path(srt_file)
    duration_val = parse_duration_to_seconds(total_duration)

    if not srt_path:
        gr.Warning("⚠️ Please upload the Translated Subtitle File (.srt) first.")
        job_manager.status = "FAILED"
        job_manager.message = "No subtitle file provided. Please upload a .srt file."
        job_manager.save_to_disk()
        yield get_dashboard_state()
        return

    if not os.path.exists(srt_path):
        err_msg = f"Subtitle file '{os.path.basename(srt_path)}' was not found on disk. Please re-upload your .srt file."
        gr.Warning(f"❌ {err_msg}")
        job_manager.status = "FAILED"
        job_manager.message = err_msg
        job_manager.log(f"❌ {err_msg}", level="ERROR")
        job_manager.save_to_disk()
        yield get_dashboard_state()
        return

    # Synchronous pre-flight validation of the SRT file
    try:
        test_blocks = parse_srt(srt_path)
        if not test_blocks:
            raise ValueError("No valid dialogue lines found in the uploaded subtitle file.")
    except Exception as validation_err:
        err_msg = f"SRT Validation Error: {validation_err}"
        gr.Warning(f"❌ {err_msg}")
        job_manager.status = "FAILED"
        job_manager.message = err_msg
        job_manager.log(f"❌ {err_msg}", level="ERROR")
        job_manager.save_to_disk()
        yield get_dashboard_state()
        return

    voice_id = SUPPORTED_VOICES.get(voice_choice, "hi-IN-SwaraNeural")
    clean_rate = rate_choice.split()[0] if rate_choice else "+0%"

    success, msg = job_manager.start_job(
        total_duration=duration_val,
        srt_file_path=srt_path,
        voice_id=voice_id,
        rate_str=clean_rate,
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


# ─── GRADIO APPLICATION LAYOUT ───────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Default(), css=CUSTOM_CSS, title="AudioGen Flow Studio") as demo:
    # 1. Header Banner
    gr.HTML(
        """
        <div style="border-bottom: 1px solid var(--border-color-primary, #334155); padding-bottom: 16px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                        <span style="font-size: 0.76rem; font-weight: 700; border: 1px solid #10b981; color: #10b981; padding: 2px 8px; border-radius: 6px;">⚡ ZERO HUGGING FACE</span>
                        <span style="font-size: 0.76rem; font-weight: 700; border: 1px solid #38bdf8; color: #38bdf8; padding: 2px 8px; border-radius: 6px;">🎙️ NEURAL SYNTHESIS</span>
                        <span style="font-size: 0.76rem; font-weight: 700; border: 1px solid #f59e0b; color: #f59e0b; padding: 2px 8px; border-radius: 6px;">🚀 GOOGLE COLAB OPTIMIZED</span>
                    </div>
                    <h1 style="font-size: 1.85rem; font-weight: 800; margin: 0; color: var(--body-text-color, #f8fafc);">🎙️ AudioGen Flow Studio</h1>
                    <p style="font-size: 0.95rem; color: #94a3b8; margin: 4px 0 0 0;">High-Speed Neural Voice SRT Dubber & FFmpeg Silent Canvas Timeline Sync</p>
                </div>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px;">
                <span style="font-size: 0.75rem; border: 1px solid var(--border-color-primary, #334155); padding: 3px 10px; border-radius: 9999px;">⚡ Zero Model Downloads</span>
                <span style="font-size: 0.75rem; border: 1px solid var(--border-color-primary, #334155); padding: 3px 10px; border-radius: 9999px;">🛡️ Zero GPU VRAM Crashes</span>
                <span style="font-size: 0.75rem; border: 1px solid var(--border-color-primary, #334155); padding: 3px 10px; border-radius: 9999px;">🔇 FFmpeg anullsrc Silent Canvas</span>
                <span style="font-size: 0.75rem; border: 1px solid var(--border-color-primary, #334155); padding: 3px 10px; border-radius: 9999px;">🎯 Sample-Accurate SRT Start Time Sync</span>
                <span style="font-size: 0.75rem; border: 1px solid var(--border-color-primary, #334155); padding: 3px 10px; border-radius: 9999px;">🗣️ Natural Hindi & Multilingual Voices</span>
            </div>
        </div>
        """
    )

    # 2. Main Workstation: Duration, Voice, Rate & Subtitle Upload
    with gr.Row():
        with gr.Column(scale=5, elem_classes=["studio-panel"]):
            gr.Markdown("### ⏱️ 1. Total Video Duration")
            total_duration_input = gr.Textbox(
                label="Total Video Duration (HH:MM:SS or Seconds)",
                value="00:02:00",
                placeholder="HH:MM:SS (e.g. 01:30:00 or 00:02:00)",
                interactive=True,
            )

            gr.Markdown("### 🗣️ 2. Neural Voice & Speech Rate")
            voice_dropdown = gr.Dropdown(
                label="Select Voice Character",
                choices=list(SUPPORTED_VOICES.keys()),
                value=DEFAULT_VOICE_LABEL,
                interactive=True,
            )
            rate_dropdown = gr.Dropdown(
                label="Speech Speed Rate",
                choices=RATE_OPTIONS,
                value="+0% (Normal)",
                interactive=True,
            )

        with gr.Column(scale=7, elem_classes=["studio-panel"]):
            gr.Markdown("### 📝 3. Translated Subtitle File (.srt)")
            srt_file_input = gr.File(
                label="Select or Drag & Drop Translated Subtitle (.srt)",
                file_types=[".srt", ".txt"],
                file_count="single",
                type="filepath",
                interactive=True,
            )
            gr.Markdown(
                """
                <div style='font-size: 0.85rem; opacity: 0.8; margin-top: 10px; line-height: 1.5;'>
                    ⏱️ <b>Sample-Accurate Timeline Sync:</b> Each subtitle block is synthesized with high-speed neural TTS and placed at its exact SRT <code>start_time</code> onto the silent base canvas.
                    <br>
                    ⚡ <b>Ultra-Fast:</b> Synthesizes dozens of lines in seconds without downloading multi-gigabyte models!
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
                lines=10,
                max_lines=12,
                elem_classes=["fixed-log-console"],
                interactive=False,
                autoscroll=True,
            )

    # Polling & Handlers
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
        inputs=[total_duration_input, srt_file_input, voice_dropdown, rate_dropdown],
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
    # Auto-purge any stale/zombie job state from prior session
    job_manager.reset_job()
    demo.launch(share=True)
