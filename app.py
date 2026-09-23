import os
import re
import sys
import time
import json
import logging
import threading
import shutil
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import gradio as gr
from pydub import AudioSegment
from pydub.effects import speedup
import yt_dlp
from faster_whisper import WhisperModel
import google.generativeai as genai
from kokoro_onnx import Kokoro
from huggingface_hub import hf_hub_download

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("yt_dubbing_pipeline")

# ─── Directories & Global Constants ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Kokoro-ONNX Configuration
KOKORO_HF_REPO = "rumbleFTW/kokoro-v1.0-onnx"
KOKORO_MODEL_FILE = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = "voices-v1.0.bin"
KOKORO_SAMPLE_RATE = 24000

# Multi-Model Fallback Hierarchy for Gemini API
GEMINI_MODEL_CASCADE = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
]


# ─── Module 1: YouTube Audio Downloader ────────────────────────────────────────
def extract_youtube_audio(url: str, output_dir: str = OUTPUT_DIR) -> Tuple[str, float]:
    """Downloads highest quality audio stream from a YouTube URL and converts it to WAV.

    Returns:
        Tuple[str, float]: (path to extracted 16kHz WAV file, total duration in seconds)
    """
    log.info(f"[YouTubeDownloader] Downloading audio stream from: {url}")
    timestamp = int(time.time())
    output_template = os.path.join(output_dir, f"yt_source_{timestamp}.%(ext)s")
    final_wav_path = os.path.join(output_dir, f"yt_source_{timestamp}.wav")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": [
            "-ar", "16000",  # Downsample to 16kHz for Whisper optimal recognition
            "-ac", "1",      # Mono channel
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        duration = float(info_dict.get("duration", 0.0))

    if not os.path.exists(final_wav_path):
        # Fallback search if yt-dlp named it slightly differently
        potential_files = [
            f for f in os.listdir(output_dir)
            if f.startswith(f"yt_source_{timestamp}") and f.endswith(".wav")
        ]
        if potential_files:
            final_wav_path = os.path.join(output_dir, potential_files[0])
        else:
            raise FileNotFoundError(f"Audio extraction failed for URL: {url}")

    if duration <= 0.0:
        audio_seg = AudioSegment.from_file(final_wav_path)
        duration = len(audio_seg) / 1000.0

    log.info(f"[YouTubeDownloader] Audio downloaded successfully: {final_wav_path} (Duration: {duration:.2f}s)")
    return final_wav_path, duration


# ─── Module 2: English ASR Transcriber (faster-whisper) ────────────────────────
_whisper_instances: Dict[str, WhisperModel] = {}
_whisper_lock = threading.Lock()

def get_whisper_model(model_size: str = "base.en") -> WhisperModel:
    """Singleton getter for faster-whisper model to prevent reloading weights into memory."""
    with _whisper_lock:
        if model_size not in _whisper_instances:
            log.info(f"[Whisper] Loading faster-whisper model '{model_size}' on CPU (int8)...")
            _whisper_instances[model_size] = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                download_root=os.path.join(MODEL_CACHE_DIR, "whisper"),
            )
        return _whisper_instances[model_size]


def transcribe_audio_whisper(audio_path: str, model_size: str = "base.en") -> List[Dict[str, Any]]:
    """Transcribes English audio into timestamped segments using faster-whisper."""
    log.info(f"[Whisper] Transcribing {os.path.basename(audio_path)} ...")
    model = get_whisper_model(model_size)
    segments, info = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    results = []
    for idx, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue
        results.append({
            "id": idx,
            "start": round(float(seg.start), 2),
            "end": round(float(seg.end), 2),
            "text": text,
        })

    log.info(f"[Whisper] Transcribed {len(results)} valid segments (Language detected: {info.language}).")
    return results


# ─── Module 3: Gemini Dubbing Director (Multi-Model Fallback) ───────────────────
GEMINI_DIRECTOR_SYSTEM_PROMPT = """
You are a Professional YouTube Audio Dubbing Director and Translator.
Your job is to translate a sequence of English transcribed speech segments into natural, conversational, and culturally accurate Hindi for voice-over dubbing.

CRITICAL DUBBING RULES:
1. SCRIPT ACCURACY: Write ONLY in pure Devanagari Hindi script. Do NOT use English alphabet (No Hinglish, no Latin letters, no brackets).
2. TIMING SYNCHRONIZATION:
   - For each segment, compare the target duration (original_end - original_start) with your Hindi translation.
   - Hindi typically has more syllables than English. Adapt the phrasing to be concise so it fits naturally.
   - Specify a "speed" parameter between 0.90 (slow dramatic) and 1.25 (fast natural) to help the audio fit the window.
3. OUTPUT FORMAT:
   - Return STRICTLY a valid JSON array of objects.
   - Do NOT include any markdown preamble, conversational filler, or formatting notes.
   - Each object must have these exact keys:
     [
       {
         "id": 0,
         "original_start": 0.0,
         "original_end": 4.5,
         "hindi_text": "अनुवादित हिंदी संवाद यहाँ लिखें",
         "speed": 1.05,
         "pitch_mod": "normal"
       }
     ]
"""

def translate_and_direct_batch(
    segments: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Translates speech segments into Hindi using Gemini with a Multi-Model Fallback Cascade."""
    resolved_key = (
        (api_key or "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY_1", "").strip()
        or os.environ.get("GEMINI_API_KEY_2", "").strip()
    )
    if not resolved_key:
        raise ValueError("Missing Gemini API Key. Please provide it in the UI or set GEMINI_API_KEY.")

    genai.configure(api_key=resolved_key)

    user_payload = json.dumps(segments, ensure_ascii=False, indent=2)
    prompt = f"{GEMINI_DIRECTOR_SYSTEM_PROMPT}\n\nHere are the transcribed segments to translate:\n{user_payload}"

    last_error = None
    for model_name in GEMINI_MODEL_CASCADE:
        log.info(f"[GeminiDirector] Attempting batch translation using model: '{model_name}' ...")
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": 0.3,
                    "response_mime_type": "application/json",
                },
            )
            response = model.generate_content(prompt)
            raw_text = response.text.strip()

            # Clean potential markdown wrapping if present
            clean_json = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
            clean_json = re.sub(r"\s*```$", "", clean_json, flags=re.MULTILINE).strip()

            try:
                parsed = json.loads(clean_json)
            except json.JSONDecodeError:
                # Extract first matching array if extra text surrounds the JSON
                match = re.search(r"\[.*\]", clean_json, flags=re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                else:
                    raise

            if isinstance(parsed, list) and len(parsed) > 0:
                log.info(f"[GeminiDirector] Successfully translated {len(parsed)} segments using '{model_name}'.")
                return parsed
            else:
                raise ValueError("Model output did not contain a valid non-empty JSON list.")

        except Exception as exc:
            error_str = str(exc)
            log.warning(f"[GeminiDirector] Model '{model_name}' failed: {error_str}")
            last_error = exc
            # If hit rate-limit (429) or model-not-found (404) or server-error (500), cascade to next
            time.sleep(1.0)
            continue

    raise RuntimeError(
        f"All Gemini models in fallback cascade failed. Last error: {last_error}"
    )


# ─── Module 4: Kokoro-ONNX Hindi TTS Engine (Singleton) ────────────────────────
def sanitize_hindi_text(text: str) -> str:
    """Aggressively purges non-Devanagari characters, brackets, and Latin artifacts."""
    if not text:
        return ""
    # Strip brackets and enclosed text
    text = re.sub(r'\[.*?\]|\(.*?\)|<.*?>|\{.*?\}|【.*?】|〔.*?〕|［.*?］', '', text, flags=re.DOTALL)
    # Strip Latin letters, numbers, and technical formatting symbols
    text = re.sub(r'[a-zA-Z0-9_:|\-\+\/\*=\\\#\[\]\(\)\{\}]+', '', text)
    # Normalize whitespace
    return re.sub(r'\s+', ' ', text).strip()


def split_into_safe_chunks(text: str, max_chars: int = 240) -> List[str]:
    """Splits Hindi text into safe chunks under Kokoro's 510 phoneme limit (~240 chars)."""
    text = sanitize_hindi_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    parts = re.split(r'([।\.!\?]+)', text)
    sentences = []
    temp = ""
    for part in parts:
        if not part:
            continue
        if re.match(r'^[।\.!\?]+$', part):
            temp += part
            sentences.append(temp.strip())
            temp = ""
        else:
            temp += part
    if temp:
        sentences.append(temp.strip())

    chunks = []
    for s in sentences:
        if len(s) <= max_chars:
            chunks.append(s)
        else:
            sub_parts = re.split(r'([,;，；\s]+)', s)
            sub_temp = ""
            for sp in sub_parts:
                if len(sub_temp) + len(sp) > max_chars:
                    if sub_temp.strip():
                        chunks.append(sub_temp.strip())
                    sub_temp = sp
                else:
                    sub_temp += sp
            if sub_temp.strip():
                chunks.append(sub_temp.strip())

    return [c for c in chunks if c]


class KokoroEngine:
    """Thread-safe Singleton managing the local Kokoro ONNX model and voice embeddings."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._ready = False
                    obj._kokoro = None
                    obj._infer_lock = threading.Lock()
                    cls._instance = obj
        return cls._instance

    def ensure_loaded(self):
        """Idempotently ensures Kokoro-ONNX weights and voice arrays are downloaded and ready."""
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            for attempt in range(1, 4):
                try:
                    log.info(f"[KokoroTTS] Singleton init attempt {attempt}/3 ...")
                    self._load()
                    self._ready = True
                    log.info("[KokoroTTS] Kokoro-ONNX engine fully initialised and ready.")
                    return
                except Exception as exc:
                    log.error(f"[KokoroTTS] Init attempt {attempt}/3 failed: {exc}")
                    if attempt < 3:
                        time.sleep(2)
            raise RuntimeError("[KokoroTTS] Failed to initialize model after 3 attempts.")

    def _load(self):
        model_cache = os.path.join(MODEL_CACHE_DIR, "kokoro")
        os.makedirs(model_cache, exist_ok=True)

        model_path = os.path.join(model_cache, KOKORO_MODEL_FILE)
        voices_path = os.path.join(model_cache, KOKORO_VOICES_FILE)

        # Download ONNX model weights
        if not (os.path.exists(model_path) and os.path.getsize(model_path) > 10_000):
            log.info(f"[KokoroTTS] Downloading {KOKORO_MODEL_FILE} from HF Hub (token-free)...")
            tmp = hf_hub_download(
                repo_id=KOKORO_HF_REPO,
                filename=KOKORO_MODEL_FILE,
                local_dir=model_cache,
                local_dir_use_symlinks=False,
            )
            if tmp != model_path and os.path.exists(tmp):
                shutil.move(tmp, model_path)

        # Download voices embedding binary
        if not (os.path.exists(voices_path) and os.path.getsize(voices_path) > 10_000):
            log.info(f"[KokoroTTS] Downloading {KOKORO_VOICES_FILE} from HF Hub (token-free)...")
            tmp = hf_hub_download(
                repo_id=KOKORO_HF_REPO,
                filename=KOKORO_VOICES_FILE,
                local_dir=model_cache,
                local_dir_use_symlinks=False,
            )
            if tmp != voices_path and os.path.exists(tmp):
                shutil.move(tmp, voices_path)

        self._kokoro = Kokoro(model_path, voices_path)

    def synthesize(self, text: str, voice: str = "hm_omega", speed: float = 1.0) -> Tuple[np.ndarray, int]:
        """Synthesizes Hindi text and returns float32 samples and sample rate."""
        with self._infer_lock:
            try:
                samples, sample_rate = self._kokoro.create(
                    text,
                    voice=voice,
                    speed=speed,
                    lang="hi",
                )
            except Exception as e:
                log.warning(f"[KokoroTTS] Synthesis with lang='hi' failed ({e}), retrying with lang='en-us' fallback.")
                samples, sample_rate = self._kokoro.create(
                    text,
                    voice=voice,
                    speed=speed,
                    lang="en-us",
                )
        if samples is None or len(samples) == 0:
            raise ValueError(f"Empty audio generated for text: '{text[:20]}...'")
        return samples, sample_rate


_kokoro_engine = KokoroEngine()


def pcm_to_audiosegment(samples: np.ndarray, sample_rate: int = KOKORO_SAMPLE_RATE) -> AudioSegment:
    """Converts float32 audio samples to a normalized 16-bit PCM AudioSegment."""
    arr = np.asarray(samples, dtype=np.float32)
    peak = np.abs(arr).max()
    if peak > 0:
        arr = (arr / peak) * 0.95  # Headroom normalization to prevent digital clipping
    pcm16 = (arr * 32767).astype(np.int16)
    return AudioSegment(
        pcm16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )


def synthesize_hindi_segment(
    text: str,
    voice: str = "hm_omega",
    speed: float = 1.0
) -> AudioSegment:
    """Synthesizes a full segment (handling sub-chunking if needed) into an AudioSegment."""
    _kokoro_engine.ensure_loaded()
    chunks = split_into_safe_chunks(text)
    if not chunks:
        return AudioSegment.silent(duration=100)

    combined = AudioSegment.empty()
    for chunk in chunks:
        try:
            samples, sr = _kokoro_engine.synthesize(chunk, voice=voice, speed=speed)
            seg = pcm_to_audiosegment(samples, sr)
            combined += seg
        except Exception as e:
            log.warning(f"[KokoroTTS] Chunk failed '{chunk[:20]}': {e}. Inserting silence pad.")
            combined += AudioSegment.silent(duration=400)

    return combined


# ─── Module 5: Audio Synchronization & Time-Stretching ────────────────────────
def fit_audio_to_timeslot(
    audio_seg: AudioSegment,
    target_duration_ms: int,
    max_stretch_factor: float = 1.35
) -> AudioSegment:
    """Synchronizes audio duration to fit target slot using speed adjustment and padding."""
    actual_len_ms = len(audio_seg)
    if actual_len_ms == 0 or target_duration_ms <= 0:
        return AudioSegment.silent(duration=max(10, target_duration_ms))

    # Case 1: Audio is shorter than or equal to target window
    if actual_len_ms <= target_duration_ms:
        return audio_seg

    # Case 2: Audio exceeds window - calculate necessary speedup ratio
    ratio = actual_len_ms / target_duration_ms

    if ratio <= max_stretch_factor:
        try:
            # Pydub speedup without major pitch distortion
            adjusted = speedup(audio_seg, playback_speed=ratio)
            return adjusted[:target_duration_ms]
        except Exception:
            # Fallback frame_rate resample speedup
            new_frame_rate = int(audio_seg.frame_rate * ratio)
            adjusted = audio_seg._spawn(audio_seg.raw_data, overrides={"frame_rate": new_frame_rate})
            return adjusted.set_frame_rate(audio_seg.frame_rate)[:target_duration_ms]
    else:
        # Overflow exceeds maximum speedup limit: apply max speedup and soft-fade trim
        try:
            adjusted = speedup(audio_seg, playback_speed=max_stretch_factor)
        except Exception:
            new_frame_rate = int(audio_seg.frame_rate * max_stretch_factor)
            adjusted = audio_seg._spawn(audio_seg.raw_data, overrides={"frame_rate": new_frame_rate})
            adjusted = adjusted.set_frame_rate(audio_seg.frame_rate)

        # Apply a smooth 50ms fade-out on the tail cut to eliminate pop/click artifacts
        fade_len = min(50, target_duration_ms)
        return adjusted[:target_duration_ms].fade_out(fade_len)


def assemble_master_dubbed_track(
    director_plan: List[Dict[str, Any]],
    total_duration_sec: float,
    voice: str = "hm_omega",
    progress_callback=None
) -> str:
    """Synthesizes, synchronizes, and stitches all speech segments onto a master audio timeline."""
    total_ms = int(total_duration_sec * 1000)
    master_track = AudioSegment.silent(duration=total_ms, frame_rate=44100)

    total_segments = len(director_plan)
    log.info(f"[AudioSync] Building master timeline ({total_duration_sec:.2f}s) across {total_segments} segments...")

    for idx, item in enumerate(director_plan):
        if progress_callback:
            progress_callback(idx / total_segments, desc=f"Dubbing segment {idx+1}/{total_segments}...")

        start_ms = int(float(item.get("original_start", 0.0)) * 1000)
        end_ms = int(float(item.get("original_end", 0.0)) * 1000)
        target_duration_ms = max(200, end_ms - start_ms)

        hindi_text = item.get("hindi_text", "")
        speed_param = float(item.get("speed", 1.0))

        # Synthesize Hindi speech
        synth_seg = synthesize_hindi_segment(hindi_text, voice=voice, speed=speed_param)

        # Convert to 44.1 kHz to match master track format
        synth_seg_44k = synth_seg.set_frame_rate(44100)

        # Time-stretch / fit to timestamp
        fitted_seg = fit_audio_to_timeslot(synth_seg_44k, target_duration_ms)

        # Overlay at exact timestamp
        master_track = master_track.overlay(fitted_seg, position=start_ms)

    timestamp = int(time.time())
    output_mp3_path = os.path.join(OUTPUT_DIR, f"dubbed_hindi_master_{timestamp}.mp3")
    log.info(f"[AudioSync] Exporting master dubbed MP3 to {output_mp3_path} ...")
    master_track.export(output_mp3_path, format="mp3", bitrate="128k")

    return output_mp3_path


# ─── Master Pipeline Coordinator ───────────────────────────────────────────────
def run_auto_dubbing_pipeline(
    youtube_url: str,
    gemini_api_key: str,
    voice_choice: str,
    whisper_model_choice: str,
    progress=gr.Progress()
):
    """Executes the full end-to-end auto audio dubbing pipeline with real-time UI progress."""
    youtube_url = (youtube_url or "").strip()
    if not youtube_url:
        raise gr.Error("Please enter a valid YouTube URL.")

    # Voice map
    voice_name = "hm_omega" if "hm_omega" in voice_choice else "hf_alpha"

    try:
        # Step 1: Download Audio
        progress(0.05, desc="Step 1/5: Extracting audio from YouTube...")
        raw_audio_path, total_duration = extract_youtube_audio(youtube_url)

        # Step 2: Whisper Transcription
        progress(0.25, desc=f"Step 2/5: Transcribing English timestamps ({whisper_model_choice})...")
        transcribed_segments = transcribe_audio_whisper(raw_audio_path, model_size=whisper_model_choice)
        if not transcribed_segments:
            raise RuntimeError("No spoken English dialogue detected in the provided video audio.")

        # Step 3: Gemini Director & Batch Translation
        progress(0.50, desc="Step 3/5: Gemini Director translating & timing dialogue...")
        director_plan = translate_and_direct_batch(transcribed_segments, api_key=gemini_api_key)

        # Step 4 & 5: Kokoro Synthesis & Audio Alignment
        progress(0.70, desc="Step 4/5: Synthesizing Kokoro-ONNX Hindi speech...")

        def sub_progress(fraction, desc=""):
            progress(0.70 + (fraction * 0.25), desc=f"Step 4/5: {desc}")

        master_mp3_path = assemble_master_dubbed_track(
            director_plan,
            total_duration_sec=total_duration,
            voice=voice_name,
            progress_callback=sub_progress
        )

        progress(1.0, desc="Dubbing completed successfully!")
        status_md = (
            f"### ✅ Dubbing Process Completed!\n"
            f"- **Video Length**: `{total_duration:.1f}s`\n"
            f"- **Total Segments Dubbed**: `{len(director_plan)}`\n"
            f"- **Voice**: `{voice_choice}`\n"
            f"- **Master File**: `{os.path.basename(master_mp3_path)}`"
        )
        return master_mp3_path, status_md, director_plan

    except Exception as e:
        log.error(f"[Pipeline] Execution error: {e}", exc_info=True)
        raise gr.Error(f"Pipeline Error: {str(e)}")


# ─── Module 6: Gradio 4.x User Interface ───────────────────────────────────────
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1050px !important;
    margin: 0 auto !important;
}
.header-box {
    text-align: center;
    padding: 24px 12px;
    background: linear-gradient(135deg, #1e1e2f 0%, #111119 100%);
    border-radius: 12px;
    margin-bottom: 24px;
    border: 1px solid #2d2d42;
}
.header-box h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
}
.header-box p {
    color: #9ba1b0;
    font-size: 1.05rem;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="rose", secondary_hue="slate"), css=CUSTOM_CSS, title="YouTube Auto Audio Dubber") as demo:
    with gr.Column(elem_classes=["header-box"]):
        gr.Markdown(
            """
            # 🎙️ YouTube Auto Audio Dubber (EN ➔ HI)
            ### AI-Powered English to Hindi Video Audio Dubbing with Timestamp Alignment
            *Powered by **faster-whisper**, **Gemini 1.5 Flash Director**, and **Kokoro-ONNX Hindi TTS***
            """
        )

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("#### 📥 Video & Credentials Input")
            yt_url_input = gr.Textbox(
                label="YouTube Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                lines=1,
            )
            api_key_input = gr.Textbox(
                label="Gemini API Key",
                placeholder="Enter Gemini API Key (or leave empty if GEMINI_API_KEY env var is set)",
                type="password",
                lines=1,
            )

            with gr.Row():
                voice_dropdown = gr.Dropdown(
                    label="Hindi Dubbing Voice",
                    choices=[
                        "hm_omega (Hindi Male Dramatic)",
                        "hf_alpha (Hindi Female Expressive)",
                    ],
                    value="hm_omega (Hindi Male Dramatic)",
                )
                whisper_size_dropdown = gr.Dropdown(
                    label="Whisper ASR Model",
                    choices=["tiny.en", "base.en", "small.en"],
                    value="base.en",
                )

            start_btn = gr.Button("🚀 Start Auto Audio Dubbing", variant="primary", size="lg")

        with gr.Column(scale=5):
            gr.Markdown("#### 🎧 Master Dubbed Output")
            audio_output = gr.Audio(
                label="Dubbed Audio Track (Hindi)",
                type="filepath",
                interactive=False,
            )
            status_output = gr.Markdown("Ready to process.")

    with gr.Accordion("🔍 Inspect Transcription & Dubbing Director Plan", open=False):
        director_json_output = gr.JSON(label="Segment Alignments & Director Metadata")

    start_btn.click(
        fn=run_auto_dubbing_pipeline,
        inputs=[
            yt_url_input,
            api_key_input,
            voice_dropdown,
            whisper_size_dropdown,
        ],
        outputs=[
            audio_output,
            status_output,
            director_json_output,
        ],
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
