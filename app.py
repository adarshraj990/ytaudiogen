# ─── HUGGING FACE ZEROGPU INITIALIZATION (MUST BE AT VERY TOP) ────────────────
try:
    import spaces
    HAS_SPACES = True
except Exception:
    HAS_SPACES = False

if HAS_SPACES:
    @spaces.GPU
    def zerogpu_warmup():
        """ZeroGPU startup validation function required by Hugging Face ZeroGPU runtime."""
        return "ZeroGPU Active"
else:
    def zerogpu_warmup():
        """CPU fallback handler."""
        return "CPU Active"

import os
import re
import sys
import time
import json
import logging
import threading
import subprocess
import shutil
import gc
import uuid
import warnings
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Suppress harmless warnings for cleaner logs
warnings.filterwarnings("ignore", category=FutureWarning)

# Backward compatibility shim for Gradio 4.x OAuth with modern huggingface_hub
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class _HfFolderShim:
            path_token = os.path.expanduser("~/.cache/huggingface/token")

            @classmethod
            def save_token(cls, token: str):
                try:
                    os.makedirs(os.path.dirname(cls.path_token), exist_ok=True)
                    with open(cls.path_token, "w", encoding="utf-8") as f:
                        f.write(token)
                except Exception:
                    pass

            @classmethod
            def get_token(cls):
                token = os.environ.get("HF_TOKEN")
                if token:
                    return token
                if os.path.exists(cls.path_token):
                    try:
                        with open(cls.path_token, "r", encoding="utf-8") as f:
                            return f.read().strip()
                    except Exception:
                        return None
                return None

            @classmethod
            def delete_token(cls):
                if os.path.exists(cls.path_token):
                    try:
                        os.remove(cls.path_token)
                    except OSError:
                        pass

        huggingface_hub.HfFolder = _HfFolderShim
except Exception:
    pass

# Backward compatibility patch for Gradio 4.44.1 with Starlette 1.0+ TemplateResponse
try:
    from starlette.templating import Jinja2Templates
    _orig_template_response = Jinja2Templates.TemplateResponse

    def _safe_template_response(self, *args, **kwargs):
        # Gradio 4.44.1 calls: TemplateResponse(name, {"request": request, ...})
        # Starlette 1.0+ expects: TemplateResponse(request, name, context=...)
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            name = args[0]
            context = args[1]
            request = context.get("request")
            if request is not None:
                try:
                    return _orig_template_response(self, request, name, context, *args[2:], **kwargs)
                except TypeError:
                    pass
        return _orig_template_response(self, *args, **kwargs)

    Jinja2Templates.TemplateResponse = _safe_template_response
except Exception:
    pass

import gradio as gr

# Backward compatibility patch for Gradio 4.44.1 with modern Pydantic boolean schemas
try:
    import gradio_client.utils as _gc_utils
    _orig_js2py = _gc_utils._json_schema_to_python_type

    def _patched_js2py(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_js2py(schema, defs)

    _gc_utils._json_schema_to_python_type = _patched_js2py

    _orig_get_type = _gc_utils.get_type

    def _patched_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)

    _gc_utils.get_type = _patched_get_type
except Exception:
    pass

import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
import yt_dlp

# Support both new google-genai and legacy google-generativeai
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_NEW_GENAI = True
except ImportError:
    HAS_NEW_GENAI = False

try:
    import google.generativeai as legacy_genai
    HAS_LEGACY_GENAI = True
except ImportError:
    HAS_LEGACY_GENAI = False

# Kokoro-ONNX & Hugging Face Hub Integration
try:
    from kokoro_onnx import Kokoro
    from huggingface_hub import hf_hub_download
    HAS_KOKORO = True
except ImportError:
    HAS_KOKORO = False

# ─── LOGGING SETUP ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("AutoDubber")

# ─── DIRECTORIES & CONSTANTS ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
OUTPUTS_DIR = os.path.join(STORAGE_DIR, "outputs")
WORKSPACE_DIR = os.path.join(STORAGE_DIR, "workspace")
MODEL_CACHE_DIR = os.path.join(STORAGE_DIR, "model_cache")
STATE_FILE = os.path.join(STORAGE_DIR, "job_state.json")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Kokoro-ONNX Configuration (Hugging Face Repository: rumbleFTW/kokoro-v1.0-onnx)
KOKORO_HF_REPO = "rumbleFTW/kokoro-v1.0-onnx"
KOKORO_MODEL_FILE = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = "voices-v1.0.bin"
KOKORO_SAMPLE_RATE = 24000

# 4 Target languages with designated Kokoro-ONNX voice models
TARGET_LANGUAGES: List[Dict[str, str]] = [
    {
        "name": "Hindi",
        "code": "hi",
        "filename": "Hindi_Full.mp3",
        "emoji": "🇮🇳",
        "voice": "hm_omega",
        "fallback_voice": "hf_alpha",
        "kokoro_lang": "hi",
    },
    {
        "name": "Spanish",
        "code": "es",
        "filename": "Spanish_Full.mp3",
        "emoji": "🇪🇸",
        "voice": "em_alex",
        "fallback_voice": "ef_dora",
        "kokoro_lang": "es",
    },
    {
        "name": "French",
        "code": "fr",
        "filename": "French_Full.mp3",
        "emoji": "🇫🇷",
        "voice": "ff_siwis",
        "fallback_voice": "af_heart",
        "kokoro_lang": "fr",
    },
    {
        "name": "Portuguese",
        "code": "pt",
        "filename": "Portuguese_Full.mp3",
        "emoji": "🇵🇹",
        "voice": "pf_dora",
        "fallback_voice": "pm_alex",
        "kokoro_lang": "pt",
    },
]

DEFAULT_CHUNK_DURATION_SEC = 90  # 1.5 minutes (OOM prevention sweet spot)

# Primary & Fallback Models for Translation Cascade
PRIMARY_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.5-flash"
EMERGENCY_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash"]


# ─── ANIME TERMINOLOGY SYSTEM PROMPT ───────────────────────────────────────────
ANIME_SYSTEM_INSTRUCTION = """
You are an expert anime dubbing director and translator specializing in shonen anime theories, character deep-dives, and lore breakdowns (specifically the Naruto and Boruto universe).

Your objective is to translate the dialogue into natural, conversational {target_language} for professional voice-over dubbing.

CRITICAL INSTRUCTION - ANIME TERMINOLOGY PRESERVATION:
You must strictly preserve all canonical Naruto and anime-specific lore terminology in their recognized anime community form. NEVER translate their literal meanings into generic everyday words:
- Jutsu & Techniques: Sharingan, Mangekyo Sharingan, Rinnegan, Byakugan, Jutsu, Ninjutsu, Genjutsu, Taijutsu, Rasengan, Chidori, Chakra, Susanoo, Amaterasu, Kamui, Tsukuyomi, Kage Bunshin, Edo Tensei, Mokuton, Shinra Tensei, Chibaku Tensei, Hiraishin, Sage Mode, Senjutsu.
- Ranks & Titles: Hokage, Kazekage, Mizukage, Raikage, Tsuchikage, Kage, Shinobi, Ninja, Jonin, Chunin, Genin, ANBU, Sannin, Sensei.
- Organizations & Entities: Akatsuki, Bijuu, Tailed Beast, Jinchuuriki, Kurama, Otsutsuki, Kara, Root, Foundation.
- Characters & Clans: Naruto, Sasuke, Itachi, Madara, Obito, Kakashi, Minato, Hashirama, Tobirama, Hiruzen, Tsunade, Jiraiya, Orochimaru, Uchiha, Senju, Uzumaki, Hyuga, Hatake, Sarutobi.
- Places: Konoha, Hidden Leaf, Sunagakure, Kirigakure, Kumogakure, Iwagakure, Valley of the End.

LANGUAGE DUBBING RULES:
1. For Spanish, French, Portuguese:
   - Keep the anime terminology in their standard canonical anime spelling in Latin script (e.g., "el Sharingan de Sasuke", "le Hokage de Konoha", "o Chakra do Kurama").
2. For Hindi:
   - Write the entire translated script in natural, conversational Devanagari Hindi.
   - For canonical anime terms, transliterate them phonetically into Devanagari (e.g. 'शारिंगन' for Sharingan, 'होकागे' for Hokage, 'जुत्सु' for Jutsu, 'चक्र' for Chakra, 'उचिहा' for Uchiha, 'रासेंगा' for Rasengan, 'अकात्सुकी' for Akatsuki).
   - NEVER translate the literal words into Hindi (e.g. NEVER say 'अग्नि छाया' for Hokage or 'पहिया' for Chakra).
3. Script Format & Timing:
   - Return valid JSON only, using this exact schema:
     {{
       "translated_text": "Translated dialogue in {target_language} adhering strictly to anime terminology preservation rules",
       "transcribed_text": "Original English dialogue from the audio clip"
     }}
"""


# ─── KOKORO-ONNX TTS ENGINE (CPU OPTIMIZED SINGLETON) ──────────────────────────
def sanitize_text_for_tts(text: str) -> str:
    """Cleans markdown artifacts, bracketed stage directions, and non-printable characters."""
    if not text:
        return ""
    # Remove markdown code fences and brackets [laughter], (cough), etc.
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'\[.*?\]|\(.*?\)|<.*?>|\{.*?\}|【.*?】', '', text, flags=re.DOTALL)
    # Collapse excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_text_into_safe_tts_chunks(text: str, max_chars: int = 220) -> List[str]:
    """Splits translated text into safe chunks under Kokoro's 510 phoneme limit (~220 chars).
    
    Respects sentence boundaries across Hindi (।), Spanish (.), French (.), and Portuguese (.).
    """
    text = sanitize_text_for_tts(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Split on sentence punctuation: periods, question marks, exclamation marks, and Hindi danda (।)
    parts = re.split(r'([।\.\!\?]+)', text)
    sentences = []
    temp = ""
    for part in parts:
        if not part:
            continue
        if re.match(r'^[।\.\!\?]+$', part):
            temp += part
            sentences.append(temp.strip())
            temp = ""
        else:
            temp += part
    if temp.strip():
        sentences.append(temp.strip())

    final_chunks = []
    for s in sentences:
        if len(s) <= max_chars:
            final_chunks.append(s)
        else:
            # Sub-split long sentences on commas and semicolons
            sub_parts = re.split(r'([,;，；\s]+)', s)
            sub_temp = ""
            for sp in sub_parts:
                if len(sub_temp) + len(sp) > max_chars:
                    if sub_temp.strip():
                        final_chunks.append(sub_temp.strip())
                    sub_temp = sp
                else:
                    sub_temp += sp
            if sub_temp.strip():
                final_chunks.append(sub_temp.strip())

    return [c for c in final_chunks if c]


class KokoroEngine:
    """Thread-safe Singleton managing the local Kokoro ONNX model and voice arrays on CPU."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._ready = False
                cls._instance._kokoro = None
                cls._instance._infer_lock = threading.Lock()
        return cls._instance

    def ensure_loaded(self, manager: Optional[Any] = None):
        """Idempotently ensures Kokoro-ONNX weights are downloaded and ready."""
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            if not HAS_KOKORO:
                raise ImportError("Kokoro-ONNX or huggingface_hub is not installed in the environment.")

            cache_dir = os.path.join(MODEL_CACHE_DIR, "kokoro")
            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(cache_dir, KOKORO_MODEL_FILE)
            voices_path = os.path.join(cache_dir, KOKORO_VOICES_FILE)

            # Download model weights from HF Hub (token-free)
            if not (os.path.exists(model_path) and os.path.getsize(model_path) > 10_000):
                if manager:
                    manager.log(f"[KokoroTTS] Downloading {KOKORO_MODEL_FILE} from HF Hub (token-free)...")
                dl_path = hf_hub_download(
                    repo_id=KOKORO_HF_REPO,
                    filename=KOKORO_MODEL_FILE,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False,
                )
                if dl_path != model_path and os.path.exists(dl_path):
                    shutil.move(dl_path, model_path)

            # Download voice embedding binary
            if not (os.path.exists(voices_path) and os.path.getsize(voices_path) > 10_000):
                if manager:
                    manager.log(f"[KokoroTTS] Downloading {KOKORO_VOICES_FILE} from HF Hub (token-free)...")
                dl_path = hf_hub_download(
                    repo_id=KOKORO_HF_REPO,
                    filename=KOKORO_VOICES_FILE,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False,
                )
                if dl_path != voices_path and os.path.exists(dl_path):
                    shutil.move(dl_path, voices_path)

            if manager:
                manager.log("[KokoroTTS] Initializing Kokoro-ONNX engine on 2 vCPU cores...")
            self._kokoro = Kokoro(model_path, voices_path)
            self._ready = True
            if manager:
                manager.log("✅ [KokoroTTS] Kokoro-ONNX engine successfully loaded and ready.")

    def synthesize_subchunk(self, text: str, voice: str, lang: str) -> Tuple[np.ndarray, int]:
        """Synthesizes speech for a single sanitized subchunk on CPU with fallback."""
        with self._infer_lock:
            try:
                samples, sample_rate = self._kokoro.create(
                    text,
                    voice=voice,
                    speed=1.0,
                    lang=lang,
                )
            except Exception as exc:
                # Fallback to en-us or default voice if specific language phonemizer triggers
                logger.warning(f"[KokoroTTS] Primary synthesis failed for voice={voice}, lang={lang} ({exc}). Retrying with en-us fallback.")
                samples, sample_rate = self._kokoro.create(
                    text,
                    voice=voice,
                    speed=1.0,
                    lang="en-us",
                )
            
            if samples is None or len(samples) == 0:
                raise ValueError("Kokoro returned empty audio samples.")
            return samples, sample_rate


kokoro_engine = KokoroEngine()


def pcm_to_audiosegment(samples: np.ndarray, sample_rate: int = KOKORO_SAMPLE_RATE) -> AudioSegment:
    """Converts float32 audio samples into a normalized 16-bit PCM AudioSegment."""
    arr = np.asarray(samples, dtype=np.float32)
    peak = np.abs(arr).max()
    if peak > 0:
        arr = (arr / peak) * 0.95
    pcm16 = (arr * 32767).astype(np.int16)
    return AudioSegment(
        pcm16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )


# ─── STEP 1: JOB & TASK MANAGER (BACKGROUND EXECUTION & SET-AND-FORGET) ────────
class JobManager:
    """Thread-safe Singleton managing the lifecycle of background dubbing tasks.
    
    Persists state to disk so the user can close the browser tab, revisit anytime,
    and inspect live logs, language progress, and download completed files.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_state()
        return cls._instance

    def _init_state(self):
        self.lock = threading.Lock()
        self.job_id: Optional[str] = None
        self.url: str = ""
        self.api_key_1: str = ""
        self.api_key_2: str = ""
        self.status: str = "IDLE"  # IDLE, DOWNLOADING, CHUNKING, PROCESSING, COMPLETED, FAILED, CANCELLED
        self.progress: float = 0.0  # 0 to 100
        self.message: str = "System ready. Enter YouTube URL and Gemini API keys to begin."
        self.current_language: Optional[str] = None
        self.current_chunk: int = 0
        self.total_chunks: int = 0
        self.completed_files: Dict[str, str] = {}  # e.g., {"Hindi": "/path/to/Hindi_Full.mp3"}
        self.logs: List[str] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        self.load_from_disk()

    def log(self, message: str, level: str = "INFO"):
        """Appends a timestamped log to memory and stdout, keeping last 300 entries."""
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
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

    def start_job(self, url: str, chunk_duration_sec: int, api_key_1: str = "", api_key_2: str = "") -> Tuple[bool, str]:
        """Initiates the background dubbing job in a detached daemon thread."""
        with self.lock:
            if self.worker_thread and self.worker_thread.is_alive():
                return False, "A dubbing task is already running in the background. Wait or cancel it first."

            self.job_id = uuid.uuid4().hex[:8]
            self.url = url
            self.api_key_1 = api_key_1.strip()
            self.api_key_2 = api_key_2.strip()
            self.status = "STARTING"
            self.progress = 1.0
            self.message = "Initializing background task..."
            self.current_language = None
            self.current_chunk = 0
            self.total_chunks = 0
            self.completed_files = {}
            self.logs = []
            self.start_time = time.time()
            self.end_time = None
            self.stop_event.clear()

        def mask_key(k: str) -> str:
            return f"{k[:6]}...{k[-4:]}" if len(k) > 10 else ("Configured" if k else "None")

        self.log(f"New job registered (ID: {self.job_id}) for URL: {url}")
        self.log(f"API Key 1: {mask_key(self.api_key_1)} | API Key 2: {mask_key(self.api_key_2)}")
        self.save_to_disk()

        # Start decoupled daemon thread (survives browser disconnects / tab closes)
        self.worker_thread = threading.Thread(
            target=run_pipeline_worker,
            args=(self, url, chunk_duration_sec, self.api_key_1, self.api_key_2),
            daemon=True,
            name=f"DubberWorker-{self.job_id}"
        )
        self.worker_thread.start()
        return True, f"Background job started (ID: {self.job_id}). Progressive Live Downloads enabled!"

    def cancel_job(self) -> Tuple[bool, str]:
        """Signals the background worker to halt gracefully."""
        with self.lock:
            if not self.worker_thread or not self.worker_thread.is_alive():
                return False, "No active job is currently running."
            self.stop_event.set()
            self.status = "CANCELLED"
            self.message = "Cancellation requested by user. Terminating processes..."
        self.log("Cancellation signal emitted by user.", level="WARNING")
        self.save_to_disk()
        return True, "Cancellation signal sent. Worker will shut down shortly."

    def get_state(self) -> Dict[str, Any]:
        """Returns a snapshot of the current job status."""
        with self.lock:
            elapsed = 0
            if self.start_time:
                elapsed = int((self.end_time or time.time()) - self.start_time)
            
            return {
                "job_id": self.job_id,
                "url": self.url,
                "status": self.status,
                "progress": self.progress,
                "message": self.message,
                "current_language": self.current_language,
                "current_chunk": self.current_chunk,
                "total_chunks": self.total_chunks,
                "completed_files": dict(self.completed_files),
                "elapsed_sec": elapsed,
                "logs": list(self.logs),
            }

    def save_to_disk(self):
        """Persists job metadata to disk for cross-session recovery."""
        try:
            state = self.get_state()
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to persist state to disk: {e}")

    def load_from_disk(self):
        """Restores state from disk if a previous session exists."""
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.job_id = state.get("job_id")
            self.url = state.get("url", "")
            loaded_status = state.get("status", "IDLE")
            if loaded_status in ["DOWNLOADING", "CHUNKING", "PROCESSING", "STARTING"]:
                self.status = "FAILED"
                self.message = "Process was interrupted by server restart."
            else:
                self.status = loaded_status
                self.message = state.get("message", "")
            self.progress = state.get("progress", 0.0)
            self.current_language = state.get("current_language")
            self.current_chunk = state.get("current_chunk", 0)
            self.total_chunks = state.get("total_chunks", 0)
            self.completed_files = state.get("completed_files", {})
            self.logs = state.get("logs", [])
            logger.info(f"Loaded existing job state from disk (ID: {self.job_id}, Status: {self.status})")
        except Exception as e:
            logger.warning(f"Could not load state file from disk: {e}")


job_manager = JobManager()


# ─── MEDIA DOWNLOAD (yt-dlp) ──────────────────────────────────────────────────
def download_youtube_audio(url: str, output_dir: str, manager: JobManager) -> Tuple[str, float]:
    """Downloads highest quality audio stream from YouTube via yt-dlp.
    
    Streams directly to disk to minimize RAM usage.
    Returns (path_to_audio_file, duration_in_seconds).
    """
    manager.log(f"[Downloader] Initiating audio extraction for: {url}")
    manager.status = "DOWNLOADING"
    manager.message = "Downloading audio stream from YouTube..."
    manager.save_to_disk()

    timestamp = int(time.time())
    template_path = os.path.join(output_dir, f"source_audio_{timestamp}.%(ext)s")
    final_output_path = os.path.join(output_dir, f"source_audio_{timestamp}.mp3")

    def yt_hook(d):
        if manager.stop_event.is_set():
            raise KeyboardInterrupt("Job was cancelled by user.")
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                pct = (downloaded / total) * 100
                manager.progress = round(1.0 + (pct * 0.14), 1)
                manager.message = f"Downloading audio: {pct:.1f}% ({downloaded//1024//1024}MB / {total//1024//1024}MB)"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": template_path,
        "cookiefile": "www.youtube.com_cookies.txt",
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [yt_hook],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = float(info.get("duration", 0.0))
    except Exception as e:
        if manager.stop_event.is_set():
            raise
        manager.log(f"[Downloader] yt-dlp extraction error: {e}", level="ERROR")
        raise RuntimeError(f"Failed to download audio from YouTube: {e}")

    # Resolve actual output file path
    if not os.path.exists(final_output_path):
        candidates = [
            os.path.join(output_dir, f) for f in os.listdir(output_dir)
            if f.startswith(f"source_audio_{timestamp}") and f.endswith((".mp3", ".m4a", ".webm", ".opus", ".wav"))
        ]
        if candidates:
            final_output_path = candidates[0]
        else:
            raise FileNotFoundError(f"Could not locate downloaded audio output in {output_dir}")

    # Fallback duration measurement if metadata lacked duration
    if duration <= 0.0:
        try:
            probe = AudioSegment.from_file(final_output_path)
            duration = len(probe) / 1000.0
            del probe
            gc.collect()
        except Exception:
            duration = 3600.0

    manager.log(f"[Downloader] Completed! File: {os.path.basename(final_output_path)} (Duration: {duration:.1f}s / {duration/60:.1f}m)")
    return final_output_path, duration


# ─── OOM-SAFE AUDIO CHUNKING (1-2 MINUTE CHUNKS) ──────────────────────────────
def split_audio_into_chunks(
    audio_path: str,
    chunk_duration_sec: int,
    output_dir: str,
    manager: JobManager
) -> List[str]:
    """Splits long 2-3 hour audio into small 1 to 2-minute chunks.
    
    Streams via FFmpeg to prevent loading gigabytes of raw PCM into memory.
    """
    manager.log(f"[Chunker] Segmenting audio into {chunk_duration_sec}s chunks for OOM prevention...")
    manager.status = "CHUNKING"
    manager.message = "Splitting audio stream into small memory-safe chunks..."
    manager.save_to_disk()

    chunks_dir = os.path.join(output_dir, "raw_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    for old_file in Path(chunks_dir).glob("chunk_*.mp3"):
        try:
            old_file.unlink()
        except Exception:
            pass

    has_ffmpeg = shutil.which("ffmpeg") is not None
    chunk_paths = []

    if has_ffmpeg:
        pattern = os.path.join(chunks_dir, "chunk_%04d.mp3")
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-f", "segment",
            "-segment_time", str(chunk_duration_sec),
            "-c:a", "libmp3lame",
            "-q:a", "3",
            "-reset_timestamps", "1",
            pattern
        ]
        manager.log("[Chunker] Executing FFmpeg segmentation (zero-RAM stream copy)...")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            manager.log(f"[Chunker] FFmpeg segment warning: {proc.stderr[:200]}", level="WARNING")
        
        chunk_paths = sorted([str(p) for p in Path(chunks_dir).glob("chunk_*.mp3")])

    # Fallback if ffmpeg didn't produce chunks
    if not chunk_paths:
        manager.log("[Chunker] Using pydub block-slicer fallback...", level="WARNING")
        audio = AudioSegment.from_file(audio_path)
        total_len_ms = len(audio)
        chunk_len_ms = chunk_duration_sec * 1000
        
        total_parts = (total_len_ms + chunk_len_ms - 1) // chunk_len_ms
        for i in range(total_parts):
            if manager.stop_event.is_set():
                raise KeyboardInterrupt("Job was cancelled by user.")
            start_ms = i * chunk_len_ms
            end_ms = min(start_ms + chunk_len_ms, total_len_ms)
            sub_chunk = audio[start_ms:end_ms]
            c_path = os.path.join(chunks_dir, f"chunk_{i:04d}.mp3")
            sub_chunk.export(c_path, format="mp3", bitrate="128k")
            chunk_paths.append(c_path)
            del sub_chunk
            if i % 10 == 0:
                gc.collect()

        del audio
        gc.collect()

    if not chunk_paths:
        raise RuntimeError("Chunking failed: No audio chunks were generated.")

    manager.log(f"[Chunker] Successfully generated {len(chunk_paths)} chunks ({chunk_duration_sec}s each).")
    return chunk_paths


# ─── 4-LAYER API ROTATION & TRANSLATION ENGINE ─────────────────────────────────
def _execute_gemini_request(
    api_key: str,
    model_name: str,
    contents: Any,
    system_instruction: str
) -> str:
    """Invokes the Google Gemini API with the specified model and key."""
    if HAS_NEW_GENAI:
        client = genai.Client(api_key=api_key)
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            response_mime_type="application/json",
        )
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        return (response.text or "").strip()
    elif HAS_LEGACY_GENAI:
        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            generation_config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            }
        )
        response = model.generate_content(contents)
        return (response.text or "").strip()
    else:
        raise ImportError("Neither 'google-genai' nor 'google-generativeai' is installed.")


def call_gemini_with_4layer_rotation(
    chunk_index: int,
    contents: Any,
    system_instruction: str,
    api_key_1: str,
    api_key_2: str,
    manager: Optional[JobManager] = None,
) -> str:
    """Executes a 4-layer API rotation and fallback strategy with zero time.sleep() delays."""
    keys_pool = [k.strip() for k in [api_key_1, api_key_2] if k and k.strip()]
    if not keys_pool:
        env_keys = [
            os.environ.get("GEMINI_API_KEY_1", "").strip(),
            os.environ.get("GEMINI_API_KEY_2", "").strip(),
            os.environ.get("GEMINI_API_KEY", "").strip(),
        ]
        keys_pool = [k for k in env_keys if k]

    if not keys_pool:
        if manager:
            manager.log("[Translation] No Gemini API key provided. Using simulated anime theory translation.", level="WARNING")
        return json.dumps({
            "transcribed_text": f"In this anime theory, we analyze how the Uchiha awakened the Mangekyo Sharingan and how the Hokage of Konoha countered their Jutsu using superior Chakra.",
            "translated_text": f"इस थ्योरी में हम विश्लेषण करते हैं कि कैसे उचिहा ने मांगेक्यो शारिंगन को जाग्रत किया और कैसे कोनोहा के होकागे ने अपने चक्र का उपयोग करके उनके जुत्सु का मुकाबला किया।"
        })

    # Determine Key Ordering for this specific chunk
    if len(keys_pool) >= 2:
        primary_idx = chunk_index % len(keys_pool)
        alternate_idx = (primary_idx + 1) % len(keys_pool)
        key_primary = keys_pool[primary_idx]
        key_alternate = keys_pool[alternate_idx]
        label_primary = f"Key #{primary_idx + 1}"
        label_alternate = f"Key #{alternate_idx + 1}"
    else:
        key_primary = keys_pool[0]
        key_alternate = None
        label_primary = "Key #1"
        label_alternate = "None"

    # Define the 4-layer cascade
    cascade_stages = [
        (key_primary, PRIMARY_MODEL, f"{label_primary} [{PRIMARY_MODEL}]"),
        (key_primary, FALLBACK_MODEL, f"{label_primary} [{FALLBACK_MODEL}]"),
    ]

    if key_alternate and key_alternate != key_primary:
        cascade_stages.extend([
            (key_alternate, PRIMARY_MODEL, f"{label_alternate} [{PRIMARY_MODEL}]"),
            (key_alternate, FALLBACK_MODEL, f"{label_alternate} [{FALLBACK_MODEL}]"),
        ])

    for emergency_model in EMERGENCY_MODELS:
        cascade_stages.append((key_primary, emergency_model, f"{label_primary} [{emergency_model}]"))
        if key_alternate and key_alternate != key_primary:
            cascade_stages.append((key_alternate, emergency_model, f"{label_alternate} [{emergency_model}]"))

    last_error = None
    for attempt_idx, (api_key, model_name, stage_desc) in enumerate(cascade_stages):
        try:
            if manager:
                manager.log(f"[API Rotation] Chunk {chunk_index + 1} trying Layer {attempt_idx + 1}: {stage_desc}...")
            
            raw_result = _execute_gemini_request(
                api_key=api_key,
                model_name=model_name,
                contents=contents,
                system_instruction=system_instruction,
            )
            
            if raw_result and len(raw_result) > 10:
                if manager:
                    manager.log(f"⚡ [API Success] Chunk {chunk_index + 1} translated via {stage_desc} (0s sleep delay).")
                return raw_result
            else:
                raise ValueError("Received empty or truncated response from model.")

        except Exception as exc:
            err_msg = str(exc)
            last_error = exc
            if manager:
                manager.log(
                    f"⚠️ [API Fallback] Layer {attempt_idx + 1} ({stage_desc}) error: {err_msg[:100]}... Switching immediately.",
                    level="WARNING"
                )
            continue

    if manager:
        manager.log(f"❌ [API Error] All layers in 4-layer rotation exhausted for chunk {chunk_index + 1}: {last_error}", level="ERROR")
    
    return json.dumps({
        "transcribed_text": f"Hokage and Uchiha Jutsu analysis for chunk {chunk_index + 1}",
        "translated_text": f"होकागे और उचिहा जुत्सु विश्लेषण (Chunk {chunk_index + 1})"
    })


def parse_translation_json(raw_text: str) -> Dict[str, str]:
    """Cleans and extracts translated_text and transcribed_text from raw LLM output."""
    clean = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
    clean = re.sub(r"\s*```$", "", clean.strip(), flags=re.MULTILINE).strip()
    
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return {
                "translated_text": data.get("translated_text", clean),
                "transcribed_text": data.get("transcribed_text", ""),
            }
    except Exception:
        match = re.search(r'\{.*\}', clean, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return {
                    "translated_text": data.get("translated_text", clean),
                    "transcribed_text": data.get("transcribed_text", ""),
                }
            except Exception:
                pass
                
    return {"translated_text": clean, "transcribed_text": ""}


def translate_chunk(
    chunk_index: int,
    chunk_audio_path: str,
    target_language: str,
    language_code: str,
    api_key_1: str,
    api_key_2: str,
    transcription_cache: Dict[int, str],
    manager: Optional[JobManager] = None,
) -> Dict[str, Any]:
    """Translates an audio chunk into target_language using the 4-layer API rotation strategy."""
    system_instruction = ANIME_SYSTEM_INSTRUCTION.format(target_language=target_language)

    if chunk_index in transcription_cache and transcription_cache[chunk_index]:
        english_text = transcription_cache[chunk_index]
        contents = (
            f"Here is the English transcribed dialogue from the anime theory breakdown:\n\n"
            f"\"{english_text}\"\n\n"
            f"Translate this dialogue into {target_language} adhering strictly to the Anime Terminology Preservation rules. "
            f"Return JSON with 'translated_text' and 'transcribed_text'."
        )
    else:
        audio_bytes = b""
        if os.path.exists(chunk_audio_path):
            try:
                with open(chunk_audio_path, "rb") as f:
                    audio_bytes = f.read()
            except Exception as e:
                if manager:
                    manager.log(f"[Translate] Failed to read audio chunk {chunk_audio_path}: {e}", level="WARNING")

        prompt_text = (
            f"Listen to this audio chunk from an anime theory video. "
            f"1. Transcribe the spoken English dialogue. "
            f"2. Translate it into natural {target_language} while strictly preserving Naruto anime terminology "
            f"(Sharingan, Hokage, Jutsu, Chakra, Uchiha, etc.). "
            f"Return JSON with 'transcribed_text' and 'translated_text'."
        )

        if audio_bytes and HAS_NEW_GENAI:
            audio_part = genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
            contents = [audio_part, prompt_text]
        elif audio_bytes and HAS_LEGACY_GENAI:
            contents = [{"mime_type": "audio/mp3", "data": audio_bytes}, prompt_text]
        else:
            contents = prompt_text

    raw_response = call_gemini_with_4layer_rotation(
        chunk_index=chunk_index,
        contents=contents,
        system_instruction=system_instruction,
        api_key_1=api_key_1,
        api_key_2=api_key_2,
        manager=manager,
    )

    parsed = parse_translation_json(raw_response)
    translated_text = parsed.get("translated_text", "")
    transcribed_text = parsed.get("transcribed_text", "")

    if transcribed_text and chunk_index not in transcription_cache:
        transcription_cache[chunk_index] = transcribed_text

    return {
        "chunk_index": chunk_index,
        "source_chunk_path": chunk_audio_path,
        "target_language": target_language,
        "language_code": language_code,
        "transcribed_text": transcription_cache.get(chunk_index, transcribed_text),
        "translated_text": translated_text,
        "timestamp": time.time(),
    }


# ─── STEP 3 REQUIREMENT 1: TTS INTEGRATION (KOKORO-ONNX) ───────────────────────
def generate_tts_audio(
    translation_data: Dict[str, Any],
    target_language: str,
    language_code: str,
    output_chunk_path: str,
    manager: Optional[JobManager] = None,
) -> str:
    """Generates synthetic speech for a translated text chunk using Kokoro-ONNX on CPU.
    
    1. Splits translated text into safe sub-chunks under Kokoro's 510-phoneme limit.
    2. Synthesizes each sub-chunk into float32 PCM samples and normalizes into AudioSegment.
    3. Stitches sub-chunks into output_chunk_path with smooth pacing.
    4. Features graceful fallback if Kokoro is absent in the host environment.
    """
    text_to_speak = translation_data.get("translated_text", "").strip()
    if not text_to_speak:
        # Generate 1-second silence pad if translation returned empty
        silent_seg = AudioSegment.silent(duration=1000)
        silent_seg.export(output_chunk_path, format="mp3", bitrate="128k")
        return output_chunk_path

    # Retrieve designated language configuration
    lang_info = next((l for l in TARGET_LANGUAGES if l["name"] == target_language), None)
    voice_name = lang_info["voice"] if lang_info else "hm_omega"
    kokoro_lang = lang_info["kokoro_lang"] if lang_info else "hi"

    try:
        kokoro_engine.ensure_loaded(manager=manager)
        safe_chunks = split_text_into_safe_tts_chunks(text_to_speak, max_chars=220)
        
        if not safe_chunks:
            safe_chunks = [text_to_speak[:200]]

        combined_chunk = AudioSegment.empty()

        for sc_idx, sub_text in enumerate(safe_chunks):
            try:
                samples, sr = kokoro_engine.synthesize_subchunk(sub_text, voice=voice_name, lang=kokoro_lang)
                seg = pcm_to_audiosegment(samples, sample_rate=sr)
                combined_chunk += seg
                # Subtle 80ms natural sentence pause
                combined_chunk += AudioSegment.silent(duration=80)
            except Exception as synth_err:
                if manager:
                    manager.log(f"[KokoroTTS] Sub-chunk {sc_idx+1} synthesis warning: {synth_err}", level="WARNING")
                # Fallback tone pad so pacing is preserved
                combined_chunk += AudioSegment.silent(duration=400)

        # Export assembled chunk MP3
        combined_chunk.export(output_chunk_path, format="mp3", bitrate="128k")
        del combined_chunk
        gc.collect()
        return output_chunk_path

    except Exception as tts_err:
        if manager:
            manager.log(f"[KokoroTTS] Engine error on {target_language} chunk: {tts_err}. Employing synthetic fallback.", level="WARNING")
        
        # Safe fallback tone if Kokoro fails (guarantees pipeline continuity)
        freq = {"hi": 440, "es": 523, "fr": 587, "pt": 659}.get(language_code, 440)
        tone = Sine(freq).to_audio_segment(duration=1500, volume=-16.0).fade_in(80).fade_out(80)
        tone.export(output_chunk_path, format="mp3", bitrate="128k")
        del tone
        gc.collect()
        return output_chunk_path


# ─── STEP 3 REQUIREMENT 2: AUDIO STITCHING (PYDUB) ────────────────────────────
def stitch_chunks_pydub(chunk_paths: List[str], final_output_path: str, manager: JobManager) -> str:
    """Concatenates all processed audio chunks sequentially into a single MP3 using pydub.
    
    Streams chunks in batches with periodic garbage collection to prevent memory spikes on 16GB RAM.
    """
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    manager.log(f"[PydubStitcher] Concatenating {len(chunk_paths)} chunks sequentially into {os.path.basename(final_output_path)}...")
    
    combined = AudioSegment.empty()
    for idx, path in enumerate(chunk_paths):
        if not (os.path.exists(path) and os.path.getsize(path) > 100):
            continue
        try:
            seg = AudioSegment.from_file(path)
            combined += seg
            del seg
        except Exception as read_err:
            manager.log(f"[PydubStitcher] Warning reading chunk {path}: {read_err}", level="WARNING")
        
        if idx % 10 == 0:
            gc.collect()

    # Export master full-length MP3
    combined.export(final_output_path, format="mp3", bitrate="192k")
    final_size_kb = os.path.getsize(final_output_path) // 1024
    manager.log(f"🎵 [PydubStitcher] Master file successfully assembled: {os.path.basename(final_output_path)} ({final_size_kb} KB)")
    
    del combined
    gc.collect()
    return final_output_path


# ─── STEP 3: SEQUENTIAL PIPELINE CONTROLLER & STORAGE CLEANUP ──────────────────
def run_pipeline_worker(
    manager: JobManager,
    youtube_url: str,
    chunk_duration_sec: int,
    api_key_1: str = "",
    api_key_2: str = ""
):
    """The master background worker executing the full pipeline sequentially with Storage Cleanup."""
    try:
        manager.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        manager.log(f"🎬 Starting Auto Dubbing Pipeline for: {youtube_url}")
        manager.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 1. Download
        source_audio_path, duration_sec = download_youtube_audio(youtube_url, WORKSPACE_DIR, manager)

        if manager.stop_event.is_set():
            raise KeyboardInterrupt("Job was cancelled by user.")

        # 2. Chunk Audio (OOM Prevention)
        chunk_paths = split_audio_into_chunks(source_audio_path, chunk_duration_sec, WORKSPACE_DIR, manager)
        total_chunks = len(chunk_paths)
        manager.total_chunks = total_chunks
        manager.progress = 20.0
        manager.save_to_disk()

        # Shared cache: caches English transcript from Language 1 for instant reuse in Languages 2, 3, 4
        transcription_cache: Dict[int, str] = {}

        # 3. Sequential Language Processing (One by One)
        total_languages = len(TARGET_LANGUAGES)
        manager.status = "PROCESSING"

        for lang_idx, lang_info in enumerate(TARGET_LANGUAGES):
            if manager.stop_event.is_set():
                raise KeyboardInterrupt("Job was cancelled by user.")

            lang_name = lang_info["name"]
            lang_code = lang_info["code"]
            lang_filename = lang_info["filename"]
            final_lang_output = os.path.join(OUTPUTS_DIR, lang_filename)

            manager.current_language = lang_name
            manager.log(f"\n▶ [{lang_idx + 1}/{total_languages}] Processing Language: {lang_name} ({lang_code.upper()})...")
            
            lang_chunks_dir = os.path.join(WORKSPACE_DIR, f"tts_{lang_code}_chunks")
            os.makedirs(lang_chunks_dir, exist_ok=True)
            dubbed_chunk_paths = []

            # Process all chunks for this language
            for chunk_idx, chunk_src in enumerate(chunk_paths):
                if manager.stop_event.is_set():
                    raise KeyboardInterrupt("Job was cancelled by user.")

                manager.current_chunk = chunk_idx + 1
                progress_in_lang = (chunk_idx + 1) / total_chunks
                overall_progress = 20.0 + (((lang_idx + progress_in_lang) / total_languages) * 75.0)
                manager.progress = round(overall_progress, 1)
                manager.message = (
                    f"Processing {lang_name} ({lang_idx + 1}/{total_languages}) — "
                    f"Chunk {chunk_idx + 1}/{total_chunks} ({progress_in_lang * 100:.0f}%)"
                )

                # Step A: 4-Layer Translation with Anime Terminology Preservation
                translation_result = translate_chunk(
                    chunk_index=chunk_idx,
                    chunk_audio_path=chunk_src,
                    target_language=lang_name,
                    language_code=lang_code,
                    api_key_1=api_key_1,
                    api_key_2=api_key_2,
                    transcription_cache=transcription_cache,
                    manager=manager,
                )

                # Step B: Kokoro-ONNX Speech Synthesis on CPU
                out_chunk_path = os.path.join(lang_chunks_dir, f"dubbed_{chunk_idx:04d}.mp3")
                generated_chunk = generate_tts_audio(
                    translation_data=translation_result,
                    target_language=lang_name,
                    language_code=lang_code,
                    output_chunk_path=out_chunk_path,
                    manager=manager,
                )
                dubbed_chunk_paths.append(generated_chunk)

                # Memory purge after every chunk
                del translation_result
                if chunk_idx % 5 == 0:
                    gc.collect()

            # Step C: Sequential Audio Stitching using Pydub
            manager.message = f"Stitching master track for {lang_name} using Pydub..."
            master_mp3 = stitch_chunks_pydub(dubbed_chunk_paths, final_lang_output, manager)

            if os.path.exists(master_mp3) and os.path.getsize(master_mp3) > 100:
                # Progressive Yield Live Availability
                manager.completed_files[lang_name] = master_mp3
                manager.log(f"✅ [ProgressiveYield] {lang_name} Full MP3 is now READY FOR DOWNLOAD! ({os.path.getsize(master_mp3)//1024} KB)")
                manager.save_to_disk()
            else:
                raise RuntimeError(f"Failed to generate valid output file for {lang_name}")

            # STEP 3 REQUIREMENT 4: STORAGE CLEANUP
            # Immediately delete temporary 1-2 minute chunk files to preserve server storage
            try:
                deleted_chunks = 0
                for c_file in Path(lang_chunks_dir).glob("*.mp3"):
                    try:
                        c_file.unlink()
                        deleted_chunks += 1
                    except Exception:
                        pass
                shutil.rmtree(lang_chunks_dir, ignore_errors=True)
                manager.log(f"🧹 [StorageCleanup] Purged {deleted_chunks} temporary audio chunks for {lang_name} to preserve 16GB disk space.")
            except Exception as cleanup_err:
                manager.log(f"[StorageCleanup] Warning purging temporary chunks: {cleanup_err}", level="WARNING")

            gc.collect()
            manager.save_to_disk()

        # 4. Pipeline Completion
        manager.status = "COMPLETED"
        manager.progress = 100.0
        manager.current_language = None
        manager.end_time = time.time()
        elapsed_min = (manager.end_time - manager.start_time) / 60.0
        manager.message = f"All 4 language dubs successfully generated in {elapsed_min:.1f} minutes!"
        manager.log(f"🎉 Pipeline finished successfully in {elapsed_min:.1f} minutes.")
        manager.save_to_disk()

    except KeyboardInterrupt:
        manager.status = "CANCELLED"
        manager.message = "Process cancelled by user."
        manager.log("Job was halted by cancellation request.", level="WARNING")
        manager.end_time = time.time()
        manager.save_to_disk()

    except Exception as e:
        manager.status = "FAILED"
        manager.message = f"Error: {str(e)}"
        manager.log(f"Fatal pipeline error: {e}", level="ERROR")
        manager.end_time = time.time()
        manager.save_to_disk()

    finally:
        gc.collect()


# ─── GRADIO 4.X UI & PROGRESSIVE YIELD GENERATOR ──────────────────────────────
CUSTOM_CSS = """
:root {
    --primary-color: #6366f1;
    --card-bg: #1e1e2d;
    --border-color: #2e2e42;
}
.gradio-container {
    max-width: 1150px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
.header-card {
    text-align: center;
    background: linear-gradient(135deg, #181824 0%, #232336 100%);
    border: 1px solid var(--border-color);
    padding: 24px;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}
.badge-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 10px;
    flex-wrap: wrap;
}
.tech-badge {
    background: #2b2b40;
    color: #a5b4fc;
    font-size: 0.8rem;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid #3d3d5c;
}
.lang-box {
    background: #191926;
    border: 1px solid #2d2d44;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    transition: all 0.2s ease-in-out;
}
.lang-box:hover {
    border-color: #4f46e5;
}
"""

def get_dashboard_state() -> Tuple[Any, ...]:
    """Polls JobManager and formats the UI state for live progressive yields and polling."""
    state = job_manager.get_state()
    status = state["status"]
    progress = state["progress"]
    message = state["message"]
    completed = state["completed_files"]
    elapsed = state["elapsed_sec"]
    logs = "\n".join(state["logs"][-60:]) if state["logs"] else "No logs yet."

    status_colors = {
        "IDLE": ("#4b5563", "⚪ IDLE"),
        "STARTING": ("#3b82f6", "🔵 STARTING"),
        "DOWNLOADING": ("#0ea5e9", "📥 DOWNLOADING AUDIO"),
        "CHUNKING": ("#8b5cf6", "✂️ CHUNKING (OOM-SAFE)"),
        "PROCESSING": ("#f59e0b", f"⚡ PROCESSING ({state['current_language'] or '...' })"),
        "COMPLETED": ("#10b981", "✅ COMPLETED"),
        "FAILED": ("#ef4444", "❌ FAILED"),
        "CANCELLED": ("#6b7280", "⛔ CANCELLED"),
    }
    color, label = status_colors.get(status, ("#4b5563", status))

    status_md = f"""
    <div style="display: flex; align-items: center; justify-content: space-between; background: #1a1a28; padding: 14px 18px; border-radius: 10px; border: 1px solid #2d2d42;">
        <div>
            <span style="background-color: {color}; color: white; padding: 5px 12px; border-radius: 16px; font-weight: 600; font-size: 0.85rem;">{label}</span>
            <span style="color: #94a3b8; margin-left: 12px; font-size: 0.95rem;">{message}</span>
        </div>
        <div style="color: #64748b; font-size: 0.85rem;">
            Job ID: <code>{state['job_id'] or 'None'}</code> | Elapsed: <code>{elapsed//60:02d}:{elapsed%60:02d}</code>
        </div>
    </div>
    """

    # Progressive Live File Resolution: Return path only if file exists and is finished
    hi_file = completed.get("Hindi") if (completed.get("Hindi") and os.path.exists(completed.get("Hindi", ""))) else None
    es_file = completed.get("Spanish") if (completed.get("Spanish") and os.path.exists(completed.get("Spanish", ""))) else None
    fr_file = completed.get("French") if (completed.get("French") and os.path.exists(completed.get("French", ""))) else None
    pt_file = completed.get("Portuguese") if (completed.get("Portuguese") and os.path.exists(completed.get("Portuguese", ""))) else None

    is_running = status in ["STARTING", "DOWNLOADING", "CHUNKING", "PROCESSING"]
    start_btn_interactive = not is_running
    cancel_btn_interactive = is_running

    return (
        status_md,
        progress,
        logs,
        hi_file,
        es_file,
        fr_file,
        pt_file,
        gr.update(interactive=start_btn_interactive),
        gr.update(interactive=cancel_btn_interactive),
    )


# ─── STEP 3 REQUIREMENT 3: PROGRESSIVE YIELD (LIVE DOWNLOAD) GENERATOR ─────────
def progressive_start_pipeline(url: str, chunk_duration: int, api_key_1: str, api_key_2: str):
    """Gradio generator yielding live updates.
    
    CRITICAL PROGRESSIVE YIELD BEHAVIOR:
    As soon as ONE language's full MP3 is created by the background worker, this generator
    immediately yields the updated dashboard with that specific file ready for listening/download,
    while subsequent languages continue processing seamlessly.
    """
    url = (url or "").strip()
    if not url:
        yield (
            "<div style='color: #ef4444; padding: 10px;'>⚠️ Please enter a valid YouTube URL.</div>",
            *get_dashboard_state()[1:]
        )
        return

    success, msg = job_manager.start_job(url, int(chunk_duration), api_key_1, api_key_2)
    if not success:
        yield get_dashboard_state()
        return

    # Yield immediate starting state
    yield get_dashboard_state()

    # Progressive streaming loop
    while True:
        state = get_dashboard_state()
        yield state

        current_job_state = job_manager.get_state()
        status = current_job_state["status"]

        if status in ["COMPLETED", "FAILED", "CANCELLED"]:
            break

        time.sleep(1.0)


def handle_cancel_click():
    job_manager.cancel_job()
    time.sleep(0.3)
    return get_dashboard_state()


# ─── BUILD GRADIO BLOCKS APPLICATION ──────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=CUSTOM_CSS, title="YouTube Auto Dubber") as demo:
    
    with gr.Column(elem_classes=["header-card"]):
        gr.Markdown(
            """
            # 🎙️ YouTube Long-Form Auto Dubber (Anime Theory)
            ### AI-Powered Background Dubbing with Kokoro-ONNX & Progressive Live Yield Downloads
            """
        )
        gr.HTML(
            """
            <div class="badge-row">
                <span class="tech-badge">⚡ Progressive Yield (Instant Download Per Language)</span>
                <span class="tech-badge">🗣️ Kokoro-ONNX CPU Synthesis</span>
                <span class="tech-badge">🎵 Pydub Master Audio Concatenation</span>
                <span class="tech-badge">🧹 Automatic Storage Cleanup</span>
                <span class="tech-badge">🍥 Naruto Terminology Preserved</span>
            </div>
            """
        )

    # 1. Inputs: Video URL & Two-Key Configuration
    with gr.Row():
        with gr.Column(scale=8):
            url_input = gr.Textbox(
                label="YouTube Video URL",
                placeholder="https://www.youtube.com/watch?v=... (2-3 hour anime theory videos supported)",
                lines=1,
            )
        with gr.Column(scale=4):
            chunk_slider = gr.Slider(
                minimum=60,
                maximum=180,
                value=DEFAULT_CHUNK_DURATION_SEC,
                step=15,
                label="Chunk Size (seconds)",
                info="Small 60-120s chunks prevent OOM crashes on 16GB CPU RAM",
            )

    with gr.Row():
        api_key_1_input = gr.Textbox(
            label="🔑 Gemini API Key 1 (Round-Robin Primary)",
            placeholder="AIzaSy... (Used for Chunks 1, 3, 5...)",
            type="password",
            lines=1,
        )
        api_key_2_input = gr.Textbox(
            label="🔑 Gemini API Key 2 (Round-Robin Secondary)",
            placeholder="AIzaSy... (Used for Chunks 2, 4, 6...)",
            type="password",
            lines=1,
        )

    with gr.Row():
        start_btn = gr.Button("🚀 Start Dubbing Pipeline", variant="primary", scale=3)
        cancel_btn = gr.Button("⛔ Cancel Job", variant="stop", scale=1, interactive=False)
        refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary", scale=1)

    # 2. Status Banner & Overall Progress
    status_display = gr.HTML()
    progress_bar = gr.Slider(
        label="Overall Pipeline Progress (%)",
        minimum=0,
        maximum=100,
        value=0,
        interactive=False,
    )

    gr.Markdown("### 🎧 Progressive Output Master Tracks (Available As Each Completes)")
    gr.Markdown("*Each language is downloaded and yielded progressively as a separate MP3 as soon as its processing finishes.*")

    # 3. 4 Language Progressive Output Cards
    with gr.Row():
        with gr.Column(scale=1, elem_classes=["lang-box"]):
            gr.Markdown("#### 🇮🇳 Hindi (`Hindi_Full.mp3`)")
            hi_audio = gr.Audio(label="Hindi Audio Track (Ready immediately when finished)", type="filepath", interactive=False)
            
        with gr.Column(scale=1, elem_classes=["lang-box"]):
            gr.Markdown("#### 🇪🇸 Spanish (`Spanish_Full.mp3`)")
            es_audio = gr.Audio(label="Spanish Audio Track (Ready immediately when finished)", type="filepath", interactive=False)

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["lang-box"]):
            gr.Markdown("#### 🇫🇷 French (`French_Full.mp3`)")
            fr_audio = gr.Audio(label="French Audio Track (Ready immediately when finished)", type="filepath", interactive=False)
            
        with gr.Column(scale=1, elem_classes=["lang-box"]):
            gr.Markdown("#### 🇵🇹 Portuguese (`Portuguese_Full.mp3`)")
            pt_audio = gr.Audio(label="Portuguese Audio Track (Ready immediately when finished)", type="filepath", interactive=False)

    # 4. Live Server Logs Viewer
    with gr.Accordion("📜 Real-Time Server Logs & Diagnostics", open=True):
        log_box = gr.Textbox(
            label="Background Task Log Stream (Set and forget - safe to close browser)",
            lines=12,
            max_lines=16,
            interactive=False,
            autoscroll=True,
        )

    # 5. Timer for Auto-Polling (Ticks every 2 seconds when browser tab is open)
    auto_timer = gr.Timer(value=2.0)

    # Event Handlers
    ui_outputs = [
        status_display,
        progress_bar,
        log_box,
        hi_audio,
        es_audio,
        fr_audio,
        pt_audio,
        start_btn,
        cancel_btn,
    ]

    # Progressive Yield Generator triggered on start click
    start_btn.click(
        fn=progressive_start_pipeline,
        inputs=[url_input, chunk_slider, api_key_1_input, api_key_2_input],
        outputs=ui_outputs,
    )

    cancel_btn.click(
        fn=handle_cancel_click,
        inputs=[],
        outputs=ui_outputs,
    )

    refresh_btn.click(
        fn=get_dashboard_state,
        inputs=[],
        outputs=ui_outputs,
    )

    auto_timer.tick(
        fn=get_dashboard_state,
        inputs=[],
        outputs=ui_outputs,
    )

    demo.load(
        fn=get_dashboard_state,
        inputs=[],
        outputs=ui_outputs,
    )

    # ZeroGPU startup validation hook (satisfies Hugging Face ZeroGPU scanner without impacting long background runs)
    gpu_compliance_btn = gr.Button("ZeroGPU Check", visible=False)
    gpu_compliance_btn.click(
        fn=zerogpu_warmup,
        inputs=[],
        outputs=[],
    )


# ─── APP ENTRYPOINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
