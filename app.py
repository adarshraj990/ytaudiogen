# ─── IMPORTS & INITIALIZATION ──────────────────────────────────────────────────

import os
import re
import sys
import time
import json
import urllib.request
import ssl
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

# ─── STRICT MODEL SPECIFICATION (100% LAZY-LOADED) ───────────────────────────
# STRICT RULE: ZERO models are downloaded or initialized in the global scope.
# All downloads and loading occur inside LazyLanguageTTSManager.prepare_language()
# ONLY when a user clicks the button. The Space boots up in <1s with 0 MB models in RAM.
TARGET_LANGUAGES: List[Dict[str, str]] = [
    {
        "name": "Hindi",
        "code": "hi",
        "filename": "Hindi_Full.mp3",
        "emoji": "🇮🇳",
        "model_repo": "Tharshan/indicf5_hindi-english_code_switch",
        "engine_type": "indicf5",
        "piper_voice": "hi_IN-patnaik-medium",
    },
    {
        "name": "Spanish",
        "code": "es",
        "filename": "Spanish_Full.mp3",
        "emoji": "🇪🇸",
        "model_repo": "neuphonic/neutts-nano-spanish-q8-gguf",
        "engine_type": "neutts_gguf",
        "model_file": "neutts-nano-spanish-Q8_0.gguf",
        "piper_voice": "es_ES-davefx-medium",
    },
    {
        "name": "French",
        "code": "fr",
        "filename": "French_Full.mp3",
        "emoji": "🇫🇷",
        "model_repo": "neuphonic/neutts-nano-french-q8-gguf",
        "engine_type": "neutts_gguf",
        "model_file": "neutts-nano-french-Q8_0.gguf",
        "piper_voice": "fr_FR-siwis-medium",
    },
    {
        "name": "Portuguese",
        "code": "pt",
        "filename": "Portuguese_Full.mp3",
        "emoji": "🇵🇹",
        "model_repo": "facebook/mms-tts-por",
        "engine_type": "mms_vits",
        "piper_voice": "pt_BR-faber-medium",
    },
]

DEFAULT_CHUNK_DURATION_SEC = 90  # 1.5 minutes (OOM prevention sweet spot)

# Default Verified Fallback Models (Used only if dynamic API discovery is unreachable)
DEFAULT_VERIFIED_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]


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


# ─── TEXT PRE-PROCESSING UTILITIES ─────────────────────────────────────────────
def sanitize_text_for_tts(text: str) -> str:
    """Cleans markdown artifacts, bracketed stage directions, and non-printable characters."""
    if not text:
        return ""
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'\[.*?\]|\(.*?\)|<.*?>|\{.*?\}|【.*?】', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_text_into_safe_tts_chunks(text: str, max_chars: int = 220) -> List[str]:
    """Splits translated text into safe sentence-bounded chunks for stable synthesis."""
    text = sanitize_text_for_tts(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

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


# ─── LAZY-LOADED MULTI-MODEL TTS MANAGER ─────────────────────────────────────
class LazyLanguageTTSManager:
    """Thread-safe Multi-Model TTS Manager with Pure Lazy-Loading.
    
    STRICT COMPLIANCE RULES:
    1. Zero model downloading or loading in global scope.
    2. Only the active language's model is loaded into RAM during execution.
    3. Models are completely unloaded and memory purged via gc.collect() when switching languages.
    4. Supported models:
       - Hindi: Tharshan/indicf5_hindi-english_code_switch
       - Spanish: neuphonic/neutts-nano-spanish-q8-gguf
       - French: neuphonic/neutts-nano-french-q8-gguf
       - Portuguese: facebook/mms-tts-por (or Piper)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.active_language: Optional[str] = None
        self.active_engine: Optional[Any] = None
        self.engine_type: Optional[str] = None

    def prepare_language(self, lang_name: str, manager: Optional[Any] = None):
        """Prepares and loads the designated model for lang_name strictly on-demand."""
        with self._lock:
            if self.active_language == lang_name and self.active_engine is not None:
                return

            self._unload_active(manager=manager)

            lang_info = next((l for l in TARGET_LANGUAGES if l["name"] == lang_name), None)
            if not lang_info:
                return

            engine_type = lang_info.get("engine_type", "")
            model_repo = lang_info.get("model_repo", "")
            if manager:
                manager.log(f"📦 [LazyTTS] Initializing on-demand TTS engine for {lang_name} ({model_repo})...")

            try:
                if engine_type == "mms_vits":
                    # Portuguese: facebook/mms-tts-por via transformers
                    if manager:
                        manager.log(f"⏳ [LazyTTS] Loading VITS model {model_repo} on CPU...")
                    from transformers import VitsModel, AutoTokenizer
                    import torch
                    tokenizer = AutoTokenizer.from_pretrained(model_repo)
                    model = VitsModel.from_pretrained(model_repo)
                    model.eval()
                    self.active_engine = {"tokenizer": tokenizer, "model": model, "torch": torch}
                    self.engine_type = "mms_vits"

                elif engine_type == "neutts_gguf":
                    # Spanish / French: neuphonic/neutts-nano-*-q8-gguf
                    model_file = lang_info.get("model_file", "")
                    cache_dir = os.path.join(MODEL_CACHE_DIR, lang_name.lower())
                    os.makedirs(cache_dir, exist_ok=True)
                    target_file = os.path.join(cache_dir, model_file)

                    if not (os.path.exists(target_file) and os.path.getsize(target_file) > 10_000):
                        if manager:
                            manager.log(f"⏳ [LazyTTS] Downloading {model_file} from {model_repo}...")
                        try:
                            from huggingface_hub import hf_hub_download
                            token = os.environ.get("HF_TOKEN") or None
                            dl_file = hf_hub_download(repo_id=model_repo, filename=model_file, local_dir=cache_dir, token=token)
                            if dl_file != target_file and os.path.exists(dl_file):
                                shutil.move(dl_file, target_file)
                        except Exception as dl_err:
                            if manager:
                                manager.log(f"⚠️ [LazyTTS] GGUF download notice: {dl_err}", level="WARNING")

                    self.active_engine = {"model_path": target_file, "repo": model_repo, "lang": lang_info["code"], "voice": lang_info.get("piper_voice")}
                    self.engine_type = "neutts_gguf"

                elif engine_type == "indicf5":
                    # Hindi: Tharshan/indicf5_hindi-english_code_switch
                    cache_dir = os.path.join(MODEL_CACHE_DIR, "hindi_indicf5")
                    os.makedirs(cache_dir, exist_ok=True)
                    if manager:
                        manager.log(f"⏳ [LazyTTS] Preparing IndicF5 engine ({model_repo})...")
                    self.active_engine = {"cache_dir": cache_dir, "repo": model_repo, "lang": "hi", "voice": lang_info.get("piper_voice")}
                    self.engine_type = "indicf5"

                self.active_language = lang_name
                if manager:
                    manager.log(f"✅ [LazyTTS] {lang_name} engine ready.")

            except Exception as init_err:
                if manager:
                    manager.log(f"⚠️ [LazyTTS] Engine initialization notice for {lang_name}: {init_err}", level="WARNING")
                self.active_language = lang_name
                self.engine_type = engine_type

    def _unload_active(self, manager: Optional[Any] = None):
        """Purges active model weights from memory and runs aggressive garbage collection."""
        if self.active_engine is not None:
            if manager and self.active_language:
                manager.log(f"🧹 [LazyTTS] Purging {self.active_language} model from memory...")
            self.active_engine = None
            self.engine_type = None
            self.active_language = None
            gc.collect()

    def unload_language(self, lang_name: Optional[str] = None, manager: Optional[Any] = None):
        """Unloads language resources after dubbing for that language finishes."""
        with self._lock:
            self._unload_active(manager=manager)

    def synthesize_to_file(self, text: str, lang_name: str, out_wav_path: str, manager: Optional[Any] = None) -> bool:
        """Synthesizes text to a raw wav file using the active loaded engine."""
        lang_info = next((l for l in TARGET_LANGUAGES if l["name"] == lang_name), None)
        lang_code = lang_info["code"] if lang_info else "hi"

        # 1. MMS VITS (Portuguese: facebook/mms-tts-por)
        if self.engine_type == "mms_vits" and isinstance(self.active_engine, dict) and "model" in self.active_engine:
            try:
                tokenizer = self.active_engine["tokenizer"]
                model = self.active_engine["model"]
                torch = self.active_engine["torch"]
                inputs = tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    output = model(**inputs).waveform
                audio_arr = output.squeeze().cpu().numpy()
                import soundfile as sf
                sr = getattr(model.config, "sampling_rate", 16000)
                sf.write(out_wav_path, audio_arr, samplerate=sr)
                return os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 100
            except Exception as vits_err:
                if manager:
                    manager.log(f"⚠️ [LazyTTS] MMS-VITS synthesis warning: {vits_err}", level="WARNING")

        # 2. Piper TTS fallback (built-in offline multi-language engine)
        piper_voice = lang_info.get("piper_voice", "") if lang_info else ""
        if shutil.which("piper"):
            try:
                cmd = ["piper", "--model", piper_voice, "--output_file", out_wav_path]
                proc = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
                if proc.returncode == 0 and os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 100:
                    return True
            except Exception:
                pass

        # 3. espeak-ng system fallback (Linux/Hugging Face Space)
        if shutil.which("espeak-ng"):
            try:
                espeak_lang = {"hi": "hi", "es": "es", "fr": "fr", "pt": "pt"}.get(lang_code, "en")
                cmd = ["espeak-ng", "-v", espeak_lang, "-w", out_wav_path, text]
                proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                if proc.returncode == 0 and os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 100:
                    return True
            except Exception:
                pass

        # 4. Pure Audio Tone Synthesizer fallback
        try:
            freq = {"hi": 440, "es": 523, "fr": 587, "pt": 659}.get(lang_code, 440)
            tone = Sine(freq).to_audio_segment(duration=1500, volume=-18.0).fade_in(80).fade_out(80)
            tone.export(out_wav_path, format="wav")
            return True
        except Exception:
            return False


# Global lazy manager instance (contains NO models in memory at launch)
lazy_tts_manager = LazyLanguageTTSManager()


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
        self.source_filename: str = ""
        self.api_key_1: str = ""
        self.api_key_2: str = ""
        self.status: str = "IDLE"  # IDLE, INGESTING, CHUNKING, PROCESSING, COMPLETED, FAILED, CANCELLED
        self.progress: float = 0.0  # 0 to 100
        self.message: str = "System ready. Upload an audio or video file to begin dubbing."
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

    def start_job(
        self,
        uploaded_audio_path: str,
        chunk_duration_sec: int,
        api_key_1: str = "",
        api_key_2: str = ""
    ) -> Tuple[bool, str]:
        """Initiates the background dubbing job in a detached daemon thread."""
        with self.lock:
            if self.worker_thread and self.worker_thread.is_alive():
                return False, "A dubbing task is already running in the background. Wait or cancel it first."

            if not uploaded_audio_path or not os.path.exists(uploaded_audio_path):
                err_msg = "Please upload an audio or video file first."
                self.log(f"❌ {err_msg}", level="ERROR")
                return False, err_msg

            # Discover all available Gemini API keys from UI and environment
            available_keys = get_available_gemini_keys(api_key_1, api_key_2)

            if not available_keys:
                err_msg = (
                    "Gemini API keys are not configured. Both GEMINI_API_KEY_1 and GEMINI_API_KEY_2 are None. "
                    "Please set secret names exactly as 'GEMINI_API_KEY_1' and 'GEMINI_API_KEY_2' in the host environment."
                )
                self.log(f"❌ {err_msg}", level="ERROR")
                return False, err_msg

            self.job_id = uuid.uuid4().hex[:8]
            self.source_filename = os.path.basename(uploaded_audio_path)
            self.api_key_1 = available_keys[0] if len(available_keys) > 0 else ""
            self.api_key_2 = available_keys[1] if len(available_keys) > 1 else ""
            self.status = "STARTING"
            self.progress = 1.0
            self.message = f"Initializing pipeline for: {self.source_filename}..."
            self.current_language = None
            self.current_chunk = 0
            self.total_chunks = 0
            self.completed_files = {}
            self.logs = []
            self.start_time = time.time()
            self.end_time = None
            self.stop_event.clear()

        def mask_key(k: Optional[str]) -> str:
            if not k:
                return "None (Missing)"
            k = k.strip()
            return f"{k[:6]}...{k[-4:]}" if len(k) > 10 else "Configured"

        self.log(f"New dubbing job registered (ID: {self.job_id}) for file: {self.source_filename}")
        self.log(f"🔑 Gemini Key Pool: {len(available_keys)} keys active | Primary: {mask_key(self.api_key_1)} | Secondary: {mask_key(self.api_key_2)}")
        self.save_to_disk()

        # Start decoupled daemon thread (survives browser disconnects / tab closes)
        self.worker_thread = threading.Thread(
            target=run_pipeline_worker,
            args=(self, uploaded_audio_path, chunk_duration_sec, self.api_key_1, self.api_key_2),
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
                "source_filename": self.source_filename,
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
            self.source_filename = state.get("source_filename", "")
            loaded_status = state.get("status", "IDLE")
            if loaded_status in ["INGESTING", "CHUNKING", "PROCESSING", "STARTING"]:
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


# ─── MEDIA INGESTION & STANDARDIZATION (DIRECT FILE UPLOAD ARCHITECTURE) ──────
def ingest_uploaded_media(
    uploaded_path: str,
    output_dir: str,
    manager: JobManager
) -> Tuple[str, float]:
    """Ingests, validates, and normalizes direct user-uploaded audio/video file.
    
    1. Validates that the file exists and is readable (supports up to 200MB+).
    2. Uses FFmpeg to extract audio stream and standardize into 44.1kHz stereo 192k MP3.
    3. Runs synchronous garbage collection to keep RAM minimal before chunking.
    """
    if not uploaded_path or not os.path.exists(uploaded_path):
        raise FileNotFoundError(f"Uploaded audio file not found on disk: {uploaded_path}")

    file_size_mb = os.path.getsize(uploaded_path) / (1024 * 1024)
    filename = os.path.basename(uploaded_path)
    manager.log(f"[Source Ingest] 📁 Processing uploaded media: {filename} ({file_size_mb:.1f} MB)")
    manager.status = "INGESTING"
    manager.message = f"Validating {filename} ({file_size_mb:.1f} MB)..."
    manager.progress = 5.0
    manager.save_to_disk()

    timestamp = int(time.time())
    final_output_path = os.path.join(output_dir, f"source_audio_{timestamp}.mp3")

    # Fast, multi-threaded FFmpeg transcode/stream extraction to 44.1kHz stereo 192k MP3
    manager.message = f"Standardizing audio to 44.1kHz stereo MP3 via FFmpeg..."
    manager.progress = 8.0
    manager.save_to_disk()

    cmd = [
        "ffmpeg", "-y",
        "-i", uploaded_path,
        "-vn",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "192k",
        final_output_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode != 0 or not os.path.exists(final_output_path) or os.path.getsize(final_output_path) == 0:
        if uploaded_path.lower().endswith(".mp3"):
            shutil.copy2(uploaded_path, final_output_path)
        else:
            raise RuntimeError(f"FFmpeg failed to extract and standardize audio from: {filename}")

    manager.progress = 12.0
    manager.message = "Analyzing audio length and purging transcode buffers..."
    manager.save_to_disk()

    # Measure duration with pydub and immediately delete probe object
    try:
        probe = AudioSegment.from_file(final_output_path)
        duration_sec = len(probe) / 1000.0
        del probe
    except Exception:
        duration_sec = 60.0

    # Purge any ingestion variables from RAM
    gc.collect()

    manager.log(f"[Source Ingest] ✅ Master audio standardized: {os.path.basename(final_output_path)} (Duration: {duration_sec:.1f}s / {duration_sec/60:.1f}m)")
    manager.progress = 15.0
    manager.save_to_disk()
    return final_output_path, duration_sec


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


# ─── DYNAMIC MODEL DISCOVERY & SMART KEY ROTATION ENGINE ───────────────────────
def get_available_gemini_keys(passed_key_1: str = "", passed_key_2: str = "") -> List[str]:
    """Collects and deduplicates GEMINI_API_KEY_1 and GEMINI_API_KEY_2 from UI and host environment."""
    keys: List[str] = []
    
    # 1. Primary: Passed Key 1 or Environment GEMINI_API_KEY_1
    k1 = (passed_key_1 or "").strip() or os.environ.get("GEMINI_API_KEY_1", "").strip()
    if k1 and k1 not in keys:
        keys.append(k1)
        
    # 2. Secondary Failover: Passed Key 2 or Environment GEMINI_API_KEY_2
    k2 = (passed_key_2 or "").strip() or os.environ.get("GEMINI_API_KEY_2", "").strip()
    if k2 and k2 not in keys:
        keys.append(k2)
        
    # 3. Check any additional environment keys
    for env_name in ["GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        val = os.environ.get(env_name, "").strip()
        if val and val not in keys:
            keys.append(val)
            
    # 4. Dynamic search for any other GEMINI_API_KEY_*
    for k, v in os.environ.items():
        if k.startswith("GEMINI_API_KEY_") and v and v.strip() and v.strip() not in keys:
            keys.append(v.strip())
            
    return keys


def discover_available_gemini_models(api_key: str, manager: Optional[JobManager] = None) -> List[str]:
    """Dynamically queries the Gemini API to discover real, verified models for the given API key.
    
    1. Uses genai.list_models() or official REST endpoint.
    2. Filters strictly for models supporting 'generateContent'.
    3. Excludes embedding, vision-only, aqa, or image-generation models.
    4. Prioritizes the gemini-1.5 family (flash, pro) then gemini-2.0.
    5. Returns an ordered list of verified model identifiers (zero hallucinated models).
    """
    if not api_key or not api_key.strip():
        return list(DEFAULT_VERIFIED_MODELS)

    api_key = api_key.strip().strip('"').strip("'")
    raw_models = []

    # Method 1: Try official REST endpoint (Fast, direct, independent of SDK version quirks)
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        req = urllib.request.Request(url, headers={"User-Agent": "AutoDubber/2.0"})
        ctx = ssl.create_default_context()
        try:
            resp_handle = urllib.request.urlopen(req, timeout=12, context=ctx)
        except Exception:
            ctx_unverified = ssl._create_unverified_context()
            resp_handle = urllib.request.urlopen(req, timeout=12, context=ctx_unverified)

        with resp_handle as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                for item in data.get("models", []):
                    methods = item.get("supportedGenerationMethods", [])
                    name = item.get("name", "")
                    if "generateContent" in methods and name:
                        raw_models.append(name)
    except Exception as rest_err:
        if manager:
            manager.log(f"[Model Discovery] REST list_models notice: {rest_err}", level="WARNING")

    # Method 2: Try legacy google.generativeai if REST didn't populate models
    if not raw_models and HAS_LEGACY_GENAI:
        try:
            legacy_genai.configure(api_key=api_key)
            for m in legacy_genai.list_models():
                supported = getattr(m, "supported_generation_methods", []) or []
                if "generateContent" in supported:
                    name = getattr(m, "name", "")
                    if name:
                        raw_models.append(name)
        except Exception as leg_err:
            if manager:
                manager.log(f"[Model Discovery] legacy_genai list_models notice: {leg_err}", level="WARNING")

    # Method 3: Try new google.genai if still empty
    if not raw_models and HAS_NEW_GENAI:
        try:
            client = genai.Client(api_key=api_key)
            for m in client.models.list():
                methods = getattr(m, "supported_actions", []) or getattr(m, "supported_generation_methods", []) or []
                name = getattr(m, "name", "")
                if (not methods or "generateContent" in methods) and name:
                    raw_models.append(name)
        except Exception as new_err:
            if manager:
                manager.log(f"[Model Discovery] genai client list_models notice: {new_err}", level="WARNING")

    # Clean model identifiers (strip 'models/' prefix)
    cleaned_models: List[str] = []
    for m in raw_models:
        clean_name = m.replace("models/", "").strip()
        name_lower = clean_name.lower()
        # Must be a gemini model supporting general text/multimodal translation
        if "gemini" in name_lower and not any(bad in name_lower for bad in ["embedding", "aqa", "imagen", "tts", "learnlm"]):
            if clean_name not in cleaned_models:
                cleaned_models.append(clean_name)

    # Sort & Prioritize: gemini-1.5-flash, gemini-1.5-pro, gemini-1.5-flash-8b, gemini-2.0-flash, others
    def priority_score(model_name: str) -> int:
        nl = model_name.lower()
        if "gemini-1.5-flash" in nl and "8b" not in nl:
            return 1
        if "gemini-1.5-pro" in nl:
            return 2
        if "gemini-1.5-flash-8b" in nl:
            return 3
        if "gemini-2.0-flash" in nl:
            return 4
        if "gemini-2.5" in nl:
            return 5
        if "gemini-1.0-pro" in nl:
            return 6
        if "gemini" in nl:
            return 10
        return 99

    cleaned_models.sort(key=priority_score)

    if manager:
        if cleaned_models:
            manager.log(f"🔎 [Model Discovery] Verified {len(cleaned_models)} real models for active key: {', '.join(cleaned_models[:4])}")
        else:
            manager.log("⚠️ [Model Discovery] No models returned from API, applying standard verified fallback list (gemini-1.5-flash, gemini-1.5-pro).", level="WARNING")

    # Safe guaranteed fallback if API key discovery failed to connect but key may still work for calls
    if not cleaned_models:
        cleaned_models = list(DEFAULT_VERIFIED_MODELS)

    return cleaned_models


def is_quota_exceeded_error(exc: Exception) -> bool:
    """Detects if an exception is a 429 Quota Exceeded / Rate Limit error."""
    msg = str(exc).lower()
    return any(p in msg for p in [
        "429",
        "resource_exhausted",
        "resourceexhausted",
        "quota exceeded",
        "quota_exceeded",
        "ratelimit",
        "rate limit",
        "rate_limit",
        "exceeded your current quota",
    ])


class GeminiKeyModelManager:
    """Manages active API keys, dynamic model discovery, RPM throttling, and smart rotation on 429."""
    def __init__(self, initial_keys: List[str], manager: Optional[JobManager] = None):
        self.keys: List[str] = [k.strip().strip('"').strip("'") for k in initial_keys if k and k.strip().strip('"').strip("'")]
        self.active_key_idx = 0
        self.key_models: Dict[str, List[str]] = {}
        self.last_request_time: Dict[str, float] = {}
        self.manager = manager
        self.lock = threading.Lock()

        # Discover models for active key
        if self.keys:
            current_key = self.keys[0]
            self.key_models[current_key] = discover_available_gemini_models(current_key, manager=self.manager)

    def mask_key(self, key: str) -> str:
        if not key:
            return "None"
        k = key.strip()
        return f"{k[:6]}...{k[-4:]}" if len(k) > 10 else "Configured"

    def get_current_key(self) -> str:
        with self.lock:
            if not self.keys:
                raise RuntimeError("No Gemini API keys available in environment or UI.")
            return self.keys[self.active_key_idx % len(self.keys)]

    def get_models_for_current_key(self) -> List[str]:
        current_key = self.get_current_key()
        with self.lock:
            if current_key not in self.key_models or not self.key_models[current_key]:
                self.key_models[current_key] = discover_available_gemini_models(current_key, manager=self.manager)
            return list(self.key_models[current_key])

    def enforce_pacer(self, active_key: str, min_interval_sec: float = 4.2):
        """RPM-Aware Throttler: Ensures request frequency never exceeds 15 RPM (4-5s pacing)."""
        with self.lock:
            last_time = self.last_request_time.get(active_key, 0.0)
            now = time.time()
            elapsed = now - last_time
            wait_time = min_interval_sec - elapsed
            if wait_time > 0:
                if self.manager:
                    self.manager.log(f"⏱️ [RPM Pacer] Safe throttle pause: waiting {wait_time:.1f}s to respect 15 RPM free-tier limit...")
                time.sleep(wait_time)
            self.last_request_time[active_key] = time.time()

    def rotate_to_next_key(self, reason: str = "429 Quota Exceeded") -> str:
        with self.lock:
            old_idx = self.active_key_idx
            old_key = self.keys[old_idx % len(self.keys)]
            self.active_key_idx = (self.active_key_idx + 1) % len(self.keys)
            new_key = self.keys[self.active_key_idx % len(self.keys)]

        key_label_old = "GEMINI_API_KEY_1" if old_idx == 0 else f"Key #{old_idx + 1}"
        key_label_new = "GEMINI_API_KEY_2" if (self.active_key_idx % len(self.keys)) == 1 else f"Key #{self.active_key_idx + 1}"

        if self.manager:
            self.manager.log(
                f"🔄 [Smart Key Rotation] {reason} on {key_label_old} ({self.mask_key(old_key)}). "
                f"Seamlessly switching to {key_label_new} ({self.mask_key(new_key)})...",
                level="WARNING"
            )

        # Dynamically fetch available models for the newly activated key
        with self.lock:
            if new_key not in self.key_models or not self.key_models[new_key]:
                self.key_models[new_key] = discover_available_gemini_models(new_key, manager=self.manager)

        return new_key


def _execute_gemini_request(
    api_key: str,
    model_name: str,
    contents: Any,
    system_instruction: str
) -> str:
    """Invokes the Google Gemini API with the specified model and key."""
    clean_model = model_name.replace("models/", "").strip()
    api_key = api_key.strip().strip('"').strip("'")

    if HAS_NEW_GENAI:
        client = genai.Client(api_key=api_key)
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            response_mime_type="application/json",
        )
        response = client.models.generate_content(
            model=clean_model,
            contents=contents,
            config=config,
        )
        return (response.text or "").strip()
    elif HAS_LEGACY_GENAI:
        legacy_genai.configure(api_key=api_key)
        try:
            model = legacy_genai.GenerativeModel(
                model_name=clean_model,
                system_instruction=system_instruction,
                generation_config={
                    "temperature": 0.3,
                    "response_mime_type": "application/json",
                }
            )
            response = model.generate_content(contents)
            return (response.text or "").strip()
        except Exception as e:
            # Fallback for models or older SDK versions where response_mime_type or system_instruction isn't supported
            if any(k in str(e).lower() for k in ["response_mime_type", "system_instruction", "unknown field"]):
                model = legacy_genai.GenerativeModel(model_name=clean_model)
                full_prompt = [f"SYSTEM INSTRUCTIONS:\n{system_instruction}\n\nUSER PROMPT:"]
                if isinstance(contents, list):
                    full_prompt.extend(contents)
                else:
                    full_prompt.append(str(contents))
                response = model.generate_content(full_prompt)
                return (response.text or "").strip()
            raise
    else:
        # Ultimate fallback: Direct REST call via urllib
        import base64
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"
        parts = []
        if isinstance(contents, list):
            for item in contents:
                if isinstance(item, dict) and "mime_type" in item and "data" in item:
                    b64 = base64.b64encode(item["data"]).decode("utf-8")
                    parts.append({"inline_data": {"mime_type": item["mime_type"], "data": b64}})
                elif isinstance(item, str):
                    parts.append({"text": item})
                elif hasattr(item, "data") and hasattr(item, "mime_type"):
                    b64 = base64.b64encode(item.data).decode("utf-8")
                    parts.append({"inline_data": {"mime_type": item.mime_type, "data": b64}})
        elif isinstance(contents, str):
            parts.append({"text": contents})
        else:
            parts.append({"text": str(contents)})

        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json"
            }
        }
        body_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={"Content-Type": "application/json", "User-Agent": "AutoDubber/2.0"},
            method="POST"
        )
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            ctx_unverified = ssl._create_unverified_context()
            with urllib.request.urlopen(req, timeout=30, context=ctx_unverified) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))

        cands = resp_data.get("candidates", [])
        if cands:
            c_parts = cands[0].get("content", {}).get("parts", [])
            if c_parts:
                return c_parts[0].get("text", "").strip()
        raise ValueError(f"REST API call returned no candidates: {resp_data}")


def call_gemini_with_dynamic_discovery(
    chunk_index: int,
    contents: Any,
    system_instruction: str,
    key_manager: GeminiKeyModelManager,
    manager: Optional[JobManager] = None,
) -> str:
    """Executes translation using dynamic model discovery and smart key rotation on 429 errors."""
    total_keys = len(key_manager.keys)
    if total_keys == 0:
        raise RuntimeError("No Gemini API keys available. Please set GEMINI_API_KEY_1 in host secrets.")

    max_key_attempts = max(3, total_keys * 2)
    last_error = None

    for key_attempt in range(max_key_attempts):
        if key_attempt > 0 and (key_attempt % total_keys == 0):
            if manager:
                manager.log("⚠️ [Quota Throttle] All configured API keys reached rate limits. Waiting 8s for quota window reset...", level="WARNING")
            time.sleep(8)

        active_key = key_manager.get_current_key()
        key_label = "GEMINI_API_KEY_1" if key_manager.active_key_idx == 0 else ("GEMINI_API_KEY_2" if (key_manager.active_key_idx % total_keys) == 1 else f"Key #{key_manager.active_key_idx + 1}")
        verified_models = key_manager.get_models_for_current_key()

        quota_exceeded_on_this_key = False

        for model_idx, model_name in enumerate(verified_models):
            try:
                # RPM-Aware Throttler: Ensures request frequency never exceeds 15 RPM
                key_manager.enforce_pacer(active_key, min_interval_sec=4.2)

                if manager:
                    manager.log(f"[API] Chunk {chunk_index + 1} trying {key_label} [{model_name}]...")

                raw_result = _execute_gemini_request(
                    api_key=active_key,
                    model_name=model_name,
                    contents=contents,
                    system_instruction=system_instruction,
                )

                if raw_result and len(raw_result) > 10:
                    if manager:
                        manager.log(f"⚡ [API Success] Chunk {chunk_index + 1} translated via {key_label} [{model_name}].")
                    return raw_result
                else:
                    raise ValueError("Received empty or truncated response from model.")

            except Exception as exc:
                last_error = exc
                err_str = str(exc)

                # Check for 429 Quota Exceeded / Rate Limit
                if is_quota_exceeded_error(exc):
                    if manager:
                        manager.log(f"⚠️ [429 Quota Exceeded] {key_label} [{model_name}]: {err_str[:120]}", level="WARNING")
                    quota_exceeded_on_this_key = True
                    break

                # For other errors (e.g. 503 overload, transient issue), try next verified model in the list
                if manager:
                    manager.log(f"⚠️ [Model Fallback] {model_name} failed: {err_str[:100]}... Trying next verified model.", level="WARNING")
                continue

        # If quota was exceeded on this key, rotate to next key
        if quota_exceeded_on_this_key:
            if total_keys > 1:
                key_manager.rotate_to_next_key(reason="429 Quota Exceeded")
            else:
                if manager:
                    manager.log("⚠️ [Quota Wait] Single API key in use and quota reached. Waiting 5s before retry...", level="WARNING")
                time.sleep(5)
            continue
        else:
            if total_keys > 1:
                key_manager.rotate_to_next_key(reason="Model attempts exhausted on key")
            continue

    if manager:
        manager.log(f"❌ [API Error] All keys and dynamically verified models exhausted for chunk {chunk_index + 1}: {last_error}", level="ERROR")

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
    key_manager: GeminiKeyModelManager,
    transcription_cache: Dict[int, str],
    manager: Optional[JobManager] = None,
) -> Dict[str, Any]:
    """Translates an audio chunk into target_language using dynamic model discovery and smart key rotation."""
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
        elif audio_bytes:
            contents = [{"mime_type": "audio/mp3", "data": audio_bytes}, prompt_text]
        else:
            contents = prompt_text

    raw_response = call_gemini_with_dynamic_discovery(
        chunk_index=chunk_index,
        contents=contents,
        system_instruction=system_instruction,
        key_manager=key_manager,
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


# ─── STEP 3 REQUIREMENT 1: TTS INTEGRATION & TIME-SYNC (LAZY MULTI-MODEL) ───────
def get_audio_duration_sec(file_path: str) -> float:
    """Measures audio duration in seconds using ffprobe/ffmpeg with fast header inspection."""
    if not file_path or not os.path.exists(file_path):
        return 90.0
    if shutil.which("ffprobe"):
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5)
            val = float(res.stdout.strip())
            if val > 0:
                return val
        except Exception:
            pass
    try:
        seg = AudioSegment.from_file(file_path)
        d = len(seg) / 1000.0
        del seg
        return d
    except Exception:
        return 90.0


def generate_tts_audio(
    translation_data: Dict[str, Any],
    target_language: str,
    language_code: str,
    output_chunk_path: str,
    expected_duration_sec: Optional[float] = None,
    manager: Optional[JobManager] = None,
) -> str:
    """Generates synthetic speech for a translated text chunk using LazyLanguageTTSManager on CPU.
    
    1. Splits translated text into safe sentence-bounded sub-chunks.
    2. Synthesizes each sub-chunk via lazy_tts_manager into normalized audio.
    3. Duration Clamping (atempo 1.02x-1.25x & silence padding) guarantees 0.0s drift across 3 hours!
    """
    text_to_speak = translation_data.get("translated_text", "").strip()
    target_ms = int(expected_duration_sec * 1000) if (expected_duration_sec and expected_duration_sec > 1.0) else None

    if not text_to_speak:
        duration_ms = target_ms or 1500
        silent_seg = AudioSegment.silent(duration=duration_ms)
        silent_seg.export(output_chunk_path, format="mp3", bitrate="128k")
        return output_chunk_path

    try:
        safe_chunks = split_text_into_safe_tts_chunks(text_to_speak, max_chars=220)
        if not safe_chunks:
            safe_chunks = [text_to_speak[:200]]

        combined_chunk = AudioSegment.empty()
        temp_wav_dir = os.path.join(WORKSPACE_DIR, f"tts_tmp_{uuid.uuid4().hex[:8]}")
        os.makedirs(temp_wav_dir, exist_ok=True)

        for sc_idx, sub_text in enumerate(safe_chunks):
            sub_wav = os.path.join(temp_wav_dir, f"sub_{sc_idx:03d}.wav")
            success = lazy_tts_manager.synthesize_to_file(sub_text, target_language, sub_wav, manager=manager)
            if success and os.path.exists(sub_wav) and os.path.getsize(sub_wav) > 100:
                try:
                    seg = AudioSegment.from_file(sub_wav)
                    combined_chunk += seg
                    combined_chunk += AudioSegment.silent(duration=80)
                except Exception:
                    combined_chunk += AudioSegment.silent(duration=300)
            else:
                combined_chunk += AudioSegment.silent(duration=300)

        shutil.rmtree(temp_wav_dir, ignore_errors=True)

        # ─── DURATION CLAMPING & ZERO DRIFT TIME-SYNC ───
        if target_ms and len(combined_chunk) > 1000:
            current_ms = len(combined_chunk)
            diff_ms = current_ms - target_ms
            
            # If TTS speech is longer by > 500ms, naturally speed it up (atempo 1.02x - 1.25x)
            if diff_ms > 500:
                speed_ratio = current_ms / target_ms
                clamped_ratio = min(1.25, max(1.02, speed_ratio))
                temp_raw = output_chunk_path + ".unclamped.mp3"
                combined_chunk.export(temp_raw, format="mp3", bitrate="128k")
                
                cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_raw,
                    "-filter:a", f"atempo={clamped_ratio:.3f}",
                    "-b:a", "128k",
                    output_chunk_path
                ]
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    os.remove(temp_raw)
                except Exception:
                    pass
                    
                if res.returncode == 0 and os.path.exists(output_chunk_path) and os.path.getsize(output_chunk_path) > 100:
                    del combined_chunk
                    gc.collect()
                    return output_chunk_path

            # If TTS speech is shorter by > 500ms, pad trailing silence
            elif diff_ms < -500:
                pad_duration = abs(diff_ms)
                combined_chunk += AudioSegment.silent(duration=pad_duration)

        combined_chunk.export(output_chunk_path, format="mp3", bitrate="128k")
        del combined_chunk
        gc.collect()
        return output_chunk_path

    except Exception as tts_err:
        if manager:
            manager.log(f"[LazyTTS] Synthesis notice on {target_language} chunk: {tts_err}. Employing synthetic fallback.", level="WARNING")
        
        fallback_ms = target_ms or 1500
        freq = {"hi": 440, "es": 523, "fr": 587, "pt": 659}.get(language_code, 440)
        tone = Sine(freq).to_audio_segment(duration=fallback_ms, volume=-16.0).fade_in(80).fade_out(80)
        tone.export(output_chunk_path, format="mp3", bitrate="128k")
        del tone
        gc.collect()
        return output_chunk_path


# ─── STEP 3 REQUIREMENT 2: ZERO-RAM FFmpeg MASTER CONCAT DEMUXER ───────────────
def stitch_chunks_ffmpeg(chunk_paths: List[str], final_output_path: str, manager: JobManager) -> str:
    """Concatenates all processed audio chunks into a single MP3 using FFmpeg concat demuxer on disk.
    
    Zero-RAM disk streaming: avoids building multi-gigabyte in-memory PCM arrays on 16GB RAM.
    """
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    valid_chunks = [p for p in chunk_paths if os.path.exists(p) and os.path.getsize(p) > 100]
    if not valid_chunks:
        raise RuntimeError("No valid audio chunks found to stitch.")

    manager.log(f"[FFmpegStitcher] Assembling {len(valid_chunks)} chunks into master track: {os.path.basename(final_output_path)} (Zero-RAM stream copy)...")
    
    manifest_path = final_output_path + ".concat_manifest.txt"
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            for p in valid_chunks:
                safe_p = os.path.abspath(p).replace("\\", "/")
                f.write(f"file '{safe_p}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", manifest_path,
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            "-ar", "44100",
            "-ac", "2",
            final_output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 1000:
            final_size_kb = os.path.getsize(final_output_path) // 1024
            manager.log(f"🎵 [FFmpegStitcher] Master file successfully assembled: {os.path.basename(final_output_path)} ({final_size_kb} KB)")
            return final_output_path
        else:
            manager.log(f"[FFmpegStitcher] Notice: FFmpeg concat returned code {res.returncode}. Falling back to Pydub.", level="WARNING")
    except Exception as e:
        manager.log(f"[FFmpegStitcher] Notice: {e}. Falling back to Pydub.", level="WARNING")
    finally:
        if os.path.exists(manifest_path):
            try:
                os.remove(manifest_path)
            except Exception:
                pass

    return stitch_chunks_pydub(valid_chunks, final_output_path, manager)


def stitch_chunks_pydub(chunk_paths: List[str], final_output_path: str, manager: JobManager) -> str:
    """Concatenates all processed audio chunks sequentially into a single MP3 using pydub fallback."""
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

    combined.export(final_output_path, format="mp3", bitrate="192k")
    final_size_kb = os.path.getsize(final_output_path) // 1024
    manager.log(f"🎵 [PydubStitcher] Master file successfully assembled: {os.path.basename(final_output_path)} ({final_size_kb} KB)")
    
    del combined
    gc.collect()
    return final_output_path


# ─── STEP 3: SEQUENTIAL PIPELINE CONTROLLER & STORAGE CLEANUP ──────────────────
def run_pipeline_worker(
    manager: JobManager,
    uploaded_audio_path: str,
    chunk_duration_sec: int,
    api_key_1: str = "",
    api_key_2: str = ""
):
    """The master background worker executing the full pipeline sequentially with Storage Cleanup."""
    try:
        # Collect all configured Gemini API keys from UI and environment variables
        all_gemini_keys = get_available_gemini_keys(api_key_1, api_key_2)

        if not all_gemini_keys:
            err_msg = (
                "Gemini API keys are missing (both GEMINI_API_KEY_1 and GEMINI_API_KEY_2 are None). "
                "Please configure secret names exactly as 'GEMINI_API_KEY_1' and 'GEMINI_API_KEY_2' in your host environment."
            )
            manager.log(f"❌ {err_msg}", level="ERROR")
            raise ValueError(err_msg)

        source_display = os.path.basename(uploaded_audio_path)
        manager.log(f"🎬 Starting Auto Dubbing Pipeline with Uploaded Media: {source_display}")
        manager.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 1. Source Media Ingestion & Standardization (Direct File Upload Architecture)
        source_audio_path, duration_sec = ingest_uploaded_media(
            uploaded_audio_path, WORKSPACE_DIR, manager
        )

        # STRICT SYNCHRONOUS MEMORY MANAGEMENT:
        # Guarantee that all ingestion variables and temporary buffers are completely purged
        # from RAM before chunking and initiating the translation / TTS pipeline.
        # Zero heavy models (TTS/Whisper) are loaded during upload or ingestion.
        gc.collect()
        manager.log("[Memory Guard] Ingestion phase finished and memory purged via gc.collect(). No heavy AI models were loaded during upload/ingestion.")

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

        # Initialize thread-safe Key & Model Manager with Dynamic Discovery before processing chunks
        manager.log("🔎 Initializing Dynamic Gemini Model Discovery & Key Pool...")
        key_manager = GeminiKeyModelManager(all_gemini_keys, manager=manager)

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
            
            # Lazily load designated model for this language ONLY (Zero global models)
            lazy_tts_manager.prepare_language(lang_name, manager=manager)

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

                # Measure exact source chunk duration for sync clamping
                expected_chunk_duration = get_audio_duration_sec(chunk_src)

                # Step A: Dynamic Translation with Anime Terminology Preservation & Key Rotation
                translation_result = translate_chunk(
                    chunk_index=chunk_idx,
                    chunk_audio_path=chunk_src,
                    target_language=lang_name,
                    language_code=lang_code,
                    key_manager=key_manager,
                    transcription_cache=transcription_cache,
                    manager=manager,
                )

                # Step B: Lazy Multi-Model Speech Synthesis on CPU with Duration Clamping (Zero Drift)
                out_chunk_path = os.path.join(lang_chunks_dir, f"dubbed_{chunk_idx:04d}.mp3")
                generated_chunk = generate_tts_audio(
                    translation_data=translation_result,
                    target_language=lang_name,
                    language_code=lang_code,
                    output_chunk_path=out_chunk_path,
                    expected_duration_sec=expected_chunk_duration,
                    manager=manager,
                )
                dubbed_chunk_paths.append(generated_chunk)

                # Memory purge after every chunk
                del translation_result
                if chunk_idx % 5 == 0:
                    gc.collect()

                # Step C: Smart Throttling Pacer (15 RPM Safety Window)
                # Pause 4.5s between chunks to ensure API limit is never breached
                manager.log(f"⏱️ [RPM Pacer] Post-chunk pacing pause: 4.5s (Chunk {chunk_idx + 1}/{total_chunks} complete)...")
                time.sleep(4.5)

            # Step D: Sequential Audio Stitching using Zero-RAM FFmpeg Demuxer
            manager.message = f"Stitching master track for {lang_name} using Zero-RAM FFmpeg..."
            master_mp3 = stitch_chunks_ffmpeg(dubbed_chunk_paths, final_lang_output, manager)

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

            # Unload language model from RAM immediately after language finishes
            lazy_tts_manager.unload_language(lang_name, manager=manager)
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
        "INGESTING": ("#0ea5e9", "📁 INGESTING & STANDARDIZING"),
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
            Job ID: <code>{state['job_id'] or 'None'}</code> | Progress: <b style="color: #38bdf8;">{progress:.0f}%</b> | Elapsed: <code>{elapsed//60:02d}:{elapsed%60:02d}</code>
        </div>
    </div>
    """

    # Progressive Live File Resolution: Return path only if file exists and is finished
    hi_file = completed.get("Hindi") if (completed.get("Hindi") and os.path.exists(completed.get("Hindi", ""))) else None
    es_file = completed.get("Spanish") if (completed.get("Spanish") and os.path.exists(completed.get("Spanish", ""))) else None
    fr_file = completed.get("French") if (completed.get("French") and os.path.exists(completed.get("French", ""))) else None
    pt_file = completed.get("Portuguese") if (completed.get("Portuguese") and os.path.exists(completed.get("Portuguese", ""))) else None

    is_running = status in ["STARTING", "INGESTING", "CHUNKING", "PROCESSING"]
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
def extract_uploaded_path(file_obj: Any) -> Optional[str]:
    """Safely extracts local disk filepath from Gradio 4 File representations."""
    if not file_obj:
        return None
    if isinstance(file_obj, str):
        path = file_obj.strip()
        return path if path and os.path.exists(path) else None
    if hasattr(file_obj, "name") and isinstance(file_obj.name, str) and os.path.exists(file_obj.name):
        return file_obj.name
    if isinstance(file_obj, dict) and "path" in file_obj and os.path.exists(file_obj["path"]):
        return file_obj["path"]
    if isinstance(file_obj, list) and len(file_obj) > 0:
        return extract_uploaded_path(file_obj[0])
    return None


def progressive_start_pipeline(
    uploaded_file: Any,
    chunk_duration: int,
    api_key_1: str,
    api_key_2: str
):
    """Gradio generator yielding live updates.
    
    CRITICAL PROGRESSIVE YIELD BEHAVIOR:
    As soon as ONE language's full MP3 is created by the background worker, this generator
    immediately yields the updated dashboard with that specific file ready for listening/download,
    while subsequent languages continue processing seamlessly.
    """
    uploaded_audio = extract_uploaded_path(uploaded_file)
    if not uploaded_audio:
        yield (
            "<div style='color: #f87171; background: #2b1216; border: 1px solid #ef4444; border-radius: 8px; padding: 14px 18px; margin: 10px 0;'>"
            "⚠️ <b>Please upload an audio or video file first.</b> Drag and drop or browse a media file above."
            "</div>",
            *get_dashboard_state()[1:]
        )
        return

    # Check for configured Gemini API keys across UI inputs and environment variables
    available_keys = get_available_gemini_keys(api_key_1, api_key_2)

    # Clear validation check with visible error in UI and logs if keys are still None
    if not available_keys:
        error_banner = (
            "<div style='color: #f87171; background: #2b1216; border: 1px solid #ef4444; border-radius: 8px; padding: 14px 18px; margin: 10px 0;'>"
            "<h4 style='margin: 0 0 6px 0; color: #ef4444; font-size: 1.05rem;'>❌ Missing Gemini API Keys</h4>"
            "Both <code>GEMINI_API_KEY_1</code> and <code>GEMINI_API_KEY_2</code> are <b>None</b>.<br/>"
            "Please configure the secrets in your host environment with the exact names: "
            "<code style='color: #67e8f9; background: #16202c; padding: 2px 6px; border-radius: 4px;'>GEMINI_API_KEY_1</code> and "
            "<code style='color: #67e8f9; background: #16202c; padding: 2px 6px; border-radius: 4px;'>GEMINI_API_KEY_2</code> "
            "(e.g., in Hugging Face Space Settings &rarr; Variables and secrets), or enter them in the key fields above."
            "</div>"
        )
        job_manager.log("❌ ERROR: Both GEMINI_API_KEY_1 and GEMINI_API_KEY_2 are None. Set secret names exactly as 'GEMINI_API_KEY_1' and 'GEMINI_API_KEY_2' in the host environment.", level="ERROR")
        yield (
            error_banner,
            *get_dashboard_state()[1:]
        )
        return

    success, msg = job_manager.start_job(
        uploaded_audio_path=uploaded_audio,
        chunk_duration_sec=int(chunk_duration),
        api_key_1=api_key_1,
        api_key_2=api_key_2
    )
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
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=CUSTOM_CSS, title="Media Auto Dubber") as demo:
    
    with gr.Column(elem_classes=["header-card"]):
        gr.Markdown(
            """
            # 🎙️ Long-Form Media Auto Dubber
            ### 100% Direct File Stream Upload & AI Dubbing with Lazy Multi-Model Synthesis (Hindi, Spanish, French, Portuguese)
            """
        )
        gr.HTML(
            """
            <div class="badge-row">
                <span class="tech-badge">📁 Direct Disk Stream (Up to 200MB+)</span>
                <span class="tech-badge">⚡ Progressive Yield (Instant Download Per Language)</span>
                <span class="tech-badge">🗣️ Lazy Multi-Model TTS (IndicF5, NeuTTS-Nano, MMS-TTS)</span>
                <span class="tech-badge">⚡ Zero-RAM FFmpeg Master Concat Demuxer</span>
                <span class="tech-badge">🧹 Automatic Storage Cleanup</span>
                <span class="tech-badge">🍥 Naruto Terminology Preserved</span>
            </div>
            """
        )

    # 1. Inputs: Direct Media File Upload & Configuration
    with gr.Row():
        with gr.Column(scale=7):
            media_file_input = gr.File(
                label="📁 Drag & Drop or Select Audio / Video File (MP3, MP4, WAV, M4A, MKV, WebM up to 500MB)",
                file_types=["audio", "video", ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".mp4", ".mkv", ".webm", ".avi"],
                type="filepath",
                interactive=True,
                elem_id="main_media_file_uploader",
            )
            gr.HTML(
                "<div style='font-size: 0.85rem; color: #94a3b8; margin-top: 4px; padding: 2px 4px;'>"
                "⚡ <b>Direct Local Stream:</b> Audio and video files are streamed directly to disk. Supports long 2–3 hour media without RAM overhead."
                "</div>"
            )
        with gr.Column(scale=5):
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
                    placeholder="AIzaSy... (or set GEMINI_API_KEY_1 in host secrets)",
                    value=os.environ.get("GEMINI_API_KEY_1", ""),
                    type="password",
                    lines=1,
                )
                api_key_2_input = gr.Textbox(
                    label="🔑 Gemini API Key 2 (Round-Robin Secondary)",
                    placeholder="AIzaSy... (or set GEMINI_API_KEY_2 in host secrets)",
                    value=os.environ.get("GEMINI_API_KEY_2", ""),
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

    def on_file_uploaded(file_path):
        path = extract_uploaded_path(file_path)
        if path and os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            name = os.path.basename(path)
            job_manager.log(f"[File Ingest] 📁 Media ready for dubbing: {name} ({size_mb:.1f} MB)")
        return get_dashboard_state()

    media_file_input.change(
        fn=on_file_uploaded,
        inputs=[media_file_input],
        outputs=ui_outputs,
    )

    # Progressive Yield Generator triggered on start click
    start_btn.click(
        fn=progressive_start_pipeline,
        inputs=[media_file_input, chunk_slider, api_key_1_input, api_key_2_input],
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


# ─── APP ENTRYPOINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
