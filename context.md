# Project Context: YouTube Long-Form Auto Dubber (Anime Theory)

## Target Platform
- **Hugging Face Spaces**: Free CPU Tier (2 vCPU, 16GB RAM, ~50GB persistent storage).
- **Python**: 3.11 with Gradio 4.44.1 SDK.
- **System Packages**: `ffmpeg`, `espeak-ng`.

## Architectural Pillars

### 1. Set & Forget Background Processing
- Detached daemon worker threads via `JobManager`.
- The user can start a 2-3 hour job and close the browser immediately without interrupting execution.
- Cross-session persistence (`storage/job_state.json`) lets users reload the page anytime to view status, logs, and downloads.

### 2. OOM-Prevention Audio Chunking
- Long 2-3 hour videos are split into 60-120 second chunks (default 90s).
- Direct stream segmentation via FFmpeg (`-f segment`) incurs negligible (<30MB) RAM overhead, completely preventing Out-Of-Memory crashes on 16GB RAM.

### 3. Sequential Language Processing & Progressive Yield
- Processes each language completely before moving to the next:
  1. Hindi ➔ `storage/outputs/Hindi_Full.mp3`
  2. Spanish ➔ `storage/outputs/Spanish_Full.mp3`
  3. French ➔ `storage/outputs/French_Full.mp3`
  4. Portuguese ➔ `storage/outputs/Portuguese_Full.mp3`
- **Progressive Yield (Generator Live Streaming)**:
  - Implemented as a Gradio generator (`progressive_start_pipeline`).
  - As soon as ONE language's full MP3 is created, the UI immediately yields that file for instant listening/download without waiting for the remaining 3 languages.
- **Zero-RAM Master Concatenation**:
  - Sequential concatenation via FFmpeg concat demuxer streaming chunks directly on disk (<15MB RAM).
- **Automatic Storage Cleanup**:
  - Once a language's full MP3 is verified and saved, the 1-2 minute temporary chunk directory is immediately purged, saving disk space on the 16GB Hugging Face instance.

### 4. 100% Lazy-Loaded Multi-Model TTS Architecture
- **Zero Global Scope Overhead**: The Space boots up in <1s with 0 MB of models in RAM. No startup timeouts (503).
- Model weights are downloaded and loaded strictly inside worker functions triggered by user action:
  - **Hindi (`hi`)**: `Tharshan/indicf5_hindi-english_code_switch`
  - **Spanish (`es`)**: `neuphonic/neutts-nano-spanish-q8-gguf`
  - **French (`fr`)**: `neuphonic/neutts-nano-french-q8-gguf`
  - **Portuguese (`pt`)**: `facebook/mms-tts-por` (or Piper)
- **Language Memory Swapping**: Only ONE language model resides in RAM at any time. When a language completes, its weights are instantly freed and memory is purged via `gc.collect()`.

### 5. Dynamic Gemini Model Discovery & Smart Key Rotation
- **Two-Key Round-Robin & 429 Failover**:
  - Uses `GEMINI_API_KEY_1` as primary and `GEMINI_API_KEY_2` as secondary.
  - Dynamically discovers real models (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash`). Zero hardcoded model names.
  - Safe 15 RPM throttling pacer (4.2s pre-call and 4.5s post-chunk pacing).
- **Anime Terminology Preservation**:
  - System prompt strictly instructs Gemini to preserve canonical Naruto anime lore (Sharingan, Hokage, Jutsu, Chakra, Uchiha, Rasengan, etc.).
  - Never translates literal meanings into generic terms (e.g. Hokage is never translated to 'fire shadow' or 'अग्नि छाया').
- **Transcription Caching**:
  - The English transcript is generated and cached during the Language 1 (Hindi) pass.
  - Languages 2 (Spanish), 3 (French), and 4 (Portuguese) reuse the cached transcript directly, eliminating redundant audio processing and guaranteeing perfect cross-language sync.
