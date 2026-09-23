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

### 3. Sequential Language Processing & Progressive Yield (Step 3)
- Processes each language completely before moving to the next:
  1. Hindi ➔ `storage/outputs/Hindi_Full.mp3`
  2. Spanish ➔ `storage/outputs/Spanish_Full.mp3`
  3. French ➔ `storage/outputs/French_Full.mp3`
  4. Portuguese ➔ `storage/outputs/Portuguese_Full.mp3`
- **Progressive Yield (Generator Live Streaming)**:
  - Implemented as a Gradio generator (`progressive_start_pipeline`).
  - As soon as ONE language's full MP3 is created, the UI immediately yields that file for instant listening/download without waiting for the remaining 3 languages.
- **Audio Stitching with Pydub**:
  - Sequential concatenation via `pydub` (`stitch_chunks_pydub`) streaming chunks in memory-safe batches.
- **Automatic Storage Cleanup**:
  - Once a language's full MP3 is verified and saved, the 1-2 minute temporary chunk directory is immediately purged, saving disk space on the 16GB Hugging Face instance.

### 4. Kokoro-ONNX CPU-Optimized TTS Engine (Step 3)
- Thread-safe `KokoroEngine` singleton running on 2 vCPU cores.
- Automatic, token-free download from Hugging Face Hub (`rumbleFTW/kokoro-v1.0-onnx`).
- Safe sub-chunking (<220 characters) respecting punctuation across Devanagari Hindi (`।`), Spanish, French, and Portuguese.
- Voice mappings:
  - Hindi (`hi`): `hm_omega` (male dramatic), fallback `hf_alpha` (female expressive)
  - Spanish (`es`): `em_alex`, fallback `ef_dora`
  - French (`fr`): `ff_siwis`, fallback `af_heart`
  - Portuguese (`pt`): `pf_dora`, fallback `pm_alex`

### 5. 4-Layer API Rotation Strategy & Translation Engine (Step 2)
- **Two-Key Round-Robin**:
  - Accepts Key 1 and Key 2 in the UI.
  - Alternates between Key 1 and Key 2 for every sequential chunk (Chunk 1: Key 1, Chunk 2: Key 2, etc.).
- **Model Fallback (Instant try-except with Zero `time.sleep()`)**:
  - Primary Model: `gemini-3.8-flash`
  - Fallback Model: `gemini-3.5-flash` (same key fallback on rate limit / 429)
  - Cross-Key Failover: Alternate key is used if the primary key hits persistent limits.
  - Emergency Models: `gemini-2.0-flash` / `gemini-1.5-flash` for high availability.
- **Anime Terminology Preservation**:
  - System prompt strictly instructs Gemini to preserve canonical Naruto anime lore (Sharingan, Hokage, Jutsu, Chakra, Uchiha, Rasengan, etc.).
  - Never translates literal meanings into generic terms (e.g. Hokage is never translated to 'fire shadow' or 'अग्नि छाया').
- **Transcription Caching**:
  - The English transcript is generated and cached during the Language 1 (Hindi) pass.
  - Languages 2 (Spanish), 3 (French), and 4 (Portuguese) reuse the cached transcript directly, eliminating redundant audio processing and guaranteeing perfect cross-language sync.
