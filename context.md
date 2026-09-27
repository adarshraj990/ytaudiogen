# Project Context: YouTube Long-Form Auto Dubber (Anime Theory)

## Target Platform
- **Google Colab & Standalone Python**: T4 / CPU runtime.
- **Python**: 3.10+ with Gradio.
- **System Packages**: `ffmpeg`.
- **Zero Hugging Face Footprint**: 100% free of Hugging Face Hub, `transformers`, `tokenizers`, and multi-GB weight downloads.

## Architectural Pillars

### 1. Set & Forget Background Processing
- Detached daemon worker threads via `JobManager`.
- The user can start a dubbing job and monitor progress or reload the page without interrupting execution.
- Cross-session persistence (`storage/job_state.json`) with auto-purge on startup to prevent zombie task freezes.
- One-click **"🧹 Reset Standby"** button to instantly clear any stale state.

### 2. Fast Concurrent Neural TTS Engine
- Powered by `edge-tts` (Microsoft Neural Voice synthesis).
- Natural Hindi, Indian English, US English, and regional Indian voices.
- Concurrent chunk synthesis via asyncio tasks: synthesizes dozens of dialogue lines in seconds.
- 0 KB crash checks with automated 1-retry fallback.

### 3. FFmpeg Silent Canvas Alignment
- Generates a completely silent base canvas audio of exact `total_duration` seconds via `anullsrc`.
- Overlays each synthesized sentence chunk at its exact SRT `start_time` timestamp.
- Trims or pads boundaries to guarantee an exact match with the target video duration.
- Exports master `.wav` file into `storage/outputs/`.

### 4. Zero Dependency Conflicts
- No heavy PyTorch model weights or Hugging Face tokenizers.
- Rapid pip installation (<10s) in Google Colab with zero library version collisions.
