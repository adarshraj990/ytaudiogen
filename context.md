# Project Context: YouTube Long-Form Auto Dubber (Indic-F5 Colab Edition)

## Target Platform
- **Google Colab (T4 GPU Runtime)**: 15.3GB VRAM, CUDA 12.
- **Python**: 3.10+ with Gradio.
- **System Packages**: `ffmpeg`.

## Architectural Pillars

### 1. Robust Synchronous SRT Parser
- Synchronous pre-flight file validation inside the UI handler before dispatching background tasks.
- Multi-encoding reader (`utf-8-sig`, `utf-8`, `latin-1`, `cp1252`, `iso-8859-1`, `utf-16`).
- Throws immediate `gr.Warning` for missing, empty, or unparseable files.

### 2. High-Throughput Indic-F5 Batch Processing (T4 GPU)
- Batch size: 15–20 sentences grouped per inference pass to push VRAM usage to 6–8 GB.
- Dynamic OOM Fallback: catches CUDA OOM, clears GPU cache, halves the batch size, and immediately retries recursively.
- 0 KB crash check with individual retry per block.

### 3. FFmpeg Silent Canvas Alignment
- Generates a completely silent base canvas audio of exact `total_duration` seconds via `anullsrc`.
- Overlays each generated sentence chunk at its exact SRT `start_time` offset.
- Trims or pads boundaries to guarantee an exact match with the target video duration.
- Exports master `.wav` file into `storage/outputs/`.

### 4. Background Job Management & Persistence
- Detached daemon worker threads via `JobManager`.
- Cross-session persistence (`storage/job_state.json`) with auto-purge on startup to prevent zombie task freezes.
- One-click **"🧹 Reset Standby"** button to instantly clear any stale state.
- `demo.launch(share=True)` at entrypoint.
