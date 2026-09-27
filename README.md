# 🎙️ Indic-F5 Audio Studio — Google Colab Edition (T4 GPU)

A high-performance, batch-optimized, sentence-level audio dubbing studio powered by the **Indic-F5 (0.3B)** Hindi-English code-switched model and **FFmpeg silent canvas timeline alignment**.

---

## ⚡ Core Architecture

1. **Robust Synchronous SRT Parser**:
   - Multi-encoding safe reader (`utf-8-sig`, `utf-8`, `latin-1`, `cp1252`, `iso-8859-1`, `utf-16`).
   - Synchronous pre-flight validation catches empty, missing, or malformed subtitle files immediately with visible `gr.Warning` notifications.
   - Eliminates silent background thread freezes.

2. **Optimized Batch Inference (T4 GPU)**:
   - Groups dialogue sentences into batches of **15–20** to maximize Google Colab T4 GPU VRAM utilization (**6–8 GB**).
   - **Active OOM Fallback**: Catches CUDA out-of-memory errors, clears cache (`torch.cuda.empty_cache()`), halves the batch size, and immediately retries without crashing or skipping dialogue.

3. **FFmpeg Silent Canvas Alignment**:
   - Creates a sample-accurate silent audio track matching the exact total video duration via `anullsrc`.
   - Overlays each batched audio segment at its exact SRT `start_time` offset.
   - Trims and pads timeline boundaries to guarantee a 1:1 match with original video length.

4. **Clean & Minimal UI**:
   - Minimal Gradio Blocks interface with live progress monitoring and activity stream.
   - Always launches with public share link enabled: `demo.launch(share=True)`.

---

## 🚀 Google Colab Quickstart

Copy and paste this snippet into your Google Colab cell (with T4 GPU runtime enabled):

```bash
# 1. Clone repository
!git clone https://github.com/adarshraj990/ytaudiogen.git
%cd ytaudiogen

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Launch application
!python app.py
```
