---
title: Audiogenflow
emoji: 🎙️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# 🎙️ YouTube Auto Audio Dubber (English ➔ Hindi)

An AI-powered automated video audio dubbing pipeline optimized for Hugging Face Spaces (Free CPU tier).

## 🚀 Architecture
1. **Downloader**: Audio extraction with `yt-dlp` (16kHz mono WAV).
2. **ASR Transcriber**: Timestamped English transcription with `faster-whisper` (CPU int8).
3. **Director**: Translation and timing coordination with Gemini Multi-Model Cascade (`gemini-1.5-flash` ➔ `gemini-2.0-flash` ➔ `gemini-1.5-pro`).
4. **TTS Engine**: Local Hindi synthesis with `kokoro-onnx` (Thread-safe Singleton, `hm_omega` / `hf_alpha`).
5. **Audio Sync**: Time-stretching, padding, and master timeline assembly via `pydub`.
6. **Interface**: Modern Gradio Blocks UI with real-time `gr.Progress()`.

## 🔑 Environment Variables / Secrets
Add under **Space Settings ➔ Repository Secrets**:
- `GEMINI_API_KEY`: Your Google Gemini API key.
