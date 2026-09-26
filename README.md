---
title: Audiogenflow
emoji: 🎙️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
python_version: 3.11
app_file: app.py
pinned: false
---

# 🎙️ AudioGen Flow Studio — Pure SRT-Driven Local Audio Dubber

A high-efficiency, offline, sentence-level audio dubbing studio engineered to run smoothly on local environments and Hugging Face Spaces (CPU Tier).

## 🚀 Architecture Highlights

1. **Two Simple Inputs**:
   - **Original Audio File** (`.wav` or `.mp3` extracted from video)
   - **Translated Subtitle File** (`.srt`)
   - **Output**: Pristine sample-synchronized mixed master `.wav`.

2. **Zero Overhead**:
   - Zero Whisper ASR logic/dependencies.
   - Zero LLM / Groq / Gemini API calls or arbitrary chunking.
   - 100% offline, sentence-level dubbing driven strictly by SRT timestamps.

3. **Core SRT Parsing Engine**:
   - Extracts millisecond-accurate `start_time`, `end_time`, and clean dialogue text.
   - Robust fail-safe automatically filters out empty or whitespace-only subtitle blocks.

4. **Sentence-Level F5-TTS Generation**:
   - Hardcoded reference audio: `core_1_ours.wav` (default female voice).
   - Strict 0 KB crash check with automated single-retry before skipping any problematic block.

5. **Precise FFmpeg Audio Syncing & Mixing**:
   - Overlays each generated sentence audio clip at its exact SRT `start_time` offset.
   - Automated audio mixing: Lowers original audio track volume to **10%** (preserving background music & sound effects) and sets F5-TTS dubbed dialogue to **100%**.
   - Exports the combined master timeline as a single pristine `.wav` file.

