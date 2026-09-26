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

# 🎙️ AudioGen Flow Studio — Indic-F5 (0.3B) SRT-Driven Audio Dubber

A high-efficiency, offline, sentence-level audio dubbing studio powered by the **Indic-F5 (0.3B parameters)** Hindi-English code-switched model and FFmpeg silent canvas alignment.

## 🚀 Architecture Highlights

1. **Streamlined UI Inputs**:
   - **Total Video Duration**: Exact duration (in seconds, numeric input with default 120s).
   - **Translated Subtitle File**: Standard subtitle file (`.srt`).
   - **Output**: Master dubbed audio file (`.wav`) matching the exact video timeline.

2. **Indic-F5 (0.3B) Code-Switched Engine**:
   - Model: [`Tharshan/indicf5_hindi-english_code_switch`](https://huggingface.co/Tharshan/indicf5_hindi-english_code_switch) (0.3B parameters).
   - Specialized for Hindi-English code-switched (Hinglish) dialogue with natural Indian voice prosody.
   - Hardcoded reference voice: `core_1_ours.wav`.
   - Strict 0 KB crash check with automated single-retry before skipping any problematic block.

3. **FFmpeg Silent Canvas Alignment**:
   - Generates a completely silent base canvas audio of exact `total_duration` seconds via `anullsrc`.
   - Overlays each generated Indic-F5 sentence chunk at its exact SRT `start_time` offset.
   - Trims and pads timeline boundaries to guarantee an exact match with the target video duration.
   - Exports the combined master timeline as a single pristine `.wav` file.

