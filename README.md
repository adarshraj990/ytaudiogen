---
title: Ytaudiogen
emoji: 🎙️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
python_version: 3.11
app_file: app.py
pinned: false
---

# 🎙️ YouTube Long-Form Auto Dubber (Anime Theory)

A production-grade, background video dubbing application engineered to run smoothly on Hugging Face Spaces (Free CPU Tier: 2 vCPU, 16GB RAM). Designed to process 2-3 hour long YouTube videos (such as anime theory breakdowns) and dub them into 4 separate languages: **Hindi, Spanish, French, and Portuguese**.

## 🚀 Key Features

1. **⚡ Progressive Yield Live Downloads**:
   - The UI generator yields finished master audio tracks (`Hindi_Full.mp3`, `Spanish_Full.mp3`, etc.) **immediately as each language completes**, without waiting for all 4 languages to finish.

2. **🗣️ Kokoro-ONNX CPU Synthesis**:
   - Native ONNX TTS engine optimized for 2 vCPU execution.
   - Distinct canonical voices for Hindi (`hm_omega`), Spanish (`em_alex`), French (`ff_siwis`), and Portuguese (`pf_dora`).
   - Sentence sub-chunking (<220 chars) to prevent phoneme truncation.

3. **🎵 Pydub Master Audio Concatenation**:
   - Chunks are assembled sequentially using `pydub` with memory-safe batching and garbage collection.

4. **🧹 Automatic Server Storage Cleanup**:
   - Once a language's full MP3 is created and verified, intermediate 1-2 minute chunk files are purged immediately to preserve server disk space.

5. **⚡ 4-Layer Zero-Sleep API Rotation**:
   - Two-Key Round-Robin load balancing across sequential chunks.
   - Primary `gemini-3.8-flash` with instant `try-except` fallback to `gemini-3.5-flash` on the same key without `time.sleep()`.

6. **🍥 Naruto Lore & Anime Terminology Preservation**:
   - Specialized prompt preserving terms like *Sharingan, Hokage, Jutsu, Chakra, Uchiha, Rasengan, Akatsuki, Konoha*.
   - Phonetic Devanagari transliteration for Hindi without literal translation.

7. **🛡️ OOM-Safe 90-Second Chunking & Background Worker**:
   - Out-of-process FFmpeg stream chunking prevents OOM crashes on 16GB RAM.
   - Decoupled daemon worker allows users to close the browser tab at any time.
