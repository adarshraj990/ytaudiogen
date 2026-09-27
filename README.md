# 🎙️ AudioGen Flow Studio — Neural Voice SRT Dubber (Google Colab Edition)

A high-performance, lightweight, sentence-level audio dubbing studio powered by ultra-fast neural speech synthesis and FFmpeg silent canvas timeline alignment.

⚡ **100% Free of Hugging Face Dependencies — Zero Model Downloads, Zero VRAM Crashes, Zero Tokenizer/Pip Conflicts.**

---

## 🚀 Key Features

1. **Zero Hugging Face Footprint**:
   - No `transformers`, `tokenizers`, `huggingface_hub`, or multi-gigabyte PyTorch weights.
   - Installs in seconds in Google Colab without dependency resolver conflicts.
   - Never runs out of GPU memory (VRAM).

2. **Ultra-Fast Neural Voice Synthesis**:
   - Human-sounding neural voices for Hindi, Indian English, US English, and regional Indian languages (Bengali, Marathi, Tamil, Telugu, Urdu).
   - Generates dialogue sentences concurrently in milliseconds.

3. **FFmpeg Silent Canvas Alignment**:
   - Generates a silent base audio track of exact `total_duration` seconds.
   - Overlays each dialogue chunk at its exact SRT `start_time` timestamp.
   - Automatically pads or trims the timeline to guarantee an exact duration match with the original video.

4. **Robust Set-and-Forget Job Manager**:
   - Non-blocking daemon background processing.
   - Live dashboard metrics, real-time activity log console, and instant progress tracking.
   - Auto-purges stale/zombie states across Colab restarts.
   - Includes a one-click **"🧹 Reset Standby"** button.

---

## 💻 Google Colab Quickstart

Run these commands in your Google Colab cell:

```bash
# 1. Clone repository
!git clone https://github.com/adarshraj990/ytaudiogen.git
%cd ytaudiogen

# 2. Install lightweight dependencies (Takes <10 seconds)
!pip install -r requirements.txt

# 3. Launch with public Gradio link
!python app.py
```

---

## 🛠️ Supported Voices

- **Hindi**:
  - `Swara (Female, Natural & Expressive)` — `hi-IN-SwaraNeural`
  - `Madhur (Male, Deep & Storyteller)` — `hi-IN-MadhurNeural`
- **English**:
  - `Neerja (Female, Indian Accent)` — `en-IN-NeerjaNeural`
  - `Prabhat (Male, Indian Accent)` — `en-IN-PrabhatNeural`
  - `Christopher (Male, Dynamic Narrator)` — `en-US-ChristopherNeural`
  - `Jenny (Female, Conversational)` — `en-US-JennyNeural`
- **Regional Languages**:
  - `Bengali (Tanishaa)` — `bn-IN-TanishaaNeural`
  - `Marathi (Aarohi)` — `mr-IN-AarohiNeural`
  - `Tamil (Pallavi)` — `ta-IN-PallaviNeural`
  - `Telugu (Shruti)` — `te-IN-ShrutiNeural`
  - `Urdu (Uzma)` — `ur-PK-UzmaNeural`
