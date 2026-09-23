FROM python:3.11-slim

# Install system dependencies
# espeak-ng: required by Kokoro for G2P text phonemization
# ffmpeg: required by pydub and yt-dlp for audio transcoding
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Pre-create output and model cache directories
RUN mkdir -p /app/output /app/model_cache

# Copy requirements
COPY requirement.txt .

# Install Python dependencies — force pre-compiled binary wheels
RUN pip install --no-cache-dir --prefer-binary -r requirement.txt

# Copy application files
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run application
CMD ["python", "app.py"]
