# VibeVoice TTS API Dockerfile
#
# Build:
#   docker build -f docker/Dockerfile -t vibevoice-api .
#
# Run directly:
#   docker run --gpus all -p 5000:5000 \
#     -e MODEL_PATH=vibevoice/VibeVoice-1.5B \
#     -e HF_HOME=/app/hf-cache \
#     -v vibevoice-models:/app/hf-cache \
#     vibevoice-api

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install uv for fast Python package management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install PyTorch with CUDA support first (large download, cache layer)
RUN uv pip install --system torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy dependency specs first for better caching
COPY pyproject.toml .
COPY demo/app/requirements.txt demo/app/

# Install project dependencies
RUN uv pip install --system -r demo/app/requirements.txt \
    && uv pip install --system \
        accelerate==1.6.0 \
        transformers==4.51.3 \
        datasets==3.5.0 \
        peft \
        bitsandbytes \
        llvmlite \
        numba \
        diffusers \
        tqdm \
        numpy \
        scipy \
        librosa \
        ml-collections \
        absl-py \
        gradio==5.50.0 \
        av \
        aiortc \
        fastapi \
        uvicorn \
        python-multipart \
        python-dotenv \
        aiofiles

# Copy the rest of the project
COPY . .

# Install VibeVoice package in editable mode
RUN uv pip install --system -e .

# Create non-root user
RUN useradd -m -u 1000 vibevoice && chown -R vibevoice:vibevoice /app
USER vibevoice

# Directories for model cache and uploaded voices
RUN mkdir -p /app/hf-cache /app/uploaded-voices

ENV HF_HOME=/app/hf-cache \
    SPEAKER_SAMPLES_DIR=/app/uploaded-voices \
    API_HOST=0.0.0.0 \
    API_PORT=5000 \
    WORKER_NUM=1

EXPOSE 5000

COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
