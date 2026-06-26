# Default:  docker build -f docker/Dockerfile -t vibevoice-api .
# Flash-Attn: docker build -f docker/Dockerfile -t vibevoice-api --build-arg ATTN_IMPLEMENTATION=flash_attention_2 .

ARG ATTN_IMPLEMENTATION=sdpa

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
ARG ATTN_IMPLEMENTATION

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv ffmpeg libsndfile1 git curl \
    && if [ "$ATTN_IMPLEMENTATION" = "flash_attention_2" ]; then \
         apt-get install -y --no-install-recommends g++ ninja-build; \
       fi \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

RUN pip install --no-cache-dir uv

WORKDIR /app

# PyTorch 2.5.1 — 2.6.x breaks flash-attn ABI
RUN uv pip install --system torch==2.5.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

ARG ATTN_IMPLEMENTATION
RUN if [ "$ATTN_IMPLEMENTATION" = "flash_attention_2" ]; then \
        uv pip install --system packaging \
        && MAX_JOBS=2 uv pip install --system flash-attn --no-build-isolation; \
    else \
        pip uninstall -y flash-attn 2>/dev/null || true; \
    fi

RUN apt-get remove -y g++ ninja-build 2>/dev/null; apt-get autoremove -y; \
    rm -rf /usr/local/cuda/bin/nvcc /usr/local/cuda/nvvm \
           /usr/local/cuda/include /usr/local/cuda/doc /usr/local/cuda/extras \
           /usr/local/cuda/samples /usr/local/cuda/libnvvp /usr/local/cuda/libnsight \
           /usr/local/cuda/compute-sanitizer /usr/local/cuda/tools \
           /usr/local/cuda/targets/*/lib/*.a; \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml demo/app/requirements.txt ./

RUN uv pip install --system -r requirements.txt \
    && uv pip install --system \
        accelerate==1.6.0 transformers==4.51.3 datasets==3.5.0 \
        peft bitsandbytes llvmlite numba diffusers tqdm numpy scipy librosa \
        ml-collections absl-py gradio==5.50.0 av aiortc \
        fastapi uvicorn python-multipart python-dotenv aiofiles

COPY . .
RUN uv pip install --system -e .

RUN useradd -m -u 1000 vibevoice && chown -R vibevoice:vibevoice /app
USER vibevoice
RUN mkdir -p /app/hf-cache /app/uploaded-voices

ENV HF_HOME=/app/hf-cache \
    SPEAKER_SAMPLES_DIR=/app/uploaded-voices \
    ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION} \
    API_HOST=0.0.0.0 API_PORT=5000 WORKER_NUM=1

EXPOSE 5000

COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
