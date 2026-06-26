#!/usr/bin/env bash
set -euo pipefail

echo "=== VibeVoice TTS API ==="
echo "MODEL_PATH: ${MODEL_PATH:?MODEL_PATH environment variable is required}"
echo "API_HOST:   ${API_HOST:-0.0.0.0}"
echo "API_PORT:   ${API_PORT:-5000}"
echo "WORKERS:    ${WORKER_NUM:-1}"
echo "HF_HOME:    ${HF_HOME:-/app/hf-cache}"
echo "========================="

# Pre-download model from HuggingFace so first request is fast.
# The HF hub cache is on a persistent volume, so this is a no-op on restarts
# once the model is cached.
echo "Pre-caching model: ${MODEL_PATH} ..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_PATH}', resume_download=True)
print('Model cached successfully.')
" || echo "WARNING: Model pre-cache failed — will download on first request"

# Create voice directories
mkdir -p /app/uploaded-voices

# Start the server
echo "Starting uvicorn on ${API_HOST:-0.0.0.0}:${API_PORT:-5000} ..."
exec python3 /app/demo/uvicorn.main.py
