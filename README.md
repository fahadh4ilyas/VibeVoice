> [!IMPORTANT]
> This is a community-maintained fork of VibeVoice. Following the removal of the official VibeVoice repository, this fork serves to preserve the codebase and maintain accessibility for the community while also introducing additional functionality (such as unofficial training/fine-tuning implementations)

## 🎙️ VibeVoice: A Frontier Long Conversational Text-to-Speech Model

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://microsoft.github.io/VibeVoice)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Models-orange?logo=huggingface)](https://huggingface.co/vibevoice)
[![Technical Report](https://img.shields.io/badge/Technical-Report-red)](https://arxiv.org/pdf/2508.19205)
[![Colab](https://img.shields.io/badge/Colab-Demo-orange?logo=googlecolab)](https://colab.research.google.com/github/vibevoice-community/VibeVoice/blob/main/demo/VibeVoice_colab.ipynb)

## Community

**Join the unofficial Discord community: https://discord.gg/ZDEYTTRxWG** - share samples, ask questions, discuss fine-tuning, etc.

## Overview

VibeVoice is a novel framework designed for generating **expressive**, **long-form**, **multi-speaker** conversational audio, such as podcasts, from text. It addresses significant challenges in traditional Text-to-Speech (TTS) systems, particularly in scalability, speaker consistency, and natural turn-taking.

A core innovation of VibeVoice is its use of continuous speech tokenizers (Acoustic and Semantic) operating at an ultra-low frame rate of 7.5 Hz. These tokenizers efficiently preserve audio fidelity while significantly boosting computational efficiency for processing long sequences. VibeVoice employs a [next-token diffusion](https://arxiv.org/abs/2412.08635) framework, leveraging a Large Language Model (LLM) to understand textual context and dialogue flow, and a diffusion head to generate high-fidelity acoustic details.

The model can synthesize speech up to **90 minutes** long with up to **4 distinct speakers**, surpassing the typical 1-2 speaker limits of many prior models.

Fine-tuning is now supported, which is incredibly powerful. You can adapt VibeVoice to a new language or a new voice - [try it out](https://github.com/vibevoice-community/VibeVoice/blob/main/FINETUNING.md)

## [Examples](./EXAMPLES.md)

## Evaluation

<p align="left">
  <img src="Figures/MOS-preference.png" alt="MOS Preference Results" height="260px">
  <img src="Figures/VibeVoice.jpg" alt="VibeVoice Overview" height="250px" style="margin-right: 10px;">
</p>


## Updates

- **[2026-06-12]** [Unofficial voice cloning](https://huggingface.co/mohammed-bahumaish/vibevoice-realtime-0.5b-with-encoder) for Streaming/Realtime model released by community member
- **[2025-12-04]** Added support for VibeVoice-Streaming-0.5B model for real-time TTS!
- **[2025-09-05]** Microsoft repo restored (without code) with statement about responsible AI use.
- **[2025-09-04]** Community backup created after Microsoft removed original repo and models.
- **[2025-08-26]** The [VibeVoice-7B](https://huggingface.co/vibevoice/VibeVoice-7B) model weights are open-sourced!
- **[2025-08-28]** [Colab Notebook](https://colab.research.google.com/github/microsoft-community/VibeVoice/blob/main/demo/VibeVoice_colab.ipynb) available. Only VibeVoice-1.5B is supported due to GPU memory limitations.

## Roadmap

- [x] Unofficial/community training code
- [ ] HF Transformers integration ([PR](https://github.com/huggingface/transformers/pull/40546))
- [ ] VibePod: End-to-end solution that creates podcasts from documents, webpages, or even a simple topic.

## Model Zoo

| Model | Context Length | Generation Length | Speakers | Weight |
|-------|----------------|-------------------|----------|--------|
| VibeVoice-Streaming-0.5B | 8K | Real-time | 1 | [HF link](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) |
| VibeVoice-1.5B | 64K | ~90 min | Up to 4 | [HF link](https://huggingface.co/vibevoice/VibeVoice-1.5B) |
| VibeVoice-Large (7B) | 32K | ~45 min | Up to 4 | [HF link](https://huggingface.co/vibevoice/VibeVoice-7B) |

### Model Comparison

- **VibeVoice-Streaming-0.5B**: Optimized for **real-time** low-latency TTS. Single speaker only. Uses pre-computed voice embeddings (.pt files) for fast inference. Best for live applications. [Unofficial voice cloning](https://huggingface.co/mohammed-bahumaish/vibevoice-realtime-0.5b-with-encoder)
- **VibeVoice-1.5B/7B**: Full-featured models for **long-form multi-speaker** content like podcasts. Support up to 4 speakers with voice cloning from audio samples.

## Installation

```bash
git clone https://github.com/vibevoice-community/VibeVoice.git
cd VibeVoice/

uv pip install -e .
```

## Docker Deployment

A Docker setup is provided for deploying the REST API server.

```bash
cd VibeVoice/

# Build and start (requires nvidia-container-toolkit)
MODEL_PATH=vibevoice/VibeVoice-1.5B docker compose -f docker/docker-compose.yaml up -d

# Or set MODEL_PATH in .env then:
# docker compose -f docker/docker-compose.yaml up -d
```

The server starts on `http://localhost:5000` with the following:

| Feature | Details |
|---------|---------|
| GPU | CUDA 12.4, NVIDIA container runtime |
| Port | 5000 (configurable via `API_PORT`) |
| Persistence | Model cache and uploaded voices survive restarts via Docker volumes |
| Health check | `GET /v1/audio/voices` |

See [`docker/docker-compose.yaml`](docker/docker-compose.yaml) for all configurable environment variables.

### Starting the API server directly (without Docker)

```bash
MODEL_PATH=vibevoice/VibeVoice-1.5B python demo/uvicorn.main.py
```

Set these environment variables (or create a `.env` file in the project root):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | *(required)* | HuggingFace model ID or local path |
| `API_HOST` | `127.0.0.1` | Bind address |
| `API_PORT` | `5000` | Server port |
| `WORKER_NUM` | `1` | Uvicorn workers |
| `CFG_SCALE` | `2.0` | Classifier-free guidance scale |
| `SEED` | `42` | Random seed |
| `MAX_BATCH_SIZE` | `4` | Max concurrent generation requests |
| `SPEAKER_SAMPLES_DIR` | `demo/uploaded-voices/` | Uploaded voice storage directory |
| `SPEAKER_MAX_UPLOADED` | `1000` | Max uploaded voices |

## Usage

### 🚨 Tips

We observed users may encounter occasional instability when synthesizing Chinese speech. We recommend:

- Using English punctuation even for Chinese text, preferably only commas and periods.
- Using the Large model variant, which is considerably more stable.
- If you found the generated voice speak too fast. Please try to chunk your text with multiple speaker turns with same speaker label.

We'd like to thank [PsiPi](https://huggingface.co/PsiPi) for sharing an interesting way for emotion control. Details can be found via [discussion #12](https://huggingface.co/microsoft/VibeVoice-1.5B/discussions/12).

**Option 1: Launch Gradio demo**

```bash
python demo/gradio_demo.py --model_path vibevoice/VibeVoice-1.5B --share
# or python demo/gradio_demo.py --model_path vibevoice/VibeVoice-7B --share
# optionally add --checkpoint_path path/to/checkpoint to load a fine-tuned adapter
# use the in-app "Disable voice cloning" setting (Advanced Settings) to skip speaker conditioning
```

**Option 2: Inference from files directly (Multi-Speaker 1.5B/7B)**

```bash
# We provide some LLM generated example scripts under demo/text_examples/ for demo
# 1 speaker
python demo/inference_from_file.py --model_path vibevoice/VibeVoice-7B --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice

# or more speakers
python demo/inference_from_file.py --model_path vibevoice/VibeVoice-7B --txt_path demo/text_examples/2p_music.txt --speaker_names Alice Frank

# load a fine-tuned LoRA checkpoint
python demo/inference_from_file.py --model_path vibevoice/VibeVoice-7B --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice --checkpoint_path path/to/checkpoint

# disable voice cloning (skip speech prefill)
python demo/inference_from_file.py --model_path vibevoice/VibeVoice-7B --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice --disable_prefill
```

**Option 3: Streaming Model (0.5B) - Real-time TTS**

The streaming model uses pre-computed voice embeddings for low-latency generation:

```bash
# Basic usage with streaming model
python demo/streaming_inference_from_file.py \
    --model_path microsoft/VibeVoice-Realtime-0.5B \
    --txt_path demo/text_examples/1p_vibevoice.txt \
    --speaker_name Emma

# Available voice presets: Carter, Davis, Emma, Frank, Grace, Mike (English), Samuel (Indian English)
# Adjust CFG scale and DDPM steps for quality/speed tradeoff
python demo/streaming_inference_from_file.py \
    --model_path microsoft/VibeVoice-Realtime-0.5B \
    --txt_path demo/text_examples/1p_vibevoice.txt \
    --speaker_name Mike \
    --cfg_scale 1.5 \
    --ddpm_steps 5
```

Voice presets are stored as `.pt` files in `demo/voices/streaming_model/`. These contain pre-computed KV cache embeddings for fast inference. Voice cloning is not supported for now.

NOTE: If you get the warning `Some weights of VibeVoiceStreamingForConditionalGenerationInference were not initialized from the model checkpoint` when loading, this is expected. This is because voice cloning capabilities have been removed from the model.

## REST API Server

VibeVoice includes an OpenAI-compatible REST API for text-to-speech generation. Start the server (see [Docker Deployment](#docker-deployment) or run directly) and use the following endpoints:

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/audio/speech` | Generate speech from text |
| `GET` | `/v1/audio/voices` | List available voices |
| `POST` | `/v1/audio/voices` | Upload a custom voice sample |
| `DELETE` | `/v1/audio/voices/{name}` | Delete an uploaded voice |
| `WS` | `/v1/audio/speech/stream` | WebSocket streaming TTS |

### Quick examples

**Generate speech (curl):**
```bash
curl -X POST http://localhost:5000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, how are you?", "voice": "alloy", "language": "en"}' \
    --output output.wav
```

**List available voices:**
```bash
curl http://localhost:5000/v1/audio/voices
```

**Upload a custom voice:**
```bash
curl -X POST http://localhost:5000/v1/audio/voices \
    -F "audio_sample=@/path/to/voice.wav" \
    -F "consent=my_consent_id" \
    -F "name=custom_speaker" \
    -F "ref_text=Optional transcript of the audio"
```

**Delete a voice:**
```bash
curl -X DELETE http://localhost:5000/v1/audio/voices/custom_speaker
```

**WebSocket streaming:**
```python
import asyncio
import websockets
import json

async def stream_tts():
    async with websockets.connect("ws://localhost:5000/v1/audio/speech/stream") as ws:
        await ws.send(json.dumps({"type": "session.config", "voice": "alloy", "language": "en"}))
        await ws.send(json.dumps({"type": "input.text", "text": "Hello world!"}))
        await ws.send(json.dumps({"type": "input.done"}))

        async for msg in ws:
            data = json.loads(msg)
            print(data["type"])  # audio.start, audio.done, session.done

asyncio.run(stream_tts())
```

The API spec is based on the [vLLM-Omni Speech API](https://docs.vllm.ai/projects/vllm-omni).

## [Finetuning](./FINETUNING.md)

NOTE: Finetuning is still **very experimental** and not well tested yet!

## FAQ

#### Q1: Is this a pretrained model?
**A:** Yes, it's a pretrained model without any post-training or benchmark-specific optimizations. In a way, this makes VibeVoice very versatile and fun to use.

#### Q2: Randomly trigger Sounds / Music / BGM.
**A:** As you can see from our demo page, the background music or sounds are spontaneous. This means we can't directly control whether they are generated or not. The model is content-aware, and these sounds are triggered based on the input text and the chosen voice prompt.

Here are a few things we've noticed:
*   If the voice prompt you use contains background music, the generated speech is more likely to have it as well. (The Large model is quite stable and effective at this—give it a try on the demo!)
*   If the voice prompt is clean (no BGM), but the input text includes introductory words or phrases like "Welcome to," "Hello," or "However," background music might still appear.
*   Speaker voice related, using "Alice" results in random BGM than others (fixed).
*   In other scenarios, the Large model is more stable and has a lower probability of generating unexpected background music.

In fact, we intentionally decided not to denoise our training data because we think it's an interesting feature for BGM to show up at just the right moment. You can think of it as a little easter egg we left for you.

#### Q3: Text normalization?
**A:** We don't perform any text normalization during training or inference. Our philosophy is that a large language model should be able to handle complex user inputs on its own. However, due to the nature of the training data, you might still run into some corner cases.

#### Q4: Singing Capability.
**A:** Our training data **doesn't contain any music data**. The ability to sing is an emergent capability of the model (which is why it might sound off-key, even on a famous song like 'See You Again'). (The Large model is more likely to exhibit this than the 1.5B).

#### Q5: Some Chinese pronunciation errors.
**A:** The volume of Chinese data in our training set is significantly smaller than the English data. Additionally, certain special characters (e.g., Chinese quotation marks) may occasionally cause pronunciation issues.

#### Q6: Instability of cross-lingual transfer.
**A:** The model does exhibit strong cross-lingual transfer capabilities, including the preservation of accents, but its performance can be unstable. This is an emergent ability of the model that we have not specifically optimized. It's possible that a satisfactory result can be achieved through repeated sampling.

## Credits

- Thanks to [Microsoft](https://github.com/microsoft/VibeVoice) for the original VibeVoice implementation.
- Huge shoutout to [Juan Pablo Gallego](https://github.com/jpgallegoar) from [VoicePowered AI](https://www.voicepowered.ai/) for the unofficial training/fine-tuning code.
- Thanks to [PsiPi](https://huggingface.co/PsiPi) for sharing an interesting way for emotion control. Details can be found via [discussion #12](https://huggingface.co/microsoft/VibeVoice-1.5B/discussions/12).

## License

The source code and models are licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

Note: Microsoft has removed the original repo and models. This fork is based off of the MIT-licensed code from Microsoft.
