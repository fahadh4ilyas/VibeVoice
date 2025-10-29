# fastapi_stream_wav.py
import os
import io
import wave
import asyncio
import threading
import traceback
import timeit
from typing import Optional, Any, Literal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Body, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, ORJSONResponse, Response, RedirectResponse

import torch
import numpy as np
from contextlib import asynccontextmanager

from .config import config, LOGGER_ACCESS, LOGGER

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.lora_loading import load_lora_assets
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AsyncAudioStreamer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

executor = ThreadPoolExecutor(max_workers=4)

# --- model placeholders (load your model in startup) ---
processor: Optional[VibeVoiceProcessor] = None
model: Optional[VibeVoiceForConditionalGenerationInference] = None
lock = threading.Lock()
voice_list = {p.stem.split('_')[0]: p for p in Path(os.path.join(ROOT_DIR, 'sample-voices')).glob('*.wav')}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model

    processor = VibeVoiceProcessor.from_pretrained(config.model_path)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)
    LOGGER.info("Startup: model should be loaded here")
    yield
    LOGGER.info("Shutdown: clean up resources if needed")

app = FastAPI(title='Super LLM API',
    description='API for generating text using LLM with OpenAI Format',
    version='1.0.0',
    lifespan=lifespan)


@app.exception_handler(Exception)
async def value_error_handler(request: Request, exc: Exception):
    return ORJSONResponse({
        'error': str(exc),
        'traceback': "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        'status_code': 500
    }, status_code=500)


@app.middleware("http")
async def logging_request(request: Request, call_next):

    client_data = ''
    if request.client:
        client_data = f'{request.client.host}:{request.client.port}'
    LOGGER_ACCESS.info(f'{client_data} - "{request.method.upper()} {request.url.path} {request.url.scheme.upper()}/1.1" START')
    params = str(request.query_params)
    body = await request.body()
    if params:
        LOGGER_ACCESS.info(f'{client_data} - "{request.method.upper()} {request.url.path} {request.url.scheme.upper()}/1.1" PARAMS: {params}')
    if body:
        LOGGER_ACCESS.info(f'{client_data} - "{request.method.upper()} {request.url.path} {request.url.scheme.upper()}/1.1" BODY: {await request.body()}')

    start = timeit.default_timer()
    request.state.is_disconnected = request.is_disconnected
    response: Response = await call_next(request)
    response.headers["X-Process-Time"] = f'{timeit.default_timer() - start:.6f}'

    return response


def make_wav_header(
    sample_rate: int,
    num_channels: int,
) -> bytes:
    """
    Build a simple WAV (RIFF) header for PCM (little-endian).
    Note: many clients will accept a header with 0 data_size for chunked streaming.
    """

    num_channels = 1
    sample_width = 2
    frame_rate = sample_rate

    wav_header = io.BytesIO()
    with wave.open(wav_header, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)

    wav_header.seek(0)
    wave_header_bytes = wav_header.read()
    wav_header.close()

    # Create a new BytesIO with the correct MIME type for Firefox
    final_wave_header = io.BytesIO()
    final_wave_header.write(wave_header_bytes)
    final_wave_header.seek(0)

    return final_wave_header.getvalue()



# converter: tensor/array -> PCM16 bytes (mono/stereo)
def chunk_to_pcm16_bytes(chunk: Any, num_channels: int) -> bytes:
    """
    Accepts:
        - torch.Tensor (1D or 2D)
        - numpy array
        - bytes (raw)
    Returns little-endian PCM16 bytes.
    """
    if isinstance(chunk, torch.Tensor):
        arr = chunk.detach().cpu().float().numpy()
    elif isinstance(chunk, np.ndarray):
        arr = chunk
    elif isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk)
    else:
        # fallback: string repr
        return str(chunk).encode("utf-8")

    # If chunk is multi-dimensional (e.g., [frames, channels]), handle channels
    if arr.ndim == 1:
        # mono
        samples = arr
    elif arr.ndim == 2:
        # shape (frames, channels) or (channels, frames) — try to detect common case
        if arr.shape[1] == num_channels:
            # e.g., (frames, channels)
            samples = arr
        elif arr.shape[0] == num_channels:
            # e.g., (channels, frames) -> transpose
            samples = arr.T
        else:
            # unknown layout -> flatten
            samples = arr.flatten()
    else:
        samples = arr.flatten()

    # Normalize & convert floats to int16
    if np.issubdtype(samples.dtype, np.floating):
        # clamp to -1..1 then scale
        clipped = np.clip(samples, -1.0, 1.0)
        int16 = (clipped * 32767).astype(np.int16)
    else:
        # integer types: convert/rescale if needed — here we cast to int16 directly
        int16 = samples.astype(np.int16)

    return int16.tobytes()


@app.get("/tts")
async def gen_wav(
    request: Request,
    text: str = Query(...),
    speaker: Literal["alloy", "ash", "echo", "nova"] = Query("alloy"),
    lang: Literal["en", "id"] = Query('id'),
    timeout: float = Query(-1)
):
    """
    Streams a TTS response produced by the model.
    - only one caller at a time (asyncio.Lock).
    - cooperative stop via audio_streamer.end().
    - streaming binary WAV via chunked transfer.
    """

    return await tts_streamer(request, text, speaker, lang, timeout)

@app.post("/tts")
async def generate_wav(
    request: Request,
    text: str = Body(...),
    speaker: Literal["alloy", "ash", "echo", "nova"] = Body("alloy"),
    lang: Literal["en", "id"] = Body('id'),
    timeout: float = Body(-1)
):
    """
    Streams a TTS response produced by the model.
    - only one caller at a time (asyncio.Lock).
    - cooperative stop via audio_streamer.end().
    - streaming binary WAV via chunked transfer.
    """

    return await tts_streamer(request, text, speaker, lang, timeout)

async def tts_streamer(
    request: Request,
    text: str,
    speaker: Literal["alloy", "ash", "echo", "nova"],
    lang: Literal["en", "id"],
    timeout: float
):
    global model, processor, executor, lock

    # Acquire single-call lock so other requests wait
    if not lock.acquire(timeout=timeout if timeout >= 0 else -1):
        return ORJSONResponse(
            {"status": "error", "error": "Another request is in progress. Please try again later."},
            status_code=429
        )
    batch_size = 1  # streaming a single sample per request; adapt for batching
    sample_rate = 24000
    num_channels = 1

    # instantiate streamer (adjust signature if needed)
    audio_streamer = AsyncAudioStreamer(batch_size=batch_size, timeout=1.0)  # type: ignore


    # blocking wrapper to call the synchronous model.generate(...) that writes into audio_streamer
    def blocking_generate(prompt_text: str):
        try:
            full_script = "Speaker 1: " + prompt_text
            voice_sample = voice_list[f"{lang}-{speaker}"].as_posix()
            inputs = processor(
                text=[full_script], # Wrap in list for batch processing
                voice_samples=[voice_sample], # Wrap in list for batch processing
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            # for k, v in inputs.items():
            #     if torch.is_tensor(v):
            #         inputs[k] = v.to(model.device)
            model.generate(
                **inputs,
                audio_streamer=audio_streamer,
                tokenizer=processor.tokenizer,
                max_new_tokens=None,
                cfg_scale=config.cfg_scale,
                is_prefill=True,
                verbose=True,
                seed=config.seed,
            )
            # clear torch cache after generation
            torch.cuda.empty_cache()
            return {"status": "ok", "info": "generate finished normally"}
        except Exception as exc:
            LOGGER.error(traceback.format_exc())
            return {"status": "error", "error": str(exc)}
        finally:
            audio_streamer.end()
            lock.release()

    loop = asyncio.get_running_loop()
    gen_future = loop.run_in_executor(executor, blocking_generate, text)


    async def disconnect_watcher():
        try:
            is_disc = await request.state.is_disconnected()
        except Exception:
            # treat errors as disconnected
            is_disc = True
        if is_disc:
            audio_streamer.end()

    disconnect_task = asyncio.create_task(disconnect_watcher())

    async def stream_generator():
        """
        Yields:
            - initial WAV header bytes (with placeholder sizes)
            - then PCM16 chunk bytes as received from audio_streamer
            - finally the generate() result metadata as a small JSON/text chunk (optional)
        """
        try:
            # send WAV header first
            header = make_wav_header(sample_rate, num_channels)
            yield header

            # iterate over audio_streamer async iterator (your AsyncAudioStreamer.__aiter__)
            async for batch_chunks in audio_streamer:
                # batch_chunks: dict mapping sample_idx -> chunk
                # We assume batch_size==1 for simplicity; stream only index 0 if present
                # But handle multiple channels if your chunk data contains them.
                if 0 in batch_chunks:
                    chunk = batch_chunks[0]
                    pcm_bytes = chunk_to_pcm16_bytes(chunk, num_channels)
                    if pcm_bytes:
                        yield pcm_bytes
                else:
                    # if other indices present, process in increasing order and interleave their PCM
                    for idx in sorted(batch_chunks.keys()):
                        chunk = batch_chunks[idx]
                        pcm_bytes = chunk_to_pcm16_bytes(chunk, num_channels)
                        if pcm_bytes:
                            yield pcm_bytes

                # let event loop schedule (helps responsiveness)
                # await asyncio.sleep(0)

            # generation finished normally; await final result from blocking generate
            gen_result = await gen_future
            # optionally yield small trailing info as text chunk (not part of wav)
            # Many clients will ignore data after WAV; if you want strictly valid WAV only, omit this.
            # We omit extra trailing bytes to keep stream pure WAV.
            return  # end generator; connection closes naturally
        except asyncio.CancelledError:
            # generator cancelled (client disconnected); ensure stop flag
            audio_streamer.end()
            raise
        except Exception as exc:
            # On error, set stop flag and re-raise so client sees connection drop
            audio_streamer.end()
            raise
        finally:
            audio_streamer.end()

    # Build binary streaming response with WAV mime type
    return StreamingResponse(stream_generator(), media_type="audio/wav")

@app.get('/', include_in_schema=False)
async def redirect():

    # return ORJSONResponse({'title': app.title, 'description': app.description, 'version': app.version})
    return RedirectResponse(app.root_path+'/docs')
