# fastapi_stream_wav.py
import os
import io
import time
import wave
import asyncio
import threading
import traceback
import timeit
import base64
from typing import Optional, Any, Literal, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Body, Query, Form, File, UploadFile
from fastapi.responses import StreamingResponse, ORJSONResponse, Response, RedirectResponse

import torch
import numpy as np
from contextlib import asynccontextmanager

from .config import config, LOGGER_ACCESS, LOGGER
from .tools import base64_audio_to_numpy

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference, VibeVoiceStepOutput
from vibevoice.modular.lora_loading import load_lora_assets
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AsyncAudioStreamer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

class DataQueue:

    def __init__(self, max_batch_size: int, model: VibeVoiceForConditionalGenerationInference, processor: VibeVoiceProcessor):
        self.active_queue: List[VibeVoiceStepOutput] = []
        self.queue: List[VibeVoiceStepOutput] = []
        self.max_batch_size = max_batch_size
        self.model = model
        self.processor = processor
    
    def put(
        self,
        text: str,
        audio_base64: Optional[str],
        lang: str,
        speaker: str,
        do_sample: bool,
        temperature: float,
        top_p: float,
        audio_streamer: AsyncAudioStreamer):
        full_script = "Speaker 1: " + text
        if audio_base64 is not None:
            voice_sample = base64_audio_to_numpy(
                audio_base64,
                mono=True,
                target_sr=24000,
                dtype="float32",
            )[0]
        else:
            voice_sample = voice_list[f"{lang}-{speaker}"].as_posix()
        inputs = self.processor(
            text=[full_script], # Wrap in list for batch processing
            voice_samples=[voice_sample], # Wrap in list for batch processing
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        item = self.model._init_step_input(
            **inputs,
            audio_streamer=audio_streamer,
            tokenizer=self.processor.tokenizer,
            max_new_tokens=None,
            cfg_scale=config.cfg_scale,
            is_prefill=True,
            verbose=True,
            seed=config.seed,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        if len(self.active_queue) < self.max_batch_size:
            self.active_queue.append(item)
        else:
            self.queue.append(item)
    
    def check_queue(self) -> bool:
        "remove finished items from active queue, fill from waiting queue, but remove cancelled items first from queue"
        removed_count = 0
        for i in range(len(self.active_queue)-1, -1, -1):
            if self.active_queue[i].finished:
                self.active_queue.pop(i)
                removed_count += 1
        for i in range(len(self.queue)-1, -1, -1):
            if self.queue[i].finished:
                self.queue.pop(i)
        for _ in range(removed_count):
            if self.queue:
                item = self.queue.pop(0)
                self.active_queue.append(item)
        return len(self.active_queue) > 0
    
    @property
    def is_empty(self):
        return len(self.active_queue) == 0
    
    def log_status(self):
        LOGGER.info(f"DataQueue status: active={len(self.active_queue)}, waiting={len(self.queue)}")

    def replace_item(self, index: int, item: VibeVoiceStepOutput):
        if 0 <= index < len(self.active_queue):
            self.active_queue[index] = item
    
    def infinite_loop_step(self):
        "infinite loop in another thread and the exit if interrupted or get killed"

        log_counter = 0
        try:
            while True:
                now = time.time()
                if log_counter >= 15.0:
                    self.log_status()
                    log_counter = 0
                if self.is_empty:
                    time.sleep(0.1)
                    log_counter += (time.time() - now)
                    continue
                self.check_queue()
                for i in range(len(self.active_queue)):
                    item = self.active_queue[i]
                    next_inputs = item.next_inputs
                    step_output = self.model._single_step_generate(
                        input_ids=next_inputs.input_ids,
                        inputs_embeds=next_inputs.inputs_embeds,
                        model_kwargs=next_inputs.model_kwargs,
                        negative_input_ids=next_inputs.negative_input_ids,
                        negative_model_kwargs=next_inputs.negative_model_kwargs,
                        stop_check_fn=next_inputs.stop_check_fn,
                        verbose=next_inputs.verbose,
                        step=next_inputs.step,
                        max_steps=next_inputs.max_steps,
                        audio_streamer=next_inputs.audio_streamer,
                        finished_tags=next_inputs.finished_tags,
                        progress_bar=next_inputs.progress_bar,
                        generation_config=next_inputs.generation_config,
                        batch_size=next_inputs.batch_size,
                        device=next_inputs.device,
                        is_prefill=next_inputs.is_prefill,
                        reach_max_step_sample=next_inputs.reach_max_step_sample,
                        speech_tensors=next_inputs.speech_tensors,
                        speech_masks=next_inputs.speech_masks,
                        speech_input_mask=next_inputs.speech_input_mask,
                        logits_processor=next_inputs.logits_processor,
                        max_step_per_sample=next_inputs.max_step_per_sample,
                        acoustic_cache=next_inputs.acoustic_cache,
                        semantic_cache=next_inputs.semantic_cache,
                        correct_cnt=next_inputs.correct_cnt,
                        cfg_scale=next_inputs.cfg_scale,
                        audio_chunks=next_inputs.audio_chunks,
                        refresh_negative=next_inputs.refresh_negative,
                    )
                    self.replace_item(i, step_output)
                torch.cuda.empty_cache()
                log_counter += (time.time() - now)
        except KeyboardInterrupt:
            LOGGER.info("infinite_loop_step interrupted, exiting")
        except Exception as exc:
            LOGGER.error(f"infinite_loop_step exception: {traceback.format_exc()}")

executor = ThreadPoolExecutor(max_workers=4)

# --- model placeholders (load your model in startup) ---
processor: Optional[VibeVoiceProcessor] = None
model: Optional[VibeVoiceForConditionalGenerationInference] = None
data_queue: Optional[DataQueue] = None
lock = threading.Lock()
voice_list = {p.stem.split('_')[0]: p for p in Path(os.path.join(ROOT_DIR, 'sample-voices')).glob('*.wav')}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, data_queue, executor

    processor = VibeVoiceProcessor.from_pretrained(config.model_path)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)
    data_queue = DataQueue(max_batch_size=config.max_batch_size, model=model, processor=processor)
    loop = asyncio.get_event_loop()
    infinite_thread = loop.run_in_executor(executor, data_queue.infinite_loop_step)
    LOGGER.info("Startup: model should be loaded here")
    yield
    infinite_thread.cancel()
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
        LOGGER_ACCESS.info(f'{client_data} - "{request.method.upper()} {request.url.path} {request.url.scheme.upper()}/1.1" BODY: {(await request.body())[:256]}')

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
    do_sample: bool = Query(False),
    temperature: float = Query(1.0, ge=0.0, le=2.0),
    top_p: float = Query(1.0, le=1.0, ge=0.0)
):
    """
    Streams a TTS response produced by the model.
    - only one caller at a time (asyncio.Lock).
    - cooperative stop via audio_streamer.end().
    - streaming binary WAV via chunked transfer.
    """

    return await tts_streamer(request, text, None, speaker, lang, do_sample, temperature, top_p)

@app.post("/tts")
async def generate_wav(
    request: Request,
    text: str = Body(...),
    speaker: Literal["alloy", "ash", "echo", "nova"] = Body("alloy"),
    lang: Literal["en", "id"] = Body('id'),
    do_sample: bool = Body(False),
    temperature: float = Body(1.0, ge=0.0, le=2.0),
    top_p: float = Body(1.0, le=1.0, ge=0.0)
):
    """
    Streams a TTS response produced by the model.
    - only one caller at a time (asyncio.Lock).
    - cooperative stop via audio_streamer.end().
    - streaming binary WAV via chunked transfer.
    """

    return await tts_streamer(request, text, None, speaker, lang, do_sample, temperature, top_p)

@app.post("/voice_clone")
async def voice_clone_form(
    request: Request,
    text: str = Form(...),
    audio_file: UploadFile = File(...),
    do_sample: bool = Form(False),
    temperature: float = Form(1.0),
    top_p: float = Form(1.0)
):
    """
    Clones a voice from the provided audio file in form upload.
    """
    audio_bytes = await audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    return await tts_streamer(request, text, audio_base64, "alloy", "id", do_sample, temperature, top_p)

async def tts_streamer(
    request: Request,
    text: str,
    audio_base64: Optional[str],
    speaker: Literal["alloy", "ash", "echo", "nova"],
    lang: Literal["en", "id"],
    do_sample: bool,
    temperature: float,
    top_p: float
):
    global data_queue

    batch_size = 1  # streaming a single sample per request; adapt for batching
    sample_rate = 24000
    num_channels = 1

    # instantiate streamer (adjust signature if needed)
    audio_streamer = AsyncAudioStreamer(batch_size=batch_size, timeout=1.0)  # type: ignore

    data_queue.put(text, audio_base64, lang, speaker, do_sample, temperature, top_p, audio_streamer)

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
