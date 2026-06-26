# fastapi_stream_wav.py
import os
import io
import re
import time
import wave
import json
import shutil
import asyncio
import aiofiles
import threading
import traceback
import timeit
import base64
from typing import Optional, Any, Literal, List, Dict, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Body, Query, Form, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, ORJSONResponse, Response, RedirectResponse, FileResponse

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

# --- Voice management system ---
# Each uploaded voice is stored as two files in VOICES_DIR:
#   {name}.wav  — the audio sample
#   {name}.json — per-voice metadata (avoids multi-worker contention on a single file)
# Built-in voices (voices/, sample-voices/) are read-only and have no sidecar JSON.
VOICES_DIR = Path(os.environ.get("SPEAKER_SAMPLES_DIR", os.path.join(ROOT_DIR, "uploaded-voices")))
MAX_UPLOADED_VOICES = int(os.environ.get("SPEAKER_MAX_UPLOADED", "1000"))


def _voice_metadata_path(name: str) -> Path:
    return VOICES_DIR / f"{name}.json"


def _voice_wav_path(name: str) -> Path:
    return VOICES_DIR / f"{name}.wav"


def _read_voice_metadata(name: str) -> Optional[dict]:
    """Read a single voice's metadata JSON. Returns None if missing or corrupt."""
    path = _voice_metadata_path(name)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        LOGGER.warning(f"Corrupt metadata for voice '{name}', ignoring")
        return None


def _write_voice_metadata(name: str, info: dict) -> None:
    """Atomically write per-voice metadata JSON (write-tmp + rename)."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    path = _voice_metadata_path(name)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(info, f, indent=2)
    os.replace(tmp, path)  # atomic on POSIX


def _load_uploaded_voices_metadata() -> Dict[str, dict]:
    """Scan VOICES_DIR for *.json sidecars and return {name: metadata}."""
    if not VOICES_DIR.is_dir():
        return {}
    result: Dict[str, dict] = {}
    for json_file in sorted(VOICES_DIR.glob("*.json")):
        name = json_file.stem
        info = _read_voice_metadata(name)
        if info is not None:
            # Only include if the WAV file also exists
            if _voice_wav_path(name).exists():
                result[name] = info
            else:
                LOGGER.warning(f"Uploaded voice '{name}': metadata exists but WAV file missing, skipping")
    return result


def _get_builtin_voices() -> Dict[str, Path]:
    """Scan built-in voice directories and return {name: path} mapping."""
    builtin: Dict[str, Path] = {}
    for scan_dir in [
        Path(os.path.join(ROOT_DIR, "voices")),
        Path(os.path.join(ROOT_DIR, "sample-voices")),
    ]:
        if scan_dir.is_dir():
            for wav_file in scan_dir.glob("*.wav"):
                name = wav_file.stem  # e.g. "en-Alice_woman"
                builtin[name] = wav_file
    return builtin


def _get_uploaded_voices() -> Dict[str, Path]:
    """Return currently uploaded voices with their file paths from disk."""
    uploaded: Dict[str, Path] = {}
    for name, info in _load_uploaded_voices_metadata().items():
        wav_path = _voice_wav_path(name)
        if wav_path.exists():
            uploaded[name] = wav_path
    return uploaded


def _get_all_available_voices() -> Dict[str, Path]:
    """Combine built-in and uploaded voices. Uploaded voices take precedence over built-in."""
    voices = _get_builtin_voices()
    voices.update(_get_uploaded_voices())
    return voices


def _count_uploaded_voices() -> int:
    """Count current uploaded voices (by scanning *.json sidecars)."""
    if not VOICES_DIR.is_dir():
        return 0
    return len(list(VOICES_DIR.glob("*.json")))


def _upload_voice(name: str, audio_bytes: bytes, consent: str,
                  ref_text: Optional[str] = None,
                  speaker_description: Optional[str] = None) -> dict:
    """Save an uploaded voice to disk with per-file metadata. Returns the voice info dict."""
    existing_count = _count_uploaded_voices()
    existing_meta = _read_voice_metadata(name)
    if existing_count >= MAX_UPLOADED_VOICES and existing_meta is None:
        raise ValueError(f"Maximum number of uploaded voices ({MAX_UPLOADED_VOICES}) reached")

    if existing_meta is not None:
        LOGGER.warning(f"Overwriting existing uploaded voice '{name}'")

    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    # Write WAV atomically (tmp + rename)
    wav_path = _voice_wav_path(name)
    wav_tmp = wav_path.with_suffix(".wav.tmp")
    wav_tmp.write_bytes(audio_bytes)
    os.replace(wav_tmp, wav_path)

    now = int(time.time())
    info: dict = {
        "name": name,
        "consent": consent,
        "created_at": now,
        "file_size": len(audio_bytes),
        "mime_type": "audio/wav",
    }
    if ref_text:
        info["ref_text"] = ref_text
    if speaker_description:
        info["speaker_description"] = speaker_description

    # Write JSON atomically (tmp + rename) — this is the "committed" marker
    _write_voice_metadata(name, info)

    # Reload voice_list to include the new upload
    _reload_voice_list()

    return info


def _delete_voice(name: str) -> None:
    """Delete an uploaded voice (both WAV and JSON sidecar)."""
    meta_path = _voice_metadata_path(name)
    wav_path = _voice_wav_path(name)

    if not meta_path.exists() and not wav_path.exists():
        raise FileNotFoundError(f"Voice '{name}' not found")

    # Remove JSON first (marks as deleted), then WAV
    if meta_path.exists():
        meta_path.unlink()
    if wav_path.exists():
        wav_path.unlink()

    # Reload voice_list
    _reload_voice_list()

# Global voice_list — used by tts_streamer to resolve speaker names to audio file paths.
# Uploaded voices take precedence over built-in ones.
voice_list: Dict[str, Path] = {}
uploaded_voices_metadata: Dict[str, dict] = {}


def _reload_voice_list() -> None:
    """Reload the voice_list from disk. Called after uploads/deletes."""
    global voice_list, uploaded_voices_metadata
    voice_list = _get_all_available_voices()
    uploaded_voices_metadata = _load_uploaded_voices_metadata()


_reload_voice_list()

# --- Language normalization: full name → short code ---
_LANGUAGE_MAP: Dict[str, str] = {
    "english": "en", "en": "en",
    "indonesian": "id", "indonesia": "id", "id": "id",
    "chinese": "zh", "mandarin": "zh", "zh": "zh",
    "indian": "in", "hindi": "in", "in": "in",
    "arabic": "ar", "ar": "ar",
    "japanese": "ja", "ja": "ja",
    "korean": "ko", "ko": "ko",
    "french": "fr", "fr": "fr",
    "german": "de", "de": "de",
    "spanish": "es", "es": "es",
    "portuguese": "pt", "pt": "pt",
    "russian": "ru", "ru": "ru",
    "italian": "it", "it": "it",
}

# Lazy-loaded language detector (requires langdetect: pip install langdetect)
_detector_factory = None


def _get_detector():
    """Return the detect function from langdetect, or None if unavailable."""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # deterministic results
        return detect
    except ImportError:
        return None


def _auto_detect_language(text: str) -> str:
    """Detect language from text using langdetect. Falls back to 'en' if unavailable."""
    detect = _get_detector()
    if detect is None:
        return "en"
    try:
        iso = detect(text)
        return _LANGUAGE_MAP.get(iso, iso)
    except Exception:
        return "en"


def _normalize_language(lang: str, text: str = "") -> str:
    """Convert a language name to a short code. If 'auto', detect from text."""
    lang = lang.strip().lower()
    if lang in ("auto", "automatic"):
        return _auto_detect_language(text)
    return _LANGUAGE_MAP.get(lang, lang)

class DataQueue:

    def __init__(self, max_batch_size: int, model: VibeVoiceForConditionalGenerationInference, processor: VibeVoiceProcessor):
        self.active_queue: List[VibeVoiceStepOutput] = []
        self.queue: List[VibeVoiceStepOutput] = []
        self.max_batch_size = max_batch_size
        self.model = model
        self.processor = processor
        self.stopped = False
    
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
            # Resolve voice: try direct match in voice_list first, then lang-speaker combo
            voice_path = None
            if speaker in voice_list:
                voice_path = voice_list[speaker]
            elif f"{lang}-{speaker}" in voice_list:
                voice_path = voice_list[f"{lang}-{speaker}"]
            else:
                # Try partial match: any key containing the speaker name
                for key in voice_list:
                    if speaker.lower() in key.lower():
                        voice_path = voice_list[key]
                        break
            if voice_path is None:
                raise ValueError(f"Voice '{speaker}' not found for language '{lang}'")
            voice_sample = voice_path.as_posix()
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

    def set_stopped(self, stopped: bool):
        self.stopped = stopped
    
    def infinite_loop_step(self):
        "infinite loop in another thread and the exit if interrupted or get killed"

        log_counter = 0
        try:
            while True:
                if self.stopped:
                    break
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, data_queue, executor

    processor = VibeVoiceProcessor.from_pretrained(config.model_path)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
        device_map="cuda",
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)
    data_queue = DataQueue(max_batch_size=config.max_batch_size, model=model, processor=processor)
    loop = asyncio.get_event_loop()
    infinite_thread = loop.run_in_executor(executor, data_queue.infinite_loop_step)
    LOGGER.info("Startup: model should be loaded here")
    yield
    data_queue.set_stopped(True)
    await infinite_thread
    LOGGER.info("Shutdown: clean up resources if needed")

app = FastAPI(title='VibeVoice TTS API',
    description='API for generating speech using VibeVoice model with streaming WAV responses.',
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


# --- helper to build a complete WAV file from PCM16 bytes ---
def build_wav_from_pcm(pcm_bytes: bytes, sample_rate: int, num_channels: int, sampwidth: int = 2) -> bytes:
    """
    Create a valid WAV file (RIFF) containing pcm_bytes (little-endian PCM16).
    sampwidth is bytes per sample (2 for PCM16).
    """
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    out.seek(0)
    return out.read()


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


@app.get("/tts", response_class=FileResponse)
async def gen_wav(
    request: Request,
    text: str = Query(...),
    speaker: str = Query("alloy"),
    lang: Literal["en", "id"] = Query('id'),
    do_sample: bool = Query(False),
    temperature: float = Query(1.0, ge=0.0, le=2.0),
    top_p: float = Query(1.0, le=1.0, ge=0.0),
    do_stream: bool = Query(False)
):
    """
    Streams a TTS response produced by the model.
    - only one caller at a time (asyncio.Lock).
    - cooperative stop via audio_streamer.end().
    - streaming binary WAV via chunked transfer.
    """

    return await tts_streamer(request, text, None, speaker, lang, do_sample, temperature, top_p, do_stream)

@app.post("/tts", response_class=FileResponse)
async def generate_wav(
    request: Request,
    text: str = Body(...),
    speaker: str = Body("alloy"),
    lang: Literal["en", "id"] = Body('id'),
    do_sample: bool = Body(False),
    temperature: float = Body(1.0, ge=0.0, le=2.0),
    top_p: float = Body(1.0, le=1.0, ge=0.0),
    do_stream: bool = Body(False)
):
    """
    Streams a TTS response produced by the model.
    - only one caller at a time (asyncio.Lock).
    - cooperative stop via audio_streamer.end().
    - streaming binary WAV via chunked transfer.
    """

    return await tts_streamer(request, text, None, speaker, lang, do_sample, temperature, top_p, do_stream)

@app.post("/voice_clone", response_class=FileResponse)
async def voice_clone_form(
    request: Request,
    text: str = Form(...),
    audio_file: UploadFile = File(...),
    do_sample: bool = Form(False),
    temperature: float = Form(1.0),
    top_p: float = Form(1.0),
    do_stream: bool = Form(False)
):
    """
    Clones a voice from the provided audio file in form upload.
    """
    audio_bytes = await audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    return await tts_streamer(request, text, audio_base64, "alloy", "id", do_sample, temperature, top_p, do_stream)

@app.post("/add_voice_sample")
async def add_voice_sample(
    request: Request,
    lang: Literal["en", "id"] = Body(...),
    speaker: str = Body(...),
    audio_file: UploadFile = File(...)
):
    """
    Adds a new voice sample to the voice list for cloning.
    Uses the unified uploaded-voices directory.
    """
    name = f"{lang}-{speaker}"
    audio_bytes = await audio_file.read()
    try:
        _upload_voice(
            name=name,
            audio_bytes=audio_bytes,
            consent="legacy",
        )
    except ValueError as exc:
        return ORJSONResponse({"error": str(exc)}, status_code=400)
    return ORJSONResponse({"message": f"Added voice sample for {speaker} in {lang}."})

@app.delete("/remove_voice_sample")
async def remove_voice_sample(
    request: Request,
    lang: Literal["en", "id"] = Query(...),
    speaker: str = Query(...)
):
    """
    Removes a voice sample from the voice list.
    Uses the unified uploaded-voices directory.
    """
    name = f"{lang}-{speaker}"
    try:
        _delete_voice(name)
        return ORJSONResponse({"message": f"Removed voice sample for {speaker} in {lang}."})
    except FileNotFoundError:
        return ORJSONResponse({"error": f"Voice sample for {speaker} in {lang} not found."}, status_code=404)

@app.get('/voices')
async def list_voices():
    """
    Lists all available voice samples.
    """
    return ORJSONResponse({
        "object": "list",
        "data": [{"lang": key.split('-')[0], "speaker": key.split('-')[1]} for key in voice_list.keys()],
    })


# =============================================================================
# OpenAI-compatible /v1/audio API endpoints
# =============================================================================

@app.post("/v1/audio/speech")
async def generate_speech(
    request: Request,
    body: Dict[str, Any] = Body(...),
):
    """
    OpenAI-compatible text-to-speech endpoint.
    Generates audio from the input text with the specified voice.

    Request body (JSON):
        - input (str, required): The text to synthesize into speech.
        - model (str, optional): Model to use. Ignored — server runs one model.
        - voice (str, optional): Speaker name. Defaults to "alloy".
        - response_format (str, optional): Audio format: wav, mp3, flac, pcm. Default "wav".
        - speed (float, optional): Playback speed (0.25-4.0). Default 1.0.
        - language (str, optional): Language code (e.g. "en", "id"). Default "en".
        - instructions (str, optional): Voice style/emotion instructions (not used by VibeVoice).
    """
    text = body.get("input", "").strip()
    if not text:
        return ORJSONResponse({
            "error": {"message": "Input text cannot be empty", "type": "BadRequestError", "param": "input", "code": 400}
        }, status_code=400)

    voice = body.get("voice", "alloy")
    lang = _normalize_language(body.get("language", "en"), text)
    response_format = body.get("response_format", "wav")
    speed = float(body.get("speed", 1.0))

    # Validate & clamp speed
    if speed < 0.25 or speed > 4.0:
        return ORJSONResponse({
            "error": {"message": "Speed must be between 0.25 and 4.0", "type": "BadRequestError", "param": "speed", "code": 400}
        }, status_code=400)

    # Note: "instructions" and "model" params are accepted but not used by VibeVoice

    return await tts_streamer(
        request,
        text=text,
        audio_base64=None,
        speaker=voice,
        lang=lang,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        do_stream=False,
    )


@app.get("/v1/audio/voices")
async def list_voices_v1():
    """
    Lists available voices for the loaded model.
    Returns both built-in voice presets and uploaded voices.
    """
    builtin_voices = _get_builtin_voices()
    uploaded = _get_uploaded_voices()
    metadata = _load_uploaded_voices_metadata()

    # All voice names (built-in + uploaded)
    all_names = sorted(set(builtin_voices.keys()) | set(uploaded.keys()))

    # Build uploaded_voices list per spec
    uploaded_list = []
    for name in sorted(uploaded.keys()):
        info = metadata.get(name, {})
        entry: dict = {
            "name": name,
            "consent": info.get("consent", ""),
            "created_at": info.get("created_at", 0),
            "file_size": info.get("file_size", 0),
            "mime_type": info.get("mime_type", "audio/wav"),
        }
        if "ref_text" in info:
            entry["ref_text"] = info["ref_text"]
        if "speaker_description" in info:
            entry["speaker_description"] = info["speaker_description"]
        uploaded_list.append(entry)

    return ORJSONResponse({
        "voices": all_names,
        "uploaded_voices": uploaded_list,
    })


@app.post("/v1/audio/voices")
async def upload_voice(
    request: Request,
    audio_sample: UploadFile = File(...),
    consent: str = Form(...),
    name: str = Form(...),
    ref_text: Optional[str] = Form(None),
    speaker_description: Optional[str] = Form(None),
):
    """
    Upload a new voice sample for voice cloning.

    Form parameters:
        - audio_sample (file, required): Audio file (max 10MB).
        - consent (str, required): Consent recording ID.
        - name (str, required): Name for the new voice.
        - ref_text (str, optional): Transcript of the audio.
        - speaker_description (str, optional): Description of the voice.
    """
    # Validate audio file size (max 10MB)
    audio_bytes = await audio_sample.read()
    max_size = 10 * 1024 * 1024  # 10 MB
    if len(audio_bytes) > max_size:
        return ORJSONResponse({
            "error": {"message": "Audio file exceeds maximum size of 10MB", "type": "BadRequestError", "code": 400}
        }, status_code=400)

    # Validate name
    if not name.strip():
        return ORJSONResponse({
            "error": {"message": "Voice name cannot be empty", "type": "BadRequestError", "code": 400}
        }, status_code=400)

    # Validate consent
    if not consent.strip():
        return ORJSONResponse({
            "error": {"message": "Consent cannot be empty", "type": "BadRequestError", "code": 400}
        }, status_code=400)

    # Detect MIME type from uploaded filename
    mime_type = audio_sample.content_type or "audio/wav"
    supported_mimes = {"audio/wav", "audio/mpeg", "audio/mp3", "audio/flac",
                       "audio/ogg", "audio/aac", "audio/webm", "video/mp4",
                       "application/octet-stream"}
    # We accept application/octet-stream as a fallback

    try:
        info = _upload_voice(
            name=name.strip(),
            audio_bytes=audio_bytes,
            consent=consent.strip(),
            ref_text=ref_text.strip() if ref_text else None,
            speaker_description=speaker_description.strip() if speaker_description else None,
        )
        return ORJSONResponse({
            "success": True,
            "voice": info,
        })
    except ValueError as exc:
        return ORJSONResponse({
            "error": {"message": str(exc), "type": "BadRequestError", "code": 400}
        }, status_code=400)


@app.delete("/v1/audio/voices/{name}")
async def delete_voice(name: str):
    """
    Delete an uploaded voice sample.

    Path parameters:
        - name (str, required): Name of the voice to delete.
    """
    try:
        _delete_voice(name)
        return ORJSONResponse({
            "success": True,
            "message": f"Voice '{name}' deleted successfully",
        })
    except FileNotFoundError:
        return ORJSONResponse({
            "success": False,
            "error": f"Voice '{name}' not found",
        }, status_code=404)


@app.websocket("/v1/audio/speech/stream")
async def websocket_speech_stream(ws: WebSocket):
    """
    WebSocket endpoint for streaming text input and streaming audio output.
    Audio is generated per-sentence while text is still being received.

    Client -> Server:
        {"type": "session.config", ...}   — first message
        {"type": "input.text", "text": "..."} — text chunks
        {"type": "input.done"}            — end of input

    Server -> Client:
        {"type": "audio.start", "sentence_index": N, "sentence_text": "...", ...}
        <binary PCM16>  (when stream_audio=true)
        {"type": "audio.done", "sentence_index": N, ...}
        {"type": "session.done", "total_sentences": N}
        {"type": "error", "message": "..."}
    """
    global data_queue

    sample_rate = 24000
    num_channels = 1
    batch_size = 1

    config_params: Dict[str, Any] = {
        "voice": "alloy",
        "lang": "en",
        "stream_audio": False,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "silence_duration": 0.3,
    }

    try:
        await ws.accept()

        # 1. Expect session.config first
        raw = await ws.receive_json()
        if raw.get("type") != "session.config":
            await ws.send_json({"type": "error", "message": "First message must be session.config"})
            await ws.close()
            return

        config_params["voice"] = raw.get("voice", config_params["voice"])
        config_params["lang"] = raw.get("language", raw.get("lang", config_params["lang"]))
        # Store raw value for per-sentence resolution (supports "auto")
        raw_lang = config_params["lang"]
        config_params["stream_audio"] = raw.get("stream_audio", config_params["stream_audio"])
        config_params["do_sample"] = raw.get("do_sample", config_params["do_sample"])
        config_params["temperature"] = float(raw.get("temperature", config_params["temperature"]))
        config_params["top_p"] = float(raw.get("top_p", config_params["top_p"]))
        config_params["silence_duration"] = float(raw.get("silence_duration", config_params["silence_duration"]))

        speaker = config_params["voice"]
        lang = config_params["lang"]
        voice_found = (
            speaker in voice_list or
            f"{lang}-{speaker}" in voice_list or
            any(speaker.lower() in key.lower() for key in voice_list)
        )
        if not voice_found:
            await ws.send_json({"type": "error", "message": f"Voice '{speaker}' not found"})
            await ws.close()
            return

        # --- concurrent receiver / sender via sentence queue ---
        sentence_queue: asyncio.Queue = asyncio.Queue()
        disconnected = asyncio.Event()

        async def receiver():
            """Read text chunks, split into sentences, push to queue."""
            buffer = ""
            sentence_endings = re.compile(r'(?<=[.!?\n])\s+')
            try:
                while True:
                    msg = await ws.receive_json()
                    msg_type = msg.get("type", "")
                    if msg_type == "input.done":
                        remaining = buffer.strip()
                        if remaining:
                            await sentence_queue.put(remaining)
                        break
                    elif msg_type == "input.text":
                        buffer += msg.get("text", "")
                        while True:
                            match = sentence_endings.search(buffer)
                            if not match:
                                break
                            end = match.start() + 1
                            sentence = buffer[:end].strip()
                            buffer = buffer[end:].lstrip()
                            if sentence:
                                await sentence_queue.put(sentence)
                    else:
                        if not disconnected.is_set():
                            await ws.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})
            except WebSocketDisconnect:
                disconnected.set()
            except Exception:
                disconnected.set()
            finally:
                await sentence_queue.put(None)  # sentinel

        async def sender():
            """Pull sentences from queue, generate audio, send to client."""
            sentence_index = 0
            sentence = await sentence_queue.get()
            while sentence is not None:
                if disconnected.is_set():
                    sentence = await sentence_queue.get()
                    continue

                # Silence: full for most, half for comma-ending sentences
                sd = float(config_params["silence_duration"])
                if re.search(r'[,،]\s*$', sentence):
                    sd = sd / 2.0
                silence_samples = int(sd * sample_rate)
                silence_bytes = np.zeros(silence_samples, dtype=np.float32).astype(np.int16).tobytes()

                audio_streamer = AsyncAudioStreamer(batch_size=batch_size, timeout=1.0)
                sentence_lang = _normalize_language(raw_lang, sentence)
                data_queue.put(
                    sentence, None, sentence_lang, speaker,
                    config_params["do_sample"],
                    config_params["temperature"],
                    config_params["top_p"],
                    audio_streamer,
                )

                try:
                    await ws.send_json({
                        "type": "audio.start",
                        "sentence_index": sentence_index,
                        "sentence_text": sentence,
                        "format": "pcm",
                        "sample_rate": sample_rate,
                    })
                except WebSocketDisconnect:
                    disconnected.set()
                    audio_streamer.end()
                    sentence = await sentence_queue.get()
                    continue

                total_bytes = 0
                pcm_chunks: List[bytes] = []
                try:
                    async for batch_chunks in audio_streamer:
                        if disconnected.is_set():
                            break
                        if 0 in batch_chunks:
                            chunk = batch_chunks[0]
                            pcm_bytes = chunk_to_pcm16_bytes(chunk, num_channels)
                            if pcm_bytes:
                                if config_params["stream_audio"]:
                                    await ws.send_bytes(pcm_bytes)
                                else:
                                    pcm_chunks.append(pcm_bytes)
                                total_bytes += len(pcm_bytes)
                except WebSocketDisconnect:
                    disconnected.set()
                except Exception:
                    pass
                finally:
                    audio_streamer.end()

                if disconnected.is_set():
                    sentence = await sentence_queue.get()
                    continue

                # Peek next sentence; if more coming, append silence
                next_sentence = await sentence_queue.get()
                if next_sentence is not None and silence_bytes:
                    pcm_chunks.append(silence_bytes)
                    total_bytes += len(silence_bytes)

                all_pcm = b"".join(pcm_chunks)
                if not config_params["stream_audio"]:
                    try:
                        await ws.send_bytes(all_pcm)
                    except WebSocketDisconnect:
                        disconnected.set()
                        sentence = next_sentence
                        continue
                elif silence_bytes and next_sentence is not None:
                    try:
                        await ws.send_bytes(silence_bytes)
                    except WebSocketDisconnect:
                        disconnected.set()
                        sentence = next_sentence
                        continue

                try:
                    await ws.send_json({
                        "type": "audio.done",
                        "sentence_index": sentence_index,
                        "total_bytes": total_bytes,
                        "error": False,
                    })
                except WebSocketDisconnect:
                    disconnected.set()
                sentence_index += 1
                sentence = next_sentence

            if not disconnected.is_set():
                await ws.send_json({
                    "type": "session.done",
                    "total_sentences": sentence_index,
                })

        await asyncio.gather(receiver(), sender())

    except WebSocketDisconnect:
        LOGGER.info("WebSocket client disconnected")
    except Exception as exc:
        LOGGER.error(f"WebSocket error: {traceback.format_exc()}")
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass

async def tts_streamer(
    request: Request,
    text: str,
    audio_base64: Optional[str],
    speaker: str,
    lang: Literal["en", "id"],
    do_sample: bool,
    temperature: float,
    top_p: float,
    do_stream: bool
):
    global data_queue

    batch_size = 1  # streaming a single sample per request; adapt for batching
    sample_rate = 24000
    num_channels = 1

    # Resolve voice availability
    if audio_base64 is None:
        voice_found = (
            speaker in voice_list or
            f"{lang}-{speaker}" in voice_list or
            any(speaker.lower() in key.lower() for key in voice_list)
        )
        if not voice_found:
            return ORJSONResponse({"error": f"Speaker '{speaker}' not found for language '{lang}'."}, status_code=404)

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

    if not do_stream:
        pcm_chunks = []
        try:
            async for batch_chunks in audio_streamer:
                if 0 in batch_chunks:
                    chunk = batch_chunks[0]
                    pcm_bytes = chunk_to_pcm16_bytes(chunk, num_channels)
                    if pcm_bytes:
                        pcm_chunks.append(pcm_bytes)
                else:
                    for idx in sorted(batch_chunks.keys()):
                        chunk = batch_chunks[idx]
                        pcm_bytes = chunk_to_pcm16_bytes(chunk, num_channels)
                        if pcm_bytes:
                            pcm_chunks.append(pcm_bytes)
            # finished streaming; assemble all pcm bytes
            all_pcm = b"".join(pcm_chunks)
            full_wav = build_wav_from_pcm(all_pcm, sample_rate, num_channels, sampwidth=2)
            headers = {
                "Content-Type": "audio/wav",
                "Content-Length": str(len(full_wav))
            }
            return Response(content=full_wav, media_type="audio/wav", headers=headers)
        except asyncio.CancelledError:
            audio_streamer.end()
            raise
        except Exception:
            audio_streamer.end()
            raise
        finally:
            audio_streamer.end()

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
