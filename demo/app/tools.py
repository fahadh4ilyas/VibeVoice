import base64
import io
from typing import Tuple, Optional

import numpy as np

# Optional imports; we try to import and gracefully fallback if not installed
try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

try:
    import librosa
except Exception:
    librosa = None


def _strip_data_uri(b64: str) -> str:
    """Remove possible data:...;base64, prefix."""
    if b64.startswith("data:"):
        return b64.split("base64,")[-1]
    return b64


def base64_audio_to_numpy(
    b64_audio: str,
    *,
    mono: bool = True,
    target_sr: Optional[int] = None,
    dtype: str = "float32",
) -> Tuple[np.ndarray, int]:
    """
    Convert a base64-encoded audio blob (or data URI) into a numpy array.

    Returns:
        waveform: np.ndarray, shape (n_samples,) for mono or (n_samples, n_channels)
                  values normalized to [-1.0, 1.0] if dtype is float32/float64.
        sr: sample rate (int)

    Parameters:
      - b64_audio: base64 string or data URI (e.g. "data:audio/wav;base64,...")
      - mono: if True, convert to mono (averages channels)
      - target_sr: if provided, resample to this sample rate (requires librosa)
      - dtype: numpy dtype string for output (commonly "float32" or "float64")

    Notes:
      - For MP3 and other compressed formats, pydub + ffmpeg is used. Make sure ffmpeg is installed.
      - For WAV/FLAC, soundfile is preferred (faster, lossless).
    """

    b64_audio = _strip_data_uri(b64_audio).strip()
    audio_bytes = base64.b64decode(b64_audio)
    buffer = io.BytesIO(audio_bytes)

    # Try soundfile first (works for WAV, FLAC and other libsndfile-supported formats)
    if sf is not None:
        try:
            buffer.seek(0)
            data, sr = sf.read(buffer, dtype="float32")
            # data is float32 in range [-1, 1] for typical files
            waveform = np.asarray(data, dtype=dtype)
            # Ensure shape: (n_samples, n_channels?) -> if 1D it's mono already
            if waveform.ndim == 1:
                pass
            elif waveform.ndim == 2:
                # soundfile shapes as (n_samples, n_channels)
                pass
            else:
                # unexpected shape -- flatten channels to last axis
                waveform = waveform.reshape(waveform.shape[0], -1)
            # Convert to mono if requested
            if mono and waveform.ndim == 2:
                waveform = np.mean(waveform, axis=1)
            # resample if requested (use librosa if available)
            if target_sr is not None and sr != target_sr:
                if librosa is not None:
                    waveform = librosa.resample(waveform.astype("float32"), orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
                    waveform = waveform.astype(dtype)
                else:
                    # fallback: try pydub resampling (requires ffmpeg) if available
                    if AudioSegment is not None:
                        # re-read with pydub and set frame_rate
                        buffer.seek(0)
                        seg = AudioSegment.from_file(buffer)
                        seg = seg.set_frame_rate(target_sr)
                        arr = np.array(seg.get_array_of_samples())
                        channels = seg.channels
                        arr = arr.reshape((-1, channels)) if channels > 1 else arr
                        # normalize based on sample width
                        sample_width = seg.sample_width  # bytes
                        max_val = float(1 << (8 * sample_width - 1))
                        arr = arr.astype("float32") / max_val
                        waveform = np.mean(arr, axis=1) if mono and channels > 1 else arr.astype(dtype)
                        sr = target_sr
                    else:
                        raise RuntimeError("Resampling requested but neither librosa nor pydub is available.")
            return waveform, sr
        except Exception:
            # fallthrough to pydub
            pass

    # If soundfile failed or wasn't available, try pydub (good for mp3/ogg/other)
    if AudioSegment is None:
        raise RuntimeError(
            "Cannot decode audio: neither soundfile nor pydub is available. "
            "Install 'soundfile' and/or 'pydub' and ensure ffmpeg is installed for pydub."
        )

    try:
        buffer.seek(0)
        # pydub will guess format from headers; if that fails, user may pass explicit format
        seg = AudioSegment.from_file(buffer)

        sr = seg.frame_rate
        channels = seg.channels
        sample_width = seg.sample_width  # bytes per sample

        arr = np.array(seg.get_array_of_samples())

        if channels > 1:
            arr = arr.reshape((-1, channels))
        else:
            # shape is (n_samples,)
            pass

        # Convert integer PCM to float in [-1, 1]
        if sample_width == 1:
            # 8-bit PCM is unsigned
            arr = arr.astype("float32") - 128.0
            max_val = 128.0
        else:
            max_val = float(1 << (8 * sample_width - 1))
            arr = arr.astype("float32")

        waveform = arr / max_val

        if mono and waveform.ndim == 2:
            waveform = np.mean(waveform, axis=1)

        # If resampling requested and librosa present, use it
        if target_sr is not None and sr != target_sr:
            if librosa is not None:
                # librosa expects 1D arrays for mono; if multi-channel, resample each channel
                if waveform.ndim == 1:
                    waveform = librosa.resample(waveform.astype("float32"), orig_sr=sr, target_sr=target_sr)
                else:
                    # resample per channel
                    channels_out = []
                    for ch in range(waveform.shape[1]):
                        ch_res = librosa.resample(waveform[:, ch].astype("float32"), orig_sr=sr, target_sr=target_sr)
                        channels_out.append(ch_res)
                    waveform = np.stack(channels_out, axis=1)
                sr = target_sr
            else:
                # try pydub resample via set_frame_rate
                seg2 = seg.set_frame_rate(target_sr)
                arr2 = np.array(seg2.get_array_of_samples())
                ch2 = seg2.channels
                if ch2 > 1:
                    arr2 = arr2.reshape((-1, ch2))
                if seg2.sample_width == 1:
                    arr2 = arr2.astype("float32") - 128.0
                    max_val2 = 128.0
                else:
                    max_val2 = float(1 << (8 * seg2.sample_width - 1))
                    arr2 = arr2.astype("float32")
                waveform = arr2 / max_val2
                if mono and waveform.ndim == 2:
                    waveform = np.mean(waveform, axis=1)
                sr = target_sr

        # ensure dtype
        waveform = waveform.astype(dtype)
        return waveform, sr

    except Exception as e:
        raise RuntimeError(f"Unable to decode audio from base64: {e}")