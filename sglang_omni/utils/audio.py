# SPDX-License-Identifier: Apache-2.0
"""Shared audio utilities."""

from __future__ import annotations

import hashlib
import io
import os
import threading
from collections.abc import Mapping
from functools import lru_cache
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
import numpy as np
import pybase64
import torch
import torchaudio

_DEFAULT_REQUEST_TIMEOUT = 5
_CPU_RESAMPLER_CACHE_SIZE = 32
_CPU_RESAMPLER_CACHE_LOCK = threading.Lock()


@lru_cache(maxsize=_CPU_RESAMPLER_CACHE_SIZE)
def _create_cpu_resampler(
    source_sample_rate: int, target_sample_rate: int
) -> torchaudio.transforms.Resample:
    return torchaudio.transforms.Resample(source_sample_rate, target_sample_rate)


def _resample_cpu_audio(
    audio: torch.Tensor,
    source_sample_rate: int,
    target_sample_rate: int,
    resample_kwargs: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    if resample_kwargs:
        return torchaudio.functional.resample(
            audio,
            source_sample_rate,
            target_sample_rate,
            **dict(resample_kwargs),
        )

    # Serialize cache misses so concurrent first requests build one sinc kernel.
    with _CPU_RESAMPLER_CACHE_LOCK:
        resampler = _create_cpu_resampler(source_sample_rate, target_sample_rate)
    return resampler(audio)


def _is_riff_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _try_fast_wav_decode(
    data: bytes,
    target_sample_rate: int,
    resample_kwargs: Mapping[str, Any] | None = None,
) -> np.ndarray | None:
    # Note (akazaakane): Keep unsupported WAV encodings on torchaudio so the fast
    # path never narrows existing format coverage.
    from sglang_omni.preprocessing.audio import _parse_wav_bytes

    try:
        audio, sample_rate = _parse_wav_bytes(data)
    except ValueError:
        return None
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if not audio.flags.writeable:
        audio = audio.copy()
    if sample_rate == target_sample_rate:
        return audio

    resampled = _resample_cpu_audio(
        torch.from_numpy(audio),
        sample_rate,
        target_sample_rate,
        resample_kwargs,
    )
    return resampled.numpy()


def decode_audio_data_uri(value: str) -> bytes | None:
    if not value.startswith("data:"):
        return None
    header, separator, payload = value.partition(",")
    if not separator or ";base64" not in header.lower() or not payload:
        raise ValueError("Invalid base64 audio data URI")
    try:
        return pybase64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 audio data URI") from exc


def load_audio(
    source: Any,
    source_name: str = "audio",
    target_sample_rate: int = 16000,
    mono: bool = True,
    trim_top_db: float | None = None,
    resample_kwargs: Mapping[str, Any] | None = None,
) -> np.ndarray:
    if isinstance(source, memoryview):
        source = source.tobytes()
    if isinstance(source, bytearray):
        source = bytes(source)
    if isinstance(source, str):
        decoded = decode_audio_data_uri(source)
        if decoded is not None:
            source = decoded
        elif source.startswith(("http://", "https://")):
            try:
                timeout = int(
                    os.getenv("REQUEST_TIMEOUT", str(_DEFAULT_REQUEST_TIMEOUT))
                )
                if timeout <= 0:
                    timeout = _DEFAULT_REQUEST_TIMEOUT
            except ValueError:
                timeout = _DEFAULT_REQUEST_TIMEOUT
            response = httpx.get(source, timeout=timeout, follow_redirects=True)
            response.raise_for_status()
            source = response.content
        elif source.startswith("file://"):
            source = unquote(urlparse(source).path)

    if isinstance(source, bytes):
        # Note (akazaakane): The direct WAV/NumPy path avoids torchaudio decoder
        # startup when mono=True without changing channel-preserving loads.
        if mono and trim_top_db is None and _is_riff_wav(source):
            fast = _try_fast_wav_decode(
                source, target_sample_rate, resample_kwargs=resample_kwargs
            )
            if fast is not None:
                return fast
        audio, sample_rate = torchaudio.load(io.BytesIO(source))
    elif isinstance(source, str):
        audio, sample_rate = torchaudio.load(source)
    else:
        raise ValueError(
            f"Unsupported {source_name} audio input: {type(source).__name__}"
        )

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if mono and audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.to(torch.float32)
    if trim_top_db is not None:
        import librosa

        trimmed, _ = librosa.effects.trim(audio.numpy(), top_db=trim_top_db)
        audio = torch.from_numpy(trimmed)
    if sample_rate != target_sample_rate:
        audio = _resample_cpu_audio(
            audio,
            int(sample_rate),
            target_sample_rate,
            resample_kwargs,
        )
    if mono:
        audio = audio.squeeze(0)
    return audio.cpu().numpy()


def audio_fingerprint(audio: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(audio, dtype=np.float32)
    return hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest()


def audio_fingerprint_int(fingerprint: str) -> int:
    return int(fingerprint[:16], 16)
