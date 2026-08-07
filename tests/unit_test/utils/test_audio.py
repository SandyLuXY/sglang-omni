# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import struct
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pybase64
import pytest
import torch

from sglang_omni.utils import audio
from sglang_omni.utils.audio import load_audio


@pytest.fixture
def clear_cpu_resampler_cache():
    audio._create_cpu_resampler.cache_clear()
    yield
    audio._create_cpu_resampler.cache_clear()


def _wav_bytes(
    num_samples: int = 1600, sample_rate: int = 16000, num_channels: int = 1
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(num_channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * num_samples * num_channels)
    return buffer.getvalue()


def test_load_audio_accepts_base64_data_uri() -> None:
    encoded = pybase64.b64encode(_wav_bytes()).decode("ascii")

    samples = load_audio(f"data:audio/wav;base64,{encoded}")

    assert samples.shape == (1600,)


def test_load_audio_rejects_non_base64_data_uri() -> None:
    with pytest.raises(ValueError, match="Invalid base64 audio data URI"):
        load_audio("data:audio/wav,not-base64")


def test_load_audio_can_preserve_channels() -> None:
    encoded = pybase64.b64encode(_wav_bytes(num_channels=2)).decode("ascii")

    samples = load_audio(f"data:audio/wav;base64,{encoded}", mono=False)

    assert samples.shape == (2, 1600)


def test_load_audio_accepts_file_uri(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    path.write_bytes(_wav_bytes())

    samples = load_audio(path.as_uri())

    assert samples.shape == (1600,)


class _FakeHTTPResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.raise_checked = False

    def raise_for_status(self) -> None:
        self.raise_checked = True


def test_load_audio_accepts_http_url(monkeypatch) -> None:
    response = _FakeHTTPResponse(_wav_bytes())
    calls = []

    def fake_get(url: str, *, timeout: int, follow_redirects: bool):
        calls.append(
            {
                "url": url,
                "timeout": timeout,
                "follow_redirects": follow_redirects,
            }
        )
        return response

    monkeypatch.setenv("REQUEST_TIMEOUT", "7")
    monkeypatch.setattr(audio.httpx, "get", fake_get)

    samples = load_audio("https://example.test/audio.wav")

    assert samples.shape == (1600,)
    assert calls == [
        {
            "url": "https://example.test/audio.wav",
            "timeout": 7,
            "follow_redirects": True,
        }
    ]
    assert response.raise_checked


def test_load_audio_uses_default_timeout_for_invalid_env(monkeypatch) -> None:
    response = _FakeHTTPResponse(_wav_bytes())
    calls = []

    def fake_get(url: str, *, timeout: int, follow_redirects: bool):
        calls.append(timeout)
        return response

    monkeypatch.setenv("REQUEST_TIMEOUT", "abc")
    monkeypatch.setattr(audio.httpx, "get", fake_get)

    samples = load_audio("https://example.test/audio.wav")

    assert samples.shape == (1600,)
    assert calls == [audio._DEFAULT_REQUEST_TIMEOUT]


def _sine_wav_bytes(
    sample_rate: int = 16000,
    num_channels: int = 1,
    duration_s: float = 0.1,
    sampwidth: int = 2,
) -> bytes:
    num_samples = int(sample_rate * duration_s)
    t = np.arange(num_samples) / sample_rate
    samples = np.sin(2 * np.pi * 440.0 * t)
    if sampwidth == 2:
        frames = (samples * 32767).astype("<i2")
    elif sampwidth == 3:
        i32 = (samples * 8388607).astype("<i4")
        frames = np.zeros(num_samples * 3, dtype=np.uint8)
        raw = i32.astype("<i4").tobytes()
        for i in range(num_samples):
            frames[i * 3 : i * 3 + 3] = np.frombuffer(
                raw[i * 4 : i * 4 + 3], dtype=np.uint8
            )
    else:
        raise ValueError(sampwidth)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(num_channels)
        wav.setsampwidth(sampwidth)
        wav.setframerate(sample_rate)
        if num_channels > 1:
            interleaved = np.repeat(frames, num_channels) if sampwidth == 2 else frames
            wav.writeframes(interleaved.tobytes())
        else:
            wav.writeframes(frames.tobytes())
    return buffer.getvalue()


def _float32_wav_bytes(sample_rate: int = 16000, num_samples: int = 1600) -> bytes:
    t = np.arange(num_samples) / sample_rate
    samples = np.sin(2 * np.pi * 440.0 * t).astype("<f4")
    data = samples.tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100, 48000])
def test_load_audio_fast_path_matches_torchaudio(monkeypatch, sample_rate) -> None:
    wav = _sine_wav_bytes(sample_rate=sample_rate)

    fast = load_audio(wav)

    monkeypatch.setattr(audio, "_is_riff_wav", lambda data: False)
    slow = load_audio(wav)

    assert fast.dtype == slow.dtype == np.float32
    assert fast.shape == slow.shape
    np.testing.assert_allclose(fast, slow, atol=1e-6)


def test_load_audio_fast_path_matches_torchaudio_stereo(monkeypatch) -> None:
    wav = _sine_wav_bytes(num_channels=2)

    fast = load_audio(wav)

    monkeypatch.setattr(audio, "_is_riff_wav", lambda data: False)
    slow = load_audio(wav)

    np.testing.assert_allclose(fast, slow, atol=1e-6)


def test_load_audio_fast_path_handles_float32_wav() -> None:
    samples = load_audio(_float32_wav_bytes())

    assert samples.shape == (1600,)
    assert samples.dtype == np.float32
    assert samples.flags.writeable
    samples[0] = 0


def test_load_audio_fast_path_skips_torchaudio(monkeypatch) -> None:
    def fail_load(*args, **kwargs):
        raise AssertionError("torchaudio.load should not be called on the fast path")

    monkeypatch.setattr(audio.torchaudio, "load", fail_load)

    samples = load_audio(_sine_wav_bytes())

    assert samples.shape == (1600,)


def test_load_audio_fast_path_resamples(monkeypatch) -> None:
    def fail_load(*args, **kwargs):
        raise AssertionError("torchaudio.load should not be called on the fast path")

    monkeypatch.setattr(audio.torchaudio, "load", fail_load)

    samples = load_audio(_sine_wav_bytes(sample_rate=48000))

    assert samples.shape == (1600,)


def test_load_audio_reuses_resampler_across_fast_and_generic_paths(
    monkeypatch, clear_cpu_resampler_cache
) -> None:
    wav = _sine_wav_bytes(sample_rate=48000)
    real_resample = audio.torchaudio.functional.resample
    constructed = []

    class FakeResampler:
        def __init__(self, source_sample_rate: int, target_sample_rate: int) -> None:
            constructed.append((source_sample_rate, target_sample_rate))
            self.source_sample_rate = source_sample_rate
            self.target_sample_rate = target_sample_rate

        def __call__(self, waveform):
            return real_resample(
                waveform, self.source_sample_rate, self.target_sample_rate
            )

    monkeypatch.setattr(audio.torchaudio.transforms, "Resample", FakeResampler)

    fast = load_audio(wav)
    monkeypatch.setattr(audio, "_is_riff_wav", lambda data: False)
    generic = load_audio(wav)

    assert constructed == [(48000, 16000)]
    np.testing.assert_allclose(fast, generic, atol=1e-6)


def test_cpu_resampler_cache_separates_sample_rate_pairs(
    monkeypatch, clear_cpu_resampler_cache
) -> None:
    real_resampler = audio.torchaudio.transforms.Resample
    constructed = []

    def counting_resampler(source_sample_rate: int, target_sample_rate: int):
        constructed.append((source_sample_rate, target_sample_rate))
        return real_resampler(source_sample_rate, target_sample_rate)

    monkeypatch.setattr(audio.torchaudio.transforms, "Resample", counting_resampler)

    load_audio(_sine_wav_bytes(sample_rate=48000), target_sample_rate=16000)
    load_audio(_sine_wav_bytes(sample_rate=48000), target_sample_rate=8000)

    assert constructed == [(48000, 16000), (48000, 8000)]


def test_cpu_resampler_cache_serializes_concurrent_first_use(
    monkeypatch, clear_cpu_resampler_cache
) -> None:
    construction_count = 0
    count_lock = threading.Lock()

    class CountingResampler:
        def __init__(self, source_sample_rate: int, target_sample_rate: int) -> None:
            nonlocal construction_count
            with count_lock:
                construction_count += 1
            time.sleep(0.01)

        def __call__(self, waveform):
            return waveform[..., ::3]

    monkeypatch.setattr(audio.torchaudio.transforms, "Resample", CountingResampler)
    waveform = np.zeros(4800, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: audio._resample_cpu_audio(
                    torch.from_numpy(waveform), 48000, 16000
                ),
                range(8),
            )
        )

    assert construction_count == 1
    assert all(result.shape == (1600,) for result in results)


def test_custom_resampling_policy_uses_functional_path(
    monkeypatch, clear_cpu_resampler_cache
) -> None:
    wav = _sine_wav_bytes(sample_rate=48000)
    real_resample = audio.torchaudio.functional.resample
    calls = []

    def fail_resampler(*args, **kwargs):
        raise AssertionError("custom resampling policy must not use cached defaults")

    def counting_resample(waveform, source_rate, target_rate, **kwargs):
        calls.append((source_rate, target_rate, kwargs))
        return real_resample(waveform, source_rate, target_rate, **kwargs)

    monkeypatch.setattr(audio.torchaudio.transforms, "Resample", fail_resampler)
    monkeypatch.setattr(audio.torchaudio.functional, "resample", counting_resample)

    samples = load_audio(wav, resample_kwargs={"lowpass_filter_width": 8})

    assert samples.shape == (1600,)
    assert calls == [(48000, 16000, {"lowpass_filter_width": 8})]


def test_load_audio_trims_before_resampling() -> None:
    import librosa
    import torch
    import torchaudio

    sample_rate = 16000
    tone = np.sin(2 * np.pi * 440 * np.arange(sample_rate // 2) / sample_rate)
    values = np.pad(tone, (sample_rate, sample_rate))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((values * 32767).astype("<i2").tobytes())
    wav = buffer.getvalue()
    raw, sample_rate = torchaudio.load(io.BytesIO(wav))
    expected, _ = librosa.effects.trim(raw.numpy(), top_db=30)
    resample_kwargs = {
        "lowpass_filter_width": 64,
        "rolloff": 0.95,
        "resampling_method": "sinc_interp_kaiser",
    }
    expected = torchaudio.functional.resample(
        torch.from_numpy(expected), sample_rate, 48000, **resample_kwargs
    )

    samples = load_audio(
        wav,
        target_sample_rate=48000,
        trim_top_db=30,
        resample_kwargs=resample_kwargs,
    )

    assert samples.shape == expected.squeeze(0).shape
    np.testing.assert_allclose(samples, expected.squeeze(0).numpy())


def test_load_audio_falls_back_when_not_mono() -> None:
    samples = load_audio(_sine_wav_bytes(num_channels=2), mono=False)

    assert samples.shape == (2, 1600)


def test_load_audio_falls_back_for_24bit_pcm() -> None:
    samples = load_audio(_sine_wav_bytes(sampwidth=3))

    assert samples.shape == (1600,)
    assert samples.dtype == np.float32


def test_load_audio_falls_back_for_non_wav_bytes() -> None:
    assert audio._try_fast_wav_decode(b"\xffnot a wav" * 10, 16000) is None
    assert not audio._is_riff_wav(b"ID3\x04" + b"\x00" * 20)
