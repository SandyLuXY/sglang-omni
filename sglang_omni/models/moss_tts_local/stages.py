# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS Local (v1.5) pipeline."""

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import queue
import tempfile
import threading
import time
import wave
from collections.abc import Sequence
from typing import Any

import torch

from sglang_omni.models.moss_tts.stages import (
    _load_moss_processor_class,
    _moss_transformers_processor_compat,
    _resolve_checkpoint,
)
from sglang_omni.models.moss_tts_local.payload_types import (
    MossTTSLocalState,
    moss_tts_local_special_token_defaults,
)
from sglang_omni.models.moss_tts_local.request_builders import (
    cleanup_prepared_moss_tts_local_request,
    make_moss_tts_local_scheduler_adapters,
    preprocess_moss_tts_local_payload,
    set_moss_tts_local_preprocessing_context,
)
from sglang_omni.preprocessing.cache_key import (
    reference_path_cache_key as _reference_path_cache_key,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_MOSS_TTS_LOCAL_INSTALL_HINT = (
    "MOSS-TTS Local support requires the upstream custom Transformers code. "
    "Launch with trust_remote_code=True and make sure the checkpoint can load "
    "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2."
)

# Containers where upstream MOSS exposes projected transformer encoder blocks.
# Compile only those blocks; broader encode wrappers, patched pretransforms and
# quantizers have shape/control-flow churn and are intentionally excluded.
_AUDIO_ENCODER_COMPILE_TARGET_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("audio_tokenizer", "encoder"),
)
_DEFAULT_AUDIO_ENCODER_WARMUP_BATCH_SIZES = (1, 2, 4, 8)
_DEFAULT_AUDIO_ENCODER_LENGTH_BUCKET_SECONDS = (
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    100.0,
)
# Startup precompile only covers common short references; longer runtime buckets
# stay available and compile on first use instead of taxing every server start.
_DEFAULT_AUDIO_ENCODER_WARMUP_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0)

# NOTE: the preprocessing and vocoder stages each load their own processor
# (and thus their own ~4.3 GB bf16 codec instance). The codec's chunked decode
# flips module-global streaming state (`model.streaming()`), so a decode on a
# shared instance corrupts any concurrently running reference encode; with
# separate instances the encoder side only ever runs stateless forwards and the
# streaming decode stays confined to the single-threaded vocoder batch loop.


def load_state(payload: StagePayload) -> MossTTSLocalState:
    return MossTTSLocalState.from_dict(payload.data)


def store_state(payload: StagePayload, state: MossTTSLocalState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _normalize_processor_config(processor: Any) -> None:
    model_config = getattr(processor, "model_config", None)
    if model_config is None:
        return
    audio_vocab_size = int(getattr(model_config, "audio_vocab_size", 1024) or 1024)
    for attr, default in moss_tts_local_special_token_defaults(audio_vocab_size):
        if getattr(model_config, attr, None) is None:
            setattr(model_config, attr, default)


def _resolve_codec_device(device: str | None, gpu_id: int | None) -> str:
    """Pick the codec GPU for the preprocessing/vocoder stages.

    The ~1B-param codec encoder costs ~0.25 GPU-seconds per reference, which
    at concurrency 16 starves the AR engine when both share one device.
    The default config passes an explicit ``device`` so the second-GPU codec
    placement is visible in the pipeline config. ``gpu_id`` remains a fallback
    for custom colocated configs and launcher-injected runtime defaults.
    """
    if device:
        return device
    if gpu_id is not None:
        return f"cuda:{int(gpu_id)}"
    return "cuda:0"


def _resolve_audio_encoder_compile_targets(
    processor: Any,
) -> list[tuple[Any, str, torch.nn.Module, str]]:
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is None:
        raise RuntimeError(
            "MOSS-TTS Local audio encoder torch.compile is enabled, but the "
            "processor has no audio_tokenizer"
        )

    for path in _AUDIO_ENCODER_COMPILE_TARGET_CANDIDATES:
        owner = processor
        for attr in path[:-1]:
            owner = getattr(owner, attr, None)
            if owner is None:
                break
        if owner is None:
            continue

        attr_name = path[-1]
        target = getattr(owner, attr_name, None)
        if isinstance(target, torch.nn.ModuleList):
            children = [
                (index, child)
                for index, child in enumerate(target)
                if isinstance(child, torch.nn.Module)
                and _is_projected_transformer_like(child)
            ]
            if children:
                return [
                    (target, str(index), child, ".".join((*path, str(index))))
                    for index, child in children
                ]
        elif isinstance(target, torch.nn.Module) and _is_projected_transformer_like(
            target
        ):
            return [(owner, attr_name, target, ".".join(path))]

    candidates = ", ".join(
        ".".join(path) for path in _AUDIO_ENCODER_COMPILE_TARGET_CANDIDATES
    )
    raise RuntimeError(
        "MOSS-TTS Local audio encoder torch.compile is enabled, but no supported "
        "projected transformer encoder block was found. Checked: "
        f"{candidates}."
    )


def _resolve_audio_encoder_compile_target(
    processor: Any,
) -> tuple[Any, str, torch.nn.Module, str]:
    return _resolve_audio_encoder_compile_targets(processor)[0]


def _is_projected_transformer_like(module: torch.nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    if "projectedtransformer" in class_name:
        return True
    return all(
        hasattr(module, attr) for attr in ("input_proj", "transformer", "output_proj")
    )


def _normalize_audio_encoder_warmup_seconds(
    warmup_seconds: Sequence[float] | None,
) -> tuple[float, ...]:
    source = (
        _DEFAULT_AUDIO_ENCODER_WARMUP_SECONDS
        if warmup_seconds is None
        else warmup_seconds
    )
    values = tuple(float(value) for value in source)
    if not values:
        raise RuntimeError("MOSS-TTS Local audio encoder warm-up is empty")
    for value in values:
        if value <= 0:
            raise RuntimeError(
                "MOSS-TTS Local audio encoder warm-up seconds must be positive"
            )
    return values


def _replace_audio_encoder_compile_target(
    owner: Any,
    attr_name: str,
    module: torch.nn.Module,
) -> None:
    if isinstance(owner, torch.nn.ModuleList) and attr_name.isdigit():
        owner[int(attr_name)] = module
    else:
        setattr(owner, attr_name, module)


def _normalize_audio_encoder_length_bucket_seconds(
    bucket_seconds: Sequence[float] | None = None,
) -> tuple[float, ...]:
    source = (
        _DEFAULT_AUDIO_ENCODER_LENGTH_BUCKET_SECONDS
        if bucket_seconds is None
        else bucket_seconds
    )
    values = tuple(float(value) for value in source)
    if not values:
        raise RuntimeError("MOSS-TTS Local audio encoder length buckets are empty")
    previous = 0.0
    for value in values:
        if value <= 0:
            raise RuntimeError(
                "MOSS-TTS Local audio encoder length buckets must be positive"
            )
        if value <= previous:
            raise RuntimeError(
                "MOSS-TTS Local audio encoder length buckets must be strictly "
                "increasing"
            )
        previous = value
    return values


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return int(math.ceil(value / multiple) * multiple)


def _normalize_audio_encoder_batch_buckets(
    max_batch_size: int | None = None,
) -> tuple[int, ...]:
    max_batch_size = (
        int(_DEFAULT_AUDIO_ENCODER_WARMUP_BATCH_SIZES[-1])
        if max_batch_size is None
        else int(max_batch_size)
    )
    if max_batch_size <= 0:
        raise RuntimeError(
            "MOSS-TTS Local audio encoder max batch size must be positive"
        )

    buckets: list[int] = []
    bucket = 1
    while bucket < max_batch_size:
        buckets.append(bucket)
        bucket *= 2
    buckets.append(bucket)
    if not buckets:
        raise RuntimeError("MOSS-TTS Local audio encoder batch buckets are empty")
    return tuple(buckets)


def _bucket_audio_encoder_batch_size(batch_size: int, buckets: Sequence[int]) -> int:
    for bucket in buckets:
        if batch_size <= bucket:
            return int(bucket)
    return int(batch_size)


def _audio_tokenizer_sample_rate(audio_tokenizer: Any) -> int:
    return int(
        getattr(audio_tokenizer, "sampling_rate", 0)
        or getattr(getattr(audio_tokenizer, "config", None), "sampling_rate", 0)
        or getattr(getattr(audio_tokenizer, "config", None), "sample_rate", 0)
        or 48000
    )


def _audio_tokenizer_downsample_rate(audio_tokenizer: Any) -> int:
    return int(
        getattr(audio_tokenizer, "downsample_rate", 0)
        or getattr(getattr(audio_tokenizer, "config", None), "downsample_rate", 0)
        or 1
    )


def _bucket_audio_encoder_input_length(
    audio_tokenizer: Any,
    current_length: int,
    buckets: Sequence[float],
) -> int:
    sampling_rate = _audio_tokenizer_sample_rate(audio_tokenizer)
    downsample_rate = _audio_tokenizer_downsample_rate(audio_tokenizer)
    target_length = current_length
    for seconds in buckets:
        bucket_length = _ceil_to_multiple(
            int(math.ceil(seconds * sampling_rate)),
            downsample_rate,
        )
        if current_length <= bucket_length:
            target_length = bucket_length
            break
    else:
        target_length = _ceil_to_multiple(current_length, downsample_rate)
    return int(target_length)


def _infer_audio_encoder_code_lengths(
    audio_tokenizer: Any,
    input_lengths: torch.Tensor,
) -> torch.Tensor:
    lengths = input_lengths.clone()
    if int(getattr(audio_tokenizer, "number_channels", 1) or 1) > 1 and bool(
        getattr(audio_tokenizer, "enable_channel_interleave", False)
    ):
        lengths = lengths * int(getattr(audio_tokenizer, "number_channels", 1) or 1)

    encoder = getattr(audio_tokenizer, "encoder", None)
    if isinstance(encoder, torch.nn.ModuleList):
        modules = encoder
    elif isinstance(encoder, torch.nn.Module):
        modules = (encoder,)
    else:
        modules = ()

    for module in modules:
        patch_size = int(getattr(module, "patch_size", 0) or 0)
        if patch_size > 0 and hasattr(module, "is_downsample"):
            if bool(getattr(module, "is_downsample")):
                lengths = torch.div(lengths, patch_size, rounding_mode="floor")
            else:
                lengths = lengths * patch_size
            continue

        ratio = int(getattr(module, "downsample_ratio", 1) or 1)
        if ratio > 1 and "pretransform" in module.__class__.__name__.lower():
            lengths = torch.div(lengths, ratio, rounding_mode="floor")

    return lengths


def _slice_audio_encoder_output(
    result: Any, batch_size: int, lengths: torch.Tensor
) -> Any:
    max_valid_length = int(lengths.max().item()) if lengths.numel() > 0 else 0
    audio_codes = getattr(result, "audio_codes", None)
    audio_codes_lengths = getattr(result, "audio_codes_lengths", None)
    encoder_hidden_states = getattr(result, "encoder_hidden_states", None)

    if audio_codes is not None:
        audio_codes = audio_codes[:, :batch_size, :max_valid_length]
    if audio_codes_lengths is not None:
        audio_codes_lengths = lengths
    if encoder_hidden_states is not None:
        encoder_hidden_states = encoder_hidden_states[:batch_size, :, :max_valid_length]

    try:
        return result.__class__(
            audio_codes=audio_codes,
            audio_codes_lengths=audio_codes_lengths,
            encoder_hidden_states=encoder_hidden_states,
        )
    except Exception:
        pass

    if getattr(result, "audio_codes", None) is not None:
        result.audio_codes = audio_codes
    if getattr(result, "audio_codes_lengths", None) is not None:
        result.audio_codes_lengths = audio_codes_lengths
    if getattr(result, "encoder_hidden_states", None) is not None:
        result.encoder_hidden_states = encoder_hidden_states
    return result


def _install_audio_encoder_compile_padding(
    processor: Any,
    *,
    bucket_seconds: Sequence[float] | None = None,
    max_batch_size: int | None = None,
) -> None:
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is None:
        raise RuntimeError(
            "MOSS-TTS Local audio encoder torch.compile is enabled, but the "
            "processor has no audio_tokenizer"
        )
    missing_hooks = [
        name
        for name in ("_prepare_waveform_batch", "_encode_frame")
        if not hasattr(audio_tokenizer, name)
    ]
    if missing_hooks:
        raise RuntimeError(
            "MOSS-TTS Local audio encoder torch.compile requires upstream "
            f"audio tokenizer hooks: {', '.join(missing_hooks)}"
        )
    if getattr(audio_tokenizer, "_sglang_omni_compile_padding_installed", False):
        return

    buckets = _normalize_audio_encoder_length_bucket_seconds(bucket_seconds)
    batch_buckets = _normalize_audio_encoder_batch_buckets(max_batch_size)
    original_prepare = audio_tokenizer._prepare_waveform_batch
    original_encode_frame = audio_tokenizer._encode_frame

    def _bucketed_prepare_waveform_batch(self, wav_list):
        input_values, lengths = original_prepare(wav_list)
        current_length = int(input_values.shape[-1])
        if current_length <= 0:
            return input_values, lengths

        target_length = _bucket_audio_encoder_input_length(
            self, current_length, buckets
        )
        if target_length <= current_length:
            return input_values, lengths
        return (
            torch.nn.functional.pad(
                input_values,
                (0, target_length - current_length),
            ),
            lengths,
        )

    def _bucketed_encode_frame(self, input_values, input_lengths=None, *args, **kwargs):
        if input_values.dim() == 1:
            input_values = input_values.view(1, 1, -1)
        elif input_values.dim() == 2:
            if int(getattr(self, "number_channels", 1) or 1) == 1:
                input_values = input_values.unsqueeze(1)
            else:
                input_values = input_values.unsqueeze(0)

        batch_size = int(input_values.shape[0])
        current_length = int(input_values.shape[-1])
        if input_lengths is None:
            input_lengths = torch.full(
                (batch_size,),
                current_length,
                device=input_values.device,
                dtype=torch.long,
            )
        else:
            input_lengths = input_lengths.to(
                device=input_values.device, dtype=torch.long
            )

        original_lengths = input_lengths[:batch_size].clone()
        target_length = _bucket_audio_encoder_input_length(
            self,
            max(
                current_length,
                int(original_lengths.max().item()) if original_lengths.numel() else 0,
            ),
            buckets,
        )
        if target_length > current_length:
            input_values = torch.nn.functional.pad(
                input_values,
                (0, target_length - current_length),
            )

        target_batch_size = _bucket_audio_encoder_batch_size(batch_size, batch_buckets)
        bucketed_lengths = torch.full(
            (batch_size,),
            target_length,
            device=input_values.device,
            dtype=torch.long,
        )
        if target_batch_size > batch_size:
            pad_shape = (
                target_batch_size - batch_size,
                *input_values.shape[1:],
            )
            input_values = torch.cat(
                [input_values, input_values.new_zeros(pad_shape)],
                dim=0,
            )
            bucketed_lengths = torch.cat(
                [
                    bucketed_lengths,
                    torch.full(
                        (target_batch_size - batch_size,),
                        target_length,
                        device=input_values.device,
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )

        result = original_encode_frame(
            input_values,
            bucketed_lengths,
            *args,
            **kwargs,
        )
        code_lengths = _infer_audio_encoder_code_lengths(self, original_lengths)
        return _slice_audio_encoder_output(result, batch_size, code_lengths)

    audio_tokenizer._prepare_waveform_batch = _bucketed_prepare_waveform_batch.__get__(
        audio_tokenizer,
        audio_tokenizer.__class__,
    )
    audio_tokenizer._encode_frame = _bucketed_encode_frame.__get__(
        audio_tokenizer,
        audio_tokenizer.__class__,
    )
    setattr(audio_tokenizer, "_sglang_omni_compile_padding_installed", True)
    setattr(
        audio_tokenizer,
        "_sglang_omni_original_prepare_waveform_batch",
        original_prepare,
    )
    setattr(
        audio_tokenizer,
        "_sglang_omni_original_encode_frame",
        original_encode_frame,
    )
    logger.info(
        "Installed MOSS-TTS Local audio encoder compile buckets: lengths=[%s], "
        "batches=[%s]",
        ", ".join(f"{seconds:g}s" for seconds in buckets),
        ", ".join(str(batch_size) for batch_size in batch_buckets),
    )


def _restore_audio_encoder_compile_padding(processor: Any) -> None:
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is None:
        return
    original_prepare = getattr(
        audio_tokenizer,
        "_sglang_omni_original_prepare_waveform_batch",
        None,
    )
    if original_prepare is not None:
        audio_tokenizer._prepare_waveform_batch = original_prepare
    original_encode_frame = getattr(
        audio_tokenizer,
        "_sglang_omni_original_encode_frame",
        None,
    )
    if original_encode_frame is not None:
        audio_tokenizer._encode_frame = original_encode_frame
    for attr in (
        "_sglang_omni_compile_padding_installed",
        "_sglang_omni_original_prepare_waveform_batch",
        "_sglang_omni_original_encode_frame",
    ):
        try:
            delattr(audio_tokenizer, attr)
        except AttributeError:
            pass


def _compile_moss_tts_local_audio_encoder(
    processor: Any,
    *,
    mode: str | None = "default",
    warmup_seconds: Sequence[float] | None = None,
    max_batch_size: int | None = None,
) -> None:
    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "MOSS-TTS Local audio encoder torch.compile is enabled, but this "
            "PyTorch build does not provide torch.compile"
        )

    targets = _resolve_audio_encoder_compile_targets(processor)
    normalized_warmup_seconds = _normalize_audio_encoder_warmup_seconds(warmup_seconds)
    batch_buckets = _normalize_audio_encoder_batch_buckets(max_batch_size)
    _install_audio_encoder_compile_padding(
        processor,
        max_batch_size=max_batch_size,
    )
    if all(
        getattr(target, "_sglang_omni_torch_compiled", False)
        for _, _, target, _ in targets
    ):
        logger.info("MOSS-TTS Local audio encoder targets are already compiled")
        return

    compile_kwargs: dict[str, Any] = {"dynamic": True}
    if mode is not None:
        compile_kwargs["mode"] = mode

    replaced: list[tuple[Any, str, torch.nn.Module, str]] = []
    try:
        for owner, attr_name, target, target_name in targets:
            if getattr(target, "_sglang_omni_torch_compiled", False):
                continue
            compiled_target = torch.compile(target, **compile_kwargs)
            try:
                setattr(compiled_target, "_sglang_omni_torch_compiled", True)
            except Exception:
                pass
            _replace_audio_encoder_compile_target(owner, attr_name, compiled_target)
            replaced.append((owner, attr_name, target, target_name))
        _warm_up_moss_tts_local_audio_encoder(
            processor,
            warmup_seconds=normalized_warmup_seconds,
            batch_sizes=batch_buckets,
        )
    except Exception as exc:
        for owner, attr_name, target, _ in reversed(replaced):
            _replace_audio_encoder_compile_target(owner, attr_name, target)
        _restore_audio_encoder_compile_padding(processor)
        target_names = ", ".join(target_name for _, _, _, target_name in targets)
        logger.exception(
            "MOSS-TTS Local audio encoder torch.compile failed for %s",
            target_names,
        )
        raise RuntimeError(
            f"MOSS-TTS Local audio encoder torch.compile failed for {target_names}"
        ) from exc

    target_names = ", ".join(target_name for _, _, _, target_name in targets)
    logger.info(
        "Compiled MOSS-TTS Local audio encoder at %s (mode=%s, dynamic=True)",
        target_names,
        mode,
    )


def _processor_sample_rate(processor: Any) -> int:
    model_config = getattr(processor, "model_config", None)
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    tokenizer_config = getattr(audio_tokenizer, "config", None)
    return int(
        getattr(model_config, "sampling_rate", 0)
        or getattr(tokenizer_config, "sampling_rate", 0)
        or 48000
    )


def _write_silent_wav(path: str, *, sample_rate: int, duration_s: float) -> None:
    frame_count = max(int(sample_rate * duration_s), 1)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frame_count)


def _warm_up_moss_tts_local_audio_encoder(
    processor: Any,
    *,
    warmup_seconds: Sequence[float],
    batch_sizes: Sequence[int] = _DEFAULT_AUDIO_ENCODER_WARMUP_BATCH_SIZES,
) -> None:
    sample_rate = _processor_sample_rate(processor)
    for seconds in warmup_seconds:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            _write_silent_wav(path, sample_rate=sample_rate, duration_s=seconds)
            for batch_size in batch_sizes:
                processor.encode_audios_from_path([path] * batch_size)
        except Exception as exc:
            raise RuntimeError(
                "MOSS-TTS Local audio encoder torch.compile warm-up failed"
            ) from exc
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


def _load_moss_tts_local_processor(
    model_path: str,
    *,
    device: str,
    enable_audio_encoder_torch_compile: bool = False,
    audio_encoder_torch_compile_mode: str | None = "default",
    audio_encoder_torch_compile_warmup_seconds: Sequence[float] | None = None,
    audio_encoder_torch_compile_max_batch_size: int | None = None,
) -> Any:
    checkpoint_dir = _resolve_checkpoint(model_path)
    logger.info(
        "Loading MOSS-TTS Local processor from %s on %s", checkpoint_dir, device
    )
    try:
        with _moss_transformers_processor_compat():
            processor_cls = _load_moss_processor_class(checkpoint_dir)
            processor = processor_cls.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
            )
    except Exception as exc:
        raise RuntimeError(_MOSS_TTS_LOCAL_INSTALL_HINT) from exc

    _normalize_processor_config(processor)
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is not None:
        if hasattr(audio_tokenizer, "eval"):
            audio_tokenizer.eval()
        if hasattr(audio_tokenizer, "to"):
            # Device move only: the v2 codec manages its own dtypes (bf16
            # encoder/decoder with an fp32 quantizer); a blanket dtype cast
            # would corrupt the quantizer codebooks.
            audio_tokenizer.to(device)
    if enable_audio_encoder_torch_compile:
        _compile_moss_tts_local_audio_encoder(
            processor,
            mode=audio_encoder_torch_compile_mode,
            warmup_seconds=audio_encoder_torch_compile_warmup_seconds,
            max_batch_size=audio_encoder_torch_compile_max_batch_size,
        )
    return processor


def _build_usage(state: MossTTSLocalState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage = {
        "prompt_tokens": int(state.prompt_tokens),
        "completion_tokens": int(state.completion_tokens),
        "total_tokens": int(state.prompt_tokens + state.completion_tokens),
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


class _BatchedReferenceEncoder:
    """Coalesces concurrent reference-audio encodes into batched codec calls.

    Each request needs its reference run through the ~1B-param codec encoder
    (~0.25 GPU-seconds). The preprocessing workers call :meth:`encode`
    concurrently; a single daemon thread drains the queue and encodes up to
    ``max_batch_size`` files in one ``batch_encode`` forward, which costs
    barely more than a single encode. Failures fall back to per-item encodes
    so one bad file only fails its own request.
    """

    # Mirrors the Higgs reference-audio cap: bounds both encoder runtime and
    # the batch-padding memory amplification.
    MAX_REFERENCE_SECONDS = 100.0
    # An encode batch takes well under a second; a result this late means the
    # worker died or wedged, so fail the request instead of hanging the slot.
    ENCODE_TIMEOUT_S = 120.0

    def __init__(
        self,
        processor: Any,
        *,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 4,
    ) -> None:
        self._processor = processor
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._queue: queue.Queue[tuple[str, concurrent.futures.Future]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker, name="moss-local-ref-encode", daemon=True
        )
        self._thread.start()

    @classmethod
    def _check_reference_duration(cls, path: str) -> None:
        try:
            import torchaudio

            info = torchaudio.info(path)
            duration = info.num_frames / max(int(info.sample_rate), 1)
        except Exception:
            return  # unreadable files fail with a clearer error in the codec
        if duration > cls.MAX_REFERENCE_SECONDS:
            raise ValueError(
                f"reference audio is {duration:.1f}s long; the limit is "
                f"{cls.MAX_REFERENCE_SECONDS:.0f}s"
            )

    def encode(self, path: str) -> torch.Tensor:
        """Encode one reference file; blocks until its batch completes."""
        path = str(path)
        self._check_reference_duration(path)
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((path, future))
        return future.result(timeout=self.ENCODE_TIMEOUT_S)

    def _drain_batch(self) -> list[tuple[str, concurrent.futures.Future]]:
        batch = [self._queue.get()]
        while len(batch) < self._max_batch_size:
            try:
                if self._max_wait_s > 0:
                    batch.append(self._queue.get(timeout=self._max_wait_s))
                else:
                    batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _worker(self) -> None:
        while True:
            batch = self._drain_batch()
            unique_paths = list(dict.fromkeys(path for path, _ in batch))
            results: dict[str, Any] = {}
            try:
                encoded = self._processor.encode_audios_from_path(unique_paths)
                results = dict(zip(unique_paths, encoded))
            except Exception:
                logger.exception(
                    "MOSS-TTS Local batched reference encode failed; "
                    "retrying per item"
                )
                for path in unique_paths:
                    try:
                        results[path] = self._processor.encode_audios_from_path([path])[
                            0
                        ]
                    except Exception as exc:
                        results[path] = exc
            for path, future in batch:
                outcome = results.get(path)
                if isinstance(outcome, Exception):
                    # Fresh exception per future: a shared instance would be
                    # mutated concurrently by every waiter's traceback raise.
                    future.set_exception(
                        RuntimeError(f"reference encode failed for {path}: {outcome}")
                    )
                elif outcome is None:
                    future.set_exception(
                        RuntimeError(f"reference encode produced no codes: {path}")
                    )
                else:
                    future.set_result(outcome)


class CachedReferenceEncoder:
    """Content-addressed LRU cache + single-flight dedup in front of _BatchedReferenceEncoder.

    Miss path returns the encoder's tensor unchanged (bit-identical to cache-off).
    Hit path returns a fresh .clone().to(long) so callers cannot mutate cached state.
    Stores codes as int32 on CPU (lossless for codebook values in [0, 1023]).
    """

    # Cadence for the periodic stats log; class attr so it is easy to tune.
    LOG_INTERVAL_S = 60.0

    def __init__(
        self,
        encoder: _BatchedReferenceEncoder,
        *,
        max_items: int = 256,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        # Fail fast on non-positive capacities: a negative max_items makes
        # StageOutputCache evict from an empty dict and KeyError at request time.
        if max_items < 1:
            raise ValueError(f"ref_audio_cache_max_items must be >= 1, got {max_items}")
        if max_bytes < 1:
            raise ValueError(f"ref_audio_cache_max_bytes must be >= 1, got {max_bytes}")
        self._encoder = encoder
        self._cache = StageOutputCache(
            max_size=max_items,
            max_bytes=max_bytes,
            cache_device="cpu",
        )
        self._lock = threading.Lock()
        self._inflight: dict[str, concurrent.futures.Future] = {}
        self._hits = 0
        self._misses = 0
        self._merged = 0
        self._last_log_time: float = 0.0

    def encode(self, path: str) -> torch.Tensor:
        path = str(path)
        # Note(Jiaxin): duration gate runs first — a >100 s ref must never reach
        # the cache or the inflight dict.
        _BatchedReferenceEncoder._check_reference_duration(path)
        # trust_stat left False (review feedback): keep the sentinel byte-read so a
        # same-size+mtime+ctime overwrite cannot stale-hit. The flag stays available
        # in reference_path_cache_key for deployments that guarantee immutable refs.
        key = _reference_path_cache_key(path)
        if key is None:
            return self._encoder.encode(path)  # uncacheable (URL/missing) -> bypass
        return self._cached_encode(
            key,
            lambda: self._encoder.encode(path),
            desc=repr(path),
            # TOCTOU re-stat: skip the put if the file changed during the encode.
            revalidate=lambda: _reference_path_cache_key(path) == key,
        )

    def _cached_encode(
        self, key: str, encode_fn, *, desc: str, revalidate=None
    ) -> torch.Tensor:
        """Single-flight skeleton shared by encode() and encode_data_uri().

        Hit -> independent .clone().to(long). Miss leader runs encode_fn and returns
        its tensor unchanged (bit-identical to cache-off). revalidate(), if given, is
        evaluated outside the lock and gates the put (TOCTOU guard for file paths).
        """
        leader_fut: concurrent.futures.Future | None = None
        follower_fut: concurrent.futures.Future | None = None

        with self._lock:
            stored = self._cache.get(key)
            if stored is not None:
                self._hits += 1
            elif key in self._inflight:
                self._merged += 1
                follower_fut = self._inflight[key]
            else:
                self._misses += 1
                leader_fut = concurrent.futures.Future()
                self._inflight[key] = leader_fut

        if stored is not None:
            # Note(Jiaxin): clone on hit so callers can't mutate the shared entry.
            self._maybe_log()
            return stored.clone().to(torch.long)

        if follower_fut is not None:
            # Note(Jiaxin): each follower raises a FRESH RuntimeError — sharing one
            # exception instance lets concurrent re-raises corrupt its traceback
            # (same lesson as _BatchedReferenceEncoder._worker).
            timeout = _BatchedReferenceEncoder.ENCODE_TIMEOUT_S + 10
            try:
                stored = follower_fut.result(timeout=timeout)
            except Exception as cause:
                raise RuntimeError(
                    f"reference encode failed for {desc}: {cause}"
                ) from cause
            return stored.clone().to(torch.long)

        assert leader_fut is not None
        try:
            result = encode_fn()
        except BaseException as exc:
            with self._lock:
                self._inflight.pop(key, None)
            leader_fut.set_exception(exc)
            raise

        do_put = revalidate() if revalidate is not None else True
        stored = result.detach().to("cpu", dtype=torch.int32)
        with self._lock:
            if do_put:
                self._cache.put(key, stored)
            self._inflight.pop(key, None)
        leader_fut.set_result(stored)
        self._maybe_log()
        return result  # original tensor: miss path stays bit-identical to cache-off

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log_time < 60.0:
            return
        with self._lock:
            if now - self._last_log_time < self.LOG_INTERVAL_S:
                return
            self._last_log_time = now
            snapshot = (
                self._hits,
                self._misses,
                self._merged,
                len(self._cache._cache),
                self._cache.current_bytes,
            )
        logger.info(
            "MOSS-TTS Local ref cache: hits=%d misses=%d merged=%d entries=%d bytes=%d",
            *snapshot,
        )

    def encode_data_uri(self, ref_audio: str, *, processor: Any) -> torch.Tensor:
        """Cache-aware encode for data-URI refs through the same LRU + single-flight
        as file paths (adds the duration check _reference_for_processor lacks).

        Note(Jiaxin): file: and bytes: keyspaces never collide — the two decode
        chains differ, so codes aren't guaranteed identical for the "same" audio.
        """
        import base64
        import io

        from sglang_omni.models.moss_tts.request_builders import _DATA_URI_RE
        from sglang_omni.preprocessing.cache_key import hash_bytes as _hash_bytes

        match = _DATA_URI_RE.match(ref_audio)
        if match is None:
            raise ValueError(f"encode_data_uri: not a data URI ({ref_audio[:40]!r}...)")

        raw = base64.b64decode(match.group("data"))
        key = f"bytes:{_hash_bytes(raw)}"

        def _encode() -> torch.Tensor:
            import soundfile as sf

            audio, sample_rate = sf.read(
                io.BytesIO(raw), dtype="float32", always_2d=True
            )
            # Note(Jiaxin): the duration check runs inside the leader (not before
            # inflight registration like the file path) so concurrent same-payload
            # requests share one sf.read of a potentially large decoded buffer.
            duration = audio.shape[0] / max(int(sample_rate), 1)
            if duration > _BatchedReferenceEncoder.MAX_REFERENCE_SECONDS:
                raise ValueError(
                    f"reference audio is {duration:.1f}s long; the limit is "
                    f"{_BatchedReferenceEncoder.MAX_REFERENCE_SECONDS:.0f}s"
                )
            wav = torch.from_numpy(audio.T)
            return processor.encode_audios_from_wav([wav], int(sample_rate))[0]

        return self._cached_encode(key, _encode, desc="data-URI")

    def stats(self) -> dict:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "merged": self._merged,
                "entries": len(self._cache._cache),
                "bytes": self._cache.current_bytes,
            }


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int = 16,
    encode_batch_size: int = 8,
    encode_batch_wait_ms: int = 4,
    ref_audio_cache: bool = True,
    ref_audio_cache_max_items: int = 256,
    ref_audio_cache_max_bytes: int = 64 * 1024 * 1024,
    enable_audio_encoder_torch_compile: bool = False,
    audio_encoder_torch_compile_mode: str | None = "default",
    audio_encoder_torch_compile_warmup_seconds: Sequence[float] | None = None,
) -> SimpleScheduler:
    # MOSS_REF_AUDIO_CACHE=0 disables the cache at startup (ops kill switch / A-B
    # toggle) without a config edit; unset => kwarg default.
    env_toggle = os.environ.get("MOSS_REF_AUDIO_CACHE")
    if env_toggle is not None:
        ref_audio_cache = env_toggle.strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
    device = _resolve_codec_device(device, gpu_id)
    processor = _load_moss_tts_local_processor(
        model_path,
        device=device,
        enable_audio_encoder_torch_compile=enable_audio_encoder_torch_compile,
        audio_encoder_torch_compile_mode=audio_encoder_torch_compile_mode,
        audio_encoder_torch_compile_warmup_seconds=(
            audio_encoder_torch_compile_warmup_seconds
        ),
        audio_encoder_torch_compile_max_batch_size=encode_batch_size,
    )
    reference_encoder: Any = _BatchedReferenceEncoder(
        processor,
        max_batch_size=encode_batch_size,
        max_batch_wait_ms=encode_batch_wait_ms,
    )
    if ref_audio_cache:
        reference_encoder = CachedReferenceEncoder(
            reference_encoder,
            max_items=ref_audio_cache_max_items,
            max_bytes=ref_audio_cache_max_bytes,
        )
    set_moss_tts_local_preprocessing_context(
        processor=processor, reference_encoder=reference_encoder
    )
    # Reference encoding runs through the ~1B-param causal codec encoder, so
    # unlike MOSS Delay the audio tokenizer must live on the GPU; threads
    # release the GIL during the codec forward, keeping the AR engine fed.
    return SimpleScheduler(
        preprocess_moss_tts_local_payload,
        abort_callback=cleanup_prepared_moss_tts_local_request,
        max_concurrency=max_concurrency,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    overrides: dict[str, Any] = {
        "dtype": dtype,
        "cuda_graph_bs": [1, 2, 4, 8, 16],
        "cuda_graph_max_bs": 16,
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "enable_torch_compile": False,
        "max_prefill_tokens": 8192,
        "max_running_requests": 16,
        # Leave headroom for the two ~4.3 GB bf16 codec instances plus their
        # activations: on multi-GPU hosts the codec lives on the second GPU
        # (0.6 of an 80 GB card still gives the 4B backbone a ~35 GB KV pool);
        # on a single GPU everything co-locates, so back off further.
        "mem_fraction_static": 0.6 if torch.cuda.device_count() > 1 else 0.5,
        "sampling_backend": "pytorch",
        "torch_compile_max_bs": 16,
        "trust_remote_code": True,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=8192,
        **overrides,
    )

    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if want_cuda_graph:
        server_args.disable_cuda_graph = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        model_arch_override="MossTTSLocalSGLangModel",
    )

    if want_cuda_graph:
        server_args.disable_cuda_graph = False

    model = model_worker.model_runner.model
    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()
        # Also graph the per-frame local-transformer decode (1 + n_vq
        # micro-steps and 13 seeded sampling passes per frame): eager it is
        # kernel-launch-bound at ~22 ms/frame independent of batch size.
        model.init_frame_decode_graphs(
            list(overrides.get("cuda_graph_bs") or [1, 2, 4, 8, 16])
        )

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    request_builder, result_adapter = make_moss_tts_local_scheduler_adapters(
        model=model
    )

    def abort_request(request_id: str) -> None:
        # Drop any prepared handoff and release any held pool row; both are
        # idempotent no-ops if the request never reached them.
        cleanup_prepared_moss_tts_local_request(request_id)
        model.reset_request(request_id)

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=MossTTSLocalModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        abort_callback=abort_request,
    )


def create_tts_engine_executor(*args, **kwargs) -> Any:
    return create_sglang_tts_engine_executor(*args, **kwargs)


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    device = _resolve_codec_device(device, gpu_id)
    processor = _load_moss_tts_local_processor(model_path, device=device)

    def _prepare_codes(
        payload: StagePayload,
    ) -> tuple[MossTTSLocalState, torch.Tensor | None]:
        state = load_state(payload)
        if state.audio_codes is None:
            raise RuntimeError("MOSS-TTS Local vocoder requires audio_codes")
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        if codes.numel() == 0:
            # Immediate stop decision: emit no audio so only this request
            # fails downstream instead of poisoning the whole decode batch.
            return state, None
        return state, codes

    def _store_vocoder_result(
        payload: StagePayload,
        state: MossTTSLocalState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        # The v2 codec is natively stereo: keep the [channels, samples]
        # layout end to end so the client receives a 2-channel waveform.
        audio_payload = audio_waveform_payload(
            wav, source_hint="MOSS-TTS Local", keep_channels=True
        )
        state.audio_codes = None
        state.sample_rate = int(sample_rate)
        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _sample_rate() -> int:
        return int(
            getattr(getattr(processor, "model_config", None), "sampling_rate", 0)
            or getattr(
                getattr(getattr(processor, "audio_tokenizer", None), "config", None),
                "sampling_rate",
                0,
            )
            or 48000
        )

    def _vocode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        prepared = [_prepare_codes(payload) for payload in payloads]
        codes_list = [codes for _, codes in prepared if codes is not None]
        decoded = iter(processor.decode_audio_codes(codes_list))
        sample_rate = _sample_rate()
        results = []
        for payload, (state, codes) in zip(payloads, prepared):
            if codes is None:
                # No audio fields: the client surfaces a per-request
                # "no audio output" error without failing batch peers.
                state.audio_codes = None
                results.append(store_state(payload, state))
                continue
            wav = torch.as_tensor(next(decoded)).detach().to("cpu")
            results.append(_store_vocoder_result(payload, state, wav, sample_rate))
        return results

    def _vocode(payload: StagePayload) -> StagePayload:
        return _vocode_batch([payload])[0]

    return SimpleScheduler(
        _vocode,
        batch_compute_fn=_vocode_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
