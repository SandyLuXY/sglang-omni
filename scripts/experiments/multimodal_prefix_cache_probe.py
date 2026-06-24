#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import heapq
import json
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sglang_omni.models.ming_omni.bootstrap import make_thinker_scheduler_adapters
from sglang_omni.models.ming_omni.io import MingOmniPipelineState
from sglang_omni.models.ming_omni.pipeline.next_stage import (
    AUDIO_STAGE as MING_AUDIO_STAGE,
    IMAGE_STAGE as MING_IMAGE_STAGE,
)
from sglang_omni.models.ming_omni import stages as ming_stages
from sglang_omni.models.qwen3_omni import request_builders as qwen_rb
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.stages import (
    _batch_audio_encoder_payloads,
    _batch_image_encoder_payloads,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.stage_cache import StageOutputCache


class _FakeTokenizer:
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) % 251 + 3 for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(32 + int(token_id) % 95) for token_id in token_ids)


@dataclass(frozen=True)
class PrefixCase:
    name: str
    prompt_len: int
    raw_common_prefix_tokens: int
    keyed_common_prefix_tokens: int
    keyed_reuse_ratio: float
    unsafe_raw_reuse_tokens_avoided: int


@dataclass(frozen=True)
class EncoderCacheResult:
    model: str
    stage: str
    batch_size: int
    duplicate_requests: int
    unique_cache_keys: int
    cold_model_calls_median: float
    warm_model_calls_median: float
    cold_processed_units_median: float
    warm_processed_units_median: float
    cold_ms_median: float
    warm_ms_median: float
    warm_speedup: float
    same_batch_duplicate_unit_reduction: float
    notes: str


@dataclass(frozen=True)
class SchedulingResult:
    strategy: str
    requests: int
    workers: int
    deadline_miss_rate: float
    audio_miss_rate: float
    video_miss_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float


class _FakeImageEncoder:
    spatial_merge_size = 2

    def __init__(self, delay_s: float = 0.003) -> None:
        self.calls = 0
        self.processed_tokens = 0
        self.delay_s = delay_s

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        merge = self.spatial_merge_size**2
        out: dict[str, Any] = {}
        image_grid = inputs.get("image_grid_thw")
        if isinstance(image_grid, torch.Tensor):
            image_counts = (image_grid.prod(dim=-1) // merge).to(dtype=torch.long)
            image_tokens = int(image_counts.sum().item())
            self.processed_tokens += image_tokens
            out.update(
                {
                    "image_grid_thw": image_grid,
                    "image_token_counts": image_counts,
                    "image_embeds": torch.ones((image_tokens, 4), dtype=torch.float32),
                    "deepstack_visual_embeds_image": [
                        torch.ones((image_tokens, 4), dtype=torch.float32)
                    ],
                }
            )
        video_grid = inputs.get("video_grid_thw")
        if isinstance(video_grid, torch.Tensor):
            video_counts = (video_grid.prod(dim=-1) // merge).to(dtype=torch.long)
            video_tokens = int(video_counts.sum().item())
            self.processed_tokens += video_tokens
            out.update(
                {
                    "video_grid_thw": video_grid,
                    "video_token_counts": video_counts,
                    "video_embeds": torch.ones((video_tokens, 4), dtype=torch.float32),
                    "deepstack_visual_embeds_video": [
                        torch.ones((video_tokens, 4), dtype=torch.float32)
                    ],
                }
            )
        return out


class _FakeAudioEncoder:
    def __init__(self, delay_s: float = 0.003) -> None:
        self.calls = 0
        self.processed_rows = 0
        self.delay_s = delay_s

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        features = inputs["input_features"]
        lengths = inputs["audio_feature_lengths"].to(dtype=torch.long).view(-1)
        self.processed_rows += int(features.shape[0])
        output_lengths = torch.clamp(lengths // 2, min=1)
        return {
            "audio_feature_lengths": lengths,
            "audio_output_lengths": output_lengths,
            "audio_embeds": torch.ones(
                (int(output_lengths.sum().item()), 4), dtype=torch.float32
            ),
        }


class _FakeMingEncoder:
    def __init__(self, output_key: str, delay_s: float = 0.003) -> None:
        self.output_key = output_key
        self.calls = 0
        self.delay_s = delay_s

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.output_key == "audio_embeds":
            lengths = inputs.get("audio_feature_lengths")
            if isinstance(lengths, torch.Tensor):
                rows = max(int(lengths.numel()), 1)
            else:
                rows = 1
            return {self.output_key: torch.ones((rows, 4), dtype=torch.float32)}
        return {self.output_key: torch.ones((4, 4), dtype=torch.float32)}


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    total = 0
    for a, b in zip(left, right):
        if a != b:
            return total
        total += 1
    return total


def _qwen_keyed_ids(
    prompt_ids: list[int],
    media_cache_keys: dict[str, str],
    *,
    image_token_id: int = 501,
    audio_token_id: int = 502,
    video_token_id: int = 503,
    vocab_size: int = 1000,
) -> list[int]:
    state = Qwen3OmniPipelineState(
        prompt={
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(prompt_ids), dtype=torch.long),
            "prompt_text": "",
        },
        thinker_inputs={
            "model_inputs": {
                "image_embeds": torch.zeros((prompt_ids.count(image_token_id), 4)),
                "audio_embeds": torch.zeros((prompt_ids.count(audio_token_id), 4)),
                "video_embeds": torch.zeros((prompt_ids.count(video_token_id), 4)),
            },
            "media_cache_keys": media_cache_keys,
        },
    )
    config = SimpleNamespace(
        image_token_id=image_token_id,
        audio_token_id=audio_token_id,
        video_token_id=video_token_id,
    )
    original_mrope = qwen_rb._compute_mrope_positions
    qwen_rb._compute_mrope_positions = lambda *args, **kwargs: None
    try:
        data = qwen_rb.build_sglang_thinker_request(
            state,
            params={"max_new_tokens": 1, "top_k": 1},
            tokenizer=_FakeTokenizer(),
            vocab_size=vocab_size,
            request_id="qwen-prefix-probe",
            thinker_config=config,
        )
    finally:
        qwen_rb._compute_mrope_positions = original_mrope
    return list(data.req.origin_input_ids)


def _ming_keyed_ids(
    prompt_ids: list[int],
    media_cache_keys: dict[str, str],
    *,
    image_token_id: int = 601,
    audio_token_id: int = 602,
    video_token_id: int = 603,
    vocab_size: int = 1000,
) -> list[int]:
    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        vocab_size=vocab_size,
        image_token_id=image_token_id,
        audio_token_id=audio_token_id,
        video_token_id=video_token_id,
    )
    state = MingOmniPipelineState(
        prompt={
            "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long),
            "prompt_text": "",
        },
        thinker_inputs={
            "model_inputs": {
                "image_embeds": torch.zeros((prompt_ids.count(image_token_id), 4)),
                "audio_embeds": torch.zeros((prompt_ids.count(audio_token_id), 4)),
                "video_embeds": torch.zeros((prompt_ids.count(video_token_id), 4)),
            },
            "media_cache_keys": media_cache_keys,
        },
    )
    payload = StagePayload(
        request_id="ming-prefix-probe",
        request=OmniRequest(inputs={}, params={"max_new_tokens": 1, "top_k": 1}),
        data=state.to_dict(),
    )
    data = request_builder(payload)
    return list(data.req.origin_input_ids)


def _prefix_cases(model: str) -> list[PrefixCase]:
    if model == "qwen3_omni":
        image_id, audio_id, video_id = 501, 502, 503
        keyed_ids = _qwen_keyed_ids
    elif model == "ming_omni":
        image_id, audio_id, video_id = 601, 602, 603
        keyed_ids = _ming_keyed_ids
    else:
        raise ValueError(model)

    base = [11, 12]
    middle = [13]
    tail_a = [21, 22, 23, 24]
    tail_b = [31, 32, 33, 34]
    prompt_a = base + [image_id] * 16 + middle + [audio_id] * 12 + tail_a
    same_media_tail_b = base + [image_id] * 16 + middle + [audio_id] * 12 + tail_b
    different_image = base + [image_id] * 16 + middle + [audio_id] * 12 + tail_a
    different_audio = base + [image_id] * 16 + middle + [audio_id] * 12 + tail_a

    keys_a = {"image": "image:asset-a", "audio": "audio:asset-a"}
    keys_same = {"image": "image:asset-a", "audio": "audio:asset-a"}
    keys_img_b = {"image": "image:asset-b", "audio": "audio:asset-a"}
    keys_audio_b = {"image": "image:asset-a", "audio": "audio:asset-b"}

    cases: list[tuple[str, list[int], list[int], dict[str, str], dict[str, str]]] = [
        ("same_media_different_question", prompt_a, same_media_tail_b, keys_a, keys_same),
        ("different_image_same_shape", prompt_a, different_image, keys_a, keys_img_b),
        ("different_audio_same_shape", prompt_a, different_audio, keys_a, keys_audio_b),
    ]

    if video_id:
        video_a = base + [video_id] * 20 + tail_a
        video_b = base + [video_id] * 20 + tail_a
        cases.append(
            (
                "different_video_same_shape",
                video_a,
                video_b,
                {"video": "video:asset-a|fps=1|max_frames=8"},
                {"video": "video:asset-a|fps=4|max_frames=8"},
            )
        )

    result = []
    for name, left, right, left_keys, right_keys in cases:
        raw_common = _common_prefix_len(left, right)
        keyed_left = keyed_ids(left, left_keys)
        keyed_right = keyed_ids(right, right_keys)
        keyed_common = _common_prefix_len(keyed_left, keyed_right)
        result.append(
            PrefixCase(
                name=name,
                prompt_len=len(left),
                raw_common_prefix_tokens=raw_common,
                keyed_common_prefix_tokens=keyed_common,
                keyed_reuse_ratio=round(keyed_common / max(1, len(left)), 4),
                unsafe_raw_reuse_tokens_avoided=max(0, raw_common - keyed_common),
            )
        )
    return result


def _qwen_image_payload(request_id: str, cache_key: str) -> StagePayload:
    state = Qwen3OmniPipelineState(
        encoder_inputs={
            "image_encoder": {
                "pixel_values": torch.ones((4, 3), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
                "cache_key": cache_key,
            }
        }
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}, params={}),
        data=state.to_dict(),
    )


def _qwen_audio_payload(request_id: str, cache_key: str) -> StagePayload:
    state = Qwen3OmniPipelineState(
        encoder_inputs={
            "audio_encoder": {
                "input_features": torch.ones((1, 8, 16), dtype=torch.float32),
                "audio_feature_lengths": torch.tensor([16], dtype=torch.long),
                "feature_attention_mask": torch.ones((1, 16), dtype=torch.bool),
                "cache_key": cache_key,
            }
        }
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}, params={}),
        data=state.to_dict(),
    )


def _median(values: list[float]) -> float:
    return round(float(statistics.median(values)), 4)


def _qwen_image_cache_probe(repeats: int) -> EncoderCacheResult:
    cold_ms: list[float] = []
    warm_ms: list[float] = []
    cold_calls: list[float] = []
    warm_calls: list[float] = []
    cold_units: list[float] = []
    warm_units: list[float] = []
    for _ in range(repeats):
        cache = StageOutputCache(max_size=64, max_bytes=64 * 1024 * 1024)
        model = _FakeImageEncoder()
        payloads = [
            _qwen_image_payload("img-a-1", "image-a"),
            _qwen_image_payload("img-a-2", "image-a"),
            _qwen_image_payload("img-b-1", "image-b"),
        ]
        t0 = time.perf_counter()
        _batch_image_encoder_payloads(payloads, model=model, cache=cache)
        cold_elapsed = (time.perf_counter() - t0) * 1000
        calls_after_cold = model.calls
        units_after_cold = model.processed_tokens

        t0 = time.perf_counter()
        _batch_image_encoder_payloads(payloads, model=model, cache=cache)
        warm_elapsed = (time.perf_counter() - t0) * 1000
        cold_ms.append(cold_elapsed)
        warm_ms.append(warm_elapsed)
        cold_calls.append(calls_after_cold)
        warm_calls.append(model.calls - calls_after_cold)
        cold_units.append(units_after_cold)
        warm_units.append(model.processed_tokens - units_after_cold)

    no_dedup_units = 12.0
    dedup_units = _median(cold_units)
    return EncoderCacheResult(
        model="qwen3_omni",
        stage="image_encoder",
        batch_size=3,
        duplicate_requests=1,
        unique_cache_keys=2,
        cold_model_calls_median=_median(cold_calls),
        warm_model_calls_median=_median(warm_calls),
        cold_processed_units_median=dedup_units,
        warm_processed_units_median=_median(warm_units),
        cold_ms_median=_median(cold_ms),
        warm_ms_median=_median(warm_ms),
        warm_speedup=round(_median(cold_ms) / max(_median(warm_ms), 1e-6), 2),
        same_batch_duplicate_unit_reduction=round(
            (no_dedup_units - dedup_units) / no_dedup_units, 4
        ),
        notes=(
            "Synthetic encoder sleeps 3 ms per batched forward; processed_units are "
            "visual tokens. The active Qwen3 image path deduplicates same-batch cache keys."
        ),
    )


def _qwen_audio_cache_probe(repeats: int) -> EncoderCacheResult:
    cold_ms: list[float] = []
    warm_ms: list[float] = []
    cold_calls: list[float] = []
    warm_calls: list[float] = []
    cold_units: list[float] = []
    warm_units: list[float] = []
    for _ in range(repeats):
        cache = StageOutputCache(max_size=64, max_bytes=64 * 1024 * 1024)
        model = _FakeAudioEncoder()
        payloads = [
            _qwen_audio_payload("aud-a-1", "audio-a"),
            _qwen_audio_payload("aud-a-2", "audio-a"),
            _qwen_audio_payload("aud-b-1", "audio-b"),
        ]
        t0 = time.perf_counter()
        _batch_audio_encoder_payloads(payloads, model=model, cache=cache)
        cold_elapsed = (time.perf_counter() - t0) * 1000
        calls_after_cold = model.calls
        units_after_cold = model.processed_rows

        t0 = time.perf_counter()
        _batch_audio_encoder_payloads(payloads, model=model, cache=cache)
        warm_elapsed = (time.perf_counter() - t0) * 1000
        cold_ms.append(cold_elapsed)
        warm_ms.append(warm_elapsed)
        cold_calls.append(calls_after_cold)
        warm_calls.append(model.calls - calls_after_cold)
        cold_units.append(units_after_cold)
        warm_units.append(model.processed_rows - units_after_cold)

    no_dedup_units = 3.0
    dedup_units = _median(cold_units)
    return EncoderCacheResult(
        model="qwen3_omni",
        stage="audio_encoder",
        batch_size=3,
        duplicate_requests=1,
        unique_cache_keys=2,
        cold_model_calls_median=_median(cold_calls),
        warm_model_calls_median=_median(warm_calls),
        cold_processed_units_median=dedup_units,
        warm_processed_units_median=_median(warm_units),
        cold_ms_median=_median(cold_ms),
        warm_ms_median=_median(warm_ms),
        warm_speedup=round(_median(cold_ms) / max(_median(warm_ms), 1e-6), 2),
        same_batch_duplicate_unit_reduction=round(
            (no_dedup_units - dedup_units) / no_dedup_units, 4
        ),
        notes=(
            "Synthetic encoder sleeps 3 ms per batched forward; processed_units are "
            "audio rows. Warm cache hits skip compute, and the active audio batching "
            "path now coalesces same-batch duplicate cache keys."
        ),
    )


def _ming_payload(stage_name: str, request_id: str, cache_key: str) -> StagePayload:
    if stage_name == MING_AUDIO_STAGE:
        encoder_inputs = {
            stage_name: {
                "cache_key": cache_key,
                "audio_placeholder_loc_lens": [(0, 1)],
                "input_features": torch.ones((1, 2, 4), dtype=torch.float32),
                "audio_feature_lengths": torch.tensor([4], dtype=torch.long),
            }
        }
    else:
        encoder_inputs = {
            stage_name: {
                "cache_key": cache_key,
                "pixel_values": torch.ones((4, 8), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
            }
        }
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}),
        data=MingOmniPipelineState(encoder_inputs=encoder_inputs).to_dict(),
    )


def _ming_encoder_cache_probe() -> dict[str, Any]:
    repeat_requests = 10
    stage_specs = [
        (
            MING_AUDIO_STAGE,
            _FakeMingEncoder("audio_embeds"),
            {"cache_key", "audio_placeholder_loc_lens"},
        ),
        (MING_IMAGE_STAGE, _FakeMingEncoder("image_embeds"), {"cache_key"}),
    ]
    stages: list[dict[str, Any]] = []
    for stage_name, model, metadata_keys in stage_specs:
        cache = ming_stages._create_encoder_cache()
        elapsed_ms: list[float] = []
        for idx in range(repeat_requests):
            t0 = time.perf_counter()
            ming_stages._run_cached_encoder_payload(
                _ming_payload(stage_name, f"ming-{stage_name}-{idx}", "same-media"),
                stage_name=stage_name,
                model=model,
                cache=cache,
                metadata_keys=metadata_keys,
            )
            elapsed_ms.append((time.perf_counter() - t0) * 1000)
        stages.append(
            {
                "stage": stage_name,
                "repeated_media_requests": repeat_requests,
                "model_calls": model.calls,
                "expected_without_cache_model_calls": repeat_requests,
                "avoidable_encoder_forwards_ratio": round(
                    (repeat_requests - model.calls) / repeat_requests, 4
                ),
                "first_request_ms": round(elapsed_ms[0], 4),
                "warm_request_ms_median": round(_median(elapsed_ms[1:]), 4),
            }
        )
    return {
        "model": "ming_omni",
        "active_encoder_cache": "wired in sglang_omni/models/ming_omni/stages.py",
        "media_key_to_radix_cache": "wired in sglang_omni/models/ming_omni/bootstrap.py",
        "stages": stages,
        "notes": (
            "Ming active stages pass cache_key through preprocessing/merge for radix "
            "safety and now use StageOutputCache in active audio/image encoder "
            "factories. This is a synthetic stage-helper probe, not a full-model run."
        ),
    }


def _job_service_ms(modality: str, cache_hit: bool) -> float:
    if modality == "audio":
        return 3.0 if cache_hit else 12.0
    if modality == "video":
        return 20.0 if cache_hit else 75.0
    return 18.0


def _simulate_scheduler(strategy: str, *, seed: int = 7) -> SchedulingResult:
    rng = random.Random(seed)
    n = 5000
    workers = 2
    arrival_rate_per_ms = 0.09
    modality_weights = [("audio", 0.45), ("text", 0.35), ("video", 0.20)]
    deadlines = {"audio": 80.0, "text": 300.0, "video": 800.0}
    cache_probs = {"audio": 0.40, "text": 0.0, "video": 0.65}
    jobs = []
    now = 0.0
    for idx in range(n):
        now += rng.expovariate(arrival_rate_per_ms)
        r = rng.random()
        cumulative = 0.0
        modality = "video"
        for candidate, weight in modality_weights:
            cumulative += weight
            if r <= cumulative:
                modality = candidate
                break
        cache_hit = rng.random() < cache_probs[modality]
        service = _job_service_ms(modality, cache_hit)
        jobs.append(
            {
                "idx": idx,
                "arrival": now,
                "modality": modality,
                "cache_hit": cache_hit,
                "service": service,
                "deadline": deadlines[modality],
            }
        )

    worker_heap = [0.0 for _ in range(workers)]
    heapq.heapify(worker_heap)
    pending: list[tuple[Any, int, dict[str, Any]]] = []
    cursor = 0
    records = []

    def priority(job: dict[str, Any]) -> Any:
        if strategy == "fifo":
            return (job["arrival"], job["idx"])
        if strategy == "edf":
            return (job["arrival"] + job["deadline"], job["idx"])
        if strategy == "hybrid_slo_cache":
            cache_bonus = 0.85 if job["cache_hit"] else 1.0
            return (job["arrival"] + job["deadline"] * cache_bonus, job["idx"])
        raise ValueError(strategy)

    while cursor < n or pending:
        free_at = heapq.heappop(worker_heap)
        while cursor < n and jobs[cursor]["arrival"] <= free_at:
            job = jobs[cursor]
            heapq.heappush(pending, (priority(job), job["idx"], job))
            cursor += 1
        if not pending:
            if cursor < n:
                free_at = max(free_at, jobs[cursor]["arrival"])
                while cursor < n and jobs[cursor]["arrival"] <= free_at:
                    job = jobs[cursor]
                    heapq.heappush(pending, (priority(job), job["idx"], job))
                    cursor += 1
            else:
                heapq.heappush(worker_heap, free_at)
                break
        _, _, job = heapq.heappop(pending)
        start = max(free_at, job["arrival"])
        finish = start + job["service"]
        heapq.heappush(worker_heap, finish)
        latency = finish - job["arrival"]
        records.append(
            {
                "modality": job["modality"],
                "latency": latency,
                "miss": latency > job["deadline"],
            }
        )

    latencies = sorted(record["latency"] for record in records)

    def percentile(p: float) -> float:
        pos = min(len(latencies) - 1, max(0, int(round((len(latencies) - 1) * p))))
        return round(latencies[pos], 4)

    def miss_rate(modality: str | None = None) -> float:
        subset = [
            record
            for record in records
            if modality is None or record["modality"] == modality
        ]
        return round(sum(1 for record in subset if record["miss"]) / len(subset), 4)

    return SchedulingResult(
        strategy=strategy,
        requests=len(records),
        workers=workers,
        deadline_miss_rate=miss_rate(),
        audio_miss_rate=miss_rate("audio"),
        video_miss_rate=miss_rate("video"),
        p50_latency_ms=percentile(0.50),
        p95_latency_ms=percentile(0.95),
        p99_latency_ms=percentile(0.99),
        mean_latency_ms=round(
            sum(record["latency"] for record in records) / len(records), 4
        ),
    )


def _write_markdown(output_dir: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Multimodal Prefix Cache Probe",
        "",
        f"- generated_at: {results['identity']['generated_at']}",
        f"- git: {results['identity']['git_commit']}",
        f"- dirty_state: {results['identity']['git_dirty_state']}",
        f"- output_json: {output_dir / 'results.json'}",
        "",
        "## Prefix Keying",
        "",
        "| model | case | prompt | raw common | keyed common | reuse | unsafe raw avoided |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, cases in results["prefix_cases"].items():
        for case in cases:
            lines.append(
                "| {model} | {name} | {prompt_len} | {raw} | {keyed} | {ratio:.2%} | {avoided} |".format(
                    model=model,
                    name=case["name"],
                    prompt_len=case["prompt_len"],
                    raw=case["raw_common_prefix_tokens"],
                    keyed=case["keyed_common_prefix_tokens"],
                    ratio=case["keyed_reuse_ratio"],
                    avoided=case["unsafe_raw_reuse_tokens_avoided"],
                )
            )
    lines.extend(
        [
            "",
            "## Encoder Cache Probe",
            "",
            "| model | stage | cold calls | warm calls | cold units | warm units | cold ms | warm ms | speedup | same-batch unit reduction |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in results["encoder_cache"]:
        lines.append(
            "| {model} | {stage} | {cold_calls} | {warm_calls} | {cold_units} | {warm_units} | {cold_ms} | {warm_ms} | {speedup}x | {reduction:.2%} |".format(
                model=item["model"],
                stage=item["stage"],
                cold_calls=item["cold_model_calls_median"],
                warm_calls=item["warm_model_calls_median"],
                cold_units=item["cold_processed_units_median"],
                warm_units=item["warm_processed_units_median"],
                cold_ms=item["cold_ms_median"],
                warm_ms=item["warm_ms_median"],
                speedup=item["warm_speedup"],
                reduction=item["same_batch_duplicate_unit_reduction"],
            )
        )
    lines.extend(
        [
            "",
            "## Ming Encoder Cache Probe",
            "",
            json.dumps(results["ming_encoder_cache"], indent=2, sort_keys=True),
            "",
            "## SLO Scheduling Simulation",
            "",
            "| strategy | miss | audio miss | video miss | p50 ms | p95 ms | p99 ms | mean ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in results["scheduling_simulation"]:
        lines.append(
            "| {strategy} | {miss:.2%} | {audio:.2%} | {video:.2%} | {p50} | {p95} | {p99} | {mean} |".format(
                strategy=item["strategy"],
                miss=item["deadline_miss_rate"],
                audio=item["audio_miss_rate"],
                video=item["video_miss_rate"],
                p50=item["p50_latency_ms"],
                p95=item["p95_latency_ms"],
                p99=item["p99_latency_ms"],
                mean=item["mean_latency_ms"],
            )
        )
    lines.extend(
        [
            "",
            "The scheduling section is a deterministic synthetic model, not a production benchmark.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def run_probe(output_dir: Path, repeats: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "identity": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
            "git_dirty_state": _run(["git", "status", "--short"]),
            "python": _run(["python", "--version"]),
            "nvidia_smi": _run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "probe_repeats": repeats,
            "valid_for_full_model_performance_claim": False,
        },
        "prefix_cases": {
            "qwen3_omni": [asdict(case) for case in _prefix_cases("qwen3_omni")],
            "ming_omni": [asdict(case) for case in _prefix_cases("ming_omni")],
        },
        "encoder_cache": [
            asdict(_qwen_image_cache_probe(repeats)),
            asdict(_qwen_audio_cache_probe(repeats)),
        ],
        "ming_encoder_cache": _ming_encoder_cache_probe(),
        "scheduling_simulation": [
            asdict(_simulate_scheduler("fifo")),
            asdict(_simulate_scheduler("edf")),
            asdict(_simulate_scheduler("hybrid_slo_cache")),
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    _write_markdown(output_dir, results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe multimodal prefix-cache mechanics without loading full models."
    )
    default_dir = (
        Path("logs")
        / f"multimodal_prefix_cache_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    results = run_probe(args.output_dir, repeats=args.repeats)
    print(json.dumps({"output_dir": str(args.output_dir), "summary": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
