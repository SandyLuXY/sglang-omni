#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import bisect
import heapq
import json
import random
import statistics
import subprocess
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_ARTIFACTS = [
    (
        "qwen3_omni",
        "image",
        "logs/benchmark_multimodal_prefix_cache_20260624_224653/client/qwen_repeated_media.json",
    ),
    (
        "qwen3_omni",
        "audio",
        "logs/benchmark_multimodal_prefix_cache_av_20260624_230319/client/qwen_audio_repeated_media.json",
    ),
    (
        "qwen3_omni",
        "video",
        "logs/benchmark_multimodal_prefix_cache_av_20260624_230319/client/qwen_video_repeated_media.json",
    ),
    (
        "ming_omni",
        "image",
        "logs/benchmark_multimodal_prefix_cache_20260624_224653/client/ming_repeated_media.json",
    ),
    (
        "ming_omni",
        "audio",
        "logs/benchmark_multimodal_prefix_cache_av_20260624_230319/client/ming_audio_repeated_media.json",
    ),
    (
        "ming_omni",
        "video",
        "logs/benchmark_multimodal_prefix_cache_av_20260624_230319/client/ming_video_repeated_media_after_patch.json",
    ),
]

DEFAULT_MODALITY_WEIGHTS = {"audio": 0.50, "image": 0.30, "video": 0.20}
DEFAULT_SLO_MS = {"audio": 600.0, "image": 1200.0, "video": 3500.0}
DEFAULT_CATALOG_SIZE = {"audio": 4096, "image": 2048, "video": 512}
DEFAULT_CACHE_CAPACITY = {"audio": 2048, "image": 512, "video": 128}


@dataclass(frozen=True)
class ArtifactSpec:
    model: str
    modality: str
    path: Path


@dataclass(frozen=True)
class MeasuredProfile:
    model: str
    modality: str
    artifact_path: str
    cold_first_delta_ms: float
    warm_first_delta_ms: float
    cold_latency_ms: float
    warm_latency_ms: float
    cold_to_warm_ttft_speedup: float
    cold_reference: str
    warm_reference: str
    different_media_first_delta_ms: float | None
    cold_requests: int
    warm_requests: int


@dataclass(frozen=True, slots=True)
class RequestTemplate:
    idx: int
    modality: str
    media_key: int
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class JobSpec:
    idx: int
    arrival_ms: float
    modality: str
    media_key: int
    cache_hit: bool
    service_ms: float
    deadline_ms: float


@dataclass(frozen=True)
class SimulationResult:
    model: str
    strategy: str
    policy: str
    cache_enabled: bool
    requests: int
    workers: int
    arrival_rate_per_s: float
    offered_load: float
    cache_hit_rate: float
    deadline_miss_rate: float
    audio_miss_rate: float
    image_miss_rate: float
    video_miss_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    p95_queue_ms: float
    mean_queue_ms: float
    mean_inflight_little_law: float


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _ok_requests(data: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    requests = [
        req
        for req in data.get("requests", [])
        if req.get("phase") == phase and req.get("ok")
    ]
    return sorted(requests, key=lambda req: int(req.get("index", 0)))


def _summary_mean(data: dict[str, Any], phase: str, metric: str) -> float:
    return float(data["summary"][phase][metric]["mean"])


def _summary_requests(data: dict[str, Any], phase: str) -> int:
    return int(data["summary"][phase]["successful"])


def parse_measured_profile(spec: ArtifactSpec) -> MeasuredProfile:
    data = _load_json(spec.path)
    cold_phase = "cold_concurrent_same_media"
    warm_phase = "warm_sequential_same_media"
    cold_requests = _ok_requests(data, cold_phase)
    if cold_requests:
        cold_req = cold_requests[0]
        cold_first_delta = float(cold_req["first_delta_seconds"]) * 1000.0
        cold_latency = float(cold_req["latency_seconds"]) * 1000.0
        cold_reference = f"{cold_phase}/index0"
    else:
        cold_first_delta = _summary_mean(data, cold_phase, "first_delta_seconds") * 1000.0
        cold_latency = _summary_mean(data, cold_phase, "latency_seconds") * 1000.0
        cold_reference = f"{cold_phase}/summary_mean"

    warm_first_delta = _summary_mean(data, warm_phase, "first_delta_seconds") * 1000.0
    warm_latency = _summary_mean(data, warm_phase, "latency_seconds") * 1000.0
    different_media = data.get("summary", {}).get("single_different_media")
    different_first_delta = None
    if different_media is not None:
        different_first_delta = (
            float(different_media["first_delta_seconds"]["mean"]) * 1000.0
        )

    return MeasuredProfile(
        model=spec.model,
        modality=spec.modality,
        artifact_path=str(spec.path),
        cold_first_delta_ms=round(cold_first_delta, 4),
        warm_first_delta_ms=round(warm_first_delta, 4),
        cold_latency_ms=round(cold_latency, 4),
        warm_latency_ms=round(warm_latency, 4),
        cold_to_warm_ttft_speedup=round(
            cold_first_delta / max(warm_first_delta, 1e-9), 4
        ),
        cold_reference=cold_reference,
        warm_reference=f"{warm_phase}/summary_mean",
        different_media_first_delta_ms=(
            round(different_first_delta, 4)
            if different_first_delta is not None
            else None
        ),
        cold_requests=_summary_requests(data, cold_phase),
        warm_requests=_summary_requests(data, warm_phase),
    )


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    total = sum(weights.values())
    point = rng.random() * total
    cumulative = 0.0
    last_key = next(iter(weights))
    for key, weight in weights.items():
        cumulative += weight
        last_key = key
        if point <= cumulative:
            return key
    return last_key


def _zipf_cdf(catalog_size: int, skew: float) -> list[float]:
    weights = [1.0 / (rank**skew) for rank in range(1, catalog_size + 1)]
    total = sum(weights)
    cdf: list[float] = []
    cumulative = 0.0
    for weight in weights:
        cumulative += weight / total
        cdf.append(cumulative)
    cdf[-1] = 1.0
    return cdf


def _sample_zipf_key(rng: random.Random, cdf: list[float]) -> int:
    return bisect.bisect_left(cdf, rng.random())


def _touch_lru(
    caches: dict[str, OrderedDict[int, None]],
    *,
    modality: str,
    key: int,
    capacity: int,
) -> bool:
    cache = caches[modality]
    if key in cache:
        cache.move_to_end(key)
        return True
    cache[key] = None
    cache.move_to_end(key)
    while len(cache) > capacity:
        cache.popitem(last=False)
    return False


def generate_templates(
    *,
    requests: int,
    seed: int,
    warmup_requests: int,
    modality_weights: dict[str, float],
    catalog_size: dict[str, int],
    cache_capacity: dict[str, int],
    zipf_skew: float,
) -> list[RequestTemplate]:
    rng = random.Random(seed)
    cdfs = {
        modality: _zipf_cdf(catalog_size[modality], zipf_skew)
        for modality in modality_weights
    }
    caches = {modality: OrderedDict() for modality in modality_weights}

    def next_request(idx: int) -> RequestTemplate:
        modality = _weighted_choice(rng, modality_weights)
        key = _sample_zipf_key(rng, cdfs[modality])
        hit = _touch_lru(
            caches,
            modality=modality,
            key=key,
            capacity=cache_capacity[modality],
        )
        return RequestTemplate(
            idx=idx,
            modality=modality,
            media_key=key,
            cache_hit=hit,
        )

    for idx in range(warmup_requests):
        next_request(-idx - 1)
    return [next_request(idx) for idx in range(requests)]


def _jobs_from_templates(
    templates: list[RequestTemplate],
    *,
    profiles: dict[str, MeasuredProfile],
    cache_enabled: bool,
    arrival_rate_per_s: float,
    seed: int,
    slo_ms: dict[str, float],
) -> list[JobSpec]:
    rng = random.Random(seed)
    arrival = 0.0
    jobs: list[JobSpec] = []
    rate_per_ms = arrival_rate_per_s / 1000.0
    for item in templates:
        arrival += rng.expovariate(rate_per_ms)
        profile = profiles[item.modality]
        cache_hit = item.cache_hit if cache_enabled else False
        service_ms = (
            profile.warm_first_delta_ms
            if cache_hit
            else profile.cold_first_delta_ms
        )
        jobs.append(
            JobSpec(
                idx=item.idx,
                arrival_ms=arrival,
                modality=item.modality,
                media_key=item.media_key,
                cache_hit=cache_hit,
                service_ms=service_ms,
                deadline_ms=slo_ms[item.modality],
            )
        )
    return jobs


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = min(
        len(values) - 1,
        max(0, int(round((len(values) - 1) * percentile))),
    )
    return round(values[pos], 4)


def simulate_jobs(
    jobs: list[JobSpec],
    *,
    model: str,
    strategy: str,
    policy: str,
    cache_enabled: bool,
    workers: int,
    arrival_rate_per_s: float,
) -> SimulationResult:
    worker_heap = [0.0 for _ in range(workers)]
    heapq.heapify(worker_heap)
    pending: list[tuple[Any, int, JobSpec]] = []
    cursor = 0
    latencies: list[float] = []
    queue_delays: list[float] = []
    misses_by_modality = {modality: 0 for modality in DEFAULT_MODALITY_WEIGHTS}
    total_by_modality = {modality: 0 for modality in DEFAULT_MODALITY_WEIGHTS}
    cache_hits = 0

    def priority(job: JobSpec) -> Any:
        if policy == "fifo":
            return (job.arrival_ms, job.idx)
        if policy == "edf":
            return (job.arrival_ms + job.deadline_ms, job.idx)
        if policy == "least_slack":
            return (
                job.arrival_ms + job.deadline_ms - job.service_ms,
                job.service_ms,
                job.idx,
            )
        if policy == "cache_weighted_edf":
            cache_bonus = min(job.deadline_ms * 0.35, job.service_ms)
            if not job.cache_hit:
                cache_bonus = 0.0
            return (
                job.arrival_ms + job.deadline_ms - cache_bonus,
                job.service_ms,
                job.idx,
            )
        raise ValueError(f"unsupported policy: {policy}")

    while cursor < len(jobs) or pending:
        free_at = heapq.heappop(worker_heap)
        while cursor < len(jobs) and jobs[cursor].arrival_ms <= free_at:
            job = jobs[cursor]
            heapq.heappush(pending, (priority(job), job.idx, job))
            cursor += 1
        if not pending:
            if cursor < len(jobs):
                free_at = max(free_at, jobs[cursor].arrival_ms)
                while cursor < len(jobs) and jobs[cursor].arrival_ms <= free_at:
                    job = jobs[cursor]
                    heapq.heappush(pending, (priority(job), job.idx, job))
                    cursor += 1
            else:
                heapq.heappush(worker_heap, free_at)
                break
        _, _, job = heapq.heappop(pending)
        start = max(free_at, job.arrival_ms)
        finish = start + job.service_ms
        heapq.heappush(worker_heap, finish)
        latency = finish - job.arrival_ms
        queue_delay = start - job.arrival_ms
        latencies.append(latency)
        queue_delays.append(queue_delay)
        total_by_modality[job.modality] += 1
        if job.cache_hit:
            cache_hits += 1
        if latency > job.deadline_ms:
            misses_by_modality[job.modality] += 1

    total_requests = len(latencies)
    mean_service = statistics.fmean(job.service_ms for job in jobs)
    offered_load = arrival_rate_per_s * (mean_service / 1000.0) / workers
    mean_latency_ms = statistics.fmean(latencies)

    def modality_miss(modality: str) -> float:
        total = total_by_modality.get(modality, 0)
        if total == 0:
            return 0.0
        return round(misses_by_modality[modality] / total, 6)

    return SimulationResult(
        model=model,
        strategy=strategy,
        policy=policy,
        cache_enabled=cache_enabled,
        requests=total_requests,
        workers=workers,
        arrival_rate_per_s=round(arrival_rate_per_s, 6),
        offered_load=round(offered_load, 6),
        cache_hit_rate=round(cache_hits / total_requests, 6),
        deadline_miss_rate=round(sum(misses_by_modality.values()) / total_requests, 6),
        audio_miss_rate=modality_miss("audio"),
        image_miss_rate=modality_miss("image"),
        video_miss_rate=modality_miss("video"),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        p99_latency_ms=_percentile(latencies, 0.99),
        mean_latency_ms=round(mean_latency_ms, 4),
        p95_queue_ms=_percentile(queue_delays, 0.95),
        mean_queue_ms=round(statistics.fmean(queue_delays), 4),
        mean_inflight_little_law=round(arrival_rate_per_s * mean_latency_ms / 1000.0, 4),
    )


def _mean_cached_service_ms(
    templates: list[RequestTemplate],
    profiles: dict[str, MeasuredProfile],
) -> float:
    values = []
    for item in templates:
        profile = profiles[item.modality]
        values.append(
            profile.warm_first_delta_ms
            if item.cache_hit
            else profile.cold_first_delta_ms
        )
    return statistics.fmean(values)


def run_simulation(
    *,
    profiles_by_model: dict[str, dict[str, MeasuredProfile]],
    requests: int,
    workers: int,
    target_utilization: float,
    seed: int,
    warmup_requests: int,
    modality_weights: dict[str, float],
    slo_ms: dict[str, float],
    catalog_size: dict[str, int],
    cache_capacity: dict[str, int],
    zipf_skew: float,
) -> list[SimulationResult]:
    results: list[SimulationResult] = []
    for model_idx, (model, profiles) in enumerate(sorted(profiles_by_model.items())):
        model_seed = seed + model_idx * 1009
        templates = generate_templates(
            requests=requests,
            seed=model_seed,
            warmup_requests=warmup_requests,
            modality_weights=modality_weights,
            catalog_size=catalog_size,
            cache_capacity=cache_capacity,
            zipf_skew=zipf_skew,
        )
        mean_cached_service = _mean_cached_service_ms(templates, profiles)
        arrival_rate_per_s = workers * target_utilization / (
            mean_cached_service / 1000.0
        )
        scenario_specs = [
            ("fifo_no_cache", "fifo", False),
            ("fifo_cache", "fifo", True),
            ("edf_cache", "edf", True),
            ("cache_weighted_edf", "cache_weighted_edf", True),
            ("least_slack_cache", "least_slack", True),
        ]
        for offset, (strategy, policy, cache_enabled) in enumerate(scenario_specs):
            jobs = _jobs_from_templates(
                templates,
                profiles=profiles,
                cache_enabled=cache_enabled,
                arrival_rate_per_s=arrival_rate_per_s,
                seed=model_seed + 17 + offset,
                slo_ms=slo_ms,
            )
            results.append(
                simulate_jobs(
                    jobs,
                    model=model,
                    strategy=strategy,
                    policy=policy,
                    cache_enabled=cache_enabled,
                    workers=workers,
                    arrival_rate_per_s=arrival_rate_per_s,
                )
            )
    return results


def _parse_key_value_map(value: str, defaults: dict[str, float]) -> dict[str, float]:
    result = dict(defaults)
    if not value:
        return result
    for part in value.split(","):
        key, raw = part.split("=", 1)
        result[key.strip()] = float(raw)
    return result


def _parse_int_key_value_map(value: str, defaults: dict[str, int]) -> dict[str, int]:
    return {key: int(raw) for key, raw in _parse_key_value_map(value, defaults).items()}


def _parse_artifact_specs(items: list[str]) -> list[ArtifactSpec]:
    specs: list[ArtifactSpec] = []
    for item in items:
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "--artifact values must use model:modality:path, got "
                f"{item!r}"
            )
        model, modality, raw_path = parts
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        specs.append(ArtifactSpec(model=model, modality=modality, path=path))
    return specs


def _write_markdown(output_path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Measured Multimodal SLO Scheduler Probe",
        "",
        f"- generated_at: {results['identity']['generated_at']}",
        f"- git: {results['identity']['git_commit']}",
        f"- dirty_state: `{results['identity']['git_dirty_state'] or 'clean'}`",
        f"- requests_per_model: {results['config']['requests']}",
        f"- workers_per_model: {results['config']['workers']}",
        f"- target_cache_enabled_utilization: {results['config']['target_utilization']}",
        f"- warmup_requests: {results['config']['warmup_requests']}",
        f"- zipf_skew: {results['config']['zipf_skew']}",
        "",
        "## Measured TTFT Inputs",
        "",
        "| model | modality | cold TTFT ms | warm TTFT ms | speedup | cold ref | warm ref | artifact |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for profile in results["measured_profiles"]:
        lines.append(
            "| {model} | {modality} | {cold:.4f} | {warm:.4f} | {speedup:.2f}x | {cold_ref} | {warm_ref} | {artifact} |".format(
                model=profile["model"],
                modality=profile["modality"],
                cold=profile["cold_first_delta_ms"],
                warm=profile["warm_first_delta_ms"],
                speedup=profile["cold_to_warm_ttft_speedup"],
                cold_ref=profile["cold_reference"],
                warm_ref=profile["warm_reference"],
                artifact=profile["artifact_path"],
            )
        )

    lines.extend(
        [
            "",
            "## Simulation Results",
            "",
            "| model | strategy | offered load | hit rate | miss | audio miss | image miss | video miss | p95 latency ms | p99 latency ms | p95 queue ms | mean inflight |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in results["simulation_results"]:
        lines.append(
            "| {model} | {strategy} | {load:.3f} | {hit:.2%} | {miss:.2%} | {audio:.2%} | {image:.2%} | {video:.2%} | {p95:.4f} | {p99:.4f} | {q95:.4f} | {inflight:.4f} |".format(
                model=item["model"],
                strategy=item["strategy"],
                load=item["offered_load"],
                hit=item["cache_hit_rate"],
                miss=item["deadline_miss_rate"],
                audio=item["audio_miss_rate"],
                image=item["image_miss_rate"],
                video=item["video_miss_rate"],
                p95=item["p95_latency_ms"],
                p99=item["p99_latency_ms"],
                q95=item["p95_queue_ms"],
                inflight=item["mean_inflight_little_law"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a deterministic queueing simulation using measured streaming first-delta latency from the committed Qwen3-Omni and Ming-Omni repeated-media serving artifacts.",
            "- It does not model standalone text traffic, distributed cache coherence, memory bytes, in-flight duplicate coalescing, or real Stage runtime backpressure.",
            "- `fifo_no_cache` uses the same arrival rate as the cache-enabled scenarios, so offered load above 1.0 means the measured cold path is not stable for that workload.",
            "- `cache_weighted_edf` is the closest proxy for a unified SLO-aware policy: it uses modality SLO, estimated service time, and cache-hit state while preserving the existing Stage scheduler boundary.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def run_probe(
    *,
    output_dir: Path,
    artifact_specs: list[ArtifactSpec],
    requests: int,
    workers: int,
    target_utilization: float,
    seed: int,
    warmup_requests: int,
    modality_weights: dict[str, float],
    slo_ms: dict[str, float],
    catalog_size: dict[str, int],
    cache_capacity: dict[str, int],
    zipf_skew: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = [parse_measured_profile(spec) for spec in artifact_specs]
    profiles_by_model: dict[str, dict[str, MeasuredProfile]] = {}
    for profile in profiles:
        profiles_by_model.setdefault(profile.model, {})[profile.modality] = profile
    for model, items in profiles_by_model.items():
        missing = set(modality_weights) - set(items)
        if missing:
            raise ValueError(f"{model} missing profiles for {sorted(missing)}")

    simulation = run_simulation(
        profiles_by_model=profiles_by_model,
        requests=requests,
        workers=workers,
        target_utilization=target_utilization,
        seed=seed,
        warmup_requests=warmup_requests,
        modality_weights=modality_weights,
        slo_ms=slo_ms,
        catalog_size=catalog_size,
        cache_capacity=cache_capacity,
        zipf_skew=zipf_skew,
    )
    results = {
        "identity": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
            "git_dirty_state": _run(["git", "status", "--short"]),
            "python": _run([sys.executable, "--version"]),
        },
        "config": {
            "requests": requests,
            "workers": workers,
            "target_utilization": target_utilization,
            "seed": seed,
            "warmup_requests": warmup_requests,
            "modality_weights": modality_weights,
            "slo_ms": slo_ms,
            "catalog_size": catalog_size,
            "cache_capacity": cache_capacity,
            "zipf_skew": zipf_skew,
            "valid_for_capacity_claim": False,
        },
        "measured_profiles": [asdict(profile) for profile in profiles],
        "simulation_results": [asdict(item) for item in simulation],
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    _write_markdown(output_dir / "summary.md", results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate mixed-modality SLO scheduling with measured Qwen3/Ming "
            "repeated-media TTFT inputs."
        )
    )
    default_output = (
        REPO_ROOT
        / "logs"
        / f"multimodal_measured_slo_scheduler_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--artifact",
        action="append",
        default=None,
        help="Measured artifact as model:modality:path. Repeat to replace defaults.",
    )
    parser.add_argument("--requests", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--target-utilization", type=float, default=0.82)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--warmup-requests", type=int, default=10_000)
    parser.add_argument("--zipf-skew", type=float, default=1.05)
    parser.add_argument(
        "--modality-weights",
        default="audio=0.50,image=0.30,video=0.20",
        help="Comma-separated modality weights, e.g. audio=0.5,image=0.3,video=0.2.",
    )
    parser.add_argument(
        "--slo-ms",
        default="audio=600,image=1200,video=3500",
        help="Comma-separated first-delta SLOs in ms.",
    )
    parser.add_argument(
        "--catalog-size",
        default="audio=4096,image=2048,video=512",
        help="Comma-separated synthetic media catalog sizes.",
    )
    parser.add_argument(
        "--cache-capacity",
        default="audio=2048,image=512,video=128",
        help="Comma-separated LRU cache capacities by entries.",
    )
    args = parser.parse_args()

    artifact_values = args.artifact
    if artifact_values is None:
        artifact_values = [
            f"{model}:{modality}:{path}" for model, modality, path in DEFAULT_ARTIFACTS
        ]
    artifact_specs = _parse_artifact_specs(artifact_values)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    results = run_probe(
        output_dir=output_dir,
        artifact_specs=artifact_specs,
        requests=args.requests,
        workers=args.workers,
        target_utilization=args.target_utilization,
        seed=args.seed,
        warmup_requests=args.warmup_requests,
        modality_weights=_parse_key_value_map(
            args.modality_weights,
            DEFAULT_MODALITY_WEIGHTS,
        ),
        slo_ms=_parse_key_value_map(args.slo_ms, DEFAULT_SLO_MS),
        catalog_size=_parse_int_key_value_map(
            args.catalog_size,
            DEFAULT_CATALOG_SIZE,
        ),
        cache_capacity=_parse_int_key_value_map(
            args.cache_capacity,
            DEFAULT_CACHE_CAPACITY,
        ),
        zipf_skew=args.zipf_skew,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": str(output_dir / "summary.md"),
                "results": str(output_dir / "results.json"),
                "simulation_results": results["simulation_results"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
