# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_probe_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "experiments"
        / "multimodal_measured_slo_scheduler_probe.py"
    )
    spec = importlib.util.spec_from_file_location("measured_slo_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_measured_profile_uses_first_cold_request_and_warm_summary(tmp_path):
    module = _load_probe_module()
    artifact = tmp_path / "probe.json"
    artifact.write_text(
        json.dumps(
            {
                "summary": {
                    "cold_concurrent_same_media": {
                        "successful": 2,
                        "first_delta_seconds": {"mean": 9.0},
                        "latency_seconds": {"mean": 10.0},
                    },
                    "warm_sequential_same_media": {
                        "successful": 2,
                        "first_delta_seconds": {"mean": 0.25},
                        "latency_seconds": {"mean": 0.30},
                    },
                    "single_different_media": {
                        "first_delta_seconds": {"mean": 0.70},
                    },
                },
                "requests": [
                    {
                        "phase": "cold_concurrent_same_media",
                        "index": 1,
                        "ok": True,
                        "first_delta_seconds": 2.0,
                        "latency_seconds": 2.5,
                    },
                    {
                        "phase": "cold_concurrent_same_media",
                        "index": 0,
                        "ok": True,
                        "first_delta_seconds": 1.0,
                        "latency_seconds": 1.5,
                    },
                ],
            }
        )
    )

    profile = module.parse_measured_profile(
        module.ArtifactSpec("toy", "audio", artifact)
    )

    assert profile.cold_first_delta_ms == 1000.0
    assert profile.cold_latency_ms == 1500.0
    assert profile.warm_first_delta_ms == 250.0
    assert profile.different_media_first_delta_ms == 700.0
    assert profile.cold_reference == "cold_concurrent_same_media/index0"


def test_least_slack_prioritizes_audio_deadline_when_jobs_are_pending():
    module = _load_probe_module()
    jobs = [
        module.JobSpec(
            idx=0,
            arrival_ms=0.0,
            modality="video",
            media_key=0,
            cache_hit=False,
            service_ms=1000.0,
            deadline_ms=5000.0,
        ),
        module.JobSpec(
            idx=1,
            arrival_ms=0.0,
            modality="audio",
            media_key=0,
            cache_hit=True,
            service_ms=50.0,
            deadline_ms=100.0,
        ),
    ]

    fifo = module.simulate_jobs(
        jobs,
        model="toy",
        strategy="fifo",
        policy="fifo",
        cache_enabled=True,
        workers=1,
        arrival_rate_per_s=1.0,
    )
    least_slack = module.simulate_jobs(
        jobs,
        model="toy",
        strategy="least_slack",
        policy="least_slack",
        cache_enabled=True,
        workers=1,
        arrival_rate_per_s=1.0,
    )
    cache_weighted = module.simulate_jobs(
        jobs,
        model="toy",
        strategy="cache_weighted_edf",
        policy="cache_weighted_edf",
        cache_enabled=True,
        workers=1,
        arrival_rate_per_s=1.0,
    )

    assert fifo.audio_miss_rate == 1.0
    assert least_slack.audio_miss_rate == 0.0
    assert cache_weighted.audio_miss_rate == 0.0
    assert least_slack.mean_latency_ms < fifo.mean_latency_ms
    assert least_slack.p95_queue_ms < fifo.p95_queue_ms
