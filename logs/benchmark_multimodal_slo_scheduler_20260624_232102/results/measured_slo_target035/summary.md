# Measured Multimodal SLO Scheduler Probe

- generated_at: 2026-06-24T23:28:08.976805+00:00
- git: 4b61fcf1
- dirty_state: `?? logs/benchmark_multimodal_slo_scheduler_20260624_232102/
?? scripts/experiments/multimodal_measured_slo_scheduler_probe.py
?? tests/unit_test/experiments/`
- requests_per_model: 100000
- workers_per_model: 2
- target_cache_enabled_utilization: 0.35
- warmup_requests: 10000
- zipf_skew: 1.05

## Measured TTFT Inputs

| model | modality | cold TTFT ms | warm TTFT ms | speedup | cold ref | warm ref | artifact |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| qwen3_omni | image | 444.6335 | 180.3376 | 2.47x | cold_concurrent_same_media/index0 | warm_sequential_same_media/summary_mean | /data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/logs/benchmark_multimodal_prefix_cache_20260624_224653/client/qwen_repeated_media.json |
| qwen3_omni | audio | 1287.0211 | 177.4352 | 7.25x | cold_concurrent_same_media/index0 | warm_sequential_same_media/summary_mean | /data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/logs/benchmark_multimodal_prefix_cache_av_20260624_230319/client/qwen_audio_repeated_media.json |
| qwen3_omni | video | 3135.5938 | 2168.6994 | 1.45x | cold_concurrent_same_media/index0 | warm_sequential_same_media/summary_mean | /data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/logs/benchmark_multimodal_prefix_cache_av_20260624_230319/client/qwen_video_repeated_media.json |
| ming_omni | image | 5237.5639 | 496.0815 | 10.56x | cold_concurrent_same_media/index0 | warm_sequential_same_media/summary_mean | /data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/logs/benchmark_multimodal_prefix_cache_20260624_224653/client/ming_repeated_media.json |
| ming_omni | audio | 1862.3025 | 199.5327 | 9.33x | cold_concurrent_same_media/index0 | warm_sequential_same_media/summary_mean | /data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/logs/benchmark_multimodal_prefix_cache_av_20260624_230319/client/ming_audio_repeated_media.json |
| ming_omni | video | 5271.7307 | 2040.5090 | 2.58x | cold_concurrent_same_media/index0 | warm_sequential_same_media/summary_mean | /data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/logs/benchmark_multimodal_prefix_cache_av_20260624_230319/client/ming_video_repeated_media_after_patch.json |

## Simulation Results

| model | strategy | offered load | hit rate | miss | audio miss | image miss | video miss | p95 latency ms | p99 latency ms | p95 queue ms | mean inflight |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ming_omni | fifo_no_cache | 1.046 | 0.00% | 100.00% | 100.00% | 100.00% | 100.00% | 7206467.2952 | 7625336.3560 | 7203359.0918 | 2172.8300 |
| ming_omni | fifo_cache | 0.350 | 84.01% | 25.23% | 21.30% | 28.54% | 30.14% | 5271.7307 | 6515.0781 | 1746.8141 | 0.8409 |
| ming_omni | edf_cache | 0.350 | 84.01% | 24.65% | 20.32% | 27.81% | 30.78% | 5271.7307 | 6541.9811 | 1560.9703 | 0.8249 |
| ming_omni | cache_weighted_edf | 0.350 | 84.01% | 24.52% | 20.12% | 27.85% | 30.60% | 5271.7307 | 6437.1852 | 1571.8367 | 0.8237 |
| ming_omni | least_slack_cache | 0.350 | 84.01% | 24.89% | 20.64% | 28.27% | 30.48% | 5271.7307 | 6611.6013 | 1731.2546 | 0.8485 |
| qwen3_omni | fifo_no_cache | 0.710 | 0.00% | 71.73% | 100.00% | 39.76% | 48.68% | 5871.7065 | 8409.2940 | 4171.5765 | 2.4635 |
| qwen3_omni | fifo_cache | 0.350 | 83.99% | 11.72% | 18.04% | 5.66% | 4.90% | 3135.5938 | 3565.6018 | 1034.8995 | 0.8314 |
| qwen3_omni | edf_cache | 0.350 | 83.99% | 11.18% | 17.04% | 4.94% | 5.79% | 3135.5938 | 3635.9021 | 930.5374 | 0.8174 |
| qwen3_omni | cache_weighted_edf | 0.350 | 83.99% | 11.02% | 16.91% | 4.70% | 5.70% | 3135.5938 | 3612.3332 | 915.0651 | 0.8162 |
| qwen3_omni | least_slack_cache | 0.350 | 83.99% | 11.85% | 17.66% | 6.22% | 5.65% | 3135.5938 | 3638.3204 | 1050.4736 | 0.8316 |

## Interpretation

- This is a deterministic queueing simulation using measured streaming first-delta latency from the committed Qwen3-Omni and Ming-Omni repeated-media serving artifacts.
- It does not model standalone text traffic, distributed cache coherence, memory bytes, in-flight duplicate coalescing, or real Stage runtime backpressure.
- `fifo_no_cache` uses the same arrival rate as the cache-enabled scenarios, so offered load above 1.0 means the measured cold path is not stable for that workload.
- `cache_weighted_edf` is the closest proxy for a unified SLO-aware policy: it uses modality SLO, estimated service time, and cache-hit state while preserving the existing Stage scheduler boundary.
