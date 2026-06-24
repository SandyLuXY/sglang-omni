# Measured Multimodal SLO Scheduler Probe

- generated_at: 2026-06-24T23:28:23.771725+00:00
- git: 4b61fcf1
- dirty_state: `?? logs/benchmark_multimodal_slo_scheduler_20260624_232102/
?? scripts/experiments/multimodal_measured_slo_scheduler_probe.py
?? tests/unit_test/experiments/`
- requests_per_model: 100000
- workers_per_model: 2
- target_cache_enabled_utilization: 0.82
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
| ming_omni | fifo_no_cache | 2.450 | 0.00% | 100.00% | 100.00% | 100.00% | 100.00% | 99640801.2440 | 103968027.4906 | 99635752.3012 | 72216.8418 |
| ming_omni | fifo_cache | 0.820 | 84.01% | 69.81% | 70.95% | 70.34% | 66.18% | 13403.3408 | 19903.2540 | 11832.8650 | 6.2516 |
| ming_omni | edf_cache | 0.820 | 84.01% | 66.95% | 65.66% | 67.80% | 68.91% | 13127.3540 | 19644.6871 | 11384.6920 | 5.7251 |
| ming_omni | cache_weighted_edf | 0.820 | 84.01% | 67.50% | 66.77% | 68.12% | 68.38% | 12842.5853 | 20185.5652 | 10889.5594 | 5.7860 |
| ming_omni | least_slack_cache | 0.820 | 84.01% | 69.20% | 68.96% | 69.96% | 68.69% | 14491.1628 | 21085.8470 | 13278.7269 | 6.7976 |
| qwen3_omni | fifo_no_cache | 1.664 | 0.00% | 100.00% | 100.00% | 99.99% | 99.99% | 26405023.9501 | 27467910.4494 | 26404043.0286 | 33138.7864 |
| qwen3_omni | fifo_cache | 0.820 | 83.99% | 58.82% | 66.09% | 52.56% | 49.88% | 7486.5904 | 11403.3829 | 6572.2627 | 6.0912 |
| qwen3_omni | edf_cache | 0.820 | 83.99% | 51.86% | 56.25% | 43.33% | 53.73% | 6980.5547 | 10644.5948 | 5541.0796 | 4.9445 |
| qwen3_omni | cache_weighted_edf | 0.820 | 83.99% | 52.10% | 56.59% | 44.69% | 51.96% | 7009.6377 | 10959.6756 | 5602.9622 | 5.0406 |
| qwen3_omni | least_slack_cache | 0.820 | 83.99% | 58.44% | 63.14% | 53.86% | 53.50% | 7438.5516 | 11070.4914 | 6505.7886 | 5.9541 |

## Interpretation

- This is a deterministic queueing simulation using measured streaming first-delta latency from the committed Qwen3-Omni and Ming-Omni repeated-media serving artifacts.
- It does not model standalone text traffic, distributed cache coherence, memory bytes, in-flight duplicate coalescing, or real Stage runtime backpressure.
- `fifo_no_cache` uses the same arrival rate as the cache-enabled scenarios, so offered load above 1.0 means the measured cold path is not stable for that workload.
- `cache_weighted_edf` is the closest proxy for a unified SLO-aware policy: it uses modality SLO, estimated service time, and cache-hit state while preserving the existing Stage scheduler boundary.
