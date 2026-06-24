# Measured Multimodal SLO Scheduler Probe

- generated_at: 2026-06-24T23:28:56.276656+00:00
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
| ming_omni | fifo_no_cache | 1.636 | 0.00% | 100.00% | 100.00% | 100.00% | 100.00% | 65339376.5792 | 68218214.0453 | 65335590.3060 | 31569.6964 |
| ming_omni | fifo_cache | 0.350 | 96.20% | 10.85% | 14.13% | 10.42% | 3.32% | 2258.5883 | 5237.5639 | 987.2491 | 0.8218 |
| ming_omni | edf_cache | 0.350 | 96.20% | 9.92% | 12.29% | 9.53% | 4.56% | 2185.7260 | 5237.5639 | 839.5152 | 0.8055 |
| ming_omni | cache_weighted_edf | 0.350 | 96.20% | 10.03% | 12.55% | 9.66% | 4.29% | 2186.8133 | 5237.5639 | 851.8623 | 0.8045 |
| ming_omni | least_slack_cache | 0.350 | 96.20% | 10.31% | 13.00% | 9.85% | 4.25% | 2221.9772 | 5237.5639 | 902.5688 | 0.8148 |
| qwen3_omni | fifo_no_cache | 0.811 | 0.00% | 79.67% | 100.00% | 56.03% | 64.07% | 8425.9912 | 12694.7674 | 6808.4681 | 3.8508 |
| qwen3_omni | fifo_cache | 0.350 | 96.22% | 8.61% | 13.32% | 4.60% | 2.75% | 2168.6994 | 3187.8045 | 943.8367 | 0.8333 |
| qwen3_omni | edf_cache | 0.350 | 96.22% | 7.99% | 12.20% | 3.85% | 3.64% | 2168.6994 | 3245.6415 | 832.8699 | 0.8175 |
| qwen3_omni | cache_weighted_edf | 0.350 | 96.22% | 8.00% | 12.17% | 3.91% | 3.63% | 2168.6994 | 3254.3766 | 834.2653 | 0.8185 |
| qwen3_omni | least_slack_cache | 0.350 | 96.22% | 8.42% | 12.54% | 4.89% | 3.32% | 2168.6994 | 3246.0123 | 909.4929 | 0.8269 |

## Interpretation

- This is a deterministic queueing simulation using measured streaming first-delta latency from the committed Qwen3-Omni and Ming-Omni repeated-media serving artifacts.
- It does not model standalone text traffic, distributed cache coherence, memory bytes, in-flight duplicate coalescing, or real Stage runtime backpressure.
- `fifo_no_cache` uses the same arrival rate as the cache-enabled scenarios, so offered load above 1.0 means the measured cold path is not stable for that workload.
- `cache_weighted_edf` is the closest proxy for a unified SLO-aware policy: it uses modality SLO, estimated service time, and cache-hit state while preserving the existing Stage scheduler boundary.
