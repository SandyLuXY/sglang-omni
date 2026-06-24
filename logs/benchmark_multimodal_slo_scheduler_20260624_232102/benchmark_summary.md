# Multimodal Measured SLO Scheduler Benchmark

- generated_at: 2026-06-24T23:28:08.976805+00:00
- candidate_ref: 4b61fcf1
- run_root: `logs/benchmark_multimodal_slo_scheduler_20260624_232102`
- baseline_ref: N/A; this is a feature-branch feasibility simulation driven by measured serving artifacts, not a main-vs-candidate benchmark.
- workload: 100,000 synthetic requests per model, 2 workers per model, audio/image/video mix 50%/30%/20%, Zipf media locality skew 1.05, 10,000 warmup arrivals.
- SLOs: audio first-delta 600 ms, image first-delta 1200 ms, video first-delta 3500 ms.
- Caveat: no standalone text traffic, distributed cache coherence, byte-accurate memory pressure, in-flight duplicate coalescing, or live Stage backpressure is modeled.

## Measured TTFT Inputs

| model | modality | cold TTFT ms | warm TTFT ms | speedup | artifact |
| --- | --- | ---: | ---: | ---: | --- |
| qwen3_omni | image | 444.6335 | 180.3376 | 2.47x | `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/logs/benchmark_multimodal_prefix_cache_20260624_224653/client/qwen_repeated_media.json` |
| qwen3_omni | audio | 1287.0211 | 177.4352 | 7.25x | `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/logs/benchmark_multimodal_prefix_cache_av_20260624_230319/client/qwen_audio_repeated_media.json` |
| qwen3_omni | video | 3135.5938 | 2168.6994 | 1.45x | `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/logs/benchmark_multimodal_prefix_cache_av_20260624_230319/client/qwen_video_repeated_media.json` |
| ming_omni | image | 5237.5639 | 496.0815 | 10.56x | `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/logs/benchmark_multimodal_prefix_cache_20260624_224653/client/ming_repeated_media.json` |
| ming_omni | audio | 1862.3025 | 199.5327 | 9.33x | `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/logs/benchmark_multimodal_prefix_cache_av_20260624_230319/client/ming_audio_repeated_media.json` |
| ming_omni | video | 5271.7307 | 2040.5090 | 2.58x | `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/logs/benchmark_multimodal_prefix_cache_av_20260624_230319/client/ming_video_repeated_media_after_patch.json` |

## Default cache, 0.35 target utilization

- artifact: `logs/benchmark_multimodal_slo_scheduler_20260624_232102/results/measured_slo_target035`
- cache_capacity: `{'audio': 2048, 'image': 512, 'video': 128}`

| model | strategy | offered load | hit rate | miss | audio miss | image miss | video miss | p95 ms | p95 queue ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ming_omni | fifo_no_cache | 1.046 | 0.00% | 100.00% | 100.00% | 100.00% | 100.00% | 7206467.2952 | 7203359.0918 |
| ming_omni | fifo_cache | 0.350 | 84.01% | 25.23% | 21.30% | 28.54% | 30.14% | 5271.7307 | 1746.8141 |
| ming_omni | edf_cache | 0.350 | 84.01% | 24.65% | 20.32% | 27.81% | 30.78% | 5271.7307 | 1560.9703 |
| ming_omni | cache_weighted_edf | 0.350 | 84.01% | 24.52% | 20.12% | 27.85% | 30.60% | 5271.7307 | 1571.8367 |
| qwen3_omni | fifo_no_cache | 0.710 | 0.00% | 71.73% | 100.00% | 39.76% | 48.68% | 5871.7065 | 4171.5765 |
| qwen3_omni | fifo_cache | 0.350 | 83.99% | 11.72% | 18.04% | 5.66% | 4.90% | 3135.5938 | 1034.8995 |
| qwen3_omni | edf_cache | 0.350 | 83.99% | 11.18% | 17.04% | 4.94% | 5.79% | 3135.5938 | 930.5374 |
| qwen3_omni | cache_weighted_edf | 0.350 | 83.99% | 11.02% | 16.91% | 4.70% | 5.70% | 3135.5938 | 915.0651 |

## Default cache, 0.82 target utilization

- artifact: `logs/benchmark_multimodal_slo_scheduler_20260624_232102/results/measured_slo_target082`
- cache_capacity: `{'audio': 2048, 'image': 512, 'video': 128}`

| model | strategy | offered load | hit rate | miss | audio miss | image miss | video miss | p95 ms | p95 queue ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ming_omni | fifo_no_cache | 2.450 | 0.00% | 100.00% | 100.00% | 100.00% | 100.00% | 99640801.2440 | 99635752.3012 |
| ming_omni | fifo_cache | 0.820 | 84.01% | 69.81% | 70.95% | 70.34% | 66.18% | 13403.3408 | 11832.8650 |
| ming_omni | edf_cache | 0.820 | 84.01% | 66.95% | 65.66% | 67.80% | 68.91% | 13127.3540 | 11384.6920 |
| ming_omni | cache_weighted_edf | 0.820 | 84.01% | 67.50% | 66.77% | 68.12% | 68.38% | 12842.5853 | 10889.5594 |
| qwen3_omni | fifo_no_cache | 1.664 | 0.00% | 100.00% | 100.00% | 99.99% | 99.99% | 26405023.9501 | 26404043.0286 |
| qwen3_omni | fifo_cache | 0.820 | 83.99% | 58.82% | 66.09% | 52.56% | 49.88% | 7486.5904 | 6572.2627 |
| qwen3_omni | edf_cache | 0.820 | 83.99% | 51.86% | 56.25% | 43.33% | 53.73% | 6980.5547 | 5541.0796 |
| qwen3_omni | cache_weighted_edf | 0.820 | 83.99% | 52.10% | 56.59% | 44.69% | 51.96% | 7009.6377 | 5602.9622 |

## High cache capacity, 0.35 target utilization

- artifact: `logs/benchmark_multimodal_slo_scheduler_20260624_232102/results/measured_slo_target035_highcache`
- cache_capacity: `{'audio': 8192, 'image': 4096, 'video': 1024}`

| model | strategy | offered load | hit rate | miss | audio miss | image miss | video miss | p95 ms | p95 queue ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ming_omni | fifo_no_cache | 1.636 | 0.00% | 100.00% | 100.00% | 100.00% | 100.00% | 65339376.5792 | 65335590.3060 |
| ming_omni | fifo_cache | 0.350 | 96.20% | 10.85% | 14.13% | 10.42% | 3.32% | 2258.5883 | 987.2491 |
| ming_omni | edf_cache | 0.350 | 96.20% | 9.92% | 12.29% | 9.53% | 4.56% | 2185.7260 | 839.5152 |
| ming_omni | cache_weighted_edf | 0.350 | 96.20% | 10.03% | 12.55% | 9.66% | 4.29% | 2186.8133 | 851.8623 |
| qwen3_omni | fifo_no_cache | 0.811 | 0.00% | 79.67% | 100.00% | 56.03% | 64.07% | 8425.9912 | 6808.4681 |
| qwen3_omni | fifo_cache | 0.350 | 96.22% | 8.61% | 13.32% | 4.60% | 2.75% | 2168.6994 | 943.8367 |
| qwen3_omni | edf_cache | 0.350 | 96.22% | 7.99% | 12.20% | 3.85% | 3.64% | 2168.6994 | 832.8699 |
| qwen3_omni | cache_weighted_edf | 0.350 | 96.22% | 8.00% | 12.17% | 3.91% | 3.63% | 2168.6994 | 834.2653 |

## Findings

- Cache is a hard requirement for this workload. With the same arrival rate as the cache-enabled scenarios, no-cache Qwen3-Omni is overloaded at 0.82 target utilization and still misses 71.73% of deadlines at the 0.35 load point; no-cache Ming-Omni is unstable even at the 0.35 load point because measured cold image/video/audio TTFT exceeds the selected SLOs.
- At default cache capacity and 0.35 target utilization, Qwen3-Omni deadline miss rate improves from 11.72% FIFO-cache to 11.02% cache-weighted EDF; Ming-Omni improves from 25.23% to 24.52%. Scheduling helps, but the improvement is smaller than the cache hit-rate effect.
- At 0.82 target utilization, queueing dominates. EDF/cache-weighted EDF reduce Qwen3-Omni misses versus FIFO-cache, but misses remain above 50%; Ming-Omni remains above 66%. This is not a deployable SLO point for the measured service times.
- Raising cache capacity to make the synthetic hit rate about 96% at 0.35 utilization lowers Qwen3-Omni misses to about 8.00% and Ming-Omni misses to about 10.03% under cache-weighted EDF, but cold misses and queueing still set a nonzero floor.
- The measured-input conclusion is narrower than the earlier synthetic-only simulation: a unified SLO scheduler is feasible as a control-plane policy, but the decisive levers are high media-cache hit rate, in-flight duplicate coalescing, and additional capacity/backpressure. Scheduler priority alone cannot rescue strict realtime SLOs when cold multimodal TTFT exceeds the deadline.

## Commands

- Sync logs: `logs/benchmark_multimodal_slo_scheduler_20260624_232102/sync`
- Command logs: `logs/benchmark_multimodal_slo_scheduler_20260624_232102/commands`
- Validation: `/data/.venv/bin/python -m py_compile ...` and focused pytest
  were run after the simulator/doc updates; result was `17 passed, 20 warnings
  in 7.01s`. Full output is in
  `logs/benchmark_multimodal_slo_scheduler_20260624_232102/validation.log`.
