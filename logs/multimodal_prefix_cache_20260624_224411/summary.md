# Multimodal Prefix Cache Probe

- generated_at: 2026-06-24T22:44:11.668959+00:00
- git: 86e73bdb
- dirty_state: ?? docs/design/multimodal_prefix_cache_feasibility.md
?? logs/
?? scripts/experiments/
- output_json: logs/multimodal_prefix_cache_20260624_224411/results.json

## Prefix Keying

| model | case | prompt | raw common | keyed common | reuse | unsafe raw avoided |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3_omni | same_media_different_question | 35 | 31 | 31 | 88.57% | 0 |
| qwen3_omni | different_image_same_shape | 35 | 35 | 2 | 5.71% | 33 |
| qwen3_omni | different_audio_same_shape | 35 | 35 | 19 | 54.29% | 16 |
| qwen3_omni | different_video_same_shape | 26 | 26 | 2 | 7.69% | 24 |
| ming_omni | same_media_different_question | 35 | 31 | 31 | 88.57% | 0 |
| ming_omni | different_image_same_shape | 35 | 35 | 2 | 5.71% | 33 |
| ming_omni | different_audio_same_shape | 35 | 35 | 19 | 54.29% | 16 |
| ming_omni | different_video_same_shape | 26 | 26 | 2 | 7.69% | 24 |

## Encoder Cache Probe

| model | stage | cold calls | warm calls | cold units | warm units | cold ms | warm ms | speedup | same-batch unit reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3_omni | image_encoder | 1.0 | 0.0 | 8.0 | 0.0 | 3.1442 | 0.0234 | 134.37x | 33.33% |
| qwen3_omni | audio_encoder | 1.0 | 0.0 | 3.0 | 0.0 | 3.1455 | 0.0219 | 143.63x | 0.00% |

## Ming Encoder Gap

{
  "active_encoder_cache": "not wired in sglang_omni/models/ming_omni/stages.py",
  "avoidable_encoder_forwards_ratio": 0.9,
  "current_expected_encoder_forwards": 10,
  "media_key_to_radix_cache": "wired in sglang_omni/models/ming_omni/bootstrap.py",
  "model": "ming_omni",
  "notes": "Ming active stages pass cache_key through preprocessing/merge for radix safety, but the active SimpleScheduler encoder factories do not use StageOutputCache. This is a static code-path estimate, not a model run.",
  "repeated_media_requests": 10,
  "with_stage_output_cache_expected_forwards": 1
}

## SLO Scheduling Simulation

| strategy | miss | audio miss | video miss | p50 ms | p95 ms | p99 ms | mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fifo | 4.76% | 10.33% | 0.00% | 28.3358 | 112.4192 | 158.5473 | 42.1834 |
| edf | 0.10% | 0.22% | 0.00% | 20.0 | 113.0682 | 211.2878 | 34.3515 |
| hybrid_slo_cache | 0.02% | 0.04% | 0.00% | 20.0 | 97.8369 | 198.9059 | 33.2316 |

The scheduling section is a deterministic synthetic model, not a production benchmark.
