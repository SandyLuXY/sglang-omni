# Multimodal Prefix Cache Probe

- generated_at: 2026-06-24T23:43:18.785169+00:00
- git: 59af39b6
- dirty_state: ?? logs/benchmark_multimodal_encoder_cache_after_fix_20260624_234242/
- output_json: logs/benchmark_multimodal_encoder_cache_after_fix_20260624_234242/synthetic/prefix_cache_probe/results.json

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
| qwen3_omni | image_encoder | 1.0 | 0.0 | 8.0 | 0.0 | 3.1452 | 0.0229 | 137.34x | 33.33% |
| qwen3_omni | audio_encoder | 1.0 | 0.0 | 2.0 | 0.0 | 3.1345 | 0.0213 | 147.16x | 33.33% |

## Ming Encoder Cache Probe

{
  "active_encoder_cache": "wired in sglang_omni/models/ming_omni/stages.py",
  "media_key_to_radix_cache": "wired in sglang_omni/models/ming_omni/bootstrap.py",
  "model": "ming_omni",
  "notes": "Ming active stages pass cache_key through preprocessing/merge for radix safety and now use StageOutputCache in active audio/image encoder factories. This is a synthetic stage-helper probe, not a full-model run.",
  "stages": [
    {
      "avoidable_encoder_forwards_ratio": 0.9,
      "expected_without_cache_model_calls": 10,
      "first_request_ms": 3.1116,
      "model_calls": 1,
      "repeated_media_requests": 10,
      "stage": "audio_encoder",
      "warm_request_ms_median": 0.0106
    },
    {
      "avoidable_encoder_forwards_ratio": 0.9,
      "expected_without_cache_model_calls": 10,
      "first_request_ms": 3.0804,
      "model_calls": 1,
      "repeated_media_requests": 10,
      "stage": "image_encoder",
      "warm_request_ms_median": 0.0106
    }
  ]
}

## SLO Scheduling Simulation

| strategy | miss | audio miss | video miss | p50 ms | p95 ms | p99 ms | mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fifo | 4.76% | 10.33% | 0.00% | 28.3358 | 112.4192 | 158.5473 | 42.1834 |
| edf | 0.10% | 0.22% | 0.00% | 20.0 | 113.0682 | 211.2878 | 34.3515 |
| hybrid_slo_cache | 0.02% | 0.04% | 0.00% | 20.0 | 97.8369 | 198.9059 | 33.2316 |

The scheduling section is a deterministic synthetic model, not a production benchmark.
