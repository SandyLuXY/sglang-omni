# Multimodal Prefix Cache Feasibility Notes

## Scope

This note covers only Qwen3-Omni and Ming-Omni on branch
`feat/omni-multimodal-prefix-cache`.

The current evidence combines source review and a lightweight probe that executes
the active request-builder/cache code paths with synthetic tensors. It is valid
for cache-mechanics feasibility. It is not a full-model latency benchmark.

Raw probe artifacts:

- `logs/multimodal_prefix_cache_20260624_224411/results.json`
- `logs/multimodal_prefix_cache_20260624_224411/summary.md`

Run command:

```bash
python scripts/experiments/multimodal_prefix_cache_probe.py --repeats 30
```

Environment captured by the probe:

- commit: `86e73bdb`
- Python: `3.12.3`
- GPUs visible: 8 x H100 80GB, idle at collection time
- full-model-performance claim: false

## Current Code Path

### Qwen3-Omni

Pipeline shape:

```mermaid
flowchart LR
  preprocessing --> image_encoder
  preprocessing --> audio_encoder
  preprocessing --> mm_aggregate
  image_encoder --> mm_aggregate
  audio_encoder --> mm_aggregate
  mm_aggregate --> thinker
  mm_aggregate --> talker_ar
  thinker -. stream .-> decode
  thinker -. stream .-> talker_ar
  thinker --> decode
  talker_ar -. stream .-> code2wav
  talker_ar --> code2wav
  decode --> client((client))
  code2wav --> client
```

Key source evidence:

- `sglang_omni/models/qwen3_omni/config.py:25` routes preprocessing to
  `image_encoder`, `audio_encoder`, and `mm_aggregate`.
- `sglang_omni/models/qwen3_omni/config.py:79` makes `mm_aggregate` wait for
  preprocessing plus active encoders and merge via `merge_for_thinker`.
- `sglang_omni/models/qwen3_omni/config.py:109` streams thinker output to
  decode and, for speech, `talker_ar`.
- `sglang_omni/models/qwen3_omni/components/preprocessor.py:343` builds media
  cache keys before media conversion and contextualizes video/audio parameters.
- `sglang_omni/models/qwen3_omni/merge.py:151` passes modality-prefixed
  `media_cache_keys` into thinker inputs.
- `sglang_omni/models/qwen3_omni/request_builders.py:534` hashes media keys into
  stable out-of-vocab pad values and substitutes generic media placeholder token
  IDs in `Req.origin_input_ids`.
- `sglang_omni/models/qwen3_omni/stages.py:316` uses `StageOutputCache` for
  encoder output lookup/store.
- `sglang_omni/models/qwen3_omni/stages.py:373` deduplicates same-batch image
  encoder requests by cache key.
- `sglang_omni/models/qwen3_omni/stages.py:636` batches audio encoder requests
  and uses warm cache hits, but does not deduplicate duplicate audio cache keys
  inside the same cold batch.

### Ming-Omni

Pipeline shape:

```mermaid
flowchart LR
  preprocessing --> audio_encoder
  preprocessing --> image_encoder
  preprocessing --> mm_aggregate
  audio_encoder --> mm_aggregate
  image_encoder --> mm_aggregate
  mm_aggregate --> thinker
  thinker -. stream .-> decode
  thinker --> decode
  thinker --> talker
  thinker -. stream .-> segmenter
  segmenter -. stream .-> talker_stream
  segmenter --> talker_stream
  decode --> client((client))
  talker --> client
  talker_stream --> client
```

Key source evidence:

- `sglang_omni/models/ming_omni/config.py:49` defines preprocessing fan-out to
  audio/image encoders and aggregate.
- `sglang_omni/models/ming_omni/config.py:94` defines aggregate fan-in and
  merge to thinker.
- `sglang_omni/models/ming_omni/config.py:117` adds the streaming speech path
  through `segmenter` and `talker_stream`.
- `sglang_omni/models/ming_omni/components/preprocessor.py:393` computes media
  cache keys before async media loading.
- `sglang_omni/models/ming_omni/components/preprocessor.py:544` always creates
  audio/image encoder input keys so aggregate receives all configured sources.
- `sglang_omni/models/ming_omni/pipeline/merge.py:119` forwards
  modality-prefixed `media_cache_keys`.
- `sglang_omni/models/ming_omni/bootstrap.py:135` hashes media keys into stable
  out-of-vocab pad values and substitutes generic media placeholder token IDs in
  `Req.origin_input_ids`.
- `sglang_omni/models/ming_omni/stages.py:145` and
  `sglang_omni/models/ming_omni/stages.py:176` run active encoder factories
  without `StageOutputCache`; `cache_key` is stripped from model inputs but not
  used to skip repeated encoder work.

## Probe Results

### Prefix Keying

The probe constructs Qwen3 and Ming `Req.origin_input_ids` through the active
request-builder paths. It compares common-prefix lengths with raw generic media
placeholder IDs versus keyed out-of-vocab media IDs.

| Model | Case | Prompt tokens | Raw common prefix | Keyed common prefix | Keyed reuse | Unsafe raw reuse avoided |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Omni | same media, different question | 35 | 31 | 31 | 88.57% | 0 |
| Qwen3-Omni | different image, same placeholder shape | 35 | 35 | 2 | 5.71% | 33 |
| Qwen3-Omni | different audio, same placeholder shape | 35 | 35 | 19 | 54.29% | 16 |
| Qwen3-Omni | different video decode params, same shape | 26 | 26 | 2 | 7.69% | 24 |
| Ming-Omni | same media, different question | 35 | 31 | 31 | 88.57% | 0 |
| Ming-Omni | different image, same placeholder shape | 35 | 35 | 2 | 5.71% | 33 |
| Ming-Omni | different audio, same placeholder shape | 35 | 35 | 19 | 54.29% | 16 |
| Ming-Omni | different video decode params, same shape | 26 | 26 | 2 | 7.69% | 24 |

Interpretation:

- Same-media prompts keep the multimodal prefix reusable. In the 35-token
  synthetic prompt, 31 tokens remain cacheable across different questions.
- Different media with identical placeholder counts no longer falsely share the
  whole prompt. The keyed path cuts common prefix to the text before the changed
  media segment.
- Video decode parameters are part of the video key, so same video bytes decoded
  with different fps/max-frame settings do not alias.

### Encoder Cache

The encoder-cache probe uses synthetic encoder modules that sleep 3 ms per
batched forward. The numbers measure the active batching/cache mechanics, not
real model FLOPs.

| Model | Stage | Cold calls | Warm calls | Cold processed units | Warm processed units | Cold median ms | Warm median ms | Warm speedup | Same-batch duplicate reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Omni | image_encoder | 1 | 0 | 8 visual tokens | 0 | 3.1442 | 0.0234 | 134.37x | 33.33% |
| Qwen3-Omni | audio_encoder | 1 | 0 | 3 audio rows | 0 | 3.1455 | 0.0219 | 143.63x | 0.00% |

Interpretation:

- Qwen3 image encoder has both warm-cache skip and same-batch duplicate
  elimination. In a 3-request batch with one duplicate image key, synthetic
  visual-token work drops from 12 to 8 tokens on the cold batch, then to 0 on
  warm cache.
- Qwen3 audio encoder has warm-cache skip but no same-batch duplicate
  elimination. A matching same-batch dedup pass would remove 1 of 3 audio rows
  in this probe.
- Ming encoder cache is not wired on the active path. For 10 repeated media
  requests, current source implies 10 encoder forwards; a Qwen-style
  `StageOutputCache` would reduce that to 1, an avoidable 90% repeated-encoder
  forward ratio under this repeated-media workload.

### SLO Scheduling Simulation

The scheduling simulation is deterministic and synthetic. It models 5,000
requests, 2 workers, mixed audio/text/video arrivals, modality-specific service
times, cache-hit probabilities, and SLO deadlines. It is a direction check for
hybrid SLO-aware scheduling, not a production benchmark.

| Strategy | Deadline miss | Audio miss | Video miss | p50 ms | p95 ms | p99 ms | Mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FIFO | 4.76% | 10.33% | 0.00% | 28.3358 | 112.4192 | 158.5473 | 42.1834 |
| EDF | 0.10% | 0.22% | 0.00% | 20.0000 | 113.0682 | 211.2878 | 34.3515 |
| Hybrid SLO + cache | 0.02% | 0.04% | 0.00% | 20.0000 | 97.8369 | 198.9059 | 33.2316 |

Interpretation:

- A simple deadline-aware policy reduces synthetic audio miss rate from 10.33%
  to 0.22%.
- Adding cache-hit bias reduces the miss rate further to 0.04% and improves p95
  latency in this workload.
- This supports feasibility of mixed SLO scheduling as a control-plane policy,
  but the numbers must be replaced by profiler-backed stage service times before
  making deployment claims.

## Feasibility Conclusion

The cache part is feasible and already partially implemented:

- Qwen3-Omni has a working two-level pattern: non-AR encoder output cache plus
  SGLang radix prefix cache made multimodal-safe by content-keyed placeholder
  substitution.
- Ming-Omni has the radix-safety part on the active thinker path, but lacks the
  non-AR encoder output cache on active encoder stages.
- The media-key approach handles the key correctness problem that would otherwise
  make generic `<imagePatch>` / `<audioPatch>` / `<videoPatch>` placeholders
  unsafe for radix prefix caching.

The scheduling part is feasible as a next framework layer, but not implemented:

- Current stage configs already expose modality-separated stages, stream edges,
  fan-in waits, and terminal stages, which gives enough structure for a scheduler
  to attach modality-specific SLOs.
- The current code does not have a unified SLO-aware scheduler across stages,
  backpressure, cache-admission policy, or distributed cache namespace.
- The synthetic scheduler result shows directionally large benefit for audio SLO
  misses, but it is not sufficient for production capacity claims.

## Engineering Gaps

1. Ming encoder cache parity.
   Add `StageOutputCache` to Ming audio/image active encoder factories and keep
   cache keys contextualized by preprocessing parameters.

2. Qwen3 audio same-batch dedup.
   Mirror image-encoder duplicate coalescing in `_batch_audio_encoder_payloads`.

3. Unified cache identity schema.
   Include model id/revision, modality, preprocessing parameters, encoder
   implementation version, dtype, and output-shape contract in keys. Current
   Qwen3/Ming keys cover content and selected preprocessing knobs, but not every
   deployment/version dimension.

4. Cache telemetry.
   Promote hit/miss/store/evict counters to structured profiler events so
   formal runs can attribute TTFT and memory changes to cache behavior.

5. SLO scheduler.
   Add a policy layer that can prioritize audio-latency-sensitive work, bound
   video memory pressure, and account for cache-hit probability without changing
   Stage or Coordinator tensor/control-plane boundaries.

6. Formal GPU validation.
   Run function-check and benchmark-full profiles for Qwen3-Omni and Ming-Omni
   with repeated-media workloads, then compare profiler-backed TTFT, encoder
   time, prefix-hit length, HBM usage, and deadline miss rate.
