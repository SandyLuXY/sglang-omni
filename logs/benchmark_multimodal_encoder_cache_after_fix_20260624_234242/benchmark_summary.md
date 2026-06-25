# Multimodal Encoder Cache After-Fix Benchmark

## Run Identity

- Run root: `logs/benchmark_multimodal_encoder_cache_after_fix_20260624_234242`
- Worktree: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache`
- Branch: `feat/omni-multimodal-prefix-cache`
- Commit under test: `59af39b6`
- Python: `/data/.venv/bin/python`

## Sync

Mandatory sync was run before serving benchmarks.

- `sglang-omni-sync --yes sync-main`
- `sglang-omni-sync --yes sync-feature /data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache`

Result:

- Main was synchronized with upstream before the run.
- Feature branch was synchronized and pushed to origin at `59af39b6`.
- Logs: `sync/sync-main.txt`, `sync/sync-feature.txt`

## Hardware

- Available GPUs before run: 8 x NVIDIA H100 80GB HBM3.
- Qwen3-Omni serving: physical GPU 0.
- Ming-Omni serving: physical GPUs 2,3,4,5 through `CUDA_VISIBLE_DEVICES=2,3,4,5`.
- Post-cleanup snapshot shows ports 8100/8101 closed and GPUs returned to idle memory.
- GPU snapshots:
  - `gpu/nvidia-smi-pre.csv`
  - `gpu/nvidia-smi-post-cleanup.csv`
  - `gpu/nvidia-smi-pmon-pre.txt`
  - `gpu/nvidia-smi-pmon-post-cleanup.txt`

## Code Under Test

This run validates commit `59af39b6`, which changed:

- Qwen3-Omni audio encoder batching now coalesces duplicate cache keys inside
  the same cold batch, matching the existing image path.
- Ming-Omni active audio/image encoder factories now use model-local
  `StageOutputCache` lookup/store logic.
- Synthetic probe coverage now includes the Ming active encoder cache helper.
- Unit tests cover Qwen audio same-batch dedup and Ming audio/image cache reuse.

## Synthetic Probe

Command transcript: `commands/synthetic_probe.sh`.

Raw artifacts:

- `synthetic/prefix_cache_probe/results.json`
- `synthetic/prefix_cache_probe/summary.md`
- `synthetic/prefix_cache_probe.stdout`

Key result:

| Model | Stage | Cold model calls | Warm model calls | Cold processed units | Warm processed units | Warm speedup | Duplicate work removed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Omni | image_encoder | 1 | 0 | 8 visual tokens | 0 | 137.34x | 33.33% same-batch reduction |
| Qwen3-Omni | audio_encoder | 1 | 0 | 2 audio rows | 0 | 147.16x | 33.33% same-batch reduction |
| Ming-Omni | audio_encoder | 1 | N/A | 10 repeated requests | N/A | N/A | 90.00% avoidable forwards removed |
| Ming-Omni | image_encoder | 1 | N/A | 10 repeated requests | N/A | N/A | 90.00% avoidable forwards removed |

Interpretation:

- The Qwen audio synthetic cold batch now computes only unique audio cache keys.
- Ming active audio/image encoder cache wiring computes once for 10 repeated
  payloads in the helper-level probe.

## Serving Probes

Each probe sent 4 concurrent cold same-media requests, 4 warm sequential
same-media requests, and one different-media request where a distinct fixture
was available. All requests used `stream=true` and `max_tokens=16`.

Server/client command transcripts:

- `commands/qwen_server.sh`
- `commands/qwen_audio_client.sh`
- `commands/ming_server.sh`
- `commands/ming_audio_client.sh`
- `commands/ming_image_client.sh`
- `commands/ming_video_client.sh`

Raw client artifacts:

- `client/qwen_audio_repeated_media_after_fix.json`
- `client/ming_audio_repeated_media_after_fix.json`
- `client/ming_image_repeated_media_after_fix.json`
- `client/ming_video_repeated_media_after_fix.json`

Client timing:

| Model | Media | Phase | Requests | Success | Mean latency s | p50 latency s | p95 latency s | Mean first delta s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Omni | audio | cold concurrent same media | 4 | 4 | 1.289 | 1.287 | 1.293 | 1.217 |
| Qwen3-Omni | audio | warm sequential same media | 4 | 4 | 0.134 | 0.133 | 0.136 | 0.106 |
| Qwen3-Omni | audio | single different media | 1 | 1 | 0.213 | 0.213 | 0.213 | 0.166 |
| Ming-Omni | audio | cold concurrent same media | 4 | 4 | 2.167 | 2.167 | 2.169 | 1.734 |
| Ming-Omni | audio | warm sequential same media | 4 | 4 | 0.223 | 0.211 | 0.256 | 0.191 |
| Ming-Omni | audio | single different media | 1 | 1 | 0.276 | 0.276 | 0.276 | 0.215 |
| Ming-Omni | image | cold concurrent same media | 4 | 4 | 4.256 | 4.256 | 4.271 | 3.871 |
| Ming-Omni | image | warm sequential same media | 4 | 4 | 0.679 | 0.678 | 0.689 | 0.558 |
| Ming-Omni | image | single different media | 1 | 1 | 0.705 | 0.705 | 0.705 | 0.586 |
| Ming-Omni | video | cold concurrent same media | 4 | 4 | 4.796 | 4.815 | 6.782 | 4.678 |
| Ming-Omni | video | warm sequential same media | 4 | 4 | 2.219 | 2.209 | 2.268 | 2.101 |
| Ming-Omni | video | single same-video control | 1 | 1 | 2.196 | 2.196 | 2.196 | 2.078 |

## Server Evidence

Qwen3-Omni audio:

- Cold same-audio group emitted one effective store for the repeated key and 3
  `dedup_same_batch` records. The trace logs still print `miss` for duplicate
  waiters before the dedup record, so `store` and `dedup_same_batch` are the
  source of truth for actual encoder forwards.
- Warm same-audio group emitted 4 `audio_encoder` hits.
- Different-audio request emitted a separate miss/store.
- Server log: `server/qwen_server.log`.

Ming-Omni:

- Audio cold same-media group emitted one miss/store followed by 3 hits; warm
  same-audio emitted 4 hits; different audio emitted a separate miss/store.
- Image cold same-media group emitted one miss/store followed by 3 hits; warm
  same-image emitted 4 hits; different image emitted a separate miss/store.
- Video reaches the image encoder cache after preprocessing. The cold same-video
  group emitted one miss/store followed by hits, and warm repeats emitted hits.
- Video still spends about 1.3-1.4 seconds per request in frame extraction, so
  encoder-output caching does not remove all warm video latency.
- Server log: `server/ming_server.log`.

Trace count sanity checks:

| Server | Stage | miss records | store records | dedup records | hit records |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-Omni | audio_encoder | 5 | 2 | 3 | 4 |
| Ming-Omni | audio_encoder | 2 | 2 | 0 | 7 |
| Ming-Omni | image_encoder | 3 | 3 | 0 | 15 |

The extra Qwen miss/store pair is the different-audio control request. Ming
image counts include image primary, image different-media, and video primary
keys.

## Comparison To Earlier Gap Run

Earlier run: `logs/benchmark_multimodal_prefix_cache_av_20260624_230319`.

| Model | Media | Earlier evidence | After-fix evidence |
| --- | --- | --- | --- |
| Qwen3-Omni | audio | Cold same-audio emitted 4 misses and 4 stores. | Cold same-audio emitted 3 same-batch dedup records and one effective store. |
| Ming-Omni | audio | No `encoder_cache` records; prefix/KV reuse only. | Active audio encoder emits miss/store/hit records. |
| Ming-Omni | image | No `encoder_cache` records in image run; prefix/KV reuse only. | Active image encoder emits miss/store/hit records. |
| Ming-Omni | video | No encoder-output cache telemetry after video compatibility patch. | Video-through-image-encoder emits miss/store/hit records. |

Latency comparison:

| Model | Media | Phase | Earlier mean latency s | After-fix mean latency s |
| --- | --- | --- | ---: | ---: |
| Qwen3-Omni | audio | cold concurrent same media | 1.372 | 1.289 |
| Qwen3-Omni | audio | warm sequential same media | 0.229 | 0.134 |
| Ming-Omni | audio | cold concurrent same media | 2.225 | 2.167 |
| Ming-Omni | audio | warm sequential same media | 0.231 | 0.223 |
| Ming-Omni | image | cold concurrent same media | 5.707 | 4.256 |
| Ming-Omni | image | warm sequential same media | 0.612 | 0.679 |
| Ming-Omni | video | cold concurrent same media | 7.052 | 4.796 |
| Ming-Omni | video | warm sequential same media | 2.154 | 2.219 |

Interpretation:

- Cache telemetry conclusively closes the Qwen audio and Ming encoder-output
  cache gaps.
- Warm latency is not expected to improve uniformly because AR prefix reuse,
  preprocessing, video decode, and generation dominate some phases.
- The strongest remaining runtime bottleneck in these fixtures is video
  preprocessing before the cached image-encoder output is reached.

## Validation

- `py_compile` succeeded for the experiment scripts and touched model files.
- Targeted pytest result: `18 passed, 20 warnings in 6.76s`.
- Validation transcript: `commands/validation.txt`.

## Caveats

- This run covers text-output serving. It does not cover text+audio output or
  websocket realtime streaming.
- The video probe uses the single local `tests/data/draw.mp4` asset, so the
  single-video control is not a different-video separation test.
- Trace logging was enabled with `SGLANG_OMNI_TRACE_ENCODER_CACHE=1`; production
  observability still needs structured profiler counters for hit/miss/store,
  eviction, output bytes, and per-stage latency attribution.
- The unified SLO scheduler/admission layer remains unimplemented. Existing SLO
  simulations show scheduler priority helps, but cache hit rate, duplicate
  coalescing, admission control, and capacity dominate strict realtime SLOs.
