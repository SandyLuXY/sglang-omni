# Multimodal Prefix Cache Benchmark Summary

## Run Identity

- Run root: `logs/benchmark_multimodal_prefix_cache_20260624_224653`
- Created at UTC: `2026-06-24T22:47:12+00:00`
- Worktree: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache`
- Branch: `feat/omni-multimodal-prefix-cache`
- Pre-sync commit: `8561ab7f`
- Post-sync benchmark commit: `c879d561`
- Python: `/data/.venv/bin/python`

## Sync

Mandatory sync was run before serving benchmarks.

- `sglang-omni-sync --yes sync-main`
- `sglang-omni-sync --yes sync-feature /data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache`

Result:

- Main fast-forwarded to `1ad75cc1`.
- Feature branch rebased and pushed to `c879d561`.
- Logs: `sync/sync-main.txt`, `sync/sync-feature.txt`

## Hardware

- Available GPUs before run: 8 x NVIDIA H100 80GB HBM3, idle.
- Qwen3-Omni serving: physical GPU 0.
- Ming-Omni serving: physical GPUs 2,3,4,5 through `CUDA_VISIBLE_DEVICES=2,3,4,5`.
- GPU snapshots: `gpu/nvidia-smi-pre.csv`, `gpu/nvidia-smi-qwen-post.csv`,
  `gpu/nvidia-smi-ming-post.csv`, `gpu/nvidia-smi-post-cleanup.csv`

## Commands

Server/client command transcripts:

- `commands/qwen_server.sh`
- `commands/qwen_client.sh`
- `commands/ming_server.sh`
- `commands/ming_client.sh`

Qwen3-Omni:

- Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Mode: text output, image input
- Cache trace: `SGLANG_OMNI_TRACE_ENCODER_CACHE=1`
- Port: `8100`

Ming-Omni:

- Model: `inclusionAI/Ming-flash-omni-2.0`
- Mode: text output, image input
- Thinker TP: 4
- Port: `8101`

## Client Results

The client sent 4 concurrent cold same-image requests, 4 warm sequential
same-image requests, and 1 different-image request with `stream=true` and
`max_tokens=16`.

| Model | Phase | Requests | Success | Mean latency s | p50 latency s | p95 latency s | Mean first delta s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Omni | cold concurrent same image | 4 | 4 | 0.558 | 0.560 | 0.560 | 0.433 |
| Qwen3-Omni | warm sequential same image | 4 | 4 | 0.264 | 0.264 | 0.274 | 0.180 |
| Qwen3-Omni | single different image | 1 | 1 | 0.615 | 0.615 | 0.615 | 0.531 |
| Ming-Omni | cold concurrent same image | 4 | 4 | 5.707 | 5.705 | 5.720 | 4.948 |
| Ming-Omni | warm sequential same image | 4 | 4 | 0.612 | 0.601 | 0.660 | 0.496 |
| Ming-Omni | single different image | 1 | 1 | 0.883 | 0.883 | 0.883 | 0.767 |

Raw client artifacts:

- `client/qwen_repeated_media.json`
- `client/qwen_repeated_media.md`
- `client/qwen_first_attempt_failure.md`
- `client/ming_repeated_media.json`
- `client/ming_repeated_media.md`

## Server Evidence

Qwen3-Omni:

- Cold same-image group: one `image_encoder` `miss` + `store`, then three
  `image_encoder` `hit` records for the same media key.
- Warm same-image group: four `image_encoder` `hit` records.
- Different-image request: one `image_encoder` `miss` + `store`.
- Prefix/KV counters for the JSON run:
  - first same-image request: `#new-token: 1484`, `#cached-token: 4`
  - remaining 3 same-image concurrent requests: `#new-token: 3`, `#cached-token: 4461`
  - each warm same-image request: `#new-token: 1`, `#cached-token: 1487`
  - different-image request: `#new-token: 3263`, `#cached-token: 4`

Ming-Omni:

- No `encoder_cache` trace records were emitted, matching the source finding
  that active Ming image/audio encoder stages do not wire `StageOutputCache`.
- Prefix/KV counters:
  - first same-image request: `#new-token: 986`, `#cached-token: 0`
  - remaining 3 same-image concurrent requests: `#new-token: 3`, `#cached-token: 2955`
  - each warm same-image request: `#new-token: 1`, `#cached-token: 985`
  - different-image request: `#new-token: 939`, `#cached-token: 21`

Server logs:

- `server/qwen_server.log`
- `server/ming_server.log`

## Caveats

- This is an image-input/text-output serving probe, not a mixed audio/video
  streaming benchmark.
- The first Qwen client attempt sent traffic but failed to write JSON because
  the benchmark script tried to serialize `Path` objects. The script was fixed,
  then rerun with uncached local images for the recorded JSON/Markdown results.
- Qwen's concurrent group did not emit `dedup_same_batch`; it showed one
  encoder compute followed by cache hits for the other same-image requests.
- Ming's warm-request speedup is supported by SGLang prefix/KV counters only;
  it is not evidence of Ming active encoder-output caching.
- Servers were interrupted after the run. The shutdown logs contain expected
  `KeyboardInterrupt`/NCCL shutdown warnings; ports 8100 and 8101 were closed
  and `gpu/nvidia-smi-post-cleanup.csv` shows GPUs returned to idle memory.

## Validation

- `/data/.venv/bin/python -m py_compile scripts/experiments/multimodal_prefix_cache_probe.py scripts/experiments/multimodal_server_repeated_media_probe.py`
- `/data/.venv/bin/python -m pytest -q tests/unit_test/qwen3_omni/test_pipeline.py::test_qwen_preprocessing_routes_only_active_encoder_branches tests/unit_test/qwen3_omni/test_pipeline.py::test_qwen_aggregate_projection_marks_uncached_active_encoder_inputs tests/unit_test/qwen3_omni/test_pipeline.py::test_qwen_mm_aggregate_keeps_lightweight_inputs_and_prunes_after_merge tests/unit_test/qwen3_omni/test_pipeline.py::test_qwen_sglang_request_hashes_media_tokens_without_changing_mrope_ids tests/unit_test/ming_omni/test_pipeline.py::test_compute_video_cache_key_changes_with_decode_params tests/unit_test/ming_omni/test_pipeline.py::test_ming_merge_extracts_video_embeds_into_thinker_inputs tests/unit_test/preprocessing/test_cache_key.py`
- Result: 12 passed, 20 warnings.
