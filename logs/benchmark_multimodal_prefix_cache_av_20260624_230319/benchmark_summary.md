# Audio/Video Multimodal Prefix Cache Benchmark Summary

## Run Identity

- Run root: `logs/benchmark_multimodal_prefix_cache_av_20260624_230319`
- Created at UTC: `2026-06-24T23:03:20+00:00`
- Worktree: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache`
- Branch: `feat/omni-multimodal-prefix-cache`
- Benchmark start commit: `a2cb475c`
- Python: `/data/.venv/bin/python`

## Sync

Mandatory sync was run before serving benchmarks.

- `sglang-omni-sync --yes sync-main`
- `sglang-omni-sync --yes sync-feature /data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache`

Result:

- Main already matched `upstream/main` at `1ad75cc1`.
- Feature branch already matched `origin/feat/omni-multimodal-prefix-cache` at `a2cb475c`.
- Logs: `sync/sync-main.txt`, `sync/sync-feature.txt`

## Hardware

- Available GPUs before run: 8 x NVIDIA H100 80GB HBM3, GPUs 0-5 idle.
- Qwen3-Omni serving: physical GPU 0.
- Ming-Omni serving: physical GPUs 2,3,4,5 through `CUDA_VISIBLE_DEVICES=2,3,4,5`.
- GPU snapshots:
  - `gpu/nvidia-smi-pre.csv`
  - `gpu/nvidia-smi-qwen-post.csv`
  - `gpu/nvidia-smi-ming-after-patch-post.csv`
  - `gpu/nvidia-smi-post-cleanup.csv`

## Commands

Server/client command transcripts:

- `commands/qwen_server.sh`
- `commands/qwen_audio_client.sh`
- `commands/qwen_video_client.sh`
- `commands/ming_server.sh`
- `commands/ming_audio_client.sh`
- `commands/ming_video_client.sh`
- `commands/ming_server_after_patch.sh`
- `commands/ming_video_client_after_patch.sh`

Qwen3-Omni:

- Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Mode: text output, audio/video input
- Cache trace: `SGLANG_OMNI_TRACE_ENCODER_CACHE=1`
- Port: `8100`

Ming-Omni:

- Model: `inclusionAI/Ming-flash-omni-2.0`
- Mode: text output, audio/video input
- Thinker TP: 4
- Port: `8101`

## Client Results

Each successful probe sent 4 concurrent cold same-media requests and 4 warm
sequential same-media requests with `stream=true` and `max_tokens=16`. Audio
also sent one different-media request.

| Model | Media | Phase | Requests | Success | Mean latency s | p50 latency s | Mean first delta s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Omni | audio | cold concurrent same media | 4 | 4 | 1.372 | 1.372 | 1.267 |
| Qwen3-Omni | audio | warm sequential same media | 4 | 4 | 0.229 | 0.162 | 0.177 |
| Qwen3-Omni | audio | single different media | 1 | 1 | 0.637 | 0.637 | 0.592 |
| Qwen3-Omni | video | cold concurrent same media | 4 | 4 | 5.810 | 5.664 | 5.758 |
| Qwen3-Omni | video | warm sequential same media | 4 | 4 | 2.220 | 2.217 | 2.169 |
| Ming-Omni | audio | cold concurrent same media | 4 | 4 | 2.225 | 2.225 | 1.797 |
| Ming-Omni | audio | warm sequential same media | 4 | 4 | 0.231 | 0.226 | 0.200 |
| Ming-Omni | audio | single different media | 1 | 1 | 0.281 | 0.281 | 0.221 |
| Ming-Omni | video pre-patch | cold concurrent same media | 4 | 0 | N/A | N/A | N/A |
| Ming-Omni | video pre-patch | warm sequential same media | 4 | 0 | N/A | N/A | N/A |
| Ming-Omni | video after patch | cold concurrent same media | 4 | 4 | 7.052 | 6.816 | 6.305 |
| Ming-Omni | video after patch | warm sequential same media | 4 | 4 | 2.154 | 2.157 | 2.041 |

Raw client artifacts:

- `client/qwen_audio_repeated_media.json`
- `client/qwen_video_repeated_media.json`
- `client/ming_audio_repeated_media.json`
- `client/ming_video_repeated_media.json`
- `client/ming_video_repeated_media_after_patch.json`

## Server Evidence

Qwen3-Omni:

- Audio cold same-media group emitted 4 `audio_encoder` misses and 4 stores,
  showing the missing same-batch audio dedup gap in the real server path.
- Audio warm same-media group emitted 4 `audio_encoder` hits.
- Audio different-media request emitted a new `audio_encoder` miss/store.
- Audio prefix counters:
  - first cold batches: `#new-token: 172, #cached-token: 0`, then
    `#new-token: 2, #cached-token: 170`
  - warm same-media: each `#new-token: 1, #cached-token: 85`
  - different media: `#new-token: 76, #cached-token: 4`
- Video cold same-media group emitted 1 `image_encoder` miss/store and 3 hits.
- Video warm same-media group emitted 4 `image_encoder` hits.
- Video prefix counters:
  - first cold request: `#new-token: 555, #cached-token: 3`
  - all repeated same-video requests: `#new-token: 1, #cached-token: 557`

Ming-Omni:

- No `encoder_cache` records were emitted in either Ming server log, matching
  the source finding that active Ming audio/image encoder stages still do not
  wire `StageOutputCache`.
- Audio prefix counters:
  - first cold request: `#new-token: 165, #cached-token: 0`
  - remaining 3 concurrent same-audio requests: `#new-token: 3, #cached-token: 492`
  - warm same-audio: each `#new-token: 1, #cached-token: 164`
  - different audio: `#new-token: 123, #cached-token: 31`
- Ming video initially failed in preprocessing with:
  `Qwen2VLImageProcessorKwargs.__init__() got an unexpected keyword argument 'videos'`.
- After the compatibility patch, Ming video prefix counters were:
  - first cold request: `#new-token: 1290, #cached-token: 0`
  - repeated same-video requests: each `#new-token: 1, #cached-token: 1289`

Server logs:

- `server/qwen_server.log`
- `server/ming_server.log`
- `server/ming_server_after_patch.log`

## Code Fix

Ming video preprocessing was patched during this run:

- `sglang_omni/models/ming_omni/components/preprocessor.py`
- `tests/unit_test/ming_omni/test_pipeline.py::test_ming_video_preprocessor_falls_back_to_frame_image_processor`

The fix falls back to calling the installed `Qwen2VLImageProcessor` with a flat
list of video frames when `videos=` is unsupported, then reconstructs per-video
`video_grid_thw` rows from the returned `image_grid_thw`.

## Caveats

- This run covers audio-input/text-output and video-input/text-output. It does
  not cover text+audio output or live websocket/realtime streaming.
- Video used the single local `tests/data/draw.mp4` asset with
  `--video-fps 1 --video-max-frames 8 --video-max-pixels 200704`, so it tests
  same-video reuse but not different-video separation.
- Ming warm-request speedups are prefix/KV reuse evidence only, not active
  encoder-output cache evidence.
- Qwen audio cold concurrent behavior confirms a real same-batch dedup gap:
  identical audio requests all missed and stored separately before later warm
  hits.
- Servers were interrupted after the run. Ports 8100 and 8101 were closed and
  `gpu/nvidia-smi-post-cleanup.csv` shows GPUs returned to idle memory.

## Validation

- `/data/.venv/bin/python -m py_compile scripts/experiments/multimodal_prefix_cache_probe.py scripts/experiments/multimodal_server_repeated_media_probe.py sglang_omni/models/ming_omni/components/preprocessor.py`
- `/data/.venv/bin/python -m pytest -q tests/unit_test/qwen3_omni/test_pipeline.py::test_qwen_preprocessing_routes_only_active_encoder_branches tests/unit_test/qwen3_omni/test_pipeline.py::test_qwen_aggregate_projection_marks_uncached_active_encoder_inputs tests/unit_test/qwen3_omni/test_pipeline.py::test_qwen_mm_aggregate_keeps_lightweight_inputs_and_prunes_after_merge tests/unit_test/qwen3_omni/test_pipeline.py::test_qwen_sglang_request_hashes_media_tokens_without_changing_mrope_ids tests/unit_test/ming_omni/test_pipeline.py::test_ming_video_preprocessor_falls_back_to_frame_image_processor tests/unit_test/ming_omni/test_pipeline.py::test_compute_video_cache_key_changes_with_decode_params tests/unit_test/ming_omni/test_pipeline.py::test_ming_merge_extracts_video_embeds_into_thinker_inputs tests/unit_test/ming_omni/test_pipeline.py::test_ming_image_encoder_forward_video_embeds_match_token_counts tests/unit_test/ming_omni/test_pipeline.py::test_ming_image_encoder_forward_handles_image_and_video_together tests/unit_test/preprocessing/test_cache_key.py`
- Result: 15 passed, 20 warnings.
