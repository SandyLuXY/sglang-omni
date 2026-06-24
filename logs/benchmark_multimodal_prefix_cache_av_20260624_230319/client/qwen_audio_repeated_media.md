# Multimodal Repeated-Media Server Probe

- model: `qwen3-omni`
- base_url: `http://127.0.0.1:8100`
- media_kind: `audio`
- primary_media: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/tests/data/query_to_draw.wav`
- stream: `True`

## Summary

| phase | requests | ok | mean latency (s) | p50 latency (s) | p95 latency (s) | mean first delta (s) |
|---|---:|---:|---:|---:|---:|---:|
| cold_concurrent_same_media | 4 | 4 | 1.372 | 1.372 | 1.375 | 1.267 |
| single_different_media | 1 | 1 | 0.637 | 0.637 | 0.637 | 0.592 |
| warm_sequential_same_media | 4 | 4 | 0.229 | 0.162 | 0.391 | 0.177 |

## Requests

| phase | request_id | status | latency (s) | first delta (s) | prompt tokens | completion tokens | error |
|---|---|---:|---:|---:|---:|---:|---|
| cold_concurrent_same_media | `qwen-audio-mmcache-20260624-230319-cold_concurrent_same_media-0` | 200 | 1.375 | 1.287 | 86 | 11 |  |
| cold_concurrent_same_media | `qwen-audio-mmcache-20260624-230319-cold_concurrent_same_media-1` | 200 | 1.369 | 1.247 | 86 | 11 |  |
| cold_concurrent_same_media | `qwen-audio-mmcache-20260624-230319-cold_concurrent_same_media-2` | 200 | 1.375 | 1.287 | 86 | 11 |  |
| cold_concurrent_same_media | `qwen-audio-mmcache-20260624-230319-cold_concurrent_same_media-3` | 200 | 1.368 | 1.246 | 86 | 11 |  |
| single_different_media | `qwen-audio-mmcache-20260624-230319-single_different_media-0` | 200 | 0.637 | 0.592 | 80 | 10 |  |
| warm_sequential_same_media | `qwen-audio-mmcache-20260624-230319-warm_sequential_same_media-0` | 200 | 0.432 | 0.377 | 86 | 11 |  |
| warm_sequential_same_media | `qwen-audio-mmcache-20260624-230319-warm_sequential_same_media-1` | 200 | 0.163 | 0.113 | 86 | 11 |  |
| warm_sequential_same_media | `qwen-audio-mmcache-20260624-230319-warm_sequential_same_media-2` | 200 | 0.161 | 0.111 | 86 | 11 |  |
| warm_sequential_same_media | `qwen-audio-mmcache-20260624-230319-warm_sequential_same_media-3` | 200 | 0.159 | 0.109 | 86 | 11 |  |
