# Multimodal Repeated-Media Server Probe

- model: `qwen3-omni`
- base_url: `http://127.0.0.1:8100`
- media_kind: `video`
- primary_media: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/tests/data/draw.mp4`
- stream: `True`

## Summary

| phase | requests | ok | mean latency (s) | p50 latency (s) | p95 latency (s) | mean first delta (s) |
|---|---:|---:|---:|---:|---:|---:|
| cold_concurrent_same_media | 4 | 4 | 5.810 | 5.664 | 8.415 | 5.758 |
| warm_sequential_same_media | 4 | 4 | 2.220 | 2.217 | 2.246 | 2.169 |

## Requests

| phase | request_id | status | latency (s) | first delta (s) | prompt tokens | completion tokens | error |
|---|---|---:|---:|---:|---:|---:|---|
| cold_concurrent_same_media | `qwen-video-mmcache-20260624-230319-cold_concurrent_same_media-0` | 200 | 3.188 | 3.136 | 558 | 11 |  |
| cold_concurrent_same_media | `qwen-video-mmcache-20260624-230319-cold_concurrent_same_media-1` | 200 | 4.662 | 4.610 | 558 | 11 |  |
| cold_concurrent_same_media | `qwen-video-mmcache-20260624-230319-cold_concurrent_same_media-2` | 200 | 8.723 | 8.671 | 558 | 11 |  |
| cold_concurrent_same_media | `qwen-video-mmcache-20260624-230319-cold_concurrent_same_media-3` | 200 | 6.667 | 6.614 | 558 | 11 |  |
| warm_sequential_same_media | `qwen-video-mmcache-20260624-230319-warm_sequential_same_media-0` | 200 | 2.209 | 2.157 | 558 | 11 |  |
| warm_sequential_same_media | `qwen-video-mmcache-20260624-230319-warm_sequential_same_media-1` | 200 | 2.250 | 2.200 | 558 | 11 |  |
| warm_sequential_same_media | `qwen-video-mmcache-20260624-230319-warm_sequential_same_media-2` | 200 | 2.196 | 2.144 | 558 | 11 |  |
| warm_sequential_same_media | `qwen-video-mmcache-20260624-230319-warm_sequential_same_media-3` | 200 | 2.226 | 2.175 | 558 | 11 |  |
