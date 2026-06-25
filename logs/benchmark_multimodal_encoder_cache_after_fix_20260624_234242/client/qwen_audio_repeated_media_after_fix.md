# Multimodal Repeated-Media Server Probe

- model: `qwen3-omni`
- base_url: `http://127.0.0.1:8100`
- media_kind: `audio`
- primary_media: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/tests/data/query_to_draw.wav`
- stream: `True`

## Summary

| phase | requests | ok | mean latency (s) | p50 latency (s) | p95 latency (s) | mean first delta (s) |
|---|---:|---:|---:|---:|---:|---:|
| cold_concurrent_same_media | 4 | 4 | 1.289 | 1.287 | 1.293 | 1.217 |
| single_different_media | 1 | 1 | 0.213 | 0.213 | 0.213 | 0.166 |
| warm_sequential_same_media | 4 | 4 | 0.134 | 0.133 | 0.136 | 0.106 |

## Requests

| phase | request_id | status | latency (s) | first delta (s) | prompt tokens | completion tokens | error |
|---|---|---:|---:|---:|---:|---:|---|
| cold_concurrent_same_media | `qwen-audio-afterfix-20260624-234242-cold_concurrent_same_media-0` | 200 | 1.295 | 1.244 | 86 | 6 |  |
| cold_concurrent_same_media | `qwen-audio-afterfix-20260624-234242-cold_concurrent_same_media-1` | 200 | 1.287 | 1.208 | 86 | 6 |  |
| cold_concurrent_same_media | `qwen-audio-afterfix-20260624-234242-cold_concurrent_same_media-2` | 200 | 1.287 | 1.208 | 86 | 6 |  |
| cold_concurrent_same_media | `qwen-audio-afterfix-20260624-234242-cold_concurrent_same_media-3` | 200 | 1.286 | 1.207 | 86 | 6 |  |
| single_different_media | `qwen-audio-afterfix-20260624-234242-single_different_media-0` | 200 | 0.213 | 0.166 | 80 | 10 |  |
| warm_sequential_same_media | `qwen-audio-afterfix-20260624-234242-warm_sequential_same_media-0` | 200 | 0.135 | 0.108 | 86 | 6 |  |
| warm_sequential_same_media | `qwen-audio-afterfix-20260624-234242-warm_sequential_same_media-1` | 200 | 0.131 | 0.105 | 86 | 6 |  |
| warm_sequential_same_media | `qwen-audio-afterfix-20260624-234242-warm_sequential_same_media-2` | 200 | 0.132 | 0.105 | 86 | 6 |  |
| warm_sequential_same_media | `qwen-audio-afterfix-20260624-234242-warm_sequential_same_media-3` | 200 | 0.137 | 0.107 | 86 | 6 |  |
