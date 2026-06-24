# Multimodal Repeated-Media Server Probe

- model: `qwen3-omni`
- base_url: `http://127.0.0.1:8100`
- media_kind: `image`
- primary_media: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/docs/_static/image/higgs-architecture.png`
- stream: `True`

## Summary

| phase | requests | ok | mean latency (s) | p50 latency (s) | p95 latency (s) | mean first delta (s) |
|---|---:|---:|---:|---:|---:|---:|
| cold_concurrent_same_media | 4 | 4 | 0.558 | 0.560 | 0.560 | 0.433 |
| single_different_media | 1 | 1 | 0.615 | 0.615 | 0.615 | 0.531 |
| warm_sequential_same_media | 4 | 4 | 0.264 | 0.264 | 0.274 | 0.180 |

## Requests

| phase | request_id | status | latency (s) | first delta (s) | prompt tokens | completion tokens | error |
|---|---|---:|---:|---:|---:|---:|---|
| cold_concurrent_same_media | `qwen-mmcache-20260624-2254-json-cold_concurrent_same_media-0` | 200 | 0.561 | 0.445 | 1488 | 16 |  |
| cold_concurrent_same_media | `qwen-mmcache-20260624-2254-json-cold_concurrent_same_media-1` | 200 | 0.551 | 0.399 | 1488 | 16 |  |
| cold_concurrent_same_media | `qwen-mmcache-20260624-2254-json-cold_concurrent_same_media-2` | 200 | 0.560 | 0.444 | 1488 | 16 |  |
| cold_concurrent_same_media | `qwen-mmcache-20260624-2254-json-cold_concurrent_same_media-3` | 200 | 0.559 | 0.443 | 1488 | 16 |  |
| single_different_media | `qwen-mmcache-20260624-2254-json-single_different_media-0` | 200 | 0.615 | 0.531 | 3267 | 16 |  |
| warm_sequential_same_media | `qwen-mmcache-20260624-2254-json-warm_sequential_same_media-0` | 200 | 0.276 | 0.192 | 1488 | 16 |  |
| warm_sequential_same_media | `qwen-mmcache-20260624-2254-json-warm_sequential_same_media-1` | 200 | 0.266 | 0.180 | 1488 | 16 |  |
| warm_sequential_same_media | `qwen-mmcache-20260624-2254-json-warm_sequential_same_media-2` | 200 | 0.252 | 0.170 | 1488 | 16 |  |
| warm_sequential_same_media | `qwen-mmcache-20260624-2254-json-warm_sequential_same_media-3` | 200 | 0.262 | 0.180 | 1488 | 16 |  |
