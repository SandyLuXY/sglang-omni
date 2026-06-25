# Multimodal Repeated-Media Server Probe

- model: `ming-omni`
- base_url: `http://127.0.0.1:8101`
- media_kind: `image`
- primary_media: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/docs/_static/image/llada2.0_uni_architecture.png`
- stream: `True`

## Summary

| phase | requests | ok | mean latency (s) | p50 latency (s) | p95 latency (s) | mean first delta (s) |
|---|---:|---:|---:|---:|---:|---:|
| cold_concurrent_same_media | 4 | 4 | 4.256 | 4.256 | 4.271 | 3.871 |
| single_different_media | 1 | 1 | 0.705 | 0.705 | 0.705 | 0.586 |
| warm_sequential_same_media | 4 | 4 | 0.679 | 0.678 | 0.689 | 0.558 |

## Requests

| phase | request_id | status | latency (s) | first delta (s) | prompt tokens | completion tokens | error |
|---|---|---:|---:|---:|---:|---:|---|
| cold_concurrent_same_media | `ming-image-afterfix-20260624-234242-cold_concurrent_same_media-0` | 200 | 4.272 | 4.194 | 983 | 16 |  |
| cold_concurrent_same_media | `ming-image-afterfix-20260624-234242-cold_concurrent_same_media-1` | 200 | 4.251 | 4.192 | 983 | 16 |  |
| cold_concurrent_same_media | `ming-image-afterfix-20260624-234242-cold_concurrent_same_media-2` | 200 | 4.261 | 4.193 | 983 | 16 |  |
| cold_concurrent_same_media | `ming-image-afterfix-20260624-234242-cold_concurrent_same_media-3` | 200 | 4.239 | 2.906 | 983 | 16 |  |
| single_different_media | `ming-image-afterfix-20260624-234242-single_different_media-0` | 200 | 0.705 | 0.586 | 957 | 16 |  |
| warm_sequential_same_media | `ming-image-afterfix-20260624-234242-warm_sequential_same_media-0` | 200 | 0.671 | 0.550 | 983 | 16 |  |
| warm_sequential_same_media | `ming-image-afterfix-20260624-234242-warm_sequential_same_media-1` | 200 | 0.690 | 0.566 | 983 | 16 |  |
| warm_sequential_same_media | `ming-image-afterfix-20260624-234242-warm_sequential_same_media-2` | 200 | 0.685 | 0.565 | 983 | 16 |  |
| warm_sequential_same_media | `ming-image-afterfix-20260624-234242-warm_sequential_same_media-3` | 200 | 0.671 | 0.552 | 983 | 16 |  |
