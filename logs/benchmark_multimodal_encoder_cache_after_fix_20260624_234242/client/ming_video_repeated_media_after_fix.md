# Multimodal Repeated-Media Server Probe

- model: `ming-omni`
- base_url: `http://127.0.0.1:8101`
- media_kind: `video`
- primary_media: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/tests/data/draw.mp4`
- stream: `True`

## Summary

| phase | requests | ok | mean latency (s) | p50 latency (s) | p95 latency (s) | mean first delta (s) |
|---|---:|---:|---:|---:|---:|---:|
| cold_concurrent_same_media | 4 | 4 | 4.796 | 4.815 | 6.782 | 4.678 |
| single_different_media | 1 | 1 | 2.196 | 2.196 | 2.196 | 2.078 |
| warm_sequential_same_media | 4 | 4 | 2.219 | 2.209 | 2.268 | 2.101 |

## Requests

| phase | request_id | status | latency (s) | first delta (s) | prompt tokens | completion tokens | error |
|---|---|---:|---:|---:|---:|---:|---|
| cold_concurrent_same_media | `ming-video-afterfix-20260624-234242-cold_concurrent_same_media-0` | 200 | 6.999 | 6.881 | 1287 | 15 |  |
| cold_concurrent_same_media | `ming-video-afterfix-20260624-234242-cold_concurrent_same_media-1` | 200 | 4.079 | 3.955 | 1287 | 15 |  |
| cold_concurrent_same_media | `ming-video-afterfix-20260624-234242-cold_concurrent_same_media-2` | 200 | 2.557 | 2.445 | 1287 | 15 |  |
| cold_concurrent_same_media | `ming-video-afterfix-20260624-234242-cold_concurrent_same_media-3` | 200 | 5.551 | 5.431 | 1287 | 15 |  |
| single_different_media | `ming-video-afterfix-20260624-234242-single_different_media-0` | 200 | 2.196 | 2.078 | 1287 | 15 |  |
| warm_sequential_same_media | `ming-video-afterfix-20260624-234242-warm_sequential_same_media-0` | 200 | 2.274 | 2.154 | 1287 | 15 |  |
| warm_sequential_same_media | `ming-video-afterfix-20260624-234242-warm_sequential_same_media-1` | 200 | 2.186 | 2.073 | 1287 | 15 |  |
| warm_sequential_same_media | `ming-video-afterfix-20260624-234242-warm_sequential_same_media-2` | 200 | 2.184 | 2.062 | 1287 | 15 |  |
| warm_sequential_same_media | `ming-video-afterfix-20260624-234242-warm_sequential_same_media-3` | 200 | 2.233 | 2.114 | 1287 | 15 |  |
