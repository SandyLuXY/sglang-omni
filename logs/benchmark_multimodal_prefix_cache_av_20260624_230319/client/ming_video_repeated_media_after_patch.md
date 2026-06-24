# Multimodal Repeated-Media Server Probe

- model: `ming-omni`
- base_url: `http://127.0.0.1:8101`
- media_kind: `video`
- primary_media: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/tests/data/draw.mp4`
- stream: `True`

## Summary

| phase | requests | ok | mean latency (s) | p50 latency (s) | p95 latency (s) | mean first delta (s) |
|---|---:|---:|---:|---:|---:|---:|
| cold_concurrent_same_media | 4 | 4 | 7.052 | 6.816 | 7.633 | 6.305 |
| warm_sequential_same_media | 4 | 4 | 2.154 | 2.157 | 2.195 | 2.041 |

## Requests

| phase | request_id | status | latency (s) | first delta (s) | prompt tokens | completion tokens | error |
|---|---|---:|---:|---:|---:|---:|---|
| cold_concurrent_same_media | `ming-video-mmcache-20260624-2312-after-patch-cold_concurrent_same_media-0` | 200 | 6.799 | 5.272 | 1290 | 15 |  |
| cold_concurrent_same_media | `ming-video-mmcache-20260624-2312-after-patch-cold_concurrent_same_media-1` | 200 | 7.775 | 7.655 | 1290 | 15 |  |
| cold_concurrent_same_media | `ming-video-mmcache-20260624-2312-after-patch-cold_concurrent_same_media-2` | 200 | 6.810 | 5.961 | 1290 | 15 |  |
| cold_concurrent_same_media | `ming-video-mmcache-20260624-2312-after-patch-cold_concurrent_same_media-3` | 200 | 6.822 | 6.331 | 1290 | 15 |  |
| warm_sequential_same_media | `ming-video-mmcache-20260624-2312-after-patch-warm_sequential_same_media-0` | 200 | 2.198 | 2.085 | 1290 | 15 |  |
| warm_sequential_same_media | `ming-video-mmcache-20260624-2312-after-patch-warm_sequential_same_media-1` | 200 | 2.139 | 2.026 | 1290 | 15 |  |
| warm_sequential_same_media | `ming-video-mmcache-20260624-2312-after-patch-warm_sequential_same_media-2` | 200 | 2.175 | 2.063 | 1290 | 15 |  |
| warm_sequential_same_media | `ming-video-mmcache-20260624-2312-after-patch-warm_sequential_same_media-3` | 200 | 2.102 | 1.988 | 1290 | 15 |  |
