# Multimodal Repeated-Media Server Probe

- model: `ming-omni`
- base_url: `http://127.0.0.1:8101`
- media_kind: `audio`
- primary_media: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/tests/data/query_to_draw.wav`
- stream: `True`

## Summary

| phase | requests | ok | mean latency (s) | p50 latency (s) | p95 latency (s) | mean first delta (s) |
|---|---:|---:|---:|---:|---:|---:|
| cold_concurrent_same_media | 4 | 4 | 2.225 | 2.225 | 2.226 | 1.797 |
| single_different_media | 1 | 1 | 0.281 | 0.281 | 0.281 | 0.221 |
| warm_sequential_same_media | 4 | 4 | 0.231 | 0.226 | 0.261 | 0.200 |

## Requests

| phase | request_id | status | latency (s) | first delta (s) | prompt tokens | completion tokens | error |
|---|---|---:|---:|---:|---:|---:|---|
| cold_concurrent_same_media | `ming-audio-mmcache-20260624-230319-cold_concurrent_same_media-0` | 200 | 2.226 | 1.862 | 165 | 5 |  |
| cold_concurrent_same_media | `ming-audio-mmcache-20260624-230319-cold_concurrent_same_media-1` | 200 | 2.225 | 1.603 | 165 | 5 |  |
| cold_concurrent_same_media | `ming-audio-mmcache-20260624-230319-cold_concurrent_same_media-2` | 200 | 2.225 | 1.861 | 165 | 5 |  |
| cold_concurrent_same_media | `ming-audio-mmcache-20260624-230319-cold_concurrent_same_media-3` | 200 | 2.225 | 1.861 | 165 | 5 |  |
| single_different_media | `ming-audio-mmcache-20260624-230319-single_different_media-0` | 200 | 0.281 | 0.221 | 154 | 9 |  |
| warm_sequential_same_media | `ming-audio-mmcache-20260624-230319-warm_sequential_same_media-0` | 200 | 0.265 | 0.232 | 165 | 5 |  |
| warm_sequential_same_media | `ming-audio-mmcache-20260624-230319-warm_sequential_same_media-1` | 200 | 0.213 | 0.182 | 165 | 5 |  |
| warm_sequential_same_media | `ming-audio-mmcache-20260624-230319-warm_sequential_same_media-2` | 200 | 0.207 | 0.177 | 165 | 5 |  |
| warm_sequential_same_media | `ming-audio-mmcache-20260624-230319-warm_sequential_same_media-3` | 200 | 0.239 | 0.208 | 165 | 5 |  |
