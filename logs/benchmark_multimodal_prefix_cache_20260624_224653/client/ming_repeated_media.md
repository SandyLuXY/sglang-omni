# Multimodal Repeated-Media Server Probe

- model: `ming-omni`
- base_url: `http://127.0.0.1:8101`
- media_kind: `image`
- primary_media: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/docs/_static/image/llada2.0_uni_architecture.png`
- stream: `True`

## Summary

| phase | requests | ok | mean latency (s) | p50 latency (s) | p95 latency (s) | mean first delta (s) |
|---|---:|---:|---:|---:|---:|---:|
| cold_concurrent_same_media | 4 | 4 | 5.707 | 5.705 | 5.720 | 4.948 |
| single_different_media | 1 | 1 | 0.883 | 0.883 | 0.883 | 0.767 |
| warm_sequential_same_media | 4 | 4 | 0.612 | 0.601 | 0.660 | 0.496 |

## Requests

| phase | request_id | status | latency (s) | first delta (s) | prompt tokens | completion tokens | error |
|---|---|---:|---:|---:|---:|---:|---|
| cold_concurrent_same_media | `ming-mmcache-20260624-224653-cold_concurrent_same_media-0` | 200 | 5.721 | 5.238 | 986 | 16 |  |
| cold_concurrent_same_media | `ming-mmcache-20260624-224653-cold_concurrent_same_media-1` | 200 | 5.696 | 4.082 | 986 | 16 |  |
| cold_concurrent_same_media | `ming-mmcache-20260624-224653-cold_concurrent_same_media-2` | 200 | 5.710 | 5.237 | 986 | 16 |  |
| cold_concurrent_same_media | `ming-mmcache-20260624-224653-cold_concurrent_same_media-3` | 200 | 5.699 | 5.236 | 986 | 16 |  |
| single_different_media | `ming-mmcache-20260624-224653-single_different_media-0` | 200 | 0.883 | 0.767 | 960 | 16 |  |
| warm_sequential_same_media | `ming-mmcache-20260624-224653-warm_sequential_same_media-0` | 200 | 0.668 | 0.549 | 986 | 16 |  |
| warm_sequential_same_media | `ming-mmcache-20260624-224653-warm_sequential_same_media-1` | 200 | 0.614 | 0.499 | 986 | 16 |  |
| warm_sequential_same_media | `ming-mmcache-20260624-224653-warm_sequential_same_media-2` | 200 | 0.576 | 0.461 | 986 | 16 |  |
| warm_sequential_same_media | `ming-mmcache-20260624-224653-warm_sequential_same_media-3` | 200 | 0.589 | 0.476 | 986 | 16 |  |
