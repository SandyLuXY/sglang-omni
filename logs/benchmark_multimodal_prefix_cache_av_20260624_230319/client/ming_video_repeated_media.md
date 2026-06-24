# Multimodal Repeated-Media Server Probe

- model: `ming-omni`
- base_url: `http://127.0.0.1:8101`
- media_kind: `video`
- primary_media: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/tests/data/draw.mp4`
- stream: `True`

## Summary

| phase | requests | ok | mean latency (s) | p50 latency (s) | p95 latency (s) | mean first delta (s) |
|---|---:|---:|---:|---:|---:|---:|
| cold_concurrent_same_media | 4 | 0 |  |  |  |  |
| warm_sequential_same_media | 4 | 0 |  |  |  |  |

## Requests

| phase | request_id | status | latency (s) | first delta (s) | prompt tokens | completion tokens | error |
|---|---|---:|---:|---:|---:|---:|---|
| cold_concurrent_same_media | `ming-video-mmcache-20260624-230319-cold_concurrent_same_media-0` | 200 | 5.853 |  |  |  | peer closed connection without sending complete message body (incomplete chunked read) |
| cold_concurrent_same_media | `ming-video-mmcache-20260624-230319-cold_concurrent_same_media-1` | 200 | 4.235 |  |  |  | peer closed connection without sending complete message body (incomplete chunked read) |
| cold_concurrent_same_media | `ming-video-mmcache-20260624-230319-cold_concurrent_same_media-2` | 200 | 2.792 |  |  |  | peer closed connection without sending complete message body (incomplete chunked read) |
| cold_concurrent_same_media | `ming-video-mmcache-20260624-230319-cold_concurrent_same_media-3` | 200 | 7.440 |  |  |  | peer closed connection without sending complete message body (incomplete chunked read) |
| warm_sequential_same_media | `ming-video-mmcache-20260624-230319-warm_sequential_same_media-0` | 200 | 1.469 |  |  |  | peer closed connection without sending complete message body (incomplete chunked read) |
| warm_sequential_same_media | `ming-video-mmcache-20260624-230319-warm_sequential_same_media-1` | 200 | 1.597 |  |  |  | peer closed connection without sending complete message body (incomplete chunked read) |
| warm_sequential_same_media | `ming-video-mmcache-20260624-230319-warm_sequential_same_media-2` | 200 | 1.582 |  |  |  | peer closed connection without sending complete message body (incomplete chunked read) |
| warm_sequential_same_media | `ming-video-mmcache-20260624-230319-warm_sequential_same_media-3` | 200 | 1.436 |  |  |  | peer closed connection without sending complete message body (incomplete chunked read) |
