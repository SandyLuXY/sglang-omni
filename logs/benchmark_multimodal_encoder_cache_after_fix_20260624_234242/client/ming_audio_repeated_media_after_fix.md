# Multimodal Repeated-Media Server Probe

- model: `ming-omni`
- base_url: `http://127.0.0.1:8101`
- media_kind: `audio`
- primary_media: `/data/sglang-omni-worktrees/feat-omni-multimodal-prefix-cache/tests/data/query_to_draw.wav`
- stream: `True`

## Summary

| phase | requests | ok | mean latency (s) | p50 latency (s) | p95 latency (s) | mean first delta (s) |
|---|---:|---:|---:|---:|---:|---:|
| cold_concurrent_same_media | 4 | 4 | 2.167 | 2.167 | 2.169 | 1.734 |
| single_different_media | 1 | 1 | 0.276 | 0.276 | 0.276 | 0.215 |
| warm_sequential_same_media | 4 | 4 | 0.223 | 0.211 | 0.256 | 0.191 |

## Requests

| phase | request_id | status | latency (s) | first delta (s) | prompt tokens | completion tokens | error |
|---|---|---:|---:|---:|---:|---:|---|
| cold_concurrent_same_media | `ming-audio-afterfix-20260624-234242-cold_concurrent_same_media-0` | 200 | 2.168 | 1.801 | 165 | 5 |  |
| cold_concurrent_same_media | `ming-audio-afterfix-20260624-234242-cold_concurrent_same_media-1` | 200 | 2.167 | 1.800 | 165 | 5 |  |
| cold_concurrent_same_media | `ming-audio-afterfix-20260624-234242-cold_concurrent_same_media-2` | 200 | 2.166 | 1.536 | 165 | 5 |  |
| cold_concurrent_same_media | `ming-audio-afterfix-20260624-234242-cold_concurrent_same_media-3` | 200 | 2.169 | 1.800 | 165 | 5 |  |
| single_different_media | `ming-audio-afterfix-20260624-234242-single_different_media-0` | 200 | 0.276 | 0.215 | 154 | 9 |  |
| warm_sequential_same_media | `ming-audio-afterfix-20260624-234242-warm_sequential_same_media-0` | 200 | 0.264 | 0.229 | 165 | 5 |  |
| warm_sequential_same_media | `ming-audio-afterfix-20260624-234242-warm_sequential_same_media-1` | 200 | 0.213 | 0.182 | 165 | 5 |  |
| warm_sequential_same_media | `ming-audio-afterfix-20260624-234242-warm_sequential_same_media-2` | 200 | 0.210 | 0.179 | 165 | 5 |  |
| warm_sequential_same_media | `ming-audio-afterfix-20260624-234242-warm_sequential_same_media-3` | 200 | 0.206 | 0.175 | 165 | 5 |  |
