# Qwen First Client Attempt Failure

- Time window: approximately `2026-06-24T22:53:28Z` to `2026-06-24T22:53:35Z`
- Server: Qwen3-Omni on `http://127.0.0.1:8100`
- Request prefix: `qwen-mmcache-20260624-224653`
- Command: recorded in the first line of `commands/qwen_client.sh`

The client successfully sent:

- 4 concurrent same-image requests using `tests/data/cars.jpg`
- 4 warm sequential same-image requests using `tests/data/cars.jpg`
- 1 different-image request using `docs/_static/image/logo.png`

The client then failed while writing JSON:

```text
TypeError: Object of type PosixPath is not JSON serializable
```

The benchmark script was fixed to stringify `Path` values in the output config,
then rerun with uncached images:

- primary: `docs/_static/image/higgs-architecture.png`
- different-image: `docs/_static/image/moss-tts-arch-local.png`

The first attempt is still useful as server-side evidence. See
`server/qwen_server.log` request IDs with prefix `qwen-mmcache-20260624-224653`.
Those logs show one `image_encoder` miss/store, same-media hits, warm hits, and
a different-media miss/store.
