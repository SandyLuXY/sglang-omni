# [Perf] Batch MOSS-TTS Local reference audio encode

## Motivation

MOSS-TTS Local voice-cloning requests repeatedly encode reference audio before generation. Under concurrent serving, those independent reference encodes can bottleneck preprocessing even though the processor can batch the path-based reference encode work. This PR batches compatible reference paths, preserves per-request fallback behavior, and keeps the existing cache protections intact.

## Modifications

1. `sglang_omni/models/moss_tts_local/stages.py`
   - Extends `_BatchedReferenceEncoder` so concurrent `encode()` calls are drained into a single reference encode batch up to `max_batch_size`.
   - Prefers the processor's `encode_audios_from_path` API, which already handles path loading, resampling, padding, and batched codec encode for mixed reference lengths.
   - Deduplicates repeated paths within a drained batch while preserving each caller's result ordering.
   - Validates batched result counts and falls back to per-path encode so malformed batch output or one bad reference does not fail unrelated requests.
   - Keeps waveform encode support as a fallback for processors without the path API, including target sample-rate discovery and per-item retry after a waveform batch failure.
   - Memoizes the reference-duration gate by stable content key inside `CachedReferenceEncoder`, while still checking duration before cache or inflight reuse for uncached content.

2. `sglang_omni/models/moss_tts_local/config.py`
   - Raises `ref_audio_cache_max_items` from `256` to `1024` for the local MOSS-TTS preprocessing stage.

3. `tests/unit_test/moss_tts_local/test_pipeline.py`
   - Adds coverage for stage wiring, concurrent batch coalescing, duplicate-path deduplication, order preservation, malformed batch output fallback, no-path waveform fallback, and path-preferred behavior when both APIs are present.
   - Covers cache-duration memoization by content key so overwritten references are rechecked while repeated identical references avoid extra `torchaudio.info` calls.

## Related Issues

N/A

## Accuracy Test

**Setup:**

- Compared refs: baseline `main @ c8ea7ad`; candidate `feat/moss-local-batched-audio-encode` worktree based on `0b551f8` with the modified files recorded in `env/current_environment.txt`.
- Artifact path: `logs/moss-local-batched-audio-encode-20260612-012805-c1-c16-r2`.
- Model: `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`, cached snapshot `748406af29543d845632c6ba0fc060b295f4cc81`.
- Server: `python -m sglang_omni.cli serve --config examples/configs/moss_tts_local.yaml --host 127.0.0.1 --port 18080/18082` for baseline/candidate runs.
- Dataset/eval: `zhaochenyang20/seed-tts-eval-arrow`, `lang=en`, 1088 samples / 666 unique audio files, `--ref-format references`, `--token-count auto`.
- ASR quality: `Qwen/Qwen3-ASR-1.7B` with ASR concurrency `32`; speaker similarity ran from the generated audio outputs.
- Hardware: `CUDA_VISIBLE_DEVICES=4,5`, `2x H100` NVIDIA H100 80GB HBM3, driver `580.126.20`, CUDA `13.0`; benchmark preflights confirmed the selected GPUs were free before the completed runs.

Lower WER is better; higher speaker similarity is better. Rows below are mean +/- sample std across two repeats. All rows completed with `quality_status=ok`, `1088/1088` WER samples evaluated, `1088/1088` similarity samples evaluated, and zero failed requests.

| concurrency | main WER | current WER | main speaker similarity | current speaker similarity | failed requests |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.26% +/- 0.24% | 2.53% +/- 0.49% | 65.46 +/- 0.12 | 65.38 +/- 0.24 | 0 |
| 16 | 2.03% +/- 0.12% | 2.26% +/- 0.02% | 65.17 +/- 0.35 | 65.41 +/- 0.11 | 0 |

Note: this latest benchmark-only log root skipped the runner's unit-test phase.

## Benchmark & Profiling

Same setup as described in the _Accuracy Test_ section. Benchmark ran concurrencies `1` and `16`, warmup `8`, two repeats per branch, and wrote the consolidated CSV to `logs/moss-local-batched-audio-encode-20260612-012805-c1-c16-r2/comparison/results.csv`.

Higher audio s/s is better; lower RTF and latency are better. Rows below are mean +/- sample std across two repeats.

| concurrency | main audio s/s | current audio s/s | delta | main RTF | current RTF | latency delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.026 +/- 0.132 | 7.120 +/- 0.029 | +1.4% | 0.1451 +/- 0.0028 | 0.1430 +/- 0.0006 | -1.4% |
| 16 | 40.326 +/- 2.498 | 49.669 +/- 6.544 | +23.2% | 0.4149 +/- 0.0256 | 0.3380 +/- 0.0460 | -18.3% |

### Summary:

- The latest c1/c16 repeated benchmark passes the throughput gate at both tested concurrencies.
- c1 is effectively flat-to-slightly-faster at `+1.4%` audio throughput with similar quality metrics.
- c16 improves audio throughput by `+23.2%` and reduces mean RTF by `18.5%`; quality remains close, with zero failed requests.

## Checklist

- [x] Format your code according with pre-commit.
- [x] Add unit tests.
- [ ] Update documentation / docstrings / example tutorials as needed.
- [x] Provide throughput / latency benchmark results and accuracy evaluation results as needed.
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.

## CI

CI runs on self-hosted GPU runners and requires a maintainer to add the
`run-ci` label. Once labeled, every subsequent push re-triggers CI as
long as the label remains. Draft PRs are skipped even if labeled.
