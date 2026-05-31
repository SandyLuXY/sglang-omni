# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import typer

from sglang_omni.cli.serve import apply_torch_compile_cli_overrides
from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.models.fishaudio_s2_pro.request_builders import (
    S2ProSGLangRequestData,
    apply_tts_result,
    build_sglang_tts_request,
    make_tts_scheduler_adapters,
)
from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
    Reference,
    S2ProTokenizerAdapter,
)
from tests.unit_test.fixtures.fish_fakes import (
    FakeFishTokenizer,
    make_s2pro_payload,
    make_s2pro_state,
)


@pytest.fixture(autouse=True)
def fast_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )


def test_fish_config_state_and_tokenizer_prompt_contracts() -> None:
    """Preserves S2-Pro topology, state tensor round-trip, and prompt VQ layout."""
    config = S2ProPipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"tts_engine": 0, "vocoder": 0}

    state = S2ProState(
        input_ids=torch.tensor([1, 2, 3]),
        vq_mask_tokens=torch.tensor([False, True, False]),
        vq_parts=[torch.tensor([[10, 11], [20, 21]])],
        output_codes=torch.tensor([[100, 101], [1, 2], [3, 4]]),
    )
    restored = S2ProState.from_dict(state.to_dict())
    assert restored.input_ids == [1, 2, 3]
    assert torch.equal(restored.vq_parts[0], torch.tensor([[10, 11], [20, 21]]))
    assert torch.equal(
        restored.output_codes, torch.tensor([[100, 101], [1, 2], [3, 4]])
    )

    tokenizer = FakeFishTokenizer()
    adapter = S2ProTokenizerAdapter(tokenizer)
    prompt = adapter.build_prompt(
        "target",
        references=[
            Reference(
                audio_bytes=b"",
                text="ref",
                vq_codes=torch.tensor([[0, 1], [10, 11]], dtype=torch.long),
            )
        ],
        num_codebooks=2,
        speaker="alice",
    )
    assert adapter.eos_token_ids == [99]
    assert prompt["vq_mask_tokens"].dtype == torch.bool
    assert prompt["vq_mask_tokens"].sum().item() == 2
    assert torch.equal(prompt["vq_parts"][0], torch.tensor([[0, 1], [10, 11]]))
    assert any("<|speaker:alice|>target" in text for text in tokenizer.encoded_texts)


def test_fish_tts_request_and_result_adapters_preserve_tensor_contracts() -> None:
    """Preserves TTS request tensor fields and result adapter output-code shape."""
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(
        input_ids=[10, 11, 12],
        vq_mask_tokens=[False, True, True],
        vq_parts=[[[1, 2], [3, 4]]],
        max_new_tokens=6,
        temperature=0.6,
    )

    req_data = build_sglang_tts_request(state, tokenizer, request_id="req-1")
    assert torch.equal(req_data.input_ids, torch.tensor([10, 11, 12]))
    assert req_data.vq_mask_tokens.dtype == torch.bool
    assert torch.equal(req_data.vq_parts[0], torch.tensor([[1, 2], [3, 4]]))
    assert req_data.req.eos_token_ids == {99}

    req_data.output_codes = [
        torch.tensor([[100], [1], [2]], dtype=torch.long),
        torch.tensor([[101], [3], [4]], dtype=torch.long),
    ]
    apply_tts_result(state, req_data)
    assert torch.equal(
        state.output_codes,
        torch.tensor([[100, 101], [1, 3], [2, 4]], dtype=torch.long),
    )
    assert state.prompt_tokens == 3
    assert state.completion_tokens == 2

    payload = make_s2pro_payload(request_id="req-2")
    request_builder, result_adapter = make_tts_scheduler_adapters(tokenizer=tokenizer)
    adapted = request_builder(payload)
    adapted.output_codes = [torch.tensor([[100], [1], [2]], dtype=torch.long)]
    result_payload = result_adapter(adapted)
    assert adapted.stage_payload is payload
    assert result_payload.request is payload.request
    assert result_payload.data["output_codes"] == [[100], [1], [2]]


@pytest.mark.parametrize("top_k", [0, 31])
def test_fish_tts_rejects_top_k_outside_graph_width(top_k: int) -> None:
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(top_k=top_k)

    with pytest.raises(ValueError, match="S2-Pro top_k must be -1 or between 1 and 30"):
        build_sglang_tts_request(state, tokenizer, request_id="bad-top-k")

    with pytest.raises(ValueError, match="S2-Pro top_k must be -1 or between 1 and 30"):
        S2ProSGLangRequestData(
            input_ids=torch.tensor([], dtype=torch.long),
            req=object(),
            top_k=top_k,
        )


def test_fish_tts_accepts_graph_top_k_width() -> None:
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(top_k=30)

    req_data = build_sglang_tts_request(state, tokenizer, request_id="top-k-30")

    assert req_data.top_k == 30


def test_fish_tts_accepts_default_top_k_sentinel() -> None:
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(top_k=-1)

    req_data = build_sglang_tts_request(state, tokenizer, request_id="top-k-default")

    assert req_data.top_k == -1


def _server_args_overrides(config: S2ProPipelineConfig, name: str) -> dict[str, object]:
    stage = next(stage for stage in config.stages if stage.name == name)
    return dict(stage.factory_args.get("server_args_overrides") or {})


@pytest.mark.parametrize(
    "talker_mode,talker_max_bs,expected",
    [
        ("on", None, {"enable_torch_compile": True}),
        ("off", None, {"enable_torch_compile": False}),
        ("default", 2, {"torch_compile_max_bs": 2}),
        ("on", 4, {"enable_torch_compile": True, "torch_compile_max_bs": 4}),
    ],
)
def test_s2pro_cli_talker_torch_compile_targets_tts_engine(
    talker_mode: str,
    talker_max_bs: int | None,
    expected: dict[str, object],
) -> None:
    config = S2ProPipelineConfig(model_path="model")

    apply_torch_compile_cli_overrides(
        config,
        thinker_torch_compile="default",
        talker_torch_compile=talker_mode,
        thinker_torch_compile_max_bs=None,
        talker_torch_compile_max_bs=talker_max_bs,
    )

    assert _server_args_overrides(config, "tts_engine") == expected
    assert _server_args_overrides(config, "vocoder") == {}


def test_s2pro_cli_talker_torch_compile_default_is_noop() -> None:
    config = S2ProPipelineConfig(model_path="model")

    apply_torch_compile_cli_overrides(
        config,
        thinker_torch_compile="default",
        talker_torch_compile="default",
        thinker_torch_compile_max_bs=None,
        talker_torch_compile_max_bs=None,
    )

    assert _server_args_overrides(config, "tts_engine") == {}


def test_s2pro_cli_talker_torch_compile_max_bs_rejects_non_positive() -> None:
    config = S2ProPipelineConfig(model_path="model")

    with pytest.raises(
        typer.BadParameter,
        match="torch compile max batch size must be >= 1",
    ):
        apply_torch_compile_cli_overrides(
            config,
            thinker_torch_compile="default",
            talker_torch_compile="default",
            thinker_torch_compile_max_bs=None,
            talker_torch_compile_max_bs=0,
        )


def test_s2pro_compile_helper_targets_forward_kvcached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", "/tmp")
    stages = importlib.import_module("sglang_omni.models.fishaudio_s2_pro.stages")

    fake_runner = ModuleType("sglang.srt.model_executor.cuda_graph_runner")
    fake_runner.set_torch_compile_config = lambda: None
    monkeypatch.setitem(
        sys.modules, "sglang.srt.model_executor.cuda_graph_runner", fake_runner
    )

    compile_calls: list[tuple[object, str | None, dict[str, object]]] = []

    def fake_compile(
        target: object, *, mode: str | None = None, **kwargs: object
    ) -> object:
        compile_calls.append((target, mode, kwargs))
        return f"compiled-{len(compile_calls)}"

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setenv("SGLANG_TORCH_COMPILE_MODE", "reduce-overhead")

    class _Layer:
        def forward_kvcached(
            self, x: torch.Tensor, freqs_cis: torch.Tensor, cache_seqlens: torch.Tensor
        ) -> torch.Tensor:
            del freqs_cis, cache_seqlens
            return x

    class _AudioDecoder:
        def __init__(self) -> None:
            self.layers = [_Layer()]

        def compile_forward_kvcached_layers(self, *, mode: str) -> int:
            self._forward_kvcached_layers = [
                torch.compile(layer.forward_kvcached, mode=mode)
                for layer in self.layers
            ]
            return len(self._forward_kvcached_layers)

    audio_decoder = _AudioDecoder()
    model = SimpleNamespace(_audio_decoder=audio_decoder)

    stages._compile_s2pro_codebook_decoder(model)

    assert len(compile_calls) == 1
    target, mode, kwargs = compile_calls[0]
    assert getattr(target, "__self__", None) is audio_decoder.layers[0]
    assert getattr(target, "__name__", "") == "forward_kvcached"
    assert mode == "reduce-overhead"
    assert kwargs == {}
    assert audio_decoder._forward_kvcached_layers == ["compiled-1"]


def test_decoder_forward_kvcached_uses_compiled_callables_when_present() -> None:
    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling import (
        FishQwen3AudioDecoder,
    )

    class _EagerLayer:
        def forward_kvcached(
            self, x: torch.Tensor, freqs_cis: torch.Tensor, cache_seqlens: torch.Tensor
        ) -> torch.Tensor:
            del x, freqs_cis, cache_seqlens
            raise AssertionError("eager layer path should be bypassed")

    decoder = object.__new__(FishQwen3AudioDecoder)
    decoder.input_pos = torch.zeros(1, dtype=torch.long)
    decoder.freqs_cis = torch.zeros(8, 1, 1, dtype=torch.float32)
    decoder.layers = [_EagerLayer(), _EagerLayer()]
    decoder.norm = lambda x: x + 1
    decoder.output = lambda x: x * 2

    seen_calls: list[str] = []

    def compiled_a(
        x: torch.Tensor, freqs_cis: torch.Tensor, cache_seqlens: torch.Tensor
    ) -> torch.Tensor:
        del freqs_cis, cache_seqlens
        seen_calls.append("a")
        return x + 2

    def compiled_b(
        x: torch.Tensor, freqs_cis: torch.Tensor, cache_seqlens: torch.Tensor
    ) -> torch.Tensor:
        del freqs_cis, cache_seqlens
        seen_calls.append("b")
        return x + 3

    decoder._forward_kvcached_layers = [compiled_a, compiled_b]

    x = torch.ones((2, 1, 4), dtype=torch.float32)
    out = FishQwen3AudioDecoder.forward_kvcached(decoder, x=x, codebook_idx=2)

    assert torch.equal(out, torch.full_like(x, 14.0))
    assert seen_calls == ["a", "b"]
