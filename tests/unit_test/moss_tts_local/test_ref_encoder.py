# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import sglang_omni.models.moss_tts_local.vocoder_decoder as vocoder_decoder
from sglang_omni.models.moss_tts_local.ref_encoder import MossTTSLocalRefEncoder
from sglang_omni.models.moss_tts_local.vocoder_decoder import (
    MossTTSLocalAttention,
    MossTTSLocalProjectedTransformer,
)


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    *,
    max_period: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    del max_period
    return torch.zeros(*positions.shape, dim, device=positions.device, dtype=dtype)


class _FakeLayerScale(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class _FakeAttention(nn.Module):
    def __init__(self, hidden_size: int, *, num_heads: int = 2) -> None:
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads
        self.causal = True
        self.context = 4
        self.rope = None
        self.attention_implementation = "sdpa"
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def resolve_attention_implementation(
        self, _: torch.Tensor, *, is_streaming: bool = False
    ) -> str:
        return "sdpa"

    def _get_backend_check_dtype(self, x: torch.Tensor) -> torch.dtype:
        return x.dtype

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return x


class _ReferenceAttention(_FakeAttention):
    def forward(self, x: torch.Tensor, *, input_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, max_seqlen, _ = x.shape
        projected = self.in_proj(x).reshape(
            batch_size, max_seqlen, 3, self.num_heads, self.head_dim
        )
        q, k, v = projected.permute(2, 0, 3, 1, 4)
        positions = torch.arange(max_seqlen, device=x.device, dtype=torch.long)
        valid_k = positions.view(1, 1, max_seqlen) < input_lengths.view(-1, 1, 1)
        delta = positions.view(1, max_seqlen, 1) - positions.view(1, 1, max_seqlen)
        attn_bias = (delta >= 0) & (delta < int(self.context))
        attn_bias = (attn_bias & valid_k)[:, None, :, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
        valid_q = positions.view(1, max_seqlen) < input_lengths.view(-1, 1)
        out = torch.where(
            valid_q.view(batch_size, 1, max_seqlen, 1),
            out,
            torch.zeros((), device=x.device, dtype=x.dtype),
        )
        out = out.transpose(1, 2).reshape(batch_size, max_seqlen, self.embed_dim)
        return self.out_proj(out)


class _FakeLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = _FakeAttention(hidden_size)
        self.layer_scale_1 = _FakeLayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.layer_scale_2 = _FakeLayerScale(hidden_size)


class _FallbackTransformer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(hidden_size)])
        self.positional_embedding = "rope"
        self.positional_scale = 1.0
        self.max_period = 10000.0

    def resolve_attention_implementation(self, _: torch.Tensor) -> str:
        return "sdpa"


class _FallbackProjectedStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(3, 6)
        self.transformer = _FallbackTransformer(6)
        self.output_proj = nn.Linear(6, 7)
        self.module_type = "Transformer"
        self.seen_input_shape: tuple[int, ...] | None = None

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen_input_shape = tuple(x.shape)
        return x + 10, input_lengths + 1


class _PatchStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module_type = "PatchedPretransform"
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append((tuple(x.shape), tuple(input_lengths.shape)))
        return x, input_lengths


def _wrapped_projected_encoder() -> tuple[
    MossTTSLocalRefEncoder, _FallbackProjectedStage
]:
    source = _FallbackProjectedStage()
    source.transformer.layers[0].self_attn.attention_implementation = (
        "flash_attention_2"
    )
    wrapped = MossTTSLocalRefEncoder(nn.ModuleList([source]))
    return wrapped, source


def test_ref_encoder_wraps_supported_stage_types() -> None:
    patch_stage = _PatchStage()
    source = nn.ModuleList([patch_stage, _FallbackProjectedStage()])
    wrapped = MossTTSLocalRefEncoder(source)

    assert len(wrapped) == 2
    assert wrapped[0] is patch_stage
    assert isinstance(wrapped[1], MossTTSLocalProjectedTransformer)
    assert list(iter(wrapped)) == [wrapped[0], wrapped[1]]


def test_ref_encoder_uses_sglang_packed_flash_path() -> None:
    wrapped, source = _wrapped_projected_encoder()
    attn = wrapped[0].transformer.layers[0].self_attn
    attn._can_run_packed_flash = lambda _: True  # type: ignore[method-assign]
    calls = []

    def fake_flash_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        calls.append((cu_q.clone(), cu_k.clone(), max_q, max_k, window_size))
        return q

    attn._flash_attn_varlen = fake_flash_attn
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 3])

    out, out_lengths = wrapped(x, lengths)

    assert source.seen_input_shape is None
    assert len(calls) == 1
    cu_q, cu_k, max_q, max_k, window_size = calls[0]
    assert cu_q.tolist() == [0, 4, 7]
    assert cu_k.tolist() == [0, 4, 7]
    assert max_q == 4
    assert max_k == 4
    assert window_size == (source.transformer.layers[0].self_attn.context - 1, 0)
    assert out.shape == (2, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_ref_encoder_packed_flash_unavailable_uses_source_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vocoder_decoder, "flash_attn_varlen_func", None)
    wrapped, source = _wrapped_projected_encoder()
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 3])

    out, out_lengths = wrapped(x, lengths)

    assert wrapped[0].transformer.layers[0].self_attn._flash_attn_varlen is None
    assert source.seen_input_shape is None
    assert out.shape == (2, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_ref_encoder_skips_flash_for_zero_valid_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped, _ = _wrapped_projected_encoder()
    attn = wrapped[0].transformer.layers[0].self_attn
    attn._can_run_packed_flash = lambda _: True  # type: ignore[method-assign]

    def fail_flash(*_: object, **__: object) -> None:
        raise AssertionError("zero-length input must not call flash attention")

    def fail_pack(*_: object) -> None:
        raise AssertionError("zero-length input must not pack padded frames")

    attn._flash_attn_varlen = fail_flash
    monkeypatch.setattr(vocoder_decoder, "_pack_padded_sequence", fail_pack)
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([0, 0])

    out, out_lengths = wrapped(x, lengths)

    assert out.shape == (2, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_ref_encoder_uses_single_unpadded_pack_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped, _ = _wrapped_projected_encoder()
    attn = wrapped[0].transformer.layers[0].self_attn
    attn._can_run_packed_flash = lambda _: True  # type: ignore[method-assign]
    calls = []

    def fail_masked_pack(_: torch.Tensor, __: torch.Tensor) -> None:
        raise AssertionError("single unpadded input should not use masked pack")

    def fake_flash_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        calls.append((q.shape, cu_q.clone(), cu_k.clone(), max_q, max_k))
        return q

    monkeypatch.setattr(vocoder_decoder, "_pack_padded_sequence", fail_masked_pack)
    attn._flash_attn_varlen = fake_flash_attn
    x = torch.randn(1, 3, 4)
    lengths = torch.tensor([4])

    out, out_lengths = wrapped(x, lengths)
    _ = wrapped(x, lengths)

    assert len(calls) == 2
    q_shape, cu_q, cu_k, max_q, max_k = calls[0]
    assert q_shape[0] == 4
    assert cu_q.tolist() == [0, 4]
    assert cu_k.tolist() == [0, 4]
    assert calls[1][1].tolist() == [0, 4]
    assert calls[1][2].tolist() == [0, 4]
    assert max_q == 4
    assert max_k == 4
    assert out.shape == (1, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_ref_encoder_uses_padded_batch_pack_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped, _ = _wrapped_projected_encoder()
    attn = wrapped[0].transformer.layers[0].self_attn
    attn._can_run_packed_flash = lambda _: True  # type: ignore[method-assign]
    original_pack = vocoder_decoder._pack_padded_sequence
    pack_calls = []
    flash_calls = []

    def spy_pack(
        x: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pack_calls.append((tuple(x.shape), input_lengths.clone()))
        return original_pack(x, input_lengths)

    def fake_flash_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        flash_calls.append((q.shape, cu_q.clone(), cu_k.clone(), max_q, max_k))
        return q

    monkeypatch.setattr(vocoder_decoder, "_pack_padded_sequence", spy_pack)
    attn._flash_attn_varlen = fake_flash_attn
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 2])

    out, out_lengths = wrapped(x, lengths)

    assert len(pack_calls) == 1
    assert pack_calls[0][0] == (2, 4, 6)
    assert pack_calls[0][1].tolist() == [4, 2]
    assert len(flash_calls) == 1
    q_shape, cu_q, cu_k, max_q, max_k = flash_calls[0]
    assert q_shape[0] == 6
    assert cu_q.tolist() == [0, 4, 6]
    assert cu_k.tolist() == [0, 4, 6]
    assert max_q == 4
    assert max_k == 4
    assert out.shape == (2, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_ref_encoder_sglang_packed_flash_matches_sdpa_reference_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if vocoder_decoder.flash_attn_varlen_func is None:
        pytest.skip("requires SGLang flash_attn_varlen_func")

    torch.manual_seed(0)
    device = torch.device("cuda")
    source = _ReferenceAttention(hidden_size=128, num_heads=2).to(
        device=device, dtype=torch.bfloat16
    )
    source.attention_implementation = "flash_attention_2"
    wrapper = MossTTSLocalAttention(source)
    x = torch.randn(2, 6, 128, device=device, dtype=torch.bfloat16)
    input_lengths = torch.tensor([6, 4], device=device)

    packed_x, valid_mask, cu_seqlens, position_ids = (
        vocoder_decoder._pack_padded_sequence(x, input_lengths)
    )
    packed_out = wrapper(
        packed_x,
        cu_seqlens=cu_seqlens,
        max_seqlen=6,
        position_ids=position_ids,
    )
    flash_out = vocoder_decoder._unpack_packed_sequence(
        packed_out, valid_mask, batch_size=2, max_seqlen=6
    )
    sdpa_out = source(x, input_lengths=input_lengths)

    torch.testing.assert_close(flash_out, sdpa_out, atol=4e-2, rtol=3e-2)


def test_ref_encoder_forward_chains_all_stages() -> None:
    first_patch = _PatchStage()
    second_patch = _PatchStage()
    transformer_stage = _FallbackProjectedStage()
    wrapped = MossTTSLocalRefEncoder(
        nn.ModuleList([first_patch, transformer_stage, second_patch])
    )
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 3])

    out, out_lengths = wrapped(x, lengths)

    assert first_patch.calls == [((2, 3, 4), (2,))]
    assert second_patch.calls == [((2, 7, 4), (2,))]
    assert transformer_stage.seen_input_shape is None
    assert out.shape == (2, 7, 4)
    assert torch.equal(out_lengths, lengths)

