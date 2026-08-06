# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sglang_omni.models.qwen3_asr.request_builders import (
    make_qwen3_asr_stream_output_builder,
)
from sglang_omni.proto import OmniRequest, StagePayload

_EOS = 999


class _ByteTokenizer:
    eos_token_id = _EOS

    def __init__(
        self,
        vocab: dict[int, bytes],
        special_token_ids: set[int] | None = None,
    ) -> None:
        self._vocab = vocab
        self._special = special_token_ids or set()

    def decode(
        self,
        ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        chunks = [
            self._vocab[token_id]
            for token_id in ids
            if not (skip_special_tokens and token_id in self._special)
        ]
        return b"".join(chunks).decode("utf-8", errors="replace")


def _make_req_data(
    *,
    stream: bool = True,
    inflight_middle_chunks: int = 0,
    finished: bool = False,
) -> Any:
    stage_payload = StagePayload(
        request_id="r",
        request=OmniRequest(
            inputs={"audio_bytes": b""},
            params={"stream": stream},
            metadata={},
        ),
        data={},
    )
    req = SimpleNamespace(
        inflight_middle_chunks=inflight_middle_chunks,
        finished=lambda: finished,
    )
    return SimpleNamespace(req=req, stage_payload=stage_payload)


def _make_req_output(token_id: int | None) -> Any:
    return SimpleNamespace(data=token_id)


def _builder(
    vocab: dict[int, bytes],
    *,
    interval_s: float = 0.0,
    special: set[int] | None = None,
):
    return make_qwen3_asr_stream_output_builder(
        tokenizer=_ByteTokenizer(vocab, special_token_ids=special),
        min_emit_interval_s=interval_s,
    )


def test_qwen3_asr_stream_emits_text_delta() -> None:
    builder = _builder({1: b"hello"})
    req_data = _make_req_data()

    messages = builder("req-1", req_data, _make_req_output(1))

    assert len(messages) == 1
    message = messages[0]
    assert message.request_id == "req-1"
    assert message.type == "stream"
    assert message.target is None
    assert message.data == {
        "text": "hello",
        "modality": "text",
        "stage_name": "asr",
    }
    assert message.metadata == {"modality": "text", "token_id": 1}


def test_qwen3_asr_stream_gates_prefill_and_non_streaming_requests() -> None:
    builder = _builder({1: b"A"})
    prefill_data = _make_req_data(inflight_middle_chunks=1)
    non_stream_data = _make_req_data(stream=False)

    assert builder("prefill", prefill_data, _make_req_output(1)) == []
    assert builder("non-stream", non_stream_data, _make_req_output(1)) == []
    assert not hasattr(prefill_data.req, "_qwen3_asr_stream_pending_ids")
    assert not hasattr(non_stream_data.req, "_qwen3_asr_stream_pending_ids")


def test_qwen3_asr_stream_rate_limit_holds_then_eos_flushes() -> None:
    builder = _builder(
        {1: b"A", 2: b"B", 3: b"C", _EOS: b"<eos>"},
        interval_s=3600.0,
    )
    req_data = _make_req_data()

    assert [
        message.data["text"] for message in builder("r", req_data, _make_req_output(1))
    ] == ["A"]
    assert builder("r", req_data, _make_req_output(2)) == []
    assert builder("r", req_data, _make_req_output(3)) == []

    messages = builder("r", req_data, _make_req_output(_EOS))

    assert [message.data["text"] for message in messages] == ["BC"]
    assert messages[0].metadata == {"modality": "text", "token_id": _EOS}


def test_qwen3_asr_stream_flush_hook_emits_non_eos_terminal_tail() -> None:
    builder = _builder({1: b"A", 2: b"B"}, interval_s=3600.0)
    req_data = _make_req_data()

    assert [
        message.data["text"] for message in builder("r", req_data, _make_req_output(1))
    ] == ["A"]
    assert builder("r", req_data, _make_req_output(2)) == []

    req_data.req.finished = lambda: True
    messages = builder.flush("r", req_data)

    assert [message.data["text"] for message in messages] == ["B"]
    assert messages[0].metadata == {"modality": "text", "token_id": None}


def test_qwen3_asr_stream_holds_incomplete_utf8_until_complete() -> None:
    builder = _builder({1: b"\xe4", 2: b"\xbd", 3: b"\xa0"})
    req_data = _make_req_data()

    assert builder("r", req_data, _make_req_output(1)) == []
    assert builder("r", req_data, _make_req_output(2)) == []
    messages = builder("r", req_data, _make_req_output(3))

    assert [message.data["text"] for message in messages] == ["你"]


def test_qwen3_asr_stream_terminal_flush_emits_incomplete_utf8_replacement() -> None:
    builder = _builder({1: b"\xe4"}, interval_s=3600.0)
    req_data = _make_req_data()

    assert builder("r", req_data, _make_req_output(1)) == []

    req_data.req.finished = lambda: True
    messages = builder.flush("r", req_data)

    assert [message.data["text"] for message in messages] == ["\ufffd"]


def test_qwen3_asr_stream_suppresses_special_tokens() -> None:
    builder = _builder(
        {1: b"hello", 2: b"<special>"},
        special={2},
    )
    req_data = _make_req_data()

    assert [
        message.data["text"] for message in builder("r", req_data, _make_req_output(1))
    ] == ["hello"]
    assert builder("r", req_data, _make_req_output(2)) == []


def test_qwen3_asr_stream_explicit_eos_overrides_tokenizer() -> None:
    builder = make_qwen3_asr_stream_output_builder(
        tokenizer=_ByteTokenizer({1: b"A", 7: b"<stop>"}),
        eos_token_id=7,
    )
    req_data = _make_req_data()

    assert [
        message.data["text"] for message in builder("r", req_data, _make_req_output(1))
    ] == ["A"]
    assert builder("r", req_data, _make_req_output(7)) == []
