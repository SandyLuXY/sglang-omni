# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.pipeline_state import PipelineStateBase
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from tests.unit_test.pipeline.helpers import run_scheduler


@dataclass
class _State(PipelineStateBase):
    value: int = 0


class _FakeVocoder(BatchVocoderBase):
    def __init__(
        self,
        *,
        fail_hook: str | None = None,
        result_count_delta: int = 0,
    ) -> None:
        self.fail_hook = fail_hook
        self.result_count_delta = result_count_delta
        self.calls: list[tuple[str, Any]] = []

    def _record_or_fail(self, hook: str, value: Any) -> None:
        self.calls.append((hook, value))
        if self.fail_hook == hook:
            raise RuntimeError(f"{hook} failed")

    def prepare_item(self, payload: StagePayload) -> tuple[_State, int]:
        value = int(payload.data["value"])
        self._record_or_fail("prepare", value)
        return _State(value=value), value

    async def decode_batch(
        self, items: list[tuple[_State, int]]
    ) -> list[tuple[torch.Tensor, int]]:
        values = tuple(int(value) for _, value in items)
        self._record_or_fail("decode", values)
        results = [(torch.tensor([value]), 24000) for value in values]
        if self.result_count_delta < 0:
            return results[: self.result_count_delta]
        if self.result_count_delta > 0:
            results.extend(
                (torch.tensor([-1]), 24000) for _ in range(self.result_count_delta)
            )
        return results

    def store_result(
        self,
        payload: StagePayload,
        state: _State,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        value = int(wav.item())
        self._record_or_fail("store", value)
        payload.data = {
            "value": value,
            "sample_rate": sample_rate,
            "state_value": state.value,
        }
        return payload


def _message(request_id: str, value: int) -> IncomingMessage:
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=value),
        data={"value": value},
    )
    return IncomingMessage(request_id, "new_request", payload)


def test_single_path_runs_hooks_in_order() -> None:
    vocoder = _FakeVocoder()
    scheduler = vocoder.build_scheduler()

    outputs = run_scheduler(scheduler, [_message("req-1", 1)], output_count=1)

    assert outputs[0].request_id == "req-1"
    assert outputs[0].type == "result"
    assert outputs[0].data.data == {
        "value": 1,
        "sample_rate": 24000,
        "state_value": 1,
    }
    assert vocoder.calls == [
        ("prepare", 1),
        ("decode", (1,)),
        ("store", 1),
    ]


def test_batch_path_preserves_order_and_scheduler_settings() -> None:
    vocoder = _FakeVocoder()
    scheduler = vocoder.build_scheduler(max_batch_size=2, max_batch_wait_ms=4)

    outputs = run_scheduler(
        scheduler,
        [_message("req-1", 1), _message("req-2", 2)],
        output_count=2,
    )

    assert scheduler._max_batch_size == 2
    assert scheduler._max_batch_wait_s == pytest.approx(0.004)
    assert [output.request_id for output in outputs] == ["req-1", "req-2"]
    assert [output.data.data["value"] for output in outputs] == [1, 2]
    assert vocoder.calls == [
        ("prepare", 1),
        ("prepare", 2),
        ("decode", (1, 2)),
        ("store", 1),
        ("store", 2),
    ]


@pytest.mark.parametrize("request_count", [1, 2])
def test_result_count_mismatch_fails_clearly(request_count: int) -> None:
    scheduler = _FakeVocoder(result_count_delta=-1).build_scheduler(
        max_batch_size=2,
        max_batch_wait_ms=4,
    )
    messages = [_message(f"req-{index}", index) for index in range(request_count)]

    outputs = run_scheduler(scheduler, messages, output_count=request_count)

    noun = "input" if request_count == 1 else "inputs"
    expected = (
        f"decode_batch returned {request_count - 1} results "
        f"for {request_count} {noun}"
    )
    assert all(output.type == "error" for output in outputs)
    assert all(isinstance(output.data, RuntimeError) for output in outputs)
    assert all(expected in str(output.data) for output in outputs)


@pytest.mark.parametrize("hook", ["prepare", "decode", "store"])
@pytest.mark.parametrize("request_count", [1, 2])
def test_hook_failure_is_emitted_for_the_failed_batch(
    hook: str, request_count: int
) -> None:
    scheduler = _FakeVocoder(fail_hook=hook).build_scheduler(
        max_batch_size=request_count,
        max_batch_wait_ms=20,
    )
    messages = [_message(f"req-{index}", index) for index in range(request_count)]

    outputs = run_scheduler(scheduler, messages, output_count=request_count)

    assert [output.request_id for output in outputs] == [
        message.request_id for message in messages
    ]
    assert all(output.type == "error" for output in outputs)
    assert all(isinstance(output.data, RuntimeError) for output in outputs)
    assert all(str(output.data) == f"{hook} failed" for output in outputs)
