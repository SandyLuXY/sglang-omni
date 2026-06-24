# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local reference audio encoder with packed attention."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from sglang_omni.models.moss_tts_local.vocoder_decoder import (
    MossTTSLocalProjectedTransformer,
)


class MossTTSLocalRefEncoder(nn.Module):
    """Iterable MOSS codec encoder with patched projected transformers."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        source_stages = list(source)
        assert source_stages, "MOSS ref encoder must be a non-empty stage list"
        self.stages = nn.ModuleList(
            [self._wrap_stage(stage) for stage in source_stages]
        )

    @staticmethod
    def _wrap_stage(stage: nn.Module) -> nn.Module:
        module_type = stage.module_type
        if module_type == "Transformer":
            return MossTTSLocalProjectedTransformer(stage)
        if module_type == "PatchedPretransform":
            return stage
        raise ValueError(
            f"unsupported MOSS ref encoder stage {stage.__class__.__name__} "
            f"with module_type={module_type!r}"
        )

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    def __getitem__(self, index: int) -> nn.Module:
        return self.stages[index]

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for stage in self.stages:
            x, input_lengths = stage(x, input_lengths)
        return x, input_lengths
