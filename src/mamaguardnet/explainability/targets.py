from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def primary_logits(output: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    """Select the highest-resolution output from a deep-supervision model."""
    if isinstance(output, torch.Tensor):
        return output
    if not output:
        raise ValueError("The model returned an empty output sequence")
    tensors = [value for value in output if isinstance(value, torch.Tensor)]
    if not tensors:
        raise TypeError("The model output sequence contains no tensors")
    return max(tensors, key=lambda value: value.ndim * 10_000 + value[0, 0].numel())


class SegmentationScoreAdapter(nn.Module):
    """Turn voxel logits into one scalar score per case for attribution methods."""

    def __init__(
        self,
        model: nn.Module,
        *,
        tumor_channel: int = 1,
        region_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.tumor_channel = tumor_channel
        if region_mask is not None:
            mask = torch.as_tensor(region_mask, dtype=torch.bool)
            if mask.ndim not in {3, 4, 5}:
                raise ValueError("region_mask must contain 3D spatial masks")
            self.register_buffer("region_mask", mask, persistent=False)
        else:
            self.region_mask = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = primary_logits(self.model(inputs))
        if logits.ndim != 5:
            raise ValueError(f"Expected B,C,D,H,W logits, got {tuple(logits.shape)}")
        if not 0 <= self.tumor_channel < logits.shape[1]:
            raise ValueError(f"tumor_channel {self.tumor_channel} is unavailable")
        tumor = logits[:, self.tumor_channel : self.tumor_channel + 1]
        if self.region_mask is None:
            return tumor.flatten(1).mean(1, keepdim=True)
        mask = self.region_mask
        while mask.ndim < 5:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and tumor.shape[0] > 1:
            mask = mask.expand(tumor.shape[0], -1, -1, -1, -1)
        if mask.shape[1] != 1:
            raise ValueError("region_mask must have one channel")
        if tuple(mask.shape[2:]) != tuple(tumor.shape[2:]):
            mask = F.interpolate(mask.float(), size=tumor.shape[2:], mode="nearest").bool()
        if torch.any(mask.flatten(1).sum(1) == 0):
            raise ValueError("region_mask must contain at least one voxel per case")
        return (tumor * mask).flatten(1).sum(1, keepdim=True) / mask.flatten(1).sum(
            1, keepdim=True
        )

