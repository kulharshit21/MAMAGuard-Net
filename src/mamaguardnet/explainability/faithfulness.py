from __future__ import annotations

import torch

from .targets import SegmentationScoreAdapter


@torch.no_grad()
def faithfulness_check(
    adapter: SegmentationScoreAdapter,
    inputs: torch.Tensor,
    attributions: torch.Tensor,
    *,
    fraction: float = 0.05,
    baseline_value: float = 0.0,
) -> dict[str, float | int | bool]:
    """Compare equal-size perturbations of most- and least-attributed input voxels."""
    if inputs.shape != attributions.shape:
        raise ValueError("inputs and attributions must have identical shapes")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    adapter.eval()
    scores = attributions.abs().flatten(1)
    count = max(1, round(scores.shape[1] * fraction))
    top_indices = scores.topk(count, dim=1, largest=True).indices
    low_indices = scores.topk(count, dim=1, largest=False).indices

    top_inputs = inputs.clone().flatten(1)
    low_inputs = inputs.clone().flatten(1)
    top_inputs.scatter_(1, top_indices, baseline_value)
    low_inputs.scatter_(1, low_indices, baseline_value)
    top_inputs = top_inputs.reshape_as(inputs)
    low_inputs = low_inputs.reshape_as(inputs)

    original = adapter(inputs).mean()
    top_score = adapter(top_inputs).mean()
    low_score = adapter(low_inputs).mean()
    top_drop = original - top_score
    low_drop = original - low_score
    return {
        "perturbed_values_per_case": count,
        "original_score": float(original.item()),
        "top_attribution_score": float(top_score.item()),
        "low_attribution_score": float(low_score.item()),
        "top_attribution_drop": float(top_drop.item()),
        "low_attribution_drop": float(low_drop.item()),
        "faithfulness_pass": bool(top_drop > low_drop),
    }

