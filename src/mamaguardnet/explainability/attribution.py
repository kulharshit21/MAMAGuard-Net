from __future__ import annotations

import torch

from .targets import SegmentationScoreAdapter


def integrated_gradients(
    adapter: SegmentationScoreAdapter,
    inputs: torch.Tensor,
    *,
    baseline: torch.Tensor | None = None,
    steps: int = 32,
) -> torch.Tensor:
    """Compute per-voxel, per-phase integrated gradients for a segmentation score."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    try:
        from captum.attr import IntegratedGradients
    except ImportError as error:  # pragma: no cover - dependency-specific path
        raise RuntimeError(
            "Captum is required for integrated gradients; install the analysis extra"
        ) from error
    adapter.eval()
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    if baseline.shape != inputs.shape:
        raise ValueError("baseline and inputs must have identical shapes")
    return IntegratedGradients(adapter).attribute(
        inputs,
        baselines=baseline,
        target=0,
        n_steps=steps,
    )


def phase_contributions(attributions: torch.Tensor) -> list[dict[str, float]]:
    """Summarize signed and absolute attribution for each DCE input phase."""
    if attributions.ndim != 5:
        raise ValueError("Expected B,C,D,H,W attributions")
    absolute = attributions.abs().sum(dim=(0, 2, 3, 4))
    total = absolute.sum().clamp_min(torch.finfo(attributions.dtype).eps)
    signed = attributions.sum(dim=(0, 2, 3, 4))
    return [
        {
            "phase_index": index,
            "absolute_attribution": float(absolute[index].item()),
            "absolute_fraction": float((absolute[index] / total).item()),
            "signed_attribution": float(signed[index].item()),
        }
        for index in range(attributions.shape[1])
    ]

