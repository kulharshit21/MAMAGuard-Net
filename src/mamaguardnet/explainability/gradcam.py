from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .targets import SegmentationScoreAdapter


def layer_gradcam(
    adapter: SegmentationScoreAdapter,
    inputs: torch.Tensor,
    layer: nn.Module,
) -> torch.Tensor:
    """Return a normalized 3D Grad-CAM map in input spatial coordinates."""
    try:
        from captum.attr import LayerGradCam
    except ImportError as error:  # pragma: no cover - dependency-specific path
        raise RuntimeError("Captum is required for Grad-CAM; install the analysis extra") from error

    adapter.eval()
    attribution = LayerGradCam(adapter, layer).attribute(inputs, target=0)
    attribution = F.interpolate(
        attribution,
        size=inputs.shape[2:],
        mode="trilinear",
        align_corners=False,
    ).relu()
    maximum = attribution.flatten(1).amax(1).view(-1, 1, 1, 1, 1)
    return attribution / maximum.clamp_min(torch.finfo(attribution.dtype).eps)

