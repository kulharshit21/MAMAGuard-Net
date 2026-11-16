"""Inference-only explainability utilities; this package is not imported by training."""

from .attribution import integrated_gradients, phase_contributions
from .faithfulness import faithfulness_check
from .gradcam import layer_gradcam
from .targets import SegmentationScoreAdapter

__all__ = [
    "SegmentationScoreAdapter",
    "faithfulness_check",
    "integrated_gradients",
    "layer_gradcam",
    "phase_contributions",
]

