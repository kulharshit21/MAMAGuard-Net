import pytest
import torch
from torch import nn

from mamaguardnet.explainability import (
    SegmentationScoreAdapter,
    faithfulness_check,
    integrated_gradients,
    layer_gradcam,
    phase_contributions,
)

captum = pytest.importorskip("captum")


class TinySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv3d(3, 4, kernel_size=1, bias=False)
        self.head = nn.Conv3d(4, 2, kernel_size=1, bias=False)
        nn.init.constant_(self.features.weight, 0.25)
        nn.init.constant_(self.head.weight, 0.5)

    def forward(self, inputs):
        return self.head(torch.relu(self.features(inputs)))


def test_gradcam_and_integrated_gradients_are_3d_and_finite():
    model = TinySegmentationModel()
    mask = torch.zeros(4, 4, 4, dtype=torch.bool)
    mask[1:3, 1:3, 1:3] = True
    adapter = SegmentationScoreAdapter(model, region_mask=mask)
    inputs = torch.rand(1, 3, 4, 4, 4, requires_grad=True)

    cam = layer_gradcam(adapter, inputs, model.features)
    attribution = integrated_gradients(adapter, inputs, steps=8)
    contributions = phase_contributions(attribution)

    assert cam.shape == (1, 1, 4, 4, 4)
    assert attribution.shape == inputs.shape
    assert torch.isfinite(cam).all() and torch.isfinite(attribution).all()
    assert sum(item["absolute_fraction"] for item in contributions) == pytest.approx(1.0)


def test_faithfulness_uses_equal_perturbation_counts():
    model = TinySegmentationModel()
    adapter = SegmentationScoreAdapter(model, tumor_channel=1)
    inputs = torch.ones(1, 3, 3, 3, 3)
    attribution = torch.zeros_like(inputs)
    attribution[:, :, 0] = 10
    result = faithfulness_check(adapter, inputs, attribution, fraction=0.1)
    assert result["perturbed_values_per_case"] == 8
    assert result["top_attribution_drop"] >= result["low_attribution_drop"]


def test_deep_supervision_adapter_selects_highest_resolution_output():
    class DeepSupervision(nn.Module):
        def forward(self, inputs):
            full = torch.stack((inputs[:, 0], inputs[:, 0] * 2), dim=1)
            small = full[:, :, ::2, ::2, ::2]
            return [small, full]

    inputs = torch.ones(1, 3, 4, 4, 4)
    score = SegmentationScoreAdapter(DeepSupervision(), tumor_channel=1)(inputs)
    assert score.item() == 2.0

