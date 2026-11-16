import numpy as np
import pytest

from mamaguardnet.evaluation.metrics import (
    dice,
    evaluate_case,
    hd95,
    intersection_over_union,
    precision,
    surface_dice,
)
from mamaguardnet.statistics.bootstrap import paired_bootstrap_ci


def test_identical_masks_have_perfect_geometry():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:5, 2:5, 2:5] = True
    result = evaluate_case(mask, mask, (1.0, 1.0, 1.0))
    assert result["dice"] == 1.0
    assert result["hd95_mm"] == 0.0
    assert result["mean_surface_distance_mm"] == 0.0
    assert result["surface_dice_2mm"] == 1.0


def test_spacing_is_used_in_hd95():
    first = np.zeros((5, 5), bool)
    second = np.zeros((5, 5), bool)
    first[1, 1] = True
    second[2, 1] = True
    assert hd95(first, second, (2.0, 1.0)) == 2.0


def test_surface_distance_uses_physical_euclidean_coordinates():
    first = np.zeros((4, 4, 4), dtype=bool)
    second = np.zeros_like(first)
    first[1, 1, 1] = True
    second[2, 2, 1] = True
    assert hd95(first, second, (2.0, 3.0, 5.0)) == pytest.approx(np.sqrt(13.0))
    assert surface_dice(first, second, (2.0, 3.0, 5.0), tolerance_mm=3.0) == 0.0
    assert surface_dice(first, second, (2.0, 3.0, 5.0), tolerance_mm=4.0) == 1.0


def test_empty_dice_convention_and_bootstrap_reproducibility():
    assert dice(np.zeros(3), np.zeros(3)) == 1.0
    candidate, baseline = np.asarray([0.8, 0.9]), np.asarray([0.7, 0.8])
    assert paired_bootstrap_ci(candidate, baseline, samples=100, seed=4) == paired_bootstrap_ci(candidate, baseline, samples=100, seed=4)


def test_overlap_metrics_and_false_positive_components():
    truth = np.zeros((6, 6, 6), dtype=bool)
    prediction = np.zeros_like(truth)
    truth[1:3, 1:3, 1:3] = True
    prediction[1:3, 1:3, 1:3] = True
    prediction[5, 5, 5] = True
    result = evaluate_case(prediction, truth, (2.0, 1.0, 1.0))
    assert result["iou"] == intersection_over_union(prediction, truth) == pytest.approx(8 / 9)
    assert result["precision"] == precision(prediction, truth) == pytest.approx(8 / 9)
    assert result["false_positive_voxels"] == 1
    assert result["false_positive_components"] == 1
    assert result["absolute_volume_error_mm3"] == 2.0
    assert not result["failed_segmentation"]


def test_empty_and_shape_mismatch_conventions():
    empty = np.zeros((3, 3, 3), dtype=bool)
    result = evaluate_case(empty, empty, (1.0, 1.0, 1.0))
    assert result["dice"] == result["iou"] == result["precision"] == 1.0
    assert result["hd95_mm"] == 0.0
    assert result["surface_dice_2mm"] == 1.0
    assert surface_dice(empty, np.ones_like(empty), (1.0, 1.0, 1.0), tolerance_mm=2.0) == 0.0
    with pytest.raises(ValueError, match="shape"):
        evaluate_case(empty, np.zeros((2, 2, 2)), (1.0, 1.0, 1.0))

