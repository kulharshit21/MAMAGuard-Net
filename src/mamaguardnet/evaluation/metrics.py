from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, label
from scipy.spatial import cKDTree


def _masks(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred, truth = np.asarray(prediction, bool), np.asarray(target, bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match target shape {truth.shape}")
    return pred, truth


def _spacing(spacing: tuple[float, ...], ndim: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in spacing)
    if len(values) != ndim or not np.isfinite(values).all() or any(value <= 0 for value in values):
        raise ValueError(f"Expected {ndim} finite positive spacing values, got {spacing}")
    return values


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, truth = _masks(prediction, target)
    denominator = pred.sum() + truth.sum()
    return 1.0 if denominator == 0 else float(2 * np.logical_and(pred, truth).sum() / denominator)


def tumor_sensitivity(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, truth = _masks(prediction, target)
    return 1.0 if truth.sum() == 0 else float(np.logical_and(pred, truth).sum() / truth.sum())


def relative_volume_error(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, truth = _masks(prediction, target)
    return 0.0 if truth.sum() == 0 and pred.sum() == 0 else float((pred.sum() - truth.sum()) / max(truth.sum(), 1))


def intersection_over_union(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, truth = _masks(prediction, target)
    union = np.logical_or(pred, truth).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, truth).sum() / union)


def precision(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, truth = _masks(prediction, target)
    predicted = pred.sum()
    if predicted == 0:
        return 1.0 if truth.sum() == 0 else 0.0
    return float(np.logical_and(pred, truth).sum() / predicted)


def voxel_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, truth = _masks(prediction, target)
    return float(np.mean(pred == truth))


def false_positive_component_count(prediction: np.ndarray, target: np.ndarray) -> int:
    pred, truth = _masks(prediction, target)
    components, count = label(pred, structure=np.ones((3,) * pred.ndim, dtype=np.uint8))
    return sum(not np.any(truth[components == component]) for component in range(1, count + 1))


def _surface(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, bool) & ~binary_erosion(np.asarray(mask, bool), border_value=0)


def surface_distances(
    prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    pred, truth = _masks(prediction, target)
    spacing = _spacing(spacing, pred.ndim)
    if not pred.any() and not truth.any():
        return np.asarray([0.0]), np.asarray([0.0])
    pred_surface, target_surface = _surface(pred), _surface(truth)
    if not pred_surface.any() or not target_surface.any():
        return np.asarray([np.inf]), np.asarray([np.inf])
    scale = np.asarray(spacing, dtype=float)
    pred_points = np.argwhere(pred_surface) * scale
    target_points = np.argwhere(target_surface) * scale
    to_target = cKDTree(target_points).query(pred_points, k=1, workers=1)[0]
    to_pred = cKDTree(pred_points).query(target_points, k=1, workers=1)[0]
    return to_target, to_pred


def hd95(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> float:
    first, second = surface_distances(prediction, target, spacing)
    return float(np.percentile(np.concatenate([first, second]), 95))


def mean_surface_distance(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> float:
    first, second = surface_distances(prediction, target, spacing)
    return float(np.mean(np.concatenate([first, second])))


def surface_dice(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...],
    *,
    tolerance_mm: float,
) -> float:
    """Symmetric surface Dice at a prespecified physical tolerance."""
    if not np.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ValueError("tolerance_mm must be finite and non-negative")
    pred, truth = _masks(prediction, target)
    if not pred.any() and not truth.any():
        return 1.0
    if not pred.any() or not truth.any():
        return 0.0
    first, second = surface_distances(pred, truth, spacing)
    within = np.count_nonzero(first <= tolerance_mm) + np.count_nonzero(second <= tolerance_mm)
    return float(within / (first.size + second.size))


def evaluate_case(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...],
    *,
    severe_failure_dice_threshold: float = 0.1,
    surface_dice_tolerance_mm: float = 2.0,
) -> dict[str, float | int | bool]:
    pred, truth = _masks(prediction, target)
    spacing = _spacing(spacing, pred.ndim)
    if not 0 <= severe_failure_dice_threshold <= 1:
        raise ValueError("severe_failure_dice_threshold must be between 0 and 1")
    score = dice(prediction, target)
    voxel_volume = float(np.prod(spacing))
    volume_difference = float((pred.sum() - truth.sum()) * voxel_volume)
    first_surface, second_surface = surface_distances(pred, truth, spacing)
    combined_surface = np.concatenate((first_surface, second_surface))
    return {
        "dice": score,
        "iou": intersection_over_union(pred, truth),
        "precision": precision(pred, truth),
        "voxel_accuracy": voxel_accuracy(pred, truth),
        "hd95_mm": float(np.percentile(combined_surface, 95)),
        "mean_surface_distance_mm": float(np.mean(combined_surface)),
        "surface_dice_2mm": surface_dice(
            pred,
            truth,
            spacing,
            tolerance_mm=surface_dice_tolerance_mm,
        ),
        "tumor_sensitivity": tumor_sensitivity(prediction, target),
        "relative_volume_error": relative_volume_error(prediction, target),
        "absolute_volume_error_mm3": abs(volume_difference),
        "signed_volume_error_mm3": volume_difference,
        "false_positive_voxels": int(np.count_nonzero(pred & ~truth)),
        "false_positive_components": false_positive_component_count(pred, truth),
        "failed_segmentation": bool(truth.any() and not pred.any()),
        "severe_failure": bool(truth.any() and score < severe_failure_dice_threshold),
    }

