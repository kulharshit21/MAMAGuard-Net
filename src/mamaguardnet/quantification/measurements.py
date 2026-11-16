from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import label

from mamaguardnet.evaluation.geometry import ImageGeometry


def _mask(mask: np.ndarray, geometry: ImageGeometry) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.shape != geometry.shape:
        raise ValueError(f"Mask shape {result.shape} does not match geometry {geometry.shape}")
    return result


def _world_centres(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((indices.astype(float), np.ones(len(indices))))
    return (homogeneous @ affine.T)[:, :3]


def _physical_aabb(centres: np.ndarray, linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half_support = 0.5 * np.sum(np.abs(linear), axis=1)
    return centres.min(axis=0) - half_support, centres.max(axis=0) + half_support


def _principal_dimensions(centres: np.ndarray, linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(centres) < 2 or np.allclose(centres, centres[0]):
        axes = np.eye(3)
    else:
        covariance = np.cov(centres, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    dimensions = []
    for axis in axes.T:
        projected = centres @ axis
        half_support = 0.5 * np.sum(np.abs(axis @ linear))
        dimensions.append(float(projected.max() - projected.min() + 2 * half_support))
    return np.asarray(dimensions), axes


def voxel_surface_area_mm2(mask: np.ndarray, affine: np.ndarray) -> float:
    """Surface area of the union of occupied affine-transformed voxels."""
    binary = np.asarray(mask, dtype=bool)
    linear = np.asarray(affine, dtype=float)[:3, :3]
    face_areas = (
        np.linalg.norm(np.cross(linear[:, 1], linear[:, 2])),
        np.linalg.norm(np.cross(linear[:, 0], linear[:, 2])),
        np.linalg.norm(np.cross(linear[:, 0], linear[:, 1])),
    )
    area = 0.0
    for axis, face_area in enumerate(face_areas):
        padded = np.pad(binary.astype(np.int8), 1, mode="constant")
        transitions = np.diff(padded, axis=axis)
        area += np.count_nonzero(transitions) * float(face_area)
    return float(area)


def quantify_mask(mask: np.ndarray, geometry: ImageGeometry) -> dict[str, Any]:
    binary = _mask(mask, geometry)
    indices = np.argwhere(binary)
    if not len(indices):
        return {
            "voxel_count": 0,
            "volume_mm3": 0.0,
            "volume_cm3": 0.0,
            "centroid_mm": None,
            "axis_aligned_min_mm": None,
            "axis_aligned_max_mm": None,
            "axis_aligned_extents_mm": None,
            "principal_dimensions_mm": None,
            "principal_axes": None,
            "surface_area_mm2": 0.0,
            "component_count": 0,
            "largest_component_fraction": 0.0,
        }

    centres = _world_centres(indices, geometry.affine)
    lower, upper = _physical_aabb(centres, geometry.affine[:3, :3])
    dimensions, axes = _principal_dimensions(centres, geometry.affine[:3, :3])
    components, count = label(binary, structure=np.ones((3, 3, 3), dtype=np.uint8))
    component_sizes = np.bincount(components.ravel())[1:]
    voxel_count = len(indices)
    volume_mm3 = voxel_count * geometry.voxel_volume_mm3
    return {
        "voxel_count": voxel_count,
        "volume_mm3": float(volume_mm3),
        "volume_cm3": float(volume_mm3 / 1000.0),
        "centroid_mm": centres.mean(axis=0).tolist(),
        "axis_aligned_min_mm": lower.tolist(),
        "axis_aligned_max_mm": upper.tolist(),
        "axis_aligned_extents_mm": (upper - lower).tolist(),
        "principal_dimensions_mm": dimensions.tolist(),
        "principal_axes": axes.tolist(),
        "surface_area_mm2": voxel_surface_area_mm2(binary, geometry.affine),
        "component_count": int(count),
        "largest_component_fraction": float(component_sizes.max() / voxel_count),
    }
