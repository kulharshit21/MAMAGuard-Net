"""Geometry guards for medical-image evaluation.

All arrays are interpreted in NIfTI voxel-index order. Affines map voxel centres
to physical coordinates. Label resampling is explicit and nearest-neighbour only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform


class GeometryMismatchError(ValueError):
    """Raised when prediction and reference grids are not directly comparable."""


@dataclass(frozen=True)
class ImageGeometry:
    shape: tuple[int, int, int]
    affine: np.ndarray

    def __post_init__(self) -> None:
        affine = np.asarray(self.affine, dtype=float)
        if len(self.shape) != 3 or any(int(size) <= 0 for size in self.shape):
            raise ValueError(f"Expected a positive three-dimensional shape, got {self.shape}")
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("Affine must be a finite 4 x 4 matrix")
        if not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), rtol=0, atol=1e-7):
            raise ValueError("Affine must use homogeneous coordinates")
        if abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            raise ValueError("Affine spatial transform is singular")
        object.__setattr__(self, "affine", affine)
        object.__setattr__(self, "shape", tuple(int(size) for size in self.shape))

    @property
    def spacing(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in np.linalg.norm(self.affine[:3, :3], axis=0))

    @property
    def voxel_volume_mm3(self) -> float:
        return float(abs(np.linalg.det(self.affine[:3, :3])))


def assert_same_geometry(
    prediction: ImageGeometry,
    target: ImageGeometry,
    *,
    affine_rtol: float = 1e-5,
    affine_atol: float = 1e-4,
) -> None:
    problems = []
    if prediction.shape != target.shape:
        problems.append(f"shape {prediction.shape} != {target.shape}")
    if not np.allclose(
        prediction.affine,
        target.affine,
        rtol=affine_rtol,
        atol=affine_atol,
    ):
        maximum = float(np.max(np.abs(prediction.affine - target.affine)))
        problems.append(f"affine mismatch (maximum absolute difference {maximum:.6g})")
    if problems:
        raise GeometryMismatchError("; ".join(problems))


def resample_label_to_geometry(
    label: np.ndarray,
    source: ImageGeometry,
    target: ImageGeometry,
    *,
    background: int | bool = 0,
) -> np.ndarray:
    """Resample a discrete label map to ``target`` using nearest neighbours.

    The mapping passed to SciPy converts output (target) voxel indices to input
    (source) voxel indices: ``inv(source_affine) @ target_affine``.
    """
    array = np.asarray(label)
    if array.shape != source.shape:
        raise ValueError(f"Label shape {array.shape} does not match source geometry {source.shape}")
    output_to_input = np.linalg.inv(source.affine) @ target.affine
    result = affine_transform(
        array,
        matrix=output_to_input[:3, :3],
        offset=output_to_input[:3, 3],
        output_shape=target.shape,
        order=0,
        mode="constant",
        cval=float(background),
        prefilter=False,
    )
    return result.astype(array.dtype, copy=False)
