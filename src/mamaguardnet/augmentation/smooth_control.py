from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom


@dataclass(frozen=True)
class SmoothAffineFields:
    gain: np.ndarray
    offset: np.ndarray


def sample_smooth_affine_fields(
    spatial_shape: tuple[int, ...],
    *,
    rng: np.random.Generator,
    phase_count: int = 3,
    shared_across_phases: bool,
    gain_log_std: float = 0.12,
    offset_std: float = 0.05,
    grid: int = 4,
) -> SmoothAffineFields:
    """Sample matched smooth positive affine fields for the A4/A5 comparison.

    Shared mode samples one field and repeats it across phases. Independent mode samples
    one identically distributed field per phase. Both modes therefore have the same grid,
    interpolation, marginal gain/offset distributions and target spatial shape.
    """
    if not spatial_shape or any(size <= 0 for size in spatial_shape):
        raise ValueError(f"Invalid spatial shape: {spatial_shape}")
    if phase_count < 1 or grid < 2:
        raise ValueError("phase_count must be positive and grid must be at least 2")
    if gain_log_std < 0 or offset_std < 0:
        raise ValueError("Standard deviations must be non-negative")
    sampled_phases = 1 if shared_across_phases else phase_count
    low_shape = (sampled_phases,) + (grid,) * len(spatial_shape)
    target_shape = (sampled_phases, *spatial_shape)
    factors = (1.0, *(size / grid for size in spatial_shape))
    log_gain = zoom(rng.normal(0.0, gain_log_std, low_shape), factors, order=1)
    offset = zoom(rng.normal(0.0, offset_std, low_shape), factors, order=1)
    slices = (slice(None), *(slice(0, size) for size in spatial_shape))
    gain = np.exp(log_gain[slices]).astype(np.float32, copy=False)
    offset = offset[slices].astype(np.float32, copy=False)
    if gain.shape != target_shape or offset.shape != target_shape:
        raise RuntimeError(
            f"Field interpolation produced {gain.shape}/{offset.shape}; expected {target_shape}"
        )
    if shared_across_phases:
        gain = np.repeat(gain, phase_count, axis=0)
        offset = np.repeat(offset, phase_count, axis=0)
    return SmoothAffineFields(gain=gain, offset=offset)


def apply_smooth_affine(phases: np.ndarray, fields: SmoothAffineFields) -> np.ndarray:
    phases = np.asarray(phases, dtype=np.float32)
    if phases.shape != fields.gain.shape or phases.shape != fields.offset.shape:
        raise ValueError(
            f"Phase shape {phases.shape} does not match fields "
            f"{fields.gain.shape}/{fields.offset.shape}"
        )
    return phases * fields.gain + fields.offset
