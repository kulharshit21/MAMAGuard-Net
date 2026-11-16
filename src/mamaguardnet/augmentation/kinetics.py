from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mamaguardnet.kinetics.constraints import ConstraintReport, constraint_report, shared_affine


@dataclass(frozen=True)
class KineticsAugmentationResult:
    image: np.ndarray
    report: ConstraintReport
    attempts: int
    fallback_used: bool


def augment_kinetics(
    phases: np.ndarray,
    *,
    rng: np.random.Generator,
    gain_range: tuple[float, float] = (0.8, 1.2),
    offset_std: float = 0.05,
    noise_std: float = 0.0,
    epsilon: float = 1e-5,
    max_attempts: int = 4,
) -> KineticsAugmentationResult:
    """Apply one positive affine field to all phases, then validate optional noise.

    Exact shared affine transforms preserve difference signs. Optional per-phase noise is
    permitted only when a post-transform check confirms the configured discrete invariants.
    """
    phases = np.asarray(phases, dtype=np.float32)
    if phases.ndim < 2 or phases.shape[0] != 3:
        raise ValueError("Expected [3, ...] phases")
    for attempt in range(1, max_attempts + 1):
        gain = float(rng.uniform(*gain_range))
        offset = float(rng.normal(0.0, offset_std))
        candidate = shared_affine(phases, gain, offset).astype(np.float32, copy=False)
        if noise_std:
            candidate = candidate + rng.normal(0.0, noise_std, phases.shape).astype(np.float32)
        report = constraint_report(phases, candidate, epsilon=epsilon)
        if report.valid:
            return KineticsAugmentationResult(candidate, report, attempt, False)
    gain = float(rng.uniform(*gain_range))
    offset = float(rng.normal(0.0, offset_std))
    fallback = shared_affine(phases, gain, offset).astype(np.float32, copy=False)
    return KineticsAugmentationResult(
        fallback,
        constraint_report(phases, fallback, epsilon=epsilon),
        max_attempts,
        True,
    )


def augment_independent(
    phases: np.ndarray, *, rng: np.random.Generator, gain_range: tuple[float, float] = (0.8, 1.2),
    offset_std: float = 0.05,
) -> np.ndarray:
    phases = np.asarray(phases, dtype=np.float32)
    gains = rng.uniform(*gain_range, size=(3,) + (1,) * (phases.ndim - 1)).astype(np.float32)
    offsets = rng.normal(0.0, offset_std, size=(3,) + (1,) * (phases.ndim - 1)).astype(np.float32)
    return gains * phases + offsets

