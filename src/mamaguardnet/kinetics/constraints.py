from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConstraintReport:
    eligible_voxels: int
    sign_violations: int
    ordering_eligible_voxels: int
    ordering_violations: int

    @property
    def valid(self) -> bool:
        return self.sign_violations == 0 and self.ordering_violations == 0

    @property
    def violation_rate(self) -> float:
        denominator = max(self.eligible_voxels + self.ordering_eligible_voxels, 1)
        return (self.sign_violations + self.ordering_violations) / denominator

    @property
    def sign_violation_rate(self) -> float:
        return self.sign_violations / max(self.eligible_voxels, 1)

    @property
    def ordering_violation_rate(self) -> float:
        return self.ordering_violations / max(self.ordering_eligible_voxels, 1)


def differences(phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if phases.shape[0] != 3:
        raise ValueError("Expected channel-first phases [pre, early, late]")
    return phases[1] - phases[0], phases[2] - phases[1]


def constraint_report(before: np.ndarray, after: np.ndarray, *, epsilon: float = 1e-5) -> ConstraintReport:
    if before.shape != after.shape:
        raise ValueError("Before/after arrays must have identical shapes")
    d10, d21 = differences(before)
    a10, a21 = differences(after)
    eligible10 = np.abs(d10) > epsilon
    eligible21 = np.abs(d21) > epsilon
    sign_violations = int(np.count_nonzero(eligible10 & (np.sign(d10) != np.sign(a10))))
    sign_violations += int(np.count_nonzero(eligible21 & (np.sign(d21) != np.sign(a21))))
    ordering_before = d10 > d21
    ordering_after = a10 > a21
    ordering_mask = (np.abs(d10 - d21) > epsilon) & eligible10 & eligible21
    ordering_violations = int(np.count_nonzero(ordering_mask & (ordering_before != ordering_after)))
    return ConstraintReport(
        eligible_voxels=int(np.count_nonzero(eligible10) + np.count_nonzero(eligible21)),
        sign_violations=sign_violations,
        ordering_eligible_voxels=int(np.count_nonzero(ordering_mask)),
        ordering_violations=ordering_violations,
    )


def shared_affine(phases: np.ndarray, gain: np.ndarray | float, offset: np.ndarray | float) -> np.ndarray:
    gain_array = np.asarray(gain)
    if np.any(gain_array <= 0):
        raise ValueError("Shared affine gain must be strictly positive")
    return gain_array * phases + np.asarray(offset)

