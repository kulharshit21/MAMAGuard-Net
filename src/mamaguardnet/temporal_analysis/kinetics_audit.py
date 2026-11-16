from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom

from mamaguardnet.kinetics.constraints import constraint_report


@dataclass(frozen=True)
class SmoothFieldResult:
    image: np.ndarray
    gain: np.ndarray
    offset: np.ndarray


def sample_a4_smooth_affine(
    phases: np.ndarray,
    *,
    rng: np.random.Generator,
    gain_log_std: float = 0.12,
    offset_std: float = 0.05,
    grid: int = 4,
) -> SmoothFieldResult:
    """Reproduce A4's smooth shared gain/offset field sampler for diagnostic use."""
    values = _phases(phases)
    if values.ndim != 4:
        raise ValueError("A4 smooth-field diagnostics require three 3D phase volumes")
    if gain_log_std < 0 or offset_std < 0 or grid < 2:
        raise ValueError("Field parameters must be non-negative and grid must be at least 2")
    low_shape = (grid,) * 3
    target = values.shape[1:]
    factors = tuple(size / grid for size in target)
    log_gain = zoom(rng.normal(0, gain_log_std, low_shape), factors, order=1)
    offset = zoom(rng.normal(0, offset_std, low_shape), factors, order=1)
    slices = tuple(slice(0, size) for size in target)
    gain = np.exp(log_gain[slices]).astype(np.float32, copy=False)
    offset = offset[slices].astype(np.float32, copy=False)
    image = gain[None] * values + offset[None]
    return SmoothFieldResult(image.astype(np.float32, copy=False), gain, offset)


def zscore_per_phase(phases: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Apply independent per-phase z-score normalization, as configured by nnU-Net."""
    values = _phases(phases)
    selected = _mask(mask, values.shape[1:])
    normalized = np.empty_like(values, dtype=np.float32)
    for phase in range(3):
        sample = values[phase][selected]
        mean, standard_deviation = float(sample.mean()), float(sample.std())
        if not standard_deviation > 0:
            raise ValueError(f"Phase {phase} has zero variance in the normalization mask")
        normalized[phase] = (values[phase] - mean) / standard_deviation
    return normalized


def _phases(phases: np.ndarray) -> np.ndarray:
    values = np.asarray(phases, dtype=np.float32)
    if values.ndim < 2 or values.shape[0] != 3:
        raise ValueError("Expected channel-first [pre, early, late] phases")
    if not np.isfinite(values).all():
        raise ValueError("Phases contain non-finite values")
    return values


def _mask(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != shape:
        raise ValueError(f"Mask shape {selected.shape} does not match phase shape {shape}")
    if not selected.any():
        raise ValueError("Mask is empty")
    return selected


def _pair_audit(
    reference: np.ndarray,
    candidate: np.ndarray,
    selected: np.ndarray,
    epsilon: float,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, first, second in (("early_minus_pre", 0, 1), ("late_minus_early", 1, 2)):
        before = reference[second] - reference[first]
        after = candidate[second] - candidate[first]
        eligible = selected & (np.abs(before) > epsilon)
        count = int(np.count_nonzero(eligible))
        violations = int(np.count_nonzero(eligible & (np.sign(before) != np.sign(after))))
        result[name] = {
            "eligible_voxels": count,
            "sign_violations": violations,
            "sign_agreement_rate": 1.0 if count == 0 else 1.0 - violations / count,
        }
    return result


def audit_kinetics_domains(
    original: np.ndarray,
    normalized: np.ndarray,
    augmented: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    epsilon: float = 1e-5,
) -> dict[str, object]:
    """Audit temporal invariants across distinct intensity domains."""
    raw, norm, aug = _phases(original), _phases(normalized), _phases(augmented)
    if raw.shape != norm.shape or raw.shape != aug.shape:
        raise ValueError("All domains must have identical shapes")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    selected = _mask(mask, raw.shape[1:])
    raw_norm = constraint_report(raw[:, selected], norm[:, selected], epsilon=epsilon)
    norm_aug = constraint_report(norm[:, selected], aug[:, selected], epsilon=epsilon)
    return {
        "epsilon": epsilon,
        "selected_voxels": int(np.count_nonzero(selected)),
        "original_to_normalized": {
            "pairs": _pair_audit(raw, norm, selected, epsilon),
            "eligible_voxels": raw_norm.eligible_voxels,
            "sign_violations": raw_norm.sign_violations,
            "sign_violation_rate": raw_norm.sign_violation_rate,
            "ordering_eligible_voxels": raw_norm.ordering_eligible_voxels,
            "ordering_violations": raw_norm.ordering_violations,
            "ordering_violation_rate": raw_norm.ordering_violation_rate,
            "valid": raw_norm.valid,
        },
        "normalized_to_augmented": {
            "pairs": _pair_audit(norm, aug, selected, epsilon),
            "eligible_voxels": norm_aug.eligible_voxels,
            "sign_violations": norm_aug.sign_violations,
            "sign_violation_rate": norm_aug.sign_violation_rate,
            "ordering_eligible_voxels": norm_aug.ordering_eligible_voxels,
            "ordering_violations": norm_aug.ordering_violations,
            "ordering_violation_rate": norm_aug.ordering_violation_rate,
            "valid": norm_aug.valid,
        },
        "interpretation": {
            "proven_domain": "normalized model-input domain",
            "claim": (
                "A shared strictly positive affine transform preserves pairwise difference "
                "signs and ordering in the domain to which it is applied."
            ),
            "boundary": (
                "Independent per-phase normalization is not guaranteed to preserve original "
                "cross-phase intensity differences."
            ),
        },
    }

