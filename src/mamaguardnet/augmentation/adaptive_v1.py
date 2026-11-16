"""Exploratory input-space reliability; never a physical kinetics estimator."""
from dataclasses import dataclass

import numpy as np

from .smooth_control import sample_smooth_affine_fields


@dataclass(frozen=True)
class AdaptiveResult:
    image: np.ndarray
    reliability: np.ndarray
    gain: np.ndarray
    residual: np.ndarray


def reliability_map(image: np.ndarray, noise_floor: float = 0.05) -> np.ndarray:
    """Minimum pairwise separation / (separation + robust spatial noise).

    Uses only ordered image channels. Independent normalization limits interpretation
    to network-input ordering. Invalid voxels receive zero reliability; transforms
    reject invalid image inputs rather than fabricate replacement signals.
    """
    x = np.asarray(image, dtype=np.float64)
    if x.ndim < 3 or x.shape[0] < 2 or not np.isfinite(noise_floor) or noise_floor <= 0:
        raise ValueError('Expected at least two channels and positive finite noise floor')
    valid = np.isfinite(x).all(axis=0)
    clean = np.where(np.isfinite(x), x, 0)
    noise = []
    for axis in range(1, x.ndim):
        delta = np.diff(x, axis=axis)
        delta = delta[np.isfinite(delta)]
        if delta.size:
            noise.append(float(np.median(np.abs(delta - np.median(delta))) / 0.67448975 / np.sqrt(2)))
    sigma = max([noise_floor, *noise])
    separation = np.min(np.diff(np.sort(clean, axis=0), axis=0), axis=0)
    return np.where(valid, separation / (separation + sigma), 0).astype(np.float32)


def adaptive_affine(image, *, rng, mode='adaptive', residual_bound=0.05,
                    noise_floor=0.05, gain_bounds=(0.5, 2.0), reliability=None,
                    gain_log_std=0.12, offset_std=0.05, grid=4):
    """Shared bounded affine + (1-r)*bounded independent smooth offset.

    At r=1 phase differences scale by positive gain exactly (up to roundoff).
    At r=0 independent residuals are permitted; universal ordering is not claimed.
    A7 uses a circular shift to preserve the histogram and alter spatial alignment.
    Nonperiodic seams and residual alignment must be audited before approval.
    """
    x = np.asarray(image, dtype=np.float32)
    if x.ndim < 3 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError('Finite multi-channel image required')
    lo, hi = gain_bounds
    if not (0 < lo <= hi < np.inf) or not (0 <= residual_bound < np.inf):
        raise ValueError('Invalid bounds')
    if mode not in ('adaptive', 'shuffled', 'shifted', 'constant'):
        raise ValueError('Unknown gate mode')
    r = reliability_map(x, noise_floor) if reliability is None else np.asarray(reliability)
    if r.shape != x.shape[1:] or not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
        raise ValueError('Invalid reliability map')
    r = r.copy()
    gate_rng = np.random.default_rng(int(rng.integers(0, 2**32-1)))
    if mode == 'shuffled':
        r = gate_rng.permutation(r.ravel()).reshape(r.shape)
    elif mode == 'shifted':
        # Translation preserves the histogram and periodic spatial differences.
        # Nonperiodic boundary effects and residual anatomical alignment need audit.
        shifts = tuple(int(gate_rng.integers(1, n)) if n > 1 else 0 for n in r.shape)
        r = np.roll(r, shifts, axis=tuple(range(r.ndim)))
    elif mode == 'constant':
        r = np.full_like(r, r.mean())
    params = dict(gain_log_std=gain_log_std, offset_std=offset_std, grid=grid)
    shared = sample_smooth_affine_fields(x.shape[1:], rng=rng, phase_count=x.shape[0], shared_across_phases=True, **params)
    independent = sample_smooth_affine_fields(x.shape[1:], rng=rng, phase_count=x.shape[0], shared_across_phases=False, **params)
    gain = np.clip(shared.gain, lo, hi)
    residual = (1-r)[None] * residual_bound * np.tanh(independent.offset / max(offset_std, np.finfo(np.float32).tiny))
    out = x * gain + shared.offset + residual
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite transform output')
    return AdaptiveResult(out, r, gain, residual)
