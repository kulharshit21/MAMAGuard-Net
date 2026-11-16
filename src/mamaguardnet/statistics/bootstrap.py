from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20261031,
) -> dict[str, float | int]:
    candidate, baseline = np.asarray(candidate, float), np.asarray(baseline, float)
    if candidate.shape != baseline.shape or candidate.ndim != 1 or candidate.size == 0:
        raise ValueError("Expected non-empty paired one-dimensional arrays")
    differences = candidate - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    draws = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2
    return {
        "n": int(differences.size),
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1 - alpha)),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "seed": seed,
    }
