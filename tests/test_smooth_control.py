import numpy as np

from mamaguardnet.augmentation.smooth_control import (
    apply_smooth_affine,
    sample_smooth_affine_fields,
)
from mamaguardnet.kinetics.constraints import constraint_report


def test_shared_and_independent_fields_match_shape_and_marginal_scale():
    shape = (17, 19, 21)
    shared = sample_smooth_affine_fields(
        shape,
        rng=np.random.default_rng(4),
        shared_across_phases=True,
    )
    independent = sample_smooth_affine_fields(
        shape,
        rng=np.random.default_rng(4),
        shared_across_phases=False,
    )
    assert shared.gain.shape == independent.gain.shape == (3, *shape)
    assert np.all(shared.gain > 0) and np.all(independent.gain > 0)
    np.testing.assert_array_equal(shared.gain[0], shared.gain[1])
    assert not np.array_equal(independent.gain[0], independent.gain[1])
    assert abs(float(np.std(np.log(shared.gain))) - float(np.std(np.log(independent.gain)))) < 0.02
    assert abs(float(np.std(shared.offset)) - float(np.std(independent.offset))) < 0.02


def test_shared_smooth_field_preserves_constraints_but_independent_can_break_them():
    phases = np.stack(
        [
            np.zeros((12, 12, 12), dtype=np.float32),
            np.ones((12, 12, 12), dtype=np.float32),
            np.full((12, 12, 12), 2.0, dtype=np.float32),
        ]
    )
    shared = sample_smooth_affine_fields(
        phases.shape[1:],
        rng=np.random.default_rng(9),
        shared_across_phases=True,
        gain_log_std=0.2,
        offset_std=0.3,
    )
    independent = sample_smooth_affine_fields(
        phases.shape[1:],
        rng=np.random.default_rng(9),
        shared_across_phases=False,
        gain_log_std=0.2,
        offset_std=1.0,
    )
    assert constraint_report(phases, apply_smooth_affine(phases, shared)).valid
    assert not constraint_report(phases, apply_smooth_affine(phases, independent)).valid
