import numpy as np
import pytest

from mamaguardnet.augmentation.kinetics import augment_independent, augment_kinetics
from mamaguardnet.kinetics.constraints import constraint_report, shared_affine


def test_positive_shared_affine_preserves_sign_and_ordering():
    phases = np.asarray([[[1.0, 2.0]], [[3.0, 1.0]], [[4.0, 4.0]]])
    transformed = shared_affine(phases, np.asarray([[0.5, 2.0]]), np.asarray([[9.0, -3.0]]))
    report = constraint_report(phases, transformed)
    assert report.valid
    assert report.sign_violation_rate == 0
    assert report.ordering_violation_rate == 0


def test_nonpositive_gain_is_rejected():
    with pytest.raises(ValueError):
        shared_affine(np.ones((3, 2, 2)), -1.0, 0.0)


def test_kinetics_augmentation_is_deterministic_and_valid():
    phases = np.arange(36, dtype=np.float32).reshape(3, 3, 4)
    first = augment_kinetics(phases, rng=np.random.default_rng(7))
    second = augment_kinetics(phases, rng=np.random.default_rng(7))
    assert np.array_equal(first.image, second.image)
    assert first.report.valid


def test_independent_policy_can_break_kinetics():
    phases = np.asarray([np.zeros((8,)), np.ones((8,)), np.full((8,), 2.0)])
    transformed = augment_independent(phases, rng=np.random.default_rng(0), gain_range=(0.1, 3.0), offset_std=2.0)
    assert not constraint_report(phases, transformed).valid

