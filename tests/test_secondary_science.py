from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import zoom

from mamaguardnet.augmentation.adaptive_v1 import adaptive_affine, reliability_map
from mamaguardnet.augmentation.smooth_control import sample_smooth_affine_fields


def test_a5_config_matches_frozen_a4_probability_and_amplitudes():
    root = Path(__file__).resolve().parents[1]
    a4 = yaml.safe_load((root/'configs/experiments/a4_kinetics.yaml').read_text())['augmentation_parameters']
    a5 = yaml.safe_load((root/'configs/candidates/a5_matched_v1.yaml').read_text())['augmentation_parameters']
    for key in ('probability','gain_log_std','offset_std'):
        assert a4[key] == a5[key]
    assert a4['field_grid'] == a5['grid']


def test_a4_sampler_exact_frozen_algorithm_equivalence():
    shape = (9, 11, 13)
    rng = np.random.default_rng(42)
    # Same construction as frozen A4, without importing nnUNet.
    gain = np.exp(zoom(rng.normal(0, .12, (4,)*3), tuple(n/4 for n in shape), order=1))
    offset = zoom(rng.normal(0, .05, (4,)*3), tuple(n/4 for n in shape), order=1)
    fields = sample_smooth_affine_fields(shape, rng=np.random.default_rng(42), shared_across_phases=True)
    np.testing.assert_allclose(fields.gain[0], gain, rtol=1e-6)
    np.testing.assert_allclose(fields.offset[0], offset, atol=1e-8)


def test_a5_empirical_marginals_and_smoothness():
    stats = {True: [], False: []}
    for shared in stats:
        for seed in range(128):
            f = sample_smooth_affine_fields((9, 11, 13), rng=np.random.default_rng(seed), shared_across_phases=shared)
            stats[shared].append([np.log(f.gain).std(), f.offset.std(),
                                  np.abs(np.diff(f.offset, axis=1)).mean()])
    a, b = (np.mean(stats[k], axis=0) for k in (True, False))
    np.testing.assert_allclose(a, b, rtol=.12, atol=.001)


def test_shift_preserves_histogram_and_periodic_smoothness():
    r = np.random.default_rng(2).random((8, 9, 10))
    x = np.stack([r, r+.1, r+.2])
    out = adaptive_affine(x, rng=np.random.default_rng(3), reliability=r, mode='shifted')
    np.testing.assert_array_equal(np.sort(r.ravel()), np.sort(out.reliability.ravel()))
    for axis in range(3):
        before = np.mean((r-np.roll(r, 1, axis))**2)
        after = np.mean((out.reliability-np.roll(out.reliability, 1, axis))**2)
        np.testing.assert_allclose(before, after)


def test_strong_weak_washout_and_noise():
    rng = np.random.default_rng(3)
    base = rng.normal(0, .02, (3, 8, 9, 10))
    weak = base + np.array([0, .001, .002])[:, None, None, None]
    strong = base + np.array([0, 2, 4])[:, None, None, None]
    washout = base + np.array([0, 4, 2])[:, None, None, None]
    assert reliability_map(strong).mean() > reliability_map(weak).mean()
    assert reliability_map(washout).mean() > .9


def test_input_conditioned_gate_reduces_residual():
    x = np.zeros((3, 8, 9, 10), dtype=np.float32)
    strong = x + np.array([0, 2, 4])[:, None, None, None]
    a = adaptive_affine(x, rng=np.random.default_rng(8))
    b = adaptive_affine(strong, rng=np.random.default_rng(8))
    assert np.abs(b.residual).mean() < np.abs(a.residual).mean() * .05


def test_gate_controls_share_identical_affine_draws():
    x = np.random.default_rng(9).normal(size=(3, 8, 9, 10))
    a = adaptive_affine(x, rng=np.random.default_rng(10))
    for mode in ('shifted', 'shuffled', 'constant'):
        b = adaptive_affine(x, rng=np.random.default_rng(10), mode=mode)
        np.testing.assert_array_equal(a.gain, b.gain)


def test_shift_control_exposes_nonperiodic_seam_limitation():
    # Histogram/periodic tests alone must not be interpreted as artifact-free.
    r = np.broadcast_to(np.linspace(0,1,32)[:,None,None],(32,8,8)).copy()
    x = np.stack([r,r+.1,r+.2])
    shifted = adaptive_affine(x,rng=np.random.default_rng(3),reliability=r,mode='shifted')
    assert np.max(np.abs(np.diff(r,axis=0))) < .04
    assert np.max(np.abs(np.diff(shifted.reliability,axis=0))) > .9
