import numpy as np
import pytest

from mamaguardnet.augmentation.adaptive_v1 import adaptive_affine, reliability_map


def image():
    return np.stack([np.full((8, 9, 10), v) for v in (0., 1., 2.)]).astype(np.float32)


def test_reliability_limits_and_invalid():
    x = image()
    assert reliability_map(x).min() > .9
    assert reliability_map(np.zeros_like(x)).max() == 0
    x[:, 0, 0, 0] = np.nan
    assert reliability_map(x)[0, 0, 0] == 0
    with pytest.raises(ValueError):
        adaptive_affine(x, rng=np.random.default_rng(1))


def test_exact_shared_limit_and_input_unchanged():
    x = image()
    before = x.copy()
    out = adaptive_affine(x, rng=np.random.default_rng(1), reliability=np.ones(x.shape[1:]))
    np.testing.assert_array_equal(x, before)
    np.testing.assert_allclose(out.image[2]-out.image[0], out.gain[0]*2, rtol=1e-6)
    assert np.count_nonzero(out.residual) == 0


def test_low_reliability_bounded_and_deterministic():
    x = image()
    kwargs = dict(reliability=np.zeros(x.shape[1:]))
    a = adaptive_affine(x, rng=np.random.default_rng(4), **kwargs)
    b = adaptive_affine(x, rng=np.random.default_rng(4), **kwargs)
    np.testing.assert_array_equal(a.image, b.image)
    assert np.abs(a.residual).max() <= .05
    assert not np.array_equal(a.residual[0], a.residual[1])
    assert .5 <= a.gain.min() <= a.gain.max() <= 2


def test_shuffled_control_preserves_histogram():
    x = image()
    r = np.random.default_rng(3).random(x.shape[1:])
    a = adaptive_affine(x, rng=np.random.default_rng(4), reliability=r, mode='shuffled')
    np.testing.assert_array_equal(np.sort(a.reliability.ravel()), np.sort(r.ravel()))
    assert not np.array_equal(a.reliability, r)
