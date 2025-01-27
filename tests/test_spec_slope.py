import numpy as np
from sharpness import spec_slope


def test_polar_average():
    fake_image = np.zeros((5, 5), dtype=np.float64)
    f, s = spec_slope.polar_average(fake_image)
    np.testing.assert_allclose(s, np.zeros((2,), dtype=float))


def test_spec_slope():
    rng = np.random.default_rng(seed=100)
    fake_image = rng.uniform(0, 1, size=(5, 5))
    assert spec_slope.spec_slope(fake_image, hanning=False) == -0.09153401789037853


def test_s1():
    rng = np.random.default_rng(seed=100)
    fake_image = rng.uniform(0, 1, size=(5, 5))
    assert spec_slope.s1(fake_image, hanning=False) == -0.09153401789037853

def test_s1_contrast_threshold():
    rng = np.random.default_rng(seed=100)
    fake_image = rng.uniform(0, 1, size=(5, 5))
    assert spec_slope.s1(fake_image, contrast_threshold=2, hanning=False) is np.nan


def test_s1_brightness_threshold():
    rng = np.random.default_rng(seed=100)
    fake_image = rng.uniform(0, 1, size=(5, 5))
    assert spec_slope.s1(fake_image, brightness_threshold=2, hanning=False) is np.nan


def test_s1_brightness_mult():
    rng = np.random.default_rng(seed=100)
    fake_image = rng.uniform(0, 1, size=(5, 5))
    assert (
        spec_slope.s1(fake_image, brightness_mult=True, hanning=False)
        - (-0.09153401789037853 * 0.5)
    ) < 1e-5
