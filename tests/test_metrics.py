import numpy as np
import pytest
from sharpness import metrics


# Standard metrics


@pytest.fixture
def fake_zeros():
    return np.zeros((5, 5), dtype=np.float64)


@pytest.fixture
def fake_ones():
    return np.full((5, 5), 1.0, dtype=np.float64)


@pytest.fixture
def fake_twos():
    return np.full((5, 5), 2.0, dtype=np.float64)


def test_rmse(fake_zeros, fake_twos):
    assert metrics.rmse(fake_zeros, fake_twos) == 2.0


def test_ssim(fake_zeros, fake_twos):
    assert metrics.ssim(fake_zeros, fake_twos, data_range=2, win_size=5) == (
        (0.01 * 2) ** 2
    ) / (4 + (0.01 * 2) ** 2)


def test_total_variation(fake_twos):
    assert metrics.total_variation(fake_twos) == 0.0


# Gradient metrics


@pytest.fixture
def fake_slope_one_gradient():
    return np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def fake_slope_two_gradient():
    return np.array(
        [
            [0, 0, 0, 0],
            [2, 2, 2, 2],
            [4, 4, 4, 4],
            [6, 6, 6, 6],
        ],
        dtype=np.float32,
    )


# All of these values were worked out by hand using reference matrices from
# docs.opencv.org
def test_mean_gradient_magnitude(fake_slope_one_gradient):
    assert metrics.mean_gradient_magnitude(fake_slope_one_gradient) == 4.0


def test_grad_total_variation(fake_slope_one_gradient):
    assert metrics.grad_total_variation(fake_slope_one_gradient) == 64.0


def test_gradient_rmse(fake_slope_one_gradient, fake_slope_two_gradient):
    assert metrics.gradient_rmse(
        fake_slope_one_gradient, fake_slope_two_gradient
    ) == np.sqrt(32.0)


def test_laplacian_rmse(fake_slope_one_gradient, fake_slope_two_gradient):
    assert metrics.laplacian_rmse(
        fake_slope_one_gradient, fake_slope_two_gradient
    ) == np.sqrt(2.0)


def test_compute_power_spectrum_all_zeros(fake_zeros):
    np.testing.assert_equal(
        metrics.compute_power_spectrum(fake_zeros, hanning=False),
        np.zeros((5, 5), dtype=np.float64),
    )


def test_compute_power_spectrum_all_ones(fake_ones):
    np.testing.assert_equal(
        metrics.compute_power_spectrum(fake_ones, hanning=False),
        np.array(
            [
                [25.0**2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
    )


def test_fourier_rmse(fake_zeros, fake_ones):
    assert metrics.fourier_rmse(fake_zeros, fake_ones, hanning=False) == np.sqrt(
        (625**2 / 25)
    )


def test_fourier_total_variation(fake_ones):
    assert metrics.fourier_total_variation(fake_ones, hanning=False) == 25.0


# Wavelet Metrics


def test_wavelet_tv_constant_image():
    fake_input = np.zeros((5, 5), dtype=float)
    assert metrics.wavelet_total_variation(fake_input) == 0.0


def test_wavelet_tv_varying_image():
    fake_input = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
        ],
        dtype=np.float64
    )
    assert (metrics.wavelet_total_variation(fake_input) - 16.) <= 1e-10


# Spectral slope metrics


def test_polar_average():
    fake_image = np.zeros((5, 5), dtype=np.float64)
    f, s = metrics.polar_average(fake_image)
    np.testing.assert_allclose(s, np.zeros((2,), dtype=float))


def test_spec_slope():
    rng = np.random.default_rng(seed=100)
    fake_image = rng.uniform(0, 1, size=(5, 5))
    assert metrics.spec_slope(fake_image, hanning=False) == -0.09153401789037853


def test_s1():
    rng = np.random.default_rng(seed=100)
    fake_image = rng.uniform(0, 1, size=(5, 5))
    assert metrics.s1(fake_image, hanning=False) == -0.09153401789037853

def test_s1_contrast_threshold():
    rng = np.random.default_rng(seed=100)
    fake_image = rng.uniform(0, 1, size=(5, 5))
    assert metrics.s1(fake_image, contrast_threshold=2, hanning=False) is np.nan


def test_s1_brightness_threshold():
    rng = np.random.default_rng(seed=100)
    fake_image = rng.uniform(0, 1, size=(5, 5))
    assert metrics.s1(fake_image, brightness_threshold=2, hanning=False) is np.nan


def test_s1_brightness_mult():
    rng = np.random.default_rng(seed=100)
    fake_image = rng.uniform(0, 1, size=(5, 5))
    assert (
        metrics.s1(fake_image, brightness_mult=True, hanning=False)
        - (-0.09153401789037853 * 0.5)
    ) < 1e-5
