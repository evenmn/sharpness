import numpy as np
import pytest

import sharpness.fourier as fm


@pytest.fixture
def comparison_X():
    return np.zeros((5, 5), dtype=np.float64)


@pytest.fixture
def comparison_T():
    return np.full((5, 5), 1.0, dtype=np.float64)


def test_compute_power_spectrum_all_zeros(comparison_X):
    np.testing.assert_equal(
        fm.compute_power_spectrum(comparison_X, hanning=False),
        np.zeros((5, 5), dtype=np.float64),
    )


def test_compute_power_spectrum_all_ones(comparison_T):
    np.testing.assert_equal(
        fm.compute_power_spectrum(comparison_T, hanning=False),
        np.array(
            [
                [25.0**2, 0, 0, 0, 0],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        ),
    )


def test_fourier_rmse(comparison_X, comparison_T):
    assert fm.fourier_rmse(comparison_X, comparison_T, hanning=False) == np.sqrt(
        (625**2 / 25)
    )

def test_fourier_total_variation(comparison_T):
    assert fm.fourier_total_variation(comparison_T, hanning=False) == 25.