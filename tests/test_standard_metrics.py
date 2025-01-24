import pytest
import numpy as np

import sharpness.standard_metrics as sm


@pytest.fixture
def comparison_X():
    return np.zeros((7, 7), dtype=np.float64)


@pytest.fixture
def comparison_T():
    return np.full((7, 7), 2.0, dtype=np.float64)


def test_mse(comparison_X, comparison_T):
    assert sm.mse(comparison_X, comparison_T) == 4.0


def test_mae(comparison_X, comparison_T):
    assert sm.mae(comparison_X, comparison_T) == 2.0


def test_rmse(comparison_X, comparison_T):
    assert sm.rmse(comparison_X, comparison_T) == 2.0


def test_ssim(comparison_X, comparison_T):
    assert sm.ssim(comparison_X, comparison_T, data_range=2) == ((0.01 * 2) ** 2) / (
        4 + (0.01 * 2) ** 2
    )

def test_total_variation(comparison_X):
    assert sm.total_variation(comparison_X) == 0.