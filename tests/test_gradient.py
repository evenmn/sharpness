import pytest
import numpy as np
import sharpness.gradient as grad


@pytest.fixture
def comparison_X():
    return np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
        ],
        dtype=np.float32
    )

@pytest.fixture
def comparison_T():
    return np.array(
        [
            [0, 0, 0, 0],
            [2, 2, 2, 2],
            [4, 4, 4, 4],
            [6, 6, 6, 6],
        ],
        dtype=np.float32
    )

# All of these values were worked out by hand using reference matrices from 
# docs.opencv.org
def test_mean_gradient_magnitude(comparison_X):
    assert grad.mean_gradient_magnitude(comparison_X) == 4.

def test_grad_total_variation(comparison_X):
    assert grad.grad_total_variation(comparison_X) == 64.

def test_gradient_rmse(comparison_X, comparison_T):
    assert grad.gradient_rmse(comparison_X, comparison_T) == np.sqrt(32.)

def test_laplacian_rmse(comparison_X, comparison_T):
    assert grad.laplacian_rmse(comparison_X, comparison_T) == np.sqrt(2.)