import numpy as np
import sharpness.wavelet as wave


def test_wavelet_tv_constant_image():
    fake_input = np.zeros((5, 5), dtype=float)
    assert wave.wavelet_total_variation(fake_input) == 0.0


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
    assert (wave.wavelet_total_variation(fake_input) - 16.) <= 1e-10
