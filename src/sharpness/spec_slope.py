"""Module containing low-level functions for spectral slope metrics"""

import numpy as np
from numpy.polynomial import Polynomial as P
import scipy.ndimage as nd


def polar_average(spectrum_2d, num_angles=360):
    """
    Utility function to compute polar average of 2D FFT

    args:
        spectrum_2d (np.array): Output of 2D FFT, either in magnitude or complex form.
        num_angles (int): How many angles to average across

    returns:
        f (np.array): 1D array of frequency radii
        s (np.array): 1D array of polar averaged values at the corresponding frequency
            radii from f.
    """
    N = spectrum_2d.shape[0]

    # Set the FFT mean to the average of the first two low-frequency values
    # This avoids throwing off the interpolated values
    spectrum_2d[0, 0] = np.mean([spectrum_2d[0, 1], spectrum_2d[1, 0]])

    # Generate grid coordinates in terms of polar coordinates
    # excluding the global average but including the Nyquist frequency at N//2.
    xs = []
    ys = []
    thetas = np.linspace(0, 2 * np.pi, num_angles + 1)[:-1]
    for r in range(1, N // 2 + 1):
        xs.append(r * np.cos(thetas))
        ys.append(r * np.sin(thetas))
    grid_coords = np.array([np.concatenate(xs), np.concatenate(ys)])

    # Interpolate values at those coordinates
    s_full = nd.map_coordinates(spectrum_2d, grid_coords, mode="grid-wrap", order=1)
    s_full = s_full.reshape(-1, num_angles)

    # Average together
    s = s_full.mean(axis=1)

    # Generate frequency coordinates
    f = np.linspace(0, 0.5, s.shape[0] + 1)

    # Exclude 0th frequency, as we didn't compute an s value for that
    f = f[1:]

    return f, s


def spec_slope(image, hanning=True):
    """Univariate -- Computes raw spectral slope metric"""
    N = image.shape[0]
    if hanning:
        # Set up 2D Hanning window to deal with edge effects
        window = np.hanning(N)
        window = np.outer(window, window)
        image = image * window

    # Compute polar averaged spectral values
    # f is the frequency radius
    # s is the average value for that frequency
    [f, s] = polar_average(np.abs(np.fft.fft2(image)))

    # Fit a line to the log-log transformed data
    line = P.fit(np.log(f), np.log(s), 1)
    res = line.coef[1]
    return res


def s1(
    image,
    contrast_threshold=None,
    brightness_threshold=None,
    brightness_mult=False,
    hanning=True,
):
    """Univariate -- Computes S1 metric from Vu et al (2009)"""

    if (contrast_threshold is not None) and (
        np.nanmax(image) - np.nanmin(image) < contrast_threshold
    ):
        val = np.nan
    elif (brightness_threshold is not None) and (np.nanmax(image) < brightness_threshold):
        val = np.nan
    else:
        if brightness_mult:
            val = spec_slope(image, hanning) * np.nanmean(image)
        else:
            val = spec_slope(image, hanning)

    return val
