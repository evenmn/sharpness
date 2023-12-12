import numpy as np
from numpy.polynomial import Polynomial as P
import scipy.ndimage as nd


# Function to compute S1 metric from Vu et al. paper
def s1(img, contrast_threshold=5):

    if img.max() - img.min() >= contrast_threshold:
        val = spec_slope(img)
    else:
        val = np.nan

    return val


# Basic contrast function, which at this point simply takes a difference
def contrast_map_overlap(block):
    return block.max() - block.min()


# Compute the spectral slope
def spec_slope(block, hanning=True):
    N = block.shape[0]
    if hanning:
        # Set up 2D Hanning window to deal with edge effects
        window = np.hanning(N)
        window = np.outer(window, window)
        block = block * window

    # Compute polar averaged spectral values
    # f is the frequency radius
    # s is the average value for that frequency
    [f, s] = polar_average(np.abs(np.fft.fft2(block)))

    # Fit a line to the log-log transformed data
    line = P.fit(np.log(f), np.log(s), 1)
    res = line.coef[1]
    return res


# Given output of FFT, compute polar averaged version
# Returns a tuple (f, a)
# f is a 1D array of frequency radii
# s is a 1D array of the same length with the polar averaged value for the corresponding radii
def polar_average(spect, num_angles=360):
    N = spect.shape[0]

    spect[0, 0] = np.mean([spect[0, 1], spect[1, 0]])

    # Generate grid coordinates in terms of polar coordinates, excluding the global average but including the Nyquist frequency at N//2.
    xs = []
    ys = []
    thetas = np.linspace(0, 2*np.pi, num_angles+1)[:-1]
    for r in range(1, N//2+1):
        xs.append(r * np.cos(thetas))
        ys.append(r * np.sin(thetas))
    grid_coords = np.array([np.concatenate(xs), np.concatenate(ys)])

    # Obtain values at those coordinates
    s_full = nd.map_coordinates(spect, grid_coords, mode='grid-wrap', order=1)
    s_full = s_full.reshape(-1, num_angles)

    # Average together
    s = s_full.mean(axis=1)

    # Generate frequency coordinates
    f = np.linspace(0, 0.5, s.shape[0] + 1)

    # Exclude 0th frequency, as we didn't compute an s value for that
    f = f[1:]

    return f, s