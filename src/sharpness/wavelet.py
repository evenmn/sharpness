"""Module containing low-level functions using wavelet techniques"""
import numpy as np
import pywt


def wavelet_total_variation(image, wavelet='haar', level=1):
    """Univariate -- metric to compute total variation using wavelet coefficients"""
    # Ensure the image is a NumPy array with float data type
    image = image.astype(float)

    # Apply wavelet transform
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # Calculate the Wavelet Total Variation
    wavelet_tv = 0
    for c in coeffs:
        wavelet_tv += np.sum(np.abs(c))

    return wavelet_tv