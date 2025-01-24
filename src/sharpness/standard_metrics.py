"""Module containing some standard metrics that don't fit in any other category"""
import numpy as np
from skimage.metrics import structural_similarity


def mse(X, T):
    """Bivariate -- Mean Squared Error"""
    return np.mean((X - T) ** 2)


def mae(X, T):
    """Bivariate -- Mean Absolute Error"""
    return np.mean(np.abs((X - T)))


def rmse(X, T):
    """Bivariate -- Root Mean Squared Error"""
    return np.sqrt(mse(X, T))


def ssim(X, T, win_size=7, data_range=255):
    """Bivariate -- SSIM from scikit-image"""
    return structural_similarity(X, T, win_size=win_size, data_range=data_range)


def total_variation(X):
    """Univariate -- Total variation of an image"""
    horizontal_tv = np.sum(np.abs(X[:, :-1] - X[:, 1:]))
    vertical_tv = np.sum(np.abs(X[:-1, :] - X[1:, :]))
    tv = horizontal_tv + vertical_tv
    return tv