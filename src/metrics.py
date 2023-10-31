import numpy as np
from skimage import filters
from sharpness.spec_slope import s1_map
from skimage.color import rgb2gray

def mse(X, T):
    """Mean Squared Error"""
    return np.mean((X - T) ** 2)


def mae(X, T):
    """Mean Absolute Error"""
    return np.mean(np.abs((X - T)))


def rmse(X, T):
    """Root Mean Squared Error"""
    return np.sqrt(mse(X, T))


def grad(X):
    """Average Magnitude of the Gradient

    Edge magnitude is computed as:
        sqrt(Gx^2 + Gy^2)
    """
    def _f(x): return np.mean(filters.sobel(x))
    return _f(X)


def total_variation(X):
    """ Total variation of an image """
    horizontal_tv = np.sum(np.abs(X[:, :-1] - X[:, 1:]))
    vertical_tv = np.sum(np.abs(X[:-1, :] - X[1:, :]))
    tv = horizontal_tv + vertical_tv
    return tv


if __name__ == '__main__':
    from skimage.data import camera
    X = camera()
    T = np.fliplr(X)

    from sharpness import compute_all_metrics
    results = compute_all_metrics(X, T)
    for metric, result in results.items():
        print(f'{metric}: {result}')
