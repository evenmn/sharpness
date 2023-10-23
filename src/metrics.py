import numpy as np
from skimage import filters
from sharpness.spec_slope import s1_map
from skimage.color import rgb2gray

def gray_and_flatten(image):
    if image.ndim == 3:
        if  image.shape[-1] > 1:
            image = rgb2gray(image)
    image = np.squeeze(image).astype(np.float32)
    return image


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

def s1(X):
    """Spectral slope method from Vu et al.
    
    Computes a sharpness map of the image, then returns the
    average sharpness of the top 1% of the results.
    
    Because this is not a comparative method, looks at each
    input separately and returns a tuple.
    """
    X = gray_and_flatten(X)
    X_map = s1_map(X, 32, 16)
    return X_map[X_map > np.percentile(X_map, 99)].mean()


if __name__ == '__main__':
    from skimage.data import camera
    X = camera()
    T = np.fliplr(X)

    from sharpness import compute_all_metrics
    results = compute_all_metrics(X, T)
    for metric, result in results.items():
        print(f'{metric}: {result}')
