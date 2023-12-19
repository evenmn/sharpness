import numpy as np


def compute_power_spectrum(image, hanning=True):
    N = image.shape[0]
    if hanning:
        # Set up 2D Hanning window to deal with edge effects
        window = np.hanning(N)
        window = np.outer(window, window)
        image = image * window

    # Compute the power spectrum of an image
    f_transform = np.fft.fft2(image)
    power_spectrum = np.abs(f_transform) ** 2
    return power_spectrum


def fourier_rmse(image1, image2, hanning=True):
    # Compute power spectra of both images
    power_spectrum1 = compute_power_spectrum(image1, hanning=hanning)
    power_spectrum2 = compute_power_spectrum(image2, hanning=hanning)

    # Compute the mean squared error between power spectra
    mse = np.mean((power_spectrum1-power_spectrum2)**2)
    return np.sqrt(mse)


def fourier_total_variation(image, hanning=True):
    N = image.shape[0]
    if hanning:
        # Set up 2D Hanning window to deal with edge effects
        window = np.hanning(N)
        window = np.outer(window, window)
        image = image * window

    f_transform = np.fft.fft2(image)
    tv = np.sum(np.abs(f_transform))
    return tv


if __name__ == '__main__':
    from skimage.data import camera
    image1 = camera()
    image2 = camera()

    fourier_rmse = fourier_rmse(image1, image2)
    print("Fourier power spectrum RMSE:", fourier_rmse)