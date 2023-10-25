import cv2
import numpy as np

def compute_power_spectrum(image):
    # Compute the power spectrum of an image
    f_transform = np.fft.fft2(image)
    power_spectrum = np.abs(f_transform) ** 2
    return power_spectrum

def fourier_image_similarity(image1, image2):
    # Compute power spectra of both images
    power_spectrum1 = compute_power_spectrum(image1)
    power_spectrum2 = compute_power_spectrum(image2)

    # Compute the normalized cross-correlation between power spectra
    cross_correlation = np.real(np.fft.ifft2(np.fft.fft2(power_spectrum1) * np.conj(np.fft.fft2(power_spectrum2))))
    cross_correlation /= np.max(cross_correlation)

    # The maximum value of the cross-correlation indicates similarity
    similarity_score = np.max(cross_correlation)

    return similarity_score


if __name__ == '__main__':
    from skimage.data import camera
    image1 = camera()
    image2 = camera()

    similarity_score = fourier_image_similarity(image1, image2)
    print("Similarity Score:", similarity_score)
