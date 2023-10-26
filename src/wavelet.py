import cv2
import numpy as np
import pywt

def compute_wavelet_energy(image, wavelet='haar', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    energy = 0
    for c in coeffs:
        energy += np.sum(np.abs(c) ** 2)
    return energy

def compute_wavelet_entropy(image, wavelet='haar', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    entropy = 0
    for c in coeffs:
        normalized_c = np.abs(c) / np.sum(np.abs(c))
        entropy += -np.sum(normalized_c * np.log2(normalized_c + np.finfo(float).eps))
    return entropy

def wavelet_image_similarity(image1, image2):
    energy1 = compute_wavelet_energy(image1)
    energy2 = compute_wavelet_energy(image2)
    entropy1 = compute_wavelet_entropy(image1)
    entropy2 = compute_wavelet_entropy(image2)

    # Calculate similarity scores based on energy and entropy
    energy_similarity = np.exp(-abs(energy1 - energy2))
    entropy_similarity = np.exp(-abs(entropy1 - entropy2))

    # A weighted combination of energy and entropy similarity can be used
    similarity_score = 0.5 * energy_similarity + 0.5 * entropy_similarity

    return similarity_score

def wavelet_total_variation(image, wavelet='haar', level=1):
    # Ensure the image is a NumPy array with float data type
    image = image.astype(float)

    # Apply wavelet transform
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # Calculate the Wavelet Total Variation
    wavelet_tv = 0
    for c in coeffs:
        wavelet_tv += np.sum(np.abs(c))

    return wavelet_tv

# Example usage:
if __name__ == '__main__':
    from skimage.data import camera
    image1 = camera()
    image2 = camera()

    similarity_score = wavelet_image_similarity(image1, image2)
    print("Similarity Score:", similarity_score)
    wavelet_tv1 = wavelet_total_variation(image1)
    wavelet_tv2 = wavelet_total_variation(image2)
    print("Total variation of the two images:", wavelet_tv1, wavelet_tv2)
