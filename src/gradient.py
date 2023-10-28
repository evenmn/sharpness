import cv2
import numpy as np
from skimage.feature import hog
from skimage.filters import sobel_h, sobel_v
from scipy.stats import pearsonr


def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    max_pixel_value = np.max(image1)
    eps = np.finfo(np.float32).tiny
    psnr_value = 20 * np.log10(max_pixel_value / (eps + np.sqrt(mse)))
    return psnr_value


def normalized_cross_correlation(image1, image2):
    ncc = np.sum(image1 * image2) / (np.sqrt(np.sum(image1 ** 2)) * np.sqrt(np.sum(image2 ** 2)))
    return ncc


def histogram_intersection(image1, image2, bins=256):
    hist1, _ = np.histogram(image1.flatten(), bins=bins, range=[0, 256])
    hist2, _ = np.histogram(image2.flatten(), bins=bins, range=[0, 256])
    intersection = np.minimum(hist1, hist2).sum() / np.maximum(hist1, hist2).sum()
    return intersection


def gradient_difference_similarity(image1, image2):
    # Ensure the image is a NumPy array with float data type
    image1 = image1.astype(float)
    image2 = image2.astype(float)
    
    gradient_x1 = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y1 = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude1 = np.sqrt(gradient_x1 ** 2 + gradient_y1 ** 2)

    gradient_x2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude2 = np.sqrt(gradient_x2 ** 2 + gradient_y2 ** 2)

    gds = np.sum(np.abs(gradient_magnitude1 - gradient_magnitude2)) / np.sum(gradient_magnitude1 + gradient_magnitude2)
    return gds

def gradient_rmse(image1, image2):
    #image1 = gray_and_flatten(image1)
    #image2 = gray_and_flatten(image2)

    gradient_x1 = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y1 = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude1 = np.sqrt(gradient_x1 ** 2 + gradient_y1 ** 2)

    gradient_x2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude2 = np.sqrt(gradient_x2 ** 2 + gradient_y2 ** 2)

    difference = np.abs(gradient_magnitude1 - gradient_magnitude2)
    rmse = np.sqrt(np.mean(difference**2))
    return rmse

def laplacian_rmse(image1, image2):
    image1 = image1.astype(np.uint8)
    image2 = image2.astype(np.uint8)

    # Compute Laplacian images
    laplacian1 = cv2.Laplacian(image1, cv2.CV_64F)
    laplacian2 = cv2.Laplacian(image2, cv2.CV_64F)

    # Calculate pixel-wise differences
    difference = np.abs(laplacian1 - laplacian2)

    # Calculate Mean Squared Error
    rmse = np.sqrt(np.mean(difference**2))

    return rmse

# def gradient_profile_difference(image1, image2):
#     image1 = gray_and_flatten(image1)
#     image2 = gray_and_flatten(image2)
#
#     gradient_x1 = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)
#     gradient_y1 = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
#     gradient_magnitude1 = np.sqrt(gradient_x1 ** 2 + gradient_y1 ** 2)
#
#     gradient_x2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
#     gradient_y2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
#     gradient_magnitude2 = np.sqrt(gradient_x2 ** 2 + gradient_y2 ** 2)
#
#     gpd = np.sum(np.abs(np.array(np.gradient(gradient_magnitude1)) - np.array(np.gradient(gradient_magnitude2))))
#     return gpd


def histogram_of_oriented_gradients(image):
    hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    return hog_features


def hog_pearson(image1, image2):
    hog_features_1 = histogram_of_oriented_gradients(image1)
    hog_features_2 = histogram_of_oriented_gradients(image2)

    #squared_diff = [(x - y) ** 2 for x, y in zip(hog_features_1, hog_features_2)]
    #distance = sum(squared_diff) ** 0.5

    # HOG features for two images (hog_features_1 and hog_features_2)
    return pearsonr(hog_features_1, hog_features_2)[0]


def mean_gradient_magnitude(image):
    # Ensure the image is a NumPy array with float data type
    image = image.astype(float)
    
    # Calculate gradients of the image
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Compute the Mean Gradient Magnitude
    mgm = np.mean(gradient_magnitude)

    return mgm

def grad_total_variation(image):
    # Ensure the image is a NumPy array with float data type
    image = image.astype(float)

    # Calculate the horizontal and vertical gradients using central differences
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the total variation as the L1 norm of the gradients
    tv = np.sum(np.abs(gradient_x)) + np.sum(np.abs(gradient_y))

    return tv


# Example usage:
if __name__ == "__main__":
    # Load an example image (make sure to replace with your own image)
    from skimage.data import camera
    image1 = camera()
    image2 = camera()

    #  Calculate psnr
    psnr_value = psnr(image1, image2)
    print("psnr:", psnr_value)

    # Calculate Normalized Cross-Correlation
    ncc_value = normalized_cross_correlation(image1, image2)
    print("NCC:", ncc_value)

    # Calculate Mean Gradient Magnitude
    mgm_value = mean_gradient_magnitude(image1)
    print("MGM:", mgm_value)
    
    # Calculate Gradient Difference Similarity
    gds_value = gradient_difference_similarity(image1, image2)
    print("GDS:", gds_value)
    
    # Calculate Gradient-MSE
    gmd_value = gradient_rmse(image1, image2)
    print("G-RMSE:", gmd_value)

    # Calculate Laplacian-MSE
    mse_lap = laplacian_rmse(image1, image2)
    print("RMSE-Laplace:", mse_lap)

    # Calculate Histogram Intersection
    hist_intersection = histogram_intersection(image1, image2)
    print("Histogram Intersection:", hist_intersection)
    
    # Calculate Gradient Profile Difference
    #gpd_value = gradient_profile_difference(image1, image2)
    #print("GPD:", gpd_value)

    # Calculate Histogram of Oriented Gradients (HOG) for the image
    hog = hog_pearson(image1, image2)
    print("HOG pearson", hog)
