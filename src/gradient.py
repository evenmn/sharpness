import numpy as np
import cv2
from skimage.feature import hog


def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    max_pixel_value = np.max(image1)
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr_value

def normalized_cross_correlation(image1, image2):
    ncc = np.sum(image1 * image2) / (np.sqrt(np.sum(image1 ** 2)) * np.sqrt(np.sum(image2 ** 2)))
    return ncc

def histogram_intersection(image1, image2, bins=256): # This assumes 1 color channel 
    hist1, _ = np.histogram(image1.flatten(), bins=bins, range=[0, 256])
    hist2, _ = np.histogram(image2.flatten(), bins=bins, range=[0, 256])
    intersection = np.minimum(hist1, hist2).sum() / np.maximum(hist1, hist2).sum()
    return intersection

def gradient_difference_similarity(image1, image2):
    # Calculate gradients of the images
    gradient_x1 = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y1 = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude1 = np.sqrt(gradient_x1**2 + gradient_y1**2)

    gradient_x2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude2 = np.sqrt(gradient_x2**2 + gradient_y2**2)

    # Compute the Gradient Difference Similarity
    gds = np.sum(np.abs(gradient_magnitude1 - gradient_magnitude2)) / np.sum(gradient_magnitude1 + gradient_magnitude2)

    return gds

def gradient_magnitude_difference(image1, image2):
    # Calculate gradients of the images
    gradient_x1 = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y1 = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude1 = np.sqrt(gradient_x1**2 + gradient_y1**2)

    gradient_x2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude2 = np.sqrt(gradient_x2**2 + gradient_y2**2)

    # Compute the Gradient Magnitude Difference
    gmd = np.sum(np.abs(gradient_magnitude1 - gradient_magnitude2))

    return gmd

def gradient_profile_difference(image1, image2):
    # Calculate gradients of the images
    gradient_x1 = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y1 = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude1 = np.sqrt(gradient_x1**2 + gradient_y1**2)

    gradient_x2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude2 = np.sqrt(gradient_x2**2 + gradient_y2**2)

    # Compute the Gradient Profile Difference
    gpd = np.sum(np.abs(np.array(np.gradient(gradient_magnitude1)) - np.array(np.gradient(gradient_magnitude2))))

    return gpd

def histogram_of_oriented_gradients(image):
    # Compute Histogram of Oriented Gradients (HOG) features
    hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1), visualize=True)
    
    return hog_features

def mean_gradient_magnitude(image):
    # Calculate gradients of the image
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Compute the Mean Gradient Magnitude
    mgm = np.mean(gradient_magnitude)

    return mgm

# Example usage:
if __name__ == "__main__":
    # Load an example image (make sure to replace with your own image)
    from skimage.data import camera
    image1 = camera()
    image2 = camera()
    #image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

    # Calculate PSNR
    psnr_value = psnr(image1, image2)
    print("PSNR:", psnr_value)

    # Calculate Normalized Cross-Correlation
    ncc_value = normalized_cross_correlation(image1, image2)
    print("NCC:", ncc_value)
    
    # Calculate Gradient Difference Similarity
    gds_value = gradient_difference_similarity(image1, image2)
    print("GDS:", gds_value)
    
    # Calculate Gradient Magnitude Difference
    gmd_value = gradient_magnitude_difference(image1, image2)
    print("GMD:", gmd_value)

    # Calculate Histogram Intersection
    hist_intersection = histogram_intersection(image1, image2)
    print("Histogram Intersection:", hist_intersection)
    
    # Calculate Gradient Profile Difference
    gpd_value = gradient_profile_difference(image1, image2)
    print("GPD:", gpd_value)

    # Calculate Histogram of Oriented Gradients (HOG) for the image
    hog_features = histogram_of_oriented_gradients(image1)

    # Calculate Mean Gradient Magnitude
    mgm_value = mean_gradient_magnitude(image1)
    print("MGM:", mgm_value)
