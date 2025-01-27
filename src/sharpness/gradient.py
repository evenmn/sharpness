"""Module containing low-level functions based on gradient analysis"""

import cv2
import numpy as np
from skimage.feature import hog
from scipy.stats import pearsonr


def mean_gradient_magnitude(image):
    """Univariate -- computes the mean of the gradient magnitude map"""
    # Ensure the image is a NumPy array with float data type
    image = image.astype(float)

    # Calculate gradients of the image -- border handling is with reflect_101 method.
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Compute the Mean Gradient Magnitude
    mgm = np.mean(gradient_magnitude)

    return mgm


def grad_total_variation(image):
    """Univariate -- computes total variation of the gradient magnitude map"""
    # Ensure the image is a NumPy array with float data type
    image = image.astype(float)

    # Calculate the horizontal and vertical gradients using central differences
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the total variation as the L1 norm of the gradients
    tv = np.sum(np.abs(gradient_x)) + np.sum(np.abs(gradient_y))

    return tv


def gradient_rmse(image1, image2):
    """Bivariate -- Computes the RMSE between gradient magnitude maps of two images"""
    image1 = image1.astype(float)
    image2 = image2.astype(float)

    gradient_x1 = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y1 = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude1 = np.sqrt(gradient_x1**2 + gradient_y1**2)

    gradient_x2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude2 = np.sqrt(gradient_x2**2 + gradient_y2**2)

    difference = np.abs(gradient_magnitude1 - gradient_magnitude2)
    rmse = np.sqrt(np.mean(difference**2))
    return rmse


def laplacian_rmse(image1, image2):
    """Bivariate -- Computes the RMSE between Laplacian maps of two images"""
    image1 = image1.astype(float)
    image2 = image2.astype(float)

    # Compute Laplacian images
    laplacian1 = cv2.Laplacian(image1, cv2.CV_64F)
    laplacian2 = cv2.Laplacian(image2, cv2.CV_64F)

    # Calculate pixel-wise differences
    difference = np.abs(laplacian1 - laplacian2)

    # Calculate Mean Squared Error
    rmse = np.sqrt(np.mean(difference**2))

    return rmse
