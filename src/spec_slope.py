import numpy as np
from numpy.polynomial import Polynomial as P
import scipy.ndimage as nd


## Function to compute S1 metric from Vu et al. paper
# Parameters:
# img ---- input image, should be a grayscale 2D image, preferably as a numpy array
# block_size ---- how big each analyzed block should be; should be even, and if block_stride is left as default, should be divisble by 8.
# pad_len ---- how much padding to apply to the sides of the image; any integer
# block_stride ---- stride length for blocks, which sets overlap; should be divisible by 2
def s1_map(img, block_size, pad_len, block_stride=None, contrast_threshold = 0):
    
    ## Do some initial setup
    if block_stride == None:
        block_stride = block_size//4
    img = np.pad(img, pad_len, mode='reflect')
    (num_rows, num_cols) = img.shape
    res = np.zeros(img.shape) - 100
    
    # Run the main loop, checking only valid blocks and striding by block_stride
    for row in range(block_size//2, num_rows - block_size//2 + 1, block_stride):
        for col in range(block_size//2, num_cols - block_size//2 + 1, block_stride):
            
            # Subset the block of interest
            block = img[row - block_size//2:row + block_size//2, col - block_size//2:col + block_size//2]
            
            # Compute contrast to check for 0 contrast case
            contrastMap = contrast_map_overlap(block)
            
            # If there is sufficient contrast, compute spectral slope and apply logistic function; else, return 0
            if contrastMap.max() > contrast_threshold:
                val = spec_slope(block)
                val_1 = val[0] # 1 - 1 / (1 + np.exp(-3*(val[0] - 2)))
            else:
                val_1 = 0
                
            # Fill result in the appropriate region with the value for this block
            res[row - block_stride//2:row + block_stride//2, col - block_stride//2:col + block_stride//2] = val_1
            
    # Crop result matrix down to remove padding
    res = res[pad_len:num_rows - pad_len, pad_len:num_cols - pad_len]
    
    return res


## Basic contrast function, which at this point simply takes a difference
def contrast_map_overlap(block):
    return block.max() - block.min()


## Compute the spectral slope 
def spec_slope(block, hanning=True):
    N = block.shape[0]
    if hanning:
        # Set up 2D Hanning window to deal with edge effects
        window = np.hanning(N)
        window = np.outer(window, window)
        block = block * window
    
    # Compute polar averaged spectral values
    # f is the frequency radius
    # s is the average value for that frequency
    [f, s] = polar_average(np.abs(np.fft.fft2(block)))
    
    # Fit a line to the log-log transformed data
    line = P.fit(np.log(f), np.log(s), 1)
    res = (-line.coef[1], line.coef[0])
    return res


## Given output of FFT, compute polar averaged version
# Returns a tuple (f, a)
# f is a 1D array of frequency radii
# s is a 1D array of the same length with the polar averaged value for the corresponding radii
def polar_average(spect, num_angles=360):
    N = spect.shape[0]
    
    spect[0, 0] = np.mean([spect[0, 1], spect[1, 0]])
    
    # Generate grid coordinates in terms of polar coordinates, excluding the global average but including the Nyquist frequency at N//2.
    xs = []
    ys = []
    thetas = np.linspace(0, 2*np.pi, num_angles+1)[:-1]
    for r in range(1, N//2+1):
        xs.append(r * np.cos(thetas))
        ys.append(r * np.sin(thetas))
    grid_coords = np.array([np.concatenate(xs), np.concatenate(ys)])
    
    # Obtain values at those coordinates
    s_full = nd.map_coordinates(spect, grid_coords, mode='grid-wrap', order=1)
    s_full = s_full.reshape(-1, num_angles)
    
    # Average together
    s = s_full.mean(axis=1)
    
    # Generate frequency coordinates
    f = np.linspace(0, 0.5, s.shape[0] + 1)
    
    # Exclude 0th frequency, as we didn't compute an s value for that
    f = f[1:]

    return f, s