from sharpness.metrics import (
    mse,
    mae,
    rmse,
    total_variation
)
from sharpness.gradient import (
    psnr,
    normalized_cross_correlation,
    gradient_difference_similarity,
    gradient_rmse,
    laplacian_rmse,
    histogram_intersection,
    hog_pearson,
    grad_total_variation,
    mean_gradient_magnitude
)
from sharpness.fourier import (
    fourier_rmse,
    fourier_total_variation
)
from sharpness.wavelet import (
    wavelet_image_similarity,
    wavelet_total_variation
)
from sharpness.spec_slope import (
    s1,
    spec_slope
)

metric_f = {
    'mse': mse,
    'mae': mae,
    'rmse': rmse,
    's1': s1,
    'spec_slope': spec_slope,
    "psnr": psnr,
    "ncc": normalized_cross_correlation,
    "mgm": mean_gradient_magnitude,
    "grad-ds": gradient_difference_similarity,
    "grad-rmse": gradient_rmse,
    "laplace-rmse": laplacian_rmse,
    "hist-int": histogram_intersection,
    # "gpd": gradient_profile_difference,
    "hog-pearson": hog_pearson,
    "fourier-rmse": fourier_rmse,
    "wavelet-similarity": wavelet_image_similarity,
    "tv": total_variation,
    "grad-tv": grad_total_variation,
    "fourier-tv": fourier_total_variation,
    "wavelet-tv": wavelet_total_variation
}

single_metrics = ["mgm", "s1", "spec_slope", "tv", "grad-tv", "fourier-tv", "wavelet-tv"]