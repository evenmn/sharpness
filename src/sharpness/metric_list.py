"""
Module aggregating metrics from other submodules in this package

The dictionary "metric_f" contains the metric functions as items, with their short names
being the keys. All these functions are callable in the format func(X, T), where X and T
are two images whose sharpness is to be compared.

The list "single_metrics" contains the keys for those functions in "metric_f" that can
be evaluated on single images -- i.e., they can be used in the format func(X), where X
is a single image.
"""

from sharpness.standard_metrics import rmse, total_variation, ssim
from sharpness.gradient import (
    gradient_rmse,
    laplacian_rmse,
    grad_total_variation,
    mean_gradient_magnitude,
)
from sharpness.fourier import fourier_rmse, fourier_total_variation
from sharpness.wavelet import wavelet_total_variation
from sharpness.spec_slope import s1, spec_slope

from functools import partial

metric_f = {
    "rmse": rmse,
    "ssim": partial(ssim, data_range=255),
    "tv": total_variation,
    "grad-mag": mean_gradient_magnitude,
    "grad-tv": grad_total_variation,
    "grad-rmse": gradient_rmse,
    "laplace-rmse": laplacian_rmse,
    "fourier-rmse": fourier_rmse,
    "fourier-tv": fourier_total_variation,
    "spec-slope": spec_slope,
    "s1": partial(
        s1, contrast_threshold=5, brightness_threshold=20, brightness_mult=False
    ),
    "wavelet-tv": wavelet_total_variation,
}
metric_f["ssim"].__name__ = "ssim"
metric_f["s1"].__name__ = "s1"

single_metrics = [
    "grad-mag",
    "spec-slope",
    "s1",
    "tv",
    "grad-tv",
    "fourier-tv",
    "wavelet-tv",
]