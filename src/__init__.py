from .metrics import mse, mae, rmse, grad, s1
from .gradient import (
    psnr,
    normalized_cross_correlation,
    gradient_difference_similarity,
    gradient_magnitude_difference,
    histogram_intersection,
    gradient_profile_difference,
    hog_pearson
)
from .fourier import fourier_image_similarity
from .wavelet import wavelet_image_similarity

metric_f = {
    'mse': mse,
    'mae': mae,
    'rmse': rmse,
    'grad': grad,
    's1': s1,
    "psnr": psnr,
    "ncc": normalized_cross_correlation,
    "gds": gradient_difference_similarity,
    "gmd": gradient_magnitude_difference,
    "hist-int": histogram_intersection,
    "gpd": gradient_profile_difference,
    "hog-pearson": hog_pearson,
    "fourier-similarity": fourier_image_similarity,
    "wavelet-similarity": wavelet_image_similarity
}

single_metrics = ["grad", "s1"]


def compute_all_metrics(X, T) -> dict:
    """Compute all evaluation metrics."""
    results = dict()
    metrics_to_compute = single_metrics if T is None else metric_f.keys()

    for metric in metrics_to_compute:
        f = metric_f.get(metric)
        if f is not None:
            try:
                if T is not None:
                    if metric in single_metrics:
                        results[metric] = f(X), f(T)
                    else:
                        results[metric] = f(X, T)
                else:
                    results[metric] = f(X)
            except Exception as e:
                print(f'Failed to compute {metric}: {e}')
        else:
            print(f'Unknown metric name: {metric}')

    return results


def compute_metric(X, T, metric: str):
    """Compute specified evaluation metric"""
    f = metric_f.get(metric)
    if f is None:
        raise ValueError(f'Unknown metric name: {metric}')

    if T is not None:
        if metric in single_metrics:
            return f(X), f(T)
        else:
            return f(X, T)
    else:
        return f(X)
