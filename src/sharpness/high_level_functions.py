"""Module containing high level routines"""
import numpy as np
from .metrics import metric_f, single_metrics
from .heatmap import Heatmap, heatmap_list


def compute_all_metrics_globally(X: np.ndarray, T: np.ndarray) -> dict:
    """
    High-level function to compute all metrics between X and T

    Note that X and T must be 2D arrays
    """
    assert (
        np.ndim(X) == 2
    ), f"Input must be 2-dimensional; got {np.ndim(X)} dimensions for X"
    if T is not None:
        assert (
            np.ndim(T) == 2
        ), f"Input must be 2-dimensional; got {np.ndim(T)} dimensions for T"

    # Compute all evaluation metrics.
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
                print(f"Failed to compute {metric}: {e}")
        else:
            print(f"Unknown metric name: {metric}")

    return results


def compute_metric_globally(X, T, metric: str):
    """
    High-level function to compute a single metric between X and T

    Note that X and T must be 2D arrays
    """
    assert (
        np.ndim(X) == 2
    ), f"Input must be 2-dimensional; got {np.ndim(X)} dimensions for X"
    if T is not None:
        assert (
            np.ndim(T) == 2
        ), f"Input must be 2-dimensional; got {np.ndim(T)} dimensions for T"

    # Compute specified evaluation metric
    f = metric_f.get(metric)
    if f is None:
        raise ValueError(f"Unknown metric name: {metric}")

    if T is not None:
        if metric in single_metrics:
            return f(X), f(T)
        else:
            return f(X, T)
    else:
        return f(X)


def compute_all_metrics_locally(
    X, T, block_size=None, pad_len=None, verbose=True
) -> dict:
    """
    High-level function to compute all metric heatmaps between X and T

    Note that X and T must be 2D arrays
    """
    assert (
        np.ndim(X) == 2
    ), f"Input must be 2-dimensional; got {np.ndim(X)} dimensions for X"
    if T is not None:
        assert (
            np.ndim(T) == 2
        ), f"Input must be 2-dimensional; got {np.ndim(T)} dimensions for T"

    # Compute all evaluation metrics."""
    metrics_to_compute = single_metrics if T is None else metric_f.keys()
    if block_size is None:
        block_size = X.shape[0] // 8
    if pad_len is None:
        pad_len = X.shape[0] // 16
    if verbose:
        print(
            f"Heatmap will be computed with blocks of size {block_size}, and has image "
            f"padding of length {pad_len}"
        )

    if T is not None:
        results = heatmap_list(X, T, metrics_to_compute, block_size, pad_len)
    else:
        results = dict()
        for metric in metrics_to_compute:
            f = metric_f.get(metric)
            results[metric] = Heatmap(X, T, f, X.shape[0] // 8, X.shape[0] // 16)

    return results


def compute_metric_locally(
    X, T, metric: str, block_size=None, pad_len=None, verbose=True
):
    """
    High-level function to compute a single metric heatmap between X and T

    Note that X and T must be 2D arrays
    """
    assert (
        np.ndim(X) == 2
    ), f"Input must be 2-dimensional; got {np.ndim(X)} dimensions for X"
    if T is not None:
        assert (
            np.ndim(T) == 2
        ), f"Input must be 2-dimensional; got {np.ndim(T)} dimensions for T"

    # Compute specified evaluation metric"""
    f = metric_f.get(metric)
    if f is None:
        raise ValueError(f"Unknown metric name: {metric}")

    if block_size is None:
        block_size = X.shape[0] // 8
    if pad_len is None:
        pad_len = X.shape[0] // 16
    if verbose:
        print(
            f"Heatmap will be computed with blocks of size {block_size}, and has image "
            f"padding of length {pad_len}"
        )

    return Heatmap(X, T, f, block_size, pad_len)
