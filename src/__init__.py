import numpy as np
from metric_list import metric_f, single_metrics
from .heatmap import Heatmap, heatmap_list

def compute_all_metrics_globally(X, T) -> dict:
    """Check that X and T are 2-dim arrays"""
    assert np.ndim(X)==2, f'Input must be 2-dimensional; got {np.ndim(X)} dimensions for X'
    if T is not None:
        assert np.ndim(T)==2, f'Input must be 2-dimensional; got {np.ndim(T)} dimensions for T'

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


def compute_metric_globally(X, T, metric: str):
    """Check that X and T are 2-dim arrays"""
    assert np.ndim(X)==2; f'Input must be 2-dimensional; got {np.ndim(X)} dimensions for X'
    if T is not None:
        assert np.ndim(T)==2; f'Input must be 2-dimensional; got {np.ndim(T)} dimensions for T'

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

def compute_all_metrics_locally(X, T) -> dict:
    """Check that X and T are 2-dim arrays"""
    assert np.ndim(X)==2, f'Input must be 2-dimensional; got {np.ndim(X)} dimensions for X'
    if T is not None:
        assert np.ndim(T)==2, f'Input must be 2-dimensional; got {np.ndim(T)} dimensions for T'

    """Compute all evaluation metrics."""
    metrics_to_compute = single_metrics if T is None else metric_f.keys()
    print(f'Computing the following metrics:')
    for metric in metrics_to_compute:
        print(metric)
    print(f'Heatmap will be computed with blocks of size {X.shape[0]//8}, and has image padding of length {X.shape[0]//16}')

    if T is not None:
        results = heatmap_list(X, T, metrics_to_compute, X.shape[0]//8, X.shape[0]//16)
    else:
        results = dict()
        for metric in metrics_to_compute:
            f = metric_f.get(metric)
            results[metric] = Heatmap(X, T, f, X.shape[0]//8, X.shape[0]//16)

    return results


def compute_metric_locally(X, T, metric: str):
    """Check that X and T are 2-dim arrays"""
    assert np.ndim(X)==2; f'Input must be 2-dimensional; got {np.ndim(X)} dimensions for X'
    if T is not None:
        assert np.ndim(T)==2; f'Input must be 2-dimensional; got {np.ndim(T)} dimensions for T'

    """Compute specified evaluation metric"""
    f = metric_f.get(metric)
    if f is None:
        raise ValueError(f'Unknown metric name: {metric}')
        
    print(f'Heatmap will be computed with blocks of size {X.shape[0]//8}, and has image padding of length {X.shape[0]//16}')
    
    return Heatmap(X, T, f, X.shape[0]//8, X.shape[0]//16)