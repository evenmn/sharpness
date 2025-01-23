"""
A package to evaluate image sharpness, particular in machine learning output.

The sharpness package provides a number of utilities for use in evaluating the sharpness
of image data. The particular focus of this package is on imagery produced as the output
of a machine learning algorithm. More details can be found in the accompanying
publication at https://www.ai2es.org/sharpness/. 

Some high level code is provided as a top-level import, with more functions available in
lower-level modules, particularly the "exp_utilities", "transforms", and "dataloader"
modules.

A limited CLI interface is provided in the "benchmark.py" file.
"""
from .metric_list import metric_f, single_metrics

from .high_level_functions import (
    compute_all_metrics_globally,
    compute_all_metrics_locally,
    compute_metric_globally,
    compute_metric_locally,
)
