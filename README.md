# ai2es-sharpness
This is a repository containing implementations and utility functions for a variety of metrics that can be used to analyze the sharpness of meteorological image, as well as transform functions and a selection of synthetic and real data for use in examples. This is work conducted under the umbrella of [The NSF AI Institute for Research on Trustworthy AI in Weather, Climate, and Coastal Oceanography (AI2ES)](https://www.ai2es.org/). To find out more about the metrics included and how they can be used, see the [accompanying preprint](https://www.ai2es.org/sharpness/), and if you make use of this repository, please cite that paper. A peer-reviewed version is expected soon.

## Installation

To install this package with all required python packages, clone this repository and from the main directory run
```bash
conda env create --name ai2es-sharpness --file=environment.yml
```
To install with the specific versions used in the creation of this package, use `environmentv2.yml` instead.

## Examples of use

There are a number of example notebooks in the [notebooks](/notebooks) folder, which serve as a set of examples for utilizing the low-level interface of this package. For a simple introduction to how to run experiments using transforms, see [experiment_demo.ipynb](/notebooks/experiment_demo.ipynb), and for more in-depth experiments (including all those described in the accompanying paper,) see [paper_experiments.ipynb](/notebooks/paper_experiments.ipynb).

## Python interface

The principal utilities offered by this package can be found in [`src/sharpness/high_level_functions.py`](/src/sharpness/high_level_functions.py) and made available as a base-level import with the sharpness package. These are the functions `compute_all_metrics_globally`, `compute_metric_globally`, `compute_all_metrics_locally`, `compute_metric_locally`, all of which take in a pair of images `X` and `T` and either compute all metrics, or just a specified single metric. In either case, computing "globally" means that the metric will be applied to the whole image (or pair of images) and a single number will be returned, while computing "locally" means that a heatmap of local metric values will be computed.

#### Input requirements

All functions take as their input a single `n x m` grayscale image. If using these methods on multi-channel imagery, either convert the image to grayscale (if it is an RGB image) or loop over the channels and compute sharpness statistics individually.

Input data should be of float type; if the data are not already floats, certain metrics which require float type inputs will internally convert the data to floats.