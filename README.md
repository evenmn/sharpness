# ai2es-sharpness
This repository serves the sharpness group.  TBC

## Benchmark

Compute evaluations from different metrics and transformations on real or synthetic datasets.


#### Input requirements

At this point, all functions (including the main `benchmark.py` script) take as their input a single `n x m` grayscale image. If using these methods on multi-channel imagery, either convert the image to grayscale (if it is an RGB image) or loop over the channels and compute sharpness statistics individually.

Note that due to the way certain packages we utilize are implemented, some metrics may change depending on the data type used as input; to ensure that images are comparable, make sure that they all have the same underlying datatype (e.g., `np.float32, np.float64, np.uint8,` etc.). For maximum compatibility, image data that is integers in the range (0, 255) should have the type np.uint8, and any data that has a float type should take values in the range (0, 1).

#### Usage

From within the `src` directory:

```bash
$ python benchmark.py -h
usage: benchmark.py [-h] [-s {sinusoidal,gaussian,bw,xor}] [-i INPUT] [-t {vflip,hflip,blur,noise,brightness,crop}] [-m {all,mse,mae,rmse,grad}] [--visualize] [-o OUTPUT]

Sharpness Benchmarks

optional arguments:
  -h, --help            show this help message and exit
  -s {sinusoidal,gaussian,bw,xor}, --synthetic {sinusoidal,gaussian,bw,xor}
                        generate synthetic data
  -i INPUT, --input INPUT
                        name of input file to load data from
  -t {vflip,hflip,blur,noise,brightness,crop}, --transformation {vflip,hflip,blur,noise,brightness,crop}
                        transformation to perform on data
  -m {all,mse,mae,rmse,grad}, --metric {all,mse,mae,rmse,grad,s1,psnr,ncc,grad-ds,grad-rmse,laplace-rmse,hist-int,hog-pearson,fourier-similarity,wavelet-similarity,tv,grad-tv,fourier-tv,wavelet-tv} 
                        evaluation metric to compute
  --visualize           visualize and save the operations
  -o OUTPUT, --output OUTPUT
                        name of output file visualization
```

#### Examples

Generate synthetic data, apply a blurring transformation, compute all metrics, and visualize/save the output.

```bash
$ python benchmark.py -s xor -t blur -m all --visualize -o ../media/synthetic.png
=> mse: 151.15780639648438
=> mae: 7.190338134765625
=> rmse: 12.294625101908736
=> s1: (3.4584340177879653, 5.106637918264621)
=> psnr: 26.336497800628308
=> ncc: 0.9965684732341967
=> grad: (6.91624727961359e-19, 4.611330123778287e-19)
=> grad-ds: 0.493354541101387
=> grad-rmse: 71.90568391934428
=> laplace-rmse: 26.71850557271238
=> hist-int: 0.645227694933976
=> hog-pearson: 0.6305659890823435
=> fourier-similarity: 1.0
=> wavelet-similarity: 0.13379014374033754
=> tv: (524288, 273158)
=> grad-tv: (3153888.0, 2151308.0)
=> fourier-tv: (38639962.24096394, 31919021.24357993)
=> wavelet-tv: (8388608.0, 8360728.0)
```
![](media/synthetic.png)

Load the default data example, apply a vertical transformation, compute only the root-mean-square error, and visualize/save the output to the default name.

```bash
$ python benchmark.py -t vflip -m rmse --visualize
Loading data from ../data/kh_ABI_C13.nc (sample 0)
=> rmse: 10.005649078875036
```

![](media/output.png)
