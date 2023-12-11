# ai2es-sharpness
This repository serves the sharpness group.  TBC

## Installation

To install this package with all required python packages, clone this repository and from the main directory run
```bash
conda env create --name ai2es-sharpness --file=environment.yml
```
After that completes, activate your new environment and run
```bash
pip install .
```

## Benchmark

Compute evaluations from different metrics and transformations on real or synthetic datasets.


#### Input requirements

At this point, all functions (including the main `benchmark.py` script) take as their input a single `n x m` grayscale image. If using these methods on multi-channel imagery, either convert the image to grayscale (if it is an RGB image) or loop over the channels and compute sharpness statistics individually.

Input data should be of float type; if the data are not already floats, certain metrics which require float type inputs will internally convert the data to floats.

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
  --heatmap             compute sharpness heatmap(s) rather than global metric
  --visualize           visualize and save the operations
  --overlay             only relevant if both heatmap and visualize are true; plots heatmaps on top of input data
  -o OUTPUT, --output OUTPUT
                        name of output file visualization
```

Note that with the `--heatmap` option, each metric will be computed on small, overlapping tiles across the image; by default, these tiles will be square with side length equal to ~1/8th the width of the input image, and the stride for these tiles will be 1/4 the side length of the tile. The image will also be padded using the "reflect" method by a number of pixels equal to ~1/16th the width of the input image. These parameters are adjustable by editing the appropriate function in `__init__.py`.

#### Examples

Generate synthetic data, apply a blurring transformation, compute all metrics, and visualize/save the output.

```bash
$ python benchmark.py -s xor -t blur -m all --visualize -o ../media/synthetic.png
=> mse: 150.1562378666429
=> mae: 7.141086141494917
=> rmse: 12.25382543806802
=> s1: (0.00339528769955455, 1.7763568394002505e-15)
=> psnr: 26.36536982276365
=> ncc: 0.9965476066368607
=> mgm: (44.19561294970306, 29.452084668360147)
=> grad-ds: 0.49620562145179087
=> grad-rmse: 71.91002205102615
=> laplace-rmse: 26.700794614133223
=> hist-int: 0.6448973445108177
=> hog-pearson: 0.5533226275817607
=> fourier-rmse: 278977716.3719768
=> wavelet-similarity: 0.485842832185395
=> tv: (524288.0, 272784.50942777185)
=> grad-tv: (3153888.0, 2149503.1447823923)
=> fourier-tv: (120947473.46739776, 120612974.58877043)
=> wavelet-tv: (4194304.0, 4248385.269167363)
```
![](media/synthetic.png)

We can re-run the above example, but with local computations of heatmaps overlaid on top of input data instead of global metrics.

```bash
$ python benchmark.py -s xor -t blur -m all --heatmap --visualize --overlay -o ../media/synthetic_heatmaps.png
Heatmap will be computed with blocks of size 32, and has image padding of length 16
=> mse average: 39.881929874420166
=> mae average: 111.61909246444702
=> rmse average: 6.259764454401335
=> s1 averages: (2.7966713163144616, 4.416437443298626)
=> psnr average: 27.90985194547143
=> ncc average: 1.0957310987654965
=> grad averages: (0.025264954381721993, 0.019396214612508057)
=> grad-ds average: 0.4854998901219907
=> grad-rmse average: 54.209045009739974
=> laplace-rmse average: 20.67599301023201
=> hist-int average: 0.5660052760923319
=> hog-pearson average: 0.6746634025245817
=> fourier-similarity average: 1.0
=> wavelet-similarity average: 0.3293994978185485
=> tv averages: (253952.0, 176779.0)
=> grad-tv averages: (40465.7734375, 30567.400390625)
=> fourier-tv averages: (358239.7522150265, 318628.08900317416)
=> wavelet-tv averages: (65607.43359375, 65929.7275390625)
```
![](media/synthetic_heatmaps.png)

Load the default data example, apply a vertical transformation, compute only the root-mean-square error, and visualize/save the output to the default name.

```bash
$ python benchmark.py -t vflip -m rmse --visualize
Loading data from ../data/kh_ABI_C13.nc (sample 0)
=> rmse: 10.005649078875036
```
![](media/output.png)

Generate synthetic data again, but only compute total variation as a heatmap.
```bash
$ python benchmark.py -s='xor' -t='blur' -m='tv' -o='../media/synth_tv.png' --heatmap --visualize
Heatmap will be computed with blocks of size 32, and has image padding of length 16
=> tv averages: (253952.0, 176779.0)
```

![](media/synth_tv.png)
