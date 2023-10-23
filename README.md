# ai2es-sharpness
This repository serves the sharpness group.  TBC

## Benchmark

Compute evaluations from different metrics and transformations on real or synthetic datasets.

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
  -m {all,mse,mae,rmse,grad}, --metric {all,mse,mae,rmse,grad,s1,psnr,ncc,gds,gmd,gpd,hist-int,hog-pearson} 
                        evaluation metric to compute
  --visualize           visualize and save the operations
  -o OUTPUT, --output OUTPUT
                        name of output file visualization
```

#### Examples

Generate synthetic data, apply a bluring transformation, compute all metrics, and visualize/save the output.

```bash
$ python benchmark.py -s xor -t blur -m all --visualize -o ../media/synthetic.png
=> mse: 151.15780639648438
=> mae: 7.190338134765625
=> rmse: 12.294625101908736
=> grad: (6.91624727961359e-19, 4.611330123778287e-19)
=> s1: (3.4584340177879653, 5.106637918264621)
=> psnr: 26.336497800628308
=> ncc: 0.9965684732341967
=> gds: 0.493354541101387
=> gmd: 2381696.5929986304
=> hist-int: 0.645227694933976
=> gpd: 2540086.6786078764
=> hog-pearson: 0.6305659890823435
```
![](media/synthetic.png)

Load the default data example, apply a vertical transformation, compute only the root-mean-square error, and visualize/save the output to the default name.

```bash
$ python benchmark.py -t vflip -m rmse --visualize
Loading data from ../data/kh_ABI_C13.nc (sample 0)
=> rmse: 10.005649078875036
```

![](media/output.png)