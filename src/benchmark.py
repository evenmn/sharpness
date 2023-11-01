import argparse
import matplotlib.pyplot as plt
import numpy as np

from sharpness.dataloader import generate_synthetic_data, load_data, synthetic_f
from sharpness.transforms import apply_transform, transform_d
from sharpness.metric_list import metric_f, single_metrics
from sharpness import compute_metric_globally, compute_metric_locally, compute_all_metrics_globally, compute_all_metrics_locally


parser = argparse.ArgumentParser(description='Sharpness Benchmarks')

# Data configuration
parser.add_argument('-s', '--synthetic', type=str, default=None,
                    help='generate synthetic data',
                    choices=list(synthetic_f.keys()))
parser.add_argument('-i', '--input', type=str, default='../data/kh_ABI_C13.nc',
                    help='name of input file to load data from')

# Computation configuration
parser.add_argument('-t', '--transformation', type=str, default='vflip',
                    help='transformation to perform on data',
                    choices=list(transform_d.keys()))
parser.add_argument('-m', '--metric', type=str, default='all',
                    help='evaluation metric to compute',
                    choices=['all'] + list(metric_f.keys()))
parser.add_argument('--heatmap', action='store_true', #default false
                    help='whether to compute sharpness heatmaps')

# Visualization configuration
parser.add_argument('--visualize', action='store_true',  # default false
                    help='visualize and save the operations')
parser.add_argument('-o', '--output', type=str, default='../media/output.png',
                    help='name of output file visualization')


def visualize(data, fname, args):
    cmap = 'gray'
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(data['X'], cmap=cmap)
    axs[1].imshow(data['T'], cmap=cmap)

    axs[0].set_title('X', weight='bold')
    axs[1].set_title('T', weight='bold')
    axs[1].set_title(f'({args.transformation})', loc='left')

    for ax in axs:
        ax.axis('off')

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    

def heatmap_visualize(data, fname, args):
    cmap = 'gray'
    if args.metric == 'all':
        if data['T'] is None:
            list_of_metrics = single_metrics
        else:
            list_of_metrics = metric_f.keys()
            
        fig, axs = plt.subplots(
            int(np.ceil(np.sqrt(len(list_of_metrics)))),
            int(np.ceil(np.sqrt(len(list_of_metrics)))),
            figsize=(12, 12)
        )
        raveled_axs = np.ravel(axs)
        for i, metric in enumerate(list_of_metrics):
            if metric not in single_metrics:
                raveled_axs[i].imshow(data['metrics'][metric], cmap=cmap)
            else:
                if data['T'] is not None:
                    raveled_axs[i].imshow(data['metrics'][metric][1], cmap=cmap)
                else:
                    raveled_axs[i].imshow(data['metrics'][metric][0], cmap=cmap)
            raveled_axs[i].set_title(metric)
            
        for ax in raveled_axs:
            ax.axis('off')
            
    else:
        if args.metric not in single_metrics:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(data['X'], cmap=cmap)
            axs[1].imshow(data['T'], cmap=cmap)
            axs[2].imshow(data['metrics'], cmap=cmap)
            
            axs[0].set_title('X', weight='bold')
            axs[1].set_title('T', weight='bold')
            axs[2].set_title(args.metric, weight='bold')
            
            for ax in axs:
                ax.axis('off')
        else:
            cmin = min(data['metrics'][0].min(), data['metrics'][1].min())
            cmax = max(data['metrics'][0].max(), data['metrics'][1].max())
            fig, axs = plt.subplots(2, 2, figsize=(10, 10), layout='constrained')
            axs[0, 0].imshow(data['X'], cmap=cmap)
            axs[1, 0].imshow(data['T'], cmap=cmap)
            axs[0, 1].imshow(data['metrics'][0], cmap=cmap, clim=(cmin, cmax))
            plt2 = axs[1, 1].imshow(data['metrics'][1], cmap=cmap, clim=(cmin, cmax))
            fig.colorbar(plt2, ax=axs[:, 1], shrink=0.6)

            axs[0, 0].set_title('X', weight='bold')
            axs[1, 0].set_title('T', weight='bold')
            axs[0, 1].set_title(args.metric + ' for X', weight='bold')
            axs[1, 1].set_title(args.metric + ' for T', weight='bold')

            for ax in np.ravel(axs):
                ax.axis('off')
    
    fig.savefig(fname, dpi=300, bbox_inches='tight')


def main(args):
    if args.synthetic:
        X = generate_synthetic_data(args.synthetic)
    else:
        X = load_data(args.input)

    T = apply_transform(X, args.transformation)
    # print(f'{X.shape=}, {X.min()=}, {X.max()=}')
    # print(f'{T.shape=}, {T.min()=}, {T.max()=}')

    metric_name = args.metric
    if metric_name == 'all':
        if not args.heatmap:
            metrics = compute_all_metrics_globally(X, T)
            for metric_name, result in metrics.items():
                print(f'=> {metric_name}: {result}')
        else:
            metrics = compute_all_metrics_locally(X, T)
            for metric_name, result in metrics.items():
                if metric_name in single_metrics:
                    print(f'=> {metric_name} averages: {(result[0].mean(), result[1].mean())}')
                else:
                    print(f'=> {metric_name} average: {result.mean()}')
                    
    else:
        if not args.heatmap:
            metrics = compute_metric_globally(X, T, metric_name)
            print(f'=> {metric_name}: {metrics}')
        else:
            metrics = compute_metric_locally(X, T, metric_name)
            if metric_name in single_metrics:
                print(f'=> {metric_name} averages: {(metrics[0].mean(), metrics[1].mean())}')
            else:
                print(f'=> {metric_name} average: {metrics.mean()}')

    data = dict()
    data['X'] = X
    data['T'] = T
    data['metrics'] = metrics

    if args.visualize:
        if args.heatmap:
            heatmap_visualize(data, args.output, args)
        else:
            visualize(data, args.output, args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)