import argparse
import matplotlib.pyplot as plt
import numpy as np

from sharpness.dataloader import generate_synthetic_data, load_data, synthetic_f
from sharpness.transforms import apply_transform, transform_d
from sharpness.metrics import metric_f, single_metrics
from sharpness import (
    compute_metric_globally,
    compute_metric_locally,
    compute_all_metrics_globally,
    compute_all_metrics_locally,
)


parser = argparse.ArgumentParser(description="Sharpness Benchmarks")

# Data configuration
parser.add_argument(
    "-s",
    "--synthetic",
    type=str,
    default=None,
    help="generate synthetic data",
    choices=list(synthetic_f.keys()),
)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="../data/kh_ABI_C13.nc",
    help="name of input file to load data from",
)

# Computation configuration
parser.add_argument(
    "-t",
    "--transformation",
    type=str,
    default="vflip",
    help="transformation to perform on data",
    choices=list(transform_d.keys()),
)
parser.add_argument(
    "-m",
    "--metric",
    type=str,
    default="all",
    help="evaluation metric to compute",
    choices=["all"] + list(metric_f.keys()),
)
parser.add_argument(
    "--heatmap",
    action="store_true",  # default false
    help="whether to compute sharpness heatmaps",
)

# Visualization configuration
parser.add_argument(
    "--visualize",
    action="store_true",  # default false
    help="visualize and save the operations",
)
parser.add_argument(
    "--overlay",
    action="store_true",  # default false
    help="if visualizing heatmaps, plot output as overlay on top of original image",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="../media/output.png",
    help="name of output file visualization",
)


def visualize(data, fname, args):
    cmap = "gray"
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(data["X"], cmap=cmap)
    axs[1].imshow(data["T"], cmap=cmap)

    axs[0].set_title("X", weight="bold")
    axs[1].set_title("T", weight="bold")
    axs[1].set_title(f"({args.transformation})", loc="left")

    for ax in axs:
        ax.axis("off")

    fig.savefig(fname, dpi=300, bbox_inches="tight")


def heatmap_visualize(data, fname, args):
    bg_cmap = "gray"
    hm_cmap = "Blues"
    hm_alpha = 0.7 if args.overlay else 1.0
    if args.metric == "all":
        cb_shrink = 0.6

        if data["T"] is None:
            list_of_metrics = single_metrics
            tot_num_to_plot = 2 + len(single_metrics)
        else:
            list_of_metrics = metric_f.keys()
            tot_num_to_plot = 2 + len(list_of_metrics) + len(single_metrics)

        possible_dims = [
            (i, int(np.ceil(tot_num_to_plot / i)))
            for i in range(4, int(np.ceil(np.sqrt(tot_num_to_plot) + 1)))
        ]
        dims = possible_dims[
            np.argmin(
                [
                    (tuple[0] * tuple[1] - tot_num_to_plot)
                    + 0.1 * (np.abs(tuple[0] - tuple[1]))
                    for tuple in possible_dims
                ]
            )
        ]

        fig, axs = plt.subplots(
            dims[0], dims[1], figsize=(3 * dims[1], 3 * dims[0]), layout="constrained"
        )
        raveled_axs = np.ravel(axs)
        raveled_axs[0].imshow(data["X"], cmap=bg_cmap)
        raveled_axs[0].set_title("X")
        raveled_axs[1].imshow(data["T"], cmap=bg_cmap)
        raveled_axs[1].set_title("T")
        i = 2
        for metric in list_of_metrics:
            if metric not in single_metrics:
                if args.overlay:
                    raveled_axs[i].imshow(data["T"], cmap=bg_cmap)
                metric_plot = raveled_axs[i].imshow(
                    data["metrics"][metric], alpha=hm_alpha, cmap=hm_cmap
                )
                raveled_axs[i].set_title(metric)
                fig.colorbar(metric_plot, ax=raveled_axs[i], shrink=cb_shrink)
                i += 1
            else:
                if data["T"] is not None:
                    if args.overlay:
                        raveled_axs[i].imshow(data["X"], cmap=bg_cmap)
                    metric_plot = raveled_axs[i].imshow(
                        data["metrics"][metric][0], alpha=hm_alpha, cmap=hm_cmap
                    )
                    raveled_axs[i].set_title(metric + " on X")
                    fig.colorbar(metric_plot, ax=raveled_axs[i], shrink=cb_shrink)
                    i += 1
                    if args.overlay:
                        raveled_axs[i].imshow(data["T"], cmap=bg_cmap)
                    metric_plot = raveled_axs[i].imshow(
                        data["metrics"][metric][1], alpha=hm_alpha, cmap=hm_cmap
                    )
                    raveled_axs[i].set_title(metric + " on T")
                    fig.colorbar(metric_plot, ax=raveled_axs[i], shrink=cb_shrink)
                    i += 1
                else:
                    if args.overlay:
                        raveled_axs[i].imshow(data["T"], cmap=bg_cmap)
                    metric_plot = raveled_axs[i].imshow(
                        data["metrics"][metric][0], alpha=hm_alpha, cmap=hm_cmap
                    )
                    raveled_axs[i].set_title(metric)
                    fig.colorbar(metric_plot, ax=raveled_axs[i], shrink=cb_shrink)
                    i += 1

        for ax in raveled_axs:
            ax.axis("off")

    else:
        if args.metric not in single_metrics:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(data["X"], cmap=bg_cmap)
            axs[1].imshow(data["T"], cmap=bg_cmap)
            if args.overlay:
                axs[2].imshow(data["T"], cmap=bg_cmap)
            axs[2].imshow(data["metrics"], alpha=hm_alpha, cmap=hm_cmap)

            axs[0].set_title("X", weight="bold")
            axs[1].set_title("T", weight="bold")
            axs[2].set_title(args.metric, weight="bold")

            for ax in axs:
                ax.axis("off")
        else:
            cmin = min(data["metrics"][0].min(), data["metrics"][1].min())
            cmax = max(data["metrics"][0].max(), data["metrics"][1].max())
            fig, axs = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
            axs[0, 0].imshow(data["X"], cmap=bg_cmap)
            axs[1, 0].imshow(data["T"], cmap=bg_cmap)
            if args.overlay:
                axs[0, 1].imshow(data["X"], cmap=bg_cmap)
            axs[0, 1].imshow(
                data["metrics"][0], alpha=hm_alpha, cmap=hm_cmap, clim=(cmin, cmax)
            )
            if args.overlay:
                axs[1, 1].imshow(data["T"], cmap=bg_cmap)
            plt2 = axs[1, 1].imshow(
                data["metrics"][1], alpha=hm_alpha, cmap=hm_cmap, clim=(cmin, cmax)
            )
            fig.colorbar(plt2, ax=axs[:, 1], shrink=0.6)

            axs[0, 0].set_title("X", weight="bold")
            axs[1, 0].set_title("T", weight="bold")
            axs[0, 1].set_title(args.metric + " on X", weight="bold")
            axs[1, 1].set_title(args.metric + " on T", weight="bold")

            for ax in np.ravel(axs):
                ax.axis("off")

    fig.savefig(fname, dpi=300, bbox_inches="tight")


def main(args):
    if args.synthetic:
        X = generate_synthetic_data(args.synthetic)
    else:
        X = load_data(args.input)

    T = apply_transform(X, args.transformation)
    # print(f'{X.shape=}, {X.min()=}, {X.max()=}')
    # print(f'{T.shape=}, {T.min()=}, {T.max()=}')

    metric_name = args.metric
    if metric_name == "all":
        if not args.heatmap:
            metrics = compute_all_metrics_globally(X, T)
            for metric_name, result in metrics.items():
                print(f"=> {metric_name}: {result}")
        else:
            metrics = compute_all_metrics_locally(X, T)
            for metric_name, result in metrics.items():
                if metric_name in single_metrics:
                    print(
                        f"=> {metric_name} averages: "
                        f"{(result[0].mean(), result[1].mean())}"
                    )
                else:
                    print(f"=> {metric_name} average: {result.mean()}")

    else:
        if not args.heatmap:
            metrics = compute_metric_globally(X, T, metric_name)
            print(f"=> {metric_name}: {metrics}")
        else:
            metrics = compute_metric_locally(X, T, metric_name)
            if metric_name in single_metrics:
                print(
                    f"=> {metric_name} averages: "
                    f"{(metrics[0].mean(), metrics[1].mean())}"
                )
            else:
                print(f"=> {metric_name} average: {metrics.mean()}")

    data = dict()
    data["X"] = X
    data["T"] = T
    data["metrics"] = metrics

    if args.visualize:
        if args.heatmap:
            heatmap_visualize(data, args.output, args)
        else:
            visualize(data, args.output, args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
