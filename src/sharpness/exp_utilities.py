from sharpness.metrics import metric_f, single_metrics
from sharpness import compute_metric_locally
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional

from time import perf_counter


def apply_transforms(input, transforms, parameters, labels, base_label):
    """
    Computes a number of transforms of a single input.

    Parameters
    ----------
    input : np.ndarray
        Input 2-dimensional array that is to be transformed
    transforms : list
        List of transforms to apply. Each should be in the same format as those in
        sharpness.transforms, e.g., the transform is a class with parameters set on
        creation with a __call__ method that takes in and returns 2-dimensional
        np.ndarrays.
    parameters : list
        List of parameter dictionaries for transforms. Must be the same length as
        transforms. Each element of the list must be a dictionary containing all the
        parameters needed for the corresponding transform.
    labels : list
        List of strings used to describe the transform/parameter combinations. Must
        be the same length as transforms and parameters.
    base_label : string
        Descriptive text for input.

    Returns
    -------
    result : dictionary
        Dictionary containing the following key:value pairs:
            base_label: input
            labels[i]: transforms[i](parameters[i])(input) for each i
    """
    assert (
        len(transforms) == len(parameters) == len(labels)
    ), "Transforms, parameters, and labels must all be the same length"

    print("Computing transforms...")
    # Initialize dictionary with input
    result = {base_label: input}

    for i in tqdm(range(len(transforms))):
        result[labels[i]] = transforms[i](**parameters[i])(input)

    return result


def compute_metrics(
    inputs,
    metrics,
    metric_cmap_codes: dict,
    plot_title=None,
    outdir="../media/",
    filename=None,
    plot=True,
    invert_yaxis=True,
    plot_intensity=True,
    min_max_caption: bool = False,
    return_vals=True,
    uni_ratios=True,
):
    """
    Computes and optionally plots metrics across a set of inputs.

    Parameters
    ----------
    inputs : dictionary
        Dictionary containing key:value pairs of the form
            descriptive text: input
        where each input should be an np.ndarry. The first input in the dictionary
        will be assumed to be the "base input" for comparison purposes, and its
        descriptive text will be used in generating the filename to save the plot to.
    metrics : list
        List of metrics to compute on all inputs. Elements of the list should be
        strings that are keys in sharpness.metric_list.metric_f.
    metric_cmap_codes : dict
        Dictionary mapping metrics to matplotlib colormap names.
    plot_title : string
        Optional title to display on plot.
    outdir : string
        Prefix for which directory to save the output plot to. Can be an absolute
        or relative path.
    filename : string
        Optional filename that will override the default naming scheme.
    plot : boolean
        If true, a plot will be generated, displayed, and saved.
    invert_yaxis : boolean
        If true, y axes for heatmaps will be inverted, as is commonly needed for image
        data.
    plot_intensity : boolean
        If false, don't plot the raw intensity images.
    min_max_caption : boolean
        If true, add minimum and maximum values as captions along the left side of each
        heatmap. Makes plots quite busy.
    return_vals : boolean
        If true, the function will return summary_stats.
    uni_ratios : boolean
        If true, univariate metrics will have an additional set of statistics computed
        and displayed on the created plot. These relative min, mean, and max are the
        min, mean, and max for that particular input divided by the corresponding min,
        mean, and max values for the same metric on the "base input".

    Returns
    -------
    summary_stats : dictionary
        Dictionary with a key for each row in the plot: "intensity" corresponds to the
        first row, while each metric has its own key. The value for each key is a
        np.ndarray with columns corresponding to the columns in the plot, and rows
        corresponding to minimum, mean, and maximum values (as displayed below each
        image in the plot) respectively.
    """

    assert (
        set(metrics) - set(metric_f) == set()
    ), f"Unknown metric detected in metrics: {set(metrics) - set(metric_f)}"

    print("Computing heatmaps for " + ", ".join(metrics))
    print("on " + ", ".join(inputs.keys()))

    base_input = list(inputs.values())[0]
    summary_stats = {
        "intensity": np.zeros((3, len(inputs))),
        **{metric: np.zeros((3, len(inputs))) for metric in metrics},
    }

    # Compute statistics for the first row of inputs
    for j, img in enumerate(inputs.values()):
        summary_stats["intensity"][0, j] = np.nanmin(img)
        summary_stats["intensity"][1, j] = np.nanmean(img)
        summary_stats["intensity"][2, j] = np.nanmax(img)

    if plot:  # Set up plot and plot first row of inputs
        ncols = len(inputs.keys())
        if plot_intensity:
            nrows = len(metrics) + 1
        else:
            nrows = len(metrics)

        F, ax = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            figsize=(ncols * 4, nrows * 4),
            sharex="col",
            sharey="row",
        )

        # Set up common scaling for first row
        if plot_intensity:
            imgs_cmin, imgs_cmax = np.nanmin(list(inputs.values())), np.nanmax(
                list(inputs.values())
            )

            for j, key in enumerate(inputs.keys()):
                my_plot = ax[0, j].imshow(
                    inputs[key],
                    clim=(imgs_cmin, imgs_cmax),
                    cmap="gray",
                    origin="lower",
                )
                ax[0, j].set_title(key, fontsize=28)
                if min_max_caption:
                    ax[0, j].set_xlabel(
                        f'Min: {summary_stats["intensity"][0, j]:.2f}  '
                        f'Mean: {summary_stats["intensity"][1, j]:.2f}  '
                        f'Max:{summary_stats["intensity"][2, j]:.2f}'
                    )
                ax[0, j].set_xticks([])
                ax[0, j].set_yticks([])
                if invert_yaxis:
                    ax[0, j].invert_yaxis()
                cbar = plt.colorbar(my_plot, shrink=0.6, ax=ax[0, j])
                cbar.formatter.set_powerlimits((-10, 10))
                cbar.ax.tick_params(labelsize=16)

    print("\n##############################")
    if plot_intensity:
        i_start = 1
    else:
        i_start = 0

    for i, metric in enumerate(metrics, start=i_start):

        print(f"\nComputing {metric}")

        heatmaps = []
        verbosity = (
            True if i == 1 else False
        )  # Only print heatmap size for first metric

        # Metric computation occurs here
        start_time = perf_counter()
        for img in inputs.values():
            heatmaps.append(
                compute_metric_locally(base_input, img, metric, verbose=verbosity)
            )
        comp_time = perf_counter() - start_time
        print(f"Done! {len(inputs.keys())} inputs took {comp_time} seconds")

        row_cmin = np.nanmin(heatmaps)
        row_cmax = np.nanmax(heatmaps)

        # Univariate metrics output a tuple, so just take the second element of the list
        # This works even for the first column, because both X and T are base_input
        if metric in single_metrics:
            heatmaps = [heatmap[1] for heatmap in heatmaps]

        summary_stats[metric][0, :] = [np.nanmin(heatmap) for heatmap in heatmaps]
        summary_stats[metric][1, :] = [np.nanmean(heatmap) for heatmap in heatmaps]
        summary_stats[metric][2, :] = [np.nanmax(heatmap) for heatmap in heatmaps]

        if (metric in single_metrics) and uni_ratios:
            relative_stats = np.ndarray((3, len(inputs) - 1))
            relative_stats[0, :] = [
                min / summary_stats[metric][0, 0]
                for min in summary_stats[metric][0, 1:]
            ]
            relative_stats[1, :] = [
                mean / summary_stats[metric][1, 0]
                for mean in summary_stats[metric][1, 1:]
            ]
            relative_stats[2, :] = [
                max / summary_stats[metric][2, 0]
                for max in summary_stats[metric][2, 1:]
            ]
            append_strings = [""]
            for k in range(relative_stats.shape[1]):
                append_strings.append(
                    f"\nRelative Min: {relative_stats[0, k]:.2f}  "
                    f"Mean: {relative_stats[1, k]:.2f}  "
                    f"Max: {relative_stats[2, k]:.2f}"
                )
        else:
            append_strings = [""] * len(inputs)

        if plot:
            my_cmap = mpl.colormaps.get_cmap(metric_cmap_codes[metric])
            my_cmap.set_bad(color="yellow")
            for j, key in enumerate(inputs.keys()):
                my_plot = ax[i, j].imshow(
                    heatmaps[j],
                    clim=(row_cmin, row_cmax),
                    cmap=my_cmap,
                    origin="lower",
                )
                ax[i, j].set_title(f"{metric}", fontsize=28)
                if min_max_caption:
                    ax[i, j].set_xlabel(
                        f"Min: {summary_stats[metric][0, j]:.2f}  "
                        f"Mean: {summary_stats[metric][1, j]:.2f}  "
                        f"Max: {summary_stats[metric][2, j]:.2f}" + append_strings[j]
                    )
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                if invert_yaxis:
                    ax[i, j].invert_yaxis()
                cbar = plt.colorbar(my_plot, shrink=0.6, ax=ax[i, j])
                cbar.formatter.set_powerlimits((-10, 10))
                cbar.ax.tick_params(labelsize=16)

    print("\n##############################\n")

    if plot:
        print("Plotting figure...\n")
        F.suptitle(plot_title)
        plt.tight_layout()
        plt.show()
        if filename is None:
            base_name = list(inputs.keys())[0].replace(" ", "_")
            output_filename = (
                outdir + base_name + "_" + str(len(metrics)) + "_metrics_plots.pdf"
            )
        else:
            output_filename = filename
        print(f"\nSaving results to {output_filename}")
        F.savefig(output_filename)

    if return_vals:
        return summary_stats


###### Transferred over from experiments notebook ######


def write_metric_summary_output(
    results: dict,
    img_dict: dict,
    format: str = "IMAGES_NO_RELATION_SPECIFIED",
    outdir: str = "../media/",
    filename: str = None,
) -> None:
    """
    Utility function to format metric statistic output

    If filename is None, prints to stdout; otherwise, writes results to a file, then
        prints that file to stdout.
    """

    image_keys = list(img_dict.keys())  # Get image names

    # Set up printing to screen versus writing to file
    if filename == None:
        file = None  # write to std output
    else:
        print(f"Writing summary statistics to file {filename}")
        file = open(outdir + filename, "w")
        print(
            f"This file is:  {filename}\n", file=file
        )  # Write filename at beginning of file for convenience of seeing parameters

    for metric_name in results.keys():  # Go through all metrics

        print(f"\n##### {metric_name} #####", file=file)
        this_metric_result = results[metric_name]
        my_stat_types = ["min", "mean", "max"]

        for stat_type_idx, this_stat_type in enumerate(my_stat_types):
            this_stat_this_metric_result_vec = this_metric_result[
                stat_type_idx
            ]  # retrieve values for this statistic across all images

            if format == "IMAGES_NO_RELATION_SPECIFIED":

                # Interpret results as a list of images without any specific relation to
                # each other
                print(f"\n{metric_name} - {this_stat_type}:", file=file)

                for image_idx, value in enumerate(this_stat_this_metric_result_vec):
                    # Left justify to align values in a second column
                    image_name = image_keys[image_idx].ljust(12)
                    print(f"\t{image_name}: {value:.2f}", file=file)

            elif format == "TRUTH_AND_ENSEMBLE_PREDICTION":

                # Interpret results as 1) Observation, followed by 2) one or more
                # ensemble members
                truth_value = this_stat_this_metric_result_vec[
                    0
                ]  # first value of vector corresponds to observed radar image
                ensemble_values = this_stat_this_metric_result_vec[
                    1:
                ]  # remaining values correspond to ensembles members of prediction

                ensemble_mean = np.nanmean(ensemble_values)
                ensemble_std = np.nanstd(ensemble_values)
                print(f"{metric_name} - {this_stat_type}:", file=file)
                print(f"\t" + f"Truth: ".ljust(20) + f"({truth_value:.2f}", file=file)
                print(
                    "\t"
                    + "Ensemble mean (std): ".ljust(20)
                    + f"{ensemble_mean:.2f} ({ensemble_std:.2f}) ",
                    file=file,
                )
                print(
                    "\t"
                    + "(mean-truth): ".ljust(20)
                    + f"{(ensemble_mean - truth_value):.2f}",
                    file=file,
                )

                if (abs(ensemble_std)) > 0.00001:
                    factor_in_std_dev = (ensemble_mean - truth_value) / ensemble_std
                    print(
                        "\t"
                        + "(mean-truth) / std dev: ".ljust(20)
                        + f"{factor_in_std_dev:.2f}",
                        file=file,
                    )

                print("\t" + "Ensemble values:".ljust(20), end="", file=file)
                for ensemble_idx, value in enumerate(ensemble_values):
                    print(f" {value:.2f}", end="", file=file)
                    if (ensemble_idx + 1) % 10 == 0:
                        print(
                            "\n\t\t", end="", file=file
                        )  # Print new line after every 10 values

                print("\n", file=file)

    # Close file at the end - but only if we created a file:
    if filename != None:
        file.close()
        with open(filename, "r") as f:
            output = f.read()
            print(output)


def generate_stats_plots(
    results: dict,
    img_dict: dict,
    outdir: Optional[str] = None,
    filename: Optional[str] = None,
    draw_plot: bool = True,
    bigger_plots: bool = False,
    include_intensity_plot: Optional[bool] = True,
    parameter_description: str = "",
    comparison_results: Optional[dict] = None,
) -> None:
    """
    Utility function to plot summary statistic plots

    Note that this displays one plot per metric showing how the metric changes across
        the set of images in img_dict.

    Parameters:
    -------
    results:  contains all min/mean/max values for all metrics considered here
    img_dict:  img_dict.keys() contains the names
    outdir:  path to directory (either relative or absolute) to which figures should be
        saved. must end in a forward slash.
    filename:  filename to save plot to. if None, figure will not be saved
    draw_plot:  whether to display the plot
    bigger_plots:  whether to use a slightly altered format, such as for GFS results
    include_intensity_plot:  whether to plot intensity results
    parameter_description:  label for x-axis to describe units of x-axis. The values
        "noise" and "sigma" are shorthand for longer descriptions -- otherwise, the
        exact string provided will be used.
    comparison_results: only used for GBE plots - contains the metric values for the
        evaluation image.

    Returns:
    -------
    None
    """

    # Intensity is the first entry in list
    if include_intensity_plot:
        metrics = results.keys()
    else:
        metrics = list(results.keys())[1:]

    n_plots = len(metrics)

    image_names = list(img_dict.keys())
    n_images = len(image_names)

    x_values = range(n_images)  # list a consecutive number for each image on the x-axis
    line_format = "o--"

    ncols = 1

    if bigger_plots:
        fig, axs = plt.subplots(
            nrows=n_plots, ncols=ncols, figsize=(12.5 * ncols, 5 * n_plots)
        )
    else:
        fig, axs = plt.subplots(
            nrows=n_plots, ncols=ncols, figsize=(12 * ncols, 4 * n_plots)
        )

    # go through all metrics to be plotted
    for current_index, current_metric in enumerate(metrics):

        # Plot title provides the name of the metric
        plot_title = f"{current_metric}"
        axs[current_index].set_title(plot_title, fontsize=36)

        # Plot the min, mean, and max results for this metric
        min = results[current_metric][0]  # extraxt min values for metric
        mean = results[current_metric][1]  # extract mean values for metric
        max = results[current_metric][2]  # extract max value for metric
        axs[current_index].plot(
            x_values, max, line_format, label="max", color="b"
        )  # print max values in blue
        axs[current_index].plot(
            x_values, mean, line_format, label="mean", color="r"
        )  # print mean values in red
        axs[current_index].plot(
            x_values, min, line_format, label="min", color="y"
        )  # print min values in yellow

        # add legend describing which colors represent min, mean, and max
        # legend = axs[my_index].legend(loc='upper right', bbox_to_anchor=(1.2, 0.5),
        # shadow=False, fontsize=24)  # 'x-large')
        legend = axs[current_index].legend(
            loc="lower right", bbox_to_anchor=(1.3, 0.0), shadow=False, fontsize=24
        )

        # If we're generating a GBE plot:  add horizontal lines representing the values
        # of the evaluation image
        if comparison_results != None:
            ymin = comparison_results[current_metric][0]
            ymean = comparison_results[current_metric][1]
            ymax = comparison_results[current_metric][2]
            axs[current_index].axhline(y=ymax, color="b", linestyle=":")
            axs[current_index].axhline(y=ymean, color="r", linestyle=":")
            axs[current_index].axhline(y=ymin, color="y", linestyle=":")

        # Create tick labels for x-axis:
        axs[current_index].set_xticks(
            x_values
        )  # set the x-values at which to print ticks.
        # axs[my_index].set_xticklabels(xlabels, rotation=90)
        xlabels = image_names  # extract labels to place at those x-values
        if bigger_plots:
            axs_fontsize = 22
        else:
            axs_fontsize = 24
        axs[current_index].set_xticklabels(xlabels, rotation=0, fontsize=axs_fontsize)

        # Set size and format of y tick labels
        axs[current_index].tick_params(axis="y", labelsize=24)  # set size of tick marks
        # Use scientific exponents only when values < 10^(-10) or values > 10^(10)
        axs[current_index].ticklabel_format(style="sci", axis="y", scilimits=(-10, 10))

        # Set label for x-axis to describe units (if available)
        if parameter_description == "noise":
            my_xlabel = "(noise level)"
        elif parameter_description == "sigma":
            my_xlabel = "(blur level, $\sigma$)"
        else:
            my_xlabel = parameter_description
        axs[current_index].set_xlabel(my_xlabel, fontsize=24)
        axs[current_index].xaxis.set_label_coords(0.9, -0.25)

    fig.tight_layout()
    my_fig = plt.gcf()  # Save figure handle
    if draw_plot:
        plt.show()

    if filename is not None:
        print(f"Saving results to {filename}")
        plt.draw()  # Redraw figure to save it
        my_fig.savefig(outdir + filename)  # Save figure to file


def compare_images(
    img_dict: dict,
    selected_metrics: dict,
    outdir: str,
    filename_prefix: str,
    format: str = "IMAGES_NO_RELATION_SPECIFIED",
    bigger_summary_plots: bool = False,
    include_intensity_summary_plot: bool = True,
    parameter_description: str = "",
    plot_heatmaps: bool = True,
    invert_yaxes: bool = True,
    heatmap_metric_cmap_codes: Optional[dict] = None,
    heatmap_min_max_caption: bool = False,
    comparison_results: Optional[dict] = None,
) -> dict:
    """
    Function to analyze sharpness across a set of images

    Parameters:
    ------
    img_dict:  dictionary of images to compare
    selected_metrics:  which metrics to compute
    outdir: path (either relative or absolute) to directory to save plots to. must end
        in a forward slash.
    filename_prefix:  beginning of filename to save plots to
    format:  either "IMAGES_NO_RELATION_SPECIFIED" or "TRUTH_AND_ENSEMBLE_PREDICTION".
        This only affects how summary statistics are displayed in the stats_summary text
        file.
    bigger_summary_plots: whether to utilize larger plots
    include_intensity_summary_plot: whether to plot summary statistics for intensity
    parameter_description:  label for x-axis to describe units of x-axis. The values
        "noise" and "sigma" are shorthand for longer descriptions -- otherwise, the
        exact string provided will be used.
    plot_heatmaps:  whether to plot heatmaps (boolean)
    invert_yaxes:  whether to invert heatmap yaxes
    heatmap_metric_cmap_codes: dictionary mapping metrics to matplotlib colormap names.
    heatmap_min_max_caption: whether to plot minimum and maximum metric values next to
        heatmaps.
    comparison_results: only used for GBE plots - contains the metric values for the
        evaluation image.

    Returns:
    ------
    dictionary of dicionaries:  contains metrics as keys, and the dictionary for each
        metric has the keys "min", "max", and "mean". each of those keys gives a list
        of metric values, with the list corresponding to the images in img_dict.
    """

    if plot_heatmaps and (heatmap_metric_cmap_codes is None):
        raise ValueError("If plotting heatmaps, heatmap_metric_codes must not be None")

    # Assign individual names accordingly
    stats_summary_filename = f"{filename_prefix}_stats_summary.txt"

    if include_intensity_summary_plot:
        heatmap_filename = f"{filename_prefix}_heatmaps.pdf"
        stats_plot_filename = f"{filename_prefix}_stats_plots.pdf"
    else:
        heatmap_filename = f"{filename_prefix}_heatmaps_no_intensity.pdf"
        stats_plot_filename = f"{filename_prefix}_stats_plots_no_intensity.pdf"

    # Generate heatmaps and save plots to image file
    results = compute_metrics(
        img_dict,
        selected_metrics,
        heatmap_metric_cmap_codes,
        outdir=outdir,
        filename=heatmap_filename,
        plot=plot_heatmaps,
        invert_yaxis=invert_yaxes,
        plot_intensity=True,
        min_max_caption=heatmap_min_max_caption,
        return_vals=True,
        uni_ratios=False,
    )

    # Generate summary statistics and save to text file
    write_metric_summary_output(
        results,
        img_dict,
        format=format,
        outdir=outdir,
        filename=stats_summary_filename,
    )

    # Generate plots of statistics and save results to image file
    generate_stats_plots(
        results,
        img_dict,
        outdir=outdir,
        filename=stats_plot_filename,
        draw_plot=True,
        bigger_plots=bigger_summary_plots,
        include_intensity_plot=include_intensity_summary_plot,
        parameter_description=parameter_description,
        comparison_results=comparison_results,
    )

    return results
