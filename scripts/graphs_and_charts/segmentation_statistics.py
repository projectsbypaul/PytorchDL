import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import pandas as pd
import os

from numpy.ma.extras import hstack

import visualization.color_templates
from visualization import color_templates
from scipy import stats

def __plot_confusion_matrix(ccm_result_loc: str, class_plate):

    class_list = color_templates.get_class_list(class_plate)

    with open(ccm_result_loc, "rb") as f:
        ccm = pickle.load(f)

    n_predictions = np.sum(ccm, axis=1)

    ccm_norm = []

    for i in range(ccm.shape[0]):
        row = ccm[i]
        if n_predictions[i] > 0:
            row = row / n_predictions[i]

        ccm_norm.append(row)

    ccm_norm = np.asarray(ccm_norm)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_red_green",
        [(0.0, "#ffffffff"),
         (0.5, "#5e65dbb8"),
         (1.0, "#00dc73ff")]
    )

    # Plot
    sns.heatmap(
        ccm_norm,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0, vmax=1,
        xticklabels=class_list,
        yticklabels=class_list,
        cbar=True,
        linewidths = 0.5,
        linecolor = "black"
    )
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    plt.title("Confusion Matrix",)

    # Put x-axis labels on top
    plt.gca().xaxis.set_ticks_position("top")
    plt.gca().xaxis.set_label_position("top")

    # Rotate all tick labels
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    ax = plt.gca()

    # Make all four spines (borders) visible and thicker
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor("black")

    # Optional: remove ticks from the bottom/right axes
    ax.tick_params(bottom=False, right=False)

    plt.tight_layout()
    plt.show()



def __histogramm_segmentation_samples(val_result_loc :  str):

    # create model signature
    model_name = os.path.basename(val_result_loc)
    model_name, _ = os.path.splitext(model_name)

    with open(val_result_loc, "rb") as f:
        sample_result = pickle.load(f)

    df = pd.DataFrame(sample_result, columns=['Sample_ID', 'ABC_ID', 'Accuracy'])

    current_path = os.path.abspath(val_result_loc)
    path_without_ext = os.path.splitext(current_path)[0]

    df.to_csv(path_without_ext + ".csv")

    sample_iou = np.array([item[2]*100 for item in sample_result])
    rounded_data = np.round(sample_iou, 2)
    total_samples = len(sample_iou)

    # Settings
    x_limit = 100
    bins = 100

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot histogram and capture bar info
    counts, bin_edges, patches = ax.hist(sample_iou, bins=bins, alpha=0.6, edgecolor='black')

    # Annotate each bar with count value
    for count, x in zip(counts, bin_edges[:-1]):
        if count > 0:
            ax.text(x + (bin_edges[1] - bin_edges[0]) / 2, count, f'{int(count)}',
                    ha='center', va='bottom', fontsize=8, rotation=90)

    # Calculate statistics
    mean_val = np.mean(sample_iou)
    median_val = np.median(sample_iou)

    mode_result = stats.mode(rounded_data, keepdims=True)
    mode_val = mode_result.mode[0]
    mode_count = mode_result.count[0]

    # Calculate percentiles
    p25 = np.percentile(sample_iou, 25)
    p75 = np.percentile(sample_iou, 75)

    # Add vertical lines
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax.axvline(mode_val, color='blue', linestyle='solid', linewidth=1.5, label=f'Mode: {mode_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='dotted', linewidth=1.5, label=f'Median: {median_val:.2f}')
    ax.axvline(p25, color='orange', linestyle='dashdot', linewidth=1.5, label=f'25th Percentile: {p25:.2f}')
    ax.axvline(p75, color='purple', linestyle='dashdot', linewidth=1.5, label=f'75th Percentile: {p75:.2f}')

    # Titles and limits
    ax.set_title(f'Histogram of IoU on ABC samples\nTotal samples: {total_samples} \n{model_name}')
    ax.set_xlim(0, x_limit)
    ax.set_xlabel("IoU Value")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    plt.show()

def __epoch_from_string(filename: str):
    after_ep = filename.split("EP", 1)[1]
    ep_number = after_ep.split("_", 1)[0]
    return int(ep_number)

def __list_matching_files(root: str, search_str: str):

    f_names = np.asarray(os.listdir(root))

    matching = np.asarray([search_str in name for name in f_names])

    matches = f_names[matching]

    matches_path = [os.path.join(root, match) for match in matches]

    return matches_path

def __extract_distribution_metrics(val_result_bin: str):
    # create model signature
    model_name = os.path.basename(val_result_bin)
    model_name, _ = os.path.splitext(model_name)

    with open(val_result_bin, "rb") as f:
        sample_result = pickle.load(f)

    df = pd.DataFrame(sample_result, columns=['Sample_ID', 'ABC_ID', 'Accuracy'])

    current_path = os.path.abspath(val_result_bin)
    path_without_ext = os.path.splitext(current_path)[0]

    df.to_csv(path_without_ext + ".csv")

    sample_iou = np.array([item[2] * 100 for item in sample_result])
    rounded_data = np.round(sample_iou, 2)
    total_samples = len(sample_iou)

    # Calculate statistics
    mean_val = np.mean(sample_iou)
    median_val = np.median(sample_iou)

    mode_result = stats.mode(rounded_data, keepdims=True)
    mode_val = mode_result.mode[0]
    mode_count = mode_result.count[0]

    # Calculate percentiles
    p25 = np.percentile(sample_iou, 25)
    p75 = np.percentile(sample_iou, 75)


    distribution_metrics={
        "mean":mean_val,
        "median":median_val,
        "mode":mode_val,
        "p25":p25,
        "p75":p75
    }

    return distribution_metrics

def __extract_diagonal_from_cmm(cmm_bin : str, class_template):

    class_list = color_templates.get_class_list(class_template)

    with open(cmm_bin, "rb") as f:
        ccm = pickle.load(f)

    n_predictions = np.sum(ccm, axis=1)

    ccm_norm = []

    for i in range(ccm.shape[0]):
        row = ccm[i]
        if n_predictions[i] > 0:
            row = row / n_predictions[i]

        ccm_norm.append(row)

    ccm_norm = np.asarray(ccm_norm)

    diagonal = np.ndarray(shape=len(class_list), dtype=float)

    for i in range(len(class_list)):
        diagonal[i] = ccm_norm[i,i]

    return diagonal

def __aggregate_diagonal(root: str, search_str : str, class_template):

    ccm_file_paths = __list_matching_files(root, search_str)

    class_list = color_templates.get_class_list(class_template)

    column_names = np.hstack([["EP"], np.asarray(class_list)])

    data_arr = np.ndarray(shape=(len(ccm_file_paths) ,len(class_list) +1))

    for i, f in enumerate(ccm_file_paths):
        diagonal = __extract_diagonal_from_cmm(f, class_template)
        ep = __epoch_from_string(os.path.basename(f))
        row = np.hstack([ep, diagonal])
        data_arr[i] = row

    diagonal_df = pd.DataFrame(data_arr, columns=column_names)

    diagonal_df_sorted = diagonal_df.sort_values(by=['EP'])

    return diagonal_df_sorted

def __aggregate_distribution_metrics(root: str, search_str : str, class_template):

    stats_file_paths = __list_matching_files(root, search_str)

    class_list = color_templates.get_class_list(class_template)

    keys = __extract_distribution_metrics(stats_file_paths[0]).keys()
    keys = [k for k in keys]

    data_arr = data_arr = np.ndarray(shape=(len(stats_file_paths) ,len(keys) +1))

    column_names = np.hstack([["EP"], np.asarray(keys)])

    for i, f in enumerate(stats_file_paths):

        dist_stats = __extract_distribution_metrics(f)
        ep = __epoch_from_string(os.path.basename(f))
        vals = dist_stats.values()
        vals = [v for v in vals]
        row = np.hstack([[ep], vals])
        data_arr[i] = row

    stat_df = pd.DataFrame(data_arr, columns=column_names)

    stat_df_sorted = stat_df.sort_values(by=['EP'])

    return stat_df_sorted

def __line_chart_from_dataframe(df: pd.DataFrame, x_label: str, y_label: str, title: str = "Line Chart from DataFrame"):

    plt.figure(figsize=(8, 5))
    for col in df.columns[1:]:  # skip the first column (X)
        plt.plot(df["EP"], df[col], label=col)

    plt.xlabel("X")
    plt.ylabel("Values")
    plt.title("Line Chart from DataFrame")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_default_line_charts():

    root = r"H:\ws_hpc_workloads\hpc_val\Balanced20k"
    search_str = ["_val_result_mcm.bin", "val_result.bin"]
    template = color_templates.inside_outside_color_template_abc()

    df_stats = __aggregate_distribution_metrics(root, search_str[1], template)
    df_ccm = __aggregate_diagonal(root, search_str[0], template)

    __line_chart_from_dataframe(df_stats[["EP", "median", "mean"]], "EP", "abs")
    class_filter = np.asarray(color_templates.get_class_list(template)[:6])
    class_filter = np.hstack(["EP", class_filter])
    __line_chart_from_dataframe(df_ccm[class_filter], "EP", "p%")

    ccm_average = pd.DataFrame({
        "EP": df_ccm["EP"],
        "ccm_mean": df_ccm.iloc[:, 1:].mean(axis=1) * 100
    })

    __line_chart_from_dataframe(ccm_average, "EP", "p%")

    df_abc = df_stats[["EP", "median", "mean"]].merge(ccm_average[["EP", "ccm_mean"]], on="EP", how="left")

    __line_chart_from_dataframe(df_abc, "EP", "p%")

def main():

    stats_file = r"H:\ws_hpc_workloads\hpc_val\TEST_INOUT\TEST_INOUT_Balance_Test_02_UNet3D_Hilbig_mfcb_EP50_val_result_mcm.bin"
    template = color_templates.inside_outside_color_template_abc()
    __plot_confusion_matrix(stats_file, template)



    result_file = r"H:\ws_hpc_workloads\hpc_val\TEST_INOUT\TEST_INOUT_Balance_Test_02_UNet3D_Hilbig_mfcb_EP50_val_result.bin"
    __histogramm_segmentation_samples(result_file)





if __name__ == "__main__":
    main()