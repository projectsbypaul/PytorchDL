import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
import pandas as pd
import os

def __histogramm_segmentation_samples():
    # Load data
    data_loc = r"../../data/training_statistics"
    data_name = "UNet_Segmentation_sample"

    with open(os.path.join(data_loc, f"{data_name}.pkl"), "rb") as f:
        sample_result = pickle.load(f)

    df = pd.DataFrame(sample_result, columns=['Sample_ID', 'ABC_ID', 'Accuracy'])

    df.to_csv(os.path.join(data_loc, f"{data_name}.csv"))

    sample_iou = np.array([item[2]*100 for item in sample_result])

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
    mode_val = mode(sample_iou, keepdims=True)[0][0]

    # Add vertical lines
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='dotted', linewidth=1.5, label=f'Median: {median_val:.2f}')
    ax.axvline(mode_val, color='blue', linestyle='solid', linewidth=1.5, label=f'Mode: {mode_val:.2f}')

    # Titles and limits
    ax.set_title('Histogram of IoU on ABC samples')
    ax.set_xlim(0, x_limit)
    ax.set_xlabel("IoU Value")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    plt.show()

def main():
    __histogramm_segmentation_samples()

if __name__ == "__main__":
    main()