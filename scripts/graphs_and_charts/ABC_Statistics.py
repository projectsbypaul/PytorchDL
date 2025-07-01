import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from utility.data_exchange import cppIOexcavator
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FixedLocator, FuncFormatter

def create_plot_dimensions():
    # Load CSV
    csv_file = r"C:\Local_Data\ABC\ABC_parsed_files\log.csv"  # Replace with your actual file path
    df = pd.read_csv(csv_file)

    # Columns to plot
    columns = ['voxel_size', 'dimx', 'dimy', 'dimz']

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    bins = {
        'voxel_size': 100,
        'dimx': 500,
        'dimy': 500,
        'dimz': 500
    }

    x_limits = {
        'voxel_size': (0, 5),
        'dimx': (0, 1000),
        'dimy': (0, 1000),
        'dimz': (0, 1000)
    }

    filtered_df = df[df['dimx'] !=float('-inf')].copy()

    filtered_df.loc[:,['voxel_size', 'dimx', 'dimy', 'dimz']] = filtered_df[['voxel_size', 'dimx', 'dimy', 'dimz']] * 1000

    filtered_df = filtered_df[filtered_df["voxel_size"]<5]

    for i, col in enumerate(columns):
        data = filtered_df[col].dropna()
        ax = axs[i]

        # Plot histogram
        ax.hist(data, bins=bins[col], alpha=0.6, edgecolor='black')

        # Calculate statistics
        mean_val = data.mean()
        median_val = data.median()
        mode_val = mode(data, keepdims=True)[0][0]

        # Add vertical lines
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='dotted', linewidth=1.5, label=f'Median: {median_val:.2f}')
        ax.axvline(mode_val, color='blue', linestyle='solid', linewidth=1.5, label=f'Mode: {mode_val:.2f}')

        ax.set_title(f'Histogram of {col}')
        ax.set_xlim(x_limits[col])
        ax.legend()

    plt.tight_layout()
    plt.show()

def create_plot_surf_types(data_dir : str):
    #Section:
    #Data Parsing

    f_name = "segmentation_data.dat"

    sub_dirs = os.listdir(data_dir)

    targets = []

    for sub in sub_dirs:
        name = os.path.join(data_dir, sub, f_name)
        if os.path.exists(name):
            targets.append(name)

    raw_type_dicts = []
    for file in targets:
        print(f"reading dict from {os.path.dirname(file)}")
        type_dict = cppIOexcavator.parse_dat_file(file)['TYPE_COUNT_MAP']
        raw_type_dicts.append(type_dict)


    key_collection = []
    for index, type_dict in enumerate(raw_type_dicts):
        print(f"extracting dict keys from {os.path.dirname(targets[index])}")
        for key in type_dict.keys():
            key_collection.append(key)

    unique_types = list(set(key_collection))

    df_part_count = dict()
    df_face_count = dict()

    for index, surf_type in enumerate(unique_types):
        df_part_count.update({surf_type: 0})
        df_face_count.update({surf_type: 0})

    for type_dict in raw_type_dicts:
        for key in type_dict:
            df_part_count[key] += 1
            df_face_count[key] += type_dict[key]

    # Section:
    # Data Visualization

    # Sort categories by face count
    sorted_items = sorted(df_face_count.items(), key=lambda item: item[1], reverse=True)
    categories = [key for key, _ in sorted_items]
    values1 = [df_part_count[cat] for cat in categories]
    values2 = [df_face_count[cat] for cat in categories]

    x = np.arange(len(categories))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # --- Define number of ticks ---
    n_ticks = 5  # same for both axes

    # --- Define baseline of ticks ---
    left_start = 10 ** 4
    right_start = 1

    # --- Left Y-Axis: Faces ---
    left_ticks = [left_start * (10 ** i) for i in range(n_ticks)]
    bars1 = ax1.bar(x - width / 2, values2, width=width,
                    label='Faces per Category', color='#006AB3')
    ax1.set_ylabel('Face Count', color='#006AB3')
    ax1.set_yscale('log')
    ax1.set_yticks(left_ticks)
    ax1.yaxis.set_major_locator(FixedLocator(left_ticks))
    ax1.tick_params(axis='y', labelcolor='#006AB3')

    # --- Add minor ticks and grid for left y-axis only ---
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
    ax1.tick_params(axis='y', which='minor', length=4, color='#006AB3')
    ax1.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.5)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)

    # --- Right Y-Axis: Parts ---
    right_ticks = [right_start * (10 ** i) for i in range(n_ticks)]
    bars2 = ax2.bar(x + width / 2, values1, width=width,
                    label='Parts per Category', color='#00893A')
    ax2.set_ylabel('Part Count', color='#00893A')
    ax2.set_yscale('log')
    ax2.set_yticks(right_ticks)
    ax2.yaxis.set_major_locator(FixedLocator(right_ticks))
    ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
    ax2.tick_params(axis='y', labelcolor='#00893A', which='both', length=4, color='#00893A')

    # --- Disable grid on right axis ---
    ax2.grid(False)

    # --- Align tick positions visually ---
    ax1.set_ylim(bottom=left_ticks[0], top=left_ticks[-1])
    ax2.set_ylim(bottom=right_ticks[0], top=right_ticks[-1])

    # --- Ensure grid is behind bars ---
    ax1.set_axisbelow(True)

    # --- Shared x-axis ---
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.set_title('Surface Type Statistics: Face vs Part Count')

    # --- Combined legend ---
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()


def main():
    target = r"H:\ABC\ABC_Datasets\Segmentation\training_samples\train_1000000_ks_16_pad_4_bw_5_vs_adaptive_n3"
    create_plot_surf_types(target)

if __name__ == "__main__":
    main()