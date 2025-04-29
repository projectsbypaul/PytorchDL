import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode

def create_plot():
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


def main():
    create_plot()

if __name__ == "__main__":
    main()