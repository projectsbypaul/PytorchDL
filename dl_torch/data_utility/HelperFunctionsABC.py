import math

import numpy as np
import re
import os
from utility.data_exchange import cppIO
import torch
from dl_torch.data_utility.AnnotationForSegmentationABC import __sub_Dataset_from_target_dir_default
from dl_torch.data_utility.AnnotationForSegmentationABC import __sub_Dataset_from_target_dir_inside_outside
from dl_torch.data_utility.AnnotationForSegmentationABC import __sub_Dataset_from_target_dir_edge
from typing import List
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from dl_torch.data_utility.DataParsing import clean_up_files
from dl_torch.data_utility import DataParsing
from utility.data_exchange import cppIOexcavator
from visualization import color_templates
from pathlib import Path
from utility.data_exchange.cppIO import read_float_matrix
import pandas as pd
import gc

def __extract_number(s):
    match = re.search(r'(\d+)(?!.*\d)', s)
    return int(match.group(1)) if match else float('inf')  # Put non-matching items at the end

def __sort_files_names_by_index(file_names):
    return sorted(file_names, key=__extract_number)

def __get_ABC_bin_array_from_segment_dir(segment_dir : str, ignore_list : list[str]):

    file_path = os.listdir(segment_dir)

    filtered_file_path = [f for f in file_path if f not in ignore_list]

    filtered_file_path = __sort_files_names_by_index(filtered_file_path)

    bin_arrays =  []

    for file in filtered_file_path:
        file_path = os.path.join(segment_dir, file)
        _, _, sdf = cppIO.read_3d_array_from_binary(file_path)
        bin_arrays.append(sdf)
    return bin_arrays

def __get_highest_count_class(class_counts):
    # Separate 'Void' count, defaulting to 0 if not present
    void_count = class_counts.get('Void', 0)

    # Create a dictionary of other classes (excluding 'Void')
    other_classes = {key: value for key, value in class_counts.items() if key != 'Void'}

    # Check if all other classes have a count of 0 or if there are no other classes
    all_others_zero = True
    if not other_classes:  # Handles case where only 'Void' might exist
        all_others_zero = True
    else:
        all_others_zero = all(count == 0 for count in other_classes.values())

    if all_others_zero:
        # If all other classes are 0, the result is 'Void' if its count > 0
        if void_count > 0:
            return 'Void'
        else:
            # All classes (including Void) are 0 or not present meaningfully
            return None  # Or "No significant class" or similar
    else:
        # Find the class with the highest count among 'other_classes'
        # Filter out items with 0 count before finding max, to avoid issues if all are 0 (covered by above)
        # but good practice if only some are zero.

        # Find the maximum count in other_classes
        max_count_other = 0
        if other_classes:  # Ensure other_classes is not empty
            max_count_other = max(other_classes.values())

        # If all remaining 'other_classes' actually ended up being zero (e.g. {'A':0, 'B':0})
        # this can happen if the initial check for all_others_zero passed because of non-zero
        # values that were then not considered the max.
        # However, our `all_others_zero` check should cover this primarily.
        # The main goal here is to find the key(s) for that max_count_other if it's > 0.

        if max_count_other == 0:
            # This case should ideally be caught by `all_others_zero` leading to checking Void.
            # If somehow reached and void_count > 0, it implies an edge case not fully handled.
            # But based on the logic, if other_classes has items and their max is 0,
            # then all_others_zero should have been true.
            # For robustness, if 'Void' is the only option left with a positive count:
            if void_count > 0:
                return 'Void'
            return None

        # Find all classes that have this maximum count
        highest_classes = [key for key, value in other_classes.items() if value == max_count_other]

        if not highest_classes:  # Should not happen if max_count_other > 0
            return None

        # If there's a tie, you might want to define how to handle it.
        # This example returns the first one found in case of a tie.
        return highest_classes[0]

def __evaluate_voxel_class_kernel(grid, target_idx, k, class_weights):
    D, _, _, C = grid.shape
    x, y, z = target_idx
    half_k = k // 2

    # Define neighborhood bounds, clamped to grid dimensions
    x_min = max(x - half_k, 0)
    x_max = min(x + half_k + 1, D)
    y_min = max(y - half_k, 0)
    y_max = min(y + half_k + 1, D)
    z_min = max(z - half_k, 0)
    z_max = min(z + half_k + 1, D)

    # Extract the neighborhood (k x k x k x C)
    neighborhood = grid[x_min:x_max, y_min:y_max, z_min:z_max, :]

    # Count occurrences of each class
    class_counts = np.sum(neighborhood, axis=(0, 1, 2))

    # Apply class weights
    weighted_counts = class_counts * np.array(class_weights)

    # Return the index of the class with the highest weighted count
    return int(np.argmax(weighted_counts))

def __permute_labels_of_dataset_dir(target_dir : str):

    dataset_names = os.listdir(target_dir)
    dataset_paths = [os.path.join(target_dir, name) for name in dataset_names]

    for subset_path in dataset_paths:

        dataset = InteractiveDataset.load_dataset(subset_path)
        dataset.labels = torch.permute(dataset.labels, (0,4,1,2,3))
        dataset.save_dataset(subset_path)


def get_ABC_segment_info_from_parquet(parquet_loc : str, source_loc : str):

    loaded_df = pd.read_parquet(parquet_loc, engine='pyarrow')

    segment_info = []

    for index, entry in loaded_df.iterrows():
        f_name= f"{entry["ID"]}_{entry["Segment"]}.bin"
        sub_dir = entry["ID"]
        segment = entry["Segment"]
        segment_info.append([sub_dir, segment, f_name])

    return segment_info

def main():
  pass

if __name__ == "__main__":
    main()
