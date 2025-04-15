import numpy as np
import re
import os
from utility.data_exchange import cppIO

def create_ABC_labels_from_bin_array(segment_dir, source_dir,bin_arrays, origins, vertex_type_map, vertex_to_index_map):



    pass

def extract_number(s):
    match = re.search(r'(\d+)(?!.*\d)', s)
    return int(match.group(1)) if match else float('inf')  # Put non-matching items at the end

def sort_files_names_by_index(file_names):
    return sorted(file_names, key=extract_number)

def get_ABC_bin_arry_from_segment_dir(segment_dir : str,ignore_list : list[str]):

    file_path = os.listdir(segment_dir)

    filtered_file_path = [f for f in file_path if f not in ignore_list]

    filtered_file_path = sort_files_names_by_index(filtered_file_path)

    bin_arrays =  []

    for file in filtered_file_path:
        file_path = os.path.join(segment_dir, file)
        _, _, sdf = cppIO.read_3d_array_from_binary(file_path)
        bin_arrays.append(sdf)
    return bin_arrays

def evaluate_voxel_class_kernel(grid, target_idx, k, class_weights):
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

def main():
    pass

if __name__ == "__main__":
    main()
