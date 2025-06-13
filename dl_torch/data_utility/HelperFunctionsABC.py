import math

import numpy as np
import re
import os
from utility.data_exchange import cppIO
import torch
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from dl_torch.data_utility.DataParsing import clean_up_files
from dl_torch.data_utility import DataParsing
from utility.data_exchange import cppIOexcavator
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

def create_ABC_Dataset_from_parquet(parquet_name :str):

    # parquet_name = "ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2_balanced_n_1000"

    segment_dir = r"C:\Local_Data\ABC\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2"
    parquet_dir = r"C:\Local_Data\ABC\ABC_statistics\balance_parquets\ABC_chunk_01"

    source_dir = r"C:\Local_Data\ABC\ABC_parsed_files"
    torch_dir = r"C:\Local_Data\ABC\ABC_torch"


    ignored_files = ["origins.bin", "VertToGridIndex.bin", "VertTypeMap.bin", "TypeCounts.bin", "FaceTypeMap.bin", "FaceToGridIndex.bin"]
    n_min_files = 5

    # Set up dictionary
    class_list = np.array(['BSpline','Cone','Cylinder','Extrusion','Other','Plane','Revolution','Sphere','Torus','Void'])
    class_list = np.sort(class_list)
    class_indices = np.arange(len(class_list))
    class_lot = dict(zip(class_list, class_indices))
    index_lot = dict(zip(class_indices, class_list, ))


    parquet_loc = os.path.join(parquet_dir, parquet_name + ".parquet")
    segment_info = get_ABC_segment_info_from_parquet(parquet_loc, segment_dir)

    segment_path = [os.path.join(segment_dir, item[0], item[2]) for item in segment_info]

    bin_arrays = []

    labels = []

    for p_index, path in enumerate(segment_path):

        segment_index = segment_info[p_index][1]

        if len(os.listdir(os.path.dirname(path))) > n_min_files:

            origins = cppIO.read_float_matrix(os.path.dirname(path) + "/origins.bin")
            face_type_map = cppIO.read_type_map_from_binary(os.path.dirname(path) + "/FaceTypeMap.bin")
            face_to_index_map = cppIO.read_float_matrix(os.path.dirname(path) + "/FaceToGridIndex.bin")
            _, _, grid = cppIO.read_3d_array_from_binary(path)

            bin_arrays.append(grid)

            df_voxel_count = dict()

            for index, surf_type in enumerate(class_list):
                df_voxel_count.update({surf_type: 0})

            grid_dim = grid.shape[0]

            origin = np.asarray(origins[segment_index])

            top = origin + [grid_dim - 1, grid_dim - 1, grid_dim - 1]

            label = np.zeros(shape=[grid_dim, grid_dim, grid_dim, class_list.shape[0]])

            write_count = 0

            for face_index, face_center in enumerate(face_to_index_map):

                if origin[0] <= face_center[0] <= top[0] and origin[1] <= face_center[1] <= top[1] and origin[2] <= \
                        face_center[2] <= \
                        top[2]:
                    grid_index = face_center - origin

                    type_string = face_type_map[face_index]
                    one_hot_index = class_lot[type_string[0]]
                    label[int(grid_index[0]), int(grid_index[1]), int(grid_index[2]), one_hot_index] += 1
                    write_count += 1

            # print(f"wrote {write_count} labels for part {path}")

            for i, j, k in np.ndindex(label.shape[0], label.shape[1], label.shape[2]):
                voxel = label[i, j, k, :]

                if np.sum(voxel) > 0:
                    max_index = np.argmax(voxel)
                    label[i, j, k, :] = np.zeros_like(voxel)
                    label[i, j, k, max_index] = 1
                    df_voxel_count[index_lot[max_index]] += 1
                else:
                    label[i, j, k, class_lot["Void"]] = 1
                    df_voxel_count['Void'] += 1

            # print(f"Writer Counter part {path} grid {grid_index}")
            # print(df_voxel_count.keys())
            # print(df_voxel_count.values())

            labels.append(label)

            print(f"Added {segment_info[p_index]} to data set... {p_index + 1} of {len(segment_path)} processed")

    data = torch.tensor(np.array(bin_arrays))
    labels = torch.tensor(np.array(labels))
    labels = torch.permute(labels, (0, 4, 1, 2, 3))
    dataset = InteractiveDataset(data, labels, class_lot, set_name=parquet_name)

    dataset.save_dataset(os.path.join(torch_dir, parquet_name + ".torch"))

def create_ABC_AE_sub_Dataset():

    data_dir = r"C:\Local_Data\ABC\ABC_AE_Data_ks_16_pad_4_bw_5_vs_adaptive"
    label_dir = r"C:\Local_Data\ABC\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2"
    torch_dir = r"C:\Local_Data\ABC\ABC_torch\AE_ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2"

    ignored_files = ["origins.bin", "VertToGridIndex.bin", "VertTypeMap.bin", "TypeCounts.bin", "FaceTypeMap.bin",
                     "FaceToGridIndex.bin"]

    n_min_files_data = 3
    n_min_files_label = 5

    data_paths = os.listdir(data_dir)
    label_path = os.listdir(label_dir)

    if not(os.path.exists(torch_dir)):
        os.makedirs(torch_dir)

    for index, path in enumerate(data_paths):

        full_data_path = os.path.join(data_dir, path)

        dir_name = os.path.basename(full_data_path)

        full_label_path = os.path.join(label_dir, dir_name)

        if not (os.path.exists(full_label_path)):
            print(f"skipped file {path} - label dir does not exist")
            continue

        n_f_data = len(os.listdir(full_data_path))
        n_f_label =  len(os.listdir(full_label_path))

        if (n_f_data < n_min_files_data) or (n_f_label < n_min_files_label):
            print(f"skipped file {path} - dir was not correctly pre processed")
            continue

        data_origins = np.asarray(read_float_matrix(os.path.join(data_dir, path, "origins.bin")))
        label_origins = np.asarray(read_float_matrix(os.path.join(label_dir, path, "origins.bin")))

        if data_origins.shape == label_origins.shape:
            delta = np.sum(np.asarray(data_origins) - np.asarray(label_origins))
            if delta == 0:
                data_arrays = __get_ABC_bin_array_from_segment_dir(full_data_path, ignored_files)
                label_arrays = __get_ABC_bin_array_from_segment_dir(full_label_path, ignored_files)

                data = torch.tensor(np.array(data_arrays))
                labels = torch.tensor(np.array(label_arrays))
                sub_dataset = InteractiveDataset(data, labels, set_name=path)

                sub_dataset.save_dataset(os.path.join(torch_dir, path + ".torch"))
            else:
                print(f"skipped file {path} - origin missaligned")
        else:
            print(f"skipped file {path} - segment ammount not equal")

def __sub_Dataset_from_target_dir(target_dir: str, class_list, class_lot, index_lot):

    bin_array_file = target_dir + "/segmentation_data_segments.bin"
    segment_info_file = target_dir + "/segmentation_data.dat"

    segment_data = cppIOexcavator.parse_dat_file(segment_info_file)

    origins = segment_data["ORIGIN_CONTAINER"]["data"]
    face_type_map = np.array(list(segment_data["FACE_TYPE_MAP"].values()))
    face_to_index_map = segment_data["FACE_TO_GRID_INDEX_CONTAINER"]["data"]
    uniques = segment_data['TYPE_COUNT_MAP']

    bin_arrays = cppIOexcavator.load_segments_from_binary(bin_array_file)

    labels = []

    for grid_index, grid in enumerate(bin_arrays):

        df_voxel_count = dict()

        for index, surf_type in enumerate(class_list):
            df_voxel_count.update({surf_type: 0})

        grid_dim = grid.shape[0]

        origin = np.asarray(origins[grid_index])

        top = origin + [grid_dim - 1, grid_dim - 1, grid_dim - 1]

        label = np.zeros(shape=[grid_dim, grid_dim, grid_dim, class_list.shape[0]])

        write_count = 0

        for face_index, face_center in enumerate(face_to_index_map):

            if origin[0] <= face_center[0] <= top[0] and origin[1] <= face_center[1] <= top[1] and origin[2] <= \
                    face_center[2] <= \
                    top[2]:
                grid_index = face_center - origin

                type_string = face_type_map[face_index]
                one_hot_index = class_lot[type_string]
                label[int(grid_index[0]), int(grid_index[1]), int(grid_index[2]), one_hot_index] += 1
                write_count += 1

        # print(f"wrote {write_count} labels for part {path}")

        for i, j, k in np.ndindex(label.shape[0], label.shape[1], label.shape[2]):
            voxel = label[i, j, k, :]

            if np.sum(voxel) > 0:
                max_index = np.argmax(voxel)
                label[i, j, k, :] = np.zeros_like(voxel)
                label[i, j, k, max_index] = 1
                df_voxel_count[index_lot[max_index]] += 1
            else:
                label[i, j, k, class_lot["Void"]] = 1
                df_voxel_count['Void'] += 1

        # print(f"Writer Counter part {path} grid {grid_index}")
        # print(df_voxel_count.keys())
        # print(df_voxel_count.values())

        labels.append(label)

    data = torch.tensor(np.array(bin_arrays))
    labels = torch.tensor(np.array(labels))

    return data, labels

def create_ABC_sub_Dataset(segment_dir : str, torch_dir : str, n_min_files :  int):

    # Set up dictionary
    class_list = np.array(['BSpline','Cone','Cylinder','Extrusion','Other','Plane','Revolution','Sphere','Torus','Void'])
    class_list = np.sort(class_list)
    class_indices = np.arange(len(class_list))
    class_lot = dict(zip(class_list, class_indices))
    index_lot = dict(zip(class_indices, class_list, ))

    segment_paths = os.listdir(segment_dir)

    if not(os.path.exists(torch_dir)):
        os.makedirs(torch_dir)

    for path in segment_paths:

        full_path = os.path.join(segment_dir, path)

        if len(os.listdir(full_path)) >= n_min_files:

            data, labels = __sub_Dataset_from_target_dir(full_path, class_list, class_lot, index_lot)

            sub_dataset = InteractiveDataset(data, labels, class_lot, set_name=path)

            sub_dataset.save_dataset(os.path.join(torch_dir, path + ".torch"))

def create_ABC_sub_Dataset_from_job(job_file: str, segment_dir : str, torch_dir : str, n_min_files :  int):

    job_targets = DataParsing.read_job_file(job_file)

    # Set up dictionary
    class_list = np.array(['BSpline','Cone','Cylinder','Extrusion','Other','Plane','Revolution','Sphere','Torus','Void'])
    class_list = np.sort(class_list)
    class_indices = np.arange(len(class_list))
    class_lot = dict(zip(class_list, class_indices))
    index_lot = dict(zip(class_indices, class_list, ))

    if not(os.path.exists(torch_dir)):
        os.makedirs(torch_dir)

    for target in job_targets:

        full_path = os.path.join(segment_dir, target)

        if len(os.listdir(full_path)) >= n_min_files:

            data, labels = __sub_Dataset_from_target_dir(full_path, class_list, class_lot, index_lot)

            labels = labels.permute(0, 4, 1, 2, 3)

            sub_dataset = InteractiveDataset(data, labels, class_lot, set_name=target)

            sub_dataset.save_dataset(os.path.join(torch_dir, target + ".torch"))

def batch_ABC_sub_Datasets(source_dir: str, target_dir, dataset_name: str, batch_count: int):
    parent_folder = Path(source_dir)
    file_extension = ".torch"
    file_paths = list(parent_folder.rglob(f"*{file_extension}"))

    batch_size = math.ceil(len(file_paths)/batch_count)
    batch_start_indicies = [index * batch_size for index in range(batch_count)]

    for index, start_index in enumerate(batch_start_indicies):
        data_set_joined = InteractiveDataset.load_dataset(file_paths[start_index])

        end_index = start_index + batch_size if index < (batch_count - 1) else len(file_paths)

        for i in range(start_index + 1, end_index):
            try:
                data_set = InteractiveDataset.load_dataset(file_paths[i])
                data_set_joined.data = torch.vstack([data_set_joined.data, data_set.data])
                data_set_joined.labels = torch.vstack([data_set_joined.labels, data_set.labels])
                print(f"merged {os.path.basename(file_paths[i])} into {data_set_joined.get_name()}")

                # Free memory
                del data_set
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"merge failed on subset {i} ({file_paths[i]}): {e}")

        batch_name = dataset_name + "_batch_" + str(index)
        save_name = f"{target_dir}/{batch_name}.torch"
        data_set_joined.save_dataset(save_name)

        print(data_set_joined.get_info())

        # Free memory of joined dataset before next batch
        del data_set_joined
        gc.collect()
        torch.cuda.empty_cache()

def main():
    __permute_labels_of_dataset_dir(r"H:\ABC\ABC_torch\ABC_chunk_00\batched_data_ks_16_pad_4_bw_5_vs_adaptive_n2_testing")

if __name__ == "__main__":
    main()
