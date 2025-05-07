import numpy as np
import re
import os
from utility.data_exchange import cppIO
import torch
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from dl_torch.data_utility.DataParsing import clean_up_files
from dl_torch.data_utility import DataParsing
from pathlib import Path
from utility.data_exchange.cppIO import read_float_matrix

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
                data_arrays = get_ABC_bin_arry_from_segment_dir(full_data_path, ignored_files)
                label_arrays = get_ABC_bin_arry_from_segment_dir(full_label_path, ignored_files)

                data = torch.tensor(np.array(data_arrays))
                labels = torch.tensor(np.array(label_arrays))
                sub_dataset = InteractiveDataset(data, labels, set_name=path)

                sub_dataset.save_dataset(os.path.join(torch_dir, path + ".torch"))
            else:
                print(f"skipped file {path} - origin missaligned")
        else:
            print(f"skipped file {path} - segment ammount not equal")

def create_ABC_sub_Dataset():

    segment_dir = r"C:\Local_Data\ABC\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2"
    source_dir = r"C:\Local_Data\ABC\ABC_parsed_files"
    torch_dir = r"C:\Local_Data\ABC\ABC_torch\torch_data_ks_16_pad_4_bw_5_vs_adaptive_n2"

    ignored_files = ["origins.bin", "VertToGridIndex.bin", "VertTypeMap.bin", "TypeCounts.bin", "FaceTypeMap.bin", "FaceToGridIndex.bin"]
    n_min_files = 5

    # Set up dictionary
    class_list = np.array(['Cone', 'Revolution', 'Sphere', 'Plane', 'Extrusion', 'Other', 'Cylinder', 'Torus', 'BSpline', 'Void'])
    class_list = np.sort(class_list)
    class_indices = np.arange(len(class_list))
    class_lot = dict(zip(class_list, class_indices))
    index_lot = dict(zip(class_indices, class_list, ))

    segment_paths = os.listdir(segment_dir)

    if not(os.path.exists(torch_dir)):
        os.makedirs(torch_dir)

    for path in segment_paths:

        full_path = os.path.join(segment_dir, path)

        if len(os.listdir(full_path)) > n_min_files:

            origins = cppIO.read_float_matrix(full_path + "/origins.bin")
            face_type_map = cppIO.read_type_map_from_binary(full_path + "/FaceTypeMap.bin")
            face_to_index_map = cppIO.read_float_matrix(full_path + "/FaceToGridIndex.bin")

            bin_arrays = get_ABC_bin_arry_from_segment_dir(full_path, ignored_files)

            obj_path = os.path.join(source_dir, path ,path + ".obj")

            _ , faces = DataParsing.parse_obj(obj_path)

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

                #print(f"Writer Counter part {path} grid {grid_index}")
                # print(df_voxel_count.keys())
                # print(df_voxel_count.values())

                labels.append(label)


            data = torch.tensor(np.array(bin_arrays))
            labels = torch.tensor(np.array(labels))
            sub_dataset = InteractiveDataset(data, labels, class_lot, set_name=path)



            sub_dataset.save_dataset(os.path.join(torch_dir, path + ".torch"))

def join_ABC_sub_Datasets():
    # Define the parent folder and file extension
    parent_folder = Path(r"C:\Local_Data\ABC\ABC_torch\AE_ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2") # Replace with your folder path
    joined_name = "AE_ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2"
    file_extension = ".torch"  # Change to your desired extension

    # Get all matching file paths
    file_paths = list(parent_folder.rglob(f"*{file_extension}"))

    data_set_joined = InteractiveDataset.load_dataset(file_paths[0])

    # clean_up_files([file_paths[0]])

    data_set_joined.set_name("joined_set")

    for index, path in enumerate(file_paths):
        if index > 0:
            try:
                data_set = InteractiveDataset.load_dataset(path)
                data_set_joined.data = torch.vstack([data_set_joined.data, data_set.data])
                data_set_joined.labels = torch.vstack([data_set_joined.labels, data_set.labels])
                print(f"merged {os.path.basename(path)} into {data_set_joined.get_name()}")
            except:
                print(f"merge failed on subset {index}")

    #data_set_joined.labels = data_set_joined.labels.permute(0, 4, 1, 2, 3)
    data_set_joined.labels = data_set_joined.labels.unsqueeze(1)
    data_set_joined.save_dataset(f"../../data/datasets/ABC/{joined_name}.torch")

    load_test = InteractiveDataset.load_dataset(f"../../data/datasets/ABC/{joined_name}.torch")

    # clean_up_files(file_paths)

    print(load_test.get_info())


def main():
    # create_ABC_AE_sub_Dataset()
    join_ABC_sub_Datasets()

if __name__ == "__main__":
    main()
