import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
import pandas as pd
import os
from  dl_torch.data_utility.HelperFunctionsABC import __get_ABC_bin_array_from_segment_dir, __get_highest_count_class
from utility.data_exchange import cppIO
from dl_torch.data_utility import DataParsing

def __balance_dataset():

    target_class_count = 5000

    save_name = f"ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2_balanced_n_{target_class_count}"

    np.random.seed = 420

    statistic_loc = r"C:\Local_Data\ABC\ABC_statistics\balance_parquets\ABC_chunk_01\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2.parquet"

    save_loc = r"C:\Local_Data\ABC\ABC_statistics"

    class_list = np.array(
        ['BSpline', 'Cone', 'Cylinder', 'Extrusion', 'Other', 'Plane', 'Revolution', 'Sphere', 'Torus', 'Void'])

    loaded_df = pd.read_parquet(statistic_loc, engine='pyarrow')

    # Extract the column as a Pandas Series
    extracted_column = loaded_df["ID"]
    print(f"\nExtracted column '{"ID"}':")
    print(extracted_column)

    # Get the unique elements of the Series
    unique_elements = extracted_column.unique()
    print(f"\nUnique elements in column '{"ID"}':")
    print(len(unique_elements))

    print(f"Loaded Dataframe with {loaded_df.shape[0]} entries")

    main_class_df = []

    for i in range(class_list.shape[0]):
        # Create a boolean condition using .isin()
        condition_multiple = loaded_df['MainClass'].isin([class_list[i]])

        # Apply the condition using .loc
        df_filtered_multiple = loaded_df.loc[condition_multiple]

        main_class_df.append(df_filtered_multiple)

    sampled_dfs = []

    for index, class_df in enumerate(main_class_df):

       if class_df.shape[0] > target_class_count:
           random_integers_array = np.random.randint(0, class_df.shape[0] - 1, size=target_class_count)
           sampled_class_df = class_df.iloc[random_integers_array]
           sampled_dfs.append(sampled_class_df)
       else:
           sampled_dfs.append(class_df)

    stacked_df = pd.concat(sampled_dfs)


    print(f"Created Dataframe with {stacked_df.shape[0]} entries")
    print(pd.Series(stacked_df["MainClass"]).value_counts())

    stacked_df.to_parquet(os.path.join(save_loc, save_name + ".parquet"), engine='pyarrow', compression='snappy')

    '''

    stacked_df = pd.concat(sampled_dfs)
    print(f"dataset with {stacked_df.shape[0]} samples")
    specific_column_sums = stacked_df[class_list].sum()
    print(f"\nSum of columns {class_list}:")
    print(specific_column_sums)

    # Extract the column as a Pandas Series
    extracted_column = stacked_df["ID"]
    print(f"\nExtracted column '{"ID"}':")
    print(extracted_column)

    # Get the unique elements of the Series
    unique_elements = extracted_column.unique()
    print(f"\nUnique elements in column '{"ID"}':")
    print(len(unique_elements))
    
    '''



    print()

def __data_class_contribution():

    segment_dir = r"C:\Local_Data\ABC\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2"
    source_dir = r"C:\Local_Data\ABC\ABC_parsed_files"
    save_dir = r"C:\Local_Data\ABC\ABC_statistics"

    ignored_files = ["origins.bin", "VertToGridIndex.bin", "VertTypeMap.bin", "TypeCounts.bin", "FaceTypeMap.bin", "FaceToGridIndex.bin"]
    n_min_files = 5

    # Set up dictionary

    class_list = np.array(['BSpline','Cone','Cylinder','Extrusion','Other','Plane','Revolution','Sphere','Torus','Void'])
    class_list = np.sort(class_list)
    class_indices = np.arange(len(class_list))
    class_lot = dict(zip(class_list, class_indices))
    index_lot = dict(zip(class_indices, class_list, ))

    segment_paths = os.listdir(segment_dir)

    df_statistics = pd.DataFrame(columns=["ID","Segment", 'MainClass','BSpline','Cone','Cylinder','Extrusion','Other','Plane','Revolution','Sphere','Torus','Void'])

    for p_index, path in enumerate(segment_paths):

        full_path = os.path.join(segment_dir, path)

        if len(os.listdir(full_path)) > n_min_files:

            origins = cppIO.read_float_matrix(full_path + "/origins.bin")
            face_type_map = cppIO.read_type_map_from_binary(full_path + "/FaceTypeMap.bin")
            face_to_index_map = cppIO.read_float_matrix(full_path + "/FaceToGridIndex.bin")

            bin_arrays = get_ABC_bin_arry_from_segment_dir(full_path, ignored_files)

            print(f"Analyzing {len(bin_arrays)} segments in directory {path} ...")

            obj_path = os.path.join(source_dir, path ,path + ".obj")

            _ , faces = DataParsing.parse_obj(obj_path)

            labels = []

            for g_index, grid in enumerate(bin_arrays):

                df_voxel_count = dict()

                for index, surf_type in enumerate(class_list):
                    df_voxel_count.update({str(surf_type): 0})

                grid_dim = grid.shape[0]

                origin = np.asarray(origins[g_index])

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

                main_class = __get_highest_count_class(df_voxel_count)

                new_df_entry = {"ID":path, "Segment":g_index, "MainClass": main_class , **df_voxel_count}

                # print(new_df_entry)

                # Determine the next available index label
                # If your DataFrame uses default integer indexing and is not empty,
                # new_index_label = df_statistics.index.max() + 1
                # If it's empty, the first label can be 0.
                # Or you can use any unique custom label.
                if df_statistics.empty:
                    new_index_label = 0
                else:
                    new_index_label = df_statistics.index.max() + 1

                df_statistics.loc[new_index_label] = new_df_entry

            print(f"Wrote data to dataframe...processing {path} done")

    # pip install pyarrow pandas
    df_statistics.to_parquet(os.path.join(save_dir,os.path.basename(segment_dir) + ".parquet"), engine='pyarrow', compression='snappy')

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
    # __balance_dataset()
    __histogramm_segmentation_samples()

if __name__ == "__main__":
    main()