import numpy as np
import torch
from utility.data_exchange import cppIO
from dl_torch.data_utility import DataParsing
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv
import os
import re
from matplotlib import cm
from pyvista import LookupTable
from matplotlib.colors import Normalize
from dl_torch.data_utility.HelperFunctionsABC import evaluate_voxel_class_kernel
from dl_torch.data_utility.HelperFunctionsABC import get_ABC_bin_arry_from_segment_dir
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from pathlib import Path

def vote_voxels():

    folder_path = r"C:\Local_Data\ABC\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2\00000002"

    f_names = []

    # Pattern to extract index number from filenames like "cropped_17.bin"
    pattern = re.compile(r"00000002_(\d+)\.bin$")

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            full_path = os.path.abspath(os.path.join(folder_path, filename))
            f_names.append((index, full_path))

    # Sort by the numeric index
    f_names.sort(key=lambda x: x[0])

    sorted_paths = [path for _, path in f_names]

    origins = cppIO.read_float_matrix(folder_path + "/origins.bin")
    vertex_type_map = cppIO.read_type_map_from_binary(folder_path + "/VertTypeMap.bin")
    vertex_to_index_map = cppIO.read_float_matrix(folder_path + "/VertToGridIndex.bin")

    obj_loc = (r"C:\Local_Data\ABC\ABC_parsed_files\00000002\00000002.obj")

    vertices, _ = DataParsing.parse_obj(obj_loc)

    for index, v_types in enumerate(vertex_type_map):
        if len(v_types) > 1:
            vertex_type_map[index] = ["Edge"]

    flat = [item for sublist in vertex_type_map for item in sublist]

    flat.append("Void")

    # class_list = np.unique(flat)
    class_list = np.array(["Cone", "Cylinder", "Edge", "Plane", "Sphere", "Torus", "Void"])

    class_indices = np.arange(len(class_list))

    class_lot = dict(zip(class_list, class_indices))

    # Define RGB (0–255) and opacity (0.0–1.0) for all classes, including 'Void'
    custom_colors = {

        'Cone': (0, 0, 255),  # blue
        'Cylinder': (255, 0, 0),  # red
        'Edge': (255, 255, 0),  # yellow
        'Plane': (255, 192, 203),  # pink
        'Sphere': (128, 0, 0),  # dark red
        'Torus': (0, 255, 255),  # cyan
        'Void': (255, 255, 255),  # white
    }
    custom_opacity = {
        'Cone': 1.0,
        'Cylinder': 1.0,
        'Edge': 1.0,
        'Plane': 1.0,
        'Sphere': 1.0,
        'Torus': 1.0,
        'Void': 0.0,
    }

    grid_segments = []

    for segment_index, segment in enumerate(sorted_paths):

        _, _, sdf = cppIO.read_3d_array_from_binary(sorted_paths[segment_index])

        grid_dim = sdf.shape[0]

        grid_spacing = 15

        spacing = np.asarray([grid_spacing, grid_spacing, grid_spacing])

        origin = np.asarray(origins[segment_index])

        top = origin + [grid_dim - 1, grid_dim - 1, grid_dim - 1]

        label = np.zeros(shape=[grid_dim, grid_dim, grid_dim, class_list.shape[0]])

        write_count = 0

        for vert_index, vert in enumerate(vertex_to_index_map):
            if origin[0] <= vert[0] <= top[0] and origin[1] <= vert[1] <= top[1] and origin[2] <= vert[2] <= top[2]:
                grid_index = vert - origin
                try:
                    type_string = vertex_type_map[vert_index]
                    one_hot_index = class_lot[type_string[0]]
                    label[int(grid_index[0]), int(grid_index[1]), int(grid_index[2]), one_hot_index] += 1
                    write_count += 1
                except:
                    print(f"Vertex {vert_index} is not mappable")

        print(f"wrote {write_count} labels")

        for x, y, z in np.ndindex((grid_dim, grid_dim, grid_dim)):
            class_vector = label[x, y, z, :]
            if class_vector.sum() > 1:
                # print(f"label[{x}, {y}, {z}] = {class_vector}")
                max_value = class_vector.max()
                indices = np.where(class_vector == max_value)[0]
                if len(indices) == 1:
                    label[x, y, z, indices[0]] = 1
                if len(indices) > 1:
                    label[x, y, z, :] = np.zeros(shape=len(class_list))
                    label[x, y, z, indices[0]] = 1
                    # label[x, y, z, class_lot["Edge"]] = 1
            else:
                label[x, y, z, class_lot["Void"]] = 1

        voxel_kernel = 1
        class_weights = [1, 1, 1, 1, 1, 1, 1 / (voxel_kernel ** 3)]
        counter = np.zeros(shape=len(class_list))

        label_dense = label.copy()

        for x, y, z in np.ndindex((grid_dim, grid_dim, grid_dim)):
            class_vector = label[x, y, z, :]
            if class_vector.argmax() == class_lot["Void"]:
                max_index = evaluate_voxel_class_kernel(label, (x, y, z), voxel_kernel, class_weights)
                counter[max_index] += 1
                if max_index < 6:
                    new_vector = np.zeros(shape=len(class_list))
                    new_vector[max_index] = 1
                    label_dense[x, y, z, :] = new_vector

        print(f"processing of grid {segment_index} done")

        grid_segments.append(label_dense)

    # Create a PyVista plotter
    plotter = pv.Plotter()

    cubes = []

    for index, grid in enumerate(grid_segments):
        # Decode labels
        label_indices = np.argmax(grid, axis=-1)

        grid_dim = grid.shape[0]

        origin = origins[index]

        # Iterate through the grid
        for x in range(grid_dim):
            for y in range(grid_dim):
                for z in range(grid_dim):
                    class_idx = label_indices[x, y, z]
                    temp_label = class_list[class_idx]

                    # print(f"plotting {x} {y} {z}")

                    # Skip invisible (Void) cubes
                    if custom_opacity[temp_label] == 0.0:
                        continue

                    # Create a cube centered at the grid location
                    cube = pv.Cube(center=(x + origin[0], y + origin[1], z + origin[2]), x_length=1.0,
                                   y_length=1.0,
                                   z_length=1.0)

                    color = custom_colors[temp_label]

                    color_rgb = tuple(c / 255 for c in color)
                    alpha = custom_opacity[temp_label]

                    rgba = np.append(color_rgb, alpha)  # [R, G, B, A]

                    cube.cell_data["colors"] = np.tile(rgba, (cube.n_cells, 1))

                    cubes.append(cube)

        combined = pv.MultiBlock(cubes).combine()

        plotter.add_mesh(combined, scalars="colors", rgba=True, show_edges=False)

        print(f"drawing of gird {index} done")



    # ---- Create Custom Legend ----
    legend_entries = []
    for label in class_list:
        if custom_opacity[label] == 0.0:
            continue
        rgb = tuple(c / 255 for c in custom_colors[label])
        legend_entries.append([label, rgb])

    plotter.add_legend(legend_entries, bcolor='white', face='circle', size=(0.2, 0.25), loc='lower right')

    plotter.show()

def create_ABC_sub_Dataset():

    segment_dir = r"C:\Local_Data\ABC\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2"
    source_dir = r"C:\Local_Data\ABC\ABC_parsed_files"

    torch_dir = r"C:\Local_Data\ABC\ABC_torch\torch_data_ks_16_pad_4_bw_5_vs_adaptive_n2"


    ignored_files = ["origins.bin", "VertToGridIndex.bin", "VertTypeMap.bin"]

    # Set up dictionary
    class_list = np.array(["Cone", "Cylinder", "Edge", "Plane", "Sphere", "Torus", "Void"])
    class_indices = np.arange(len(class_list))
    class_lot = dict(zip(class_list, class_indices))

    segment_paths = os.listdir(segment_dir)

    for path in segment_paths:

        full_path = os.path.join(segment_dir, path)

        if len(os.listdir(full_path)) > 1:

            origins = cppIO.read_float_matrix(full_path + "/origins.bin")
            vertex_type_map = cppIO.read_type_map_from_binary(full_path + "/VertTypeMap.bin")
            vertex_to_index_map = cppIO.read_float_matrix(full_path + "/VertToGridIndex.bin")

            bin_arrays = get_ABC_bin_arry_from_segment_dir(full_path, ignored_files)

            obj_path = os.path.join(source_dir, path ,path + ".obj")

            vertices, _ = DataParsing.parse_obj(obj_path)

            for index, v_types in enumerate(vertex_type_map):
                if len(v_types) > 1:
                    vertex_type_map[index] = ["Edge"]

            labels = []

            for grid_index, grid in enumerate(bin_arrays):

                grid_dim = grid.shape[0]

                origin = np.asarray(origins[grid_index])

                top = origin + [grid_dim - 1, grid_dim - 1, grid_dim - 1]

                label = np.zeros(shape=[grid_dim, grid_dim, grid_dim, class_list.shape[0]])

                write_count = 0

                for vert_index, vert in enumerate(vertex_to_index_map):
                    if origin[0] <= vert[0] <= top[0] and origin[1] <= vert[1] <= top[1] and origin[2] <= vert[2] <= \
                            top[2]:
                        grid_index = vert - origin
                        try:
                            type_string = vertex_type_map[vert_index]
                            one_hot_index = class_lot[type_string[0]]
                            label[int(grid_index[0]), int(grid_index[1]), int(grid_index[2]), one_hot_index] += 1
                            write_count += 1
                        except:
                            #print(f"Vertex {vert_index} is not mappable")
                             pass

                # print(f"wrote {write_count} labels")

                labels.append(label)


            data = torch.tensor(np.array(bin_arrays))
            labels = torch.tensor(np.array(labels))
            sub_dataset = InteractiveDataset(data, labels, class_lot, set_name=path)



            sub_dataset.save_dataset(os.path.join(torch_dir, path + ".torch"))



def join_ABC_sub_Datasets():
    # Define the parent folder and file extension
    parent_folder = Path(r"C:\Local_Data\ABC\ABC_torch\torch_data_ks_16_pad_4_bw_5_vs_adaptive_n2") # Replace with your folder path
    joined_name = "ABC_Data_ks_32_pad_4_bw_5_vs_adaptive_n2"
    file_extension = ".torch"  # Change to your desired extension

    # Get all matching file paths
    file_paths = list(parent_folder.rglob(f"*{file_extension}"))

    data_set_joined = InteractiveDataset.load_dataset(file_paths[0])

    data_set_joined.set_name("joined_set")

    for index, path in enumerate(file_paths):
        if index > 0:
            data_set = InteractiveDataset.load_dataset(path)
            data_set_joined.data = torch.vstack([data_set_joined.data, data_set.data])
            data_set_joined.labels = torch.vstack([data_set_joined.labels, data_set.labels])

    data_set_joined.save_dataset(r"../data/datasets/ABC/ABC_Data_ks_32_pad_4_bw_5_vs_adaptive_n2.torch")

    load_test = InteractiveDataset.load_dataset(r"../data/datasets/ABC/ABC_Data_ks_32_pad_4_bw_5_vs_adaptive_n2.torch")

    print(load_test.get_info())





def main():
    # vote_voxels()
    # create_ABC_sub_Dataset()
    join_ABC_sub_Datasets()

if __name__ == "__main__":
    main()