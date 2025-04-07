import numpy as np

from utility.data_exchange import cppIO
from dl_torch.data_utility import DataParsing
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv
import os
import re

def vote_voxels():
    folder_path = r"C:\Local_Data\cropping_test"

    f_names = []

    # Pattern to extract index number from filenames like "cropped_17.bin"
    pattern = re.compile(r"cropped_(\d+)\.bin$")

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

    obj_loc = (r"C:\Local_Data\ABC\obj\abc_meta_files"
               r"\abc_0000_obj_v00\00000002\00000002_1ffb81a71e5b402e966b9341_trimesh_001.obj")

    vertices, _ = DataParsing.parse_obj(obj_loc)

    for index, v_types in enumerate(vertex_type_map):
        if len(v_types) > 1:
            vertex_type_map[index] = ["Edge"]

    flat = [item for sublist in vertex_type_map for item in sublist]

    flat.append("Void")

    class_list = np.unique(flat)

    class_indices = np.arange(len(class_list))

    class_lot = dict(zip(class_list, class_indices))

    segment_index = 15

    _, _, sdf = cppIO.read_3d_array_from_binary(sorted_paths[segment_index])

    grid_dim = sdf.shape[0]

    origin = np.asarray(origins[segment_index])

    top = origin + [grid_dim -1, grid_dim -1, grid_dim-1]

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
            print(f"label[{x}, {y}, {z}] = {class_vector}")
            max_value = class_vector.max()
            indices = np.where(class_vector == max_value)[0]
            if len(indices) == 1:
                label[x,y,z, indices[0]] = 1
            if len(indices) > 1:
                label[x, y, z, : ] = np.zeros(shape=len(class_list))
                label[x, y, z, class_lot["Edge"]] = 1
        else:
            label[x, y, z, class_lot["Void"]] = 1

    print()









def main():
    vote_voxels()

if __name__ == "__main__":
    main()