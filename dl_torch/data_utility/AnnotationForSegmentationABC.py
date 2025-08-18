import numpy as np
import os
import torch
from typing import List
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from dl_torch.data_utility import DataParsing
from utility.data_exchange import cppIOexcavator
from visualization import color_templates
from pathlib import Path
import gc

def __sub_Dataset_from_target_dir_default(target_dir: str, class_list, class_lot, index_lot):

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

        # face based annotations
        for face_index, face_center in enumerate(face_to_index_map):

            if origin[0] <= face_center[0] <= top[0] and origin[1] <= face_center[1] <= top[1] and origin[2] <= \
                    face_center[2] <= \
                    top[2]:
                grid_coord = face_center - origin

                type_string = face_type_map[face_index]
                one_hot_index = class_lot[type_string]
                label[int(grid_coord[0]), int(grid_coord[1]), int(grid_coord[2]), one_hot_index] += 1
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

def __sub_Dataset_from_target_dir_inside_outside(target_dir: str, class_list, class_lot, index_lot):

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

        # face based annotations
        for face_index, face_center in enumerate(face_to_index_map):

            if origin[0] <= face_center[0] <= top[0] and origin[1] <= face_center[1] <= top[1] and origin[2] <= \
                    face_center[2] <= \
                    top[2]:
                grid_coord = face_center - origin

                type_string = face_type_map[face_index]
                one_hot_index = class_lot[type_string]
                label[int(grid_coord[0]), int(grid_coord[1]), int(grid_coord[2]), one_hot_index] += 1
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
                if grid[i,j,k] < 0:
                  label[i, j, k, class_lot["Inside"]] = 1
                  df_voxel_count['Inside'] += 1
                elif grid[i,j,k] > 0:
                    label[i, j, k, class_lot["Outside"]] = 1
                    df_voxel_count['Outside'] += 1
                else:
                    print("Error: Unassign able voxel found")
                    continue

        # print(f"Writer Counter part {path} grid {grid_index}")
        # print(df_voxel_count.keys())
        # print(df_voxel_count.values())

        labels.append(label)

    data = torch.tensor(np.array(bin_arrays))
    labels = torch.tensor(np.array(labels))

    return data, labels

def __sub_Dataset_from_target_dir_edge(target_dir: str, class_list, class_lot, index_lot):

    bin_array_file = target_dir + "/segmentation_data_segments.bin"
    segment_info_file = target_dir + "/segmentation_data.dat"

    segment_data = cppIOexcavator.parse_dat_file(segment_info_file)

    origins = segment_data["ORIGIN_CONTAINER"]["data"]
    face_type_map = np.array(list(segment_data["FACE_TYPE_MAP"].values()))
    vert_type_map = list(segment_data["VERT_TYPE_MAP"].values())
    face_to_index_map = segment_data["FACE_TO_GRID_INDEX_CONTAINER"]["data"]
    vert_to_index_map = segment_data["VERT_TO_GRID_INDEX_CONTAINER"]["data"]
    uniques = segment_data['TYPE_COUNT_MAP']

    # get edge index of edge vertices
    edge_vertex_indices = []
    for v_index, vertex in enumerate(vert_type_map):
        entries = vertex.split(',')
        if len(entries) > 1:
            edge_vertex_indices.append(v_index)


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

        # face based annotations
        for face_index, face_center in enumerate(face_to_index_map):

            if origin[0] <= face_center[0] <= top[0] and origin[1] <= face_center[1] <= top[1] and origin[2] <= \
                    face_center[2] <= \
                    top[2]:
                grid_coord = face_center - origin

                type_string = face_type_map[face_index]
                one_hot_index = class_lot[type_string]
                label[int(grid_coord[0]), int(grid_coord[1]), int(grid_coord[2]), one_hot_index] += 1
                write_count += 1

        #edge classification
        for vertex in edge_vertex_indices:
            vertex_on_grid = vert_to_index_map[vertex]
            if origin[0] <= vertex_on_grid[0] <= top[0] and origin[1] <= vertex_on_grid[1] <= top[1] and origin[2] <= \
                    vertex_on_grid[2] <= \
                    top[2]:
                grid_coord = vertex_on_grid - origin
                type_string = "Edge"
                one_hot_index = class_lot[type_string]
                label[int(grid_coord[0]), int(grid_coord[1]), int(grid_coord[2]), one_hot_index] += 1
                write_count += 1


        # inside - outside classification
        for i, j, k in np.ndindex(label.shape[0], label.shape[1], label.shape[2]):
            voxel = label[i, j, k, :]

            if np.sum(voxel) > 0:
                # Enforce edge class
                if voxel[class_lot["Edge"]] > 0:
                    max_index = class_lot["Edge"]
                else:
                    max_index = np.argmax(voxel)

                label[i, j, k, :] = np.zeros_like(voxel)
                label[i, j, k, max_index] = 1
                df_voxel_count[index_lot[max_index]] += 1
            else:
                if grid[i,j,k] < 0:
                  label[i, j, k, class_lot["Inside"]] = 1
                  df_voxel_count['Inside'] += 1
                elif grid[i,j,k] > 0:
                    label[i, j, k, class_lot["Outside"]] = 1
                    df_voxel_count['Outside'] += 1
                else:
                    print("Error: Unassign able voxel found")
                    continue

        labels.append(label)

    data = torch.tensor(np.array(bin_arrays))
    labels = torch.tensor(np.array(labels))

    return data, labels

def __sub_Dataset_from_target_dir_edge_only(target_dir: str, class_list, class_lot, index_lot):

    bin_array_file = target_dir + "/segmentation_data_segments.bin"
    segment_info_file = target_dir + "/segmentation_data.dat"

    segment_data = cppIOexcavator.parse_dat_file(segment_info_file)

    origins = segment_data["ORIGIN_CONTAINER"]["data"]
    face_type_map = np.array(list(segment_data["FACE_TYPE_MAP"].values()))
    vert_type_map = list(segment_data["VERT_TYPE_MAP"].values())
    face_to_index_map = segment_data["FACE_TO_GRID_INDEX_CONTAINER"]["data"]
    vert_to_index_map = segment_data["VERT_TO_GRID_INDEX_CONTAINER"]["data"]
    uniques = segment_data['TYPE_COUNT_MAP']

    # get edge index of edge vertices
    edge_vertex_indices = []
    for v_index, vertex in enumerate(vert_type_map):
        entries = vertex.split(',')
        if len(entries) > 1:
            edge_vertex_indices.append(v_index)


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

        # face based annotations
        for face_index, face_center in enumerate(face_to_index_map):

            if origin[0] <= face_center[0] <= top[0] and origin[1] <= face_center[1] <= top[1] and origin[2] <= \
                    face_center[2] <= \
                    top[2]:
                grid_coord = face_center - origin

                type_string = "Surface"
                one_hot_index = class_lot[type_string]
                label[int(grid_coord[0]), int(grid_coord[1]), int(grid_coord[2]), one_hot_index] += 1
                write_count += 1

        #edge classification
        for vertex in edge_vertex_indices:
            vertex_on_grid = vert_to_index_map[vertex]
            if origin[0] <= vertex_on_grid[0] <= top[0] and origin[1] <= vertex_on_grid[1] <= top[1] and origin[2] <= \
                    vertex_on_grid[2] <= \
                    top[2]:
                grid_coord = vertex_on_grid - origin
                type_string = "Edge"
                one_hot_index = class_lot[type_string]
                label[int(grid_coord[0]), int(grid_coord[1]), int(grid_coord[2]), one_hot_index] += 1
                write_count += 1


        # inside - outside classification
        for i, j, k in np.ndindex(label.shape[0], label.shape[1], label.shape[2]):
            voxel = label[i, j, k, :]

            if np.sum(voxel) > 0:
                # Enforce edge class
                if voxel[class_lot["Edge"]] > 0:
                    max_index = class_lot["Edge"]
                else:
                    max_index = np.argmax(voxel)

                label[i, j, k, :] = np.zeros_like(voxel)
                label[i, j, k, max_index] = 1
                df_voxel_count[index_lot[max_index]] += 1
            else:
                if grid[i,j,k] < 0:
                  label[i, j, k, class_lot["Inside"]] = 1
                  df_voxel_count['Inside'] += 1
                elif grid[i,j,k] > 0:
                    label[i, j, k, class_lot["Outside"]] = 1
                    df_voxel_count['Outside'] += 1
                else:
                    print("Error: Unassign able voxel found")
                    continue

        labels.append(label)

    data = torch.tensor(np.array(bin_arrays))
    labels = torch.tensor(np.array(labels))

    return data, labels

def create_ABC_sub_Dataset(segment_dir : str, torch_dir : str, n_min_files :  int, template: str = "default"):

    # Set up dictionary
    color_template = None

    template_list = {
        "default"         : 0,
        "edge"            : 1,
        "inside_outside"  : 2,
        "edge_only"       : 3
    }



    match template_list[template]:
        case 0 : color_template = color_templates.default_color_template_abc()
        case 1 : color_template = color_templates.edge_color_template_abc()
        case 2 : color_template = color_templates.inside_outside_color_template_abc()
        case 3 : color_template = color_templates.edge_only_color_template_abc()


    if template is not None:
        class_keys = list(color_template.keys())

        class_list = np.array(class_keys)
        class_list = np.sort(class_list)
        class_indices = np.arange(len(class_list))
        class_lot = dict(zip(class_list, class_indices))
        index_lot = dict(zip(class_indices, class_list, ))

        segment_paths = os.listdir(segment_dir)

        if not (os.path.exists(torch_dir)):
            os.makedirs(torch_dir)

        for path in segment_paths:

            full_path = os.path.join(segment_dir, path)

            if len(os.listdir(full_path)) >= n_min_files:

                data, labels = None, None

                match template_list[template]:

                    case 0: data, labels = __sub_Dataset_from_target_dir_default(full_path, class_list, class_lot, index_lot)
                    case 1: data, labels = __sub_Dataset_from_target_dir_edge(full_path, class_list, class_lot, index_lot)
                    case 2: data, labels = __sub_Dataset_from_target_dir_inside_outside(full_path, class_list, class_lot, index_lot)
                    case 3: data, labels = __sub_Dataset_from_target_dir_edge_only(full_path, class_list, class_lot, index_lot)

                if data is not None and labels is not None:

                    sub_dataset = InteractiveDataset(data, labels, class_lot, set_name=path)
                    sub_dataset.save_dataset(os.path.join(torch_dir, path + ".torch"))

                else:
                    print("Error: Data Annotation failed...")

    else:
        print('Error: Invalid template_mode selection for subset creation')


def create_ABC_sub_Dataset_from_job(job_file: str, segment_dir : str, torch_dir : str, n_min_files :  int, template: str = "default"):

    job_targets = DataParsing.read_job_file(job_file)

    # Set up dictionary
    color_template = None

    template_list = {
        "default": 0,
        "edge": 1,
        "inside_outside": 2,
        "edge_only": 3
    }

    match template_list[template]:
        case 0:
            color_template = color_templates.default_color_template_abc()
        case 1:
            color_template = color_templates.edge_color_template_abc()
        case 2:
            color_template = color_templates.inside_outside_color_template_abc()
        case 3:
            color_template = color_templates.edge_only_color_template_abc()

    if template is not None:

        class_keys = list(color_template.keys())

        class_list = np.array(class_keys)
        class_list = np.sort(class_list)
        class_indices = np.arange(len(class_list))
        class_lot = dict(zip(class_list, class_indices))
        index_lot = dict(zip(class_indices, class_list, ))

        segment_paths = os.listdir(segment_dir)

        if not(os.path.exists(torch_dir)):
            os.makedirs(torch_dir)

        for target in job_targets:

            full_path = os.path.join(segment_dir, target)

            if len(os.listdir(full_path)) >= n_min_files:

                data, labels = None, None

                match template_list[template]:

                    case 0:
                        data, labels = __sub_Dataset_from_target_dir_default(full_path, class_list, class_lot,
                                                                             index_lot)
                    case 1:
                        data, labels = __sub_Dataset_from_target_dir_edge(full_path, class_list, class_lot, index_lot)
                    case 2:
                        data, labels = __sub_Dataset_from_target_dir_inside_outside(full_path, class_list, class_lot,
                                                                                    index_lot)
                    case 3:
                        data, labels = __sub_Dataset_from_target_dir_edge_only(full_path, class_list, class_lot,
                                                                                    index_lot)

                if data is not None and labels is not None:

                    labels = labels.permute(0, 4, 1, 2, 3)

                    sub_dataset = InteractiveDataset(data, labels, class_lot, set_name=target)

                    sub_dataset.save_dataset(os.path.join(torch_dir, target + ".torch"))

                else:
                    
                    print("Error: Data Annotation failed...")


def batch_ABC_sub_Datasets(source_dir: str, target_dir: str, dataset_name: str, batch_count: int):
    src = Path(source_dir)
    out = Path(target_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect .torch files deterministically
    file_paths: List[Path] = sorted(src.rglob("*.torch"))
    n_files = len(file_paths)

    if n_files == 0:
        raise RuntimeError(f"No .torch subsets found in {source_dir}")

    # If batch_count is larger than available files, cap it
    batch_count = max(1, min(batch_count, n_files))

    # Split into ~equal batches without going OOB
    # sizes like: [ceil(n/b), ... first (n % b) times, then floor(n/b) ...]
    base = n_files // batch_count
    rem = n_files % batch_count
    sizes = [base + 1] * rem + [base] * (batch_count - rem)

    # Compute batch index ranges
    starts = []
    s = 0
    for sz in sizes:
        starts.append((s, s + sz))
        s += sz

    for batch_idx, (start, end) in enumerate(starts):
        group = file_paths[start:end]  # never OOB, may be size 1

        # Load the first dataset in the group
        first_fp = group[0]
        data_set_joined = InteractiveDataset.load_dataset(str(first_fp))

        # Merge the rest
        for fp in group[1:]:
            try:
                ds = InteractiveDataset.load_dataset(str(fp))
                # Assumes same shapes; otherwise add shape checks here.
                data_set_joined.data = torch.vstack([data_set_joined.data, ds.data])
                data_set_joined.labels = torch.vstack([data_set_joined.labels, ds.labels])
                print(f"merged {os.path.basename(str(fp))} into {data_set_joined.get_name()}")

                # Free memory
                del ds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                # Don't index into file_paths with i; use the actual fp
                print(f"merge failed on file '{fp}': {e}")
                raise

        # Save the batch
        batch_name = f"{dataset_name}_batch_{batch_idx}"
        save_path = out / f"{batch_name}.torch"
        data_set_joined.save_dataset(str(save_path))

        # Optional info
        try:
            print(data_set_joined.get_info())
        except Exception:
            pass

        # Cleanup
        del data_set_joined
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    segment_dir=r"H:\ABC\ABC_Benchmark\Outputs_Benchmark"
    torch_dir=r"H:\ABC\ABC_Benchmark\torch_benchmark\edge_only"
    job_file = r"H:\ABC\ABC_Benchmark\torch_job\Instance001.job"
    create_ABC_sub_Dataset_from_job(job_file, segment_dir, torch_dir, 2, "edge_only" )

if __name__ == "__main__":
    main()
