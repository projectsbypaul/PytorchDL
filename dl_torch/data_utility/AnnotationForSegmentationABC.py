import numpy as np
import os
import torch
from typing import List

from markdown.extensions.extra import extensions

from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from dl_torch.data_utility import DataParsing
from utility.data_exchange import cppIOexcavator
from visualization import color_templates
from pathlib import Path
import gc
import zipfile
import shutil
from utility.job_utility import job_creation

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

import numpy as np
import torch

def __sub_Dataset_from_target_dir_inside_outside(
    target_dir: str,
    class_list,                # list/array of class names, must include "Inside" and "Outside"
    class_lot: dict,           # {class_name -> class_index}
    index_lot: dict,           # {class_index -> class_name}, optional (kept for compatibility)
    epsilon: float = 0.0       # treat |grid| <= epsilon as 0; set small >0 to avoid "unassignable"
):
    """
    Vectorized rebuild of __sub_Dataset_from_target_dir_inside_outside.

    Returns:
        data   : torch.FloatTensor of shape [N, D, D, D] (same dtype as loaded grids, cast to float32)
        labels : torch.UInt8Tensor of shape [N, D, D, D, C] one-hot per voxel
    """
    bin_array_file   = f"{target_dir}/segmentation_data_segments.bin"
    segment_info_file= f"{target_dir}/segmentation_data.dat"

    segment_data = cppIOexcavator.parse_dat_file(segment_info_file)

    # Load metadata
    origins = np.asarray(segment_data["ORIGIN_CONTAINER"]["data"], dtype=np.int64)       # [N, 3]
    face_type_map = np.asarray(list(segment_data["FACE_TYPE_MAP"].values()))            # [F] (assumed aligned)
    face_to_index_map = np.asarray(segment_data["FACE_TO_GRID_INDEX_CONTAINER"]["data"], dtype=np.int64)  # [F, 3]

    # Load voxel grids
    bin_arrays = cppIOexcavator.load_segments_from_binary(bin_array_file)  # list of [D, D, D] arrays
    num_parts = len(bin_arrays)
    num_classes = len(class_list)

    labels_out = []
    for part_idx, grid in enumerate(bin_arrays):
        grid = np.asarray(grid)  # ensure ndarray
        D = grid.shape[0]
        origin = origins[part_idx]                         # [3]
        top = origin + (D - 1)                             # inclusive bounds

        # Face-based annotations: pick faces that land inside this grid’s cube
        fc = face_to_index_map                             # [F, 3] (global coords)
        inside_mask = (fc >= origin).all(axis=1) & (fc <= top).all(axis=1)
        if inside_mask.any():
            fc_local = fc[inside_mask] - origin            # [M, 3] local coords
            face_types = face_type_map[inside_mask]        # [M] strings
            # map types -> class indices
            face_cls_idx = np.fromiter((class_lot[t] for t in face_types), dtype=np.int64, count=face_types.shape[0])
        else:
            fc_local = np.empty((0,3), dtype=np.int64)
            face_cls_idx = np.empty((0,), dtype=np.int64)

        # Accumulate per-voxel class vote counts
        # counts shape: [D, D, D, C], uint16 should be enough for vote tallies
        counts = np.zeros((D, D, D, num_classes), dtype=np.uint16)
        if fc_local.size > 0:
            xi, yi, zi = fc_local[:,0], fc_local[:,1], fc_local[:,2]
            ci = face_cls_idx
            # Multiple faces may map to same voxel/class -> use add.at for safe accumulation
            np.add.at(counts, (xi, yi, zi, ci), 1)

        # Resolve face votes -> one-hot where present
        max_vals = counts.max(axis=-1)                     # [D, D, D]
        max_idx  = counts.argmax(axis=-1)                  # [D, D, D]
        has_face = max_vals > 0

        # Prepare final one-hot label volume (compact dtype)
        label = np.zeros((D, D, D, num_classes), dtype=np.uint8)

        if has_face.any():
            xi, yi, zi = np.where(has_face)
            label[xi, yi, zi, max_idx[has_face]] = 1

        # Fill remaining voxels using SDF sign
        unlabeled = ~has_face
        if epsilon > 0:
            neg_mask = (grid < -epsilon) & unlabeled
            pos_mask = (grid >  epsilon) & unlabeled
            zero_mask = (~neg_mask) & (~pos_mask) & unlabeled  # |grid| <= eps
        else:
            neg_mask = (grid < 0) & unlabeled
            pos_mask = (grid > 0) & unlabeled
            zero_mask = (grid == 0) & unlabeled

        # Require "Inside" and "Outside" in class_lot
        inside_idx  = class_lot["Inside"]
        outside_idx = class_lot["Outside"]
        if neg_mask.any():
            label[neg_mask, inside_idx] = 1
        if pos_mask.any():
            label[pos_mask, outside_idx] = 1

        # Handle true zeros deterministically:
        # Option A: push zeros to nearest sign (choose Outside by default)
        # Option B: if you have a dedicated "Surface" class, map zero_mask to that.
        if zero_mask.any():
            # Fallback: treat as Outside (change if you prefer Inside or a "Surface" class)
            label[zero_mask, outside_idx] = 1
            # Or, if "Surface" in class_lot:
            # surf_idx = class_lot.get("Surface", outside_idx)
            # label[zero_mask, surf_idx] = 1

        labels_out.append(label)

    # Stack & convert with minimal copies
    data_np   = np.stack([np.asarray(g, dtype=np.float32) for g in bin_arrays], axis=0)  # [N, D, D, D]
    labels_np = np.stack(labels_out, axis=0)                                             # [N, D, D, D, C]

    data_t   = torch.from_numpy(data_np)     # float32
    labels_t = torch.from_numpy(labels_np)   # uint8

    return data_t, labels_t


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
                    print(f"Error: Unassign able voxel found val: {grid[i, j, k]}")
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
                    print(f"Error: Unassign able voxel found val: {grid[i,j,k]}")
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


def compressed_segment_dir_to_dataset_from_job(zip_source_dir, job_file : str, workspace_dir :  str, template : str = "default", batch_count = 1):

    job_targets = DataParsing.read_job_file(job_file)
    job_targets = [os.path.join(zip_source_dir, entry) for entry in job_targets]

    compressed_segment_dir_to_dataset(job_targets, workspace_dir, template, batch_count)


def compressed_segment_dir_to_dataset(segment_dir_zip : [str], workspace_dir, template : str = "default", batch_count = 1):

    for seg_dir in segment_dir_zip:
        unpack_dir = os.path.join(workspace_dir, "unpacked_files")
        torch_dir = os.path.join(workspace_dir, "torch_files")
        shards_dir = os.path.join(workspace_dir, "shards")

        if not os.path.exists(torch_dir): os.mkdir(torch_dir)
        if not os.path.exists(shards_dir): os.mkdir(shards_dir)

        print(f"Starting extraction of {os.path.split(seg_dir)[1]}")

        try:
            with zipfile.ZipFile(seg_dir, 'r') as zf:
                zf.extractall(unpack_dir)
            print(f"{os.path.split(seg_dir)[1]} was unpacked to {unpack_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to extract {seg_dir}: {e}")
            continue  # skip this archive, move on

        # try subset creation
        try:
            create_ABC_sub_Dataset(unpack_dir, torch_dir, 2, template)
        except Exception as e:
            print(f"[ERROR] create_ABC_sub_Dataset failed for {seg_dir}: {e}")
            shutil.rmtree(unpack_dir, ignore_errors=True)
            shutil.rmtree(torch_dir, ignore_errors=True)
            continue  # skip batching, move to next archive

        # try batching
        try:
            _, zip_filename = os.path.split(seg_dir)
            zip_filename = zip_filename.split('.')[0]
            batch_ABC_sub_Datasets(torch_dir, shards_dir, zip_filename, batch_count)
        except Exception as e:
            print(f"[ERROR] batch_ABC_sub_Datasets failed for {seg_dir}: {e}")

        # cleanup (always)
        shutil.rmtree(torch_dir, ignore_errors=True)
        shutil.rmtree(unpack_dir, ignore_errors=True)


def main():

    job_file = r"W:\hpc_workloads\hpc_datasets\jobs_Block_A\Instance001.job"
    workspace = r"W:\hpc_workloads\hpc_datasets\Block_A\train_A_10000_16_pd0_bw12_vs2_20250825-084440\workspace"
    source = r"W:\hpc_workloads\hpc_datasets\Block_A\train_A_10000_16_pd0_bw12_vs2_20250825-084440\output_dir"



if __name__ == "__main__":
    main()
