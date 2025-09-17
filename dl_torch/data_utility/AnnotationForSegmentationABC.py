import numpy as np
import os
import torch
from typing import List

from markdown.extensions.extra import extensions
from numpy.ma.core import masked_inside

from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from dl_torch.data_utility import DataParsing
from utility.data_exchange import cppIOexcavator
from visualization import color_templates
from pathlib import Path
import gc
import zipfile
import shutil
from utility.job_utility import job_creation


# -----------------------------
# Helpers
# -----------------------------
def _normalize_for_ce_save(data: torch.Tensor, labels: torch.Tensor):
    """
    Normalize shapes/dtypes for CrossEntropyLoss training:
      - data:  [N, D, H, W]     -> [N, 1, D, H, W] (float32)
      - labels:[N, D, H, W, C]  -> [N, D, H, W]    (long), via argmax on last dim
      - labels:[N, C, D, H, W]  -> [N, D, H, W]    (long), via argmax on dim=1
    """
    # Data: ensure float32 and channel-first with 1 channel
    if data is None or labels is None:
        return data, labels

    if data.ndim == 4:
        data = data.unsqueeze(1)  # [N, 1, D, H, W]
    elif data.ndim == 5:
        # assume already [N, C, D, H, W]; keep as is
        pass
    else:
        raise RuntimeError(f"Unexpected data ndim={data.ndim}, shape={tuple(data.shape)}")

    data = data.float()

    # Labels: convert to indices [N, D, H, W]
    if labels.ndim == 5:
        # Could be channel-last [N, D, H, W, C] or channel-first [N, C, D, H, W]
        # Heuristic: compare spatial dims to data's spatial dims
        # data spatial dims:
        _, _, D, H, W = data.shape
        if labels.shape[1:4] == (D, H, W) and labels.shape[-1] > 1:
            # [N, D, H, W, C] -> argmax last
            labels = labels.argmax(dim=-1)
        elif labels.shape[2:5] == (D, H, W) and labels.shape[1] > 1:
            # [N, C, D, H, W] -> argmax dim=1
            labels = labels.argmax(dim=1)
        else:
            # If it's one-hot but doesn't match expected layout, fall back to last dim
            labels = labels.argmax(dim=-1)
    elif labels.ndim == 4:
        # already indices
        pass
    else:
        raise RuntimeError(f"Unexpected labels ndim={labels.ndim}, shape={tuple(labels.shape)}")

    labels = labels.long()
    return data, labels


# -----------------------------
# Subset builders
# -----------------------------
import numpy as np
import torch

def __sub_Dataset_for_test(target_dir: str, class_list, class_lot, index_lot):
    bin_array_file = target_dir + "/segmentation_data_segments.bin"
    segment_info_file = target_dir + "/segmentation_data.dat"

    segment_data = cppIOexcavator.parse_dat_file(segment_info_file)

    background = segment_data["SCALARS"]["background"]
    voxel_size = segment_data["SCALARS"]["voxel_size"]
    origins = segment_data["ORIGIN_CONTAINER"]["data"]                     # shape (num_grids, 3)
    face_type_map = np.array(list(segment_data["FACE_TYPE_MAP"].values()))  # (n_faces,) strings
    face_to_index_map = segment_data["FACE_TO_GRID_INDEX_CONTAINER"]["data"]  # (n_faces, 3)
    uniques = segment_data['TYPE_COUNT_MAP']

    bin_arrays = cppIOexcavator.load_segments_from_binary(bin_array_file)  # list of (D,D,D) arrays

    labels = []

    for grid_index, grid in enumerate(bin_arrays):
        num_classes = len(class_lot)
        grid_dim = grid.shape[0]

        # ---------------- Masks (disjoint) ----------------
        surface_threshold = voxel_size / background
        mask_inside = (grid < -surface_threshold)
        mask_outside = (grid >  surface_threshold)
        mask_narrowband = ~(mask_inside | mask_outside)  # same as abs(grid) <= surface_threshold

        # integer voxel coordinates for each region
        coords_inside      = np.argwhere(mask_inside)         # (Mi,3) int
        coords_outside     = np.argwhere(mask_outside)        # (Mo,3) int
        coords_narrowband  = np.argwhere(mask_narrowband)     # (Mn,3) int

        # ---------------- Label tensor ----------------
        label = np.zeros((grid_dim, grid_dim, grid_dim, num_classes), dtype=np.uint8)
        idx_inside  = class_lot["Inside"]
        idx_outside = class_lot["Outside"]

        # write Inside/Outside channels in one shot
        label[..., idx_inside]  = mask_inside.astype(np.uint8)
        label[..., idx_outside] = mask_outside.astype(np.uint8)

        # ---------------- Nearest-face mapping for narrowband ----------------
        # geometry coords (float) = voxel index + origin
        origin = origins[grid_index]                                  # (3,)
        nb_coords_float = coords_narrowband.astype(np.float32) + origin  # (Mn,3) float

        nb = torch.as_tensor(nb_coords_float, dtype=torch.float32)    # (Mn,3)
        fp = torch.as_tensor(face_to_index_map, dtype=torch.float32)  # (N,3)

        # NOTE: cdist is O(M*N) memory/time; switch to KD-tree if this gets big
        dists = torch.cdist(nb, fp)                                   # (Mn,N)
        _, nearest_face_indices = torch.min(dists, dim=1)             # (Mn,)

        # convert to NumPy before indexing a NumPy array of strings
        nearest_face_indices_np = nearest_face_indices.cpu().numpy()  # (Mn,)
        nearest_face_types = face_type_map[nearest_face_indices_np]   # (Mn,) np.str_ / str

        # map face types -> class indices, default unknown → Outside
        face_keys = nearest_face_types.astype(object).astype(str)
        class_idx_arr = np.fromiter(
            (class_lot.get(k, idx_outside) for k in face_keys),
            dtype=np.intp,
            count=face_keys.size
        )  # (Mn,)

        # vectorized write of narrowband classes
        ii, jj, kk = coords_narrowband[:, 0], coords_narrowband[:, 1], coords_narrowband[:, 2]
        label[ii, jj, kk, class_idx_arr] = 1

        # ---------------- Edge assignment (ignore Inside/Outside) ----------------
        X, Y, Z, C = label.shape
        idx_edge = class_lot["Edge"]

        # only consider non Inside/Outside classes for neighborhood logic
        class_mask = np.ones(C, dtype=bool)
        class_mask[[idx_inside, idx_outside]] = False
        lbl = label[..., class_mask].astype(bool)  # (X,Y,Z,C_eff)
        C_eff = lbl.shape[3]

        # pad for 3x3x3 neighborhood scan
        padded = np.pad(lbl, ((1,1), (1,1), (1,1), (0,0)),
                        mode="constant", constant_values=False)

        # count neighbors per class in 3x3x3 (EXCLUDING the center)
        neigh_counts = np.zeros_like(lbl, dtype=np.uint8)  # max 26 fits in uint8
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    xs = slice(1 + dx, 1 + dx + X)
                    ys = slice(1 + dy, 1 + dy + Y)
                    zs = slice(1 + dz, 1 + dz + Z)
                    neigh_counts += padded[xs, ys, zs, :]

        # center must ALREADY be a considered class (prevents outer shell)
        center_in_considered = lbl.any(axis=3)  # (X,Y,Z) bool

        # center's own class (within the reduced channel set)
        own_class = lbl.argmax(axis=3)          # (X,Y,Z) int
        flat_counts = neigh_counts.reshape(-1, C_eff)
        own_flat = own_class.ravel()
        same_class_neigh = flat_counts[np.arange(flat_counts.shape[0]), own_flat].reshape(X, Y, Z)

        # neighbors with different class (voxel count)
        total_neigh = neigh_counts.sum(axis=3)                      # (X,Y,Z)
        different_voxel_count = (total_neigh - same_class_neigh)    # (X,Y,Z)

        # distinct other classes present
        neigh_any = (neigh_counts > 0)
        distinct_total = neigh_any.sum(axis=3)                      # (X,Y,Z)
        own_present = neigh_any.reshape(-1, C_eff)[np.arange(flat_counts.shape[0]), own_flat] \
                         .reshape(X, Y, Z)                          # (X,Y,Z) bool
        distinct_classes_diff = (distinct_total - own_present.astype(np.int32))  # (X,Y,Z)

        # --- sensitivity: "min voxel count" means >= sensitivity ---
        sensitivity = 1  # tweak as needed

        edge_mask = center_in_considered & (distinct_classes_diff > 0) & (different_voxel_count >= sensitivity)

        # write Edge one-hot
        label[edge_mask, :] = 0
        label[edge_mask, idx_edge] = 1

        labels.append(label)

    # pack tensors
    data_np = np.stack([np.asarray(g, dtype=np.float32) for g in bin_arrays], axis=0)  # [N, D, D, D]
    labels_np = np.stack(labels, axis=0)                                               # [N, D, D, D, C]

    data_t = torch.from_numpy(data_np)     # float32
    labels_t = torch.from_numpy(labels_np) # uint8

    return data_t, labels_t


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

        labels.append(label)

    data = torch.tensor(np.array(bin_arrays))
    labels = torch.tensor(np.array(labels))

    return data, labels


def __sub_Dataset_from_target_dir_inside_outside(
    target_dir: str,
    class_lot: dict,           # {class_name -> class_index}
):
    bin_array_file = target_dir + "/segmentation_data_segments.bin"
    segment_info_file = target_dir + "/segmentation_data.dat"

    segment_data = cppIOexcavator.parse_dat_file(segment_info_file)


    background = segment_data["SCALARS"]["background"]
    voxel_size = segment_data["SCALARS"]["voxel_size"]
    origins = segment_data["ORIGIN_CONTAINER"]["data"]
    face_type_map = np.array(list(segment_data["FACE_TYPE_MAP"].values()))
    face_to_index_map = segment_data["FACE_TO_GRID_INDEX_CONTAINER"]["data"]
    uniques = segment_data['TYPE_COUNT_MAP']

    bin_arrays = cppIOexcavator.load_segments_from_binary(bin_array_file)

    labels = []

    for grid_index, grid in enumerate(bin_arrays):

        num_classes = len(class_lot)
        grid_dim = grid.shape[0]

        surface_threshold = voxel_size / background

        mask_inside = grid < -surface_threshold
        mask_outside = grid > surface_threshold
        mask_narrowband = (grid >= -surface_threshold) & (grid <= surface_threshold)

        grid_index_inside = np.argwhere(mask_inside)
        grid_index_outside = np.argwhere(mask_outside)
        grid_index_narrowband = np.argwhere(mask_narrowband)

        label = np.zeros((grid_dim, grid_dim, grid_dim, num_classes), dtype=np.uint8)
        idx_inside = class_lot["Inside"]
        idx_outside = class_lot["Outside"]

        #bool to integer conversion
        mask_inside_int = mask_inside.astype(np.uint8)
        mask_outside_int = mask_outside.astype(np.uint8)

        #labl[..., channel] -> set channel to mask
        label[..., idx_inside] = mask_inside_int
        label[..., idx_outside] = mask_outside_int

        #calculate closest face
        origin = origins[grid_index]  # (3,)
        grid_index_narrowband = grid_index_narrowband.astype(np.float32)  # (M,3)
        narrowband_coord = grid_index_narrowband + origin  # (M,3)

        nb = torch.as_tensor(narrowband_coord, dtype=torch.float32)  # (M,3)
        fp = torch.as_tensor(face_to_index_map, dtype=torch.float32)  # (N,3)

        d = torch.cdist(nb, fp)  # (M,N)
        nearest_dists, nearest_face_indices = torch.min(d, dim=1)  # (M,), (M,)

        #get face types
        nearest_face_types = face_type_map[nearest_face_indices]  # (M,) array of strings

        for idx, coord in enumerate(grid_index_narrowband):
            try:
                class_idx = int(class_lot[str(nearest_face_types[idx])])
                x, y, z = map(int, coord)  # guarantees integer indices
                label[x, y, z, class_idx] = 1
            except KeyError:
                face_type = nearest_face_types[idx]
                print(f"Surface Unknown {face_type}, assigning Outside")
                x, y, z = map(int, coord)  # guarantees integer indices
                label[x, y, z, idx_outside] = 1

        labels.append(label)

    data_np = np.stack([np.asarray(g, dtype=np.float32) for g in bin_arrays], axis=0)  # [N, D, D, D]
    labels_np = np.stack(labels, axis=0)  # [N, D, D, D, C]

    data_t = torch.from_numpy(data_np)  # float32
    labels_t = torch.from_numpy(labels_np)  # uint8

    return data_t, labels_t


def __sub_Dataset_from_target_dir_edge(target_dir: str, class_lot):
    bin_array_file = target_dir + "/segmentation_data_segments.bin"
    segment_info_file = target_dir + "/segmentation_data.dat"

    segment_data = cppIOexcavator.parse_dat_file(segment_info_file)

    background = segment_data["SCALARS"]["background"]
    voxel_size = segment_data["SCALARS"]["voxel_size"]
    origins = segment_data["ORIGIN_CONTAINER"]["data"]  # shape (num_grids, 3)
    face_type_map = np.array(list(segment_data["FACE_TYPE_MAP"].values()))  # (n_faces,) strings
    face_to_index_map = segment_data["FACE_TO_GRID_INDEX_CONTAINER"]["data"]  # (n_faces, 3)
    uniques = segment_data['TYPE_COUNT_MAP']

    bin_arrays = cppIOexcavator.load_segments_from_binary(bin_array_file)  # list of (D,D,D) arrays

    labels = []

    for grid_index, grid in enumerate(bin_arrays):
        num_classes = len(class_lot)
        grid_dim = grid.shape[0]

        # ---------------- Masks (disjoint) ----------------
        surface_threshold = voxel_size / background
        mask_inside = (grid < -surface_threshold)
        mask_outside = (grid > surface_threshold)
        mask_narrowband = ~(mask_inside | mask_outside)  # same as abs(grid) <= surface_threshold

        # integer voxel coordinates for each region
        coords_inside = np.argwhere(mask_inside)  # (Mi,3) int
        coords_outside = np.argwhere(mask_outside)  # (Mo,3) int
        coords_narrowband = np.argwhere(mask_narrowband)  # (Mn,3) int

        # ---------------- Label tensor ----------------
        label = np.zeros((grid_dim, grid_dim, grid_dim, num_classes), dtype=np.uint8)
        idx_inside = class_lot["Inside"]
        idx_outside = class_lot["Outside"]

        # write Inside/Outside channels in one shot
        label[..., idx_inside] = mask_inside.astype(np.uint8)
        label[..., idx_outside] = mask_outside.astype(np.uint8)

        # ---------------- Nearest-face mapping for narrowband ----------------
        # geometry coords (float) = voxel index + origin
        origin = origins[grid_index]  # (3,)
        nb_coords_float = coords_narrowband.astype(np.float32) + origin  # (Mn,3) float

        nb = torch.as_tensor(nb_coords_float, dtype=torch.float32)  # (Mn,3)
        fp = torch.as_tensor(face_to_index_map, dtype=torch.float32)  # (N,3)

        # NOTE: cdist is O(M*N) memory/time; switch to KD-tree if this gets big
        dists = torch.cdist(nb, fp)  # (Mn,N)
        _, nearest_face_indices = torch.min(dists, dim=1)  # (Mn,)

        # convert to NumPy before indexing a NumPy array of strings
        nearest_face_indices_np = nearest_face_indices.cpu().numpy()  # (Mn,)
        nearest_face_types = face_type_map[nearest_face_indices_np]  # (Mn,) np.str_ / str

        # map face types -> class indices, default unknown → Outside
        face_keys = nearest_face_types.astype(object).astype(str)
        class_idx_arr = np.fromiter(
            (class_lot.get(k, idx_outside) for k in face_keys),
            dtype=np.intp,
            count=face_keys.size
        )  # (Mn,)

        # vectorized write of narrowband classes
        ii, jj, kk = coords_narrowband[:, 0], coords_narrowband[:, 1], coords_narrowband[:, 2]
        label[ii, jj, kk, class_idx_arr] = 1

        # ---------------- Edge assignment (ignore Inside/Outside) ----------------
        X, Y, Z, C = label.shape
        idx_edge = class_lot["Edge"]

        # only consider non Inside/Outside classes for neighborhood logic
        class_mask = np.ones(C, dtype=bool)
        class_mask[[idx_inside, idx_outside]] = False
        lbl = label[..., class_mask].astype(bool)  # (X,Y,Z,C_eff)
        C_eff = lbl.shape[3]

        # pad for 3x3x3 neighborhood scan
        padded = np.pad(lbl, ((1, 1), (1, 1), (1, 1), (0, 0)),
                        mode="constant", constant_values=False)

        # count neighbors per class in 3x3x3 (EXCLUDING the center)
        neigh_counts = np.zeros_like(lbl, dtype=np.uint8)  # max 26 fits in uint8
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    xs = slice(1 + dx, 1 + dx + X)
                    ys = slice(1 + dy, 1 + dy + Y)
                    zs = slice(1 + dz, 1 + dz + Z)
                    neigh_counts += padded[xs, ys, zs, :]

        # center must ALREADY be a considered class (prevents outer shell)
        center_in_considered = lbl.any(axis=3)  # (X,Y,Z) bool

        # center's own class (within the reduced channel set)
        own_class = lbl.argmax(axis=3)  # (X,Y,Z) int
        flat_counts = neigh_counts.reshape(-1, C_eff)
        own_flat = own_class.ravel()
        same_class_neigh = flat_counts[np.arange(flat_counts.shape[0]), own_flat].reshape(X, Y, Z)

        # neighbors with different class (voxel count)
        total_neigh = neigh_counts.sum(axis=3)  # (X,Y,Z)
        different_voxel_count = (total_neigh - same_class_neigh)  # (X,Y,Z)

        # distinct other classes present
        neigh_any = (neigh_counts > 0)
        distinct_total = neigh_any.sum(axis=3)  # (X,Y,Z)
        own_present = neigh_any.reshape(-1, C_eff)[np.arange(flat_counts.shape[0]), own_flat] \
            .reshape(X, Y, Z)  # (X,Y,Z) bool
        distinct_classes_diff = (distinct_total - own_present.astype(np.int32))  # (X,Y,Z)

        # --- sensitivity: "min voxel count" means >= sensitivity ---
        sensitivity = 1  # tweak as needed

        edge_mask = center_in_considered & (distinct_classes_diff > 0) & (different_voxel_count >= sensitivity)

        # write Edge one-hot
        label[edge_mask, :] = 0
        label[edge_mask, idx_edge] = 1

        labels.append(label)

    # pack tensors
    data_np = np.stack([np.asarray(g, dtype=np.float32) for g in bin_arrays], axis=0)  # [N, D, D, D]
    labels_np = np.stack(labels, axis=0)  # [N, D, D, D, C]

    data_t = torch.from_numpy(data_np)  # float32
    labels_t = torch.from_numpy(labels_np)  # uint8

    return data_t, labels_t




# -----------------------------
# Dataset creators
# -----------------------------
def create_ABC_sub_Dataset(segment_dir : str, torch_dir : str, n_min_files :  int, template: str = "default"):

    # Set up dictionary
    color_template = None

    template_list = {
        "default"         : 0,
        "edge"            : 1,
        "inside_outside"  : 2
    }

    match template_list[template]:
        case 0 : color_template = color_templates.default_color_template_abc()
        case 1 : color_template = color_templates.edge_color_template_abc()
        case 2 : color_template = color_templates.inside_outside_color_template_abc()

    if template is not None:
        class_keys = list(color_template.keys())

        class_list = np.array(class_keys)
        # class_list = np.sort(class_list)
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
                    case 0:
                        data, labels = __sub_Dataset_from_target_dir_default(full_path, class_list, class_lot, index_lot)
                    case 1:
                        data, labels = __sub_Dataset_from_target_dir_edge(full_path, class_lot)
                    case 2:
                        data, labels = __sub_Dataset_from_target_dir_inside_outside(full_path, class_lot)


                if data is not None and labels is not None:
                    # Normalize for CE (indices for labels, channel for data)
                    data, labels = _normalize_for_ce_save(data, labels)

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
        # class_list = np.sort(class_list)
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
                        data, labels = __sub_Dataset_from_target_dir_default(full_path, class_list, class_lot, index_lot)
                    case 1:
                        data, labels = __sub_Dataset_from_target_dir_edge(full_path, class_list, class_lot, index_lot)
                    case 2:
                        data, labels = __sub_Dataset_from_target_dir_inside_outside(full_path, class_lot)

                if data is not None and labels is not None:
                    # !! removed incorrect permutation of labels
                    # labels = labels.permute(0, 4, 1, 2, 3)

                    # Normalize for CE (indices for labels, channel for data)
                    data, labels = _normalize_for_ce_save(data, labels)

                    sub_dataset = InteractiveDataset(data, labels, class_lot, set_name=target)
                    sub_dataset.save_dataset(os.path.join(torch_dir, target + ".torch"))

                else:
                    print("Error: Data Annotation failed...")


# -----------------------------
# Batching / Zips
# -----------------------------
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
    source = r"W:\label_debug\target"
    torch_dir = r"W:\label_debug\subsets\edge"
    create_ABC_sub_Dataset(source, torch_dir, 2 , "edge")

if __name__ == "__main__":
    main()
