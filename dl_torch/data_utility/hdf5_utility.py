import os
import argparse
import pickle
from typing import Optional

import torch
from pyarrow import int64
from sympy.series.limitseq import dominant
from concurrent.futures import ProcessPoolExecutor, as_completed

from dl_torch.data_utility.HDF5Dataset import HDF5Dataset
from visualization import color_templates
import numpy as np
import pandas as pd
import random

def _suffix_path(src_path: str, n_samples: int) -> str:
    base, ext = os.path.splitext(src_path)
    return f"{base}_{n_samples}{ext or '.h5'}"

def crop_hdf_dataset(
    src_path: str,
    n_samples: int,
    *,
    crop_mode: str = "crop_start",
    out_path: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> str:
    """
    Create a cropped copy of an existing HDF5 dataset (no compression).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    # Print summary
    print("Original dataset info")
    HDF5Dataset.print_file_info(src_path)

    # Cropped view
    ds = HDF5Dataset(
        src_path,
        fixed_length=n_samples,
        crop_mode=crop_mode,
        random_seed=random_seed,
    )

    # Output file name
    out_path = out_path or _suffix_path(src_path, n_samples)

    # Export (no compression)
    ds.export_subset(
        out_path,
        compression=None,
        compression_opts=None,
        save_source_indices=True,
        copy_root_attrs=True,
    )
    ds.close()

    print(f"[OK] wrote cropped dataset -> {out_path}")

    print("Cropped dataset info")
    HDF5Dataset.print_file_info(out_path)

    return out_path

def crop_hdf_by_class(src_path: str, result_loc: str, out_path: str, n_samples: int, ignore_index, rng_seed = None):

    uniques_classes, index_container = __get_indicis_by_class(result_loc, ignore_index)
    sampled_index_container = __sample_indicis_per_class(index_container, n_samples, rng_seed)
    sampled_index_container = __flatten_index_container(sampled_index_container)

    hdf_dataset = HDF5Dataset(src_path)
    hdf_dataset.set_active_selection(sampled_index_container)
    hdf_dataset.export_active_selection(out_path)


def screen_hdf_dataset(src_path: str, result_loc: str,  template: str = "inside_outside"):
    # Print summary
    print("Screening dataset:")
    HDF5Dataset.print_file_info(src_path)

    ds = HDF5Dataset(src_path)

    ds_len = ds.__len__()

    if template=="inside_outside":
        class_temp = color_templates.inside_outside_color_template_abc()
    elif template =="edge":
        class_temp = color_templates.edge_color_template_abc()
    else:
        raise NotImplementedError(f"Template '{template}' is not implemented.")

    class_list = color_templates.get_class_list(class_temp)

    class_to_index = color_templates.get_class_to_index_dict(class_temp)

    count_collection = np.zeros((ds_len, len(class_list)), dtype=np.int32)

    for i in range(ds_len):

        ds_item = ds.__getitem__(i)

        labels = ds_item[1]


        counter = np.zeros(len(class_list))

        for item in class_list:
            index = int(class_to_index[item])
            count = np.count_nonzero(labels == index)
            counter[index] += count

        count_collection[i] = counter
        print(f"Evaluated class count of item: {i}/{ds_len}")

    with open(result_loc, "wb") as f:
        pickle.dump(count_collection,f)

def __flatten_index_container(index_container):

    n_entries = 0
    glob_index = 0

    for element in index_container:
        n_entries += len(element)

    flat_index_container = np.ndarray(shape=n_entries)


    for entry in index_container:
        for element in entry:
            flat_index_container[glob_index] = element
            glob_index+= 1

    if n_entries != glob_index:
        raise ValueError(
            f"Mismatch: not all flat containers entries received a values "
        )

    return  flat_index_container


def __sample_indicis_per_class(index_container, n_sample : int, rng_seed = None):

    random.seed(rng_seed)

    sampled_index_container = []
    for entry in index_container:
        if len(entry) > n_sample:
            sampled_entry = random.sample(entry, n_sample)
            sampled_index_container.append(sampled_entry)
        else:
            sampled_index_container.append(entry)

    return sampled_index_container

def __get_indicis_by_class(result_loc, ignore_index):

    # ---- load ----
    with open(result_loc, "rb") as f:
        count_collection = pickle.load(f)
    if not isinstance(count_collection, np.ndarray):
        count_collection = np.asarray(count_collection)

    cc_ignores = count_collection.copy()

    cc_max_index = []

    for index in ignore_index:
      cc_ignores[:, index] = 0

    for i in range(count_collection.shape[0]):
        row_ignored = cc_ignores[i]
        row_raw = count_collection[i]

        if np.sum(row_ignored) > 0:
            argmax = np.argmax(row_ignored)
        else:
            argmax = np.argmax(row_raw)

        cc_max_index.append(argmax)

    unique_classes = np.unique(cc_max_index)

    index_container = []

    for element in unique_classes:
        indices_per_element = [index for index, entry in enumerate(cc_max_index) if entry == element]
        index_container.append(indices_per_element)

    return unique_classes, index_container

def get_class_distribution(result_loc,
                     ignore_names=("Inside", "Outside"),
                     use_tiebreak_noise=False,
                     mark_empty_as_unknown=True):
    """
    - Loads count_collection from `result_loc` (pickle of [N, C] counts).
    - Prints per-class voxel totals.
    - Picks dominant primitive per row, ignoring `ignore_names`.
    - Prints distribution (class names + counts).

    Args:
      result_loc: path to pickle file with numpy array [N, C]
      ignore_names: class names to exclude from argmax (e.g., "Inside", "Outside")
      use_tiebreak_noise: add tiny random noise to break ties deterministically
      mark_empty_as_unknown: mark rows with all-zero/NaN/Inf as -1 ("Unknown")
    """

    # ---- load ----
    with open(result_loc, "rb") as f:
        count_collection = pickle.load(f)
    if not isinstance(count_collection, np.ndarray):
        count_collection = np.asarray(count_collection)

    # ---- classes & mappings ----
    class_temp = color_templates.inside_outside_color_template_abc()
    class_list = color_templates.get_class_list(class_temp)
    class_to_index = color_templates.get_class_to_index_dict(class_temp)
    index_to_class = color_templates.get_index_to_class_dict(class_temp)

    # sanity: dims vs classes
    assert count_collection.shape[1] == len(class_list), (
        f"Columns ({count_collection.shape[1]}) != classes ({len(class_list)})"
    )

    # ---- totals ----
    class_count = np.sum(count_collection, axis=0)
    print("voxels per class:")
    for i, name in enumerate(class_list):
        print(f"{name:10s}: {class_count[i]:,}")

    # ---- choose primitives only (ignore Inside/Outside) ----
    # Primitive names = all classes except ignore_names
    primitive_names = [n for n in class_list if n not in ignore_names]
    primitive_idxs = torch.tensor([class_to_index[n] for n in primitive_names], dtype=torch.long)

    # to torch
    C = torch.from_numpy(count_collection).float()
    P = C[:, primitive_idxs]  # [N, P] primitives only

    # diagnostics
    rows_nan = torch.isnan(P).any(dim=1)
    rows_inf = torch.isinf(P).any(dim=1)
    all_equal = (P == P[:, :1]).all(dim=1)  # every value equal within a row
    all_zero = (P == 0).all(dim=1)

    print("\nDiagnostics (primitives slice):")
    print(f"rows with NaN: {rows_nan.sum().item():,}")
    print(f"rows with ±Inf: {rows_inf.sum().item():,}")
    print(f"rows all-equal: {all_equal.sum().item():,}")
    print(f"rows all-zero: {all_zero.sum().item():,}")

    # mask for primitives all-zero
    # Identify primitive-only slice
    prims_all_zero = (P == 0).all(dim=1)

    # Grab Inside / Outside counts
    inside_idx = class_to_index["Inside"]
    outside_idx = class_to_index["Outside"]

    inside_counts = C[:, inside_idx]
    outside_counts = C[:, outside_idx]

    # Split
    inside_only = prims_all_zero & (inside_counts > 0) & (outside_counts == 0)
    outside_only = prims_all_zero & (outside_counts > 0) & (inside_counts == 0)
    both_io = prims_all_zero & (inside_counts > 0) & (outside_counts > 0)
    truly_empty = prims_all_zero & (inside_counts == 0) & (outside_counts == 0)

    print("\nBreakdown of rows where all primitives are zero:")
    print(f"Inside only : {inside_only.sum().item():,}")
    print(f"Outside only: {outside_only.sum().item():,}")
    print(f"Both I+O    : {both_io.sum().item():,}")
    print(f"Truly empty : {truly_empty.sum().item():,}")

    # tie-breaking noise (doesn't change clear maxima, only ties)
    if use_tiebreak_noise:
        eps = 1e-6 * torch.rand_like(P)
        P = P + eps

    # argmax over primitives (local indices)
    pred_local = torch.argmax(P, dim=1)  # 0..len(primitive_idxs)-1
    pred_global = primitive_idxs[pred_local].clone()  # map back to global class indices

    # mark "unknown" for degenerate rows if requested
    if mark_empty_as_unknown:
        unknown_mask = rows_nan | rows_inf | all_zero
        # also optionally consider rows where all-equal and value==0 as unknown:
        all_equal_and_zero = all_equal & all_zero
        unknown_mask = unknown_mask | all_equal_and_zero
        pred_global[unknown_mask] = -1  # sentinel

    # ---- count & print ----
    uniq, cnts = torch.unique(pred_global, return_counts=True)
    uniq = uniq.cpu().tolist()
    cnts = cnts.cpu().tolist()

    print("\nClass distribution after argmax (ignoring Inside/Outside):")
    # sort by global index but keep -1 at the end
    ordered = sorted(zip(uniq, cnts), key=lambda x: (x[0] == -1, x[0]))
    for gid, n in ordered:
        if gid == -1:
            print(f"Unknown/Empty/Ignored: {n:,}")
        else:
            print(f"{index_to_class[gid]}: {n:,}")

    # optional: return predictions if you need them downstream
    return pred_global.numpy()


def main():

    ds_path = r"H:\ws_abc_labelling\export\ABC_chunk_01_ks32swo4nbw8nk3_20250929-101945\ABC_chunk_01_ks32swo4nbw8nk3_20250929-101945_results.h5"
    result_loc =r"H:\ws_abc_labelling\export\ABC_chunk_01_ks32swo4nbw8nk3_20250929-101945\result.bin"

    HDF5Dataset.print_file_info(ds_path)

    screen_hdf_dataset(ds_path, result_loc)
    get_class_distribution(result_loc)


if __name__ == "__main__":
    main()