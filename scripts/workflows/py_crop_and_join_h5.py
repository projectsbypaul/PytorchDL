import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gzip
import logging
import os
import shutil
from dl_torch.data_utility.HDF5Dataset import HDF5Dataset
from dl_torch.data_utility import hdf5_utility
from pathlib import Path
from torch import split
import numpy as np
import pickle
from utility.logging import config_logger
import csv

def __safe_remove_file(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        # fine, nothing to remove
        pass
    except Exception as e:
        # avoid logging here if logger is suspect
        print(f"[WARN] Failed to remove file {path}: {e}")

def __safe_rmtree(path: str):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[WARN] Failed to remove directory {path}: {e}")


def __find_gzipped_datasets(target_dir: str, search_str: str) -> list[str]:
    target_dir = Path(target_dir)
    matches = list(target_dir.rglob(f"*{search_str}*"))
    return matches

def __setup_workspace(work_space: str) -> list [str]:
    """
    Set up a workspace with subdirectories for intermediate processing.

    This function creates (if not already existing) three subdirectories inside
    the given `work_space` path:

    1. **temp_unpacked** (`unpack_dir`)
       - Location where unpacked `.h5` files (from `.h5.gz`) are stored.

    2. **temp_cropped** (`cropped_dir`)
       - Location where cropped/intermediate processed files are saved.

    4. **stats_bin** (`stats_dir`)
       - Location where statistics or binary analysis outputs are stored.

    Parameters
    ----------
    work_space : str
        The base directory in which to create the workspace subdirectories.

    Returns
    -------
    list[str]
        A list of three paths:
        `[unpack_dir, cropped_dir, stats_dir]`
    """
    if not os.path.exists(work_space): os.mkdir(work_space)

    unpack_dir = os.path.join(work_space, f"temp_unpacked")
    if not os.path.exists(unpack_dir): os.mkdir(unpack_dir)

    cropped_dir = os.path.join(work_space, f"temp_cropped")
    if not os.path.exists(cropped_dir): os.mkdir(cropped_dir)

    stats_dir = os.path.join(work_space, f"stats_bin")
    if not os.path.exists(stats_dir): os.mkdir(stats_dir)

    return [unpack_dir, cropped_dir, stats_dir]


def __crop_gzipped_dataset(gzipped_dataset, unpack_dir, stats_dir, cropped_dir, n_samples, template, ignore_index):

    dataset_name = os.path.basename(gzipped_dataset)
    dataset_name = dataset_name.split('.')[0]

    h5_unpacked = os.path.join(unpack_dir, f"{dataset_name}.h5")
    h5_stat_file = os.path.join(stats_dir, f"{dataset_name}_stats.bin")
    h5_cropped = os.path.join(cropped_dir, f"{dataset_name}_crp{n_samples}.h5")

    logging.info(f"Unpacking {dataset_name} -> {h5_unpacked}")
    with gzip.open(gzipped_dataset, "rb") as f_in:
        with open(h5_unpacked, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    logging.info(f"Unpacking {dataset_name} done")

    logging.info(f"Screening {dataset_name}")
    hdf5_utility.screen_hdf_dataset(h5_unpacked, h5_stat_file, template)
    logging.info(f"Screening {dataset_name} done")

    logging.info(f"Cropping {dataset_name}")
    hdf5_utility.crop_hdf_by_class(h5_unpacked, h5_stat_file, h5_cropped, n_samples, ignore_index)
    logging.info(f"Cropping {dataset_name} done")

    os.remove(h5_unpacked)

    return h5_cropped

def process_block(n_samples, template, ignore_index, search_str, h5_out_name, target_dir):

    workspace = rf"H:\ws_{h5_out_name}"
    log_file = rf"{workspace}/{h5_out_name}.log"
    h5_out = os.path.join(workspace, f"{h5_out_name}.h5")
    stat_bin = os.path.join(workspace, f"{h5_out_name}_stats.bin")

    if not os.path.exists(workspace): os.mkdir(workspace)
    config_logger.init_log(log_file, capture_print=False)

    gzipped_dataset = __find_gzipped_datasets(target_dir, search_str)
    logging.info(f"Found {gzipped_dataset.__len__()} entries containing '{search_str}'")

    unpack_dir, cropped_dir, stats_dir = __setup_workspace(workspace)

    cropped_h5_paths = []

    for dataset in gzipped_dataset:
        logging.info(f"Start cropping by class of {dataset} to {n_samples} samples per class)")
        logging.info(f"Using template '{template}', ignoring classes {ignore_index}")

        h5_cropped_pth = __crop_gzipped_dataset(
            dataset, unpack_dir, stats_dir, cropped_dir, n_samples, template, ignore_index)

        cropped_h5_paths.append(h5_cropped_pth)

        logging.info(f"Cropping of {dataset} to {n_samples} samples per class)")
        HDF5Dataset.print_file_info(h5_cropped_pth)

    logging.info(f"Joining {cropped_h5_paths.__len__()} subsets into {h5_out}")
    HDF5Dataset.join_hdf5_files(cropped_h5_paths, h5_out)
    logging.info(f"Subset joined successfully")

    HDF5Dataset.print_file_info(h5_out)
    h5_out_gz = f"{h5_out}.gz"

    logging.info(f"Compressing {h5_out} -> {h5_out_gz} ")
    with open(h5_out, "rb") as f_in:
        with gzip.open(h5_out_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    logging.info(f"Compression done")

    logging.info(f"Computing class distribution")
    hdf5_utility.screen_hdf_dataset(h5_out, stat_bin, template=template)
    hdf5_utility.get_class_distribution(stat_bin, class_template=template)
    logging.info(f"Computing class distribution done")

    __safe_remove_file(h5_out)
    __safe_rmtree(unpack_dir)
    __safe_rmtree(cropped_dir)

def main():

    n_samples = 20000
    template = "primitive_edge"
    ignore_index = [6, 7]
    search_str = "h5.gz"
    h5_out_name = f"abc_ks16_rot_primitive_edge_1f0_crp{n_samples}"
    target_dir = r"H:\ws_abc_labelling\abc_ks16_rot_primitive_edge_1f0_labels"

    process_block(n_samples, template, ignore_index, search_str, h5_out_name, target_dir)

    n_samples = 20000
    template = "primitive_edge"
    ignore_index = [6, 7]
    search_str = "h5.gz"
    h5_out_name = f"abc_ks16_rot_primitive_edge_3f9_crp{n_samples}"
    target_dir = r"H:\ws_abc_labelling\abc_ks16_rot_primitive_edge_3f9_labels"

    process_block(n_samples, template, ignore_index, search_str, h5_out_name, target_dir)


if __name__ == "__main__":
    main()