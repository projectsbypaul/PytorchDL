import sys
from pathlib import Path

from setuptools.wheel import unpack

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


def __crop_gzipped_dataset_n_from_start(gzipped_dataset, h5_out_name,unpack_dir, stats_dir, cropped_dir, template ,n_samples):

    dataset_name = os.path.basename(gzipped_dataset)
    dataset_name = dataset_name.split('.')[0]

    h5_unpacked = os.path.join(unpack_dir, f"{dataset_name}.h5")
    h5_stat_file = os.path.join(stats_dir, f"{dataset_name}_stats.bin")
    h5_cropped = os.path.join(cropped_dir, f"{h5_out_name}.h5")
    h5_cropped_stat_file = os.path.join(cropped_dir, f"{h5_out_name}_stats.bin")

    logging.info(f"Unpacking {dataset_name} -> {h5_unpacked}")
    with gzip.open(gzipped_dataset, "rb") as f_in:
        with open(h5_unpacked, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    logging.info(f"Unpacking {dataset_name} done")

    logging.info(f"Screening {dataset_name}")
    hdf5_utility.screen_hdf_dataset(h5_unpacked, h5_stat_file, template)
    logging.info(f"Screening {dataset_name} done")

    crop_index = np.arange(start=0, stop=n_samples, step=1)

    unpacked_dataset = HDF5Dataset(h5_unpacked)
    unpacked_dataset.set_active_selection(crop_index)
    unpacked_dataset.export_active_selection(h5_cropped)

    logging.info(f"Screening {h5_out_name}")
    hdf5_utility.screen_hdf_dataset(h5_cropped, h5_cropped_stat_file, template)
    logging.info(f"Screening {h5_out_name} done")

    os.remove(h5_unpacked)

    return h5_cropped


def main():
    n_samples = 100000
    template = "inside_outside"
    search_str = "h5.gz"
    target_dir = r"H:\ws_design_2026\00_datagen\Block_A"
    result_dir = r"H:\ws_design_2026\01_labels\Block_A"
    outfile_pattern="{dataset_name}_start_{n}"
    log_file=os.path.join(target_dir, "dir_crop.log")
    with open(log_file, "w"): pass

    config_logger.init_log(log_file, capture_print=True)

    gzipped_dataset = __find_gzipped_datasets(target_dir, search_str)
    logging.info(f"Found {gzipped_dataset.__len__()} entries containing '{search_str}'")

    cropped_h5_paths = []

    for dataset in gzipped_dataset:

        logging.info(f"Start cropping from start of {dataset} to {n_samples} samples per class)")
        logging.info(f"Using template '{template}")
        parent_dir = os.path.dirname(dataset)
        parent_dir_name = os.path.basename(parent_dir)
        dataset_name = os.path.basename(dataset)
        dataset_name = dataset_name.split('.')[0]
        out_name = outfile_pattern.format(dataset_name=dataset_name, n=n_samples)
        workspace = os.path.join(result_dir, parent_dir_name)
        unpacked_dir = os.path.join(workspace, f"unpacked_{dataset_name}")
        if not os.path.exists(workspace): os.makedirs(workspace)
        if not os.path.exists(unpacked_dir): os.mkdir(unpacked_dir)


        h5_cropped_pth = __crop_gzipped_dataset_n_from_start(
            dataset, out_name, unpacked_dir, workspace, workspace, template, n_samples)

        cropped_h5_paths.append(h5_cropped_pth)

        logging.info(f"Cropping of {dataset} to {n_samples} samples per class)")
        HDF5Dataset.print_file_info(h5_cropped_pth)

        with open(h5_cropped_pth, "rb") as f_in:
            with gzip.open(f"{h5_cropped_pth}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        shutil.rmtree(unpacked_dir)
        os.remove(h5_cropped_pth)

if __name__ == "__main__":
    main()