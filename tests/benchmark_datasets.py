import os
from datetime import datetime
import logging

import time
import torch
from torch.utils.data import DataLoader

from dl_torch.data_utility.InteractiveDatasetManager import InteractiveDatasetManager
from dl_torch.data_utility.HDF5Dataset import HDF5Dataset

# Set up logging
def setup_logging(log_file: str = "dataset_loading.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()  # Also prints to console
        ]
    )

def load_ManagedDataset(managed_directory: str, global_split: float, global_batch_size: int, set_name: str = "default"):
    dataset: InteractiveDatasetManager = InteractiveDatasetManager(managed_directory, global_split, global_batch_size, set_name)
    dataset.set_split(global_split)

    logging.info(f"Target: {managed_directory}")
    logging.info(f"Loading {dataset.get_subset_count()} Managed Datasets")

    t_start = datetime.now()
    logging.info(f"[ENTER] Loading")

    train_size, val_size = dataset.get_train_test_size()
    logging.info(f"Training samples {train_size}, Validation samples {val_size}")

    t_end = datetime.now()
    logging.info(f"[EXIT] Loading")
    logging.info(f"Time delta = {t_end - t_start}")

    del dataset
def run_epoch(loader):
    for batch in loader:
        x, y = batch
        # dummy operation to simulate training step
        _ = x + 1


def func1_run_managed_epoch(managed_dir, batch_size, split):
    print("[MANAGED DATASET] Starting full pass through all subsets...")
    dataset_manager = InteractiveDatasetManager(managed_dir, split, batch_size)
    subset_count = dataset_manager.get_subset_count()

    start = time.time()
    for i in range(subset_count):
        dataset_manager.activate_subset_by_index(i)
        train_loader = dataset_manager.get_active_train_loader()
        run_epoch(train_loader)
    end = time.time()

    print(f"[MANAGED DATASET] Total epoch across {subset_count} subsets took {end - start:.2f} seconds")


def func2_run_hdf5_epoch(hdf5_path, batch_size):
    print("[HDF5 DATASET] Starting...")
    dataset = HDF5Dataset(hdf5_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    start = time.time()
    run_epoch(loader)
    end = time.time()

    print(f"[HDF5 DATASET] One epoch took {end - start:.2f} seconds")



def main():
    managed_dir = r"H:\ABC\ABC_torch\ABC_training\train_1f0_mio_ks_16_pad_4_bw_5_vs_adaptive_n3\batch_iter_01"
    hdf5_path = r"H:\ABC\ABC_torch\ABC_training\train_1f0_mio_ks_16_pad_4_bw_5_vs_adaptive_n3\dataset.hdf5"
    batch_size = 32
    split = 0.9

    print("=" * 50)
    func1_run_managed_epoch(managed_dir, batch_size, split)
    print("=" * 50)
    func2_run_hdf5_epoch(hdf5_path, batch_size)
    print("=" * 50)


if __name__ == "__main__":
    main()