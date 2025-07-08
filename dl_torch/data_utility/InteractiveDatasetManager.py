import os
import gc
from typing import Optional
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset

class InteractiveDatasetManager:
    def __init__(self, managed_directory: str, global_split: float, global_batch_size: int, set_name: str = "default"):
        self.__managed_directory = managed_directory
        self.__global_split = global_split
        self.__global_batch_size = global_batch_size
        self.__set_name = set_name

        self.__subset_list = sorted(os.listdir(managed_directory))
        self.__subset_count = len(self.__subset_list)

        self.__active_set: Optional[InteractiveDataset] = None
        self.__train_loader = None
        self.__test_loader = None

    def set_managed_dir(self, managed_dir: str):
        self.__managed_directory = managed_dir
        self.__subset_list = sorted(os.listdir(managed_dir))
        self.__subset_count = len(self.__subset_list)

    def set_split(self, split: float):
        self.__global_split = split

    def set_batch_size(self, batch_size: int):
        self.__global_batch_size = batch_size

    def get_managed_dir(self):
        return self.__managed_directory

    def get_subset_list(self):
        return self.__subset_list

    def get_split(self):
        return self.__global_split

    def get_batch_size(self):
        return self.__global_batch_size

    def get_subset_count(self):
        return self.__subset_count

    def get_active_train_loader(self):
        return self.__train_loader if self.__train_loader is not None else 0

    def get_active_test_loader(self):
        return self.__test_loader if self.__test_loader is not None else 0

    def get_train_test_size(self):
        glob_train_size = 0
        glob_test_size = 0
        for subset_index in range(self.__subset_count):
            self.activate_subset_by_index(subset_index)
            glob_train_size += len(self.__active_set.get_train_dataset())
            glob_test_size += len(self.__active_set.get_test_dataset())
        return glob_train_size, glob_test_size

    def get_active_subset_info(self):
        return self.__active_set.get_info()

    def activate_subset_by_index(self, subset_index: int):
        if 0 <= subset_index < self.__subset_count:
            if self.__active_set is not None:
                del self.__active_set
                del self.__train_loader
                del self.__test_loader
                gc.collect()

            subset_path = os.path.join(self.__managed_directory, self.__subset_list[subset_index])
            print(f"[INFO] Loading subset {subset_index}: {subset_path}")

            self.__active_set = InteractiveDataset.load_dataset(subset_path)
            self.__active_set.set_split(self.__global_split)
            self.__active_set.split_dataset()

            self.__train_loader = self.__active_set.get_train_loader(self.__global_batch_size)
            self.__test_loader = self.__active_set.get_test_loader(self.__global_batch_size)

def main():
    dataset_manager = InteractiveDatasetManager(r"H:\ABC\ABC_torch\ABC_training\train_1f0_mio_ks_16_pad_4_bw_5_vs_adaptive_n3\batch_iter_01", 0.9, 16)
    train_size, val_size = dataset_manager.get_train_test_size()

    print(f"Train on {dataset_manager.get_subset_count()} Managed Dataset samples")
    print(f"Training samples {train_size}, Validation samples {val_size} ")

if __name__ == "__main__":
    main()
