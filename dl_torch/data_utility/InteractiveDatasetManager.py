from typing import List, Tuple, Dict, Optional
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
import os
import gc

class InteractiveDatasetManager:
    def __init__(self, managed_directory : str , global_split : float, global_batch_size : int ,set_name:str = "default"):
        """
        Args:
            managed_directory (str): directory containing managed interactive datasets
            global_split (float): split applied to all subsets.
            global_batch_size (int): batch_size applied to all subsets.
        """
        self.__managed_directory = managed_directory
        self.__active_set: Optional[InteractiveDataset] = None  # <- This enables IDE tooltips

        self.__global_split = global_split
        self.__global_batch_size = global_batch_size
        self.__set_name = set_name

        self.__subset_list = os.listdir(managed_directory)
        self.__subset_count = len(self.__subset_list)

    def set_managed_dir(self, managed_dir : str):
        self.__managed_directory = managed_dir
        self.__subset_list = os.listdir(managed_dir)
        self.__subset_count = len(self.__subset_list)

    def set_split(self, split: float):
        self.__global_split = split

    def set_batch_size(self, batch_size: float):
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
        if self.__active_set is not None:
          return self.__active_set.get_train_loader(self.__global_batch_size)
        else:
            return 0

    def get_active_test_loader(self):
        if self.__active_set is not None:
          return self.__active_set.get_test_loader(self.__global_batch_size)
        else:
            return 0

    def get_train_test_size(self):
        glob_train_loader_size = 0
        glob_test_loader_size = 0

        for subset_index in range(self.__subset_count):
            self.activate_subset_by_index(subset_index)
            glob_train_loader_size += len(self.__active_set.get_train_loader(1))
            glob_test_loader_size += len(self.__active_set.get_test_loader(1))

        return glob_train_loader_size, glob_test_loader_size

    def get_active_subset_info(self):

        return self.__active_set.get_info()

    def activate_subset_by_index(self, subset_index):
        if 0 <= subset_index < self.__subset_count:

            if self.__active_set is not None:

                del self.__active_set
                gc.collect()  # Explicit garbage collection

            subset_path = os.path.join(self.__managed_directory, self.__subset_list[subset_index])

            self.__active_set = InteractiveDataset.load_dataset(subset_path)
            self.__active_set.set_split(self.__global_split)

def main() -> None:
    managed_dir = r"H:\ABC\ABC_torch\ABC_chunk_00\batched_data_ks_16_pad_4_bw_5_vs_adaptive_n2_testing"
    batch_size = 4
    split = 0.9
    dataset_manager = InteractiveDatasetManager(managed_dir, split, batch_size)
    for set_index in range(dataset_manager.get_subset_count()):
        dataset_manager.activate_subset_by_index(set_index)
        print(dataset_manager.get_active_subset_info())

if __name__ == "__main__":
    main()


