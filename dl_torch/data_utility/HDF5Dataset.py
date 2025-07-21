import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.h5_file = None

        with h5py.File(self.hdf5_path, 'r') as f:
            self.length = f['features'].shape[0]

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
        x = self.h5_file['features'][idx]
        y = self.h5_file['labels'][idx]
        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return self.length

    @staticmethod
    def convert_pt_to_hdf5(pt_folder, hdf5_path):
        files = sorted([f for f in os.listdir(pt_folder) if f.endswith('.torch')])

        total_samples = 0
        feature_shape = None
        label_shape = None

        # First pass — get dataset shapes and total size
        for filename in files:
            print(f"[INFO] parsing {filename} to get shape information")
            path = os.path.join(pt_folder, filename)
            dataset: InteractiveDataset = InteractiveDataset.load_dataset(path)
            data = dataset.data
            labels = dataset.labels

            if feature_shape is None:
                feature_shape = data[0].shape
                label_shape = labels[0].shape

            total_samples += len(data)

        if feature_shape is None or label_shape is None:
            raise ValueError("No valid samples found.")

        # Second pass — write to HDF5
        with h5py.File(hdf5_path, 'w') as f:
            dset_x = f.create_dataset(
                'features',
                shape=(total_samples, *feature_shape),
                dtype='float32'
            )
            dset_y = f.create_dataset(
                'labels',
                shape=(total_samples, *label_shape),
                dtype='float32'
            )

            offset = 0
            for filename in files:
                print(f"[INFO] added {filename} to {os.path.basename(hdf5_path)}")
                path = os.path.join(pt_folder, filename)
                dataset: InteractiveDataset = InteractiveDataset.load_dataset(path)
                data = dataset.data
                labels = dataset.labels

                size = data.shape[0]
                dset_x[offset:offset + size] = data.numpy()
                dset_y[offset:offset + size] = labels.numpy()
                offset += size


def main():
    source = r"H:\ABC\ABC_torch\ABC_training\train_500k_ks_16_pad_4_bw_5_vs_adaptive_n3\batch_iter_01"
    target = r"H:\ABC\ABC_torch\ABC_training\\train_500k_ks_16_pad_4_bw_5_vs_adaptive_n3\dataset.hdf5"
    HDF5Dataset.convert_pt_to_hdf5(source, target)

if __name__ == "__main__":
    main()
