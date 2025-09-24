import os
import argparse
from typing import Optional

from pyarrow import int64

from dl_torch.data_utility.HDF5Dataset import HDF5Dataset
from visualization import color_templates
import numpy as np
import pandas as pd

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

def screen_hdf_dataset(src_path: str, template: str = "default"):
    # Print summary
    print("Screening dataset:")
    HDF5Dataset.print_file_info(src_path)

    ds = HDF5Dataset(src_path)

    ds_len = ds.__len__()

    class_temp = color_templates.inside_outside_color_template_abc()

    class_list = color_templates.get_class_list(class_temp)
    index_to_class = color_templates.get_index_to_class_dict(class_temp)
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
        print(f"Evaluated class count of item: {i} of {ds_len}")


    h5_df = pd.DataFrame(columns=class_list, data=count_collection, dtype=np.int32)








def main():
    ds_path = r"W:\hpc_workloads\hpc_datasets\train_data\inside_outside_A_32_pd0_bw8_nk3_20250915-110203\inside_outside_A_32_pd0_bw8_nk3_20250915-110203_dataset.h5"
    screen_hdf_dataset(ds_path)

if __name__ == "__main__":
    main()