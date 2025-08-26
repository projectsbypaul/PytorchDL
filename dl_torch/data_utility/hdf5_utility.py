import os
import argparse
from typing import Optional
from dl_torch.data_utility.HDF5Dataset import HDF5Dataset

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

def main():
    pass

if __name__ == "__main__":
    main()