import os
import numpy as np
import h5py
import torch
from numpy.ma.core import shape
from torch.utils.data import Dataset
from typing import Optional, Tuple, Iterable
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset


class HDF5Dataset(Dataset):
    """
    HDF5-backed dataset with optional fixed-length cropping.

    Crop modes:
      - 'crop_start'  : keep the first N samples (default)
      - 'crop_end'    : keep the last N samples
      - 'crop_random' : keep N random samples (sorted order); set random_seed for determinism

    You can also crop during conversion from .torch shards to reduce the physical file size.

    Example (read a cropped view):
        ds = HDF5Dataset("dataset.hdf5", fixed_length=100_000, crop_mode="crop_start")

    Example (save that cropped view to a new file):
        ds.export_subset("dataset_100k_end.hdf5", compression="lzf")

    Example (convert shards with cropping to make a smaller HDF5 on disk):
        HDF5Dataset.convert_pt_to_hdf5(
            pt_folder="path/to/shards",
            hdf5_path="dataset_small.hdf5",
            fixed_length=200_000,
            crop_mode="crop_random",
            random_seed=42,
            compression="lzf",
        )
    """

    # ------------- lifecycle -------------

    def __init__(
        self,
        hdf5_path: str,
        fixed_length: Optional[int] = None,
        crop_mode: str = "crop_start",
        random_seed: Optional[int] = None,
    ):
        self.hdf5_path = hdf5_path
        self.h5_file: Optional[h5py.File] = None

        with h5py.File(self.hdf5_path, "r") as f:
            total = int(f["features"].shape[0])

        self.indices = self._choose_indices(total, fixed_length, crop_mode, random_seed)
        self.length = int(self.indices.size)
        self.active_selection = self.indices

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self.h5_file is None:
            # Lazy-open per worker to avoid shared handles
            self.h5_file = h5py.File(self.hdf5_path, "r")
        real_idx = int(self.indices[idx])
        x = self.h5_file["features"][real_idx]
        y = self.h5_file["labels"][real_idx]
        return torch.from_numpy(x), torch.from_numpy(y)

    def close(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            finally:
                self.h5_file = None

    def __del__(self):
        self.close()

    # ------------- public exports -------------

    def export_subset(
        self,
        out_path: str,
        *,
        compression: Optional[str] = None,      # e.g. "lzf" (fast) or "gzip"
        compression_opts: Optional[int] = None, # e.g. 1..9 for gzip level
        save_source_indices: bool = True,
        copy_root_attrs: bool = True,
    ):
        """
        Persist the currently selected indices (self.indices) into a new HDF5.
        Data is copied in contiguous runs for speed and low memory use.
        """
        sel = np.sort(self.indices.astype(np.int64))
        if sel.size == 0:
            raise ValueError("No samples selected; nothing to export.")

        with h5py.File(self.hdf5_path, "r") as src, h5py.File(out_path, "w") as dst:
            fx, fy = src["features"], src["labels"]
            N = int(sel.size)

            dset_x = dst.create_dataset(
                "features",
                shape=(N, *fx.shape[1:]),
                dtype=fx.dtype,
                compression=compression,
                compression_opts=compression_opts,
            )
            dset_y = dst.create_dataset(
                "labels",
                shape=(N, *fy.shape[1:]),
                dtype=fy.dtype,
                compression=compression,
                compression_opts=compression_opts,
            )

            if copy_root_attrs:
                for k, v in src.attrs.items():
                    dst.attrs[k] = v

            write_off = 0
            for s, e in self._runs_from_indices(sel):
                blk = e - s
                dset_x[write_off:write_off + blk] = fx[s:e]
                dset_y[write_off:write_off + blk] = fy[s:e]
                write_off += blk

            if save_source_indices:
                dst.create_dataset("source_indices", data=sel, dtype="int64")


    def export_active_selection(
        self,
        out_path: str,
        *,
        compression: Optional[str] = None,      # e.g. "lzf" (fast) or "gzip"
        compression_opts: Optional[int] = None, # e.g. 1..9 for gzip level
        save_source_indices: bool = True,
        copy_root_attrs: bool = True,
    ):
        """
        Exports the active selection (self.active) into a new HDF5.
        Data is copied in contiguous runs for speed and low memory use.
        """
        sel = np.sort(self.active_selection.astype(np.int64))
        if sel.size == 0:
            raise ValueError("No samples selected; nothing to export.")

        with h5py.File(self.hdf5_path, "r") as src, h5py.File(out_path, "w") as dst:
            fx, fy = src["features"], src["labels"]
            N = int(sel.size)

            dset_x = dst.create_dataset(
                "features",
                shape=(N, *fx.shape[1:]),
                dtype=fx.dtype,
                compression=compression,
                compression_opts=compression_opts,
            )
            dset_y = dst.create_dataset(
                "labels",
                shape=(N, *fy.shape[1:]),
                dtype=fy.dtype,
                compression=compression,
                compression_opts=compression_opts,
            )

            if copy_root_attrs:
                for k, v in src.attrs.items():
                    dst.attrs[k] = v

            write_off = 0
            for s, e in self._runs_from_indices(sel):
                blk = e - s
                dset_x[write_off:write_off + blk] = fx[s:e]
                dset_y[write_off:write_off + blk] = fy[s:e]
                write_off += blk

            if save_source_indices:
                dst.create_dataset("source_indices", data=sel, dtype="int64")

    def set_active_selection(self, selection: np.ndarray):
        # check if valid
        if not isinstance(selection, np.ndarray):
            raise TypeError(
                f"Invalid type {type(selection)}. Expected numpy.ndarray."
            )

        if selection.ndim != 1:
            raise ValueError(
                f"Invalid shape {selection.shape}. Expected 1D array (shape: (n,))."
            )

        max_index = self.__len__() - 1

        for element in selection:
            if element > max_index:
                raise ValueError(
                    f"Selection out if bounds. Element {element})."
                )

        self.active_selection = selection

    # ------------- conversion from shards -------------

    @staticmethod
    def convert_pt_to_hdf5(
        pt_folder: str,
        hdf5_path: str,
        *,
        fixed_length: Optional[int] = None,
        crop_mode: str = "crop_start",
        random_seed: Optional[int] = None,
        dtype_x: str = "float32",
        dtype_y: str = "float32",
        compression: Optional[str] = None,      # e.g. "lzf" or "gzip"
        compression_opts: Optional[int] = None, # gzip level 1..9
    ):
        """
        Convert *.torch shards (InteractiveDataset) into a single HDF5.
        Optionally crop to a fixed_length using crop_mode.

        Notes:
          - If fixed_length is None or >= total, keeps all samples.
          - crop_random uses random_seed for reproducibility.
          - Data is copied in contiguous blocks for speed.
        """
        files = sorted([f for f in os.listdir(pt_folder) if f.endswith(".torch")])
        if not files:
            raise ValueError(f"No .torch files found in: {pt_folder}")

        # Pass 1: scan sizes and shapes
        file_sizes: list[int] = []
        total_samples = 0
        feature_shape: Optional[Tuple[int, ...]] = None
        label_shape: Optional[Tuple[int, ...]] = None

        for filename in files:
            print(f"[INFO] scanning {filename} for shape/size")
            path = os.path.join(pt_folder, filename)
            dataset: InteractiveDataset = InteractiveDataset.load_dataset(path)
            data = dataset.data
            labels = dataset.labels

            if feature_shape is None:
                feature_shape = tuple(data[0].shape)
                label_shape = tuple(labels[0].shape)

            sz = int(data.shape[0])
            file_sizes.append(sz)
            total_samples += sz

        assert feature_shape is not None and label_shape is not None, "No valid samples found."

        # Decide which global indices to keep
        selected = HDF5Dataset._choose_indices(total_samples, fixed_length, crop_mode, random_seed)
        keep_count = int(selected.size)
        if keep_count == 0:
            raise ValueError("Requested fixed_length=0 or no samples selected after cropping.")

        print(f"[INFO] total={total_samples}, keeping={keep_count}, mode={HDF5Dataset._normalize_mode(crop_mode)}")

        # Compute per-file index ranges (global index space)
        file_starts = np.cumsum([0] + file_sizes[:-1]).astype(np.int64)
        file_ends = (file_starts + np.array(file_sizes, dtype=np.int64)).astype(np.int64)

        # Pass 2: write selected samples
        with h5py.File(hdf5_path, "w") as f:
            dset_x = f.create_dataset(
                "features",
                shape=(keep_count, *feature_shape),
                dtype=dtype_x,
                compression=compression,
                compression_opts=compression_opts,
            )
            dset_y = f.create_dataset(
                "labels",
                shape=(keep_count, *label_shape),
                dtype=dtype_y,
                compression=compression,
                compression_opts=compression_opts,
            )

            write_offset = 0
            sel_ptr = 0  # pointer into 'selected' (sorted by _choose_indices)

            for filename, fstart, fend in zip(files, file_starts, file_ends):
                if sel_ptr >= keep_count:
                    break

                # advance to first index inside this file's [fstart, fend)
                while sel_ptr < keep_count and selected[sel_ptr] < fstart:
                    sel_ptr += 1
                if sel_ptr >= keep_count or selected[sel_ptr] >= fend:
                    continue

                # gather all indices in this file
                local_sel = []
                cur = sel_ptr
                while cur < keep_count and selected[cur] < fend:
                    local_sel.append(int(selected[cur] - fstart))  # local offset
                    cur += 1

                if not local_sel:
                    continue

                print(f"[INFO] writing from {filename} ({len(local_sel)} samples) -> {os.path.basename(hdf5_path)}")
                path = os.path.join(pt_folder, filename)
                dataset: InteractiveDataset = InteractiveDataset.load_dataset(path)
                data = dataset.data
                labels = dataset.labels

                local_sorted = np.array(sorted(local_sel), dtype=np.int64)
                for rs, re in HDF5Dataset._runs_from_indices(local_sorted):
                    nx = re - rs
                    block_x = data[rs:re].detach().cpu().numpy()
                    block_y = labels[rs:re].detach().cpu().numpy()
                    dset_x[write_offset:write_offset + nx] = block_x
                    dset_y[write_offset:write_offset + nx] = block_y
                    write_offset += nx

                sel_ptr = cur

            assert write_offset == keep_count, f"Write mismatch: wrote {write_offset}, expected {keep_count}"

    # ------------- helpers -------------

    @staticmethod
    def _normalize_mode(mode: Optional[str]) -> str:
        m = (mode or "crop_start").strip().lower()
        aliases = {
            "start": "crop_start",
            "begin": "crop_start",
            "head": "crop_start",
            "end": "crop_end",
            "tail": "crop_end",
            "random": "crop_random",
            "rand": "crop_random",
        }
        return aliases.get(m, m)

    @staticmethod
    def _choose_indices(
        total: int,
        fixed_length: Optional[int],
        crop_mode: str,
        seed: Optional[int],
    ) -> np.ndarray:
        """
        Return a sorted numpy array of selected global indices in [0, total).
        If fixed_length is None or >= total, returns all indices [0..total).
        """
        if total < 0:
            raise ValueError("total must be >= 0")

        if fixed_length is None or fixed_length >= total:
            return np.arange(total, dtype=np.int64)

        n = int(fixed_length)
        if n <= 0:
            return np.array([], dtype=np.int64)

        mode = HDF5Dataset._normalize_mode(crop_mode)
        if mode == "crop_start":
            return np.arange(n, dtype=np.int64)
        if mode == "crop_end":
            start = total - n
            return np.arange(start, total, dtype=np.int64)
        if mode == "crop_random":
            rng = np.random.default_rng(seed)
            sel = rng.choice(total, size=n, replace=False)
            sel.sort()
            return sel.astype(np.int64)

        raise ValueError(f"Unknown crop mode: {crop_mode}")

    @staticmethod
    def _runs_from_indices(indices: np.ndarray) -> Iterable[Tuple[int, int]]:
        """
        Yield (start, end_exclusive) of consecutive runs from a sorted 1D array.
        Example: [2,3,4, 7,8, 12] -> (2,5), (7,9), (12,13)
        """
        if indices.size == 0:
            return
        indices = np.asarray(indices, dtype=np.int64)
        starts = [int(indices[0])]
        ends = []
        diffs = np.diff(indices)
        split_points = np.where(diffs != 1)[0]
        for sp in split_points:
            ends.append(int(indices[sp] + 1))
            starts.append(int(indices[sp + 1]))
        ends.append(int(indices[-1] + 1))
        for s, e in zip(starts, ends):
            yield s, e

    @staticmethod
    def _guess_sample_layout(shape: tuple) -> str | None:
        """
        Heuristic hint for common layouts. Purely informational.
        """
        if not shape:
            return None
        # 3D shapes often mean images; 4D often volumes
        if len(shape) == 3:
            c, h, w = shape
            # channels-first if the first dim looks like channels
            if c in (1, 2, 3, 4, 6, 8, 16):
                return "channels-first (C,H,W)"
            # channels-last if last dim looks like channels
            if shape[-1] in (1, 2, 3, 4, 6, 8, 16):
                return "channels-last (H,W,C)"
            return "3D tensor"
        if len(shape) == 4:
            # Could be (C,D,H,W) or (D,H,W,C)
            if shape[0] in (1, 2, 3, 4, 6, 8, 16):
                return "channels-first 3D (C,D,H,W)"
            if shape[-1] in (1, 2, 3, 4, 6, 8, 16):
                return "channels-last 3D (D,H,W,C)"
            return "4D tensor"
        return f"{len(shape)}D tensor"

    @staticmethod
    def print_file_info(hdf5_path: str):
        """Print info without constructing a dataset/crop."""
        with h5py.File(hdf5_path, "r") as f:
            fx = f["features"];
            fy = f["labels"]
            total = int(fx.shape[0])
            fshape = tuple(fx.shape[1:]);
            lshape = tuple(fy.shape[1:])
            fmt_hint = HDF5Dataset._guess_sample_layout(fshape)

            print("HDF5 file summary")
            print(f"  file              : {hdf5_path}")
            print(f"  total samples     : {total}")
            print("  features          :")
            print(f"    dtype           : {fx.dtype}")
            print(f"    per-sample shape: {fshape} {f'({fmt_hint})' if fmt_hint else ''}")
            print("  labels            :")
            print(f"    dtype           : {fy.dtype}")
            print(f"    per-sample shape: {lshape}")

            if f.attrs:
                print("  root attrs        :")
                for k, v in f.attrs.items():
                    print(f"    {k} = {v}")

    @staticmethod
    def join_hdf5_files(in_paths: list[str], out_path: str, *, compression=None, compression_opts=None):
        """
        Concatenate multiple HDF5 datasets into one.
        Assumes each file has 'features' and 'labels' with the same per-sample shape.
        """
        if not in_paths:
            raise ValueError("No input files provided.")

        # Scan shapes
        total = 0
        feature_shape = None
        label_shape = None
        dtype_x = None
        dtype_y = None
        for p in in_paths:
            with h5py.File(p, "r") as f:
                fx, fy = f["features"], f["labels"]
                if feature_shape is None:
                    feature_shape = tuple(fx.shape[1:])
                    label_shape = tuple(fy.shape[1:])
                    dtype_x, dtype_y = fx.dtype, fy.dtype
                else:
                    assert fx.shape[1:] == feature_shape, f"Shape mismatch in {p}"
                    assert fy.shape[1:] == label_shape, f"Label shape mismatch in {p}"
                total += fx.shape[0]

        # Create output
        with h5py.File(out_path, "w") as dst:
            dset_x = dst.create_dataset(
                "features", shape=(total, *feature_shape), dtype=dtype_x,
                compression=compression, compression_opts=compression_opts
            )
            dset_y = dst.create_dataset(
                "labels", shape=(total, *label_shape), dtype=dtype_y,
                compression=compression, compression_opts=compression_opts
            )

            # Copy data in blocks
            write_off = 0
            for p in in_paths:
                with h5py.File(p, "r") as f:
                    fx, fy = f["features"], f["labels"]
                    n = fx.shape[0]
                    dset_x[write_off:write_off + n] = fx[:]
                    dset_y[write_off:write_off + n] = fy[:]
                    write_off += n

def main():
    h5_path = r"H:\ws_abc_chunks\testing\inside_outside_A_32_pd0_bw8_nk3_20250915-110203_dataset_cropped.h5"
    h5_path_test = r"H:\ws_abc_chunks\testing\test.h5"
    h5_path_joined = r"H:\ws_abc_chunks\testing\joined.h5"

    HDF5Dataset.join_hdf5_files([h5_path, h5_path_test], h5_path_joined)
    HDF5Dataset.print_file_info(h5_path)
    HDF5Dataset.print_file_info(h5_path_test)
    HDF5Dataset.print_file_info(h5_path_joined)






if __name__ == "__main__":
    main()
