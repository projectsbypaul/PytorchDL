import os
import numpy as np
import h5py
import torch
from numpy.ma.core import shape
from torch.utils.data import Dataset
from typing import Optional, Tuple, Iterable, Callable, Literal
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from utility.data_exchange import cppIOexcavator


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

    # ------------- create from bin tree -------------
    @staticmethod
    def __list_label_and_segment_files(
            root: str,
            segment_signature="segmentation_data_segments.bin",
            label_signature="segmentation_data_labels.bin"):

        subdir_names = os.listdir(root)
        subdir_paths = [os.path.join(root, p) for p in subdir_names]

        data_files = []
        label_files = []

        for i, p in enumerate(subdir_paths):

            data_f_path = os.path.join(p, segment_signature)
            label_f_path = os.path.join(p, label_signature)

            data_exist = os.path.exists(data_f_path)
            label_exist = os.path.exists(label_f_path)

            if not data_exist or not label_exist:
                print(f"Subdir {subdir_names[i]}: binary file for segments or labels missing -> skipping")
                continue

            data_files.append(data_f_path)
            label_files.append(label_f_path)

        return data_files, label_files


    @staticmethod
    def convert_bin_tree_to_hdf5(
            root: str,
            out_hdf5: str,
            segment_signature: str = "segmentation_data_segments.bin",
            label_signature: str = "segmentation_data_labels.bin",
            *,
            compression: Optional[str] = None,  # e.g. "lzf" or "gzip"
            compression_opts: Optional[int] = None,  # gzip level 1..9
            one_pair_per_dir: bool = True,
            progress: Optional[Callable[[int, int, str], None]] = None,
            # NEW controls:
            add_channel_axis: bool = True,
            channel_position: Literal["first", "last"] = "first",
            feature_dtype: str = "float32",
            label_dtype: str = "float32",
    ):
        """
        Writes:
          /features: (TOTAL, C?, 32,32,32) if add_channel_axis else (TOTAL, 32,32,32)
          /labels  : (TOTAL, 32,32,32)

        - Forces dtypes to feature_dtype / label_dtype.
        - If add_channel_axis=True and channel_position='first', features become (N,1,D,H,W).
        """
        # --------- discover files ----------
        seg_paths, lab_paths = HDF5Dataset.__list_label_and_segment_files(
            root, segment_signature, label_signature
        )
        if not seg_paths:
            raise RuntimeError(f"No matching .bin pairs found under {root}")
        if len(seg_paths) != len(lab_paths):
            raise RuntimeError(f"Mismatched file counts: segments={len(seg_paths)} labels={len(lab_paths)}")

        # --------- pass 1: scan ----------
        file_counts: list[int] = []
        total_samples = 0
        base_feat_shape: Optional[Tuple[int, ...]] = None  # shape WITHOUT channel axis
        label_shape: Optional[Tuple[int, ...]] = None

        for sp, lp in zip(seg_paths, lab_paths):
            seg_arr = np.asarray(cppIOexcavator.load_segments_from_binary(sp))  # (Ni, *feat)
            lab_arr = np.asarray(cppIOexcavator.load_labels_from_binary(lp))  # (Ni, *lab)

            parent_name = os.path.basename(os.path.dirname(sp))
            print(f"[INFO] scanning {parent_name} for shape/size")

            if seg_arr.shape[0] != lab_arr.shape[0]:
                raise ValueError(f"Count mismatch:\n  {sp}\n  {lp}\n  {seg_arr.shape[0]} vs {lab_arr.shape[0]}")

            Ni = int(seg_arr.shape[0])
            file_counts.append(Ni)
            total_samples += Ni

            if Ni == 0:
                continue

            cur_feat_shape = tuple(seg_arr.shape[1:])  # (D,H,W) currently
            cur_lab_shape = tuple(lab_arr.shape[1:])  # (D,H,W)

            if base_feat_shape is None:
                base_feat_shape = cur_feat_shape
                label_shape = cur_lab_shape
            else:
                if cur_feat_shape != base_feat_shape:
                    raise ValueError(
                        f"Inconsistent feature shapes: expected {base_feat_shape}, got {cur_feat_shape} in {sp}")
                if cur_lab_shape != label_shape:
                    raise ValueError(f"Inconsistent label shapes: expected {label_shape}, got {cur_lab_shape} in {lp}")

        if base_feat_shape is None or label_shape is None or total_samples == 0:
            raise RuntimeError("No samples found across the discovered .bin pairs.")

        # compute final feature shape (per-sample, WITHOUT batch)
        if add_channel_axis:
            if channel_position == "first":
                final_feat_shape = (1, *base_feat_shape)  # (C=1,D,H,W)
            else:
                final_feat_shape = (*base_feat_shape, 1)  # (D,H,W,C=1)
        else:
            final_feat_shape = base_feat_shape  # (D,H,W)

        # --------- pass 2: write ----------
        with h5py.File(out_hdf5, "w") as f:
            dset_x = f.create_dataset(
                "features",
                shape=(total_samples, *final_feat_shape),
                dtype=feature_dtype,
                compression=compression,
                compression_opts=compression_opts,
            )
            dset_y = f.create_dataset(
                "labels",
                shape=(total_samples, *label_shape),
                dtype=label_dtype,
                compression=compression,
                compression_opts=compression_opts,
            )

            write_offset = 0
            written = 0

            for (sp, lp), Ni in zip(zip(seg_paths, lab_paths), file_counts):
                parent = os.path.dirname(sp)
                parent_name = os.path.basename(parent)

                if Ni == 0:
                    if progress is not None:
                        progress(written, total_samples, parent)
                    continue

                seg_arr = np.asarray(cppIOexcavator.load_segments_from_binary(sp))  # (Ni, D,H,W)
                lab_arr = np.asarray(cppIOexcavator.load_labels_from_binary(lp))  # (Ni, D,H,W)

                # transform features: cast + add channel axis
                x = seg_arr.astype(feature_dtype, copy=False)
                if add_channel_axis:
                    if channel_position == "first":
                        if x.ndim != 4:  # (N,D,H,W)
                            raise RuntimeError(f"Unexpected feature ndim={x.ndim} in {sp}, expected 4")
                        x = x[:, None, ...]  # -> (N,1,D,H,W)
                    else:
                        x = x[..., None]  # -> (N,D,H,W,1)

                # transform labels: cast only
                y = lab_arr.astype(label_dtype, copy=False)

                # sanity
                if x.shape[0] != Ni or y.shape[0] != Ni:
                    raise RuntimeError(f"File changed between passes or load error in:\n  {sp}\n  {lp}")
                if tuple(x.shape[1:]) != final_feat_shape:
                    raise RuntimeError(
                        f"Feature shape drift: expected {final_feat_shape}, got {tuple(x.shape[1:])} in {sp}")
                if tuple(y.shape[1:]) != label_shape:
                    raise RuntimeError(f"Label shape drift: expected {label_shape}, got {tuple(y.shape[1:])} in {lp}")

                # write block
                dset_x[write_offset:write_offset + Ni] = x
                dset_y[write_offset:write_offset + Ni] = y
                write_offset += Ni
                written += Ni

                print(f"[INFO] writing from {parent_name} ({Ni} samples) -> {os.path.basename(out_hdf5)}")
                if progress is not None:
                    progress(written, total_samples, parent)

            assert write_offset == total_samples, f"Write mismatch: wrote {write_offset}, expected {total_samples}"

            # metadata reflects FINAL per-sample shapes
            f.attrs["source_root"] = os.path.abspath(root)
            f.attrs["samples"] = np.int64(total_samples)
            f.attrs["feature_shape"] = np.array(final_feat_shape, dtype=np.int64)
            f.attrs["label_shape"] = np.array(label_shape, dtype=np.int64)

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

        print(f"Joining datasets into {os.path.split(out_path)[1]}")

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
                print(f"Adding {os.path.split(p)[1]}")
                with h5py.File(p, "r") as f:
                    fx, fy = f["features"], f["labels"]
                    n = fx.shape[0]
                    dset_x[write_off:write_off + n] = fx[:]
                    dset_y[write_off:write_off + n] = fy[:]
                    write_off += n

def main():
    data_root = r"H:\ws_abc_chunks\source\ABC_chunk_01_ks32swo4nbw8nk3_20250929-101945\ABC_chunk_01_labeled"
    test_h5 = r"H:\ws_abc_chunks\dataset\inside_outside\cropped\ABC_inside_outside_ks32swo4nbw8nk3_dataset.h5"
    out_h5 = r"H:\ws_abc_chunks\source\ABC_chunk_01_ks32swo4nbw8nk3_20250929-101945\chunk01.h5"
    HDF5Dataset.convert_bin_tree_to_hdf5(data_root, out_h5)
    print("---test---")
    HDF5Dataset.print_file_info(out_h5)
    print("---reference---")
    HDF5Dataset.print_file_info(test_h5)







if __name__ == "__main__":
    main()
