import os.path
from dbm import error

from visualization import color_templates
import torch
import pickle
from utility.data_exchange import cppIOexcavator
import numpy as np
from dl_torch.models.UNet3D_Segmentation import UNet3D_16EL
from dl_torch.model_utility import Custom_Metrics
from pathlib import Path


def validate_segmentation_model(val_dataset_loc: str, weights_loc: str, save_loc: str, kernel_size: int, padding: int):

    val_sample_names = os.listdir(val_dataset_loc)
    val_sample_path = [os.path.join(val_dataset_loc, name) for name in val_sample_names]

    val_sample_path = [
        p for p in val_sample_path
        if Path(p).is_dir() and
           (Path(p) / "segmentation_data.dat").exists() and
           (Path(p) / "segmentation_data_segments.bin").exists()
    ]

    sample_result = []

    for s_index, sample in enumerate(val_sample_path):
        sample = Path(sample)
        sample_name = sample.name
        dat_path = sample / "segmentation_data.dat"
        bin_path = sample / "segmentation_data_segments.bin"

        if not dat_path.exists() or not bin_path.exists():
            print(f"[WARN] missing {'' if dat_path.exists() else dat_path.name} "
                  f"{'' if bin_path.exists() else bin_path.name} in {sample} — skipping")
            continue

        try:
            segment_data = cppIOexcavator.parse_dat_file(dat_path)
            sdf_grids = cppIOexcavator.load_segments_from_binary(bin_path)
        except FileNotFoundError as e:
            print(f"[WARN] parse_dat_file failed or load_segments_from_binary for {sample}: {e} — skipping")
            continue

        origins = segment_data["ORIGIN_CONTAINER"]["data"]
        face_to_grid_index = segment_data["FACE_TO_GRID_INDEX_CONTAINER"]["data"]
        ftm_ground_truth = segment_data["FACE_TYPE_MAP"]

        # data torch
        model_input = torch.tensor(np.array(sdf_grids))
        model_input = model_input.unsqueeze(1)  # (N, 1, D, H, W)

        # -------------------------
        # load model (fixed to 8 classes)
        # -------------------------
        print("Evaluating Model")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", device)

        model = UNet3D_16EL(in_channels=1, out_channels=8)

        # Safe checkpoint load (supports raw or 'state_dict'-wrapped)
        ckpt = torch.load(weights_loc, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # Hard check: must be 8-class checkpoint; otherwise skip cleanly
        head_w = state_dict.get('final_conv.weight')
        if head_w is not None and head_w.shape[0] != 8:
            print(f"[ERROR] Checkpoint head has {head_w.shape[0]} classes, "
                  f"but model is initialized for 8. Skipping sample '{sample_name}'.")
            continue

        # strict=True so size mismatches don't slip through silently
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            print(f"[INFO] load_state_dict: missing={missing}, unexpected={unexpected}")

        model.to(device)
        model.eval()

        # use model
        with torch.no_grad():
            model_input = model_input.to(device)
            model_output = model(model_input)  # (N, C, D, H, W)
            model_output = model_output.cpu()

            _, prediction = torch.max(model_output, 1)  # (N, D, H, W)
            prediction = prediction.numpy()
            model_output = model_output.numpy()

        # -------------------------
        # assemble outputs (EXCLUSIVE upper bound to avoid off-by-one)
        # -------------------------
        origins_array = np.asarray(origins, dtype=np.int64)  # (N, 3)
        bottom_coord = np.min(origins_array, axis=0)                     # inclusive
        top_excl = np.max(origins_array + kernel_size, axis=0)           # exclusive
        dim_vec = top_excl - bottom_coord

        # Offsets from bottom (vectorized & robust)
        offsets = [origin - bottom_coord for origin in origins_array]

        full_grid = np.zeros(shape=(int(dim_vec[0]), int(dim_vec[1]), int(dim_vec[2])), dtype=np.int32)

        pad_lo = int(padding * 0.5)
        pad_hi = kernel_size - int(padding * 0.5)  # end exclusive in range()

        # --- OOB logging for voxel writes ---
        oob_voxel_writes = 0
        oob_voxel_examples = []
        MAX_OOB_LOG = 10

        for g_index in range(prediction.shape[0]):
            grid = prediction[g_index, :]  # (D, H, W)
            ox, oy, oz = map(int, offsets[g_index])

            for x in range(pad_lo, pad_hi):
                gx = ox + x
                if gx < 0 or gx >= full_grid.shape[0]:
                    if len(oob_voxel_examples) < MAX_OOB_LOG:
                        oob_voxel_examples.append(("x", g_index, gx, full_grid.shape[0], x, ox))
                    oob_voxel_writes += (pad_hi - pad_lo) * (pad_hi - pad_lo)
                    continue

                for y in range(pad_lo, pad_hi):
                    gy = oy + y
                    if gy < 0 or gy >= full_grid.shape[1]:
                        if len(oob_voxel_examples) < MAX_OOB_LOG:
                            oob_voxel_examples.append(("y", g_index, gy, full_grid.shape[1], y, oy))
                        oob_voxel_writes += (pad_hi - pad_lo)
                        continue

                    for z in range(pad_lo, pad_hi):
                        gz = oz + z
                        if 0 <= gz < full_grid.shape[2]:
                            full_grid[gx, gy, gz] = int(grid[x, y, z])
                        else:
                            if len(oob_voxel_examples) < MAX_OOB_LOG:
                                oob_voxel_examples.append(("z", g_index, gz, full_grid.shape[2], z, oz))
                            oob_voxel_writes += 1

        if oob_voxel_writes > 0:
            print(f"[OOB][voxels] Sample {s_index} '{sample_name}': "
                  f"{oob_voxel_writes} voxel writes skipped. Examples (dim, patch, idx, dim_size, local, offset): "
                  f"{oob_voxel_examples}")

        # color/classes template (your chosen template)
        color_temp = color_templates.inside_outside_color_template_abc()
        index_to_class = color_templates.get_index_to_class_dict(color_temp)
        class_to_index = color_templates.get_class_to_index_dict(color_temp)

        # map color to faces
        ftm_prediction = []

        # --- OOB logging for face reads ---
        oob_face_reads = 0
        oob_face_examples = []

        for face_idx, face_index in enumerate(face_to_grid_index):
            face_idx_arr = np.asarray(face_index, dtype=np.int64)
            grid_coord = face_idx_arr - bottom_coord  # still in exclusive-top space
            gx, gy, gz = int(grid_coord[0]), int(grid_coord[1]), int(grid_coord[2])

            if (0 <= gx < full_grid.shape[0] and
                0 <= gy < full_grid.shape[1] and
                0 <= gz < full_grid.shape[2]):
                face_class_index = int(full_grid[gx, gy, gz])
            else:
                oob_face_reads += 1
                if len(oob_face_examples) < MAX_OOB_LOG:
                    oob_face_examples.append((face_idx, tuple(map(int, face_idx_arr.tolist())),
                                              (gx, gy, gz), tuple(full_grid.shape)))
                face_class_index = 7  # fallback to outside

            ftm_prediction.append(face_class_index)

        if oob_face_reads > 0:
            print(f"[OOB][faces] Sample {s_index} '{sample_name}': "
                  f"{oob_face_reads}/{len(face_to_grid_index)} face lookups OOB. "
                  f"Examples (i, face_idx, grid_idx, grid_shape): {oob_face_examples}")

        ftm_ground_truth = list(ftm_ground_truth.values())
        ftm_ground_truth = [class_to_index[item] for item in ftm_ground_truth]

        sample_iou = Custom_Metrics.mesh_IOU(ftm_prediction, ftm_ground_truth).item()

        print(f"Sample {s_index}: Mesh {sample_name} Intersection Over Union {sample_iou:.6f}")

        sample_result.append([s_index, sample_name, sample_iou])

    # saving
    with open(save_loc, "wb") as f:
        pickle.dump(sample_result, f)


def main():
    # w_loc_0 = r"H:\ABC\ABC_torch\temp_models\UNet3D_SDF_16EL_n_class_10_multiset_250k\UNet3D_SDF_16EL_n_class_10_multiset_250k_lr[0.0001]_lrdc[1e-01]_bs16_save_0.pth"
    w_loc_1 = r"H:\ABC\ABC_torch\temp_models\UNet3D_SDF_16EL_n_class_10_multiset_1f0_mio\UNet3D_SDF_16EL_n_class_10_multiset_1f0_mio_lr[0.0001]_lrdc[1e-01]_bs16_save_90.pth"

    w_loc = [w_loc_1]

    save_loc = r"H:\ABC\ABC_statistics\val_segmentation\val_sample_2500"
    kernel_size = 16

    val_dataset_path = r"H:\ABC\ABC_Datasets\Segmentation\validation_samples\val_2500_ks_16_pad_4_bw_5_vs_adaptive_n3"

    for w in w_loc:
        save_name = os.path.splitext(w)[0] + ".pkl"
        save_name = os.path.basename(save_name)
        save_name = os.path.join(save_loc, save_name)
        validate_segmentation_model(val_dataset_path, w, save_name, kernel_size, 4)


if __name__ == "__main__":
    main()
