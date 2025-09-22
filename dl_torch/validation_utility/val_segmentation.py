import os.path
from dbm import error
import pandas as pd
from scipy import stats
from visualization import color_templates
import torch
import pickle
from utility.data_exchange import cppIOexcavator
import numpy as np
from dl_torch.models.UNet3D_Segmentation import UNet_Hilbig, UNet3D_16EL
from dl_torch.model_utility import Custom_Metrics
from pathlib import Path

def val_segmentation_stats_on_dir(val_result_dir :  str, output_file):
    input_files = [f for f in os.listdir(val_result_dir) if f.endswith(".bin")]
    input_paths = [os.path.join(val_result_dir, f) for f in input_files]

    results = []

    for file_path in input_paths:
        try:
            save_name = os.path.splitext(os.path.basename(file_path))[0]
            res_stats = val_segmentation_stats_on_file(file_path)
            results.append((save_name, *res_stats))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    df = pd.DataFrame(results,
                      columns=['Save_Name', 'Mean_IoU', 'Median_IoU', 'Mode_IoU', '25th_Percentile', '75th_Percentile'])
    df.to_csv(output_file, sep=';', decimal=',',index=False)
    print(f"Saved summary to: {output_file}")

def val_segmentation_stats_on_file(val_result_loc :  str):
    # create model signature
    model_name = os.path.basename(val_result_loc)
    model_name, _ = os.path.splitext(model_name)

    with open(val_result_loc, "rb") as f:
        sample_result = pickle.load(f)

    df = pd.DataFrame(sample_result, columns=['Sample_ID', 'ABC_ID', 'Accuracy'])

    current_path = os.path.abspath(val_result_loc)
    path_without_ext = os.path.splitext(current_path)[0]

    df.to_csv(path_without_ext + ".csv")

    sample_iou = np.array([item[2] * 100 for item in sample_result])
    rounded_data = np.round(sample_iou, 2)
    total_samples = len(sample_iou)


    # Calculate statistics
    mean_val = np.mean(sample_iou)
    median_val = np.median(sample_iou)

    mode_result = stats.mode(rounded_data, keepdims=True)
    mode_val = mode_result.mode[0]
    mode_count = mode_result.count[0]

    # Calculate percentiles
    p25 = np.percentile(sample_iou, 25)
    p75 = np.percentile(sample_iou, 75)

    return mean_val, median_val, mode_val, p25, p75

def validate_segmentation_model(
    val_dataset_loc: str,
    weights_loc: str,
    save_loc: str,
    kernel_size: int,
    padding: int,
    n_classes: int,
    model_type: str = "default"
):
    """
    Validate a 3D segmentation model on a dataset of pre-processed samples.

    Args:
        val_dataset_loc (str):
            Path to the validation dataset root directory.
            Each sample is expected to be a subdirectory containing:
              - "segmentation_data.dat" (metadata, origins, face/grid maps)
              - "segmentation_data_segments.bin" (voxelized SDF segments)

        weights_loc (str):
            Path to the model checkpoint file (.pth or similar).
            Supports both raw `state_dict` checkpoints and dicts with "state_dict".

        save_loc (str):
            Output path for storing the evaluation results.
            Results are written as a pickle file containing a list of tuples:
            [sample_index, sample_name, IoU_score].

        kernel_size (int):
            Side length (in voxels) of each cubic patch used when reconstructing
            the prediction into a full voxel grid.

        padding (int):
            Overlap/trim applied to each cubic patch during voxel reconstruction.
            Helps avoid double-counting at patch borders.
            The effective patch region written into the full grid is
            [padding//2 : kernel_size - padding//2].

        n_classes (int):
            Number of segmentation classes to predict.
            The model is constructed with this many output channels.

        model_type (str, default="default"):
            Which segmentation network variant to use.
            Must be one of:
                - "default"     =>  UNet_Hilbig
                - "UNet_Hilbig" =>  UNet_Hilbig
                - "UNet_16EL"   =>  UNet3D_16EL
    """

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

        # Set up dictionary
        model = None

        model_list = {
            "default": 0,
            "UNet_Hilbig": 1,
            "UNet_16EL": 2
        }

        match model_list[model_type]:
            case 0:
                model = UNet_Hilbig(in_channels=1, out_channels=n_classes)
            case 1:
                model = UNet_Hilbig(in_channels=1, out_channels=n_classes)
            case 2:
                model = UNet3D_16EL(in_channels=1, out_channels=n_classes)

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
    stat_dir = r"W:\hpc_workloads\hpc_val\taguchi_L9"
    stat_out = r"W:\hpc_workloads\hpc_val\Block_B_result_2.csv"

    val_segmentation_stats_on_dir(stat_dir, stat_out)



if __name__ == "__main__":
    main()
