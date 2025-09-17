from visualization import color_templates
import torch
import pickle
from dl_torch.models.UNet3D_Segmentation import UNet3D_16EL, UNet_Hilbig
from utility.data_exchange import cppIOexcavator
from dl_torch.data_utility import InteractiveDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


def visu_mesh_label(data_loc : str, save_loc : str):


    # parameters
    bin_array_file = data_loc + "/segmentation_data_segments.bin"
    segment_info_file = data_loc + "/segmentation_data.dat"

    segment_data = cppIOexcavator.parse_dat_file(segment_info_file)


    face_type_map = segment_data["FACE_TYPE_MAP"]

    # Define RGB (0–255) and opacity (0.0–1.0) for all classes, including 'Void'
    # Define RGB (0–255) for all classes
    color_temp = color_templates.default_color_template_abc()

    custom_colors = color_templates.get_color_dict(color_temp)
    custom_opacity = color_templates.get_opacity_dict(color_temp)

    face_colors = []

    face_labels = list(face_type_map.values())

    for label in face_labels:
        if label in custom_colors:
            rgb = custom_colors[label]
            rgba = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0)  # Normalize RGB to 0–1 and add alpha
            face_colors.append(rgba)
        else:
            # Default color if label missing
            face_colors.append((1.0, 1.0, 1.0, 1.0))

    # Now save it
    with open(save_loc, "wb") as f:
        pickle.dump(face_colors, f)

def visu_mesh_model_on_dir(data_loc : str,weights_loc : str, save_loc : str, kernel_size : int, padding : int, n_classes):
    # parameters
    bin_array_file = data_loc + "/segmentation_data_segments.bin"
    segment_info_file = data_loc + "/segmentation_data.dat"

    segment_data = cppIOexcavator.parse_dat_file(segment_info_file)

    origins = segment_data["ORIGIN_CONTAINER"]["data"]
    face_to_grid_index = segment_data["FACE_TO_GRID_INDEX_CONTAINER"]["data"]
    sdf_grid = cppIOexcavator.load_segments_from_binary(bin_array_file)
    # data torch
    model_input = torch.tensor(np.array(sdf_grid))
    model_input = model_input.unsqueeze(1)

    # load model
    print("Evaluating Model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = UNet_Hilbig(in_channels=1, out_channels=n_classes)
    state_dict = torch.load(weights_loc)

    model.load_state_dict(state_dict)  # it takes the loaded dictionary, not the path file itself
    model.to(device)
    model.eval()

    # use model
    with torch.no_grad():
        model_input = model_input.to(device)
        model_output = model(model_input)
        model_output = model_output.cpu()

        _, prediction = torch.max(model_output, 1)
        prediction = prediction.cpu().numpy()
        model_output = model_output.numpy()

    # assemble outputs
    origins_array = np.asarray(origins)
    bottom_coord = np.min(origins_array, axis=0)
    top_coord = np.max(origins_array + kernel_size - 1, axis=0)

    offsets = [[0,0,0] - bottom_coord + origin for origin in origins]

    dim_vec = top_coord - bottom_coord

    full_grid = np.zeros(shape=(int(dim_vec[0]), int(dim_vec[1]), int(dim_vec[2])))

    for g_index in range(prediction.shape[0]):

        grid = prediction[g_index,:]

        offset = offsets[g_index]

        for x in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
            for y in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                for z in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                    full_grid[int(offset[0]) + x,int(offset[1]) + y, int(offset[2]) + z] = grid[x,y,z]

    color_temp = color_templates.edge_color_template_abc_sorted()

    class_list = color_templates.get_class_list(color_temp)

    custom_colors = color_templates.get_color_dict(color_temp)
    custom_opacity = color_templates.get_opacity_dict(color_temp)

    index_to_class = color_templates.get_index_to_class_dict(color_temp)
    class_to_index = color_templates.get_class_to_index_dict(color_temp)

    #map color to faces

    face_colors = []
    ftm_prediction = []

    for face_index in face_to_grid_index:
        gird_coord = face_index - bottom_coord
        face_class_index = full_grid[int(gird_coord[0]), int(gird_coord[1]), int(gird_coord[2])]
        face_class = index_to_class[int(face_class_index)]

        ftm_prediction.append(face_class)

        if face_class in custom_colors:
            rgb = custom_colors[face_class]
            rgba = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0)  # Normalize RGB to 0–1 and add alpha
            face_colors.append(rgba)
        else:
            # Default color if label missing
            face_colors.append((1.0, 1.0, 1.0, 1.0))

    # Now save it
    with open(save_loc, "wb") as f:
        pickle.dump(face_colors, f)

    # Derive .txt path from save_loc using os.path
    txt_save_path = os.path.splitext(save_loc)[0] + ".txt"

    # Write face_index: surface_type mapping to the .txt file
    with open(txt_save_path, "w") as f_txt:
        for idx, label in enumerate(ftm_prediction):
            f_txt.write(f"{idx}: {label}\n")

    # ftm_ground_truth = cppIO.read_type_map_from_binary(os.path.join(data_loc, "FaceTypeMap.bin"))

    # ftm_ground_truth = [item[0] for item in ftm_ground_truth]

    # ftm_ground_truth = [class_to_index[item] for item in ftm_ground_truth]

    # ftm_prediction = [class_to_index[item] for item in ftm_prediction]

    # print(f"Mesh Intersection Over Union {Custom_Metrics.mesh_IOU(ftm_prediction, ftm_ground_truth)}")

def visu_histogram_segmentation_samples(val_result_loc :  str):

    # create model signature
    model_name = os.path.basename(val_result_loc)
    model_name, _ = os.path.splitext(model_name)

    with open(val_result_loc, "rb") as f:
        sample_result = pickle.load(f)

    df = pd.DataFrame(sample_result, columns=['Sample_ID', 'ABC_ID', 'Accuracy'])

    current_path = os.path.abspath(val_result_loc)
    path_without_ext = os.path.splitext(current_path)[0]

    df.to_csv(path_without_ext + ".csv")

    sample_iou = np.array([item[2]*100 for item in sample_result])
    rounded_data = np.round(sample_iou, 2)
    total_samples = len(sample_iou)



    # Settings
    x_limit = 100
    bins = 100

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot histogram and capture bar info
    counts, bin_edges, patches = ax.hist(sample_iou, bins=bins, alpha=0.6, edgecolor='black')

    # Annotate each bar with count value
    for count, x in zip(counts, bin_edges[:-1]):
        if count > 0:
            ax.text(x + (bin_edges[1] - bin_edges[0]) / 2, count, f'{int(count)}',
                    ha='center', va='bottom', fontsize=8, rotation=90)

    # Calculate statistics
    mean_val = np.mean(sample_iou)
    median_val = np.median(sample_iou)

    mode_result = stats.mode(rounded_data, keepdims=True)
    mode_val = mode_result.mode[0]
    mode_count = mode_result.count[0]

    # Calculate percentiles
    p25 = np.percentile(sample_iou, 25)
    p75 = np.percentile(sample_iou, 75)

    # Add vertical lines
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax.axvline(mode_val, color='blue', linestyle='solid', linewidth=1.5, label=f'Mode: {mode_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='dotted', linewidth=1.5, label=f'Median: {median_val:.2f}')
    ax.axvline(p25, color='orange', linestyle='dashdot', linewidth=1.5, label=f'25th Percentile: {p25:.2f}')
    ax.axvline(p75, color='purple', linestyle='dashdot', linewidth=1.5, label=f'75th Percentile: {p75:.2f}')

    # Titles and limits
    ax.set_title(f'Histogram of IoU on ABC samples\nTotal samples: {total_samples} \n{model_name}')
    ax.set_xlim(0, x_limit)
    ax.set_xlabel("IoU Value")
    ax.set_ylabel("Frequency")
    ax.legend()



    plt.tight_layout()
    plt.show()

import os
import numpy as np
import torch
import pyvista as pv
from matplotlib.colors import ListedColormap

# Assumes these exist in your codebase:
# - cppIOexcavator.load_segments_from_binary
# - cppIOexcavator.parse_dat_file
# - color_templates.edge_color_template_abc / get_class_list / get_color_dict / get_opacity_dict
# - UNet_Hilbig



def visu_voxel_on_dir(data_loc: str, weights_loc: str, kernel_size: int, padding: int, n_classes: int,
                      stride: int = 1, surface_only: bool = False):
    # -----------------------------
    # 1) Load segments + metadata
    # -----------------------------
    data_arrays = cppIOexcavator.load_segments_from_binary(
        os.path.join(data_loc, "segmentation_data_segments.bin")
    )
    seg_info = cppIOexcavator.parse_dat_file(
        os.path.join(data_loc, "segmentation_data.dat")
    )

    # Model input: (N,1,ks,ks,ks)
    model_input = torch.tensor(np.array(data_arrays)).unsqueeze(1)

    # -----------------------------
    # 2) Load model + predict
    # -----------------------------
    print("Evaluating Model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = UNet_Hilbig(in_channels=1, out_channels=n_classes)
    state_dict = torch.load(weights_loc, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("model predicting outputs...")
    with torch.no_grad():
        model_input = model_input.to(device)
        model_output = model(model_input)          # (N, C, ks, ks, ks)
        model_output = model_output.cpu()
        _, prediction = torch.max(model_output, 1) # (N, ks, ks, ks)
        prediction = prediction.numpy()            # int64

    # -----------------------------
    # 3) Assemble full label grid
    # -----------------------------
    origins = seg_info["ORIGIN_CONTAINER"]["data"]  # list of (x,y,z)
    origins_array = np.asarray(origins, dtype=int)

    bottom_coord   = np.min(origins_array, axis=0)
    top_inclusive  = np.max(origins_array + int(kernel_size) - 1, axis=0)
    top_exclusive  = top_inclusive + 1
    dim_vec        = (top_exclusive - bottom_coord).astype(int)

    # Offsets per block (global placement of local [0..ks) coords)
    offsets = [(-bottom_coord + np.array(o, dtype=int)) for o in origins]

    # Color/opacity template
    color_temp     = color_templates.edge_color_template_abc_sorted()
    class_list     = color_templates.get_class_list(color_temp)          # ordered names -> indices
    custom_colors  = color_templates.get_color_dict(color_temp)          # {name: (R,G,B)}
    custom_opacity = color_templates.get_opacity_dict(color_temp)        # {name: alpha}

    # Default fill: Outside
    void_idx = class_list.index('Outside')

    # Compact dtype for labels
    if n_classes <= 255:
        label_dtype = np.uint8
    elif n_classes <= 65535:
        label_dtype = np.uint16
    else:
        label_dtype = np.int32

    full_grid = np.full(tuple(dim_vec), fill_value=void_idx, dtype=label_dtype)

    print("assembling model outputs...")
    ks       = int(kernel_size)
    pad_half = int(padding // 2)
    x0, x1   = pad_half, ks - pad_half
    if x1 <= x0:
        print("Padding too large for kernel_size; nothing to paste.")
        return

    # Paste each predicted block (vectorized slice; no inner xyz loops)
    for g_index, off in enumerate(offsets):
        block = prediction[g_index]                  # (ks, ks, ks), int64
        crop  = block[x0:x1, x0:x1, x0:x1]          # remove padding border
        dx, dy, dz = crop.shape
        ox, oy, oz = (int(off[0]) + x0, int(off[1]) + x0, int(off[2]) + x0)
        full_grid[ox:ox+dx, oy:oy+dy, oz:oz+dz] = crop.astype(label_dtype, copy=False)

    # -----------------------------
    # 4) Build draw set: non-Inside/Outside voxels
    # -----------------------------
    labels = full_grid.astype(np.int32, copy=False)

    hidden = {'Outside'}
    visible_class_mask = np.array([name not in hidden for name in class_list], dtype=bool)
    keep = visible_class_mask[labels]   # True where voxel should be drawn

    if not np.any(keep):
        print("No voxels to render (all are Inside/Outside).")
        return

    if surface_only:
        # Keep only boundary voxels of the visible set (optional toggle)
        vm  = keep
        pad = np.pad(vm, 1, constant_values=False)
        fully_surrounded = (
            pad[2:,1:-1,1:-1] & pad[:-2,1:-1,1:-1] &
            pad[1:-1,2:,1:-1] & pad[1:-1,:-2,1:-1] &
            pad[1:-1,1:-1,2:] & pad[1:-1,1:-1,:-2]
        )
        keep = vm & ~fully_surrounded
        if not np.any(keep):
            print("No surface voxels remain after filtering.")
            return

    coords       = np.argwhere(keep)   # (K, 3)
    kept_labels  = labels[keep]

    if stride > 1:
        coords      = coords[::stride]
        kept_labels = kept_labels[::stride]

    # -----------------------------
    # 5) Colors (per-voxel RGBA, uint8)
    # -----------------------------
    lut_rgba = np.zeros((len(class_list), 4), dtype=np.uint8)
    for idx, name in enumerate(class_list):
        r, g, b = custom_colors[name]
        a = 0.0 if name in hidden else float(custom_opacity.get(name, 1.0))
        lut_rgba[idx] = (r, g, b, int(round(a * 255)))

    colors_rgba = lut_rgba[kept_labels]  # (K, 4) uint8

    # -----------------------------
    # 6) Glyph render (one actor)
    # -----------------------------
    points = coords.astype(np.float32)   # voxel centers at integer coords
    cloud  = pv.PolyData(points)
    cloud['rgba'] = colors_rgba

    cube   = pv.Cube(center=(0, 0, 0), x_length=1.0, y_length=1.0, z_length=1.0)
    glyphs = cloud.glyph(geom=cube, scale=False, orient=False)

    print("PyVista version:", pv.__version__)
    p = pv.Plotter()
    p.add_mesh(
        glyphs,
        scalars='rgba',
        rgba=True,
        lighting=False,          # flat label colors
        culling='back',          # reduce overdraw
        show_edges=False,
        render_lines_as_tubes=False,
    )
    p.enable_eye_dome_lighting()

    # Legend for visible classes only
    legend_entries = [[name, tuple(c/255 for c in custom_colors[name])]
                      for name in class_list if name not in hidden]
    if legend_entries:
        p.add_legend(legend_entries, bcolor='white', face='circle',
                     size=(0.2, 0.25), loc='lower right')

    # Optional: interactive clip plane to peek inside solid regions
    # p.add_mesh_clip_plane(glyphs)

    p.show()

def visu_label_on_dir(data_loc: str, torch_path: str, kernel_size: int, padding: int, n_classes: int,
                      stride: int = 1, surface_only: bool = False):
    # -----------------------------
    # 1) Load segments + metadata
    # -----------------------------
    data_arrays = cppIOexcavator.load_segments_from_binary(
        os.path.join(data_loc, "segmentation_data_segments.bin")
    )
    seg_info = cppIOexcavator.parse_dat_file(
        os.path.join(data_loc, "segmentation_data.dat")
    )

    dataset = InteractiveDataset.InteractiveDataset.load_dataset(torch_path)
    print(dataset.get_info())

    torch_label = dataset.labels.numpy()  # [N, C, X, Y, Z] or similar
    class_dict = dataset.get_class_dictionary()  # {label_name: class_index}
    torch_class_list = list(class_dict.keys())

    # -----------------------------
    # 3) Assemble full label grid
    # -----------------------------
    origins = seg_info["ORIGIN_CONTAINER"]["data"]  # list of (x,y,z)
    origins_array = np.asarray(origins, dtype=int)

    bottom_coord   = np.min(origins_array, axis=0)
    top_inclusive  = np.max(origins_array + int(kernel_size) - 1, axis=0)
    top_exclusive  = top_inclusive + 1
    dim_vec        = (top_exclusive - bottom_coord).astype(int)

    # Offsets per block (global placement of local [0..ks) coords)
    offsets = [(-bottom_coord + np.array(o, dtype=int)) for o in origins]

    # Color/opacity template
    color_temp     = color_templates.edge_color_template_abc()
    class_list     = color_templates.get_class_list(color_temp)          # ordered names -> indices
    custom_colors  = color_templates.get_color_dict(color_temp)          # {name: (R,G,B)}
    custom_opacity = color_templates.get_opacity_dict(color_temp)        # {name: alpha}

    # Default fill: Outside
    void_idx = class_list.index('Outside')

    # Compact dtype for labels
    if n_classes <= 255:
        label_dtype = np.uint8
    elif n_classes <= 65535:
        label_dtype = np.uint16
    else:
        label_dtype = np.int32

    full_grid = np.full(tuple(dim_vec), fill_value=void_idx, dtype=label_dtype)

    print("assembling model outputs...")
    ks       = int(kernel_size)
    pad_half = int(padding // 2)
    x0, x1   = pad_half, ks - pad_half
    if x1 <= x0:
        print("Padding too large for kernel_size; nothing to paste.")
        return

    # Paste each predicted block (vectorized slice; no inner xyz loops)
    for g_index, off in enumerate(offsets):
        block = torch_label[g_index]                  # (ks, ks, ks), int64
        crop  = block[x0:x1, x0:x1, x0:x1]          # remove padding border
        dx, dy, dz = crop.shape
        ox, oy, oz = (int(off[0]) + x0, int(off[1]) + x0, int(off[2]) + x0)
        full_grid[ox:ox+dx, oy:oy+dy, oz:oz+dz] = crop.astype(label_dtype, copy=False)

    # -----------------------------
    # 4) Build draw set: non-Inside/Outside voxels
    # -----------------------------
    labels = full_grid.astype(np.int32, copy=False)

    hidden = {'Outside'}
    visible_class_mask = np.array([name not in hidden for name in class_list], dtype=bool)
    keep = visible_class_mask[labels]   # True where voxel should be drawn

    if not np.any(keep):
        print("No voxels to render (all are Inside/Outside).")
        return

    if surface_only:
        # Keep only boundary voxels of the visible set (optional toggle)
        vm  = keep
        pad = np.pad(vm, 1, constant_values=False)
        fully_surrounded = (
            pad[2:,1:-1,1:-1] & pad[:-2,1:-1,1:-1] &
            pad[1:-1,2:,1:-1] & pad[1:-1,:-2,1:-1] &
            pad[1:-1,1:-1,2:] & pad[1:-1,1:-1,:-2]
        )
        keep = vm & ~fully_surrounded
        if not np.any(keep):
            print("No surface voxels remain after filtering.")
            return

    coords       = np.argwhere(keep)   # (K, 3)
    kept_labels  = labels[keep]

    if stride > 1:
        coords      = coords[::stride]
        kept_labels = kept_labels[::stride]

    # -----------------------------
    # 5) Colors (per-voxel RGBA, uint8)
    # -----------------------------
    lut_rgba = np.zeros((len(class_list), 4), dtype=np.uint8)
    for idx, name in enumerate(class_list):
        r, g, b = custom_colors[name]
        a = 0.0 if name in hidden else float(custom_opacity.get(name, 1.0))
        lut_rgba[idx] = (r, g, b, int(round(a * 255)))

    colors_rgba = lut_rgba[kept_labels]  # (K, 4) uint8

    # -----------------------------
    # 6) Glyph render (one actor)
    # -----------------------------
    points = coords.astype(np.float32)   # voxel centers at integer coords
    cloud  = pv.PolyData(points)
    cloud['rgba'] = colors_rgba

    cube   = pv.Cube(center=(0, 0, 0), x_length=1.0, y_length=1.0, z_length=1.0)
    glyphs = cloud.glyph(geom=cube, scale=False, orient=False)

    print("PyVista version:", pv.__version__)
    p = pv.Plotter()
    p.add_mesh(
        glyphs,
        scalars='rgba',
        rgba=True,
        lighting=False,          # flat label colors
        culling='back',          # reduce overdraw
        show_edges=False,
        render_lines_as_tubes=False,
    )
    p.enable_eye_dome_lighting()

    # Legend for visible classes only
    legend_entries = [[name, tuple(c/255 for c in custom_colors[name])]
                      for name in class_list if name not in hidden]
    if legend_entries:
        p.add_legend(legend_entries, bcolor='white', face='circle',
                     size=(0.2, 0.25), loc='lower right')

    # Optional: interactive clip plane to peek inside solid regions
    # p.add_mesh_clip_plane(glyphs)

    p.show()


def get_vdb_from_dir(data_loc: str, weights_loc: str, kernel_size: int, padding: int, n_classes, grid_name: str):

    data_arrays = cppIOexcavator.load_segments_from_binary(os.path.join(data_loc ,"segmentation_data_segments.bin"))
    seg_info = cppIOexcavator.parse_dat_file(os.path.join(data_loc , "segmentation_data.dat"))
    # data torch
    model_input = torch.tensor(np.array(data_arrays))
    model_input = model_input.unsqueeze(1)

    # load model
    print("Evaluating Model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = UNet3D_16EL(in_channels=1, out_channels=n_classes)
    state_dict = torch.load(weights_loc)

    model.load_state_dict(state_dict)  # it takes the loaded dictionary, not the path file itself
    model.to(device)
    model.eval()

    print("model predicting outputs...")

    # use model
    with torch.no_grad():
        model_input = model_input.to(device)
        model_output = model(model_input)
        model_output = model_output.cpu()

        _, prediction = torch.max(model_output, 1)
        prediction = prediction.cpu().numpy()
        model_output = model_output.numpy()

    # assemble outputs

    # 'prediction' is your output from the model, shape: (num_segments, D, H, W)
    # If it is (num_segments, D, H, W), convert to a list of 3D arrays
    segments = [np.asarray(prediction[i], dtype=np.float32) for i in range(prediction.shape[0])]
    cppIOexcavator.save_segments_to_binary(os.path.join(data_loc, 'predicted_segments.bin'), segments)

def draw_voxel_input_slice_from_dir(
    data_loc: str,
    kernel_size: int,
    padding: int,
    slice_axis: int = 2,
    slice_index: int = None,
    cmap: str = 'viridis'  # Or 'gray', 'plasma', etc
):
    # 1. LOAD DATA
    data_arrays = cppIOexcavator.load_segments_from_binary(
        os.path.join(data_loc, "segmentation_data_segments.bin")
    )
    seg_info = cppIOexcavator.parse_dat_file(
        os.path.join(data_loc, "segmentation_data.dat")
    )

    # Convert to float numpy array if not already
    data_arrays = np.array(data_arrays, dtype=np.float32)

    # 2. Assemble full input grid
    origins = seg_info["ORIGIN_CONTAINER"]["data"]
    origins_array = np.asarray(origins)
    bottom_coord = np.min(origins_array, axis=0)
    top_coord = np.max(origins_array + kernel_size - 1, axis=0)
    offsets = [np.array([0, 0, 0]) - bottom_coord + origin for origin in origins]
    dim_vec = top_coord - bottom_coord


    full_input_grid = np.full(
        shape=(int(dim_vec[0]), int(dim_vec[1]), int(dim_vec[2])),
        fill_value=1,
        dtype=np.float32
    )


    for g_index in range(data_arrays.shape[0]):
        grid = data_arrays[g_index, :]
        offset = offsets[g_index]
        for x in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
            for y in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                for z in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                    full_input_grid[
                        int(offset[0]) + x,
                        int(offset[1]) + y,
                        int(offset[2]) + z
                    ] = grid[x, y, z]

    # 3. Select slice
    if slice_index is None:
        slice_index = full_input_grid.shape[slice_axis] // 2  # center by default

    if slice_axis == 0:
        img = full_input_grid[slice_index, :, :]
    elif slice_axis == 1:
        img = full_input_grid[:, slice_index, :]
    elif slice_axis == 2:
        img = full_input_grid[:, :, slice_index]
    else:
        raise ValueError("slice_axis must be 0, 1, or 2")

    # 4. Plot
    plt.figure(figsize=(8, 8))
    im = plt.imshow(img, cmap=cmap, origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Input Float Value")
    plt.title(f"Input values slice (axis={slice_axis}, index={slice_index})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def draw_voxel_slice_from_dir(
    data_loc: str,
    weights_loc: str,
    kernel_size: int,
    padding: int,
    n_classes: int,
    slice_axis: int = 2,
    slice_index: int = None
):
    # 1. LOAD DATA
    data_arrays = cppIOexcavator.load_segments_from_binary(
        os.path.join(data_loc, "segmentation_data_segments.bin")
    )
    seg_info = cppIOexcavator.parse_dat_file(
        os.path.join(data_loc, "segmentation_data.dat")
    )

    # 2. Prepare Torch Input
    model_input = torch.tensor(np.array(data_arrays))
    model_input = model_input.unsqueeze(1)

    # 3. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D_16EL(in_channels=1, out_channels=n_classes)
    state_dict = torch.load(weights_loc, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Model Predict
    with torch.no_grad():
        model_input = model_input.to(device)
        model_output = model(model_input)
        model_output = model_output.cpu()
        _, prediction = torch.max(model_output, 1)
        prediction = prediction.cpu().numpy()

    # 5. Assemble full voxel grid
    origins = seg_info["ORIGIN_CONTAINER"]["data"]
    origins_array = np.asarray(origins)
    bottom_coord = np.min(origins_array, axis=0)
    top_coord = np.max(origins_array + kernel_size - 1, axis=0)
    offsets = [np.array([0, 0, 0]) - bottom_coord + origin for origin in origins]
    dim_vec = top_coord - bottom_coord

    color_temp = color_templates.default_color_template_abc()
    class_list = color_templates.get_class_list(color_temp)

    void_class_name = 'Void'
    void_class_idx = class_list.index(void_class_name)
    full_grid = np.full(
        shape=(int(dim_vec[0]), int(dim_vec[1]), int(dim_vec[2])),
        fill_value=void_class_idx,
        dtype=np.float32
    )

    for g_index in range(prediction.shape[0]):
        grid = prediction[g_index, :]
        offset = offsets[g_index]
        for x in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
            for y in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                for z in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                    full_grid[
                        int(offset[0]) + x,
                        int(offset[1]) + y,
                        int(offset[2]) + z
                    ] = grid[x, y, z]

    # 6. Color setup
    color_temp = color_templates.default_color_template_abc()
    class_list = color_templates.get_class_list(color_temp)
    custom_colors = color_templates.get_color_dict(color_temp)
    custom_opacity = color_templates.get_opacity_dict(color_temp)
    color_list = [tuple(c/255 for c in custom_colors[lbl]) for lbl in class_list]
    opacity_list = [custom_opacity[lbl] for lbl in class_list]

    # 7. Select slice
    if slice_index is None:
        slice_index = full_grid.shape[slice_axis] // 2  # center by default

    if slice_axis == 0:
        img = full_grid[slice_index, :, :]
    elif slice_axis == 1:
        img = full_grid[:, slice_index, :]
    elif slice_axis == 2:
        img = full_grid[:, :, slice_index]
    else:
        raise ValueError("slice_axis must be 0, 1, or 2")

    # ----
    # Always include the Void class in the color mapping and legend!
    # So, we include all classes (not just visible ones)
    img_vis = img.astype(int)
    cmap = ListedColormap(color_list)

    legend_elements = [
        Patch(facecolor=color_list[i], label=class_list[i])
        for i in range(len(class_list))
    ]

    plt.figure(figsize=(8, 8))
    plt.imshow(img_vis, cmap=cmap, origin='lower', vmin=0, vmax=len(class_list)-1)
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.title(f"Slice axis={slice_axis}, index={slice_index}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    data_loc = r"W:\label_debug\target\00000003"
    torch_loc = r"W:\label_debug\subsets\edge\00000003.torch"

    visu_label_on_dir(data_loc, torch_loc, 32,  0, 9)

if __name__=="__main__":
    main()