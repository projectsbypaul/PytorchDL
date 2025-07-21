from visualization import color_templates
import torch
import pickle
from dl_torch.models.UNet3D_Segmentation import UNet3D_16EL
from utility.data_exchange import cppIOexcavator
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

    model = UNet3D_16EL(in_channels=1, out_channels=10)
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

    color_temp = color_templates.default_color_template_abc()

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

def visu_voxel_on_dir(data_loc: str, weights_loc: str, kernel_size: int, padding: int, n_classes):


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

    origins = seg_info["ORIGIN_CONTAINER"]["data"]
    bottom_coord = np.asarray(origins[0])
    top_coord = np.asarray(origins[len(origins) - 1])
    top_coord += [kernel_size - 1, kernel_size - 1, kernel_size - 1]
    offsets = [[0, 0, 0] - bottom_coord + origin for origin in origins]

    color_temp = color_templates.default_color_template_abc()

    class_list = color_templates.get_class_list(color_temp)

    custom_colors = color_templates.get_color_dict(color_temp)
    custom_opacity = color_templates.get_opacity_dict(color_temp)

    # assemble outputs
    origins_array = np.asarray(origins)
    bottom_coord = np.min(origins_array, axis=0)
    top_coord = np.max(origins_array + kernel_size - 1, axis=0)

    offsets = [[0, 0, 0] - bottom_coord + origin for origin in origins]

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

    print("assembling model outputs...")

    for g_index in range(prediction.shape[0]):

        grid = prediction[g_index, :]

        offset = offsets[g_index]

        for x in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
            for y in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                for z in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                    full_grid[int(offset[0]) + x, int(offset[1]) + y, int(offset[2]) + z] = grid[x, y, z]


    # Create a PyVista plotter
    plotter = pv.Plotter()

    cubes = []

    print("drawing full grid outputs...")

    counter = 0
    n_cubes = full_grid.shape[0] * full_grid.shape[1] * full_grid.shape[1]

    for x in range(full_grid.shape[0]):
        for y in range(full_grid.shape[1]):
            for z in range(full_grid.shape[2]):
                class_idx = int(full_grid[x, y, z])
                temp_label = class_list[class_idx]

                # print(f"plotting {x} {y} {z}")

                # Skip invisible (Void) cubes
                if custom_opacity[temp_label] == 0.0:
                    continue

                # Create a cube centered at the grid location
                cube = pv.Cube(center=(x, y, z), x_length=1.0,
                               y_length=1.0,
                               z_length=1.0)

                color = custom_colors[temp_label]

                color_rgb = tuple(c / 255 for c in color)
                alpha = custom_opacity[temp_label]

                rgba = np.append(color_rgb, alpha)  # [R, G, B, A]

                cube.cell_data["colors"] = np.tile(rgba, (cube.n_cells, 1))

                counter += 1
                print(f"added cubes {counter} / {n_cubes}")

                cubes.append(cube)

    combined = pv.MultiBlock(cubes).combine()

    plotter.add_mesh(combined, scalars="colors", rgba=True, show_edges=False)

    # ---- Create Custom Legend ----
    legend_entries = []
    for label in class_list:
        if custom_opacity[label] == 0.0:
            continue
        rgb = tuple(c / 255 for c in custom_colors[label])
        legend_entries.append([label, rgb])

    plotter.add_legend(legend_entries, bcolor='white', face='circle', size=(0.2, 0.25), loc='lower right')

    plotter.show()

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
    data_loc = r"H:\ABC_Demo\temp"
    # data_loc = r"H:\ABC\ABC_Datasets\Segmentation\val_ks_16_pad_4_bw_5_vs_adaptive_n3\ABC_chunk_00\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n3\00000002"
    weights_loc = r"C:\Users\pschuster\source\repos\PytorchDL\data\model_weights\UNet3D_SDF_16EL_n_class_10_multiset_1f0_mio\UNet3D_SDF_16EL_n_class_10_multiset_1f0_mio_lr[0.0001]_lrdc[1e-01]_bs16_save_120.pth"
    gird_name = "vdb_test"
    # get_vdb_from_dir(data_loc, weights_loc, 16, 4, 10, gird_name)
    get_vdb_from_dir(data_loc, weights_loc ,16, 4, 10, "test")
    # draw_voxel_slice_from_dir(data_loc, weights_loc, 16, 4, 10, 0)

if __name__=="__main__":
    main()