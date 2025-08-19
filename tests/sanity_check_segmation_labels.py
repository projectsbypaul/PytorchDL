import os.path
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from utility.data_exchange import cppIO
import torch
import numpy as np
import pyvista as pv
from visualization import color_templates
from utility.data_exchange import cppIOexcavator

def run_sanity_check_on_labels_parquet():

    sample_grid_size = 10

    grid_spacing = 0.7

    data_path = r"../data/datasets/ABC/ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2_balanced_n_1000.torch"

    dataset = InteractiveDataset.load_dataset(data_path)
    print(dataset.get_info())

    label = dataset.labels

    labels = torch.permute(dataset.labels, (0,2,3,4,1)).numpy()

    class_dict = dataset.get_class_dictionary()

    class_list = list(class_dict.keys())

    color_temp = color_templates.default_color_template_abc()

    custom_colors = color_templates.get_color_dict(color_temp)
    custom_opacity = color_templates.get_opacity_dict(color_temp)

    np.random.seed = 420

    random_integers_array = np.random.randint(0, labels.shape[0] - 1, size=sample_grid_size**2)

    sampled_array = [labels[rand_int] for rand_int in random_integers_array]

    start_point = np.asarray([0, 0, 0])
    grid_width = sampled_array[0].shape[0]
    gap = grid_width*grid_spacing
    origins = []

    for i in range(sample_grid_size):
        for j in range(sample_grid_size):
            origin = start_point + [ (grid_width + grid_spacing) * i, 0, 0] + [0, (grid_width + grid_spacing) * j, 0]
            origins.append(origin)

    # Create a PyVista plotter
    plotter = pv.Plotter()

    cubes = []

    for index, grid in enumerate(sampled_array):
        # Decode labels
        label_indices = np.argmax(grid, axis=-1)

        grid_dim = grid.shape[0]

        origin = origins[index]

        # Iterate through the grid
        for x in range(grid_dim):
            for y in range(grid_dim):
                for z in range(grid_dim):
                    class_idx = label_indices[x, y, z]
                    temp_label = class_list[class_idx]

                    # print(f"plotting {x} {y} {z}")

                    # Skip invisible (Void) cubes
                    if custom_opacity[temp_label] == 0.0:
                        continue

                    # Create a cube centered at the grid location
                    cube = pv.Cube(center=(x + origin[0], y + origin[1], z + origin[2]), x_length=1.0,
                                   y_length=1.0,
                                   z_length=1.0)

                    color = custom_colors[temp_label]

                    color_rgb = tuple(c / 255 for c in color)
                    alpha = custom_opacity[temp_label]

                    rgba = np.append(color_rgb, alpha)  # [R, G, B, A]

                    cube.cell_data["colors"] = np.tile(rgba, (cube.n_cells, 1))

                    cubes.append(cube)

        if not cubes:
            print("No cubes generated for grid", index)
            continue

        combined = pv.MultiBlock(cubes).combine()

        plotter.add_mesh(combined, scalars="colors", rgba=True, show_edges=False)

        print(f"drawing of gird {index} done")

    # ---- Create Custom Legend ----
    legend_entries = []
    for label in class_list:
        if custom_opacity[label] == 0.0:
            continue
        rgb = tuple(c / 255 for c in custom_colors[label])
        legend_entries.append([label, rgb])

    plotter.add_legend(legend_entries, bcolor='white', face='circle', size=(0.2, 0.25), loc='lower right')

    plotter.show()

def run_sanity_check_on_labels_rework():
    id = "00000000"

    data_path = os.path.join(r"H:\ABC\ABC_Benchmark\torch_benchmark\inside_outside", id + ".torch")
    target_dir  = fr"H:\ABC\ABC_Benchmark\Outputs_Benchmark\{id}"

    segment_info_file = os.path.join(target_dir, "segmentation_data.dat")
    segment_data = cppIOexcavator.parse_dat_file(segment_info_file)
    origins = segment_data["ORIGIN_CONTAINER"]["data"]

    dataset = InteractiveDataset.load_dataset(data_path)
    print(dataset.get_info())

    labels = dataset.labels.numpy()             # [N, C, X, Y, Z] or similar
    class_dict = dataset.get_class_dictionary() # {label_name: class_index}
    class_list = list(class_dict.keys())

    color_temp = color_templates.inside_outside_color_template_abc()
    custom_colors  = color_templates.get_color_dict(color_temp)   # {label_name: (r,g,b) 0..255}
    custom_opacity = color_templates.get_opacity_dict(color_temp) # {label_name: alpha 0..1}

    # Build a palette aligned with class indices: index -> [r,g,b,a] in 0..1
    num_classes = len(class_list)
    palette = np.zeros((num_classes, 4), dtype=np.float32)
    for i, lbl in enumerate(class_list):
        rgb = np.array(custom_colors[lbl], dtype=np.float32) / 255.0
        a   = float(custom_opacity[lbl])
        palette[i, :3] = rgb
        palette[i,  3] = a

    plotter = pv.Plotter()
    plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)

    # A single unit cube glyph to be instanced at each visible voxel
    cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)

    for index, grid in enumerate(labels):
        # grid: [C, X, Y, Z] -> [X, Y, Z, C]
        grid = np.moveaxis(grid, 0, -1)
        X, Y, Z, C = grid.shape

        # Argmax to class indices
        label_idx = np.argmax(grid, axis=-1).astype(np.int32)  # [X, Y, Z]

        # RGBA volume via vectorized lookup
        rgba_vol = palette[label_idx]                           # [X, Y, Z, 4]
        visible_mask = rgba_vol[..., 3] > 0.0                   # opacity > 0

        if not np.any(visible_mask):
            print(f"No visible voxels for grid {index}")
            continue

        # (Optional) keep only surface voxels to cut draw calls drastically
        # Comment this block if you truly want all filled voxels.
        vm = visible_mask
        # pad to avoid roll edge-wrap
        pad = np.pad(vm, 1, mode='constant', constant_values=False)
        neighbors = (
            pad[2:,1:-1,1:-1] & pad[:-2,1:-1,1:-1] &
            pad[1:-1,2:,1:-1] & pad[1:-1,:-2,1:-1] &
            pad[1:-1,1:-1,2:] & pad[1:-1,1:-1,:-2]
        )
        surface_mask = vm & ~neighbors
        mask = surface_mask  # or vm for all voxels

        # Coordinates of voxels to draw
        coords = np.argwhere(mask)                              # [N, 3] in (x,y,z) index space
        origin = np.asarray(origins[index], dtype=np.float32)
        points = coords.astype(np.float32) + origin[None, :]

        # Per-voxel RGBA
        rgba = rgba_vol[mask]                                   # [N, 4] float 0..1

        # Build a point cloud; attach per-point RGBA
        cloud = pv.PolyData(points)
        cloud["rgba"] = rgba

        # Instance the cube at each point (no scaling/orient)
        glyphs = cloud.glyph(geom=cube, scale=False, orient=False)

        # Draw; PyVista will use per-vertex RGBA, giving each cube a uniform color
        plotter.add_mesh(glyphs, scalars="rgba", rgba=True, show_edges=False)

        print(f"drawing of grid {index} done")

    # ---- Legend (same idea, but using your template) ----
    legend_entries = []
    for lbl in class_list:
        if custom_opacity[lbl] == 0.0:
            continue
        rgb = tuple(c/255 for c in custom_colors[lbl])
        legend_entries.append([lbl, rgb])
    plotter.add_legend(legend_entries, bcolor='white', face='circle', size=(0.2, 0.25), loc='lower right')

    plotter.show()

def run_sanity_check_on_sdf_bins(bin_path):
    sdf_segment = cppIOexcavator.load_segments_from_binary(bin_path)

    max_val, min_val = 0, 0

    grid_dim = sdf_segment[0].shape[0]

    for segment in sdf_segment:
        for i in range(grid_dim):
            for j in range(grid_dim):
                for k in range(grid_dim):
                    entry = segment[i,j,k,]
                    if entry > max_val:
                        max_val = entry
                    if entry < min_val:
                        min_val = entry


    return min_val, max_val

def run_sanity_check_on_dir(dataset_path, result_path):
    entry_names = os.listdir(dataset_path)
    bin_array_names = [os.path.join(dataset_path, name, "segmentation_data_segments.bin") for name in entry_names]

    results = []

    for index, name in enumerate(bin_array_names):

        if os.path.exists(name):

            min_val, max_val = run_sanity_check_on_sdf_bins(name)

            if min_val >= -1 and max_val <= 1:
                output = f"{entry_names[index]}: min: {min_val} >= -1; max: {max_val} <= 1: ==> PASSED"
                print(output)
                results.append(output)
            else:
                output = f"{entry_names[index]}: min: {min_val} !>= -1; max: {max_val} !<= 1: ==> FAILED"
                print(output)
                results.append(output)

        else:
            output = f"{entry_names[index]}: no data found ==> MISSING"
            print(output)
            results.append(output)

    # Define the output file path
    output_file = os.path.join(dataset_path, result_path)

    # Write results to file
    with open(output_file, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Results saved to {output_file}")

def main():

    run_sanity_check_on_labels_rework()

if __name__ == "__main__":
    main()
