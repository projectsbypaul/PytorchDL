import os.path

from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from utility.data_exchange import cppIO
import torch
import numpy as np
import pyvista as pv
def run_sanity_check_on_labels():

    id = "00000002"

    data_path = os.path.join(r"C:\Local_Data\ABC\ABC_torch\torch_data_ks_16_pad_4_bw_5_vs_adaptive_n2", id + ".torch")
    orgins_path  = r"C:\Local_Data\ABC\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2\00000002\origins.bin"

    dataset = InteractiveDataset.load_dataset(data_path)
    print(dataset.get_info())

    origins = cppIO.read_float_matrix(orgins_path)

    labels = dataset.labels.numpy()

    class_dict = dataset.get_class_dictionary()

    class_list = list(class_dict.keys())

    print()

    # Define RGB (0–255) for all classes
    custom_colors = {
        'Cone': (0, 0, 255),  # blue
        'Cylinder': (255, 0, 0),  # red
        'Edge': (255, 255, 0),  # yellow
        'Plane': (255, 20, 147),  # deep pink
        'Sphere': (128, 0, 0),  # dark red
        'Torus': (0, 255, 255),  # cyan
        'Revolution': (0, 128, 0),  # dark green
        'Extrusion': (255, 165, 0),  # orange
        'Other': (128, 128, 128),  # gray (you may want to update this too)
        'BSpline': (138, 43, 226),  # blue violet
        'Void': (0, 0, 0),  # black
    }

    custom_opacity = {
        'Cone': 1.0,
        'Cylinder': 1.0,
        'Edge': 1.0,
        'Plane': 1.0,
        'Sphere': 1.0,
        'Torus': 1.0,
        'Revolution': 1.0,
        'Extrusion': 1.0,
        'Other': 1.0,
        'BSpline': 1.0,
        'Void': 0.0,
    }

    # Create a PyVista plotter
    plotter = pv.Plotter()

    cubes = []

    for index, grid in enumerate(labels):
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

def main():
    run_sanity_check_on_labels()

if __name__ == "__main__":
    main()
