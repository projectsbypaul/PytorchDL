import torch
from networkx.algorithms.distance_measures import radius

from dl_torch.models.UNet3D_Segmentation import UNet3D_16EL
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
import numpy as np
import pyvista as pv

def __visu_train_result():
    weights_loc = r'../data/model_weights/UNet3D_SDF_16EL/UNet3D_SDF_16EL_lr[1e-05]_lrdc[1e-01]_bs4_save_400.pth'
    data_loc = r'../data/datasets/ABC/ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2.torch'

    print("Evaluating Model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = UNet3D_16EL()
    state_dict = torch.load(weights_loc)

    model.load_state_dict(state_dict)  # it takes the loaded dictionary, not the path file itself
    model.to(device)
    model.eval()

    dataset = InteractiveDataset.load_dataset(data_loc)
    print(dataset.get_info())

    dataset.set_split(0.75)
    dataset.split_dataset()
    test_loader = dataset.get_test_loader(batch_size=1)

    # class_list = np.unique(flat)
    class_list = np.array(["Cone", "Cylinder", "Edge", "Plane", "Sphere", "Torus", "Void"])

    class_indices = np.arange(len(class_list))

    class_lot = dict(zip(class_list, class_indices))

    # Define RGB (0–255) and opacity (0.0–1.0) for all classes, including 'Void'
    custom_colors = {

        'Cone': (0, 0, 255),  # blue
        'Cylinder': (255, 0, 0),  # red
        'Edge': (255, 255, 0),  # yellow
        'Plane': (255, 192, 203),  # pink
        'Sphere': (128, 0, 0),  # dark red
        'Torus': (0, 255, 255),  # cyan
        'Void': (255, 255, 125),  # white
    }

    custom_opacity = {
        'Cone': 0.0,
        'Cylinder': 1.0,
        'Edge': 1.0,
        'Plane': 1.0,
        'Sphere': 1.0,
        'Torus': 1.0,
        'Void': 1.0,
    }

    with torch.no_grad():

        for data, labels in test_loader:

            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)


            _, predicted = torch.max(outputs, 1)
            _, label = torch.max(labels,1)

            predicted = predicted.cpu()
            predicted = predicted.numpy()
            label = label.cpu()
            label = label.cpu()

            # Create a PyVista plotter
            plotter = pv.Plotter()

            elements = []

            grid_dim = predicted.shape[1]

            # Iterate through the output
            for x in range(grid_dim):
                for y in range(grid_dim):
                    for z in range(grid_dim):
                        class_idx = predicted[0, x, y, z]
                        temp_label = class_list[class_idx]

                        # print(f"plotting {x} {y} {z}")

                        # Skip invisible (Void) cubes
                        if custom_opacity[temp_label] == 0.0:
                            continue

                        # Create a cube centered at the grid location
                        sphere = pv.Sphere(center=(x, y, z), radius=0.5)

                        color = custom_colors[temp_label]

                        color_rgb = tuple(c / 255 for c in color)
                        alpha = custom_opacity[temp_label]

                        rgba = np.append(color_rgb, alpha)  # [R, G, B, A]

                        sphere.cell_data["colors"] = np.tile(rgba, (sphere.n_cells, 1))

                        elements.append(sphere)

            # Iterate through the label
            for x in range(grid_dim):
                for y in range(grid_dim):
                    for z in range(grid_dim):
                        class_idx = label[0, x, y, z]
                        temp_label = class_list[class_idx]

                        # print(f"plotting {x} {y} {z}")

                        # Skip invisible (Void) cubes
                        if custom_opacity[temp_label] == 0.0:
                            continue

                        # Create a cube centered at the grid location
                        sphere = pv.Icosphere(center=(x + grid_dim*1.5, y , z), radius=0.5, nsub=1)

                        color = custom_colors[temp_label]

                        color_rgb = tuple(c / 255 for c in color)
                        alpha = custom_opacity[temp_label]

                        rgba = np.append(color_rgb, alpha)  # [R, G, B, A]

                        sphere.cell_data["colors"] = np.tile(rgba, (sphere.n_cells, 1))

                        elements.append(sphere)

            if len(elements) > 0:

                combined = pv.MultiBlock(elements).combine()

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

            else:
                print("only void in plot --> skipped")


def main():
    __visu_train_result()

if __name__=="__main__":
    main()