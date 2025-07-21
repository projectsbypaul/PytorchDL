import os.path
from visualization import color_templates
import torch
import pickle
from utility.data_exchange import cppIO
from dl_torch.models.UNet3D_Segmentation import UNet3D_16EL
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from dl_torch.data_utility.HelperFunctionsABC import get_ABC_bin_arry_from_segment_dir
import numpy as np
import pyvista as pv
from dl_torch.model_utility import Custom_Metrics

def __eval_model_on_random_sample():
    #Setup data and model
    data_loc = r'C:\Local_Data\ABC\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2'
    weights_loc = r'../../data\model_weights\UNet3D_SDF_16EL_n_class_10_bln_5000\UNet3D_SDF_16EL_n_class_10_bln_5000_lr[1e-06]_lrdc[0.1]bs4_save_last.pth'
    kernel_size = 16
    padding = 4
    n_classes = 10
    sample_size = 500
    # Load data
    ignored_files = ["origins.bin", "VertToGridIndex.bin", "VertTypeMap.bin", "TypeCounts.bin", "FaceTypeMap.bin",
                     "FaceToGridIndex.bin"]

    data_dir = os.listdir(data_loc)

    np.random.seed(42)

    sample_indices = np.random.randint(low=0, high=len(data_dir)-1, size=sample_size)

    sample_result = []

    for n, index in enumerate(sample_indices):

        if len(os.listdir(os.path.join(data_loc, data_dir[index]))) < len(ignored_files):
            print(f"Sample {n}: skipped mesh {data_dir[index]} -> incorrect preprocessing")
            continue

        data_arrays = get_ABC_bin_arry_from_segment_dir(os.path.join(data_loc, data_dir[index]), ignored_files)

        # data torch
        model_input = torch.tensor(np.array(data_arrays))
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

        origins = cppIO.read_float_matrix(os.path.join(data_loc, data_dir[index], "origins.bin"))
        bottom_coord = np.asarray(origins[0])
        top_coord = np.asarray(origins[len(origins) - 1])
        top_coord += [kernel_size - 1, kernel_size - 1, kernel_size - 1]
        offsets = [[0, 0, 0] - bottom_coord + origin for origin in origins]

        dim_vec = top_coord - bottom_coord

        full_grid = np.zeros(shape=(int(dim_vec[0]), int(dim_vec[1]), int(dim_vec[2])))

        for g_index in range(prediction.shape[0]):

            grid = prediction[g_index, :]

            offset = offsets[g_index]

            for x in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                for y in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                    for z in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                        full_grid[int(offset[0]) + x, int(offset[1]) + y, int(offset[2]) + z] = grid[x, y, z]

        color_temp = color_templates.default_color_template_abc()

        index_to_class = color_templates.get_index_to_class_dict(color_temp)
        class_to_index = color_templates.get_class_to_index_dict(color_temp)

        # map color to faces

        FaceToGridIndex = cppIO.read_float_matrix(os.path.join(data_loc, data_dir[index], "FaceToGridIndex.bin"))

        face_colors = []
        ftm_prediction = []

        for face_index in FaceToGridIndex:
            gird_coord = face_index - bottom_coord
            face_class_index = full_grid[int(gird_coord[0]), int(gird_coord[1]), int(gird_coord[2])]

            ftm_prediction.append(face_class_index)

        ftm_ground_truth = cppIO.read_type_map_from_binary(os.path.join(data_loc, data_dir[index], "FaceTypeMap.bin"))

        ftm_ground_truth = [item[0] for item in ftm_ground_truth]

        ftm_ground_truth = [class_to_index[item] for item in ftm_ground_truth]

        sample_iou = Custom_Metrics.mesh_IOU(ftm_prediction, ftm_ground_truth).item()

        print(f"Sample {n}: Mesh {data_dir[index]} Intersection Over Union {sample_iou}")

        sample_result.append([n, data_dir[index], sample_iou])

    # saving
    with open(r"../../data/training_statistics/UNet_Segmentation_sample.pkl", "wb") as f:
        pickle.dump(sample_result, f)

def __visu_mesh_model_on_dir(data_loc : str, weights_loc : str, kernel_size : int, padding : int, n_classes):
    # parameters
    data_loc = r'C:\Local_Data\Segmentation_Alex\hx_gyroid_2'
    weights_loc = r'../../data/model_weights/UNet3D_SDF_16EL_n_class_10/UNet3D_SDF_16EL_n_class_10_lr[1e-05]_lrdc[1e-01]_bs4_save_200.pth'
    kernel_size = 16
    padding = 4
    n_classes = 10
    # Load data
    ignored_files = ["origins.bin", "VertToGridIndex.bin", "VertTypeMap.bin", "TypeCounts.bin", "FaceTypeMap.bin",
                     "FaceToGridIndex.bin"]
    data_arrays = get_ABC_bin_arry_from_segment_dir(data_loc, ignored_files)

    # data torch
    model_input = torch.tensor(np.array(data_arrays))
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

    origins = cppIO.read_float_matrix(os.path.join(data_loc, "origins.bin"))
    bottom_coord = np.asarray(origins[0])
    top_coord = np.asarray(origins[len(origins) - 1])
    top_coord += [kernel_size - 1, kernel_size - 1, kernel_size - 1]
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

    FaceToGridIndex = cppIO.read_float_matrix(os.path.join(data_loc, "FaceToGridIndex.bin"))

    face_colors = []
    ftm_prediction = []

    for face_index in FaceToGridIndex:
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
    with open(r"../../data/blender_export/color_map_learned.pkl", "wb") as f:
        pickle.dump(face_colors, f)

    # ftm_ground_truth = cppIO.read_type_map_from_binary(os.path.join(data_loc, "FaceTypeMap.bin"))

    # ftm_ground_truth = [item[0] for item in ftm_ground_truth]

    # ftm_ground_truth = [class_to_index[item] for item in ftm_ground_truth]

    # ftm_prediction = [class_to_index[item] for item in ftm_prediction]

    # print(f"Mesh Intersection Over Union {Custom_Metrics.mesh_IOU(ftm_prediction, ftm_ground_truth)}")

def __visu_voxel_model_on_dir():
    # parameters
    data_loc = r'C:\Local_Data\ABC\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2\00000002'
    weights_loc = r'../../data\model_weights\UNet3D_SDF_16EL_n_class_10_bln_5000.pth'
    kernel_size = 16
    padding = 4
    n_classes = 7
    # Load data
    ignored_files = ["origins.bin", "VertToGridIndex.bin", "VertTypeMap.bin", "TypeCounts.bin", "FaceTypeMap.bin",
                     "FaceToGridIndex.bin"]
    data_arrays = get_ABC_bin_arry_from_segment_dir(data_loc, ignored_files)

    # data torch
    model_input = torch.tensor(np.array(data_arrays))
    model_input = model_input.unsqueeze(1)

    # load model
    print("Evaluating Model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = UNet3D_16EL()
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

    origins = cppIO.read_float_matrix(os.path.join(data_loc, "origins.bin"))
    bottom_coord = np.asarray(origins[0])
    top_coord = np.asarray(origins[len(origins) - 1])
    top_coord += [kernel_size - 1, kernel_size - 1, kernel_size - 1]
    offsets = [[0, 0, 0] - bottom_coord + origin for origin in origins]

    color_temp = color_templates.small_ABC_template()

    class_list = color_templates.get_class_list(color_temp)

    custom_colors = color_templates.get_color_dict(color_temp)
    custom_opacity = color_templates.get_opacity_dict(color_temp)

    dim_vec = top_coord - bottom_coord

    full_grid = np.zeros(shape=(int(dim_vec[0]), int(dim_vec[1]), int(dim_vec[2])))

    for g_index in range(prediction.shape[0]):

        grid = prediction[g_index, :]

        offset = offsets[g_index]

        for x in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
            for y in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                for z in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                    full_grid[int(offset[0]) + x, int(offset[1]) + y, int(offset[2]) + z] = grid[x, y, z]

    color_temp = color_templates.small_ABC_template()

    class_list = color_templates.get_class_list(color_temp)

    custom_colors = color_templates.get_color_dict(color_temp)
    custom_opacity = color_templates.get_opacity_dict(color_temp)

    # Create a PyVista plotter
    plotter = pv.Plotter()

    cubes = []

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
    
def __visu_train_result():
    weights_loc = r'../../data\model_weights\UNet3D_SDF_16EL_n_class_10_bln_5000\UNet3D_SDF_16EL_n_class_10_bln_5000_lr[1e-05]_lrdc[0.1]bs4_save_last.pth'
    data_loc = r'../data/datasets/ABC/ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2_samples_37273.torch'

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
    data_loc = r'C:\Local_Data\Segmentation_Alex\hx_gyroid_2'
    weights_loc = r'../../data/model_weights/UNet3D_SDF_16EL_n_class_10/UNet3D_SDF_16EL_n_class_10_lr[1e-05]_lrdc[1e-01]_bs4_save_200.pth'
    kernel_size = 16
    padding = 4
    n_classes = 10
    __visu_voxel_model_on_dir(data_loc, weights_loc)

if __name__=="__main__":
    main()