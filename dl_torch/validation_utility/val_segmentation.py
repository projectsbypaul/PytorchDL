import os.path
from visualization import color_templates
import torch
import pickle
from utility.data_exchange import cppIOexcavator
import numpy as np
from dl_torch.models.UNet3D_Segmentation import UNet3D_16EL
from dl_torch.model_utility import Custom_Metrics

def validate_segmentation_model(val_dataset_loc : str, weights_loc : str, save_loc : str, kernel_size : int, padding : int):

    val_sample_names = os.listdir(val_dataset_loc)
    val_sample_path = [os.path.join(val_dataset_loc, name) for name in val_sample_names]

    sample_result = []

    for s_index, sample in enumerate(val_sample_path):

        segment_data = cppIOexcavator.parse_dat_file(os.path.join(sample, "segmentation_data.dat"))
        origins = segment_data["ORIGIN_CONTAINER"]["data"]
        face_to_grid_index = segment_data["FACE_TO_GRID_INDEX_CONTAINER"]["data"]
        ftm_ground_truth = segment_data["FACE_TYPE_MAP"]

        sdf_grids = cppIOexcavator.load_segments_from_binary(os.path.join(sample, "segmentation_data_segments.bin"))

        # data torch
        model_input = torch.tensor(np.array(sdf_grids))
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

        face_colors = []
        ftm_prediction = []

        for face_index in face_to_grid_index:
            gird_coord = face_index - bottom_coord
            face_class_index = full_grid[int(gird_coord[0]), int(gird_coord[1]), int(gird_coord[2])]

            ftm_prediction.append(face_class_index)

        ftm_ground_truth = list(ftm_ground_truth.values())

        ftm_ground_truth = [class_to_index[item] for item in ftm_ground_truth]

        sample_iou = Custom_Metrics.mesh_IOU(ftm_prediction, ftm_ground_truth).item()

        print(f"Sample {s_index}: Mesh {val_sample_names[s_index]} Intersection Over Union {sample_iou}")

        sample_result.append([s_index, val_sample_names[s_index], sample_iou])

    # saving
    with open(save_loc, "wb") as f:
        pickle.dump(sample_result, f)


def main():
    val_dataset_path = r"H:\ABC\ABC_Datasets\Segmentation\validation_samples\val_1000_ks_16_pad_4_bw_5_vs_adaptive_n3"
    weights_loc = r'C:\src\repos\PytorchDL\data\model_weights\UNet3D_SDF_16EL_n_class_10_multiset_1f0_mio\UNet3D_SDF_16EL_n_class_10_multiset_1f0_mio_lr[0.0001]_lrdc[1e-01]_bs16_save_110.pth'
    save_file = r"H:\ABC\ABC_Testing\val_UNet3D_SDF_16EL_n_class_10_multiset_1f0_mio_lr[0.0001]_lrdc[1e-01]_bs16_save_110.pkl"
    kernel_size = 16
    padding = 4
    validate_segmentation_model(val_dataset_path, weights_loc, save_file,  kernel_size, padding)


if __name__ == "__main__":
    main()