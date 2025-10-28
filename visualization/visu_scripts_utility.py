from visualization import color_templates
import torch
import pickle
from dl_torch.models.UNet3D_Segmentation import UNet3D_16EL, UNet_Hilbig
from visualization import visu_helpers
from utility.data_exchange import cppIOexcavator
import numpy as np
import os

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
    custom_colors = color_templates.get_color_dict(color_temp)
    index_to_class = color_templates.get_index_to_class_dict(color_temp)

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

def run_prediction_on_dir(data_loc: str, weights_loc: str, n_classes: int, model_type:str = "UNet_Hilbig"):

    # -----------------------------
    # 1) Load segments + metadata
    # -----------------------------
    data_arrays = cppIOexcavator.load_segments_from_binary(
        os.path.join(data_loc, "segmentation_data_segments.bin")
    )

    predictions_bin_pth = os.path.join(data_loc, "segmentation_data_predictions.bin")

    # Model input: (N,1,ks,ks,ks)
    model_input = torch.tensor(np.array(data_arrays)).unsqueeze(1)


    # -----------------------------
    # 2) Load model + predict
    # -----------------------------
    print("Evaluating Model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)


    if model_type == "UNet_Hilbig":
        model = UNet_Hilbig(in_channels=1, out_channels=n_classes)
    elif model_type == "UNet_16EL":
        model = UNet3D_16EL(in_channels=1, out_channels=n_classes)
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented")

    state_dict = torch.load(weights_loc, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("model predicting outputs...")
    device = torch.device("cuda", 0)
    batch_size = 4  # start tiny, then increase if it fits
    prediction = visu_helpers.__predict_in_batches(
        model, model_input, batch_size, device, use_amp=True, dtype=torch.bfloat16)


    # -----------------------------
    # 3) Save predictions
    # -----------------------------
    print(f"Saving model predictions to {predictions_bin_pth}")
    print(prediction.dtype)
    prediction = prediction.astype(np.int32)
    print(prediction.dtype)
    print(prediction.shape)
    prediction_list = [prediction[i] for i in range(prediction.shape[0])]
    print(prediction_list[0].shape)  # → (32, 32, 32)
    print(prediction_list[0].dtype)  # → int32
    cppIOexcavator.save_predictions(predictions_bin_pth, prediction)

def main():
    data_loc = r"H:\ws_seg_vdb\output_adaptive"
    weights_loc = r"H:\ws_hpc_workloads\hpc_models\Balanced20k_Edge32_LRE-04\Balanced20k_Edge32_LRE-04_save_10.pth"
    n_classes = 9
    model_type = "UNet_Hilbig"
    arr = np.arange(10, dtype=np.int32)
    arr.tofile(os.path.join(data_loc,"test.bin"))
    run_prediction_on_dir(data_loc, weights_loc, n_classes, model_type)



if __name__=="__main__":
    main()