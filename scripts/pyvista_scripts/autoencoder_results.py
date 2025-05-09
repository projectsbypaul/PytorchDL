from utility.data_exchange import cppIO
from dl_torch.models.UNet3D_Autoencoder import AutoEncoder3D
from dl_torch.data_utility.HelperFunctionsABC import get_ABC_bin_arry_from_segment_dir
from scipy.ndimage import zoom
from skimage import measure
from dl_torch.data_utility import Mapping
import pyvista as pv

import os.path
from visualization import color_templates
import torch
import numpy as np

def __visu_sdf_model_on_dir():
    # parameters
    id = "00000003"
    '''
    
    data_loc = f'C:/Local_Data/ABC/ABC_AE_Data_ks_16_pad_4_bw_5_vs_adaptive/{id}'
    '''

    data_loc = r"C:\Local_Data\cropping_test\sdf_segments"
    save_location = r"C:\Local_Data\cropping_test"

    weights_loc = r'../../data/model_weights/Autoencoder_UNEt/Autoencoder_UNEt_lr[1e-05]cdod[0.1]bs4_save_500.pth'
    #mapping
    map_min = 0
    map_max = 1
    min_val = -0.925676
    max_val = 8.96953
    #dl params
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

    model = AutoEncoder3D()
    state_dict = torch.load(weights_loc)

    model.load_state_dict(state_dict)  # it takes the loaded dictionary, not the path file itself
    model.to(device)
    model.eval()

    # use model
    with torch.no_grad():
        model_input = model_input.to(device)
        model_output = model(model_input)
        model_output = model_output.cpu()

        prediction = model_output.cpu().squeeze(1).numpy()

    # assemble outputs

    origins = cppIO.read_float_matrix(os.path.join(data_loc, "origins.bin"))
    bottom_coord = np.asarray(origins[0])
    top_coord = np.asarray(origins[len(origins) - 1])
    top_coord += [kernel_size - 1, kernel_size - 1, kernel_size - 1]
    offsets = [[0, 0, 0] - bottom_coord + origin for origin in origins]

    dim_vec = top_coord - bottom_coord

    full_grid = np.ones(shape=(int(dim_vec[0]), int(dim_vec[1]), int(dim_vec[2])))

    for g_index in range(prediction.shape[0]):

        grid = prediction[g_index, :]

        offset = offsets[g_index]

        for x in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
            for y in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                for z in range(int(padding * 0.5), kernel_size - int(padding * 0.5)):
                    mapped_val = Mapping.linear_mapping(min_val, max_val, map_min, map_max, grid[x, y, z])
                    full_grid[int(offset[0]) + x, int(offset[1]) + y, int(offset[2]) + z] = mapped_val

    #Upsample full_grid (e.g., by 2x in each dimension)
    up_sampled_grid = zoom(full_grid, zoom=2, order=1)  # order=1 → trilinear interpolation
    # Apply marching cubes to get a mesh from the SDF
    verts, faces, normals, _ = measure.marching_cubes(up_sampled_grid, level=0)

    # Convert faces to the format PyVista expects (include face sizes)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)

    # Create a PyVista PolyData mesh
    mesh = pv.PolyData(verts, faces_pv)

    # Visualize it
    mesh.plot(color="lightblue", show_edges=True)

    # Assuming `verts` and `faces` are already created
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
    mesh = pv.PolyData(verts, faces_pv)

    # Save as OBJ
    mesh.save(os.path.join(save_location,f"reconstructed_mesh_{id}.obj"))


def main():
   __visu_sdf_model_on_dir()

if __name__=="__main__":
    main()