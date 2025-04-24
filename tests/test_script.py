import numpy as np

from utility.data_exchange import cppIO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv
import os
import re

def read_array_test():
    # Load the array
    voxel_size, background, array = cppIO.read_3d_array_from_binary(r"C:\Local_Data\ABC\ABC_Data_ks_32_pad_4_bw_5_vs_adaptive_n2\00000006\00000006_15.bin")

    print(f"Voxel Size: {voxel_size}")
    print(f"Background: {background}")
    # print(array)

    # Define a custom colormap
    colors = ["blue", "white", "red"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Plot the array
    plt.imshow(array[:, 26, :], cmap="seismic")
    plt.colorbar()  # Add a colorbar to show value mapping
    plt.title("2D NumPy Array Visualization")
    plt.show()

def read_array_pyvista():

    # Load the array
    voxel_size, background, sdf = cppIO.read_3d_array_from_binary(r"C:\Local_Data\cropping_test\cropped_32")

    print(f"Voxel Size: {voxel_size}")
    print(f"Background: {background}")
    # print(array)

    # Create a UniformGrid for the SDF
    grid = pv.ImageData()
    grid.dimensions = sdf.shape
    grid.spacing = (1, 1, 1)  # spacing between voxels
    grid.origin = (0, 0, 0)
    grid.cell_data["sdf"] = sdf[:-1, :-1, :-1].flatten(order="F")  # Assign to cells

    # Threshold to get only inside voxels (sdf < 0)
    inside_voxels = grid.threshold(value=0, scalars="sdf", invert=True)

    # Visualization
    plotter = pv.Plotter()
    plotter.add_mesh(inside_voxels, show_edges=True, color="orange", opacity=1.0)
    plotter.show()

    # Visualization
    plotter = pv.Plotter()
    plotter.add_mesh(
        inside_voxels,
        show_edges=False,
        scalars="sdf",  # Use the sdf values for coloring
        cmap="coolwarm",  # Choose a colormap (optional, e.g., "viridis", "plasma", "coolwarm")
        opacity=1.0,
    )
    plotter.show()

def read_dir_of_arrays_pyvista():

    folder_path = r"C:\Local_Data\cropping_test"

    f_names = []

    # Pattern to extract index number from filenames like "cropped_17.bin"
    pattern = re.compile(r"cropped_(\d+)\.bin$")

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            full_path = os.path.abspath(os.path.join(folder_path, filename))
            f_names.append((index, full_path))

    # Sort by the numeric index
    f_names.sort(key=lambda x: x[0])

    sorted_paths = [path for _, path in f_names]

    origins = cppIO.read_float_matrix(folder_path + "/origins.bin")


    plotter = pv.Plotter()

    count = 0

    scale_factor = 1.5

    S = np.array([
        [scale_factor, 0, 0],
        [0, scale_factor, 0],
        [0, 0, scale_factor]
    ])

    for index, f in enumerate(sorted_paths):
              _, background, sdf = cppIO.read_3d_array_from_binary(sorted_paths[index])

              k_size = sdf.shape[0]

              # Create a UniformGrid for the SDF
              grid = pv.ImageData()
              grid.dimensions = sdf.shape
              grid.spacing = (1, 1, 1)  # spacing between voxels
              grid.origin = np.dot(np.asarray(origins[index]), S)
              grid.cell_data["sdf"] = sdf[:-1, :-1, :-1].flatten(order="F")  # Assign to cells

              # Threshold to get only inside voxels (sdf < 0)
              inside_voxels = grid.threshold(value=1, scalars="sdf", invert=True)

              try:
                  plotter.add_mesh(
                      inside_voxels,
                      show_edges=False,
                      scalars="sdf",  # Use the sdf values for coloring
                      cmap="coolwarm",  # Choose a colormap (optional, e.g., "viridis", "plasma", "coolwarm")
                      opacity=1.0,
                      clim=[-1, 1],
                  )
                  print(f"{sorted_paths[count]} added")

              except:
                  print(f"{sorted_paths[count]} has no inside voxels")

              count+=1

    # Optional: set initial camera position
    plotter.camera_position = 'xz'  # or 'xz', 'yz', etc.

    # Rotate the camera around the scene
    # plotter.camera.azimuth = 45  # rotate 45 degrees around z-axi
    plotter.camera.elevation = -60  # tilt camera

    plotter.show()
    '''
    n_frames = 100
    plotter.open_gif(r"../data/pyvista_outputs/sdf_spin.gif")  # Optional: save animation

    for i in range(n_frames):
        plotter.camera.azimuth += 360 / n_frames
        plotter.render()
        plotter.write_frame()  # Save each frame to GIF

    plotter.close()  # Only needed if you're writing to GIF
    
    '''

def read_test_type_maps():
    face_map_loc = r"C:\Local_Data\cropping_test\FaceTypeMap.bin"
    vert_map_loc = r"C:\Local_Data\cropping_test\VertTypeMap.bin"

    vert_map = cppIO.read_type_map_from_binary(vert_map_loc)
    face_map = cppIO.read_type_map_from_binary(face_map_loc)

    print()


def read_float_matrix_test():
    origin_matrix_loc = r"C:\Local_Data\cropping_test\origins.bin"
    origin_matrix = cppIO.load_float_matrix(origin_matrix_loc)

    print()



def main() -> None:
   read_dir_of_arrays_pyvista()

if __name__ == "__main__":
    main()

