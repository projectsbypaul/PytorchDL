import pyvista as pv
import numpy as np
from dl_torch.data_utility import DataParsing
from utility.data_exchange import cppIO
from utility.data_exchange import cppIOexcavator
import os
import random
from matplotlib.colors import ListedColormap
from dl_torch.data_utility import HelperFunctionsABC
from visualization import color_templates


def draw_from_yml():
    id = "00000002"

    obj_loc = f"C:/Local_Data/ABC/ABC_parsed_files/{id}/{id}.obj"
    yml_loc = f"C:/Local_Data/ABC/ABC_parsed_files/{id}/{id}.yml"

    vertices, faces = DataParsing.parse_obj(obj_loc)
    data = DataParsing.parse_yaml(yml_loc)
    surface_metadata = data["surfaces"]

    # Create face type list
    face_type_list = [""] * len(faces)
    for surface in surface_metadata:
        faces_of_surf = surface["face_indices"]
        surf_type = surface["type"]
        for face_index in faces_of_surf:
            face_type_list[face_index] = surf_type

    # set up colors
    color_temp = color_templates.default_color_template_abc()

    custom_colors = color_templates.get_color_dict(color_temp)
    custom_opacity = color_templates.get_opacity_dict(color_temp)

    # Map face types to RGBA colors
    face_colors = []
    for face_type in face_type_list:
        rgb = custom_colors.get(face_type, (128, 128, 128))  # fallback grey
        opacity = custom_opacity.get(face_type, 1.0)         # fallback 1.0
        rgba = [c / 255 for c in rgb] + [opacity]
        face_colors.append(rgba)
    face_colors = np.array(face_colors)

    # Prepare the mesh
    pvy_vertices = np.array(vertices)
    pvy_faces = np.hstack([[3] + list(f) for f in faces])
    mesh = pv.PolyData(pvy_vertices, pvy_faces)

    # Add face colors directly
    mesh.cell_data['colors'] = face_colors

    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars='colors',
        rgba=True,
        show_edges=True,
        show_scalar_bar=False,
    )

    # ------ Legend creation ------
    legend_entries = []
    for label, rgb in custom_colors.items():
        color_normalized = tuple(c / 255.0 for c in rgb)
        legend_entries.append([label, color_normalized])

    plotter.add_legend(
        legend_entries,
        bcolor=None,
        border=True,
        size=(0.25, 0.3)
    )

    plotter.show()


def draw_from_bin():
    id = "00000002"

    obj_loc = f"C:/Local_Data/ABC/ABC_parsed_files/{id}/{id}.obj"

    face_type_map_loc = (f"C:\\Local_Data\\ABC\\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2"
                         f"\\{id}\\FaceTypeMap.bin")

    face_type_map = cppIO.read_type_map_from_binary(face_type_map_loc)
    face_type_map = [inner[0] for inner in face_type_map]  # flatten

    vertices, faces = DataParsing.parse_obj(obj_loc)

    #set up colors
    color_temp = color_templates.default_color_template_abc()

    custom_colors = color_templates.get_color_dict(color_temp)
    custom_opacity = color_templates.get_opacity_dict(color_temp)

    # Map face types to RGBA colors
    face_colors = []
    for face_type in face_type_map:
        rgb = custom_colors.get(face_type, (128, 128, 128))  # fallback grey
        opacity = custom_opacity.get(face_type, 1.0)  # fallback 1.0
        rgba = [c / 255 for c in rgb] + [opacity]
        face_colors.append(rgba)

    face_colors = np.array(face_colors)

    # Prepare the mesh
    pvy_vertices = np.array(vertices)
    pvy_faces = np.hstack([[3] + list(f) for f in faces])
    mesh = pv.PolyData(pvy_vertices, pvy_faces)

    # Add face colors directly
    mesh.cell_data['colors'] = face_colors

    # Plot
    plotter = pv.Plotter()

    plotter.add_mesh(
        mesh,
        scalars='colors',
        rgba=True,
        show_edges=True,
        show_scalar_bar=False,
    )

    # ------ Legend creation ------
    legend_entries = []
    for label, rgb in custom_colors.items():
        color_normalized = tuple(c / 255.0 for c in rgb)
        legend_entries.append([label, color_normalized])

    plotter.add_legend(
        legend_entries,
        bcolor=None,
        border=True,
        size=(0.25, 0.3)
    )

    plotter.show()

def draw_labels_from_bin():

    id = "00010084"

    target_dir = f"H:\ABC\ABC_Datasets\Segmentation\ABC_Chunk_01\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2/{id}"

    bin_array_file = target_dir + "/segmentation_data_segments.bin"
    segment_info_file = target_dir + "/segmentation_data.dat"

    segment_data = cppIOexcavator.parse_dat_file(segment_info_file)

    origins = segment_data["ORIGIN_CONTAINER"]["data"]
    face_type_map =  np.array(list(segment_data["FACE_TYPE_MAP"].values()))
    face_to_index_map = segment_data["FACE_TO_GRID_INDEX_CONTAINER"]["data"]
    uniques = segment_data['TYPE_COUNT_MAP']

    bin_arrays = cppIOexcavator.load_segments_from_binary(bin_array_file)

    # Set up dictionary
    class_list = np.array(
        ['Cone', 'Revolution', 'Sphere', 'Plane', 'Extrusion', 'Other', 'Cylinder', 'Torus', 'BSpline', 'Void'])

    class_indices = np.arange(len(class_list))
    class_lot = dict(zip(class_list, class_indices))
    index_lot =  dict(zip(class_indices, class_list,))

    labels = []

    for grid_index, grid in enumerate(bin_arrays):

        grid_dim = grid.shape[0]

        origin = np.asarray(origins[grid_index])

        top = origin + [grid_dim - 1, grid_dim - 1, grid_dim - 1]

        l = np.zeros([grid_dim, grid_dim, grid_dim, class_list.shape[0]])

        write_count = 0

        df_voxel_count = dict()

        for index, surf_type in enumerate(class_list):
            df_voxel_count.update({surf_type: 0})

        for face_index, face_center in enumerate(face_to_index_map):

            if origin[0] <= face_center[0] <= top[0] and origin[1] <= face_center[1] <= top[1] and origin[2] <= \
                    face_center[2] <= \
                    top[2]:

                grid_index = face_center - origin

                type_string = face_type_map[face_index]
                one_hot_index = class_lot[type_string]
                l[int(grid_index[0]), int(grid_index[1]), int(grid_index[2]), one_hot_index] += 1

        print(f"wrote {write_count} labels")

        for i, j, k in np.ndindex(l.shape[0],l.shape[1],l.shape[2]):
            voxel = l[i, j, k, :]

            if np.sum(voxel) > 0:
                max_index = np.argmax(voxel)
                l[i, j, k, :] = np.zeros_like(voxel)
                l[i, j, k, max_index] = 1
                # df_voxel_count[index_lot[max_index]] += 1
            else:
                l[i, j, k, class_lot["Void"]] = 1
                # df_voxel_count['Void'] += 1

        # print(df_voxel_count.keys())
        # print(df_voxel_count.values())

        labels.append(l)

    color_temp = color_templates.default_color_template_abc()

    custom_colors = color_templates.get_color_dict(color_temp)
    custom_opacity = color_templates.get_opacity_dict(color_temp)

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

def draw_sdf_from_bin():
    id = "00010084"

    target_dir = f"H:\ABC\ABC_Datasets\Segmentation\ABC_Chunk_01\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2/{id}"

    bin_array_file = target_dir + "/segmentation_data_segments.bin"
    segment_info_file = target_dir + "/segmentation_data.dat"

    segment_data = cppIOexcavator.parse_dat_file(segment_info_file)

    origins = segment_data["ORIGIN_CONTAINER"]["data"]
    sdf_grid = cppIOexcavator.load_segments_from_binary(bin_array_file)

    count = 0

    scale_factor = 1.5

    S = np.array([
        [scale_factor, 0, 0],
        [0, scale_factor, 0],
        [0, 0, scale_factor]
    ])

    plotter = pv.Plotter()

    for index, sdf in enumerate(sdf_grid):

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
                clim=[0, 1],
            )
            print(f"gird {index} added")

        except:
            print(f"grid {index} has no inside voxels")

        count += 1

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


def main():
   # draw_labels_from_bin()
   draw_sdf_from_bin()

if __name__ =="__main__":
    main()