import pyvista as pv
import numpy as np
from dl_torch.data_utility import DataParsing
from utility.data_exchange import cppIO
import random
from matplotlib.colors import ListedColormap


def test_draw_from_yml():
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

    # Define custom colors (RGB 0-255) and opacity
    custom_colors = {
        'Cone': (0, 0, 255),  # blue
        'Cylinder': (255, 0, 0),  # red
        'Edge': (255, 255, 0),  # yellow
        'Plane': (255, 192, 203),  # pink
        'Sphere': (128, 0, 0),  # dark red
        'Torus': (0, 255, 255),  # cyan
        'BSpline': (100, 0, 100)  # magenta
    }
    custom_opacity = {
        'Cone': 1.0,
        'Cylinder': 1.0,
        'Edge': 1.0,
        'Plane': 1.0,
        'Sphere': 1.0,
        'Torus': 1.0,
        'BSpline': 1.0,
    }

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


def test_draw_from_bin():

    obj_loc = r"C:\Local_Data\ABC\ABC_parsed_files\00000066\00000066.obj"
    face_type_map_loc = r"C:\Local_Data\ABC\ABC_Testing\FaceTypeMap.bin"

    type_counts = cppIO.read_type_counts_from_binary(r"C:\Local_Data\ABC\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2\00000002\TypeCounts.bin")
    print(type_counts)

    face_type_map = cppIO.read_type_map_from_binary(face_type_map_loc)
    face_type_map = [inner[0] for inner in face_type_map]  # flatten

    vertices, faces = DataParsing.parse_obj(obj_loc)

    # Define custom colors (RGB 0-255) and opacity
    custom_colors = {
        'Cone': (0, 0, 255),  # blue
        'Cylinder': (255, 0, 0),  # red
        'Edge': (255, 255, 0),  # yellow
        'Plane': (255, 192, 203),  # pink
        'Sphere': (128, 0, 0),  # dark red
        'Torus': (0, 255, 255),  # cyan
        'BSpline': (100, 0, 100) # magenta
    }
    custom_opacity = {
        'Cone': 1.0,
        'Cylinder': 1.0,
        'Edge': 1.0,
        'Plane': 1.0,
        'Sphere': 1.0,
        'Torus': 1.0,
        'BSpline': 1.0,
    }

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
    # Add a legend
    legend_entries = []
    for label, rgb in custom_colors.items():
        color_normalized = tuple(c / 255.0 for c in rgb)
        legend_entries.append([label, color_normalized])

    plotter.add_legend(legend_entries, bcolor=None, border=True, size=(0.25, 0.3))
    plotter.show()


def main():
    # test_draw_from_yml()
    test_draw_from_bin()

if __name__ =="__main__":
    main()