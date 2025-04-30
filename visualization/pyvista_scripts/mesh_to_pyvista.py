import numpy as np
import pyvista as pv
from utility.data_exchange import cppIO
from dl_torch.data_utility import DataParsing

def colored_mesh_pyvista():
    vtm_loc = r"C:\Local_Data\ABC\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2\00000004\VertTypeMap.bin"
    obj_loc = r"C:\Local_Data\ABC\ABC_parsed_files\00000004\00000004.obj"

    VertTypeMap = cppIO.read_type_map_from_binary(vtm_loc)
    vertices, faces = DataParsing.parse_obj(obj_loc)

    VertTypeMapWithEdges = []
    FaceTypeMap = []

    edge_count = 0

    for vert in VertTypeMap:
        if len(vert) == 1:
            VertTypeMapWithEdges.append(vert[0])
        else:
            VertTypeMapWithEdges.append("Edge")
            edge_count += 1

    print(f"added {edge_count} edges")

    for f in faces:
        temp_list = [
            VertTypeMapWithEdges[f[0]],
            VertTypeMapWithEdges[f[1]],
            VertTypeMapWithEdges[f[2]],
        ]

        unique_items = list(set(temp_list))

        if len(unique_items) == 1:
            FaceTypeMap.append(unique_items[0])
        else:
            FaceTypeMap.append("Edge")

    uft = list(set(FaceTypeMap))

    # Define RGB (0–255) for all classes
    custom_colors = {
        'Cone': (0, 0, 255),      # blue
        'Cylinder': (255, 0, 0),  # red
        'Edge': (255, 255, 0),    # yellow
        'Plane': (255, 192, 203), # pink
        'Sphere': (128, 0, 0),    # dark red
        'Torus': (0, 255, 255),   # cyan
    }

    custom_opacity = {
        'Cone': 1.0,
        'Cylinder': 1.0,
        'Edge': 1.0,
        'Plane': 1.0,
        'Sphere': 1.0,
        'Torus': 1.0,
    }

    ### --- NOW: Visualize with PyVista --- ###

    # Create PyVista mesh
    vertices = np.array(vertices)
    faces_flat = np.hstack([np.full((len(faces), 1), 3), np.array(faces)])  # 3 indicates triangles
    mesh = pv.PolyData(vertices, faces_flat)

    # Assign face colors
    face_colors = []
    for face_type in FaceTypeMap:
        color = custom_colors.get(face_type, (128, 128, 128))  # Default gray if missing
        face_colors.append(color)

    face_colors = np.array(face_colors) / 255.0  # Normalize to [0,1] for PyVista

    # Create plotter
    plotter = pv.Plotter()
    mesh.cell_data['colors'] = face_colors  # Add colors per face
    plotter.add_mesh(mesh, scalars='colors', rgb=True, show_edges=True)

    # Add a legend
    legend_entries = []
    for label, rgb in custom_colors.items():
        color_normalized = tuple(c / 255.0 for c in rgb)
        legend_entries.append([label, color_normalized])

    plotter.add_legend(legend_entries, bcolor=None, border=True, size=(0.25, 0.3))

    plotter.show()

def main():
    colored_mesh_pyvista()

if __name__ == "__main__":
    main()