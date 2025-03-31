import pyvista as pv
import numpy as np
from dl_torch.data_utility import DataParsing
import random
from matplotlib.colors import ListedColormap


def test_pyvista():
    obj_loc = (r"C:\Local_Data\ABC\obj\abc_meta_files"
               r"\abc_0000_obj_v00\00000002\00000002_1ffb81a71e5b402e966b9341_trimesh_001.obj")

    yml_loc = (r"C:\Local_Data\ABC\feat\abc_meta_files"
               r"\abc_0000_feat_v00\00000002\00000002_1ffb81a71e5b402e966b9341_features_001.yml")

    vertices, faces = DataParsing.parse_obj(obj_loc)

    data = DataParsing.parse_yaml(yml_loc)

    surface_metadata = data["surfaces"]

    face_type_list = [""] * len(faces)

    for surface in surface_metadata:

        faces_of_surf = surface["face_indices"]
        surf_type = surface["type"]

        for face_index in faces_of_surf:
            face_type_list[face_index] = surf_type

    # Create mapping for unique types
    unique_types = sorted(set(face_type_list))
    type_to_id = {typ: i for i, typ in enumerate(unique_types)}
    face_type_ids = np.array([type_to_id[t] for t in face_type_list])

    # Prepare mesh
    pvy_vertices = np.array(vertices)
    pvy_faces = np.hstack([[3] + list(f) for f in faces])
    mesh = pv.PolyData(pvy_vertices, pvy_faces)
    mesh.cell_data['surface_type'] = face_type_ids

    c1 = np.array([12 / 256, 238 / 256, 246 / 256, 1.0])
    c2 = np.array([148 / 256, 0 / 256, 211 / 256, 1.0])
    c3 = np.array([0 / 256, 128 / 256, 0 / 256, 1.0])
    c4 = np.array([255 / 256, 247 / 256, 0 / 256, 1.0])
    c5 = np.array([1.0, 0.0, 0.0, 1.0])

    rgba_colors = [c1, c2, c3, c4, c5]

    colormap = ListedColormap(rgba_colors)

    # Manually add colored squares using rectangles
    plotter = pv.Plotter()

    actor = plotter.add_mesh(
        mesh,
        scalars='surface_type',
        cmap=colormap,
        show_edges=True,
        show_scalar_bar=False,
    )

    # Overwrite legend with black text and custom colored squares

    """
       plotter.add_legend(
        labels=[
            [typ, tuple(color[:3])] for typ, color in zip(unique_types, rgba_colors)
        ],
        bcolor='white',
        face='rectangle',  # shows colored box
        border=True,
        size=(0.2, 0.3)
    )

    """

    plotter.show()




    print()


def main():
    test_pyvista()

if __name__ =="__main__":
    main()