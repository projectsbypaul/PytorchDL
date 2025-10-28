from networkx.readwrite.edgelist import write_weighted_edgelist

from visualization import color_templates
import torch
from utility.data_exchange import cppIOexcavator
import numpy as np
import os
import pyvista as pv
from visualization import visu_helpers
from pyvistaqt import BackgroundPlotter
from qtpy import QtWidgets
import sys

def visu_mesh_input_on_dir(obj_loc: str, render: bool = True):

    print("Loading input mesh:", obj_loc)
    mesh = pv.read(obj_loc)

    if render:
        print("PyVista version:", pv.__version__)
        p = pv.Plotter()
        p.add_mesh(
            mesh,
            color="lightgray",
            smooth_shading=True,
            show_edges=True,
            edge_color="black",
            line_width=0.5,
            edge_opacity=0.5
        )
        p.add_text("Input Mesh", font_size=10)
        p.enable_eye_dome_lighting()
        p.show()
    else:
        return mesh

def visu_mesh_prediction_on_dir(data_loc: str, weights_loc: str, obj_loc: str, model_type: str, class_template: str,   kernel_size: int, padding: int, n_classes: int,
                      stride: int = 1, surface_only: bool = False, render=True):
    # -----------------------------
    # 1) Load segments + metadata
    # -----------------------------
    data_arrays = cppIOexcavator.load_segments_from_binary(
        os.path.join(data_loc, "segmentation_data_segments.bin")
    )
    seg_info = cppIOexcavator.parse_dat_file(
        os.path.join(data_loc, "segmentation_data.dat")
    )

    model_input = torch.tensor(np.array(data_arrays)).unsqueeze(1)

    # -----------------------------
    # 2) Load model + predict
    # -----------------------------
    prediction = visu_helpers.__run_prediction_on_dir(
        data_loc, weights_loc, n_classes, model_type
    )

    # -----------------------------
    # 3) Assemble full label grid
    # -----------------------------
    origins = seg_info["ORIGIN_CONTAINER"]["data"]
    face_to_grid_index = seg_info["FACE_TO_GRID_INDEX_CONTAINER"]["data"]

    # Color/opacity template
    if class_template == "inside_outside":
        color_temp = color_templates.inside_outside_color_template_abc()
    elif class_template == "edge":
        color_temp = color_templates.edge_color_template_abc()
    else:
        raise NotImplementedError(f"Class Template '{class_template}' not implemented")

    class_list = color_templates.get_class_list(color_temp)
    index_to_class = color_templates.get_index_to_class_dict(color_temp)
    custom_colors = color_templates.get_color_dict(color_temp)
    custom_opacity = color_templates.get_opacity_dict(color_temp)
    class_to_idx = color_templates.get_class_to_index_dict(color_temp)

    print("assembling model outputs...")
    fill_idx = class_to_idx["Outside"]
    full_grid = visu_helpers.__assemble_grids_by_origin(
        prediction, origins, kernel_size, padding, fill_idx
    )

    # -----------------------------
    # 4) Assign face colors
    # -----------------------------
    origins_array = np.asarray(origins)
    bottom_coord = np.min(origins_array, axis=0)

    face_colors = []
    ftm_prediction = []

    for face_index in face_to_grid_index:
        grid_coord = face_index - bottom_coord
        gx, gy, gz = grid_coord.astype(int)
        face_class_index = full_grid[gx, gy, gz]
        face_class = index_to_class[int(face_class_index)]

        ftm_prediction.append(face_class)

        if face_class in custom_colors:
            rgb = custom_colors[face_class]
            a = float(custom_opacity.get(face_class, 1.0))
            face_colors.append((rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, a))
        else:
            face_colors.append((1.0, 1.0, 1.0, 1.0))

    face_colors = np.array(face_colors)

    # -----------------------------
    # 5) Load mesh and apply colors
    # -----------------------------
    mesh = pv.read(obj_loc)

    if len(face_colors) != mesh.n_faces:
        print(f"Warning: number of faces ({mesh.n_faces}) ≠ number of predicted colors ({len(face_colors)})")
        min_len = min(len(face_colors), mesh.n_faces)
        face_colors = face_colors[:min_len]

    mesh.cell_data["rgba"] = face_colors

    # -----------------------------
    # 6) Render or return
    # -----------------------------
    if render:
        print("PyVista version:", pv.__version__)
        p = pv.Plotter()
        p.add_mesh(
            mesh,
            scalars="rgba",
            rgba=True,
            lighting=False,
            culling="back",
            show_edges=False,
        )
        p.enable_eye_dome_lighting()

        legend_entries = [
            [name, tuple(c / 255 for c in custom_colors[name])]
            for name in class_list if name != "Outside"
        ]
        if legend_entries:
            p.add_legend(
                legend_entries, bcolor="white", face="circle",
                size=(0.2, 0.25), loc="lower right"
            )

        p.show()
    else:
        return mesh, ftm_prediction, class_list, custom_colors

def visu_mesh_label_on_dir(data_loc: str, obj_loc: str, class_template: str, kernel_size: int, padding: int, n_classes: int,
                      stride: int = 1, surface_only: bool = False, render=True):
    # -----------------------------
    # 1) Load segments + metadata
    # -----------------------------
    data_arrays = cppIOexcavator.load_segments_from_binary(
        os.path.join(data_loc, "segmentation_data_segments.bin")
    )
    seg_info = cppIOexcavator.parse_dat_file(
        os.path.join(data_loc, "segmentation_data.dat")
    )



    # -----------------------------
    # 2) Load model + predict
    # -----------------------------
    label = cppIOexcavator.load_labels_from_binary(
        os.path.join(data_loc, "segmentation_data_labels.bin")
    )

    # -----------------------------
    # 3) Assemble full label grid
    # -----------------------------
    origins = seg_info["ORIGIN_CONTAINER"]["data"]
    face_to_grid_index = seg_info["FACE_TO_GRID_INDEX_CONTAINER"]["data"]
    ftm_label = seg_info["FACE_TYPE_MAP"]

    # Color/opacity template
    if class_template == "inside_outside":
        color_temp = color_templates.inside_outside_color_template_abc()
    elif class_template == "edge":
        color_temp = color_templates.edge_color_template_abc()
    else:
        raise NotImplementedError(f"Class Template '{class_template}' not implemented")

    class_list = color_templates.get_class_list(color_temp)
    index_to_class = color_templates.get_index_to_class_dict(color_temp)
    custom_colors = color_templates.get_color_dict(color_temp)
    custom_opacity = color_templates.get_opacity_dict(color_temp)
    class_to_idx = color_templates.get_class_to_index_dict(color_temp)

    print("assembling model outputs...")
    fill_idx = class_to_idx["Outside"]
    full_grid = visu_helpers.__assemble_grids_by_origin(
        label, origins, kernel_size, padding, fill_idx
    )

    # -----------------------------
    # 4) Assign face colors
    # -----------------------------
    origins_array = np.asarray(origins)
    bottom_coord = np.min(origins_array, axis=0)

    face_colors = []
    ftm_label_list = [str(v) for v in ftm_label.values()]

    for face_class in ftm_label_list:
        if face_class in custom_colors:
            rgb = custom_colors[face_class]
            a = float(custom_opacity.get(face_class, 1.0))
            face_colors.append((rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, a))
        else:
            face_colors.append((1.0, 1.0, 1.0, 1.0))

    face_colors = np.array(face_colors)

    # -----------------------------
    # 5) Load mesh and apply colors
    # -----------------------------
    mesh = pv.read(obj_loc)

    if len(face_colors) != mesh.n_faces:
        print(f"Warning: number of faces ({mesh.n_faces}) ≠ number of predicted colors ({len(face_colors)})")
        min_len = min(len(face_colors), mesh.n_faces)
        face_colors = face_colors[:min_len]

    mesh.cell_data["rgba"] = face_colors

    # -----------------------------
    # 6) Render or return
    # -----------------------------
    if render:
        print("PyVista version:", pv.__version__)
        p = pv.Plotter()
        p.add_mesh(
            mesh,
            scalars="rgba",
            rgba=True,
            lighting=False,
            culling="back",
            show_edges=False,
        )
        p.enable_eye_dome_lighting()

        legend_entries = [
            [name, tuple(c / 255 for c in custom_colors[name])]
            for name in class_list if name != "Outside"
        ]
        if legend_entries:
            p.add_legend(
                legend_entries, bcolor="white", face="circle",
                size=(0.2, 0.25), loc="lower right"
            )

        p.show()
    else:
        return mesh, ftm_label_list, class_list, custom_colors

def visu_mesh_label_and_prediction(data_loc, weights_loc, obj_loc, model_type, class_template,
                                    kernel_size, padding, n_classes,
                                    stride=1, surface_only=False):
    """Visualize voxel label vs. prediction side-by-side in one interactive Qt window."""

    pred_mesh, ftm_prediction, class_list, custom_colors = visu_mesh_prediction_on_dir(
        data_loc, weights_loc, obj_loc,model_type, class_template,
        kernel_size, padding, n_classes,
        stride=stride, surface_only=surface_only, render=False
    )
    label_mesh, ftm_label, _, _ = visu_mesh_label_on_dir(
        data_loc, obj_loc, class_template, kernel_size, padding,
        n_classes, stride=stride, surface_only=surface_only, render=False
    )

    legend_entries = [
        [name, tuple(c / 255 for c in custom_colors[name])]
        for name in class_list if name != 'Outside'
    ]

    # Ensure a running QApplication
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    # Create PyVistaQt background plotter (tuple for window_size!)
    p = BackgroundPlotter(shape=(1, 2), window_size=(1600, 800))

    ambient = 0.52
    diffuse = 0.4
    specular = 0.1

    # Left subplot — ground truth
    p.subplot(0, 0)
    p.add_mesh(label_mesh, scalars='rgba', rgba=True, lighting=True, ambient=ambient, diffuse=diffuse, specular=specular)
    p.add_text("Ground Truth", font_size=10)
    if legend_entries:
        p.add_legend(legend_entries, bcolor='white', face='circle',
                     size=(0.2, 0.25), loc='lower right')

    # Right subplot — prediction
    p.subplot(0, 1)
    p.add_mesh(pred_mesh, scalars='rgba', rgba=True, lighting=True, ambient=ambient, diffuse=diffuse, specular=specular)
    p.add_text("Prediction", font_size=10)
    if legend_entries:
        p.add_legend(legend_entries, bcolor='white', face='circle',
                     size=(0.2, 0.25), loc='lower right')


    p.link_views()
    app.exec_()

def main():
    data_loc = r"H:\ws_label_test\label\00013045"
    obj_loc = r"H:\ws_label_test\source\00013045\00013045.obj"
    weights_loc = r"H:\ws_hpc_workloads\hpc_models\Balanced20k_Edge32_LRE-04\Balanced20k_Edge32_LRE-04_save_10.pth"
    template = "edge"
    model_type = "UNet_Hilbig"
    n_classes = 9
    ks = 32
    pd = 0

    # visu_mesh_prediction_on_dir(data_loc, weights_loc, obj_loc, model_type,template, ks, pd, n_classes)
    # visu_mesh_label_on_dir(data_loc, obj_loc, template, ks, pd, n_classes)
    visu_mesh_input_on_dir(obj_loc, render=True)
    visu_mesh_label_and_prediction(data_loc, weights_loc, obj_loc, model_type, template, ks, pd, n_classes)


if __name__=="__main__":
    main()