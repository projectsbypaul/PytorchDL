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

from visualization.visu_scripts_voxel import visu_voxel_prediction_on_dir, visu_voxel_label_on_dir
from visualization.visu_scripts_mesh import visu_mesh_input_on_dir, visu_mesh_label_on_dir, visu_mesh_prediction_on_dir

def apply_scaled_camera(p, subplot_index, camera_mode="isometric", distance_scale=2.5):
    """Set camera orientation for one subplot, scaled to its bounding box."""
    p.subplot(0, subplot_index)

    # Get this subplot’s bounds
    bounds = p.bounds
    center = (
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2
    )
    max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

    # Orientation direction vectors (normalized later)
    CAMERA_PRESETS = {
        "isometric": np.array([1, 1, 1]),
        "front": np.array([0, -1, 0]),
        "top": np.array([0, 0, 1]),
        "side": np.array([1, 0, 0]),
        "custom": np.array([-1, -1, 1])
    }

    direction = CAMERA_PRESETS.get(camera_mode, CAMERA_PRESETS["isometric"])
    direction = direction / np.linalg.norm(direction)

    # Compute camera eye position
    eye = np.array(center) + direction * max_dim * distance_scale

    # Choose up vector intelligently
    up = (0, 0, 1) if not np.allclose(direction, [0, 0, 1]) else (0, 1, 0)

    # Apply
    p.camera_position = [tuple(eye), tuple(center), up]
    p.camera_set = True

def visu_input_prediction_mesh(data_loc, weights_loc, obj_loc, model_type, class_template,
                                    kernel_size, padding, n_classes,
                                    stride=1, surface_only=False):
    """Visualize voxel label vs. prediction side-by-side in one interactive Qt window."""

    input_mesh = visu_mesh_input_on_dir(obj_loc, render=False)

    pred_voxel, class_list, custom_colors = visu_voxel_prediction_on_dir(
        data_loc, weights_loc, model_type, class_template,
        kernel_size, padding, n_classes,
        stride=stride, surface_only=surface_only, render=False
    )

    pred_mesh, ftm_prediction, _, _ = visu_mesh_prediction_on_dir(
        data_loc, weights_loc, obj_loc,model_type, class_template,
        kernel_size, padding, n_classes,
        stride=stride, surface_only=surface_only, render=False
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
    p = BackgroundPlotter(shape=(1, 3), window_size=(1600, 800))

    ambient = 0.52
    diffuse = 0.4
    specular = 0.1

    # Left subplot — input
    p.subplot(0, 0)
    p.add_mesh(pred_mesh, color='lightgray', smooth_shading=True, edge_opacity=0.3, show_edges=True, lighting=True, ambient=ambient, diffuse=diffuse, specular=specular)
    p.add_text("Input Mesh", font_size=10)

    # middle subplot — voxel prediction
    p.subplot(0, 1)
    p.add_mesh(pred_voxel, scalars='rgba', rgba=True, lighting=True, ambient=ambient, diffuse=diffuse, specular=specular)
    p.add_text("Prediction Voxel", font_size=10)
    if legend_entries:
        p.add_legend(legend_entries, bcolor='white', face='circle',
                     size=(0.2, 0.25), loc='lower right')

    # Right subplot — prediction
    p.subplot(0, 2)
    p.add_mesh(pred_mesh, scalars='rgba', rgba=True, lighting=True, ambient=ambient, diffuse=diffuse, specular=specular)
    p.add_text("Prediction Mesh", font_size=10)
    if legend_entries:
        p.add_legend(legend_entries, bcolor='white', face='circle',
                     size=(0.2, 0.25), loc='lower right')

    for i in range(3):
        apply_scaled_camera(p, i, camera_mode="custom", distance_scale=3)

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
    # visu_mesh_input_on_dir(obj_loc, render=True)
    visu_input_prediction_mesh(data_loc, weights_loc, obj_loc, model_type, template, ks, pd, n_classes)




if __name__=="__main__":
    main()