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

def visu_compared_views_geom(data_locs, weights_loc, model_type, class_template,
                             kernel_size, padding, n_classes,
                             stride=1, surface_only=False):

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    p = BackgroundPlotter(shape=(1, len(data_locs)), window_size=(1600, 800))

    ambient, diffuse, specular = 0.52, 0.4, 0.1

    for i, data_loc in enumerate(data_locs):
        pred_voxel, class_list, custom_colors = visu_voxel_prediction_on_dir(
            data_loc, weights_loc, model_type, class_template,
            kernel_size, padding, n_classes,
            stride=stride, surface_only=surface_only, render=False
        )

        # --- force opaque voxels for comparison ---
        if 'rgba' in pred_voxel.array_names:
            rgba = pred_voxel['rgba'].copy()
            rgba[:, 3] = 255
            pred_voxel['rgba'] = rgba
        # ------------------------------------------

        legend_entries = [
            [name, tuple(c / 255 for c in custom_colors[name])]
            for name in class_list if name != 'Outside'
        ]

        plot_title = os.path.basename(data_loc)

        p.subplot(0, i)
        p.add_mesh(pred_voxel, scalars='rgba', rgba=True, lighting=True,
                   ambient=ambient, diffuse=diffuse, specular=specular)
        p.add_text(plot_title, font_size=10)

        if legend_entries:
            p.add_legend(legend_entries, bcolor='white', face='circle',
                         size=(0.2, 0.25), loc='lower right')

    p.link_views()
    app.exec_()

def visu_compared_views_model(data_loc, weights_locs, model_types, class_templates,
                              kernel_size, padding, n_classes,
                              stride=1, surface_only=False):

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    p = BackgroundPlotter(shape=(1, len(weights_locs)), window_size=(1600, 800))

    ambient, diffuse, specular = 0.52, 0.4, 0.1

    for i, weights_loc in enumerate(weights_locs):
        model_type = model_types[i]
        class_template = class_templates[i]

        pred_voxel, class_list, custom_colors = visu_voxel_prediction_on_dir(
            data_loc, weights_loc, model_type, class_template,
            kernel_size, padding, n_classes,
            stride=stride, surface_only=surface_only, render=False
        )

        # --- force opaque voxels for comparison ---
        if 'rgba' in pred_voxel.array_names:
            rgba = pred_voxel['rgba'].copy()
            rgba[:, 3] = 255
            pred_voxel['rgba'] = rgba
        # ------------------------------------------

        legend_entries = [
            [name, tuple(c / 255 for c in custom_colors[name])]
            for name in class_list if name != 'Outside'
        ]

        plot_title = os.path.basename(weights_loc).split('.')[0]

        p.subplot(0, i)
        p.add_mesh(pred_voxel, scalars='rgba', rgba=True, lighting=True,
                   ambient=ambient, diffuse=diffuse, specular=specular)
        p.add_text(plot_title, font_size=10)

        if legend_entries:
            p.add_legend(legend_entries, bcolor='white', face='circle',
                         size=(0.2, 0.25), loc='lower right')

    p.link_views()
    app.exec_()


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

    bounds = p.bounds
    center = (
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2
    )
    max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

    CAMERA_PRESETS = {
        "isometric": np.array([1, 1, 1]),
        "front": np.array([0, -1, 0]),
        "top": np.array([0, 0, 1]),
        "side": np.array([1, 0, 0]),
        "custom": np.array([-1, -1, 1])
    }

    direction = CAMERA_PRESETS.get(camera_mode, CAMERA_PRESETS["isometric"])
    direction = direction / np.linalg.norm(direction)

    eye = np.array(center) + direction * max_dim * distance_scale

    up = (0, 0, 1) if not np.allclose(direction, [0, 0, 1]) else (0, 1, 0)

    p.camera_position = [tuple(eye), tuple(center), up]
    p.camera_set = True



def visu_input_prediction_mesh(data_loc, weights_loc, obj_loc, model_type, class_template,
                               kernel_size, padding, n_classes,
                               stride=1, surface_only=False):

    """Visualize input mesh, voxel prediction, and prediction mesh side-by-side."""

    # Load input mesh
    input_mesh = visu_mesh_input_on_dir(obj_loc, render=False)

    # Load voxel prediction
    pred_voxel, class_list, custom_colors = visu_voxel_prediction_on_dir(
        data_loc, weights_loc, model_type, class_template,
        kernel_size, padding, n_classes,
        stride=stride, surface_only=surface_only, render=False
    )

    # Load mesh prediction
    pred_mesh, ftm_prediction, _, _ = visu_mesh_prediction_on_dir(
        data_loc, weights_loc, obj_loc, model_type, class_template,
        kernel_size, padding, n_classes,
        stride=stride, surface_only=surface_only, render=False
    )

    input_mesh_copy = input_mesh.copy(deep=True)
    vox_copy = pred_voxel.copy(deep=True)
    mesh_copy = pred_mesh.copy(deep=True)

    # ---------- FORCE OPAQUE RGBA ----------
    def force_opaque_rgba_voxel(obj):
        if "rgba" in obj.array_names:
            rgba = obj["rgba"].astype(np.float32)
            rgba[:, 3] = 255.0
            rgba /= 255.0
            obj["rgba"] = rgba

    force_opaque_rgba_voxel(vox_copy)
    # ----------------------------------------

    legend_entries = [
        [name, tuple(c / 255 for c in custom_colors[name])]
        for name in class_list if name != 'Outside'
    ]

    # Ensure QApplication exists
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    p = BackgroundPlotter(shape=(1, 3), window_size=(1800, 800))

    ambient = 0.52
    diffuse = 0.4
    specular = 0.1

    # -------------------------------------------
    # LEFT SUBPLOT — Input mesh
    # -------------------------------------------
    p.subplot(0, 0)
    p.add_mesh(
        input_mesh_copy,
        color='lightgray',
        smooth_shading=True,
        edge_opacity=0.3,
        show_edges=True,
        lighting=True,
        ambient=ambient, diffuse=diffuse, specular=specular
    )
    p.reset_camera()     # <<< FIX BLACK VIEW
    p.add_text("Input Mesh", font_size=10)

    # -------------------------------------------
    # MIDDLE SUBPLOT — Voxel prediction
    # -------------------------------------------
    p.subplot(0, 1)
    p.add_mesh(
        vox_copy,
        scalars="rgba",
        rgba=True,
        lighting=True,
        ambient=ambient, diffuse=diffuse, specular=specular
    )
    p.reset_camera()     # <<< FIX BLACK VIEW
    p.add_text("Voxel Prediction", font_size=10)

    # -------------------------------------------
    # RIGHT SUBPLOT — Mesh prediction
    # -------------------------------------------

    p.subplot(0, 2)
    p.add_mesh(
        mesh_copy,
        scalars=mesh_copy["rgba"][:, :3],  # RGB only
        rgb=True,  # DO NOT use rgba=True
        opacity=1.0,  # Force full opacity
        smooth_shading=False,
        show_edges=False
    )
    p.reset_camera()     # <<< FIX BLACK VIEW
    p.add_text("Prediction Mesh", font_size=10)

    # SINGLE LEGEND (optional)
    if legend_entries:
        p.add_legend(
            legend_entries, bcolor='white', face='circle',
            size=(0.2, 0.25), loc='lower right'
        )

    p.link_views()
    app.exec_()


def main():
    ''''''

    f_name = "REBeleg_Refined"
    data_loc = rf"H:\ws_seg_test\debug_output\{f_name}"
    obj_loc = rf"H:\ws_seg_test\source\{f_name}.obj"
    weights_loc = r"H:\ws_hpc_workloads\hpc_models\EdgeMCFB_00_UNet_16EL_3f9_crp20000_LR0f0001_DC0f03\EdgeMCFB_00_UNet_16EL_3f9_crp20000_LR0f0001_DC0f03_save_135.pth"
    template = "primitive_edge"
    model_type = "UNet_16EL"
    n_classes = 8
    ks = 16
    pd = 8

    # visu_mesh_prediction_on_dir(data_loc, weights_loc, obj_loc, model_type,template, ks, pd, n_classes)
    # visu_mesh_label_on_dir(data_loc, obj_loc, template, ks, pd, n_classes)
    # visu_mesh_input_on_dir(obj_loc, render=True)
    visu_input_prediction_mesh(data_loc, weights_loc, obj_loc, model_type, template, ks, pd, n_classes)

    '''  
    weights_loc_0 = r"H:\ws_hpc_workloads\hpc_models\SegDemoEdge_32\SegDemoEdge_32_save_50.pth"
    weights_loc_1 = r"H:\ws_hpc_workloads\hpc_models\Balanced20k_Edge32_LRE-05\Balanced20k_Edge32_LRE-05_save_50.pth"
    weights_loc_2 = r"H:\ws_hpc_workloads\hpc_models\mfcb_Edge_01_UNet3D_Hilbig_crp10000\mfcb_Edge_01_UNet3D_Hilbig_crp10000_save_50.pth"

    templates = ["edge", "edge", "edge"]
    models = ["UNet_Hilbig", "UNet_Hilbig", "UNet_Hilbig"]

    data_loc = r"H:\ws_seg_test\debug_output\REBeleg_Refined"
    n_classes = 9
    ks = 32
    pd = 16

    visu_compared_views_model(
        data_loc, [weights_loc_0, weights_loc_1, weights_loc_2], models, templates, ks, pd, n_classes
    )

    '''

    '''
    weights_loc = r"H:\ws_hpc_workloads\hpc_models\GN_Test_UNet_16EL_1f0_crp20000_LR0f0001_DC0f03\GN_Test_UNet_16EL_1f0_crp20000_LR0f0001_DC0f03_save_30.pth"
    n_classes = 7
    pd = 8
    ks = 16
    template = "primitive"
    model = "UNet_16EL"

    data_0 = r"H:\ws_seg_test\debug_output\rot_test_x_up"
    data_1 = r"H:\ws_seg_test\debug_output\rot_test_z_up"
    data_2 = r"H:\ws_seg_test\debug_output\rot_test_odd_up"

    visu_compared_views_geom([data_0, data_1, data_2], weights_loc, model, template, ks, pd, n_classes)

    '''

MAIN = main()

if __name__=="__main__":
    pass