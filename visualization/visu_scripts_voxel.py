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
import h5py

def visu_voxel_prediction_from_h5(h5_file: str, class_template: str,   kernel_size: int, padding: int, n_classes: int,
                      stride: int = 1, surface_only=True, render=True):

    with h5py.File(h5_file, "r") as f:
        full_grid = f["flat_predictions"][:]  # read entire dataset into memory

    if class_template == "inside_outside":
        color_temp = color_templates.inside_outside_color_template_abc()
    elif class_template == "edge":
        color_temp = color_templates.edge_color_template_abc()
    else:
        raise NotImplementedError(f"Class Template '{class_template}' not implemented")

    class_list = color_templates.get_class_list(color_temp)  # ordered names -> indices
    custom_colors = color_templates.get_color_dict(color_temp)  # {name: (R,G,B)}
    custom_opacity = color_templates.get_opacity_dict(color_temp)  # {name: alpha}
    class_to_idx = color_templates.get_class_to_index_dict(color_temp)

    print("assembling model outputs...")
    fill_idx = class_to_idx["Outside"]

    # -----------------------------
    # 4) Build draw set: non-Inside/Outside voxels
    # -----------------------------
    labels = full_grid.astype(np.int32, copy=False)

    hidden = {'Outside'}
    visible_class_mask = np.array([name not in hidden for name in class_list], dtype=bool)
    keep = visible_class_mask[labels]  # True where voxel should be drawn

    if not np.any(keep):
        print("No voxels to render (all are Inside/Outside).")
        return

    if surface_only:
        # Keep only boundary voxels of the visible set (optional toggle)
        vm = keep
        pad = np.pad(vm, 1, constant_values=False)
        fully_surrounded = (
                pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1] &
                pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1] &
                pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
        )
        keep = vm & ~fully_surrounded
        if not np.any(keep):
            print("No surface voxels remain after filtering.")
            return

    coords = np.argwhere(keep)  # (K, 3)
    kept_labels = labels[keep]

    if stride > 1:
        coords = coords[::stride]
        kept_labels = kept_labels[::stride]

    # -----------------------------
    # 5) Colors (per-voxel RGBA, uint8)
    # -----------------------------
    lut_rgba = np.zeros((len(class_list), 4), dtype=np.uint8)
    for idx, name in enumerate(class_list):
        r, g, b = custom_colors[name]
        a = 0.0 if name in hidden else float(custom_opacity.get(name, 1.0))
        lut_rgba[idx] = (r, g, b, int(round(a * 255)))

    colors_rgba = lut_rgba[kept_labels]  # (K, 4) uint8

    # -----------------------------
    # 6) Glyph render (one actor)
    # -----------------------------
    points = coords.astype(np.float32)  # voxel centers at integer coords
    # points = [tpl * vs for tpl in points]
    cloud = pv.PolyData(points)
    cloud['rgba'] = colors_rgba

    cube = pv.Cube(center=(0, 0, 0), x_length=1.0, y_length=1.0, z_length=1.0)
    glyphs = cloud.glyph(geom=cube, scale=False, orient=False)

    if render:
        print("PyVista version:", pv.__version__)
        p = pv.Plotter()
        p.add_mesh(
            glyphs,
            scalars='rgba',
            rgba=True,
            lighting=False,  # flat label colors
            culling='back',  # reduce overdraw
            show_edges=False,
            render_lines_as_tubes=False,
        )
        p.enable_eye_dome_lighting()

        # Legend for visible classes only
        legend_entries = [[name, tuple(c / 255 for c in custom_colors[name])]
                          for name in class_list if name not in hidden]
        if legend_entries:
            p.add_legend(legend_entries, bcolor='white', face='circle',
                         size=(0.2, 0.25), loc='lower right')

        p.show()
    else:
        return glyphs, class_list, custom_colors

def visu_voxel_prediction_on_dir(data_loc: str, weights_loc: str, model_type: str, class_template: str,   kernel_size: int, padding: int, n_classes: int,
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

    # Model input: (N,1,ks,ks,ks)
    model_input = torch.tensor(np.array(data_arrays)).unsqueeze(1)

    # -----------------------------
    # 2) Load model + predict
    # -----------------------------
    prediction = visu_helpers.__run_prediction_on_dir(data_loc, weights_loc, n_classes, model_type)

    # -----------------------------
    # 3) Assemble full label grid
    # -----------------------------
    origins = seg_info["ORIGIN_CONTAINER"]["data"]  # list of (x,y,z)
    vs = seg_info["SCALARS"]["voxel_size"]
    # Color/opacity template
    if class_template == "inside_outside":
        color_temp = color_templates.inside_outside_color_template_abc()
    elif class_template == "edge":
        color_temp = color_templates.edge_color_template_abc()
    else:
        raise NotImplementedError(f"Class Template '{class_template}' not implemented")

    class_list     = color_templates.get_class_list(color_temp)          # ordered names -> indices
    custom_colors  = color_templates.get_color_dict(color_temp)          # {name: (R,G,B)}
    custom_opacity = color_templates.get_opacity_dict(color_temp)        # {name: alpha}
    class_to_idx   =  color_templates.get_class_to_index_dict(color_temp)

    print("assembling model outputs...")
    fill_idx = class_to_idx["Outside"]
    full_grid = visu_helpers.__assemble_grids_by_origin(prediction, origins, kernel_size, padding, fill_idx)


    # -----------------------------
    # 4) Build draw set: non-Inside/Outside voxels
    # -----------------------------
    labels = full_grid.astype(np.int32, copy=False)

    hidden = {'Outside'}
    visible_class_mask = np.array([name not in hidden for name in class_list], dtype=bool)
    keep = visible_class_mask[labels]   # True where voxel should be drawn

    if not np.any(keep):
        print("No voxels to render (all are Inside/Outside).")
        return

    if surface_only:
        # Keep only boundary voxels of the visible set (optional toggle)
        vm  = keep
        pad = np.pad(vm, 1, constant_values=False)
        fully_surrounded = (
            pad[2:,1:-1,1:-1] & pad[:-2,1:-1,1:-1] &
            pad[1:-1,2:,1:-1] & pad[1:-1,:-2,1:-1] &
            pad[1:-1,1:-1,2:] & pad[1:-1,1:-1,:-2]
        )
        keep = vm & ~fully_surrounded
        if not np.any(keep):
            print("No surface voxels remain after filtering.")
            return

    coords = np.argwhere(keep)   # (K, 3)
    kept_labels  = labels[keep]

    if stride > 1:
        coords = coords[::stride]
        kept_labels = kept_labels[::stride]

    # -----------------------------
    # 5) Colors (per-voxel RGBA, uint8)
    # -----------------------------
    lut_rgba = np.zeros((len(class_list), 4), dtype=np.uint8)
    for idx, name in enumerate(class_list):
        r, g, b = custom_colors[name]
        a = 0.0 if name in hidden else float(custom_opacity.get(name, 1.0))
        lut_rgba[idx] = (r, g, b, int(round(a * 255)))

    colors_rgba = lut_rgba[kept_labels]  # (K, 4) uint8

    # -----------------------------
    # 6) Glyph render (one actor)
    # -----------------------------
    points = coords.astype(np.float32)   # voxel centers at integer coords
    points = [tpl*vs for tpl in points]
    cloud  = pv.PolyData(points)
    cloud['rgba'] = colors_rgba

    cube   = pv.Cube(center=(0, 0, 0), x_length=1.0*vs, y_length=1.0*vs, z_length=1.0*vs)
    glyphs = cloud.glyph(geom=cube, scale=False, orient=False)

    if render:
        print("PyVista version:", pv.__version__)
        p = pv.Plotter()
        p.add_mesh(
            glyphs,
            scalars='rgba',
            rgba=True,
            lighting=False,  # flat label colors
            culling='back',  # reduce overdraw
            show_edges=False,
            render_lines_as_tubes=False,
        )
        p.enable_eye_dome_lighting()

        # Legend for visible classes only
        legend_entries = [[name, tuple(c / 255 for c in custom_colors[name])]
                          for name in class_list if name not in hidden]
        if legend_entries:
            p.add_legend(legend_entries, bcolor='white', face='circle',
                         size=(0.2, 0.25), loc='lower right')

        p.show()
    else:
        return glyphs, class_list, custom_colors

def visu_voxel_label_on_dir(data_loc: str, kernel_size: int, padding: int, class_template: str, n_classes: int,
                              stride: int = 1, surface_only: bool = False, render=True):

        # -----------------------------
        # 1) Load segments + metadata
        # -----------------------------
        label_arrays = cppIOexcavator.load_labels_from_binary(
            os.path.join(data_loc, "segmentation_data_labels.bin")
        )
        seg_info = cppIOexcavator.parse_dat_file(
            os.path.join(data_loc, "segmentation_data.dat")
        )

        # 3) Assemble full label grid
        origins = seg_info["ORIGIN_CONTAINER"]["data"]  # should be (N, 3)
        vs = seg_info["SCALARS"]["voxel_size"]
        # Color/opacity template
        if class_template == "inside_outside":
            color_temp = color_templates.inside_outside_color_template_abc()
        elif class_template == "edge":
            color_temp = color_templates.edge_color_template_abc()
        else:
            raise NotImplementedError(f"Class Template '{class_template}' not implemented")

        class_list = color_templates.get_class_list(color_temp)  # ordered names -> indices
        custom_colors = color_templates.get_color_dict(color_temp)  # {name: (R,G,B)}
        custom_opacity = color_templates.get_opacity_dict(color_temp)  # {name: alpha}
        class_to_idx = color_templates.get_class_to_index_dict(color_temp)

        print("assembling model outputs...")
        fill_idx = class_to_idx["Outside"]
        full_grid = visu_helpers.__assemble_grids_by_origin(label_arrays, origins, kernel_size, padding, fill_idx)

        labels = full_grid.astype(np.int32, copy=False)

        hidden = {'Outside'}
        visible_class_mask = np.array([name not in hidden for name in class_list], dtype=bool)
        keep = visible_class_mask[labels]   # True where voxel should be drawn

        if not np.any(keep):
            print("No voxels to render (all are Inside/Outside).")
            return

        if surface_only:
            # Keep only boundary voxels of the visible set (optional toggle)
            vm  = keep
            pad = np.pad(vm, 1, constant_values=False)
            fully_surrounded = (
                pad[2:,1:-1,1:-1] & pad[:-2,1:-1,1:-1] &
                pad[1:-1,2:,1:-1] & pad[1:-1,:-2,1:-1] &
                pad[1:-1,1:-1,2:] & pad[1:-1,1:-1,:-2]
            )
            keep = vm & ~fully_surrounded
            if not np.any(keep):
                print("No surface voxels remain after filtering.")
                return

        coords = np.argwhere(keep)   # (K, 3)
        kept_labels  = labels[keep]

        if stride > 1:
            coords = coords[::stride]
            kept_labels = kept_labels[::stride]

        # -----------------------------
        # 5) Colors (per-voxel RGBA, uint8)
        # -----------------------------
        lut_rgba = np.zeros((len(class_list), 4), dtype=np.uint8)
        for idx, name in enumerate(class_list):
            r, g, b = custom_colors[name]
            a = 0.0 if name in hidden else float(custom_opacity.get(name, 1.0))
            lut_rgba[idx] = (r, g, b, int(round(a * 255)))

        colors_rgba = lut_rgba[kept_labels]  # (K, 4) uint8

        # -----------------------------
        # 6) Glyph render (one actor)
        # -----------------------------
        points = coords.astype(np.float32)   # voxel centers at integer coords
        points = [tpl*vs for tpl in points]
        cloud  = pv.PolyData(points)
        cloud['rgba'] = colors_rgba
        vs = seg_info["SCALARS"]["voxel_size"]
        cube   = pv.Cube(center=(0, 0, 0), x_length=1.0*vs, y_length=1.0*vs, z_length=1.0*vs)
        glyphs = cloud.glyph(geom=cube, scale=False, orient=False)

        if render:
            print("PyVista version:", pv.__version__)
            p = pv.Plotter()
            p.add_mesh(
                glyphs,
                scalars='rgba',
                rgba=True,
                lighting=False,  # flat label colors
                culling='back',  # reduce overdraw
                show_edges=False,
                render_lines_as_tubes=False,
            )
            p.enable_eye_dome_lighting()

            # Legend for visible classes only
            legend_entries = [[name, tuple(c / 255 for c in custom_colors[name])]
                              for name in class_list if name not in hidden]
            if legend_entries:
                p.add_legend(legend_entries, bcolor='white', face='circle',
                             size=(0.2, 0.25), loc='lower right')
            p.show()
        else:
            return glyphs, class_list, custom_colors

def visu_voxel_label_and_prediction(data_loc, weights_loc, model_type, class_template,
                                    kernel_size, padding, n_classes,
                                    stride=1, surface_only=False):
    """Visualize voxel label vs. prediction side-by-side in one interactive Qt window."""

    pred_mesh, class_list, custom_colors = visu_voxel_prediction_on_dir(
        data_loc, weights_loc, model_type, class_template,
        kernel_size, padding, n_classes,
        stride=stride, surface_only=surface_only, render=False
    )
    label_mesh, _, _ = visu_voxel_label_on_dir(
        data_loc, kernel_size, padding, class_template,
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
    data_loc = r"H:\ws_seg_test\debug_output\REBeleg_Refined"
    weights_loc = r"H:\ws_hpc_workloads\hpc_models\fcb_Edge_01_UNet3D_Hilbig_crp10000\fcb_Edge_01_UNet3D_Hilbig_crp10000_save_20.pth"
    h5_path = r"H:\ws_seg_vdb\vdb_cyl_test\int_grid_predictions.h5"
    template = "edge"
    model_type = "UNet_Hilbig"
    n_classes = 9
    ks = 32
    pd = 8
    #visu_voxel_label_and_prediction(data_loc, weights_loc, model_type, template, ks, pd, n_classes)
    visu_voxel_prediction_on_dir(data_loc, weights_loc, model_type,template, ks, pd, n_classes)
    #visu_voxel_label_on_dir(data_loc, ks, pd, template, n_classes)
    #visu_voxel_prediction_from_h5(h5_path, template, ks, pd, n_classes)


if __name__=="__main__":
    main()