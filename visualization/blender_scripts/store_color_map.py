import numpy as np
from utility.data_exchange import cppIO
from dl_torch.data_utility import DataParsing
import pickle

def FaceTypeMap_blender_export():
    ftm_loc = r"C:\Local_Data\ABC\ABC_Testing\FaceTypeMap.bin"
    obj_loc = r"C:\Local_Data\ABC\ABC_parsed_files\00000004\00000004.obj"

    FaceTypeMap = cppIO.read_type_map_from_binary(ftm_loc)
    # flatten list
    FaceTypeMap = [inner[0] for inner in FaceTypeMap]

    unique_types = sorted(set(FaceTypeMap))

    vertices, faces = DataParsing.parse_obj(obj_loc)

    # Define RGB (0–255) and opacity (0.0–1.0) for all classes, including 'Void'
    custom_colors = {

        'Cone': (0, 0, 255),  # blue
        'Cylinder': (255, 0, 0),  # red
        'Edge': (255, 255, 0),  # yellow
        'Plane': (255, 192, 203),  # pink
        'Sphere': (128, 0, 0),  # dark red
        'Torus': (0, 255, 255),  # cyan
    }

    custom_opacity = {
        'Cone': 1.0,
        'Cylinder': 1.0,
        'Edge': 1.0,
        'Plane': 1.0,
        'Sphere': 1.0,
        'Torus': 1.0,
        }

    face_colors = []

    for face_label in FaceTypeMap:
        if face_label in custom_colors:
            rgb = custom_colors[face_label]
            rgba = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0)  # Normalize RGB to 0–1 and add alpha
            face_colors.append(rgba)
        else:
            # Default color if label missing
            face_colors.append((1.0, 1.0, 1.0, 1.0))

    # Now save it
    with open(r"../../data/blender_export/color_map.pkl", "wb") as f:
        pickle.dump(face_colors, f)

    print()



def main():
    FaceTypeMap_blender_export()

if __name__ == "__main__":
   main()