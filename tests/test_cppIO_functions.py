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

def read_test_type_maps():
    face_map_loc = r"C:\Local_Data\cropping_test\FaceTypeMap.bin"
    vert_map_loc = r"C:\Local_Data\cropping_test\VertTypeMap.bin"

    vert_map = cppIO.read_type_map_from_binary(vert_map_loc)
    face_map = cppIO.read_type_map_from_binary(face_map_loc)

    print()


def read_float_matrix_test():
    origin_matrix_loc = r"C:\Local_Data\ABC\ABC_AE_Data_ks_16_pad_4_bw_5_vs_adaptive\00000004\origins.bin"
    matrix_0 = cppIO.read_float_matrix(origin_matrix_loc)

    origin_matrix_loc = r"C:\Local_Data\ABC\ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2\00000004\origins.bin"
    matrix_1 = cppIO.read_float_matrix(origin_matrix_loc)

    print()



def main() -> None:
    pass


if __name__ == "__main__":
    main()

