import os.path

import numpy as np
import h5py
from utility.data_exchange import cppIOexcavator

def data_to_hdf5(data_loc : str):

    p_meta_data = os.path.join(data_loc, "segmentation_data.dat")
    p_segments = os.path.join(data_loc, "segmentation_data_segments.bin")
    p_labels = os.path.join(data_loc, "segmentation_data_labels.bin")

    meta_data = cppIOexcavator.parse_dat_file(p_meta_data)
    segments  = cppIOexcavator.load_segments_from_binary(p_segments)
    labels = cppIOexcavator.load_labels_from_binary(p_labels)

    with h5py.File(os.path.join(data_loc,"file.h5"), 'w') as f:
        samples = f.create_group('samples')
 #   print()

def main():
    data_loc = r"H:\ws_abc_chunks\source\ABC_chunk_01_ks32swo4nbw8nk3_20250929-101945\ABC_chunk_01_labeled\00010089"
    data_to_hdf5(data_loc)

if __name__ == "__main__":
    main()