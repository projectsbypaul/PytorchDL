import numpy as np
import pickle

def get_voxel_count_from_stats_binary(stat_bin_path: str):

    with open(stat_bin_path, "rb") as f:
        h5_voxel_count = pickle.load(f)

    class_count = np.sum(h5_voxel_count, axis=0)

    return class_count

def get_fcb_weights_from_voxel_count(voxel_count: np.array, min_val : float = None,  max_val : float = None):

    total_count = np.sum(voxel_count, axis=0)

    #weights for pure frequency balancing
    f_c =  voxel_count / (total_count + 1e-6)
    w_fcb = 1 / f_c

    if min_val is not None:
        w_fcb = np.maximum(w_fcb, min_val)

    if max_val is not None:
        w_fcb = np.minimum(w_fcb, max_val)

    return w_fcb

def get_fcb_median_weights_from_class_count(voxel_count: np.array, min_val : float = None,  max_val : float = None):


    total_count = np.sum(voxel_count, axis=0)

    # weights for median frequency balancing
    f_c = voxel_count / (total_count + 1e-6)
    f_median = np.median(f_c)
    w_median_fcb = f_median / (f_c + 1e-6)

    if min_val is not None:
        w_median_fcb = np.maximum(w_median_fcb, min_val)

    if max_val is not None:
        w_median_fcb = np.minimum(w_median_fcb, max_val)

    return w_median_fcb

def main():
    pass

if __name__ == "__main__":
    main()
