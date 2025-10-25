import numpy as np
import pickle

def get_voxel_count_from_stats_binary(stat_bin_path: str):

    with open(stat_bin_path, "rb") as f:
        h5_voxel_count = pickle.load(f)

    class_count = np.sum(h5_voxel_count, axis=0)

    return class_count

def get_fcb_weights_from_voxel_count(voxel_count: np.array):

    total_count = np.sum(voxel_count, axis=0)

    #weights for pure frequency balancing
    f_c =  voxel_count / total_count
    w_fcb = 1 / f_c

    return w_fcb

def get_fcb_median_weights_from_class_count(voxel_count: np.array):


    total_count = np.sum(voxel_count, axis=0)

    # weights for median frequency balancing
    f_c = voxel_count / total_count
    f_median = np.median(f_c)
    w_median_fcb = f_median / f_c

    return w_median_fcb

def main():
    pass

if __name__ == "__main__":
    main()
