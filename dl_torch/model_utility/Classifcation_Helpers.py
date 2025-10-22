import numpy as np

def get_fcb_weights_from_voxel_count(voxel_count: np.array):

    class_count = np.sum(voxel_count, axis=0)
    total_count = np.sum(class_count, axis=0)

    #weights for pure frequency balancing
    f_c = class_count / total_count
    w_fcb = 1 / f_c

    return w_fcb

def get_fcb_median_weights_from_class_count(voxel_count: np.array):

    class_count = np.sum(voxel_count, axis=0)
    total_count = np.sum(class_count, axis=0)

    # weights for median frequency balancing
    f_c = class_count / total_count
    f_median = np.median(f_c)
    w_median_fcb = f_median / f_c

    return w_median_fcb

def main():
    pass

if __name__ == "__main__":
    main()
