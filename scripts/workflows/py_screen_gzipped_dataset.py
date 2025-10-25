import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import csv
import os
from dl_torch.data_utility import hdf5_utility
from dl_torch.model_utility import Classifcation_Helpers
import gzip
import shutil


def main():

    gzipped_h5 = r"H:\ws_export\ABC_InOut_ks32swo4nbw8nk3_crp50000.h5.gz"
    template = "inside_outside"

    h5_name = os.path.basename(gzipped_h5).split('.')[0]
    workspace = os.path.dirname(gzipped_h5)
    h5_src = os.path.join(workspace,h5_name)
    stat_bin = os.path.join(workspace, f"{h5_name}_stats.bin")
    stat_csv = os.path.join(workspace, f"{h5_name}_wnc.csv")

    with gzip.open(gzipped_h5, "rb") as f_in:
        with open(h5_src, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    hdf5_utility.screen_hdf_dataset(h5_src, stat_bin)
    
    os.remove(h5_src)


    with open(stat_bin, "rb") as f:
        h5_voxel_count = pickle.load(f)

    hdf5_utility.get_class_distribution(stat_bin)

    class_count = np.sum(h5_voxel_count, axis=0)

    total_count = np.sum(class_count, axis=0)

    # weights for frequency balancing
    f_c = class_count / total_count

    w_fcb = 1 / f_c

    f_median = np.median(f_c)

    w_median_fcb = f_median / f_c

    data = []

    for i, entry in enumerate(class_count):
        data.append({"class": i, "voxel_count": entry, "weights_fcb": w_fcb[i], "weights_median_fcb" : w_median_fcb[i]})

    with open(stat_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "voxel_count", "weights_fcb", "weights_median_fcb"], delimiter=";")
        writer.writeheader()
        writer.writerows(data)

    print(f"CSV file written to {stat_csv}")

if __name__ == "__main__":
    main()