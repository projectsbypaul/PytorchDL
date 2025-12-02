import pickle
from pathlib import Path
import numpy as np
import os
import pandas as pd
from numpy.ma.core import filled
from scipy import stats
from scipy.integrate import lebedev_rule

from visualization import color_templates

def __find_files_by_name(target_dir: str, search_str: str) -> list[str]:
    target_dir = Path(target_dir)
    matches = list(target_dir.rglob(f"*{search_str}*"))
    return matches

def __extract_distribution_metrics(val_result_bin: str):
    # create model signature
    model_name = os.path.basename(val_result_bin)
    model_name, _ = os.path.splitext(model_name)

    with open(val_result_bin, "rb") as f:
        sample_result = pickle.load(f)

    df = pd.DataFrame(sample_result, columns=['Sample_ID', 'ABC_ID', 'Accuracy'])

    current_path = os.path.abspath(val_result_bin)
    path_without_ext = os.path.splitext(current_path)[0]

    df.to_csv(path_without_ext + ".csv")

    sample_iou = np.array([item[2] * 100 for item in sample_result])
    rounded_data = np.round(sample_iou, 2)
    total_samples = len(sample_iou)

    # Calculate statistics
    mean_val = np.mean(sample_iou)
    median_val = np.median(sample_iou)

    mode_result = stats.mode(rounded_data, keepdims=True)
    mode_val = mode_result.mode[0]
    mode_count = mode_result.count[0]

    # Calculate percentiles
    p25 = np.percentile(sample_iou, 25)
    p75 = np.percentile(sample_iou, 75)


    distribution_metrics={
        "mean":mean_val,
        "median":median_val,
        "mode":mode_val,
        "p25":p25,
        "p75":p75
    }

    return distribution_metrics

def __extract_diagonal_from_cmm(cmm_bin : str, class_template):

    class_list = color_templates.get_class_list(class_template)

    with open(cmm_bin, "rb") as f:
        ccm = pickle.load(f)
        face_sums = np.sum(ccm, axis=1)
        face_total = np.sum(face_sums)
        print()

    n_predictions = np.sum(ccm, axis=1)

    ccm_norm = []

    for i in range(ccm.shape[0]):
        row = ccm[i]
        if n_predictions[i] > 0:
            row = row / n_predictions[i]

        ccm_norm.append(row)

    ccm_norm = np.asarray(ccm_norm)

    diagonal = np.ndarray(shape=len(class_list), dtype=float)

    for i in range(len(class_list)):
        diagonal[i] = ccm_norm[i,i]

    return diagonal

def taguchi_results_csv(result_dir: str, search_str: str, out_file: str):

    res_bin_name = __find_files_by_name(result_dir, search_str)
    result_array = np.full((len(res_bin_name),6),-1, dtype=float)
    for r in res_bin_name:
        exp_id = int(os.path.basename(r).split('_')[3])
        r_stats = __extract_distribution_metrics(r)
        row = [exp_id, r_stats["mean"], r_stats["median"], r_stats["mode"], r_stats["p25"], r_stats["p75"]]
        result_array[exp_id-1] = row

    df = pd.DataFrame(result_array, columns=['EXP_ID', 'mean', 'median', "mode", "p25", "p75"])
    df["EXP_ID"] = df["EXP_ID"].astype(int)
    df.to_csv(out_file, index=False, sep=';')

def ccm_results_to_csv(result_dir: str, search_str: str, out_file: str):

    template = color_templates.inside_outside_color_template_abc()

    class_list = color_templates.get_class_list(template)

    res_bin_name = __find_files_by_name(result_dir, search_str)

    result_array = np.full((len(res_bin_name), len(class_list) + 1), -1, dtype=float)

    for r in res_bin_name:
        exp_id = int(os.path.basename(r).split('_')[3])
        ccm_diagonal = __extract_diagonal_from_cmm(r, template)
        row = [exp_id, *ccm_diagonal]

        row_p = [val * 100 for val in row]
        row_p[0] = row[0]

        result_array[exp_id-1] = row_p

    df = pd.DataFrame(result_array, columns=['EXP_ID', *class_list])
    df = df.round(8)
    df["EXP_ID"] = df["EXP_ID"].astype(int)
    df.to_csv(out_file, index=False, sep=';')

def main():
    res_dir = r"H:\ws_design_2026\03_val_results\CIRP_REV_01"
    search_str = "_val_result_mcm.bin"
    out_csv = r"H:\ws_design_2026\03_val_results\CIRP_REV_01_mcm.csv"
    ccm_results_to_csv(res_dir, search_str, out_csv)

if __name__ == "__main__":
    main()