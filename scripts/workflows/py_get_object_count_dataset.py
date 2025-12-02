import os.path
import shutil
from pathlib import Path
import tarfile
import zipfile
from utility.data_exchange.cppIOexcavator import  load_segments_from_binary
import numpy as np

def __find_files_by_name(target_dir: str, search_str: str) -> list[str]:
    target_dir = Path(target_dir)
    matches = list(target_dir.rglob(f"*{search_str}*"))
    return matches

def __extract_zip(src_path, dest_dir):
    src_path = Path(src_path)
    dest_dir = Path(dest_dir)

    with zipfile.ZipFile(src_path, 'r') as z:
        z.extractall(dest_dir)

def __extract_tar_gz(src_path, dest_dir):
    src_path = Path(src_path)
    dest_dir = Path(dest_dir)

    with tarfile.open(src_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)

def get_object_count_in_dataset(data_set_dir):

    data_set = __find_files_by_name(data_set_dir, "tar.gz")[0]
    targz_unpacked = os.path.join(data_set_dir, "targz_unpacked")
    zip_unpacked = os.path.join(data_set_dir, "zip_unpacked")

    if not os.path.exists(targz_unpacked) : os.mkdir(targz_unpacked)
    if not os.path.exists(zip_unpacked): os.mkdir(zip_unpacked)
    print("Unpacking .tar.gz")
    __extract_tar_gz(data_set, targz_unpacked)

    print("Unpacking zip files")
    zip_segments = os.listdir(targz_unpacked)
    for z in zip_segments:
        __extract_zip(os.path.join(targz_unpacked, z), zip_unpacked)
    
    if os.path.exists(os.path.join(zip_unpacked, "temp")) : shutil.rmtree(os.path.join(zip_unpacked, "temp"))
    
    shutil.rmtree(targz_unpacked)

    sub_dir = os.listdir(zip_unpacked)
    n_subdir = len(sub_dir)
    seg_count = np.zeros(len(sub_dir))

    print("Parsing segments count from bin files")
    for i,sd in enumerate(sub_dir):
        #print(f"parsing {i+1}|{n_subdir}")
        bin_file = os.path.join(zip_unpacked, sd, "segmentation_data_segments.bin")
        segments = load_segments_from_binary(bin_file)
        seg_count[i] = len(segments)

    seg_count_mean = np.mean(seg_count)
    seg_count_min = np.min(seg_count)
    seg_count_max = np.max(seg_count)

    print(f"{os.path.basename(data_set_dir)} - seg_mean: {seg_count_mean}, seg_min: {seg_count_min}, seg_max: {seg_count_max}\n")

    shutil.rmtree(zip_unpacked)

def main():

    dataset = [
        r"H:\ws_design_2026\00_datagen\Block_A\train_A_10000_16_pd0_bw12_vs2_20250825-084440",
        r"H:\ws_design_2026\00_datagen\Block_B\train_B_10000_16_pd0_bw12_vs2_20250825-101708",
        r"H:\ws_design_2026\00_datagen\Block_C\train_C_10000_16_pd0_bw12_vs2_20250825-130218"
    ]

    for d in dataset:
        get_object_count_in_dataset(d)

if __name__=="__main__":
    main()