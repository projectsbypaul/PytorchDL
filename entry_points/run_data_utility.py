from dl_torch.data_utility import AnnotationForSegmentationABC
from dl_torch.data_utility.HDF5Dataset import HDF5Dataset

class RunABCHelperFunctions:
    @staticmethod
    def run_create_ABC_sub_Dataset_from_job(job_file : str, segment_dir : str, torch_dir : str ,n_min_files : int, template : str):
        AnnotationForSegmentationABC.create_ABC_sub_Dataset_from_job(job_file, segment_dir, torch_dir, n_min_files, template)
        return 0

    @staticmethod
    def run_batch_ABC_sub_Datasets(source_dir : str, target_dir: str , dataset_name : str, batch_count : int):
        AnnotationForSegmentationABC.batch_ABC_sub_Datasets(source_dir, target_dir, dataset_name, batch_count)
        return 0

    @staticmethod
    def run_torch_to_hdf5(torch_dir: str, out_file: str):
       HDF5Dataset.convert_pt_to_hdf5(torch_dir, out_file)
       return 0

def main():
    pass

if __name__ == "__main__":
    main()