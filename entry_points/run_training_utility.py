import os.path
import numpy as np

from dl_torch.model_utility import Classifcation_Helpers
from dl_torch.training_utility.hdf5_train_UNet_3D_Segmentation import train_model_hdf5

class RunTrainingUtility:

    @staticmethod
    def run_hdf5_train_UNet_3D_Segmentation(model_name : str,
                     hdf5_path : str,
                     model_weights_loc : str,
                     epochs : int,
                     backup_epochs : int,
                     batch_size : int,
                     lr,
                     decay_order,
                     split,
                     use_amp: bool,
                     val_batch_factor: int,
                     workers: int,
                     n_classes: int,
                     model_seed: int,
                     model_type:str,
                     raw_ep_resume: str):


        if raw_ep_resume is None or (isinstance(raw_ep_resume, str) and raw_ep_resume.lower() == "none"):
            resume_epoch = None
        else:
            resume_epoch = int(raw_ep_resume)

        train_model_hdf5(
            model_name, hdf5_path, model_weights_loc, epochs, backup_epochs, batch_size,
            lr, decay_order, split, use_amp, val_batch_factor, workers,
            n_classes=n_classes, model_seed=model_seed, model_type=model_type, resume_epoch=resume_epoch
        )

    @staticmethod
    def run_hdf5_train_fcb_UNet_3D_Segmentation(model_name: str,
                                            hdf5_path: str,
                                            model_weights_loc: str,
                                            epochs: int,
                                            backup_epochs: int,
                                            batch_size: int,
                                            lr,
                                            decay_order,
                                            split,
                                            use_amp: bool,
                                            val_batch_factor: int,
                                            workers: int,
                                            n_classes: int,
                                            model_seed: int,
                                            model_type: str,
                                            raw_ep_resume: str):

        if raw_ep_resume is None or (isinstance(raw_ep_resume, str) and raw_ep_resume.lower() == "none"):
            resume_epoch = None
        else:
            resume_epoch = int(raw_ep_resume)

        hdf5_stub = hdf5_path.split('.')[0]
        stat_bin = f"{hdf5_stub}_stats.bin"

        if not os.path.exists(stat_bin):
            raise FileNotFoundError(f"Not stats file found: {stat_bin}")


        voxel_count = Classifcation_Helpers.get_voxel_count_from_stats_binary(stat_bin)
        fcb_weights = Classifcation_Helpers.get_fcb_weights_from_voxel_count(voxel_count, min_val=0.1)
        fcb_weights = np.atleast_1d(fcb_weights.astype(float))
        fcb_weights_list = fcb_weights.tolist()

        train_model_hdf5(
            model_name, hdf5_path, model_weights_loc, epochs, backup_epochs, batch_size,
            lr, decay_order, split, use_amp, val_batch_factor, workers,
            n_classes=n_classes, model_seed=model_seed, model_type=model_type, resume_epoch=resume_epoch, class_weights=fcb_weights_list
        )

    @staticmethod
    def run_hdf5_train_mfcb_UNet_3D_Segmentation(model_name: str,
                                                hdf5_path: str,
                                                model_weights_loc: str,
                                                epochs: int,
                                                backup_epochs: int,
                                                batch_size: int,
                                                lr,
                                                decay_order,
                                                split,
                                                use_amp: bool,
                                                val_batch_factor: int,
                                                workers: int,
                                                n_classes: int,
                                                model_seed: int,
                                                model_type: str,
                                                raw_ep_resume: str):

        if raw_ep_resume is None or (isinstance(raw_ep_resume, str) and raw_ep_resume.lower() == "none"):
            resume_epoch = None
        else:
            resume_epoch = int(raw_ep_resume)

        hdf5_stub = hdf5_path.split('.')[0]
        stat_bin = f"{hdf5_stub}_stats.bin"

        if not os.path.exists(stat_bin):
            raise FileNotFoundError(f"Not stats file found: {stat_bin}")

        voxel_count = Classifcation_Helpers.get_voxel_count_from_stats_binary(stat_bin)
        mfcb_weights = Classifcation_Helpers.get_fcb_median_weights_from_class_count(voxel_count, min_val=0.1)
        mfcb_weights = np.atleast_1d(mfcb_weights.astype(float))
        mfcb_weights_list = mfcb_weights.tolist()

        train_model_hdf5(
            model_name, hdf5_path, model_weights_loc, epochs, backup_epochs, batch_size,
            lr, decay_order, split, use_amp, val_batch_factor, workers,
            n_classes=n_classes, model_seed=model_seed, model_type=model_type, resume_epoch=resume_epoch,
            class_weights=mfcb_weights_list
        )


def main():
    pass

if __name__ == "__main__":

    main()