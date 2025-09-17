from dl_torch.training_utility.managed_train_UNet_3D_Segmentation import training_routine
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


def main():
    pass

if __name__ == "__main__":
    main()