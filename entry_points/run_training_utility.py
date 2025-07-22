from dl_torch.training_utility.managed_train_UNet_3D_Segmentation import training_routine
from dl_torch.training_utility.hdf5_train_UNet_3D_Segmentation import training_routine_hdf5

class RunTrainingUtility:
    @staticmethod
    def run_train_UNet_3D_Segmentation(model_name : str,
                     dataset_dir : str,
                     model_weights_loc : str,
                     epochs : int,
                     backup_epochs : int,
                     batch_size : int,
                     lr,
                     decay_order,
                     split,
                     use_amp = True,
                     num_workers: int = 0
                     ):
        training_routine(
            model_name, dataset_dir, model_weights_loc, epochs, backup_epochs, batch_size, lr, decay_order, split, num_workers, use_amp
        )

        return 0

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
                     workers: int):

        training_routine_hdf5(
            model_name, hdf5_path, model_weights_loc, epochs, backup_epochs, batch_size,
            lr, decay_order, split, use_amp, val_batch_factor, workers
        )


def main():
    pass

if __name__ == "__main__":
    main()