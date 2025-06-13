from dl_torch.training_utility.managed_train_UNet_3D_Segmentation import training_routine

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
                     split
                     ):
        training_routine(
            model_name, dataset_dir, model_weights_loc, epochs, backup_epochs, batch_size, lr, decay_order, split
        )

        return 0

def main():
    pass

if __name__ == "__main__":
    main()