from dl_torch.validation_utility import val_segmentation

class RunValidation:
    @staticmethod
    def run_validate_segmentation_model(
            val_data_set : str, weights_loc : str, save_loc : str, kernel_size : int, padding
    ):
        val_segmentation.validate_segmentation_model(val_data_set, weights_loc, save_loc, kernel_size, padding)
        return 0

def main():
    pass

if __name__ == "__main__":
    main()