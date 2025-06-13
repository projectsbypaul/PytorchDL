from visualization import visu_scripts

class RunVisu:
    @staticmethod
    def run_visu_mesh_model_on_dir(
            data_loc : str, weights_loc : str, save_loc : str, kernel_size : int, padding : int, n_classes : int
    ):
        visu_scripts.visu_mesh_model_on_dir(data_loc, weights_loc, save_loc, kernel_size, padding, n_classes)
        return 0

def main():
    pass

if __name__ == "__main__":
    main()