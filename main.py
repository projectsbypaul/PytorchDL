#packages
import argparse
import sys

#project includes
from entry_points.run_visu import RunVisu
from entry_points.run_data_utility import RunABCHelperFunctions
from entry_points.run_training_utility import RunTrainingUtility


def main():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='module', required=True)

    # module data_utility
    p_data_utility = subparsers.add_parser('data_utility')
    p_data_utility.add_argument('mode', choices=['create_subsets', 'batch_subsets','help'])
    p_data_utility.add_argument('arg0', type=str, nargs='?')
    p_data_utility.add_argument('arg1', type=str, nargs='?')
    p_data_utility.add_argument('arg2', type=str, nargs='?')
    p_data_utility.add_argument('arg3', type=str, nargs='?')

    # module data_utility
    p_data_utility = subparsers.add_parser('visualization')
    p_data_utility.add_argument('mode', choices=['model_on_mesh', 'help'])
    p_data_utility.add_argument('arg0', type=str, nargs='?')
    p_data_utility.add_argument('arg1', type=str, nargs='?')
    p_data_utility.add_argument('arg2', type=str, nargs='?')
    p_data_utility.add_argument('arg3', type=str, nargs='?')
    p_data_utility.add_argument('arg4', type=str, nargs='?')
    p_data_utility.add_argument('arg5', type=str, nargs='?')

    # module data_utility
    p_train_utility = subparsers.add_parser('train_utility')
    p_train_utility.add_argument('mode', choices=['train_UNet3D', 'train_hdf5_UNet3D', 'help'])
    p_train_utility.add_argument('arg0', type=str, nargs='?')
    p_train_utility.add_argument('arg1', type=str, nargs='?')
    p_train_utility.add_argument('arg2', type=str, nargs='?')
    p_train_utility.add_argument('arg3', type=str, nargs='?')
    p_train_utility.add_argument('arg4', type=str, nargs='?')
    p_train_utility.add_argument('arg5', type=str, nargs='?')
    p_train_utility.add_argument('arg6', type=str, nargs='?')
    p_train_utility.add_argument('arg7', type=str, nargs='?')
    p_train_utility.add_argument('arg8', type=str, nargs='?')
    p_train_utility.add_argument('arg9', type=str, nargs='?')
    p_train_utility.add_argument('arg10', type=str, nargs='?')
    p_train_utility.add_argument('arg11', type=str, nargs='?')

    args = parser.parse_args()

    if args.module == 'data_utility':
        if args.mode == 'help' or args.arg0 is None:
            print("Usage:")
            print("main.py data_utility help")
            print("main.py data_utility create_subsets <job_file> <source_dir> <target_dir> <n_min_files>")
            print("main.py data_utility batch_subsets <source_dir> <target_dir> <dataset_name> <batch_count>")
            print("main.py data_utility torch_to_hdf5 <torch_dir> <out_file>")
            sys.exit(0)
        elif args.mode == 'create_subsets':
            try:
                job_file: str = args.arg0
                source_dir: str = args.arg1
                target_dir: str = args.arg2
                n_min_files: int = int(args.arg3)
            except (TypeError, ValueError):
                print("[ERROR] Invalid or missing arguments for 'create_subsets'.")
                p_data_utility.print_help()
                sys.exit(1)
            # Replace this with your actual function call
            RunABCHelperFunctions.run_create_ABC_sub_Dataset_from_job(
                job_file, source_dir, target_dir, n_min_files
            )
        elif args.mode == 'batch_subsets':
            try:
                source_dir : str = args.arg0
                target_dir: str = args.arg1
                dataset_name : str = args.arg2
                batch_count: int = int(args.arg3)
            except (TypeError, ValueError):
                print("[ERROR] Invalid or missing arguments for 'create_subsets'.")
                p_data_utility.print_help()
                sys.exit(1)
            # Replace this with your actual function call
            RunABCHelperFunctions.run_batch_ABC_sub_Datasets(
                source_dir, target_dir, dataset_name, batch_count
            )
        elif args.mode == 'torch_to_hdf5':
            try:
                torch_dir: str = args.arg0
                out_file: str = args.arg1
            except (TypeError, ValueError):
                print("[ERROR] Invalid or missing arguments for 'create_subsets'.")
                p_data_utility.print_help()
                sys.exit(1)
            # Replace this with your actual function call
            RunABCHelperFunctions.run_torch_to_hdf5(
                torch_dir, out_file
            )
        else:
            print("[ERROR] Invalid mode for data_utility.")
            p_data_utility.print_help()
            sys.exit(1)

    elif args.module == 'visualization':
        if args.mode == 'help' or args.arg0 is None:
          print("Usage:\n")
          print("")
          print("main.py visualization help")
          print("main.py visualization model_on_mesh <data_loc> <weight_loc> <save_loc> <kernel_size> <padding> <n_classes>")
          sys.exit(0)
        elif args.mode == 'model_on_mesh':
            try:
                data_loc: str = args.arg0
                weight_loc: str = args.arg1
                save_loc: str = args.arg2
                kernel_size: int = int(args.arg3)
                padding : int = int(args.arg4)
                n_classes : int = int(args.arg5)
            except (TypeError, ValueError):
                print("[ERROR] Invalid or missing arguments for 'create_subsets'.")
                p_data_utility.print_help()
                sys.exit(1)

            # Replace this with your actual function call
            RunVisu.run_visu_mesh_model_on_dir(data_loc, weight_loc, save_loc, kernel_size, padding, n_classes)
        else:
            print("[ERROR] Invalid mode for data_utility.")
            p_data_utility.print_help()
            sys.exit(1)

    elif args.module == 'train_utility':
        if args.mode == 'help' or args.arg0 is None:
            print("Usage:\n")
            print("  main.py train_utility help")
            print()
            print("  main.py train_utility train_UNet3D \\")
            print("      <model_name> <dataset_dir> <model_weights_loc> <epoch> \\")
            print("      <backup_epochs> <batch_size> <lr> <decay_order> <split> "
                  "      <use_amp> <workers>")
            print()
            print("  main.py train_utility train_hdf5_UNet3D \\")
            print("      <model_name> <hdf5_path> <model_weights_loc> <epoch> \\")
            print("      <backup_epochs> <batch_size> <lr> <decay_order> <split> \\")
            print("      <use_amp> <val_batch_factor> <workers>")
            sys.exit(0)
        elif args.mode == 'train_UNet3D':
            try:
                model_name : str = args.arg0
                dataset_dir: str = args.arg1
                model_weights_loc: str = args.arg2
                epoch: int = int(args.arg3)
                backup_epochs : int = int(args.arg4)
                batch_size : int = int(args.arg5)
                lr : float = float(args.arg6)
                decay_order : float = float(args.arg7)
                split: float = float(args.arg8)
                use_amp: bool = bool(args.arg9)
                workers: int = int(args.arg10)


            except (TypeError, ValueError):
                print("[ERROR] Invalid or missing arguments for 'create_subsets'.")
                p_data_utility.print_help()
                sys.exit(1)

            # Replace this with your actual function call
            RunTrainingUtility.run_train_UNet_3D_Segmentation(
                model_name, dataset_dir, model_weights_loc, epoch, backup_epochs, batch_size,
                lr, decay_order, split, use_amp, workers
            )
        elif args.mode == 'train_hdf5_UNet3D':
            try:
                model_name: str = args.arg0
                hdf5_path: str = args.arg1
                model_weights_loc: str = args.arg2
                epoch: int = int(args.arg3)
                backup_epochs: int = int(args.arg4)
                batch_size: int = int(args.arg5)
                lr: float = float(args.arg6)
                decay_order: float = float(args.arg7)
                split: float = float(args.arg8)
                use_amp: bool = bool(args.arg9)
                val_batch_factor: int = int(args.arg10)
                workers: int = int(args.arg11)

            except (TypeError, ValueError):
                print("[ERROR] Invalid or missing arguments for 'create_subsets'.")
                p_data_utility.print_help()
                sys.exit(1)

            # Replace this with your actual function call
            RunTrainingUtility.run_hdf5_train_UNet_3D_Segmentation(
            model_name, hdf5_path, model_weights_loc, epoch, backup_epochs, batch_size,
            lr, decay_order, split, use_amp, val_batch_factor, workers
        )

        else:
            print("[ERROR] Invalid mode for data_utility.")
            p_data_utility.print_help()
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    # For debugging, uncomment the next line:
    # data_loc = r'H:\ABC_Demo\target'
    # weights_loc = r'H:\ABC_Demo\weights\UNet3D_SDF_16EL_n_class_10_lr[1e-05]_lrdc[1e-01]_bs4_save_100.pth'
    # save_loc = r"H:/ABC_Demo/blender/color_map_learned.pkl"
    # kernel_size = 16
    # padding = 4
    # n_classes = 10
    # sys.argv = ["main.py", "visualization", "model_on_mesh", data_loc, weights_loc, save_loc, str(kernel_size), str(padding), str(n_classes)]
    main()