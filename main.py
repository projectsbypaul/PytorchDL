# packages
import argparse
import sys
import os
from torch.utils.hipify.hipify_python import str2bool

# project includes
from entry_points.run_visu import RunVisu
from entry_points.run_data_utility import RunABCHelperFunctions
from entry_points.run_training_utility import RunTrainingUtility
from entry_points.run_validation_utility import RunValidation
from entry_points.run_job_utility import RunJobUtility


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='module', required=True)

    # -----------------------------
    # module: data_utility
    # -----------------------------
    p_data_utility = subparsers.add_parser('data_utility')
    p_data_utility.add_argument(
        'mode',
        choices=[
            'create_subsets',
            'create_subsets_from_zip',
            'batch_subsets',
            'torch_to_hdf5',
            'crop_hdf5',
            'help',
        ],
    )
    p_data_utility.add_argument('arg0', type=str, nargs='?')
    p_data_utility.add_argument('arg1', type=str, nargs='?')
    p_data_utility.add_argument('arg2', type=str, nargs='?')
    p_data_utility.add_argument('arg3', type=str, nargs='?')
    p_data_utility.add_argument('arg4', type=str, nargs='?')

    # -----------------------------
    # module: visualization
    # -----------------------------
    p_visualization = subparsers.add_parser('visualization')
    p_visualization.add_argument('mode', choices=['model_on_mesh', 'help'])
    p_visualization.add_argument('arg0', type=str, nargs='?')
    p_visualization.add_argument('arg1', type=str, nargs='?')
    p_visualization.add_argument('arg2', type=str, nargs='?')
    p_visualization.add_argument('arg3', type=str, nargs='?')
    p_visualization.add_argument('arg4', type=str, nargs='?')
    p_visualization.add_argument('arg5', type=str, nargs='?')

    # -----------------------------
    # module: train_utility
    # -----------------------------
    p_train_utility = subparsers.add_parser('train_utility')
    p_train_utility.add_argument(
        'mode', choices=['train_UNet_16EL', 'train_UNet_Hilbig', 'help']
    )
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
    p_train_utility.add_argument('arg12', type=str, nargs='?')
    p_train_utility.add_argument('arg13', type=str, nargs='?')
    p_train_utility.add_argument('arg14', type=str, nargs='?')

    # -----------------------------
    # module: validation_utility
    # -----------------------------
    p_validation_utility = subparsers.add_parser('validation_utility')
    p_validation_utility.add_argument(
        'mode', choices=['val_segmentation_UNet', 'help']
    )
    p_validation_utility.add_argument('arg0', type=str, nargs='?')
    p_validation_utility.add_argument('arg1', type=str, nargs='?')
    p_validation_utility.add_argument('arg2', type=str, nargs='?')
    p_validation_utility.add_argument('arg3', type=str, nargs='?')
    p_validation_utility.add_argument('arg4', type=str, nargs='?')
    p_validation_utility.add_argument('arg5', type=str, nargs='?')
    p_validation_utility.add_argument('arg6', type=str, nargs='?')
    p_validation_utility.add_argument('arg7', type=str, nargs='?')

    # -----------------------------
    # module: job_utility
    # -----------------------------
    p_job_utility = subparsers.add_parser('job_utility')
    p_job_utility.add_argument(
        'mode', choices=['j_create_all', 'j_create_ext', 'j_create_dirs', 'help']
    )
    p_job_utility.add_argument('arg0', type=str, nargs='?')
    p_job_utility.add_argument('arg1', type=str, nargs='?')
    p_job_utility.add_argument('arg2', type=str, nargs='?')
    p_job_utility.add_argument('arg3', type=str, nargs='?')
    p_job_utility.add_argument('arg4', type=str, nargs='?')
    p_job_utility.add_argument('arg5', type=str, nargs='?')

    args = parser.parse_args()

    # -------------- data_utility --------------
    if args.module == 'data_utility':
        if args.mode == 'help' or args.arg0 is None:
            print("Usage:")
            print("main.py data_utility help")
            print(
                "main.py data_utility create_subsets <job_file> <source_dir> <target_dir> <n_min_files> <template>"
            )
            print(
                "main.py data_utility create_subsets_from_zip <source_dir> <job_file> <workspace> <template> <batch_count>"
            )
            print(
                "main.py data_utility batch_subsets <source_dir> <target_dir> <dataset_name> <batch_count>"
            )
            print("main.py data_utility torch_to_hdf5 <torch_dir> <out_file>")
            print("main.py data_utility crop_hdf5 <target> <n_samples>")
            sys.exit(0)

        elif args.mode == 'create_subsets':
            try:
                job_file: str = args.arg0
                source_dir: str = args.arg1
                target_dir: str = args.arg2
                n_min_files: int = int(args.arg3)
                template: str = args.arg4
            except (TypeError, ValueError):
                print(
                    "[ERROR] Invalid or missing arguments for 'create_subsets'."
                )
                p_data_utility.print_help()
                sys.exit(1)

            RunABCHelperFunctions.run_create_ABC_sub_Dataset_from_job(
                job_file, source_dir, target_dir, n_min_files, template
            )

        elif args.mode == 'create_subsets_from_zip':
            try:
                source_dir: str = args.arg0
                job_file: str = args.arg1
                workspace: str = args.arg2
                template: str = args.arg3
                batch_count: int = int(args.arg4)
            except (TypeError, ValueError):
                print(
                    "[ERROR] Invalid or missing arguments for 'create_subsets_from_zip'."
                )
                p_data_utility.print_help()
                sys.exit(1)

            RunABCHelperFunctions.run_compressed_segment_dir_to_dataset_from_job(
                source_dir, job_file, workspace, template, batch_count
            )

        elif args.mode == 'batch_subsets':
            try:
                source_dir: str = args.arg0
                target_dir: str = args.arg1
                dataset_name: str = args.arg2
                batch_count: int = int(args.arg3)
            except (TypeError, ValueError):
                print(
                    "[ERROR] Invalid or missing arguments for 'batch_subsets'."
                )
                p_data_utility.print_help()
                sys.exit(1)

            RunABCHelperFunctions.run_batch_ABC_sub_Datasets(
                source_dir, target_dir, dataset_name, batch_count
            )

        elif args.mode == 'torch_to_hdf5':
            try:
                torch_dir: str = args.arg0
                out_file: str = args.arg1
            except (TypeError, ValueError):
                print(
                    "[ERROR] Invalid or missing arguments for 'torch_to_hdf5'."
                )
                p_data_utility.print_help()
                sys.exit(1)

            RunABCHelperFunctions.run_torch_to_hdf5(torch_dir, out_file)

        elif args.mode == 'crop_hdf5':
            try:
                target: str = args.arg0
                n_samples: int = int(args.arg1)
            except (TypeError, ValueError):
                print("[ERROR] Invalid or missing arguments for 'crop_hdf5'.")
                p_data_utility.print_help()
                sys.exit(1)

            RunABCHelperFunctions.run_crop_hdf5_dataset(target, n_samples)

        else:
            print("[ERROR] Invalid mode for data_utility.")
            p_data_utility.print_help()
            sys.exit(1)

    # -------------- visualization --------------
    elif args.module == 'visualization':
        if args.mode == 'help' or args.arg0 is None:
            print("Usage:\n")
            print("main.py visualization help")
            print(
                "main.py visualization model_on_mesh <data_loc> <weight_loc> <save_loc> <kernel_size> <padding> <n_classes>"
            )
            sys.exit(0)

        elif args.mode == 'model_on_mesh':
            try:
                data_loc: str = args.arg0
                weight_loc: str = args.arg1
                save_loc: str = args.arg2
                kernel_size: int = int(args.arg3)
                padding: int = int(args.arg4)
                n_classes: int = int(args.arg5)
            except (TypeError, ValueError):
                print(
                    "[ERROR] Invalid or missing arguments for 'model_on_mesh'."
                )
                p_visualization.print_help()
                sys.exit(1)

            RunVisu.run_visu_mesh_model_on_dir(
                data_loc, weight_loc, save_loc, kernel_size, padding, n_classes
            )

        else:
            print("[ERROR] Invalid mode for visualization.")
            p_visualization.print_help()
            sys.exit(1)

    # -------------- train_utility --------------
    elif args.module == 'train_utility':
        if args.mode == 'help' or args.arg0 is None:
            print("Usage:\n")
            print("  main.py train_utility help\n")
            print("  main.py train_utility train_UNet_16EL \\")
            print(
                "      <model_name> <hdf5_path> <model_weights_loc> <epoch> \\"
            )
            print(
                "      <backup_epochs> <batch_size> <lr> <decay_order> <split> \\"
            )
            print(
                "      <use_amp> <val_batch_factor> <workers> <n_classes> <model_seed> <ep_resume>\n"
            )
            print("  main.py train_utility train_UNet_Hilbig \\")
            print(
                "      <model_name> <hdf5_path> <model_weights_loc> <epoch> \\"
            )
            print(
                "      <backup_epochs> <batch_size> <lr> <decay_order> <split> \\"
            )
            print(
                "      <use_amp> <val_batch_factor> <workers> <n_classes> <model_seed> <ep_resume>"
            )
            sys.exit(0)

        elif args.mode == 'train_UNet_16EL':
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
                use_amp: bool = str2bool(args.arg9)
                val_batch_factor: int = int(args.arg10)
                workers: int = int(args.arg11)
                n_classes: int = int(args.arg12)
                model_seed: int = int(args.arg13)
                ep_resume_raw: str | None = args.arg14
            except (TypeError, ValueError):
                print(
                    "[ERROR] Invalid or missing arguments for 'train_UNet_16EL'."
                )
                p_train_utility.print_help()
                sys.exit(1)

            RunTrainingUtility.run_hdf5_train_UNet_3D_Segmentation(
                model_name,
                hdf5_path,
                model_weights_loc,
                epoch,
                backup_epochs,
                batch_size,
                lr,
                decay_order,
                split,
                use_amp,
                val_batch_factor,
                workers,
                n_classes,
                model_seed,
                model_type="UNet_16EL",
                raw_ep_resume=ep_resume_raw,  # pass raw; callee normalizes
            )

        elif args.mode == 'train_UNet_Hilbig':
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
                use_amp: bool = str2bool(args.arg9)
                val_batch_factor: int = int(args.arg10)
                workers: int = int(args.arg11)
                n_classes: int = int(args.arg12)
                model_seed: int = int(args.arg13)
                ep_resume_raw: str | None = args.arg14
            except (TypeError, ValueError):
                print(
                    "[ERROR] Invalid or missing arguments for 'train_UNet_Hilbig'."
                )
                p_train_utility.print_help()
                sys.exit(1)

            RunTrainingUtility.run_hdf5_train_UNet_3D_Segmentation(
                model_name,
                hdf5_path,
                model_weights_loc,
                epoch,
                backup_epochs,
                batch_size,
                lr,
                decay_order,
                split,
                use_amp,
                val_batch_factor,
                workers,
                n_classes,
                model_seed,
                model_type="UNet_Hilbig",
                raw_ep_resume=ep_resume_raw,  # pass raw; callee normalizes
            )

        else:
            print("[ERROR] Invalid mode for train_utility.")
            p_train_utility.print_help()
            sys.exit(1)

    # -------------- validation_utility --------------
    elif args.module == 'validation_utility':
        if args.mode == 'help' or args.arg0 is None:
            print("Usage:\n")
            print("main.py validation_utility help")
            print(
                "main.py validation_utility val_segmentation_UNet <val_dataset> <weights_loc> <save_loc> <kernel_size> <padding> <n_classes> <model_type> <template>"
            )
            sys.exit(0)

        elif args.mode == 'val_segmentation_UNet':
            try:
                data_loc: str = args.arg0
                weight_loc: str = args.arg1
                save_loc: str = args.arg2
                kernel_size: int = int(args.arg3)
                padding: int = int(args.arg4)
                n_classes: int = int(args.arg5)
                model_type: str = args.arg6
                template: str = args.arg7
            except (TypeError, ValueError):
                print(
                    "[ERROR] Invalid or missing arguments for 'val_segmentation_UNet'."
                )
                p_validation_utility.print_help()
                sys.exit(1)

            RunValidation.run_validate_segmentation_model(
                data_loc, weight_loc, save_loc, kernel_size, padding, n_classes, model_type, template
            )

        else:
            print("[ERROR] Invalid mode for validation_utility.")
            p_validation_utility.print_help()
            sys.exit(1)

    # -------------- job_utility --------------
    elif args.module == 'job_utility':
        if args.mode == 'help' or args.arg0 is None:
            print("Usage:\n")
            print("main.py job_utility help")
            print(
                "main.py job_utility j_create_all <root> <instance_count> <output_dir> [abs_path:False] [recursive:False]\n"
                "main.py job_utility j_create_ext <root> <instance_count> <extensions> <output_dir> [abs_path:False] [recursive:False]\n"
                "main.py job_utility j_create_dirs <root> <instance_count> <output_dir> [abs_path:False]\n"
            )
            sys.exit(0)

        elif args.mode == 'j_create_all':
            try:
                root: str = args.arg0
                instance_count: int = int(args.arg1)
                output_dir: str = args.arg2
                abs_path: bool = str2bool(args.arg3)
                recursive: bool = str2bool(args.arg4)
            except (TypeError, ValueError):
                print(
                    "[ERROR] Invalid or missing arguments for 'j_create_all'."
                )
                p_job_utility.print_help()
                sys.exit(1)

            RunJobUtility.run_make_jobs_all(
                root, instance_count, output_dir, abs_path, recursive
            )

        elif args.mode == 'j_create_ext':
            try:
                root: str = args.arg0
                instance_count: int = int(args.arg1)
                ext_arg = args.arg2
                extensions: [str] = [
                    e if e.startswith(".") else "." + e
                    for e in ext_arg.replace(",", " ").split()
                ]
                output_dir: str = args.arg3
                abs_path: bool = str2bool(args.arg4)
                recursive: bool = str2bool(args.arg5)
            except (TypeError, ValueError):
                print(
                    "[ERROR] Invalid or missing arguments for 'j_create_ext'."
                )
                p_job_utility.print_help()
                sys.exit(1)

            RunJobUtility.run_make_jobs_ext(
                root, instance_count, extensions, output_dir, abs_path, recursive
            )

        elif args.mode == 'j_create_dirs':
            try:
                root: str = args.arg0
                instance_count: int = int(args.arg1)
                output_dir: str = args.arg2
                abs_path: bool = str2bool(args.arg3)
            except (TypeError, ValueError):
                print(
                    "[ERROR] Invalid or missing arguments for 'j_create_dirs'."
                )
                p_job_utility.print_help()
                sys.exit(1)

            RunJobUtility.run_make_jobs_dirs(
                root, instance_count, output_dir, abs_path
            )

        else:
            print("[ERROR] Invalid mode for job_utility.")
            p_job_utility.print_help()
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    # Debug toggles (uncomment while diagnosing CLI issues):
    # print("[DEBUG] __file__:", os.path.abspath(__file__), file=sys.stderr)
    # print("[DEBUG] sys.argv:", sys.argv, file=sys.stderr)
    main()
