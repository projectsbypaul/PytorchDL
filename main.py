# packages
import argparse
import sys
from torch.utils.hipify.hipify_python import str2bool

# project includes
from entry_points.run_visu import RunVisu
from entry_points.run_data_utility import RunABCHelperFunctions
from entry_points.run_training_utility import RunTrainingUtility
from entry_points.run_validation_utility import RunValidation
from entry_points.run_job_utility import RunJobUtility

MODULE_HELP = {
    "job_utility": {
        "j_create_all": "root cnt out abs_p rec",
        "j_create_dirs": "root cnt out abs_p",
        "j_create_ext": "root cnt ext out abs_p rec",
    },
    "data_utility": {
        "create_subsets": "job_file src tgt n_min tmpl",
        "torch_to_hdf5": "src tgt",
        "crop_hdf5": "hdf5 n",
        "tree_to_hdf5": "root hdf5",
    },
    "train_utility": {
        "train": (
            "model_type model_name hdf5 weights epoch backup batch "
            "lr decay split amp val_factor workers n_classes seed ep_resume"
        ),
        "train_fcb": "(same as train)",
        "train_mfcb": "(same as train)",
        "train_hardcoded_weights": "(same as train)"
    },
    "validation_utility": {
        "val_segmentation_UNet": "model weights data k p n save_dir tag",
    },
    "visualization": {
        "model_on_mesh": "data weights save k p n",
    },
}

def print_module_help(module):
    print(f"\nOptions {module}:")
    modes = MODULE_HELP.get(module)

    if not modes:
        print("  (no help available)")
        return

    for mode, args in modes.items():
        print(f"  {mode:<20} {args}")

def sys_exit_with_help(module, error):
    print(f"[ERROR] {error}")
    print_module_help(module)
    sys.exit(1)

def require_n(args, n, name):
    if len(args) != n:
        raise ValueError(f"{name} expects exactly {n} arguments, got {len(args)}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="module", required=True)

    # ---------------- data_utility ----------------
    p = subparsers.add_parser("data_utility")
    p.add_argument(
        "mode",
        choices=[
            "create_subsets",
            "create_subsets_from_zip",
            "batch_subsets",
            "torch_to_hdf5",
            "crop_hdf5",
            "tree_to_hdf5",
            "join_hdf5",
            "screen_hdf5",
            "crop_hdf5_by_class",
            "help",
        ],
    )
    p.add_argument("extra", nargs="*")

    # ---------------- visualization ----------------
    p = subparsers.add_parser("visualization")
    p.add_argument("mode", choices=["model_on_mesh", "help"])
    p.add_argument("extra", nargs="*")

    # ---------------- train_utility ----------------
    p = subparsers.add_parser("train_utility")
    p.add_argument(
        "mode",
        choices=["train", "train_hardcoded_weights",  "train_fcb", "train_mfcb", "help"],
    )
    p.add_argument("extra", nargs="*")

    # ---------------- validation_utility ----------------
    p = subparsers.add_parser("validation_utility")
    p.add_argument("mode", choices=["val_segmentation_UNet", "help"])
    p.add_argument("extra", nargs="*")

    # ---------------- job_utility ----------------
    p = subparsers.add_parser("job_utility")
    p.add_argument(
        "mode",
        choices=["j_create_all", "j_create_ext", "j_create_dirs", "help"],
    )
    p.add_argument("extra", nargs="*")

    args = parser.parse_args()

    # ================= data_utility =================
    if args.module == "data_utility":
        if args.mode == "help":
            parser.parse_args(["data_utility", "--help"])
            sys.exit(0)

        try:
            if args.mode == "create_subsets":
                require_n(args.extra, 5, "create_subsets")
                job_file, src, tgt, n_min, tmpl = args.extra
                RunABCHelperFunctions.run_create_ABC_sub_Dataset_from_job(
                    job_file, src, tgt, int(n_min), tmpl
                )

            elif args.mode == "torch_to_hdf5":
                require_n(args.extra, 2, "torch_to_hdf5")
                RunABCHelperFunctions.run_torch_to_hdf5(*args.extra)

            elif args.mode == "crop_hdf5":
                require_n(args.extra, 2, "crop_hdf5")
                RunABCHelperFunctions.run_crop_hdf5_dataset(
                    args.extra[0], int(args.extra[1])
                )

            elif args.mode == "tree_to_hdf5":
                require_n(args.extra, 2, "tree_to_hdf5")
                RunABCHelperFunctions.run_convert_bin_tree_to_hdf5(
                    args.extra[0], args.extra[1]
                )

            else:
                raise ValueError("Unsupported data_utility mode")

        except Exception as e:
            sys_exit_with_help(args.module, e)

    # ================= visualization =================
    elif args.module == "visualization":
        if args.mode == "help":
            sys.exit(0)

        try:
            require_n(args.extra, 6, "model_on_mesh")
            data, weights, save, k, p, n = args.extra
            RunVisu.run_visu_mesh_model_on_dir(
                data, weights, save, int(k), int(p), int(n)
            )
        except Exception as e:
            sys_exit_with_help(args.module, e)

    # ================= train_utility =================
    elif args.module == "train_utility":
        if args.mode == "help":
            sys.exit(0)

        try:
            require_n(args.extra, 16, args.mode)
            (
                model_type,
                model_name,
                hdf5,
                weights,
                epoch,
                backup,
                batch,
                lr,
                decay,
                split,
                amp,
                val_factor,
                workers,
                n_classes,
                seed,
                ep_resume,
            ) = args.extra

            dispatch = {
                "train": RunTrainingUtility.run_hdf5_train,
                'train_hardcoded_weights': RunTrainingUtility.run_hdf5_train_hardcoded_weights,
                "train_fcb": RunTrainingUtility.run_hdf5_train_fcb,
                "train_mfcb": RunTrainingUtility.run_hdf5_train_mfcb,
            }

            runner = dispatch[args.mode]

            if runner is None:
                raise ValueError(f"Unsupported train mode: {args.mode}")

            runner(
                model_name,
                hdf5,
                weights,
                int(epoch),
                int(backup),
                int(batch),
                float(lr),
                float(decay),
                float(split),
                str2bool(amp),
                int(val_factor),
                int(workers),
                int(n_classes),
                int(seed),
                model_type=model_type,
                raw_ep_resume=ep_resume,
            )

        except Exception as e:
            sys_exit_with_help(args.module, e)

    # ================= validation_utility =================
    elif args.module == "validation_utility":
        if args.mode == "help":
            sys.exit(0)

        try:
            require_n(args.extra, 8, "val_segmentation_UNet")
            RunValidation.run_validate_segmentation_model(
                args.extra[0],
                args.extra[1],
                args.extra[2],
                int(args.extra[3]),
                int(args.extra[4]),
                int(args.extra[5]),
                args.extra[6],
                args.extra[7],
            )
        except Exception as e:
            sys_exit_with_help(args.module, e)

    # ================= job_utility =================
    elif args.module == "job_utility":
        if args.mode == "help":
            sys.exit(0)

        try:
            if args.mode == "j_create_all":
                require_n(args.extra, 5, "j_create_all")
                root, cnt, out, abs_p, rec = args.extra
                RunJobUtility.run_make_jobs_all(
                    root, int(cnt), out, str2bool(abs_p), str2bool(rec)
                )

            elif args.mode == "j_create_dirs":
                require_n(args.extra, 4, "j_create_dirs")
                root, cnt, out, abs_p = args.extra
                RunJobUtility.run_make_jobs_dirs(
                    root, int(cnt), out, str2bool(abs_p)
                )

            elif args.mode == "j_create_ext":
                require_n(args.extra, 6, "j_create_ext")
                root, cnt, ext, out, abs_p, rec = args.extra
                extensions = [
                    e if e.startswith(".") else "." + e
                    for e in ext.replace(",", " ").split()
                ]
                RunJobUtility.run_make_jobs_ext(
                    root, int(cnt), extensions, out, str2bool(abs_p), str2bool(rec)
                )

        except Exception as e:
            sys_exit_with_help(args.module, e)


if __name__ == "__main__":
    main()
