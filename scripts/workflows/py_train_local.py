import os, shutil, gzip
import gzip
import shutil
import os
import re
import sys
from multiprocessing.pool import worker

sys.path.append("/mnt/c/Users/pschuster/source/repos/PytorchDL")
from entry_points.run_training_utility import RunTrainingUtility

class LocalRun:
    model_name : str
    dataset_loc: str
    workspace: str
    epochs:    int
    backup_epochs: int
    batch_size: int
    lr: float
    decay_order: float
    n_classes: int
    model_type: str
    model_seed: int

    split: float = 0.9
    use_amp: bool = True
    val_batch_factor: int = 4
    workers: int = 1
    resume_epoch: int = 0
    class_weights_mode: str = 'default'
    model_weights_loc: str = None
    hdf5_path: str = None

    def __init__(
        self,
        model_name: str,
        dataset_loc: str,
        workspace: str,
        model_weights_loc: str,
        epochs: int,
        backup_epochs: int,
        batch_size: int,
        lr: float,
        decay_order: float,
        n_classes: int,
        model_type: str,
        model_seed: int,
        *,
        split: float = 0.9,
        use_amp: bool = True,
        val_batch_factor: int = 4,
        workers: int = 1,
        resume_epoch: int = 0,
        class_weights_mode: str = 'default',

        hdf5_path: str = None,
    ):

        # --- validations ---
        assert isinstance(model_name, str) and model_name, "model_name must be non-empty string"
        assert os.path.exists(dataset_loc), f"dataset_loc does not exist: {dataset_loc}"
        # ensure workspace exists or create it
        os.makedirs(workspace, exist_ok=True)
        assert epochs >= 1, "epochs must be >=1"
        assert backup_epochs >= 0, "backup_epochs must be >=0"
        assert batch_size >= 1, "batch_size must be >=1"
        assert 1e-7 < lr <= 10, "lr must be in (0, 10]"
        #assert 0 <= decay_order <= 10, "decay_order must be in [0,10]"
        assert n_classes >= 1, "n_classes must be >=1"
        assert isinstance(model_type, str) and model_type, "model_type must be non-empty string"
        assert model_seed >= 0, "model_seed must be >=0"
        assert 0 < split <= 1, "split must be in (0,1]"
        assert workers >= 0, "workers must be >=0"
        assert resume_epoch >= 0, "resume_epoch must be >=0"

        valid_wm = {"default","fcb","mfcb"}
        assert class_weights_mode in valid_wm, f"class_weights_mode must be one of {valid_wm}"

        # --- assignments ---
        self.model_name = model_name
        self.dataset_loc = dataset_loc
        self.workspace = workspace
        self.epochs = epochs
        self.backup_epochs = backup_epochs
        self.backup_epochs = backup_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.decay_order = decay_order
        self.n_classes = n_classes
        self.model_type = model_type
        self.model_seed = model_seed

        self.split = split
        self.use_amp = use_amp
        self.use_amp = use_amp
        self.val_batch_factor = val_batch_factor
        self.workers = workers
        self.resume_epoch = resume_epoch
        self.class_weights_mode = class_weights_mode
        self.model_weights_loc = model_weights_loc
        self.hdf5_path = hdf5_path

    def __setup_workspace(self):
        print(f"Preparing workspace at {self.workspace}")
        dataset_name = os.path.basename(self.dataset_loc).split('.')[0]
        gzipped_h5 = os.path.join(self.dataset_loc, dataset_name + ".h5.gz")
        self.hdf5_path = os.path.join(self.workspace, dataset_name + ".h5")

        print(f"Copy stat.bin to workspace at {self.workspace}")

        stat_file = os.path.join(self.dataset_loc, dataset_name + "_stats.bin")
        stat_file_loc = os.path.join(self.workspace, dataset_name + "_stats.bin")
        if not os.path.exists(stat_file_loc) and os.path.exists(stat_file):
            shutil.copy(stat_file, self.workspace)

        print(f"Unpack gzipped dataset into {self.workspace}")
        if not os.path.exists(self.hdf5_path):
            with gzip.open(gzipped_h5, "rb") as f_in:
                with open(self.hdf5_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

    def __clean_workspace(self):
        print(f"Cleaning workspace at {self.workspace}")
        dataset_name = os.path.basename(self.dataset_loc).split('.')[0]
        stat_file_loc = os.path.join(self.workspace, dataset_name + "_stats.bin")

        # Remove HDF5 file if it exists
        if os.path.isfile(self.hdf5_path):
            os.remove(self.hdf5_path)

        # Remove stats file if it exists
        if os.path.isfile(stat_file_loc):
            os.remove(stat_file_loc)

    def do_run(self):

        self.__setup_workspace()

        class_weights_modes = {
            "default": 0,
            "fcb": 1,
            "mfcb": 2
        }

        match class_weights_modes[self.class_weights_mode]:
            case 0:
                RunTrainingUtility.run_hdf5_train_UNet_3D_Segmentation(
                    self.model_name, self.hdf5_path, self.model_weights_loc, self.epochs, self.backup_epochs, self.batch_size,
                    self.lr, self.decay_order, self.split, self.use_amp, self.val_batch_factor, self.workers,
                    n_classes = self.n_classes, model_seed=self.model_seed, model_type=self.model_type,
                    raw_ep_resume=str(self.resume_epoch)
                )
            case 1:
                RunTrainingUtility.run_hdf5_train_fcb_UNet_3D_Segmentation(
                    self.model_name, self.hdf5_path, self.model_weights_loc, self.epochs, self.backup_epochs, self.batch_size,
                    self.lr, self.decay_order, self.split, self.use_amp, self.val_batch_factor, self.workers,
                    n_classes=self.n_classes, model_seed=self.model_seed, model_type=self.model_type,
                    raw_ep_resume=str(self.resume_epoch)
                )
            case 2:
                RunTrainingUtility.run_hdf5_train_mfcb_UNet_3D_Segmentation(
                    self.model_name, self.hdf5_path, self.model_weights_loc, self.epochs, self.backup_epochs, self.batch_size,
                    self.lr, self.decay_order, self.split, self.use_amp, self.val_batch_factor, self.workers,
                    n_classes=self.n_classes, model_seed=self.model_seed, model_type=self.model_type,
                    raw_ep_resume=str(self.resume_epoch)
                )

        self.__clean_workspace()

def setup_runs():
    local_1 = LocalRun(
        model_name="test_model",
        dataset_loc=r"/mnt/h/abc_ks16_rot_InOut_1f0_crp20000",
        workspace=r"/mnt/h/ws_training_local",
        model_weights_loc=r"/mnt/h/ws_training_local/model_weights/{model_name}/{run_name}_save_{epoch}.pth",
        epochs=2,
        backup_epochs= 1,
        batch_size=4,
        lr=1e-5,
        decay_order=-0.03,
        n_classes=8,
        model_type="UNet_16EL",
        model_seed=1337,
        class_weights_mode="mfcb",
        workers=1 #for windows workers = 0
    )

    local_1.do_run()

def main():
    setup_runs()

if __name__=="__main__":
    main()