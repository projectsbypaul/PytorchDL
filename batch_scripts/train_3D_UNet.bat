@echo off
REM Batch script to launch python
REM --- Configuration ---

REM Path to your main entrypoint and venv  (using relative path from script location)
SET "VENV_PATH=C:\Users\pschuster\source\repos\PytorchDL\.venv\Scripts\activate.bat"
SET "MAIN_PY=C:\Users\pschuster\source\repos\PytorchDL\main.py"
REM Arg for CLI
SET "MODULE=train_utility"
SET "MODE=train_UNet3D"
REM Args for training script
SET "MODEL_NAME=UNet3D_SDF_16EL_n_class_10_multiset_1f0_mio"
SET "DATASET_DIR=H:\ABC\ABC_torch\ABC_training\train_1000000_ks_16_pad_4_bw_5_vs_adaptive_n3\batch_iter_01"
SET "MODEL_WEIGHTS_LOC=C:\src\repos\PytorchDL\data\model_weights\{model_name}\{run_name}_save_{epoch}.pth"
SET "EPOCHS=200"
SET "BACKUP_EP=10"
SET "BATCH_SIZE=16"
SET "LR=1e-4"
SET "DECAY_ORDER=1e-1"
SET "SPLIT=0.9"

call %VENV_PATH%

"%MAIN_PY%" "%MODULE%" "%MODE%" "%MODEL_NAME%" "%DATASET_DIR%" "%MODEL_WEIGHTS_LOC%" "%EPOCHS%" "%BACKUP_EP%" "%BATCH_SIZE%" "%LR%" "%DECAY_ORDER%" "%SPLIT%"

pause

call deactivate
