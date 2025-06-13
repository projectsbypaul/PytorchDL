@echo off
REM Batch script to launch python
REM --- Configuration ---

REM Path to your main entrypoint and venv  (using relative path from script location)
SET "VENV_PATH=C:\src\repos\PytorchDL\.venv\Scripts\activate.bat"
SET "MAIN_PY=C:\src\repos\PytorchDL\main.py"
REM Arg for CLI
SET "MODULE=train_utility"
SET "MODE=train_UNet3D"
REM Args for training script
SET "MODEL_NAME=UNet3D_SDF_16EL_n_class_10_multiset"
SET "DATASET_DIR=H:\ABC\ABC_torch\ABC_chunk_00\batched_data_ks_16_pad_4_bw_5_vs_adaptive_n2_testing"
SET "MODEL_WEIGHTS_LOC=C:\src\repos\PytorchDL\data\model_weights\{model_name}\{run_name}_save_{epoch}.pth"
SET "EPOCHS=200"
SET "BACKUP_EP=100"
SET "BATCH_SIZE=4"
SET "LR=1e-4"
SET "DECAY_ORDER=1e-1"
SET "SPLIT=0.8"

call %VENV_PATH%

"%MAIN_PY%" "%MODULE%" "%MODE%" "%MODEL_NAME%" "%DATASET_DIR%" "%MODEL_WEIGHTS_LOC%" "%EPOCHS%" "%BACKUP_EP%" "%BATCH_SIZE%" "%LR%" "%DECAY_ORDER%" "%SPLIT%"

pause

call deactivate
