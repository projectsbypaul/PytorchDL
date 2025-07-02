@echo off
REM Batch script to launch training

REM --- Configuration ---
SET "VENV_PYTHON=C:\Users\pschuster\source\repos\PytorchDL\.venv\Scripts\python.exe"
SET "MAIN_PY=C:\Users\pschuster\source\repos\PytorchDL\main.py"

SET "MODULE=train_utility"
SET "MODE=train_UNet3D"
SET "MODEL_NAME=UNet3D_SDF_16EL_n_class_10_multiset_500k"
SET "DATASET_DIR=H:\ABC\ABC_torch\ABC_training\train_500k_ks_16_pad_4_bw_5_vs_adaptive_n3\batch_iter_01"

REM Construct weights path dynamically
SET "RUN_NAME=%MODEL_NAME%"
SET "MODEL_WEIGHTS_LOC=C:\src\repos\PytorchDL\data\model_weights\{model_name}\{run_name}_save_{epoch}.pth"

SET "EPOCHS=200"
SET "BACKUP_EP=10"
SET "BATCH_SIZE=16"
SET "LR=1e-4"pi
SET "DECAY_ORDER=1e-1"
SET "SPLIT=0.9"

REM Run the training script using the virtual environment's Python
"%VENV_PYTHON%" "%MAIN_PY%" %MODULE% %MODE% %MODEL_NAME% %DATASET_DIR% "%MODEL_WEIGHTS_LOC%" %EPOCHS% %BACKUP_EP% %BATCH_SIZE% %LR% %DECAY_ORDER% %SPLIT%

echo.
echo Training finished with code %ERRORLEVEL%
pause
