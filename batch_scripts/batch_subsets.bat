@echo off
REM Batch script to launch python
REM --- Configuration ---

REM Path to your main entrypoint and venv  (using relative path from script location)
SET "VENV_PATH=C:\Users\pschuster\source\repos\PytorchDL\.venv\Scripts"
SET "MAIN_PY=C:\Users\pschuster\source\repos\PytorchDL\main.py"
REM Args for main scripts
SET "MODULE=data_utility"
SET "MODE=batch_subsets"
SET "SOURCE=H:\ABC\ABC_torch\ABC_training\train_500k_ks_16_pad_4_bw_5_vs_adaptive_n3\subset"
SET "TARGET=H:\ABC\ABC_torch\ABC_training\train_500k_ks_16_pad_4_bw_5_vs_adaptive_n3\batch_iter_01"
SET "DATASET_NAME=batch_iter_01"
SET "BATCH_COUNT=20"

IF NOT EXIST "%TARGET%" (
    echo Creating glob target directory: %TARGET%
    MKDIR "%TARGET%"
)

call "%VENV_PATH%\python.exe" "%MAIN_PY%" "%MODULE%" "%MODE%" "%SOURCE%" "%TARGET%" "%DATASET_NAME%" "%BATCH_COUNT%"

pause