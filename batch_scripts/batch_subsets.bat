@echo off
REM Batch script to launch python
REM --- Configuration ---

REM Path to your main entrypoint and venv  (using relative path from script location)
SET "VENV_PATH=C:\src\repos\PytorchDL\.venv\Scripts\activate.bat"
SET "MAIN_PY=C:\src\repos\PytorchDL\main.py"
REM Args for main scripts
SET "MODULE=data_utility"
SET "MODE=batch_subsets"
SET "SOURCE=H:\ABC\ABC_torch\ABC_training\train_1000000_ks_16_pad_4_bw_5_vs_adaptive_n3\subset"
SET "TARGET=H:\ABC\ABC_torch\ABC_training\train_1000000_ks_16_pad_4_bw_5_vs_adaptive_n3\batch_iter_01"
SET "DATASET_NAME=batch_iter_01"
SET "BATCH_COUNT=40"

IF NOT EXIST "%TARGET%" (
    echo Creating glob target directory: %TARGET%
    MKDIR "%TARGET%"
)



call %VENV_PATH%

"%MAIN_PY%" "%MODULE%" "%MODE%" "%SOURCE%" "%TARGET%" "%DATASET_NAME%" "%BATCH_COUNT%"
deactivate

pause