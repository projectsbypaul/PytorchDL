@echo off
REM Batch script to launch python
REM --- Configuration ---

REM Path to your main entrypoint and venv  (using relative path from script location)
SET "VENV_PATH=C:\Users\pschuster\source\repos\PytorchDL\.venv\Scripts\activate.bat"
SET "MAIN_PY=C:\Users\pschuster\source\repos\PytorchDL\main.py"
REM Args for main scripts
SET "SAMPLES=1000000"
SET "MODULE=data_utility"
SET "MODE=create_subsets"
SET "JOB_LOCATION=h:\ABC\ABC_jobs\job_train\job_train_%SAMPLES%"
SET "INPUT_DIR=H:\ABC\ABC_Datasets\Segmentation\training_samples\train_%SAMPLES%_ks_16_pad_4_bw_5_vs_adaptive_n3"
SET "OUTPUT_DIR=H:\ABC\ABC_torch\ABC_training\train_%SAMPLES%_ks_16_pad_4_bw_5_vs_adaptive_n3\subset"
SET "MIN_FILES=2"
SET "N_THREADS=16"


IF NOT EXIST "%OUTPUT_DIR%" (
    echo Creating glob target directory: %OUTPUT_DIR%
    MKDIR "%OUTPUT_DIR%"
)


call %VENV_PATH%

FOR /L %%I IN (1,1,%N_THREADS%) DO (
    SETLOCAL ENABLEDELAYEDEXPANSION
    SET "ID=00%%I"
    SET "ID=!ID:~-3!"
    SET "INSTANCE_ID=Instance!ID!"
    SET "JOB_FILE=%JOB_LOCATION%\!INSTANCE_ID!.job"
    echo Starting !INSTANCE_ID!...
    START "!INSTANCE_ID! Process" /B %MAIN_PY% %MODULE% %MODE% "!JOB_FILE!" %INPUT_DIR% %OUTPUT_DIR% %MIN_FILES%
    ENDLOCAL
)

pause
deactivate
