@echo off
REM Batch script to launch python
REM --- Configuration ---

REM Path to your main entrypoint and venv  (using relative path from script location)
SET "VENV_PATH=C:\src\repos\PytorchDL\.venv\Scripts\activate.bat"
SET "MAIN_PY=C:\src\repos\PytorchDL\main.py"
REM Args for main scripts
SET "MODULE=data_utility"
SET "MODE=batch_subsets"
SET "SOURCE=H:\ABC\ABC_torch\ABC_chunk_01\torch_data_ks_16_pad_4_bw_5_vs_adaptive_n2"
SET "TARGET=H:\ABC\ABC_torch\ABC_chunk_01\batch_data_ks_16_pad_4_bw_5_vs_adaptive_n2"
SET "DATASET_NAME=torch_batched"
SET "BATCH_COUNT=20"


call %VENV_PATH%

"%MAIN_PY%" "%MODULE%" "%MODE%" "%SOURCE%" "%TARGET%" "%DATASET_NAME%" "%BATCH_COUNT%"
deactivate

pause