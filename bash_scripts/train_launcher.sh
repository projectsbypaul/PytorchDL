#!/bin/bash
# train_launcher.sh -- Bash script to run training with separate output and error logs

# --- Configuration ---
MAIN_PY="/workspace/PytorchDL/main.py"  # Adjust as needed
SAMPLES="1f0_mio"
MODULE="train_utility"
MODE="train_hdf5_UNet3D"
MODEL_NAME="UNet3D_SDF_16EL_n_class_10_multiset_${SAMPLES}"
DATASET_DIR="/datasets/train_${SAMPLES}_ks_16_pad_4_bw_5_vs_adaptive_n3.hdf5"
RUN_NAME="${MODEL_NAME}"
MODEL_WEIGHTS_LOC="/model_weights/${MODEL_NAME}/${RUN_NAME}_save_{epoch}.pth"
EPOCHS=2
BACKUP_EP=1
BATCH_SIZE=16
LR=1e-4
DECAY_ORDER=1e-1
SPLIT=0.9
USE_AMP="True"
VAL_BATCH_FACTOR=4
WORKERS=14

# Log file locations
LOG_DIR="./workspace/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_LOG="${LOG_DIR}/output_${RUN_NAME}_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/error_${RUN_NAME}_${TIMESTAMP}.log"

# Print some info
echo "Starting training: ${RUN_NAME}"
echo "Stdout will go to: $OUTPUT_LOG"
echo "Stderr will go to: $ERROR_LOG"
echo

# Run the training script and log output/error separately
python "$MAIN_PY" \
    $MODULE \
    $MODE \
    $MODEL_NAME \
    $DATASET_DIR \
    "$MODEL_WEIGHTS_LOC" \
    $EPOCHS \
    $BACKUP_EP \
    $BATCH_SIZE \
    $LR \
    $DECAY_ORDER \
    $SPLIT \
    $USE_AMP \
    $VAL_BATCH_FACTOR \
    $WORKERS \
    > "$OUTPUT_LOG" 2> "$ERROR_LOG"

EXIT_CODE=$?

echo
echo "Training finished with exit code $EXIT_CODE"
echo "Output log: $OUTPUT_LOG"
echo "Error log: $ERROR_LOG"

