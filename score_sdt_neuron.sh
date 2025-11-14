#!/bin/bash
conda deactivate
conda activate robust-dst
# Check required environment variables
if [ -z ${EXPERIMENT_NAME+x} ]; then
  echo "Please set EXPERIMENT_NAME."
  exit 1
fi
if [ -z ${PROMPT_ID+x} ]; then
  echo "Please set PROMPT_ID (e.g., v0, v1, v2, etc.)."
  exit 1
fi
if [ -z ${SPLIT+x} ]; then
  echo "Please specify the split: dev | test."
  exit 1
fi
if [ -z ${CHECKPOINTS+x} ]; then
  CHECKPOINTS=( $(seq 1600 1600 55000) ) # Default checkpoint range
else
  IFS=',' read -r -a CHECKPOINTS <<< "$CHECKPOINTS"
fi
if [ "$SPLIT" == "test" ] && [ -z ${CHECKPOINTS+x} ]; then
  echo "Error: CHECKPOINTS must be specified when SPLIT is set to 'test'. Exiting."
  exit 1
fi
if [ -z ${SAVE_FILES+x} ]; then
  echo "Frame metrics wil not be saved."
  SAVE_FILES=false
fi

# Define paths
VERSION=1
PREFIX="/scratch/dev/robust-dst-v2"
LOG_FILE="scoring_status-${EXPERIMENT_NAME}-${SPLIT}-${VERSION}.txt"

# Create log file if not exists
if [ ! -f "$LOG_FILE" ]; then
    true > "$LOG_FILE"
fi
echo "$(date +'%Y-%m-%d %H:%M:%S') - Running scoring for $EXPERIMENT_NAME - Split: $SPLIT" >> "$LOG_FILE"

# Loop through all checkpoints
for ckpt in "${CHECKPOINTS[@]}"; do
    PREDICTION_FILE_PATH="${PREFIX}/models/${EXPERIMENT_NAME}/version_1/checkpoint-${ckpt}/generated_predictions.txt"
    REFS_FILE_PATH="${PREFIX}/data/processed/SGD_SDT/${PROMPT_ID}/${SPLIT}/version_${VERSION}/data.json"
    OUT_DIR="${PREFIX}/metrics/${EXPERIMENT_NAME}/original/${SPLIT}/version_${VERSION}"
    HYP_DIR="${PREFIX}/hyps/${EXPERIMENT_NAME}/original/${SPLIT}/version_${VERSION}"
    mkdir -p "$OUT_DIR"
    mkdir -p "$HYP_DIR"
    OUT_FILE="${OUT_DIR}/metrics_${ckpt}.json"

    echo "Scoring checkpoint: $ckpt" | tee -a "$LOG_FILE"

    python scripts/sdt_scoring.py \
        --demonstration_id "${PROMPT_ID}" \
        --predictions_file "${PREDICTION_FILE_PATH}" \
        --refs_file "${REFS_FILE_PATH}" \
        --data_type "${SPLIT}" \
        --output_file "${OUT_FILE}" \
        --hyp_dir "${HYP_DIR}" \
        --save_files "${SAVE_FILES}" \

    if [ $? -eq 0 ]; then
        echo "$(date +'%Y-%m-%d %H:%M:%S') - Checkpoint $ckpt - SUCCESS" >> "$LOG_FILE"
    else
        echo "$(date +'%Y-%m-%d %H:%M:%S') - Checkpoint $ckpt - FAILURE" >> "$LOG_FILE"
    fi
done

echo "Scoring completed for $EXPERIMENT_NAME." | tee -a "$LOG_FILE"
