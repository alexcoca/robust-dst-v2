#!/bin/bash

# Ensure the user provides necessary arguments
if [ -z "${EXPERIMENT_NAME+x}" ]; then
  echo "Error: Please set EXPERIMENT_NAME."
  exit 1
fi

if [ -z "${KEEP_CHECKPOINTS+x}" ]; then
  echo "Error: Please specify the checkpoints to keep as a comma-separated list."
  exit 1
fi

# Convert the comma-separated list to an array
IFS=',' read -r -a KEEP_ARRAY <<< "$KEEP_CHECKPOINTS"

# Define the base directory where checkpoints are stored
PREFIX="/home/ac2123/rds/rds-wjb31-nmt2020/ac2123/robust-dst-v2/models"
CHECKPOINT_DIR="${PREFIX}/${EXPERIMENT_NAME}/version_1"

# Create a set of checkpoints to keep (for faster lookups)
declare -A KEEP_SET
for ckpt in "${KEEP_ARRAY[@]}"; do
    KEEP_SET["checkpoint-$ckpt"]=1
done

# Find and delete checkpoints not in the keep list
for dir in "${CHECKPOINT_DIR}"/checkpoint-*; do
    ckpt_name=$(basename "$dir")

    if [[ ! ${KEEP_SET[$ckpt_name]+_} ]]; then
        echo "Deleting $ckpt_name..."
        rm -rf "$dir"
    else
        echo "Keeping $ckpt_name."
    fi
done

echo "Checkpoint cleanup complete!"
