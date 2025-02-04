#!/bin/bash

# Ensure required variables are set
if [ -z "${EXPERIMENT_NAME+x}" ]; then
  echo "Error: Please set EXPERIMENT_NAME."
  exit 1
fi

if [ -z "${KEEP_CHECKPOINTS+x}" ]; then
  echo "Error: Please specify the checkpoints to keep as a comma-separated list."
  exit 1
fi

# Convert comma-separated list to an array
IFS=',' read -r -a KEEP_ARRAY <<< "$KEEP_CHECKPOINTS"

# Define the checkpoint directory
PREFIX="/scratch/dev/robust-dst-v2/models"
CHECKPOINT_DIR="${PREFIX}/${EXPERIMENT_NAME}/version_1"

# Verify if the directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

# Convert to a set for fast lookup
declare -A KEEP_SET
for ckpt in "${KEEP_ARRAY[@]}"; do
    KEEP_SET["checkpoint-$ckpt"]=1
done

# Track if any deletion happens
deleted_any=false

# Iterate over checkpoint directories and delete the ones not in KEEP_SET
for dir in "${CHECKPOINT_DIR}"/checkpoint-*; do
    # Skip if no matching directories are found
    [ -e "$dir" ] || continue

    ckpt_name=$(basename "$dir")

    if [[ ! ${KEEP_SET[$ckpt_name]+_} ]]; then
        echo "Deleting $ckpt_name..."
        rm -rf "$dir"
        deleted_any=true
    else
        echo "Keeping $ckpt_name."
    fi
done

if ! $deleted_any; then
    echo "No checkpoints were deleted. Make sure the specified ones exist."
fi

echo "Checkpoint cleanup complete!"
