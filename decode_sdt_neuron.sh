#!/bin/bash

if [ -z ${EXPERIMENT_NAME+x} ]; then
  echo "Please prepend the experiment name eg EXPERIMENT_NAME=cool_experiment."
  exit
fi
if [ -z ${PROMPT_ID+x} ]; then
  echo "Please specify the prompt index: PROMPT_ID=(v0|v1|v2|v3|v4)"
  exit
fi
if [ -z ${SPLIT+x} ]; then
  echo "Please specify the split: dev|test"
  exit
fi
if [ -z ${CHECKPOINTS+x} ]; then
  CHECKPOINTS=(
    $(seq 1600 1600 55000)
  )
else
  IFS=',' read -r -a CHECKPOINTS <<< "$CHECKPOINTS"
fi
if [ "$SPLIT" == "test" ] && [ -z ${CHECKPOINTS+x} ]; then
  echo "Error: CHECKPOINTS must be specified when SPLIT is set to 'test'. Exiting."
  exit 1
fi

INFERENCE_BATCH_SIZE=64
file_name="decoding_status-$EXPERIMENT_NAME-$EXP_NAME_SUFFIX-$SPLIT-$VERSION.txt"

if [ ! -f "$file_name" ]; then
    true > $file_name
fi
echo "$(date +'%Y-%m-%d %H:%M:%S') - $EXPERIMENT_NAME - $VERSION - $SPLIT" >> "$file_name"

for ckpt in "${CHECKPOINTS[@]}"; do
    mkdir -p logs/"${EXPERIMENT_NAME}"
    LOG=logs/${EXPERIMENT_NAME}/${ckpt}.log
    ERR=logs/${EXPERIMENT_NAME}/${ckpt}.err

    CHECKPOINT_DIR=${PREFIX}/${EXPERIMENT_NAME}/version_1/checkpoint-${ckpt}

    echo "Running inference for checkpoint: $ckpt" | tee -a "$file_name"

    python -m scripts.sdt_inference \
      --model_name_or_path $CHECKPOINT_DIR \
      --output_dir $CHECKPOINT_DIR \
      --test_file data/processed/SGD_SDT/"${PROMPT_ID}"/"${SPLIT}"/version_$VERSION/data.json \
      --start_batch_size $INFERENCE_BATCH_SIZE \
      --max_target_length 512 > $LOG 2> $ERR

    if [ $? -eq 0 ]; then
        echo "$(date +'%Y-%m-%d %H:%M:%S') - Checkpoint $ckpt - SUCCESS" >> "$file_name"
    else
        echo "$(date +'%Y-%m-%d %H:%M:%S') - Checkpoint $ckpt - FAILURE" >> "$file_name"
    fi
done
