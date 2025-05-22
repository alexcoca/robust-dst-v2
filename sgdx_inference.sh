#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

VERSION=1
PROC_NUM_WORK=16
WANDB_ENTITY=byrne-lab
WANDB_PROJECT=pegasussa
INFERENCE_BATCH_SIZE=64
LOGFILE="run.log"
declare -a SHARDS=("original" "v1" "v2" "v3" "v4" "v5")
#declare -a CKPTS=("/scratch/dev/robust-dst-v2/models/seed_20230110_d3st_centroids/version_1/checkpoint-20000")
declare -a CKPTS=("/scratch/dev/robust-dst-v2/models/seed_20230111_d3st_centroids/version_1/checkpoint-30000")
for SGD_SHARD in "${SHARDS[@]}"; do
  for CHECKPOINT_DIR in "${CKPTS[@]}"; do
    printf '%s  shard=%s  ckpt=%s\n' "$(date -Iseconds)" "$SGD_SHARD" \
           "$(basename "$CHECKPOINT_DIR")" >> "$LOGFILE"

    python -m scripts.run_dialogue_state_tracking \
      --model_name_or_path "$CHECKPOINT_DIR" \
      --output_dir "$CHECKPOINT_DIR" \
      --cache_dir cache \
      --test_file data/processed/"$SGD_SHARD"/test/version_$VERSION/data.json \
      --test_template_dir data/interim/blank_dialogue_templates/"$SGD_SHARD"/test \
      --test_ref_dir data/raw/"$SGD_SHARD"/test \
      --preprocessing_num_workers "$PROC_NUM_WORK" \
      --per_device_eval_batch_size "$INFERENCE_BATCH_SIZE" \
      --report_to wandb \
      --wandb_entity "$WANDB_ENTITY" \
      --wandb_project "$WANDB_PROJECT" \
      --do_predict \
      --predict_with_generate \
      --max_target_length 512 \
      --val_max_target_length 512
  done
done