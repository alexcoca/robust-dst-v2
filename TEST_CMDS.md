torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_backend c10d \
  --rdzv_endpoint localhost:0 \
  -m scripts.run_dialogue_state_tracking "configs/d3st_flant5_large_test_run_sanity_check.json" 2>&1 | tee test_d3st_flant5_setup.log

torchrun \
  --nproc_per_node 1 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_backend c10d \
  --rdzv_endpoint localhost:0 \
  -m scripts.run_dialogue_state_tracking "configs/replicate_sdt_v0.json" 2>&1 | tee test_sdt_setup.log




VERSION=1
PROC_NUM_WORK=128
INFERENCE_BATCH_SIZE=128 
CHECKPOINT_DIR=/home/ac2123/rds/rds-wjb31-nmt2020/ac2123/robust-dst-v2/models/seed_420_replicate_sdt_v0/version_1/checkpoint-2500


python -m scripts.run_dialogue_state_tracking \
  --model_name_or_path $CHECKPOINT_DIR \
  --output_dir $CHECKPOINT_DIR \
  --cache_dir cache \
  --test_file data/processed/original/dev/version_$VERSION/data.json \
  --test_template_dir data/interim/blank_dialogue_templates/original/dev \
  --test_ref_dir data/raw/original/dev \
  --preprocessing_num_workers $PROC_NUM_WORK \
  --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
  --do_predict \
  --report_to "none"\
  --predict_with_generate \
  --max_target_length 512 \
  --val_max_target_length 512



EXPERIMENT_NAME=seed_4_d3st_flant5_large_test_sanity_check 
SPLIT=dev 
CHECKPOINTS=900,1800
$SLURM_ARRAY_TASK_ID=1

if [ -z ${EXPERIMENT_NAME+x} ]; then
  echo "Please prepend the experiment name eg EXPERIMENT_NAME=cool_experiment."
  exit
fi
if [ -z ${SPLIT+x} ]; then
  echo "Please specify the split: dev|test"
  exit
fi
if [ -z ${CHECKPOINTS+x} ]; then
  CHECKPOINTS=(
    900 1800 2700 3600 4500 5400 6300 7200 8100 9000 9900 10800 11700 12600 13500 14400 15300 16200 17100 18000 18900 19800 20700 21600 22500 23400 24300 25200 26100 27000 27900 28800 29700 30600 31500 32400 33300 34200 35100 36000 36900 37800 38700 39600 40500 41400 42300 43200 44100 45000 45900 46800 47700 48600 49500 50400 51300 52200 53100 54000 54900
  )
else
  IFS=',' read -r -a CHECKPOINTS <<< "$CHECKPOINTS"
fi
if [ "$SPLIT" == "test" ] && [ -z ${CHECKPOINTS+x} ]; then
  echo "Error: CHECKPOINTS must be specified when SPLIT is set to 'test'. Exiting."
  exit 1
fi

ckpt=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}


LOG=logs/"$SLURM_JOB_ID"_replicate_d3st_${SPLIT}_${ckpt}.log
ERR=logs/"$SLURM_JOB_ID"_replicate_d3st_${SPLIT}_${ckpt}.err

VERSION=1
PROC_NUM_WORK=128
INFERENCE_BATCH_SIZE=128
CHECKPOINT_DIR=/home/ac2123/rds/rds-wjb31-nmt2020/ac2123/robust-dst-v2/models/${EXPERIMENT_NAME}/version_1/checkpoint-${ckpt}

python -m scripts.run_dialogue_state_tracking \
  --model_name_or_path $CHECKPOINT_DIR \
  --output_dir $CHECKPOINT_DIR \
  --cache_dir cache \
  --test_file data/processed/original/"${SPLIT}"/version_$VERSION/data.json \
  --test_template_dir data/interim/blank_dialogue_templates/original/"${SPLIT}" \
  --test_ref_dir data/raw/original/"${SPLIT}" \
  --preprocessing_num_workers $PROC_NUM_WORK \
  --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
  --do_predict \
  --report_to "none"\
  --predict_with_generate \
  --max_target_length 512 \
  --val_max_target_length 512 > $LOG 2> $ERR


VERSION=1
PROC_NUM_WORK=128
INFERENCE_BATCH_SIZE=128
ckpt=3200
SPLIT=dev
PROMPT_ID=v0
EXPERIMENT_NAME=seed_420_replicate_sdt_v0
CHECKPOINT_DIR=/home/ac2123/rds/rds-wjb31-nmt2020/ac2123/robust-dst-v2/models/${EXPERIMENT_NAME}/version_1/checkpoint-${ckpt}

python -m scripts.sdt_inference \
  --model_name_or_path $CHECKPOINT_DIR \
  --output_dir $CHECKPOINT_DIR \
  --test_file data/processed/SGD_SDT/"${PROMPT_ID}"/"${SPLIT}"/version_$VERSION/data.json \
  --start_batch_size $INFERENCE_BATCH_SIZE \
  --max_target_length 512 2>&1 | tee debug_inference.log
