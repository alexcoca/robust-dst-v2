#!/bin/bash
#SBATCH -A GASIC-BHT26-SL2-GPU
#SBATCH -J D3ST-DECODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:55:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ac2123@cam.ac.uk
#SBATCH --array=0-59
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere
#! ############################################################

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp
export OMP_NUM_THREADS=1

module load slurm
module load anaconda/3.2019-10
eval "$(conda shell.bash hook)"
conda activate robust-dst

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