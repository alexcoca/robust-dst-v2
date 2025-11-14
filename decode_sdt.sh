#!/bin/bash
#SBATCH -A GASIC-BHT26-SL2-GPU
#SBATCH -J SDT-DECODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:20:00
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
eval "$(conda shell.bash hook)"
conda deactivate
conda info --envs
conda activate robust-dst

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


ckpt=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}
mkdir -p logs/"${EXPERIMENT_NAME}"
LOG=logs/${EXPERIMENT_NAME}/${ckpt}.log
ERR=logs/${EXPERIMENT_NAME}/${ckpt}.err

PREFIX=/home/ac2123/rds/rds-wjb31-nmt2020/ac2123/robust-dst-v2/models
VERSION=1
INFERENCE_BATCH_SIZE=128
CHECKPOINT_DIR=${PREFIX}/${EXPERIMENT_NAME}/version_1/checkpoint-${ckpt}

python -m scripts.sdt_inference \
  --model_name_or_path $CHECKPOINT_DIR \
  --output_dir $CHECKPOINT_DIR \
  --test_file data/processed/SGD_SDT/"${PROMPT_ID}"/"${SPLIT}"/version_$VERSION/data.json \
  --start_batch_size $INFERENCE_BATCH_SIZE \
  --max_target_length 512 > $LOG 2> $ERR
