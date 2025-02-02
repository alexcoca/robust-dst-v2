#!/bin/bash
#SBATCH -A GASIC-BHT26-SL2-GPU
#SBATCH -J D3ST-TRAIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ac2123@cam.ac.uk
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

if [ -z ${CONFIG_FILE+x} ]; then
  echo "Please specify the json file with the training args"
  exit
fi

LOG=logs/train_"$SLURM_JOB_ID"_replicate_d3st.log
ERR=logs/train_"$SLURM_JOB_ID"_replicate_d3st.err

torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_backend c10d \
  --rdzv_endpoint localhost:0 \
  -m scripts.run_dialogue_state_tracking "$CONFIG_FILE" > $LOG 2> $ERR