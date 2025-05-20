#!/bin/bash
#SBATCH -A GASIC-BHT26-SL2-GPU
#SBATCH -J D3ST-TRAIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ac2123@cam.ac.uk
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere
#! ############################################################

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp
export OMP_NUM_THREADS=1
module load miniconda/3
eval "$(conda shell.bash hook)"
which python
conda activate /home/ac2123/anaconda3/envs/robust-dst
which python

workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.
if [ -z ${CONFIG_FILE+x} ]; then
  echo "Please specify the json file with the training args"
  exit
fi

LOG=logs/train_"$SLURM_JOB_ID"_replicate_d3st.log
ERR=logs/train_"$SLURM_JOB_ID"_replicate_d3st.err

CMD="torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_backend c10d \
  --rdzv_endpoint localhost:0 \
  -m scripts.run_dialogue_state_tracking $CONFIG_FILE"


cd "$workdir" || exit
echo -e "Changed directory to $(pwd).\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: $(date)"
echo "Running on master node: $(hostname)"
echo "Current directory: $(pwd)"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=$(generate_pbs_nodefile)
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD