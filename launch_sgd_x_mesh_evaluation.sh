#!/bin/bash
#!
#! Example SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#! Last updated: Fri 30 Jul 11:07:58 BST 2021
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J SGDX-INFERENCE
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A GASIC-BHT26-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 32 cpus per GPU.
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=0:30:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ac2123@cam.ac.uk
#SBATCH --array=0-5
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by GPU number*walltime.

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
module load miniconda/3
eval "$(conda shell.bash hook)"
conda activate /home/ac2123/anaconda3/envs/robust-dst
which python

SHARDS=("original" "v1" "v2" "v3" "v4" "v5")
SGD_SHARD=${SHARDS[$SLURM_ARRAY_TASK_ID]}

if [ -z ${CHECKPOINT_DIR+x} ]; then
  echo "Please pass the path to the directory of the checkpoint you want to run inference on by prepending
  CHECKPOINT_DIR=abs/path/to/the/args to your command. This should end in */checkpoint-[STEP]."
  exit
fi
if [ -z ${VERSION+x} ]; then
  echo "Please specify the version of the data you want to run inference for by prepending VERSION=int to the command. 
  For example, to run inference on version 1 pass prepend VERSION=1. This should match the /version_*/ portion of
  CHECKPOINT_DIR"
  exit
fi
if [ -z ${PROC_NUM_WORK+x} ]; then
  PROC_NUM_WORK=128
  echo "Preprocessing data on $PROC_NUM_WORK cores"
fi
if [ -z ${BATCH_SIZE+x} ]; then
  BATCH_SIZE=200
  echo "Decoding with batch size $BATCH_SIZE"
fi
#! Full path to application executable:
application="python -m src.run_dialogue_state_tracking"

#! Run options for the application:
options="--model_name_or_path $CHECKPOINT_DIR \
--output_dir $CHECKPOINT_DIR \
--cache_dir cache_$SLURM_JOB_ID \
--test_file data/preprocessed/$SGD_SHARD/test/version_$VERSION/data.json \
--test_template_dir data/interim/blank_dialogue_templates/$SGD_SHARD/test \
--test_ref_dir data/raw/sgd_x/$SGD_SHARD/test \
--preprocessing_num_workers $PROC_NUM_WORK \
--per_device_eval_batch_size $BATCH_SIZE \
--report_to wandb \
--wandb_entity byrne-lab \
--wandb_project pegasussa \
--do_predict \
--predict_with_generate \
--max_target_length 512 \
--val_max_target_length 512"

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

#! Choose this for a MPI code using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD
