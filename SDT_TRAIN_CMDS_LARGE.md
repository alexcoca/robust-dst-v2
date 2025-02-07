# v0

python scripts/run_dialogue_state_tracking.py "configs/replicate_sdt_v0_neuron_large.json"

## Inference

PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=1600 EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large sbatch -J decode_dev --array=0-0 --time=00:35:00 decode_sdt.sh
Submitted batch job 4727030 (done)
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=3200 EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large bash decode_sdt.sh && exit
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=4800 EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large bash decode_sdt.sh && exit
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=8000 EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large sbatch -J decode_dev --array=0-0 --time=00:35:00 decode_sdt.sh
Submitted batch job 4734015 (done)
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=11200,12800 EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large bash decode_sdt_intr.sh 
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=6400,9600 EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large bash decode_sdt_intr.sh 
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=14400 EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large decode_sdt_neuron.sh
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=16000 EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large sbatch -J decode_dev --array=0-0 --time=00:28:00 decode_sdt.sh
Submitted batch job 4744509 (done)
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=17600 EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large sbatch -J decode_dev --array=0-0 --time=00:28:00 decode_sdt.sh
Submitted batch job 4745306 (done)
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=$(seq 35200 -1600 20800 | paste -sd ",") EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large sbatch -J decode_dev --array=0-9 --time=00:24:00 decode_sdt.sh
Submitted batch job 4755198
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=20800,19200 EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large bash decode_sdt_intr.sh && exit
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=22400,24000 EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large bash decode_sdt_intr.sh && exit
PROMPT_ID=v0 SPLIT=test CHECKPOINTS=4800 EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large bash decode_sdt_intr.sh && exit




## Scoring 

EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=$(seq 1600 1600 22400 | paste -sd ",") bash score_sdt.sh
EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large PROMPT_ID=v0 SPLIT=test CHECKPOINTS=4800 bash score_sdt.sh


## Clean-up 


EXPERIMENT_NAME=seed_420_replicate_sdt_v0_large KEEP_CHECKPOINTS=$(echo $(seq 24400 1600 41600) 4800 | tr ' ' '\n' | sort -n | paste -sd ",") bash cleanup_checkpoints.sh

# v1 

python scripts/run_dialogue_state_tracking.py "configs/replicate_sdt_v1_neuron_large.json"

## Inference

PROMPT_ID=v1 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v1_large CHECKPOINTS=1600,3200 bash decode_sdt_intr.sh && exit
PROMPT_ID=v1 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v1_large CHECKPOINTS=4800,6400 bash decode_sdt_intr.sh && exit
PROMPT_ID=v1 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v1_large CHECKPOINTS=8000 bash decode_sdt_intr.sh && exit
PROMPT_ID=v1 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v1_large CHECKPOINTS=9600 bash decode_sdt_intr.sh && exit
PROMPT_ID=v1 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v1_large CHECKPOINTS=11200,12800 bash decode_sdt_intr.sh && exit
PROMPT_ID=v1 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v1_large CHECKPOINTS=14400,16000 bash decode_sdt_intr.sh && exit
PROMPT_ID=v1 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v1_large CHECKPOINTS=9600,17600 bash decode_sdt_intr.sh && exit

- current interactive:
PROMPT_ID=v1 SPLIT=test EXPERIMENT_NAME=seed_420_replicate_sdt_v1_large CHECKPOINTS=4800 bash decode_sdt_intr.sh && exit

- next interactive: - 


## Scoring
EXPERIMENT_NAME=seed_420_replicate_sdt_v1_large PROMPT_ID=v1 SPLIT=dev CHECKPOINTS=$(seq 1600 1600 17600 | paste -sd ",") bash score_sdt.sh
EXPERIMENT_NAME=seed_420_replicate_sdt_v1_large PROMPT_ID=v1 SPLIT=test CHECKPOINTS=4800 bash score_sdt.sh

## Cleaup
EXPERIMENT_NAME=seed_420_replicate_sdt_v1_large KEEP_CHECKPOINTS=4800 bash cleanup_checkpoints_neuron.sh

#TODO: cleanup HPC

# v2 

python scripts/run_dialogue_state_tracking.py "configs/replicate_sdt_v2_neuron_large.json"

## Inference
PROMPT_ID=v2 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v2_large CHECKPOINTS=1600,3200 bash decode_sdt_intr.sh && exit
PROMPT_ID=v2 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v2_large CHECKPOINTS=4800,6400 bash decode_sdt_intr.sh && exit
PROMPT_ID=v2 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v2_large CHECKPOINTS=8000,9600 bash decode_sdt_intr.sh && exit
PROMPT_ID=v2 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v2_large CHECKPOINTS=11200,12800 bash decode_sdt_intr.sh && exit
PROMPT_ID=v2 SPLIT=test EXPERIMENT_NAME=seed_420_replicate_sdt_v2_large CHECKPOINTS=3200 bash decode_sdt_intr.sh && exit


## Scoring 
EXPERIMENT_NAME=seed_420_replicate_sdt_v2_large PROMPT_ID=v2 SPLIT=dev CHECKPOINTS=$(seq 1600 1600 16000 | paste -sd ",") bash score_sdt.sh
SAVE_FILES=true EXPERIMENT_NAME=seed_420_replicate_sdt_v2_large PROMPT_ID=v2 SPLIT=test CHECKPOINTS=3200 bash score_sdt.sh


## Cleanup

EXPERIMENT_NAME=seed_420_replicate_sdt_v2_large KEEP_CHECKPOINTS=3200 bash cleanup_checkpoints_neuron.sh


# v3

python scripts/run_dialogue_state_tracking.py "configs/replicate_sdt_v3_neuron_large.json"

## Inference
PROMPT_ID=v3 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large CHECKPOINTS=1600,3200 bash decode_sdt_intr.sh && exit
PROMPT_ID=v3 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large CHECKPOINTS=4800,6400 bash decode_sdt_intr.sh && exit
PROMPT_ID=v3 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large CHECKPOINTS=8000,9600 bash decode_sdt_intr.sh && exit
PROMPT_ID=v3 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large CHECKPOINTS=11200,12800 bash decode_sdt_intr.sh && exit
PROMPT_ID=v3 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large CHECKPOINTS=14400,16000 bash decode_sdt_intr.sh && exit
PROMPT_ID=v3 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large CHECKPOINTS=17600,19200 bash decode_sdt_intr.sh && exit
PROMPT_ID=v3 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large CHECKPOINTS=20800,22400 bash decode_sdt_intr.sh && exit
PROMPT_ID=v3 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large CHECKPOINTS=25600,27200 bash decode_sdt_intr.sh && exit
PROMPT_ID=v3 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large CHECKPOINTS=24000 bash decode_sdt_intr.sh && exit


- current interactive 
PROMPT_ID=v3 SPLIT=test EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large CHECKPOINTS=14400 bash decode_sdt_intr.sh && exit

- next interactive

PROMPT_ID=v3 SPLIT=test EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large CHECKPOINTS=6400 bash decode_sdt_intr.sh && exit

 
## Scoring 
EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large PROMPT_ID=v3 SPLIT=dev CHECKPOINTS=$(seq 1600 1600 16000 | paste -sd ",") bash score_sdt.sh
EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large PROMPT_ID=v3 SPLIT=dev CHECKPOINTS=$(seq 17600 1600 27200 | paste -sd ",") bash score_sdt.sh
EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large PROMPT_ID=v3 SPLIT=test CHECKPOINTS=14400 bash score_sdt.sh

## Cleanup

EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large KEEP_CHECKPOINTS=6400,14400 bash cleanup_checkpoints_neuron.sh
EXPERIMENT_NAME=seed_420_replicate_sdt_v3_large KEEP_CHECKPOINTS=6400,14400 bash cleanup_checkpoints.sh


# v4

python scripts/run_dialogue_state_tracking.py "configs/replicate_sdt_v4_neuron_large.json"

## Inference

PROMPT_ID=v4 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v4_large CHECKPOINTS=1600,3200 bash decode_sdt_intr.sh && exit

- current interactive
PROMPT_ID=v4 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v4_large CHECKPOINTS=4800,6400 bash decode_sdt_intr.sh && exit

- next_interactive
PROMPT_ID=v4 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v4_large CHECKPOINTS=8000,9600 bash decode_sdt_intr.sh && exit
PROMPT_ID=v4 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v4_large CHECKPOINTS=11200,12800 bash decode_sdt_intr.sh && exit
PROMPT_ID=v4 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v4_large CHECKPOINTS=14400,16000 bash decode_sdt_intr.sh && exit



# 1 epoch:  10976 (11200)
# 3 epochs: 32958 
# 2 epochs: 21952 (22400)