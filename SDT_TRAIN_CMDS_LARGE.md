# v0

python scripts/run_dialogue_state_tracking.py "configs/replicate_sdt_v0_neuron_large.json"

## Inference

PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=3200 EXPERIMENT_NAME=seed_420_replicate_sdt_v0_neuron_large bash decode_sdt_neuron.sh

## Scoring 

EXPERIMENT_NAME=seed_420_replicate_sdt_v0_neuron_large PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=$(seq 1600 1600 54400 | paste -sd ",") bash score_sdt_neuron.sh

# v1 

python scripts/run_dialogue_state_tracking.py "configs/replicate_sdt_v1_neuron.json"

## Inference

PROMPT_ID=v1 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v1 bash decode_sdt_neuron.sh
CHECKPOINT=14400 PROMPT_ID=v1 SPLIT=test EXPERIMENT_NAME=seed_420_replicate_sdt_v1 bash decode_sdt_neuron.sh

## Scoring
EXPERIMENT_NAME=seed_420_replicate_sdt_v1 PROMPT_ID=v1 SPLIT=dev CHECKPOINTS=$(seq 1600 1600 32000 | paste -sd ",") bash score_sdt_neuron.sh

