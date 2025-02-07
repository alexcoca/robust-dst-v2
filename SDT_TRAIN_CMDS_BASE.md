# v0

python scripts/run_dialogue_state_tracking.py "configs/replicate_sdt_v0_neuron.json"

## Inference

PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=3200 EXPERIMENT_NAME=seed_420_replicate_sdt_v0 sbatch -J decode_dev --array=0-1 decode_sdt.sh
Submitted batch job 4645516
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=1600,4800,6400,8000,9600,11200,12800,14400,16000,17600,19200,20800,22400,24000,25600,27200 EXPERIMENT_NAME=seed_420_replicate_sdt_v0 sbatch -J decode_dev --array=0-15 decode_sdt.sh
Submitted batch job 4650028
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=28800,30400,32000,33600,35200,36800,38400 EXPERIMENT_NAME=seed_420_replicate_sdt_v0 sbatch -J decode_dev --array=0-6 decode_sdt.sh
Submitted batch job 4665191
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=40000,41600,43200,44800,46400,48000,49600,51200,52800,54400 EXPERIMENT_NAME=seed_420_replicate_sdt_v0 sbatch -J decode_dev --array=0-9 decode_sdt.sh
Submitted batch job 4677359
PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=17600,19200 EXPERIMENT_NAME=seed_420_replicate_sdt_v0 sbatch -J decode_dev --array=0-1 decode_sdt.sh
Submitted batch job 4708378

PROMPT_ID=v0 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v0 CHECKPOINTS=17600,19200 bash decode_sdt_neuron.sh
PROMPT_ID=v0 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v0 CHECKPOINTS=16000 bash decode_sdt_neuron.sh


# scoring 
EXPERIMENT_NAME=seed_420_replicate_sdt_v0 PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=$(seq 4800 1600 16000 | paste -sd ",") bash score_sdt.sh
- nb: not sure why the array skipped 17600,19200
EXPERIMENT_NAME=seed_420_replicate_sdt_v0 PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=$(seq 22400 1600 54400 | paste -sd ",") bash score_sdt.sh

EXPERIMENT_NAME=seed_420_replicate_sdt_v0 PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=20800 bash score_sdt.sh
EXPERIMENT_NAME=seed_420_replicate_sdt_v0 PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=22400 bash score_sdt.sh
EXPERIMENT_NAME=seed_420_replicate_sdt_v0 PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=17600,19200 bash score_sdt_neuron.sh
EXPERIMENT_NAME=seed_420_replicate_sdt_v0 PROMPT_ID=v0 SPLIT=dev CHECKPOINTS=17600 bash score_sdt_neuron.sh

## cleanup 
EXPERIMENT_NAME=seed_420_replicate_sdt_v0 KEEP_CHECKPOINTS=16000  bash cleanup_checkpoints.sh


EXPERIMENT_NAME=seed_420_replicate_sdt_v0
ckpt=4800
SPLIT=dev
PROMPT_ID=v0
VERSION=1
PREDICTION_FILE_PATH=/home/ac2123/rds/rds-wjb31-nmt2020/ac2123/robust-dst-v2/models/${EXPERIMENT_NAME}/version_1/checkpoint-${ckpt}/generated_predictions.txt
CONFIG_FILE_PATH=/home/ac2123/rds/rds-wjb31-nmt2020/ac2123/robust-dst-v2/configs/replicate_sdt_${PROMPT_ID}.json
REFS_FILE_PATH=/home/ac2123/rds/rds-wjb31-nmt2020/ac2123/robust-dst-v2/data/processed/SGD_SDT/${PROMPT_ID}/"${SPLIT}"/version_"${VERSION}/data.json"
OUT_DIR=/home/ac2123/rds/rds-wjb31-nmt2020/ac2123/robust-dst-v2/metrics/${EXPERIMENT_NAME}/original/${SPLIT}/version_"${VERSION}"
HYP_DIR=/home/ac2123/rds/rds-wjb31-nmt2020/ac2123/robust-dst-v2/hyps/${EXPERIMENT_NAME}/original/${SPLIT}/version_"${VERSION}"
mkdir -p $OUT_DIR
mkdir -p $HYP_DIR
OUT_FILE=${OUT_DIR}/metrics_${ckpt}.json

python scripts/sdt_scoring.py --initialisation_file_path ${CONFIG_FILE_PATH} --predictions_file "${PREDICTION_FILE_PATH}" --refs_file ${REFS_FILE_PATH} --output_file ${OUT_FILE}

# v1 

python scripts/run_dialogue_state_tracking.py "configs/replicate_sdt_v1_neuron.json"

## Inference

PROMPT_ID=v1 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v1 bash decode_sdt_neuron.sh
CHECKPOINTS=14400 PROMPT_ID=v1 SPLIT=test EXPERIMENT_NAME=seed_420_replicate_sdt_v1 bash decode_sdt_neuron.sh

## Scoring
EXPERIMENT_NAME=seed_420_replicate_sdt_v1 PROMPT_ID=v1 SPLIT=dev CHECKPOINTS=$(seq 1600 1600 32000 | paste -sd ",") bash score_sdt_neuron.sh
EXPERIMENT_NAME=seed_420_replicate_sdt_v1 PROMPT_ID=v1 SPLIT=dev CHECKPOINTS=$(seq 33600 1600 54400 | paste -sd ",") bash score_sdt_neuron.sh
EXPERIMENT_NAME=seed_420_replicate_sdt_v1 PROMPT_ID=v1 SPLIT=test CHECKPOINTS=14400 bash score_sdt_neuron.sh

# v2

python scripts/run_dialogue_state_tracking.py "configs/replicate_sdt_v2_neuron.json"
PROMPT_ID=v2 SPLIT=dev EXPERIMENT_NAME=seed_420_replicate_sdt_v2 bash decode_sdt_neuron.sh


