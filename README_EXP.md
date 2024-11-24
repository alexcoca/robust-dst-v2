Need to set the following environment variables:
* `JOB_ID`
  For `torchrun`, this can be set by SLURM
* `HOST`
* `PORT`
  **Use a unique port for each job if it's possible that they are assigned to the same node.**
* `CHECKPOINT_DIR`
  Directory of the model to decode, use the best checkpoint from training
* `VERSION`
  Version of processed dataset, this is automatically incremented if datasets already exist in the specified output directory. Check the data output directory for the preprocessing step and use the latest version. Only inlucde the digits, e.g. `VERSION=2`.
* `PROC_NUM_WORK`
  Usually set this to 128
* `INFERENCE_BATCH_SIZE`
  This is used for inference only, set this to 200 unless specified otherwise
* `WANDB_ENTITY`
* `WANDB_PROJECT`

Also need to fill out the config files with the correct paths and wandb configs.

**Do not copy commands blindly, check paths to datasets. If a config file specified do not exist in repo, modify the baseline config following the stated changes. Inference commands within for loops are for illustration purposes only, need to use the desired checkpoint for each command call.**

> Note that in [zhangGroundingDescriptionDrivenDialogue2023], the randomness across the SGD-X variants is not controlled. However, randomness across the different models during Turn decoding experiments are controlled (as for TurnSlot decoding).


# SGD

## Baseline

Results in Table 1 [zhangGroundingDescriptionDrivenDialogue2023].

Seeds used by W Zhang:
1. 20230110
2. 202301102
3. 2023


```bash
# if you test on sgd-x
python -m scripts.preprocess_d3st_sgd \
  -d data/raw/original/ \
  -d data/raw/v1/ \
  -d data/raw/v2/ \
  -d data/raw/v3/ \
  -d data/raw/v4/ \
  -d data/raw/v5/ \
  -o data/processed/ \
  -c configs/data_processing_d3st_sgd.yaml \
  --all \
  -vv

# otherwise, for just training on SGD
python -m scripts.preprocess_d3st_sgd \
  -d data/raw/original/ \
  -o data/processed/ \
  -c configs/data_processing_d3st_sgd.yaml \
  --all \
  -vv

torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_id "$JOB_ID" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$HOST:$PORT" \
  -m scripts.run_dialogue_state_tracking "configs/train_d3st_sgd_baseline.json"

# SGD-X evaluation
declare -a SHARDS=("original" "v1" "v2" "v3" "v4" "v5")
# SGD evaluation
declare -a SHARDS=("original")
for SGD_SHARD in "${SHARDS[@]}"
do
  python -m scripts.run_dialogue_state_tracking \
    --model_name_or_path $CHECKPOINT_DIR \
    --output_dir $CHECKPOINT_DIR \
    --cache_dir cache \
    --test_file data/processed/$SGD_SHARD/test/version_$VERSION/data.json \
    --test_template_dir data/interim/blank_dialogue_templates/$SGD_SHARD/test \
    --test_ref_dir data/raw/$SGD_SHARD/test \
    --preprocessing_num_workers $PROC_NUM_WORK \
    --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
    --report_to wandb \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --do_predict \
    --predict_with_generate \
    --max_target_length 512 \
    --val_max_target_length 512

done
# NB: -mod can be specified multiple times together with the --average flag to average results across seeds
python -m scripts.compute_sgd_x_metrics \
  -ver $VERSION \
  -h hyps \
  -mod $CHECKPOINT_DIR
```

> From the models achive:
> 1. sgd_baseline/seed_20230110_d3st_baseline/version_5/checkpoint-20000
> 2. sgd_baseline/seed_202301102_d3st_baseline_2/version_5/checkpoint-15000
> 3. sgd_baseline/seed_2023_d3st_baseline_3/version_5/checkpoint-25000


## Data Augmentation

### `KSTRandom`

Results in Table 1, row `KSTRandom` [zhangGroundingDescriptionDrivenDialogue2023].

In the training config, `augment_style` should be `"REPLACE"`.

Seeds used by W Zhang:
1. 202302261
2. 202302262
3. 202302263


```bash
python -m scripts.preprocess_d3st_sgd \
  -d data/raw/original/ \
  -d data/raw/v1/ \
  -d data/raw/v2/ \
  -d data/raw/v3/ \
  -d data/raw/v4/ \
  -d data/raw/v5/ \
  -o data/processed/ \
  -c configs/data_processing_d3st_sgd.yaml \
  --all \
  -vv

torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_id "$JOB_ID" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$HOST:$PORT" \
  -m scripts.run_dialogue_state_tracking "configs/train_d3st_sgd_sample_schema.json"

declare -a SHARDS=("original" "v1" "v2" "v3" "v4" "v5")
for SGD_SHARD in "${SHARDS[@]}"
do
  python -m scripts.run_dialogue_state_tracking \
    --model_name_or_path $CHECKPOINT_DIR \
    --output_dir $CHECKPOINT_DIR \
    --cache_dir cache \
    --test_file data/processed/$SGD_SHARD/test/version_$VERSION/data.json \
    --test_template_dir data/interim/blank_dialogue_templates/$SGD_SHARD/test \
    --test_ref_dir data/raw/$SGD_SHARD/test \
    --preprocessing_num_workers $PROC_NUM_WORK \
    --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
    --report_to wandb \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --do_predict \
    --predict_with_generate \
    --max_target_length 512 \
    --val_max_target_length 512

done

python -m scripts.compute_sgd_x_metrics \
  -ver $VERSION \
  -h hyps \
  -mod $CHECKPOINT_DIR
```

> From the models achive:
> 1. sgd_da/seed_202302261_d3st_sgd_sample_dialogue/version_9/checkpoint-35000
> 2. sgd_da/seed_202302262_d3st_sgd_sample_dialogue/version_9/checkpoint-20000
> 3. sgd_da/seed_202302263_d3st_sgd_sample_dialogue/version_9/checkpoint-25000


### `KSTRandomConcat`

Results in Table 1, row `KSTRandomConcat` [zhangGroundingDescriptionDrivenDialogue2023].

In the training config, `augment_style` should be `"DA"`.

Seeds used by W Zhang:
1. 202302264
2. 202302265
3. 202302266


```bash
python -m scripts.preprocess_d3st_sgd \
  -d data/raw/original/ \
  -d data/raw/v1/ \
  -d data/raw/v2/ \
  -d data/raw/v3/ \
  -d data/raw/v4/ \
  -d data/raw/v5/ \
  -o data/processed/ \
  -c configs/data_processing_d3st_sgd.yaml \
  --all \
  -vv

torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_id "$JOB_ID" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$HOST:$PORT" \
  -m scripts.run_dialogue_state_tracking "configs/train_d3st_sgd_sample_schema_dialogue.json"

declare -a SHARDS=("original" "v1" "v2" "v3" "v4" "v5")
for SGD_SHARD in "${SHARDS[@]}"
do
  python -m scripts.run_dialogue_state_tracking \
    --model_name_or_path $CHECKPOINT_DIR \
    --output_dir $CHECKPOINT_DIR \
    --cache_dir cache \
    --test_file data/processed/$SGD_SHARD/test/version_$VERSION/data.json \
    --test_template_dir data/interim/blank_dialogue_templates/$SGD_SHARD/test \
    --test_ref_dir data/raw/$SGD_SHARD/test \
    --preprocessing_num_workers $PROC_NUM_WORK \
    --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
    --report_to wandb \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --do_predict \
    --predict_with_generate \
    --max_target_length 512 \
    --val_max_target_length 512

done

python -m scripts.compute_sgd_x_metrics \
  -ver $VERSION \
  -h hyps \
  -mod $CHECKPOINT_DIR
```

> From the models achive:
> 1. sgd_da/seed_202302264_d3st_sgd_sample_schema_dialogue/version_9/checkpoint-20000
> 2. sgd_da/seed_202302265_d3st_sgd_sample_schema_dialogue/version_9/checkpoint-30000
> 3. sgd_da/seed_202302266_d3st_sgd_sample_schema_dialogue/version_9/checkpoint-60000


## Training with Grounded Prompts

### Turn

Results in Table 9, row `RandomTurn` [zhangGroundingDescriptionDrivenDialogue2023].

In the training config, `augment_style` should be `"TURN"`.

Use 100 as decoding batch size.

Seeds used by W Zhang:
1. 2023060401
2. 2023060402
3. 2023060403

```bash
python -m scripts.preprocess_d3st_sgd \
  -d data/raw/original/ \
  -d data/raw/v1/ \
  -d data/raw/v2/ \
  -d data/raw/v3/ \
  -d data/raw/v4/ \
  -d data/raw/v5/ \
  -o data/processed/ \
  -c configs/data_processing_d3st_sgd.yaml \
  --all \
  -vv

# Collect the turns from the schemas variants used by Coca et al. to a table
# The table is used in the next step to ground the descriptions at inference time in
# the knowledge seeking turns used by Coca et al.
python -m scripts.build_kst_table_from_schema_variants \
  -s data/external/sampled_kst_schema_variants/v1/schema.json \
  -s data/external/sampled_kst_schema_variants/v2/schema.json \
  -s data/external/sampled_kst_schema_variants/v3/schema.json \
  -s data/external/sampled_kst_schema_variants/v4/schema.json \
  -s data/external/sampled_kst_schema_variants/v5/schema.json \
  -o data/external \
  -d data/processed/original/train/version_$VERSION/data.json \
  -vv

# Processing test data for all SGD-X variants to use the same turns as in Coca et al.
# Seed is fixed to ensure the examples across the SGD-X variants are grounded with the
# same turn and concatenation order is consistent across schema variants
# (this is not the case in [zhangGroundingDescriptionDrivenDialogue2023])

# Note that for a given example, the turn choice and concatenation order are not the
# same as in Coca et al.
declare -a SHARDS=("original" "v1" "v2" "v3" "v4" "v5")
for SGD_SHARD in "${SHARDS[@]}"
do
  python -m scripts.ground_with_sampled_ksts \
    --seed 100 \
    --kst_table data/external/kst_table_$SGD_SHARD.json \
    --data data/processed/original/test/version_$VERSION/data.json \
    --out data/processed/turn/$SGD_SHARD \
    --augment_style "TURN" \
    -v
done

torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_id "$JOB_ID" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$HOST:$PORT" \
  -m scripts.run_dialogue_state_tracking "configs/train_d3st_sgd_turn.json"
# SGD-X evaluation
declare -a SHARDS=("original" "v1" "v2" "v3" "v4" "v5")
for SGD_SHARD in "${SHARDS[@]}"
do
  python -m scripts.run_dialogue_state_tracking \
    --model_name_or_path $CHECKPOINT_DIR \
    --output_dir $CHECKPOINT_DIR \
    --cache_dir cache \
    --test_file data/processed/turn/$SGD_SHARD/test/version_$VERSION/data.json \
    --test_template_dir data/interim/blank_dialogue_templates/$SGD_SHARD/test \
    --test_ref_dir data/raw/$SGD_SHARD/test \
    --preprocessing_num_workers $PROC_NUM_WORK \
    --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
    --report_to wandb \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --do_predict \
    --predict_with_generate \
    --max_target_length 512 \
    --val_max_target_length 512

done

python -m scripts.compute_sgd_x_metrics \
  -ver $VERSION \
  -h hyps \
  -mod $CHECKPOINT_DIR
```


> From the models achive:
> 1. sgd_turn_turnslot/seed_2023060401_d3st_sgd_turn/version_9/checkpoint-30000
> 2. sgd_turn_turnslot/seed_2023051402_d3st_sgd_turn/version_9/checkpoint-20000
> 3. sgd_turn_turnslot/seed_2023051403_d3st_sgd_turn/version_9/checkpoint-45000


### TurnSlot

Results in Table 9, row `RandomTurnSlot` [zhangGroundingDescriptionDrivenDialogue2023].

In the training config, `augment_style` should be `"TURNSLOT"`.

Use 100 as decoding batch size.

Seeds used by W Zhang:
1. 2023051404
2. 2023051405
3. 2023051407


```bash
python -m scripts.preprocess_d3st_sgd \
  -d data/raw/original/ \
  -d data/raw/v1/ \
  -d data/raw/v2/ \
  -d data/raw/v3/ \
  -d data/raw/v4/ \
  -d data/raw/v5/ \
  -o data/processed/ \
  -c configs/data_processing_d3st_sgd.yaml \
  --all \
  -vv

# Collect the turns from the schemas variants used by Coca et al. to a table
# The table is used in the next step to ground the descriptions at inference time in
# the knowledge seeking turns used by Coca et al.
python -m scripts.build_kst_table_from_schema_variants \
  -s data/external/sampled_kst_schema_variants/v1/schema.json \
  -s data/external/sampled_kst_schema_variants/v2/schema.json \
  -s data/external/sampled_kst_schema_variants/v3/schema.json \
  -s data/external/sampled_kst_schema_variants/v4/schema.json \
  -s data/external/sampled_kst_schema_variants/v5/schema.json \
  -o data/external \
  -d data/processed/original/train/version_$VERSION/data.json \
  -vv

# Processing test data for all SGD-X variants to use the same turns as in Coca et al.
# Seed is fixed to ensure the examples across the SGD-X variants are grounded with the
# same turn and concatenation order is consistent across schema variants
# (this is not the case in [zhangGroundingDescriptionDrivenDialogue2023])

# Note that for a given example, the turn choice and concatenation order are not the
# same as in Coca et al.
declare -a SHARDS=("original" "v1" "v2" "v3" "v4" "v5")
for SGD_SHARD in "${SHARDS[@]}"
do
  python -m scripts.ground_with_sampled_ksts \
    --seed 100 \
    --kst_table data/external/kst_table_$SGD_SHARD.json \
    --data data/processed/original/test/version_$VERSION/data.json \
    --out data/processed/turnslot/$SGD_SHARD \
    --augment_style "TURNSLOT" \
    -v
done

torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_id "$JOB_ID" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$HOST:$PORT" \
  -m scripts.run_dialogue_state_tracking "configs/train_d3st_sgd_turnslot.json"

declare -a SHARDS=("original" "v1" "v2" "v3" "v4" "v5")
for SGD_SHARD in "${SHARDS[@]}"
do
  python -m scripts.run_dialogue_state_tracking \
    --model_name_or_path $CHECKPOINT_DIR \
    --output_dir $CHECKPOINT_DIR \
    --cache_dir cache \
    --test_file data/processed/turnslot/$SGD_SHARD/test/version_$VERSION/data.json \
    --test_template_dir data/interim/blank_dialogue_templates/$SGD_SHARD/test \
    --test_ref_dir data/raw/$SGD_SHARD/test \
    --preprocessing_num_workers $PROC_NUM_WORK \
    --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
    --report_to wandb \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --do_predict \
    --predict_with_generate \
    --max_target_length 512 \
    --val_max_target_length 512

done

python -m scripts.compute_sgd_x_metrics \
  -ver $VERSION \
  -h hyps \
  -mod $CHECKPOINT_DIR
```

> From the models achive:
> 1. sgd_turn_turnslot/seed_2023051404_d3st_sgd_turnslot/version_9/checkpoint-45000
> 2. sgd_turn_turnslot/seed_2023051405_d3st_sgd_turnslot/version_9/checkpoint-35000
> 3. sgd_turn_turnslot/seed_2023051407_d3st_sgd_turnslot/version_9/checkpoint-35000



## Grounded Decoding

### Turn

```bash
python -m scripts.preprocess_d3st_sgd \
  -d data/raw/original/ \
  -d data/raw/v1/ \
  -d data/raw/v2/ \
  -d data/raw/v3/ \
  -d data/raw/v4/ \
  -d data/raw/v5/ \
  -o data/processed/ \
  -c configs/data_processing_d3st_sgd.yaml \
  --all \
  -vv

# Collect the turns from the schemas variants used by Coca et al. to a table
# The table is used in the next step to ground the descriptions at inference time in
# the knowledge seeking turns used by Coca et al.
python -m scripts.build_kst_table_from_schema_variants \
  -s data/external/sampled_kst_schema_variants/v1/schema.json \
  -s data/external/sampled_kst_schema_variants/v2/schema.json \
  -s data/external/sampled_kst_schema_variants/v3/schema.json \
  -s data/external/sampled_kst_schema_variants/v4/schema.json \
  -s data/external/sampled_kst_schema_variants/v5/schema.json \
  -o data/external \
  -d data/processed/original/train/version_$VERSION/data.json \
  -vv

# Processing test data for all SGD-X variants to use the same turns as in Coca et al.
# Seed is fixed to ensure the examples across the SGD-X variants are grounded with the
# same turn and concatenation order is consistent across schema variants
# (this is not the case in [zhangGroundingDescriptionDrivenDialogue2023])

# Note that for a given example, the turn choice and concatenation order are not the
# same as in Coca et al.
declare -a SHARDS=("original" "v1" "v2" "v3" "v4" "v5")
for SGD_SHARD in "${SHARDS[@]}"
do
  python -m scripts.ground_with_sampled_ksts \
    --seed 100 \
    --kst_table data/external/kst_table_$SGD_SHARD.json \
    --data data/processed/original/test/version_$VERSION/data.json \
    --out data/processed/turn/$SGD_SHARD \
    --augment_style "TURN" \
    -v
done


declare -a SHARDS=("original" "v1" "v2" "v3" "v4" "v5")
for SGD_SHARD in "${SHARDS[@]}"
do
  python -m scripts.run_dialogue_state_tracking \
    --model_name_or_path $CHECKPOINT_DIR \
    --output_dir $CHECKPOINT_DIR \
    --cache_dir cache \
    --test_file data/processed/turn/$SGD_SHARD/test/version_$VERSION/data.json \
    --test_template_dir data/interim/blank_dialogue_templates/$SGD_SHARD/test \
    --test_ref_dir data/raw/$SGD_SHARD/test \
    --preprocessing_num_workers $PROC_NUM_WORK \
    --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
    --report_to wandb \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --do_predict \
    --predict_with_generate \
    --max_target_length 512 \
    --val_max_target_length 512

done

python -m scripts.compute_sgd_x_metrics \
  -ver $VERSION \
  -h hyps \
  -mod $CHECKPOINT_DIR
```


> From the models achive:
> 1. sgd_da/seed_202302261_d3st_sgd_sample_dialogue/version_9/checkpoint-35000
> 2. sgd_da/seed_202302262_d3st_sgd_sample_dialogue/version_9/checkpoint-20000
> 3. sgd_da/seed_202302263_d3st_sgd_sample_dialogue/version_9/checkpoint-25000
> 4. sgd_da/seed_202302264_d3st_sgd_sample_schema_dialogue/version_9/checkpoint-20000
> 5. sgd_da/seed_202302265_d3st_sgd_sample_schema_dialogue/version_9/checkpoint-30000
> 6. sgd_da/seed_202302266_d3st_sgd_sample_schema_dialogue/version_9/checkpoint-60000


### TurnSlot


```bash
python -m scripts.preprocess_d3st_sgd \
  -d data/raw/original/ \
  -d data/raw/v1/ \
  -d data/raw/v2/ \
  -d data/raw/v3/ \
  -d data/raw/v4/ \
  -d data/raw/v5/ \
  -o data/processed/ \
  -c configs/data_processing_d3st_sgd.yaml \
  --all \
  -vv

# Collect the turns from the schemas variants used by Coca et al. to a table
# The table is used in the next step to ground the descriptions at inference time in
# the knowledge seeking turns used by Coca et al.
python -m scripts.build_kst_table_from_schema_variants \
  -s data/external/sampled_kst_schema_variants/v1/schema.json \
  -s data/external/sampled_kst_schema_variants/v2/schema.json \
  -s data/external/sampled_kst_schema_variants/v3/schema.json \
  -s data/external/sampled_kst_schema_variants/v4/schema.json \
  -s data/external/sampled_kst_schema_variants/v5/schema.json \
  -o data/external \
  -d data/processed/original/train/version_$VERSION/data.json \
  -vv

# Processing test data for all SGD-X variants to use the same turns as in Coca et al.
# Seed is fixed to ensure the examples across the SGD-X variants are grounded with the
# same turn and concatenation order is consistent across schema variants
# (this is not the case in [zhangGroundingDescriptionDrivenDialogue2023])

# Note that for a given example, the turn choice and concatenation order are not the
# same as in Coca et al.
declare -a SHARDS=("original" "v1" "v2" "v3" "v4" "v5")
for SGD_SHARD in "${SHARDS[@]}"
do
  python -m scripts.ground_with_sampled_ksts \
    --seed 100 \
    --kst_table data/external/kst_table_$SGD_SHARD.json \
    --data data/processed/original/test/version_$VERSION/data.json \
    --out data/processed/turnslot/$SGD_SHARD \
    --augment_style "TURN" \
    -v
done


declare -a SHARDS=("original" "v1" "v2" "v3" "v4" "v5")
for SGD_SHARD in "${SHARDS[@]}"
do
  python -m scripts.run_dialogue_state_tracking \
    --model_name_or_path $CHECKPOINT_DIR \
    --output_dir $CHECKPOINT_DIR \
    --cache_dir cache \
    --test_file data/processed/turnslot/$SGD_SHARD/test/version_$VERSION/data.json \
    --test_template_dir data/interim/blank_dialogue_templates/$SGD_SHARD/test \
    --test_ref_dir data/raw/$SGD_SHARD/test \
    --preprocessing_num_workers $PROC_NUM_WORK \
    --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
    --report_to wandb \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --do_predict \
    --predict_with_generate \
    --max_target_length 512 \
    --val_max_target_length 512

done

python -m scripts.compute_sgd_x_metrics \
  -ver $VERSION \
  -h hyps \
  -mod $CHECKPOINT_DIR
```


> From the models achive:
> 1. sgd_da/seed_202302261_d3st_sgd_sample_dialogue/version_9/checkpoint-35000
> 2. sgd_da/seed_202302262_d3st_sgd_sample_dialogue/version_9/checkpoint-20000
> 3. sgd_da/seed_202302263_d3st_sgd_sample_dialogue/version_9/checkpoint-25000
> 4. sgd_da/seed_202302264_d3st_sgd_sample_schema_dialogue/version_9/checkpoint-20000
> 5. sgd_da/seed_202302265_d3st_sgd_sample_schema_dialogue/version_9/checkpoint-30000
> 6. sgd_da/seed_202302266_d3st_sgd_sample_schema_dialogue/version_9/checkpoint-60000



# MultiWOZ

## Bseline

Results in Table 5 [zhangGroundingDescriptionDrivenDialogue2023].

Use 100 as decoding batch size.

Seeds used by W Zhang:
1. 20230224
2. 202302242
3. 202302243

```bash
python -m scripts.preprocess_d3st_multiwoz \
  -d data/raw/multiwoz/ \
  --schema_file data/raw/multiwoz/schema.json \
  --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
  -o data/processed/ \
  -c configs/data_processing_d3st_multiwoz.yaml \
  --all \
  -vv

torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_id "$JOB_ID" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$HOST:$PORT" \
  -m scripts.run_dialogue_state_tracking "configs/train_d3st_multiwoz_baseline.json"

python -m scripts.run_dialogue_state_tracking \
  --model_name_or_path $CHECKPOINT_DIR \
  --output_dir $CHECKPOINT_DIR \
  --cache_dir cache \
  --test_file data/processed/multiwoz/test/version_$VERSION/data.json \
  --preprocessing_num_workers $PROC_NUM_WORK \
  --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
  --report_to wandb \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --do_predict \
  --predict_with_generate \
  --max_target_length 512 \
  --val_max_target_length 512"
```

> From the models achive:
> 1. multiwoz_baseline/seed_20230224_multiwoz_baseline/version_9/checkpoint-5100
> 2. multiwoz_baseline/seed_202302242_multiwoz_baseline/version_9/checkpoint-10200
> 3. multiwoz_baseline/seed_202302243_multiwoz_baseline/version_9/checkpoint-6800


## Data Augmentation

### `KSTRandom`

Results in Table 5, row `KSTRandom` [zhangGroundingDescriptionDrivenDialogue2023].

In the training config, `augment_style` should be `"REPLACE"`.

Seeds used by W Zhang:
1. 202302281
2. 202302282
3. 202302283


```bash
python -m scripts.preprocess_d3st_multiwoz \
  -d data/raw/multiwoz/ \
  --schema_file data/raw/multiwoz/schema.json \
  --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
  -o data/processed/ \
  -c configs/data_processing_d3st_multiwoz.yaml \
  --all \
  -vv

torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_id "$JOB_ID" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$HOST:$PORT" \
  -m scripts.run_dialogue_state_tracking "configs/train_d3st_multiwoz_sample_dialogue.json"

python -m scripts.run_dialogue_state_tracking \
  --model_name_or_path $CHECKPOINT_DIR \
  --output_dir $CHECKPOINT_DIR \
  --cache_dir cache \
  --test_file data/processed/multiwoz/test/version_$VERSION/data.json \
  --preprocessing_num_workers $PROC_NUM_WORK \
  --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
  --report_to wandb \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --do_predict \
  --predict_with_generate \
  --max_target_length 512 \
  --val_max_target_length 512"
```

> From the models achive:
> 1. multiwoz_da/seed_202302281_multiwoz_sample_dialogue/version_9/checkpoint-8500
> 2. multiwoz_da/seed_202302282_multiwoz_sample_dialogue/version_9/checkpoint-13600
> 3. multiwoz_da/seed_202302283_multiwoz_sample_dialogue/version_9/checkpoint-10200


### `KSTRandomConcat`

Results in Table 5, row `KSTRandomConcat` [zhangGroundingDescriptionDrivenDialogue2023].

In the training config, `augment_style` should be `"REPLACE"`.

Seeds used by W Zhang:
1. 202302284
2. 202302285
3. 202302286


```bash
python -m scripts.preprocess_d3st_multiwoz \
  -d data/raw/multiwoz/ \
  --schema_file data/raw/multiwoz/schema.json \
  --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
  -o data/processed/ \
  -c configs/data_processing_d3st_multiwoz.yaml \
  --all \
  -vv

torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_id "$JOB_ID" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$HOST:$PORT" \
  -m scripts.run_dialogue_state_tracking "configs/train_d3st_multiwoz_sample_schema_dialogue.json"

python -m scripts.run_dialogue_state_tracking \
  --model_name_or_path $CHECKPOINT_DIR \
  --output_dir $CHECKPOINT_DIR \
  --cache_dir cache \
  --test_file data/processed/multiwoz/test/version_$VERSION/data.json \
  --preprocessing_num_workers $PROC_NUM_WORK \
  --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
  --report_to wandb \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --do_predict \
  --predict_with_generate \
  --max_target_length 512 \
  --val_max_target_length 512"
```

> From the models achive:
> 1. multiwoz_da/seed_202302284_multiwoz_sample_schema_dialogue/version_9/checkpoint-6800
> 2. multiwoz_da/seed_202302285_multiwoz_sample_schema_dialogue/version_9/checkpoint-6800
> 3. multiwoz_da/seed_202302286_multiwoz_sample_schema_dialogue/version_9/checkpoint-6800


## Leave-One-Domain-Out

### Baselines

Results in Table 6 [zhangGroundingDescriptionDrivenDialogue2023].

Seeds used by W Zhang:
```json
"attraction": [
    "2023042801",
    "2023042802",
    "2023042803",
],
"hotel": [
    "2023042804",
    "2023042805",
    "2023042806",
],
"restaurant": [
    "2023042807",
    "2023042808",
    "2023042809",
],
"taxi": [
    "2023042810",
    "2023042811",
    "2023042812",
],
"train": [
    "2023042813",
    "2023042814",
    "2023042815",
],
```

```bash
declare -a DOMAINS=("train" "taxi" "restaurant" "hotel" "attraction")
for DOMAIN in "${DOMAINS[@]}"
do
  python -m scripts.preprocess_d3st_multiwoz \
    -d data/raw/multiwoz/ \
    --schema_file data/raw/multiwoz/schema.json \
    --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
    -o data/processed/multiwoz_loo/$DOMAIN/ \
    -c configs/data_processing_d3st_multiwoz_loo__$DOMAIN.yaml \
    --train \
    -vv

  python -m scripts.preprocess_d3st_multiwoz \
    -d data/raw/multiwoz/ \
    --schema_file data/raw/multiwoz/schema.json \
    --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
    -o data/processed/multiwoz_loo/$DOMAIN/ \
    -c configs/data_processing_d3st_multiwoz_loo__$DOMAIN.yaml \
    --dev \
    -vv

  python -m scripts.preprocess_d3st_multiwoz \
    -d data/raw/multiwoz/ \
    --schema_file data/raw/multiwoz/schema.json \
    --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
    -o data/processed/multiwoz_loo/$DOMAIN/ \
    -c configs/data_processing_d3st_multiwoz_loo__${DOMAIN}_test.yaml \
    --test \
    -vv

  torchrun \
    --nproc_per_node 2 \
    --nnodes 1 \
    --max_restarts 0 \
    --rdzv_id "$JOB_ID" \
    --rdzv_backend c10d \
    --rdzv_endpoint "$HOST:$PORT" \
    -m scripts.run_dialogue_state_tracking "configs/train_d3st_multiwoz_loo_baseline__$DOMAIN.json"

  python -m scripts.run_dialogue_state_tracking \
    --model_name_or_path $CHECKPOINT_DIR \
    --output_dir $CHECKPOINT_DIR \
    --cache_dir cache \
    --test_file data/processed/multiwoz_loo/$DOMAIN/multiwoz/test/version_$VERSION/data.json \
    --preprocessing_num_workers $PROC_NUM_WORK \
    --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
    --report_to wandb \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --do_predict \
    --predict_with_generate \
    --max_target_length 512 \
    --val_max_target_length 512"

done
```

> In the models achive under `multiwoz_cross_domain/active/`
```json
"attraction": [
    "seed_2023042801_d3st_multiwoz_loo_baseline_attraction/version_1/checkpoint-6800",
    "seed_2023042802_d3st_multiwoz_loo_baseline_attraction/version_1/checkpoint-5100",
    "seed_2023042803_d3st_multiwoz_loo_baseline_attraction/version_1/checkpoint-8500",
],
"hotel": [
    "seed_2023042804_d3st_multiwoz_loo_baseline_hotel/version_1/checkpoint-8500",
    "seed_2023042805_d3st_multiwoz_loo_baseline_hotel/version_1/checkpoint-5100",
    "seed_2023042806_d3st_multiwoz_loo_baseline_hotel/version_1/checkpoint-6800",
],
"restaurant": [
    "seed_2023042807_d3st_multiwoz_loo_baseline_restaurant/version_1/checkpoint-6800",
    "seed_2023042808_d3st_multiwoz_loo_baseline_restaurant/version_1/checkpoint-6800",
    "seed_2023042809_d3st_multiwoz_loo_baseline_restaurant/version_1/checkpoint-6800",
],
"taxi": [
    "seed_2023042810_d3st_multiwoz_loo_baseline_taxi/version_1/checkpoint-5100",
    "seed_2023042811_d3st_multiwoz_loo_baseline_taxi/version_1/checkpoint-11900",
    "seed_2023042812_d3st_multiwoz_loo_baseline_taxi/version_1/checkpoint-6800",
],
"train": [
    "seed_2023042813_d3st_multiwoz_loo_baseline_train/version_1/checkpoint-5100",
    "seed_2023042814_d3st_multiwoz_loo_baseline_train/version_1/checkpoint-3400",
    "seed_2023042815_d3st_multiwoz_loo_baseline_train/version_1/checkpoint-6800",
],
```


### Data Augmentation

In the training config, `augment_style` should be `"DA"`.

Results in Table 6, row `KSTRandomConcat` [zhangGroundingDescriptionDrivenDialogue2023].

Seeds used by W Zhang:
```json
"attraction": [
    "2023042901",
    "2023042902",
    "2023042903",
],
"hotel": [
    "2023042904",
    "2023042905",
    "2023042906",
],
"restaurant": [
    "2023042907",
    "2023042908",
    "2023042909",
],
"taxi": [
    "2023042910",
    "2023042911",
    "2023042912",
],
"train": [
    "2023042913",
    "2023042914",
    "2023042915",
],
```

```bash
declare -a DOMAINS=("train" "taxi" "restaurant" "hotel" "attraction")
for DOMAIN in "${DOMAINS[@]}"
do
  python -m scripts.preprocess_d3st_multiwoz \
    -d data/raw/multiwoz/ \
    --schema_file data/raw/multiwoz/schema.json \
    --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
    -o data/processed/multiwoz_loo/$DOMAIN/ \
    -c configs/data_processing_d3st_multiwoz_loo__$DOMAIN.yaml \
    --train \
    -vv

  python -m scripts.preprocess_d3st_multiwoz \
    -d data/raw/multiwoz/ \
    --schema_file data/raw/multiwoz/schema.json \
    --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
    -o data/processed/multiwoz_loo/$DOMAIN/ \
    -c configs/data_processing_d3st_multiwoz_loo__$DOMAIN.yaml \
    --dev \
    -vv

  python -m scripts.preprocess_d3st_multiwoz \
    -d data/raw/multiwoz/ \
    --schema_file data/raw/multiwoz/schema.json \
    --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
    -o data/processed/multiwoz_loo/$DOMAIN/ \
    -c configs/data_processing_d3st_multiwoz_loo__${DOMAIN}_test.yaml \
    --test \
    -vv

  torchrun \
    --nproc_per_node 2 \
    --nnodes 1 \
    --max_restarts 0 \
    --rdzv_id "$JOB_ID" \
    --rdzv_backend c10d \
    --rdzv_endpoint "$HOST:$PORT" \
    -m scripts.run_dialogue_state_tracking "configs/train_d3st_multiwoz_loo_sample_schema_dialogue__$DOMAIN.json"

  python -m scripts.run_dialogue_state_tracking \
    --model_name_or_path $CHECKPOINT_DIR \
    --output_dir $CHECKPOINT_DIR \
    --cache_dir cache \
    --test_file data/processed/multiwoz_loo/$DOMAIN/multiwoz/test/version_$VERSION/data.json \
    --preprocessing_num_workers $PROC_NUM_WORK \
    --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
    --report_to wandb \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --do_predict \
    --predict_with_generate \
    --max_target_length 512 \
    --val_max_target_length 512"

done
```

> In the models achive under `multiwoz_cross_domain/active/`
```json
"attraction": [
    "seed_2023042901_d3st_multiwoz_loo_sample_schema_dialogue_attraction/version_1/checkpoint-8500",
    "seed_2023042902_d3st_multiwoz_loo_sample_schema_dialogue_attraction/version_1/checkpoint-8500",
    "seed_2023042903_d3st_multiwoz_loo_sample_schema_dialogue_attraction/version_1/checkpoint-8500",
],
"hotel": [
    "seed_2023042904_d3st_multiwoz_loo_sample_schema_dialogue_hotel/version_1/checkpoint-8500",
    "seed_2023042905_d3st_multiwoz_loo_sample_schema_dialogue_hotel/version_1/checkpoint-6800",
    "seed_2023042906_d3st_multiwoz_loo_sample_schema_dialogue_hotel/version_1/checkpoint-6800",
],
"restaurant": [
    "seed_2023042907_d3st_multiwoz_loo_sample_schema_dialogue_restaurant/version_1/checkpoint-6800",
    "seed_2023042908_d3st_multiwoz_loo_sample_schema_dialogue_restaurant/version_1/checkpoint-6800",
    "seed_2023042909_d3st_multiwoz_loo_sample_schema_dialogue_restaurant/version_1/checkpoint-18700",
],
"taxi": [
    "seed_2023042910_d3st_multiwoz_loo_sample_schema_dialogue_taxi/version_1/checkpoint-5100",
    "seed_2023042911_d3st_multiwoz_loo_sample_schema_dialogue_taxi/version_1/checkpoint-5100",
    "seed_2023042912_d3st_multiwoz_loo_sample_schema_dialogue_taxi/version_1/checkpoint-8500",
],
"train": [
    "seed_2023042913_d3st_multiwoz_loo_sample_schema_dialogue_train/version_1/checkpoint-5100",
    "seed_2023042914_d3st_multiwoz_loo_sample_schema_dialogue_train/version_1/checkpoint-5100",
    "seed_2023042915_d3st_multiwoz_loo_sample_schema_dialogue_train/version_1/checkpoint-6800",
],
```


## Unseen Description Transfer

Modify `scripts/preprocess_d3st_multiwoz.py`

```python
            # Decide description for this slot.
            # slot_descriptions.json has multiple descriptions for each slot,
            # for now only use the first one.
            full_desc = slot_descriptions[slot_name][0]
```

into

```python
            # Decide description for this slot.
            # slot_descriptions.json has multiple descriptions for each slot,
            # for now only use the first one.
            full_desc = slot_descriptions[slot_name][1]
```

Then follow the commands in MultiWOZ Baseline and/or Data Augmentation.

### Baseline

```bash
python -m scripts.preprocess_d3st_multiwoz \
  -d data/raw/multiwoz/ \
  --schema_file data/raw/multiwoz/schema.json \
  --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
  -o data/processed/multiwoz_x/ \
  -c configs/data_processing_d3st_multiwoz.yaml \
  --test \
  -vv

# Decode the models from MultiWOZ Baseline section on this new dataset
python -m scripts.run_dialogue_state_tracking \
  --model_name_or_path $CHECKPOINT_DIR \
  --output_dir $CHECKPOINT_DIR \
  --cache_dir cache \
  --test_file data/processed/multiwoz_x/multiwoz/test/version_$VERSION/data.json \
  --preprocessing_num_workers $PROC_NUM_WORK \
  --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
  --report_to wandb \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --do_predict \
  --predict_with_generate \
  --max_target_length 512 \
  --val_max_target_length 512"
```

> From the models achive:
> 1. multiwoz_baseline/seed_20230224_multiwoz_baseline/version_9/checkpoint-5100
> 2. multiwoz_baseline/seed_202302242_multiwoz_baseline/version_9/checkpoint-10200
> 3. multiwoz_baseline/seed_202302243_multiwoz_baseline/version_9/checkpoint-6800
> 4. multiwoz_da/seed_202302281_multiwoz_sample_dialogue/version_9/checkpoint-8500
> 5. multiwoz_da/seed_202302282_multiwoz_sample_dialogue/version_9/checkpoint-13600
> 6. multiwoz_da/seed_202302283_multiwoz_sample_dialogue/version_9/checkpoint-10200
> 7. multiwoz_da/seed_202302284_multiwoz_sample_schema_dialogue/version_9/checkpoint-6800
> 8. multiwoz_da/seed_202302285_multiwoz_sample_schema_dialogue/version_9/checkpoint-6800
> 9. multiwoz_da/seed_202302286_multiwoz_sample_schema_dialogue/version_9/checkpoint-6800



### Leave-One-Domain-Out

```bash
declare -a DOMAINS=("train" "taxi" "restaurant" "hotel" "attraction")
for DOMAIN in "${DOMAINS[@]}"
do
  python -m scripts.preprocess_d3st_multiwoz \
    -d data/raw/multiwoz/ \
    --schema_file data/raw/multiwoz/schema.json \
    --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
    -o data/processed/multiwoz_x/$DOMAIN/ \
    -c configs/data_processing_d3st_multiwoz_loo__${DOMAIN}_test.yaml \
    --test \
    -vv

done

# Decode the models from MultiWOZ Leave-One-Out section on this new dataset
declare -a DOMAINS=("train" "taxi" "restaurant" "hotel" "attraction")
for DOMAIN in "${DOMAINS[@]}"
do
  python -m scripts.run_dialogue_state_tracking \
    --model_name_or_path $CHECKPOINT_DIR \
    --output_dir $CHECKPOINT_DIR \
    --cache_dir cache \
    --test_file data/processed/multiwoz_x/$DOMAIN/multiwoz/test/version_$VERSION/data.json \
    --preprocessing_num_workers $PROC_NUM_WORK \
    --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
    --report_to wandb \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --do_predict \
    --predict_with_generate \
    --max_target_length 512 \
    --val_max_target_length 512"

done
```


# Cross Dataset Transfer

## SGD -> MultiWOZ

Use the models from SGD Baseline and Data Augmentation sections.

Decode the models on MultiWOZ produced by

```bash
python -m scripts.preprocess_d3st_multiwoz \
  -d data/raw/multiwoz/ \
  --schema_file data/raw/multiwoz/schema.json \
  --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
  -o data/processed/multiwoz_active \
  -c configs/data_processing_d3st_multiwoz__active.yaml \
  --all \
  -vv
```

using

```bash
python -m scripts.run_dialogue_state_tracking \
  --model_name_or_path $CHECKPOINT_DIR \
  --output_dir $CHECKPOINT_DIR \
  --cache_dir cache \
  --test_file data/processed/multiwoz_active/multiwoz/test/version_$VERSION/data.json \
  --preprocessing_num_workers $PROC_NUM_WORK \
  --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
  --report_to wandb \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --do_predict \
  --predict_with_generate \
  --max_target_length 512 \
  --val_max_target_length 512"
```

> From the models achive:
> 1. sgd_baseline/seed_20230110_d3st_baseline/version_5/checkpoint-20000
> 2. sgd_baseline/seed_202301102_d3st_baseline_2/version_5/checkpoint-15000
> 3. sgd_baseline/seed_2023_d3st_baseline_3/version_5/checkpoint-25000
> 4. sgd_da/seed_202302261_d3st_sgd_sample_dialogue/version_9/checkpoint-35000
> 5. sgd_da/seed_202302262_d3st_sgd_sample_dialogue/version_9/checkpoint-20000
> 6. sgd_da/seed_202302263_d3st_sgd_sample_dialogue/version_9/checkpoint-25000
> 7. sgd_da/seed_202302264_d3st_sgd_sample_schema_dialogue/version_9/checkpoint-20000
> 8. sgd_da/seed_202302265_d3st_sgd_sample_schema_dialogue/version_9/checkpoint-30000
> 9. sgd_da/seed_202302266_d3st_sgd_sample_schema_dialogue/version_9/checkpoint-60000k


## MultiWOZ -> SGD

### Baseline

Results in Table 12 [zhangGroundingDescriptionDrivenDialogue2023].

Seeds used by W Zhang:
1. 2023043001
2. 2023043002
3. 2023043003

```bash
python -m scripts.preprocess_d3st_multiwoz \
  -d data/raw/multiwoz/ \
  --schema_file data/raw/multiwoz/schema.json \
  --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
  -o data/processed/multiwoz_active \
  -c configs/data_processing_d3st_multiwoz__active.yaml \
  --all \
  -vv

# change train dataset path
torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_id "$JOB_ID" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$HOST:$PORT" \
  -m scripts.run_dialogue_state_tracking "configs/train_d3st_multiwoz.json"

python -m scripts.run_dialogue_state_tracking \
  --model_name_or_path $CHECKPOINT_DIR \
  --output_dir $CHECKPOINT_DIR \
  --cache_dir cache \
  --test_file data/processed/original/test/version_$VERSION/data.json \
  --test_template_dir data/interim/blank_dialogue_templates/$SGD_SHARD/test \
  --test_ref_dir data/raw/original/test \
  --preprocessing_num_workers $PROC_NUM_WORK \
  --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
  --report_to wandb \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --do_predict \
  --predict_with_generate \
  --max_target_length 512 \
  --val_max_target_length 512
```

> From the models achive:
> 1. multiwoz_cross_dataset/seed_2023043001_d3st_multiwoz_baseline_active/version_1/checkpoint-5100
> 2. multiwoz_cross_dataset/seed_2023043002_d3st_multiwoz_baseline_active/version_1/checkpoint-5100
> 3. multiwoz_cross_dataset/seed_2023043003_d3st_multiwoz_baseline_active/version_1/checkpoint-10200


### `KSTRandomConcat`

In the training config, `augment_style` should be `"DA"`.

Results in Table 12 [zhangGroundingDescriptionDrivenDialogue2023].

Seeds used by W Zhang:
1. 2023043004
2. 2023043005
3. 2023043006


```bash
python -m scripts.preprocess_d3st_multiwoz \
  -d data/raw/multiwoz/ \
  --schema_file data/raw/multiwoz/schema.json \
  --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
  -o data/processed/multiwoz_active \
  -c configs/data_processing_d3st_multiwoz__active.yaml \
  --all \
  -vv

# change train dataset path
torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --max_restarts 0 \
  --rdzv_id "$JOB_ID" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$HOST:$PORT" \
  -m scripts.run_dialogue_state_tracking "configs/train_d3st_multiwoz_sample_schema_dialogue.json"

python -m scripts.run_dialogue_state_tracking \
  --model_name_or_path $CHECKPOINT_DIR \
  --output_dir $CHECKPOINT_DIR \
  --cache_dir cache \
  --test_file data/processed/original/test/version_$VERSION/data.json \
  --test_template_dir data/interim/blank_dialogue_templates/$SGD_SHARD/test \
  --test_ref_dir data/raw/original/test \
  --preprocessing_num_workers $PROC_NUM_WORK \
  --per_device_eval_batch_size $INFERENCE_BATCH_SIZE \
  --report_to wandb \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --do_predict \
  --predict_with_generate \
  --max_target_length 512 \
  --val_max_target_length 512
```

> From the models achive:
> 1. multiwoz_cross_dataset/seed_2023043004_multiwoz_sample_schema_dialogue_active/version_1/checkpoint-10200
> 2. multiwoz_cross_dataset/seed_2023043005_multiwoz_sample_schema_dialogue_active/version_1/checkpoint-13600
> 3. multiwoz_cross_dataset/seed_2023043006_multiwoz_sample_schema_dialogue_active/version_1/checkpoint-6800
