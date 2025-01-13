#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
import json
import logging
import os
import re
import sys
from itertools import chain
from pathlib import Path
from typing import Union

import datasets
import transformers
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from transformers import (
    Adafactor,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    get_constant_schedule_with_warmup,
    is_wandb_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from robust_dst.callbacks import CacheManagerCallback, CustomWandbCallback, JGAEarlyStoppingCallback
from robust_dst.cli import (
    CustomSeq2SeqTrainingArguments,
    DataTrainingArguments,
    ModelArguments,
)
from robust_dst.evaluation import get_metrics
from robust_dst.parser import D3STParser, T5DSTParser, SDTParser
from robust_dst.preprocessor import D3STPreprocessor, T5DSTPreprocessor, SDTPreprocessor
from robust_dst.scoring_utils import (
    flatten_metrics_dict,
    setup_evaluator_output_dirs,
    setup_sgd_evaluation,
)
from robust_dst.trainer import CustomTrainer
from robust_dst.utils import (
    infer_data_version_from_path,
    infer_schema_variant_from_path,
)

# Will error if the minimal version of Transformers is not installed.
# Remove at your own risks.
check_min_version("4.19.2")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/summarization/requirements.txt",
)

logger = logging.getLogger(__name__)



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    arg_parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomSeq2SeqTrainingArguments)
    )
    if sys.argv[-1].endswith(".json"):
        # parse arguments passed in a .json file
        json_file_path = os.path.abspath(sys.argv[-1])
        logger.info(f"Parsing arguments in .json format at path {json_file_path}")
        model_args, data_args, training_args = arg_parser.parse_json_file(
            json_file=json_file_path
        )
    else:
        logger.info("Parsing arguments into dataclasses")
        model_args, data_args, training_args = arg_parser.parse_args_into_dataclasses()
    if not Path(model_args.cache_dir).exists():
        Path(model_args.cache_dir).resolve().mkdir(parents=True, exist_ok=True)
    if training_args.do_predict and training_args.do_eval:
        raise ValueError(
            "Predict and eval cannot be active at the same time. Setting do_eval=False"
        )
    # Setup logging
    output_dir_pth = Path(training_args.output_dir)
    if training_args.do_predict:
        schema_variant = infer_schema_variant_from_path(data_args.test_file)
        file_name = f"predict_{schema_variant}_{Path(__file__).stem}"
        file_handler = logging.FileHandler(
            f"{output_dir_pth.joinpath(file_name)}.log",
        )
    else:
        assert training_args.do_train
        # save checkpoints in a hierarchy that depends on experiment name
        # and model input data version
        model_input_data_versions = list(
            {infer_data_version_from_path(p) for p in data_args.train_file}
        )
        assert (
            len(model_input_data_versions) == 1
        ), "Cannot train on multiple input data versions."
        model_input_data_version = model_input_data_versions[0]
        if not training_args.resume_from_checkpoint:
            output_dir = os.path.join(
                training_args.output_dir,
                training_args.experiment_name,
                model_input_data_version,
            )
            log_dir = os.path.join(
                training_args.output_dir,
                training_args.experiment_name,
                "logs",
                model_input_data_version,
            )
            if not Path(log_dir).exists():
                Path(log_dir).mkdir(exist_ok=True, parents=True)
            training_args.output_dir = output_dir
            training_args.run_name = (
                f"{training_args.run_name}_{model_input_data_version}"
            )
            logger.info(f"Updated training_args.run_name to {training_args.run_name}")
            file_handler = logging.FileHandler(
                f"{Path(log_dir).joinpath(f'{Path(__file__).stem}')}.log"
            )
        else:
            # adapt wandb run name
            logger.info(f"Current wandb run name is {training_args.run_name}")
            training_args.run_name = (
                f"{training_args.experiment_name}_{model_input_data_version}"
            )
            logger.info(
                f"Changed run name to include data version: {training_args.run_name}"
            )
            if "checkpoint" in output_dir_pth.name:
                logs_dir = output_dir_pth.parent.parent
            else:
                logs_dir = output_dir_pth.parent
            logs_dir = logs_dir.joinpath(
                "logs", model_input_data_version, f"{Path(__file__).stem}.log"
            )
            file_handler = logging.FileHandler(logs_dir, mode="a")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            file_handler,
        ],
        force=True,
    )
    log_level = training_args.get_process_log_level()
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device},"
        f" n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits"
        f" training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 2:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is"
                " not empty. Use --overwrite_output_dir to overcome. Found"
                f" {list(os.listdir(training_args.output_dir))}"
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid"
                " this behavior, change the `--output_dir` or add"
                " `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    preprocessing_configs = {}  # type: dict[str, DictConfig]
    data_files = {}  # type: dict[str, str]
    extension = "json"
    if data_args.train_file is not None and data_args.train_file:
        data_files["train"] = data_args.train_file
        preprocessing_configs["train"] = OmegaConf.load(
            Path(data_args.train_file[0]).parent.joinpath("preprocessing_config.yaml")
        )
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
        data_version = infer_data_version_from_path(data_args.validation_file)
        logger.info(f"Inferred data version {data_version} for validation file ...")
        preprocessing_configs["validation"] = OmegaConf.load(
            Path(data_args.validation_file).parent.joinpath("preprocessing_config.yaml")
        )
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        data_version = infer_data_version_from_path(data_args.test_file)
        logger.info(f"Inferred data version {data_version} for validation file ...")
        preprocessing_configs["test"] = OmegaConf.load(
            Path(data_args.test_file).parent.joinpath("preprocessing_config.yaml")
        )
        extension = data_args.test_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension, data_files=data_files, cache_dir=model_args.cache_dir, field="data"
    )
    # keep a copy of the original data structure for parsing
    raw_preprocessed_refs = {}  # type: dict[str, list[dict]]
    for split in raw_datasets:
        if split == "train":
            continue
        raw_preprocessed_refs[split] = raw_datasets.data[split].table.to_pylist()
        if split == "test" and data_args.max_predict_samples is not None:
            raw_preprocessed_refs[split] = raw_preprocessed_refs[split][
                : data_args.max_predict_samples
            ]
        if split == "validation" and data_args.max_eval_samples is not None:
            raw_preprocessed_refs[split] = raw_preprocessed_refs[split][
                : data_args.max_eval_samples
            ]

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if data_args.val_max_target_length is not None:
        config.max_length = data_args.val_max_target_length
    if data_args.num_beams is not None:
        config.num_beams = data_args.num_beams

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Must make sure the order of additional special tokens to be consistent
    additional_special_tokens = None
    for preprocessing_config in preprocessing_configs.values():
        split_special_tokens = sorted(
            preprocessing_config.preprocessing.get("special_tokens", [])
        )
        if additional_special_tokens is None:
            additional_special_tokens = split_special_tokens
        else:
            if additional_special_tokens != split_special_tokens:
                raise ValueError(
                    "Data preprocessing configs of the splits contain different"
                    "special tokens"
                )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )
    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to"
                f" {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but"
                f" the model only has {model.config.max_position_embeddings} position"
                " encodings. Consider either reducing `--max_source_length` to"
                f" {model.config.max_position_embeddings} or to automatically resize"
                " the model's position encodings by passing"
                " `--resize_position_embeddings`."
            )

    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the"
            " `prepare_decoder_input_ids_from_labels` method is not defined"
            f" for`{model.__class__.__name__}`. This will lead to loss being calculated"
            " twice and will take up more memory"
        )

    if not (
        training_args.do_train or training_args.do_eval or training_args.do_predict
    ):
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or"
            " `do_predict`."
        )
        return

    # Data preprocessing
    data_formats = {
        preprocessing_config.metadata.data_format
        for preprocessing_config in preprocessing_configs.values()
    }
    if len(data_formats) > 1:
        raise RuntimeError(
            "train, validation and test datasets have different data formats"
        )
    data_format = data_formats.pop().lower()

    preprocessor_init_kwargs = {
        "tokenizer": tokenizer,
        "max_source_length": data_args.max_source_length,
        "max_target_length": data_args.max_target_length,
        "padding": "max_length" if data_args.pad_to_max_length else False,
        "ignore_pad_token_for_loss": data_args.ignore_pad_token_for_loss,
        "num_proc": data_args.preprocessing_num_workers,
        "load_from_cache_file": not data_args.overwrite_cache,
        "source_column": data_args.source_column,
        "target_column": data_args.target_column,
        "input_prefix": data_args.source_prefix
        if data_args.source_prefix is not None
        else "",
    }
    if "d3st" in data_format:
        delimiters = {
            preprocessing_config.preprocessing.delimiter
            for preprocessing_config in preprocessing_configs.values()
        }
        if len(delimiters) > 1:
            raise RuntimeError(
                "train, validation and test datasets have different delimiters"
            )

        domain_in_desc = False
        preprocessor = D3STPreprocessor(
            delimiter=delimiters.pop(),
            domain_in_desc=domain_in_desc,
            **preprocessor_init_kwargs,
        )
    elif "sdt" in data_format:
        preprocessor = SDTPreprocessor(
            **preprocessor_init_kwargs,
        )
    else:
        preprocessor = T5DSTPreprocessor(
            **preprocessor_init_kwargs,
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        logger.info(
            f"Processing training dataset, KST augmentation: {data_args.augment_style}"
        )
        with training_args.main_process_first(
            local=False, desc="train dataset map pre-processing"
        ):
            train_dataset = preprocessor.process(
                train_dataset,
                desc="Running tokenizer on train dataset",
                augment_style=data_args.augment_style,
                truncation=False,
                omit_confirmation_turns=data_args.omit_confirmation_turns,
                discard_truncated_examples=data_args.discard_truncated_examples,
            )
            if data_args.augment_style != "NONE":
                # this is tokenized so the tokenizer needs to be loaded to detokenize
                # the data
                train_dataset.to_json(f"{training_args.output_dir}/train_dataset.json")

    if training_args.do_eval:
        preprocessor.max_target_length = data_args.val_max_target_length
        training_args.data_variant = "original"
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        logger.info("Processing development dataset...")
        with training_args.main_process_first(
            local=False, desc="validation dataset map pre-processing"
        ):
            eval_dataset = preprocessor.process(
                eval_dataset,
                truncation=False,
                desc="Running tokenizer on validation dataset",
            )
        parser_inputs, sgd_evaluator_inputs = setup_sgd_evaluation(
            data_args, preprocessing_configs, raw_preprocessed_refs, "validation"
        )

    if training_args.do_predict:
        data_variant = infer_schema_variant_from_path(data_args.test_file)
        training_args.run_name = f"{training_args.run_name}_{data_variant}"
        logger.info(f"Prediction results will be logged under {training_args.run_name}")
        if is_wandb_available():
            if training_args.wandb_tags is None:
                training_args.wandb_tags = [data_variant, "evaluation"]
            elif data_variant not in training_args.wandb_tags:
                training_args.wandb_tags.append(data_variant)
                if "evaluation" not in training_args.wandb_tags:
                    training_args.wandb_tags.append("evaluation")
            logger.info(
                f"Added automatic tags for inference run: {training_args.wandb_tags}"
            )
        training_args.data_variant = data_variant
        preprocessor.max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(
                range(data_args.max_predict_samples)
            )
        with training_args.main_process_first(
            local=False, desc="prediction dataset map pre-processing"
        ):
            if data_args.augment_style not in ("NONE", "TURN", "TURNSLOT"):
                raise ValueError(
                    "For prediction, augment style must be one of one of NONE, TURN,"
                    " TURNSLOT"
                )
            kst_table = None
            if data_args.augment_style in ("TURN", "TURNSLOT"):
                if data_args.test_kst_table_dir is not None:
                    with open(data_args.test_kst_table_dir, "r") as f:
                        kst_table = json.load(f)
            predict_dataset = preprocessor.process(
                predict_dataset,
                augment_style=data_args.augment_style,
                kst_table=kst_table,
                truncation=False,  # do not truncate inputs when predict
                desc="Running tokenizer on prediction dataset",
                iterative_decoding=data_args.iterative_decoding,
            )
        parser_inputs, sgd_evaluator_inputs = setup_sgd_evaluation(
            data_args, preprocessing_configs, raw_preprocessed_refs, "test"
        )

    # Optimizer and scheduler
    optimizer, scheduler = None, None
    if training_args.optimizer:
        # If training_args.optimizer is set by user
        # Setup optimizer/scheduler according to training args
        if training_args.do_train:
            if training_args.optimizer == "adamw":
                optimizer = AdamW(
                    model.parameters(),
                    lr=training_args.learning_rate,
                )
            elif training_args.optimizer == "adam":
                optimizer = Adam(
                    model.parameters(),
                    lr=training_args.learning_rate,
                )
            elif training_args.optimizer == "adafactor":
                optimizer = Adafactor(
                    model.parameters(),
                    lr=training_args.learning_rate,
                    relative_step=False,
                    scale_parameter=False,
                )
            else:
                logger.warning(
                    f"Unknown optimizer {training_args.optimizer}, defaulting to"
                    " huggingface defaults!"
                )
            if training_args.use_scheduler:
                scheduler = get_constant_schedule_with_warmup(
                    optimizer, num_warmup_steps=training_args.scheduler_warmup_steps
                )

    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metrics
    parser_init_kwargs = {
        "template_dir": parser_inputs["template_dir"],
        "schema_path": parser_inputs["schema_path"],
        "files_to_parse": parser_inputs["files_to_parse"],
        "dialogues_to_parse": parser_inputs["dialogues_decoded"],
    }
    if "d3st" in data_format:
        parser = D3STParser(
            value_separator=parser_inputs["preproc_config"]["value_separator"],
            target_slot_index_separator=parser_inputs["preproc_config"]["delimiter"],
            restore_categorical_case=parser_inputs["preproc_config"]["lowercase"],
            **parser_init_kwargs,
        )
    elif "sdt" in data_format:
        parser = SDTParser(
            data_format=data_format,
            **parser_init_kwargs)
    else:
        parser = T5DSTParser(
            data_format=data_format,
            **parser_init_kwargs,
        )

    def remove_eos_token_inplace(predictions: list[str]) -> None:
        """Remove the eos token.

        Do not call tokenizer.batch_decode with skip_special_tokens because eos may
        not be the only special token.
        """
        for idx, pred in enumerate(predictions):
            try:
                predictions[idx] = (
                    re.search("(.*)" + re.escape(tokenizer.eos_token), pred)
                    .group(1)
                    .strip()
                )
            except AttributeError:
                logger.error(
                    f"No EOS token {tokenizer.eos_token} can be found in prediction:"
                    f" {pred}"
                )
                continue

    def compute_metrics(eval_preds) -> dict:
        """Compute DST metrics.

        Works for both SGD and MultiWOZ dataset, using the corresponding offical
        evaluation scripts.

        Returns:
            A dictionary of the following format:
                {
                    **$flattened_metrics,
                    "all_metrics_aggregate": $possibly_nested_metrics,
                    "file_to_hyp_dials": $file_to_hype_dials,
                    "raw_predictions": $raw_model_predicitions_list,
                }

            $file_to_hyp_dials is a mapping
                - For SGD: from SGD dialogue filenames (e.g., `dialogues_001.json`) to
                    lists of SGD-format dialogues, or

            The key-values "all_metric_aggregate", "file_to_hyp_dials" and
            "raw_predictions" are popped off and used to save metrics and
            predictions locally later. The $flattened_metrics are then reported
            to services like wandb.
        """
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        predictions = tokenizer.batch_decode(preds)
        remove_eos_token_inplace(predictions)

        logger.info("Computing metrics")
        preproc_refs = parser_inputs["preprocessed_refs"]
        file_to_hyp_dials = parser.convert_to_sgd_format(
            preprocessed_refs=preproc_refs,
            predictions=predictions,
        )
        assert len(predictions) == len(
            preproc_refs
        ), f"Expected {len(preproc_refs)} predictions but got {len(predictions)}"

        dataset_hyp = {
            dial["dialogue_id"]: dial for dial in chain(*file_to_hyp_dials.values())
        }
        all_metrics_aggregate, _ = get_metrics(
            dataset_ref=sgd_evaluator_inputs["dataset_ref"],
            dataset_hyp=dataset_hyp,
            service_schemas=sgd_evaluator_inputs["eval_services"],
            in_domain_services=sgd_evaluator_inputs["in_domain_services"],
        )
        result = flatten_metrics_dict(all_metrics_aggregate)  # type: dict

        logger.info("all metrics aggregate", all_metrics_aggregate)
        result["all_metrics_aggregate"] = all_metrics_aggregate
        result["file_to_hyp_dials"] = file_to_hyp_dials
        result["raw_predictions"] = predictions
        return result

    # Callbacks
    early_stopping_callback = JGAEarlyStoppingCallback(
        training_args.early_stopping_patience
    )
    if is_wandb_available():
        callbacks = [CacheManagerCallback, early_stopping_callback, CustomWandbCallback]
    else:
        callbacks = [CacheManagerCallback, early_stopping_callback]

    if training_args.generation_max_length is None:
        training_args.generation_max_length = data_args.val_max_target_length
    if training_args.generation_num_beams is None:
        training_args.generation_num_beams = data_args.num_beams
    logger.info(
        f"Generation max length: {training_args.generation_max_length},"
        f" generation num beams: {training_args.generation_num_beams}"
    )

    logger.info("Initialising trainer...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler) if training_args.do_train else (None, None),
        callbacks=callbacks,
        compute_metrics=compute_metrics
        if training_args.predict_with_generate
        else None,
    )

    def create_and_save_model_config(path: Union[str, Path]) -> None:
        config = {
            "data": preprocessing_configs,
            "train_arguments": {
                k: v
                for k, v in training_args.__dict__.items()
                if not k.startswith("__")
            },
            "data_args": data_args.__dict__,
            "model_args": model_args.__dict__,
        }

        # it's better to use pydantic here, but HuggingFace config classes have
        # inconsistencies in type annoations at the time of writing this code
        if "distributed_state" in config["train_arguments"]:
            distributed_state_str = str(config["train_arguments"]["distributed_state"])
            config["train_arguments"]["distributed_state"] = distributed_state_str

        model_config = OmegaConf.create(config)
        # needed for post-hoc parsing of raw predictions
        OmegaConf.save(config=model_config, f=path)


    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        create_and_save_model_config(output_dir_pth.joinpath("experiment_config.yaml"))

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        logger.info("SGD metrics", metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        assert (
            not training_args.do_train and not training_args.do_eval
        ), "Cannot run script in training and inference mode simultaneously"
        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        logger.info("SGD metrics", metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                checkpoint_dir = output_dir_pth
                step = int(checkpoint_dir.name.split("-")[1])
                hyp_dir, metrics_dir = setup_evaluator_output_dirs(
                    training_args, "test", step
                )
                model_config = OmegaConf.create(
                    {
                        "data": preprocessing_configs,
                        "train_arguments": {
                            k: v
                            for k, v in training_args.__dict__.items()
                            if not k.startswith("__")
                        },
                        "data_args": data_args.__dict__,
                        "model_args": model_args.__dict__,
                    }
                )
                OmegaConf.save(
                    config=model_config, f=hyp_dir.joinpath("experiment_config.yaml")
                )

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "schema_guided_dialogue_state_tracking",
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
