from __future__ import annotations

import glob
import json
import logging
import pathlib
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional

from omegaconf import DictConfig

from robust_dst.cli import CustomSeq2SeqTrainingArguments, DataTrainingArguments

PER_FRAME_OUTPUT_FILENAME = "metrics_and_dialogues.json"
ALL_SERVICES = "#ALL_SERVICES"
SEEN_SERVICES = "#SEEN_SERVICES"
UNSEEN_SERVICES = "#UNSEEN_SERVICES"
_TRACKED_METRICS = [
    "average_goal_accuracy",
    "average_cat_accuracy",
    "average_noncat_accuracy",
    "joint_goal_accuracy",
    "joint_cat_accuracy",
    "joint_noncat_accuracy",
]

logger = logging.getLogger(__name__)


def get_dataset_as_dict(file_path_patterns, decoded_only: Optional[list[str]] = None):
    """Read the DSTC8 json dialog data as dictionary with dialog ID as keys.

    Parameters
    ----------
    decoded_only
        Used for code testing with few dialogues. Should contain valid dialogue IDs.
    """
    dataset_dict = {}
    if isinstance(file_path_patterns, list):
        list_fp = file_path_patterns
    else:
        list_fp = sorted(glob.glob(file_path_patterns))
    for fp in list_fp:
        if PER_FRAME_OUTPUT_FILENAME in fp or "belief" in fp:
            continue
        logger.info("Loading file: %s", fp)
        with open(fp, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                for dial in data:
                    dial_id = dial["dialogue_id"]
                    if decoded_only is not None and dial_id not in decoded_only:
                        continue
                    dataset_dict[dial_id] = dial
            elif isinstance(data, dict):
                dataset_dict.update(data)
    return dataset_dict


def get_service_set(schema_path):
    """Get the set of all services present in a schema."""
    service_set = set()
    with open(schema_path, "r") as f:
        schema = json.load(f)
        for service in schema:
            service_set.add(service["service_name"])
    return service_set


def get_in_domain_services(schema_path_1, schema_path_2):
    """Get the set of common services between two schemas."""
    return get_service_set(schema_path_1) & get_service_set(schema_path_2)


def setup_sgd_evaluator_inputs(
    ref_dir: pathlib.Path, decoded_only: Optional[list[str]] = None
) -> dict:
    """Helper function for calling evaluation from training script.

    Parameters
    ----------
    ref_dir
        The directory where the SGD-formatted dialogue files "dialogues_***.json"
        are located.
    decoded_only:
        IDs of dialogues which have been decoded.
        Used to test code with a subsample of the data.

    Returns
    -------
    A mapping containing the positional arguments of the official SGD evaluation script.
    """
    ref_data = get_dataset_as_dict(
        str(ref_dir.joinpath("dialogues_*.json")), decoded_only=decoded_only
    )
    eval_schema_path = ref_dir.joinpath("schema.json")
    with open(eval_schema_path, "r") as f:
        eval_services = {}
        list_services = json.load(f)
        for service in list_services:
            eval_services[service["service_name"]] = service
    # this is the schema of the SGD-X train dataset. We consider "seen" all
    # dialogues which appear in the training set. So even though the schema for
    # e.g., Homes_15 is not seen in training, for evaluation the dialogues in
    # test Homes_15 are considered seen - we saw the dialogues but not the descriptions.
    eval_variant_train_schema_path = ref_dir.parent.joinpath("train", "schema.json")
    assert (
        eval_variant_train_schema_path.exists()
    ), "Could not find the train/ subdir when setting up SGD evaluation"
    in_domain_services = get_in_domain_services(
        eval_schema_path, eval_variant_train_schema_path
    )
    logger.info(f"In domain services: {list(in_domain_services)}")
    logger.info(f"Evaluation schema path {eval_schema_path}")
    logger.info(f"Evaluation services {list(eval_services.keys())}")
    return {
        "eval_services": eval_services,
        "in_domain_services": in_domain_services,
        "dataset_ref": ref_data,
    }


def get_dialogue_ids(preprocessed_refs: list[dict]) -> list[str]:
    return list({ref["dialogue_id"] for ref in preprocessed_refs})


def get_dialogue_filenames(preprocessed_refs: list[dict]) -> list[str]:
    return list({ref["file_name"] for ref in preprocessed_refs})


def get_refs_subset(
    preprocessed_refs: list[dict], dialogue_ids: list[str], dialogue_fnames: list[str]
) -> list[dict]:
    if any((dialogue_ids is None, dialogue_fnames is None)):
        return preprocessed_refs
    preprocessed_refs_subset = []
    for r in preprocessed_refs:
        if r["file_name"] in dialogue_fnames and r["dialogue_id"] in dialogue_ids:
            preprocessed_refs_subset.append(r)
    assert preprocessed_refs_subset
    return preprocessed_refs_subset


def flatten_metrics_dict(sgd_metrics_dict: dict) -> dict[str, float]:
    """
    Turn SGD evaluator output into a mapping from key to values so that it can
    be logged by services such as Tensorboard or ``wandb``.
    """

    service_categories = [ALL_SERVICES, UNSEEN_SERVICES, SEEN_SERVICES]
    flattened_dict = {}
    for service_type in service_categories:
        if service_type not in sgd_metrics_dict:
            logger.warning(
                f"No {service_type} were scored. This is expected if you are testing"
                " code with few samples."
            )
            continue
        this_service_type_metrics = sgd_metrics_dict[service_type]
        for metric_name, value in this_service_type_metrics.items():
            if metric_name in _TRACKED_METRICS:
                flattened_dict[f"{service_type}/{metric_name}"] = value
    # capture domain-level metrics
    for metric_key in sgd_metrics_dict:
        if metric_key not in service_categories and "_" not in metric_key:
            for metric_name, value in sgd_metrics_dict[metric_key].items():
                if metric_name in _TRACKED_METRICS:
                    flattened_dict[f"{metric_key}/{metric_name}"] = value
    return flattened_dict


def setup_sgd_evaluation(
    data_args: DataTrainingArguments | SimpleNamespace,
    preprocessing_configs: dict[str, DictConfig],
    raw_preprocessed_refs: dict[str, list[dict]],
    split: Literal["validation", "test"],
) -> tuple[dict, dict]:
    assert split in ["validation", "test"], (
        f"Cannot setup SGD evaluation for split {split}. "
        "Valid options are 'validation' and 'test'."
    )
    logger.info("Setting up SGD evaluation")
    files_decoded, dialogues_decoded = None, None
    if any(
        (
            data_args.max_eval_samples is not None,
            data_args.max_predict_samples is not None,
        )
    ):
        files_decoded = get_dialogue_filenames(raw_preprocessed_refs[split])
        dialogues_decoded = get_dialogue_ids(raw_preprocessed_refs[split])
    schema_path = Path(getattr(data_args, f"{split}_ref_dir")).joinpath("schema.json")
    logger.info(f"Retrieving schema from path {schema_path}")
    parser_inputs = {
        "preproc_config": preprocessing_configs[split].get("preprocessing"),
        "template_dir": getattr(data_args, f"{split}_template_dir"),
        "schema_path": schema_path,
        "files_to_parse": files_decoded,
        "dialogues_decoded": dialogues_decoded,
        "preprocessed_refs": get_refs_subset(
            raw_preprocessed_refs[split],
            dialogues_decoded,
            files_decoded,
        ),
    }
    evaluator_inputs = setup_sgd_evaluator_inputs(
        Path(getattr(data_args, f"{split}_ref_dir")), decoded_only=dialogues_decoded
    )
    return parser_inputs, evaluator_inputs


def setup_multiwoz_evaluation(
    data_args: DataTrainingArguments,
    preprocessing_configs: dict[str, DictConfig],
    raw_preprocessed_refs: dict[str, list[dict]],
    split: Literal["validation", "test"],
) -> dict:
    assert split in ["validation", "test"], (
        f"Cannot setup MultiWOZ evaluation for split {split}. "
        "Valid options are 'validation' and 'test'."
    )
    logger.info("Setting up MultiWOZ evaluation")
    files_decoded, dialogues_decoded = None, None
    if any(
        (
            data_args.max_eval_samples is not None,
            data_args.max_predict_samples is not None,
        )
    ):
        files_decoded = get_dialogue_filenames(raw_preprocessed_refs[split])
        dialogues_decoded = get_dialogue_ids(raw_preprocessed_refs[split])
    schema_path = Path(preprocessing_configs[split].metadata.raw_data_path).joinpath(
        "schema.json"
    )
    logger.info(f"Retrieving schema from path {schema_path}")
    parser_inputs = {
        "preproc_config": preprocessing_configs[split].get("preprocessing"),
        "template_dir": None,
        "schema_path": schema_path,
        "files_to_parse": files_decoded,
        "dialogues_decoded": dialogues_decoded,
        "preprocessed_refs": get_refs_subset(
            raw_preprocessed_refs[split],
            dialogues_decoded,
            files_decoded,
        ),
    }
    return parser_inputs


def infer_step(training_args: CustomSeq2SeqTrainingArguments) -> int:
    checkpoint_dir = Path(training_args.output_dir)
    if "checkpoint-" in checkpoint_dir.name:
        return int(checkpoint_dir.name.split("-")[1])
    raise ValueError(
        f"Wrong output_dir setting in training_args. Got {checkpoint_dir}."
    )


def setup_evaluator_output_dirs(
    training_args: CustomSeq2SeqTrainingArguments,
    split: Literal["test", "dev"],
    step: Optional[int] = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Create the folder hierarchy for storing evaluator aggregated and
    frame-level outputs.

    This is used both during traing to save dev set model predictions and
    task-oriented eval results as well as during inference.
    """

    assert split in [
        "test",
        "dev",
    ], f"Expected split to be either 'dev' or 'test' but got {split}"
    logger.info("Setting up directories to store evaluation output")
    checkpoint_dir = Path(training_args.output_dir)
    logger.info(f"Checkpoint directory is {checkpoint_dir}")
    # checkpoint dir is suffixed with /checkpoint-[step] in inference
    if "checkpoint-" in checkpoint_dir.name:
        if step is None:
            step = infer_step(training_args)
        input_data_version = checkpoint_dir.parent.name
        experiment_name = checkpoint_dir.parent.parent.name
    else:
        assert step is not None
        input_data_version = checkpoint_dir.name
        experiment_name = checkpoint_dir.parent.name
        try:
            assert (
                "models" in checkpoint_dir.parent.parent.name
            ), f"checkpoint dir ancestor is {checkpoint_dir.parent.parent.name}"
        except AssertionError:
            assert (
                "output" in checkpoint_dir.parent.parent.name
            ), f"checkpoint dir ancestor is {checkpoint_dir.parent.parent.name}"
    assert (
        "version_" in input_data_version
    ), f"Input data version is {input_data_version}"
    assert "version_" not in experiment_name, f"Experiment name is {experiment_name}"
    data_variant = training_args.data_variant
    assert data_variant is not None
    logger.info(f"Inferred input data version: {input_data_version}")
    logger.info(f"Inferred experiment name: {experiment_name}")
    logger.info(f"Inferred schema variant: {data_variant}")
    dir_hierarchy = (
        experiment_name,
        data_variant,
        split,
        input_data_version,
    )
    hyp_dir = (
        Path(training_args.hyps_dir)
        .resolve()
        .joinpath(*dir_hierarchy, f"checkpoint-{step}")
    )
    metrics_dir = Path(training_args.metrics_dir).resolve().joinpath(*dir_hierarchy)
    if not hyp_dir.exists():
        logger.info(f"Creating hyps directory {str(hyp_dir)}")
        hyp_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_dir.exists():
        logger.info(f"Creating metrics directory {str(metrics_dir)}")
        metrics_dir.mkdir(parents=True, exist_ok=True)
    return hyp_dir, metrics_dir


def save_evaluator_outputs(
    hyp_dir: Path,
    metrics_dir: Path,
    all_metrics_aggregate: dict,
    file_to_hyp_dials: dict[str, list[dict]],
    predictions: list[str],
    step: int,
):
    """
    Save evaluator aggregated metrics, frame level output and raw model predictions.

    Parameters
    ----------
    all_metrics_aggregate
        Aggregated metrics output by official evaluation scripts.
    file_to_hyp_dials:
        Mapping from SGD dialogue filenames (e.g., `dialogues_001.json`) to lists
        of SGD-format dialogues, or the key "data.json" to state predicitions formatted
        according to the official MultiWOZ evaluation script
        (https://github.com/Tomiinek/MultiWOZ_Evaluation).
    hyp_dir, metrics_dir:
        Directories where the evaluator outputs are saved. The detailed,
        frame level output is saved in `hyp_dir` whereas the aggregated metrics
        in `metrics_dir`. `hyp_dir` is joined with `checkpoint-[step]`
        so that SGD-fromat dialogue files containing model predictions and
        detailed output (or the MultiWOZ states) are saved for a given checkpoint.
    predictions:
        Raw model predictions.
    step:
        Represents the training step (unit is number of weight updates) at which
        evaluation was carried out. Used to save the evaluator results in a
        subdirectory of `hyp_dir` that corresponds to the evaluated checkpoint.
    """

    if "data.json" in file_to_hyp_dials:
        dataset_hyp = file_to_hyp_dials["data.json"]
    else:
        dataset_hyp = {
            dial["dialogue_id"]: dial for dial in chain(*file_to_hyp_dials.values())
        }
    current_step_metrics_pth = metrics_dir.joinpath(f"model_{step}_metrics.json")
    logger.info(f"Saving aggregated evaluator output at {current_step_metrics_pth}")
    with open(current_step_metrics_pth, "w") as f:
        json.dump(
            all_metrics_aggregate, f, indent=2, separators=(",", ": "), sort_keys=True
        )
    current_step_detailed_output_pth = hyp_dir.joinpath(PER_FRAME_OUTPUT_FILENAME)
    logger.info(
        f"Saving frame-level evaluator output at {current_step_detailed_output_pth}"
    )
    with open(current_step_detailed_output_pth, "w") as f:
        json.dump(dataset_hyp, f, indent=2, separators=(",", ": "))
    for fname, this_file_dials in file_to_hyp_dials.items():
        current_step_sgd_format_predictions_pth = hyp_dir.joinpath(fname)
        logger.info(
            "Saving formatted predictions at"
            f" {current_step_sgd_format_predictions_pth}"
        )
        with open(current_step_sgd_format_predictions_pth, "w") as f:
            json.dump(this_file_dials, f, indent=2)
    # save raw predictions and experiment config to support parsing & scoring
    # independent of this script
    output_prediction_file = hyp_dir.joinpath("generated_predictions.txt")
    logger.info(f"Saving raw predictions at {output_prediction_file}")
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(predictions))
