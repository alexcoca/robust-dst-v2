from __future__ import annotations

import json
import logging
import pathlib
import random
import re
import sys
from copy import deepcopy
from typing import Literal, Optional

import click
from omegaconf import DictConfig, OmegaConf

from robust_dst.parser_metadata import SPECIAL_SLOT_VALUES
from robust_dst.utils import (
    get_datetime,
    infer_schema_variant_from_path,
    save_data,
    set_seed,
)

logger = logging.getLogger(__name__)


def get_qa_string(
    values: list[str], target_value: str, randomize_order: bool = True
) -> tuple[str, str, dict[str, str]]:
    # convert a list of values into a string of choices,
    # e.g., ['N', 'E', "S", 'W'] -> "a) N b) E c) S d) W"
    if randomize_order:
        random.shuffle(values)

    qa_str = ""
    option2value = {}
    for i, value in enumerate(values):
        option = chr(ord("a") + i)  # a, b, c ..
        qa_str += option + ") " + value + " "
        option2value[option] = value
        # get the target value option
        if target_value == value:
            target_value = option
    if target_value not in ["NONE", "dontcare"]:
        assert target_value in option2value.keys()
    return qa_str, target_value, option2value


def preprocess_file(
    raw_dialogues: list[dict],
    fname: str,
    service2schema,
    data_format: Literal["T5DST", "T5DST2", "google"],
    downsample_factor: int = 1,
) -> dict[str, list[dict]]:
    this_file_dialogues, this_file_dialogues_metadata = [], []
    for dial_idx in range(len(raw_dialogues)):
        if dial_idx % downsample_factor != 0:
            continue
        dial = raw_dialogues[dial_idx]
        cur_dial = ""
        dialogue_id = dial["dialogue_id"]
        for turn_idx, turn in enumerate(dial["turns"]):
            speaker = " [" + turn["speaker"] + "] "
            uttr = turn["utterance"]
            cur_dial += speaker
            cur_dial += uttr
            if turn["speaker"] == "SYSTEM":
                continue
            # prepare examples per user turn
            active_slot_values = {}
            ground_truth_services = (
                []
            )  # use list to remove randomness when iterating elements in set
            for frame_idx in range(len(turn["frames"])):
                frame = turn["frames"][frame_idx]
                # get service
                service = frame["service"]
                ground_truth_services.append(service)
                # get state
                for slot, values in frame["state"]["slot_values"].items():
                    # make slot dependent with service in state dict
                    # to differentiate same slot names across services
                    service_slot = service + "-" + slot
                    if data_format in ["T5DST", "T5DST2"]:
                        value = random.sample(values, 1)[0]
                        active_slot_values[service_slot] = value
                    else:  # google format with multiple values as target
                        active_slot_values[service_slot] = " || ".join(values)
            # iterate all slots in ground-truth services
            for service_idx, service in enumerate(ground_truth_services):
                service_wo_index = service.split("_")[0]
                service_schema = service2schema[service]
                for slot_metadata in service_schema["slots"]:
                    slot = slot_metadata["name"]
                    slot_wi_space = " ".join(slot.split("_"))
                    # generate schema prompt w/ or w/o natural language descriptions
                    schema_prompt = ""
                    if data_format in ["T5DST", "T5DST2"]:
                        schema_prompt += (
                            " [domain] "
                            + service_wo_index
                            + " "
                            + service_schema["description"]
                        )
                        if data_format == "T5DST":
                            schema_prompt += (
                                " [slot] "
                                + slot_wi_space
                                + " "
                                + slot_metadata["description"]
                            )
                        else:  # T5DST2
                            schema_prompt += " [slot] " + slot_metadata["description"]
                    else:  # google format with only slot description
                        schema_prompt += " [slot] " + slot_metadata["description"]

                    service_slot = service + "-" + slot
                    if service_slot in active_slot_values.keys():
                        target_value = active_slot_values[service_slot]
                    else:
                        target_value = "NONE"  # special token for inactive slots

                    # only append possible values if the slot is categorical
                    if slot_metadata["is_categorical"]:
                        if data_format in ["T5DST", "T5DST2"]:
                            this_slot_values = deepcopy(
                                slot_metadata["possible_values"]
                            )
                            if (
                                service in SPECIAL_SLOT_VALUES
                                and slot in SPECIAL_SLOT_VALUES[service]
                            ):
                                replacement_value = SPECIAL_SLOT_VALUES[service][slot]
                                ambiguous_val_index = this_slot_values.index("None")
                                this_slot_values[
                                    ambiguous_val_index
                                ] = replacement_value
                            allowed_values = ", ".join(this_slot_values)
                            schema_prompt += " [PVs] " + allowed_values
                        else:
                            qa_str, target_value, option2value = get_qa_string(
                                slot_metadata["possible_values"], target_value
                            )
                            schema_prompt += " " + qa_str

                    # write data
                    if data_format in ["T5DST", "T5DST2"]:
                        processed_dialogue = {
                            "dialogue": (cur_dial + schema_prompt),
                            "state": target_value,
                        }
                    else:
                        processed_dialogue = {
                            "dialogue": (cur_dial + schema_prompt).lower(),
                            "state": target_value.lower(),
                        }

                    this_file_dialogues.append(processed_dialogue)
                    # write idx file for post-processing
                    turn_metadata = {
                        "dialogue_idx": str(dial_idx),
                        "dialogue_id": dialogue_id,
                        "turn_idx": str(turn_idx),
                        "service_idx": service_idx,
                        "service": service,
                        "slot": slot,
                        "file_name": fname,
                    }
                    if data_format == "google":
                        if slot_metadata["is_categorical"]:
                            turn_metadata["option2value"] = json.dumps(option2value)
                        else:
                            turn_metadata["option2value"] = json.dumps({})
                    else:
                        turn_metadata["option2value"] = json.dumps({})
                    processed_dialogue.update(turn_metadata)
                    this_file_dialogues_metadata.append(turn_metadata)
    return {
        "proc_dials": this_file_dialogues,
        "metadata": this_file_dialogues_metadata,
    }


def process_schema(schema: list[dict]) -> dict[str, dict]:
    """Create a mapping from service names to service schemas."""
    service2schema = {}
    for service_meta in schema:
        service = service_meta["service_name"]
        service2schema[service] = service_meta
    return service2schema


def update_data_config(config: DictConfig):
    """Update data configuration with a description of the linearization strategy."""
    data_version = config.linearization_version
    if data_version == "version_1":
        config.metadata.data_format = "google"
        config.metadata.data_desc = (
            "This is google data format, which follows what the mail says. Key points:"
            " 1) no slot names in input 2) use QA format for categorical slots 3)"
            " everything lower case"
        )
    elif data_version == "version_2":
        config.metadata.data_format = "T5DST"
        config.metadata.data_desc = (
            "This is the data format that was useed in the T5DST paper, but with SGD"
            " corpus."
        )
    elif data_version == "version_3":
        config.metadata.data_format = "T5DST2"
        config.metadata.data_desc = (
            "Same format as version_2 but without slot name in input"
        )
    else:
        print("Wrong data version")
        sys.exit(1)
    assert config.metadata.data_format in [
        "T5DST",
        "google",
        "T5DST2",
    ]  # T5DST2 has no slot name


@click.command()
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option(
    "-c",
    "--config",
    "cfg_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to data preprocessing config file.",
)
@click.option(
    "-d",
    "--data_paths",
    "data_paths",
    required=True,
    type=click.Path(exists=True),
    help="Path to one or more raw SGD data directories.",
    multiple=True,
)
@click.option(
    "-o",
    "--output_path",
    "output_path",
    required=True,
    type=click.Path(exists=False),
    help="Directory where processed data is output.",
)
@click.option(
    "-ver",
    "--version",
    "version",
    default=None,
    help=(
        "By default, the version is incremented automatically when the pre-processing"
        " script is run. Use this option when you pre-process data with a given data"
        " format for differentexperiments to avoid version discrepancies and errors"
        " while decoding."
    ),
)
@click.option(
    "--override", is_flag=True, default=False, help="Override previous results."
)
@click.option("--train", "split", flag_value="train")
@click.option("--dev", "split", flag_value="dev")
@click.option("--dev_small", "split", flag_value="dev_small")
@click.option("--test", "split", flag_value="test")
def main(
    cfg_path: pathlib.Path,
    log_level: int,
    data_paths: tuple[str],
    output_path: pathlib.Path,
    override: bool,
    split: str,
    version: Optional[int],
):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = OmegaConf.load(cfg_path)
    set_seed(config.reproduce)
    config.metadata.date = get_datetime()
    config.metadata.raw_data_path = [p for p in data_paths]
    config.metadata.split = split
    config.metadata.output_path = output_path
    output_path = pathlib.Path(output_path)
    data_paths = [pathlib.Path(p) for p in data_paths]
    update_data_config(config)
    for shard_path in data_paths:
        logger.info(
            f"--------Preprocessing {split} set for shard {shard_path}---------"
        )
        this_shard_data_dir = shard_path.joinpath(split)
        schema_variant = infer_schema_variant_from_path(str(this_shard_data_dir))
        logger.info(f"Inferred schema variant: {schema_variant}")
        config.metadata.schema_variant = schema_variant
        if shard_path.joinpath(f"{split}_generator_config.yaml").exists():
            gen_config = OmegaConf.load(
                shard_path.joinpath(f"{split}_generator_config.yaml")
            )
            config.metadata.generator = gen_config
        with open(this_shard_data_dir.joinpath("schema.json"), "r") as f:
            schema = json.load(f)
        service2schema = process_schema(schema)
        preprocessed_dialogues = {"data": []}
        metadata_out = []
        pattern = re.compile(r"dialogues_[0-9]+\.json")
        this_shard_files = sorted(
            [f for f in this_shard_data_dir.iterdir() if pattern.match(f.name)],
            key=lambda fpath: int((fpath.name.split("_")[1]).split(".")[0]),
        )
        for file in this_shard_files:
            if pattern.match(file.name):
                logger.info(f"Processing file {file.name}")
                with open(file, "r") as f:
                    raw_dialogues = json.load(f)
                preproc_file = preprocess_file(
                    raw_dialogues,
                    file.name,
                    service2schema,
                    config.metadata.data_format,
                    downsample_factor=config.downsample_factor,
                )
                this_file_metadata = preproc_file["metadata"]
                preprocessed_dialogues["data"].extend(preproc_file["proc_dials"])
                metadata_out.extend(this_file_metadata)
        logger.info(
            f"Extracted examples {len(preprocessed_dialogues['data'])} for shard"
            f" {shard_path}, split {split}"
        )
        save_data(
            preprocessed_dialogues,
            output_path.joinpath(schema_variant, split),
            metadata=config,
            version=version,
            override=override,
        )
        logger.info("--------Finished Preprocessing---------")


if __name__ == "__main__":
    main()
