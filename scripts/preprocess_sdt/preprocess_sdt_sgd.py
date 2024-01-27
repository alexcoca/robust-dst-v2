from __future__ import annotations

import copy
import dataclasses
import json
import csv
import logging
import pathlib
import random
import re
import string
import sys
from collections import OrderedDict, defaultdict
from functools import partial
from itertools import product
from multiprocessing import Pool
from typing import Any, Optional, Union

import click
from omegaconf import DictConfig, OmegaConf

from robust_dst.utils import (
    get_datetime,
    infer_schema_variant_from_path,
    save_data,
    set_seed,
)

logger = logging.getLogger(__name__)


def extract_categorical_mapping(source):
    pattern = r"\[categorical_mapping_prompt\](.*?)\[example\](.*)"
    match = re.match(pattern, source)
    if not match:
        raise ValueError(f"Could not find categorical mapping prompt in {source}")
    mapping = match.group(1)
    if mapping == "{}":
        mapping = None
    source = "[example]" + match.group(2)
    return mapping, source

def process_interim(data_path:pathlib.Path,
                    output_path:pathlib.Path,
                    config:DictConfig,
                    version:Optional[int],
                    override:bool):
    """Pre-processes the data from the interim folder.

    Converts the .tsv files from the interim folder into a .json file in
    processed that is ready to be used for training."""
    logger.info(f"Processing data from {data_path}")
    data = []
    headers = ['categorical_slots_mapping', 'source', 'target', 'dialogue_id', 'turn_id', 'frame_id']
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            source, target, dialogue_id, turn_id, frame_id = row
            slots_mapping, source = extract_categorical_mapping(source)
            elements = [slots_mapping, source, target, dialogue_id, turn_id, frame_id]
            data.append({header: element for header, element in zip(headers, elements)})
    logger.info(f"Loaded {len(data)} examples")
    processed_data = {"data": data}
    save_data(processed_data,
              output_path,
              metadata=config,
              version=version,
              override=override)


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
    "--data_path",
    "data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to .tsv file with SDT data in source: target format for SGD.",
)
@click.option(
    "-r",
    "--raw_data_path",
    "raw_data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to raw data used to generate the .tsv file.",
)
@click.option(
    "-p",
    "--prompt_indices",
    "prompt_indices",
    required=True,
    type=int,
    help="Prompt Indice from sdt_prompts.py used to generate the demonstration prompt for SDT.",
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
    type=int,
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
def main(
    cfg_path: pathlib.Path,
    log_level: int,
    data_path: pathlib.Path,
    raw_data_path: pathlib.Path,
    prompt_indices: int,
    output_path: pathlib.Path,
    override: bool,
    version: Optional[int],
):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = OmegaConf.load(cfg_path)
    config.metadata.date = get_datetime()
    config.metadata.raw_data_path = raw_data_path
    config.metadata.prompt_indices = prompt_indices
    config.metadata.output_path = output_path
    output_path = pathlib.Path(output_path)
    process_interim(data_path, output_path, config, version, override)
    logger.info("Done")


if __name__ == "__main__":
    main()
