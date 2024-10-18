import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import click
from datasets import load_dataset
from omegaconf import OmegaConf

from robust_dst.preprocessor import D3STPreprocessor
from robust_dst.utils import set_seed

logger = logging.getLogger(__name__)


@dataclass
class CudnnConfig:
    enabled: bool = True
    deterministic: bool = False
    benchmark: bool = True


@dataclass
class ReproduceConfig:
    seed: int
    cudnn: CudnnConfig


@click.command()
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option(
    "-s",
    "--seed",
    "seed",
    required=False,
    type=int,
    default=None,
    help=("Set the random seed for reproducibility"),
)
@click.option(
    "-t",
    "--kst_table",
    "kst_table_path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=("The path to a user defined KST table to sample from if desired"),
)
@click.option(
    "-d",
    "--data",
    "data_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help=("Path to the directory containing preprocessed data"),
)
@click.option(
    "-o",
    "--out",
    "output_path",
    required=True,
    type=click.Path(exists=False, path_type=Path),
    help="Directory where processed data is output.",
)
@click.option(
    "--max_source_length",
    required=False,
    type=int,
    default=1024,
    help=(
        "The maximum total input sequence length after tokenization. Sequences"
        " longer than this will be truncated, sequences shorter will be padded."
    ),
)
@click.option(
    "--max_target_length",
    required=False,
    type=int,
    default=512,
    help=(
        "The maximum total sequence length for target text after tokenization."
        " Sequences longer than this will be truncated, sequences shorter will"
        " be padded."
    ),
)
@click.option(
    "--augment_style",
    required=False,
    type=str,
    default="TURN",
    help=(
        "How the sampled knowledge-seeking turns are added to the prompts,"
        " can be NONE, REPLACE, DA, TURN, TURNSLOT."
        " * NONE: Do not sample or add KSTs."
        " * REPLACE: Use sampled KSTs instead of schema descriptions, fallback"
        " if no KST available."
        " * DA: Data augmentation, where the dataset is augmented with prompts"
        " containing only the KSTs, forming a dataset twice as large."
        " * TURN: For each turn, the KSTs are concatenated to schema"
        " descriptions in random order, forming a dataset of the same size."
        " * TURNSLOT: For each turn, the KSTs and slot names are concatenated"
        " to the schema descriptions in random order, forming a dataset of the"
        " same size."
    ),
)
@click.option(
    "--omit_confirmation_turns",
    is_flag=True,
    default=True,
    help=(
        "Whether to skip sampling KSTs that are confirmations. These are turns"
        " annotated with REQUEST(slot=value) (e.g. 'Did you book for 2 people?'"
        " may be annotated with REQUEST(travellers=2)). In contrast knowledge"
        " seeking turns are annotated REQUEST(slot) (e.g. 'What time did you"
        " want to travel?' may be annotated with REQUEST(journey_start))"
    ),
)
@click.option(
    "--discard_truncated_examples",
    is_flag=True,
    default=False,
    help=(
        "Whether to discard examples with inputs exceeding the max_length when"
        " tokenized."
    ),
)
def main(
    log_level,
    seed,
    kst_table_path,
    data_path,
    output_path,
    augment_style,
    max_source_length,
    max_target_length,
    omit_confirmation_turns,
    discard_truncated_examples,
):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if seed is not None:
        logger.info(f"Setting random seed to {seed}")
        set_seed(ReproduceConfig(seed, CudnnConfig()))

    preprocessing_config = OmegaConf.load(
        data_path.parent / "preprocessing_config.yaml"
    )
    data_format = preprocessing_config.metadata.data_format.lower()

    preprocessor_init_kwargs = {
        "max_source_length": max_source_length,
        "max_target_length": max_target_length,
        "load_from_cache_file": False,
        "tokenize": False,
    }
    if "d3st" in data_format:
        delimiter = preprocessing_config.preprocessing.delimiter

        if "multiwoz" in data_format:
            desc_format = preprocessing_config.preprocessing.description_type
            domain_in_desc = desc_format == "full_desc_with_domain"
        else:
            domain_in_desc = False

        preprocessor = D3STPreprocessor(
            delimiter=delimiter,
            domain_in_desc=domain_in_desc,
            **preprocessor_init_kwargs,
        )
    else:
        raise NotImplementedError("This script only supports D3ST format.")

    kst_table = None
    if kst_table_path is not None:
        logger.info(f"Using KST table at {kst_table_path}")
        with kst_table_path.open() as f:
            kst_table = json.load(f)

    logger.info("Processing dataset")
    processed_dataset = preprocessor.process(
        load_dataset(
            "json",
            data_files=str(data_path),
            field="data",
            split="train",
        ),
        augment_style=augment_style,
        kst_table=kst_table,
        omit_confirmation_turns=omit_confirmation_turns,
        discard_truncated_examples=discard_truncated_examples,
    )

    processed_dataset.to_json(str(output_path / "data.json"))
    with (output_path / "data.json").open() as f:
        examples = f.readlines()
    nested_data = {"data": []}
    for example in examples:
        nested_data["data"].append(json.loads(example))
    with (output_path / "data.json").open(mode="w") as f:
        json.dump(nested_data, f, indent=4)

    OmegaConf.save(
        config=preprocessing_config, f=output_path / "preprocessing_config.yaml"
    )


if __name__ == "__main__":
    main()
