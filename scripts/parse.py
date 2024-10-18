import json
import logging
import pathlib
import sys
from distutils.dir_util import copy_tree
from pathlib import Path
from typing import Optional, Union

import click
from omegaconf import OmegaConf

from robust_dst.parser import D3STParser, T5DSTParser

logger = logging.getLogger(__name__)


@click.command()
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option(
    "-b",
    "--belief_path",
    "belief_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Absolute path to the directory containing the belief file to be decoded.",
)
@click.option(
    "-o",
    "--output_dir",
    "output_dir",
    required=False,
    type=click.Path(path_type=Path),
    help=(
        "Absolute path to the directory where SGD-formatted dialogues containing"
        " predictionsas opposed to annotations are output. If not passed, the"
        " dialogues are saved in the same directory as the parser input,"
        " (i.e., -b argument)."
    ),
    default=None,
)
@click.option(
    "-s",
    "--schema_path",
    "schema_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Absolute path to the the schema of the data to be parsed.",
)
@click.option(
    "-templates",
    "--template_dir",
    "template_dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Absolute to the directory containing blank dialogue files for the split"
        " parsed."
    ),
)
@click.option(
    "-t",
    "--test_data",
    "test_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Path to pre-processed test data for which model predictions are to be"
        " parsed.Used to retrieve mappings from indices to slot/intent names which are"
        " required torecover slot names from predicted indices"
    ),
)
@click.option(
    "-f",
    "--file",
    "files_to_parse",
    type=str,
    default=None,
    multiple=True,
    help=(
        "Which file to parse. Possible file names are all files listed under dialogue"
        " templates,that is, the filenames are the same as the SGD train/dev/test"
        " dialogue files. For example,passing 'dialogues_001.json' will parse"
        " predictions for the dialogues in the corresponding files.."
    ),
)
def main(
    belief_path: pathlib.Path,
    schema_path: pathlib.Path,
    output_dir: Union[pathlib.Path, None],
    template_dir: pathlib.Path,
    test_path: pathlib.Path,
    log_level: int,
    files_to_parse: Optional[str],
):
    assert (
        test_path.name == "data.json"
    ), "-t/--test_data must be the preprocessed data for the split to be parsed"
    if output_dir is None:
        output_dir = belief_path
    else:
        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True, parents=True)

    with open(belief_path.joinpath("experiment_config.yaml"), "r") as f:
        experiment_config = OmegaConf.load(f)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(
            f'{output_dir.joinpath("parse")}.log',
            mode="w",
        ),
    ]
    logging.basicConfig(
        handlers=handlers,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)
    data_format = experiment_config.data.metadata.data_format
    assert data_format in ["T5DST", "google", "T5DST2", "D3ST"]
    copy_tree(str(template_dir), str(output_dir))

    if data_format == "D3ST":
        parser = D3STParser(
            template_dir=output_dir,
            schema_path=str(schema_path),
            files_to_parse=files_to_parse if files_to_parse else None,
        )
    else:
        parser = T5DSTParser(
            template_dir=output_dir,
            schema_path=str(schema_path),
            data_format=data_format,
            files_to_parse=files_to_parse if files_to_parse else None,
        )

    with open(belief_path.joinpath("generated_predictions.txt"), "r") as f:
        predicted_values = [l.rstrip("\n") for l in f.readlines()]  # noqa
    with open(test_path, "r") as f:
        preprocessed_refs = json.load(f)["data"]
    logger.info(f"Parsing {belief_path} directory.")
    logger.info(f"Outputs will be found in {output_dir} directory.")
    assert len(predicted_values) == len(
        preprocessed_refs
    ), "For some references, no prediction was made"

    parser.convert_to_sgd_format(
        preprocessed_refs=preprocessed_refs,
        predictions=predicted_values,
        write_to_disk=True,
    )


if __name__ == "__main__":
    main()
