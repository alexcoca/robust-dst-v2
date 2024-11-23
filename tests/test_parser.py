import logging
import pathlib
from itertools import chain
from typing import List, Tuple

import pytest
from datasets import Dataset, load_dataset
from numpy.testing import assert_almost_equal
from omegaconf import DictConfig, OmegaConf

from robust_dst.evaluation import get_metrics
from robust_dst.parser import D3STParser
from robust_dst.scoring_utils import setup_sgd_evaluator_inputs

logger = logging.getLogger(__name__)

PROCESSED_DATASET_ROOT_PATH = pathlib.Path("data/processed/")
RAW_DATASET_PATH_ROOT = pathlib.Path("data/raw")
DIALOGUE_TEMPLATES_ROOT_PATH = pathlib.Path("data/interim/blank_dialogue_templates/")
SGD_VARIANTS = ["original", "v1", "v2", "v3", "v4", "v5"]
SPLITS = ["train", "dev", "test"]


def load_dataset_and_config(path: str) -> Tuple[Dataset, list, DictConfig]:
    dataset = load_dataset(
        "json", data_files=path, field="data", split="train", cache_dir="cache"
    )
    # keep a copy of the original data structure for parsing
    preprocessed_ref = dataset.data.table.to_pylist()

    preprocessing_config = OmegaConf.load(
        pathlib.Path(path).parent.joinpath("preprocessing_config.yaml")
    )["preprocessing"]

    return dataset, preprocessed_ref, preprocessing_config


def get_fake_predictions(dataset: Dataset, target_column: str = "state") -> List[str]:
    return dataset.map(
        lambda examples: {"prediction": examples[target_column]},
        batched=True,
    )["prediction"]


@pytest.mark.parametrize("variant", SGD_VARIANTS, ids="variant={}".format)
@pytest.mark.parametrize("split", SPLITS, ids="split={}".format)
@pytest.mark.parametrize(
    "data_version",
    ["version_8"],
    ids="data_version={}".format,
)
def test_d3st_parser(
    split: str,
    variant: str,
    data_version: str,
):
    """This test requires preprocessing.cumulate_slots to be set to False"""
    dataset, preprocessed_ref, preprocessing_config = load_dataset_and_config(
        str(PROCESSED_DATASET_ROOT_PATH / variant / split / data_version / "data.json")
    )

    parser = D3STParser(
        template_dir=DIALOGUE_TEMPLATES_ROOT_PATH / variant / split,
        schema_path=RAW_DATASET_PATH_ROOT / variant / split / "schema.json",
        value_separator=preprocessing_config["value_separator"],
        target_slot_index_separator=preprocessing_config["delimiter"],
        restore_categorical_case=preprocessing_config["lowercase"],
    )

    predictions = get_fake_predictions(dataset)

    file_to_hyp_dials = parser.convert_to_sgd_format(
        preprocessed_refs=preprocessed_ref,
        predictions=predictions,
    )
    assert len(predictions) == len(
        preprocessed_ref
    ), f"Expected {len(preprocessed_ref)} predictions but got {len(predictions)}"
    dataset_hyp = {
        dial["dialogue_id"]: dial for dial in chain(*file_to_hyp_dials.values())
    }

    evaluator_inputs = setup_sgd_evaluator_inputs(
        ref_dir=RAW_DATASET_PATH_ROOT / variant / split
    )
    all_metrics_aggregate, _ = get_metrics(
        evaluator_inputs["dataset_ref"],
        dataset_hyp,
        evaluator_inputs["eval_services"],
        evaluator_inputs["in_domain_services"],
    )
    # assert_almost_equal(
    #     all_metrics_aggregate["#ALL_SERVICES"]["active_intent_accuracy"], 1.0
    # )
    # assert_almost_equal(
    #     all_metrics_aggregate["#ALL_SERVICES"]["requested_slots_f1"], 1.0
    # )
    assert_almost_equal(
        all_metrics_aggregate["#ALL_SERVICES"]["joint_goal_accuracy"], 1.0
    )
