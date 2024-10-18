import json
import logging
import random
import re
from collections import defaultdict
from datetime import datetime
from functools import partial
from itertools import repeat
from operator import methodcaller
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

_EXPECTED_SCHEMA_VARIANTS = ["v1", "v2", "v3", "v4", "v5"]

logger = logging.getLogger(__name__)


def get_datetime() -> str:
    """Returns the current date and time."""
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


def set_seed(args):
    # For reproduction
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.cudnn.enabled
    torch.backends.cudnn.enabled = args.cudnn.deterministic
    torch.backends.cudnn.benchmark = args.cudnn.benchmark


def infer_schema_variant_from_path(path: str) -> str:
    """Extracts the schema version from the data path."""
    match = re.search(r"\bv[1-9]\b", path)  # noqa
    if match is not None:
        schema_version = path[match.start() : match.end()]
        assert schema_version in _EXPECTED_SCHEMA_VARIANTS
    else:
        schema_version = "original"
    return schema_version


def infer_data_version_from_path(path: str) -> str:
    match = re.search(r"\bversion_\d+\b", path)  # noqa
    if match is not None:
        version = path[match.start() : match.end()]
    else:
        logger.warning(f"Could not detect data version in path {path}")
        version = ""
    return version


def nested_defaultdict(default_factory: Callable, depth: int = 1):
    """Creates a nested default dictionary of arbitrary depth with a specified callable
    as leaf.
    """
    if not depth:
        return default_factory()
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()


def save_data(
    data: Union[dict, list],
    path: Union[Path, str],
    fname: str = "data.json",
    metadata: Optional[DictConfig] = None,
    version: Optional[int] = None,
    override: bool = False,
) -> int:
    """Saves data along with the configuration that created it.

    Args:
        override:
        version:
    """
    path = Path(path)
    if version is None:
        if path.exists():
            existing_version = sorted(
                [
                    int(p.name.split("_")[1])
                    for p in path.iterdir()
                    if "version" in str(p)
                ]
            )  # type: list[int]
            if existing_version:
                version = existing_version[-1] + 1
            else:
                version = 1
        else:
            version = 1
    path = path.joinpath(f"version_{version}")
    if path.joinpath(fname).exists():
        if not override:
            logger.warning(
                f"Cannot override content of {path.joinpath(fname)}, existing data will"
                " not be overwritten. Use --override flag to achieve this behaviour."
            )
            return
    path.mkdir(parents=True, exist_ok=True)
    if metadata:
        logger.info(
            "Saving data processing info at path"
            f" {path.joinpath('preprocessing_config.yaml')}"
        )
        metadata.metadata.version = version
        OmegaConf.save(config=metadata, f=path.joinpath("preprocessing_config.yaml"))
    logger.info(f"Saving data at path {path.joinpath(fname)}")
    with open(path.joinpath(fname), "w") as f:
        json.dump(data, f, indent=4)
    return version


def default_to_regular(d: defaultdict) -> dict:
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def append_to_values(result: dict, new_data: dict):
    """Recursively appends to the values of `result` the values in
    `new_data` that have the same keys. If the keys in `new_data`
    do not exist in `result`, they are recursively added. The keys of
    `new_data` can be either lists or single float objects that
    are to be appended to existing `result` keys. List concatenation is
    performed in former case.

    Parameters
    ----------
    result
        Mapping whose values are to be extended with corresponding values from
        `new_data_map`
    new_data
        Data with which the values of `result_map` are extended.
    """

    def dict_factory():
        return defaultdict(list)

    for key in new_data:
        # recursively add any new keys to the result mapping
        if key not in result:
            if isinstance(new_data[key], dict):
                result[key] = dict_factory()
                append_to_values(result[key], new_data[key])
            else:
                if isinstance(new_data[key], float):
                    result[key] = [new_data[key]]
                elif isinstance(new_data[key], list):
                    result[key] = [*new_data[key]]
                else:
                    raise ValueError("Unexpected key type.")
        # updated existing values with the value present in `new_data_map`
        else:
            if isinstance(result[key], dict):
                append_to_values(result[key], new_data[key])
            else:
                if isinstance(new_data[key], list):
                    result[key] += new_data[key]
                elif isinstance(new_data[key], float):
                    result[key].append(new_data[key])
                else:
                    raise ValueError("Unexpected key type")


def aggregate_values(
    mapping: dict, agg_fcn: Literal["mean", "prod"], reduce: bool = True
):
    """Aggregates the values of the input (nested) mapping according to the
    specified aggregation method. This function modifies the input in place.

    Parameters
    ---------
    mapping
        The mapping to be aggregated.
    agg_fcn
        Aggregation function. Only  `mean` or `prod` aggregation supported.
    reduce
        If False, the aggregator will keep the first dimension of the value to be
        aggregated.

    Example
    -------
    >>> mapping = {'a': {'b': [[1, 2], [3, 4]]}}
    >>> agg_fcn = 'mean'
    >>> aggregate_values(mapping, agg_fcn, reduce=False)
    >>> {'a': {'b': [1.5, 3.5]}}

    """

    for key, value in mapping.items():
        if isinstance(value, dict):
            aggregate_values(mapping[key], agg_fcn, reduce=reduce)
        else:
            if reduce:
                aggregator = methodcaller(agg_fcn, value)
                mapping[key] = aggregator(np)
            else:
                if isinstance(mapping[key], list) and isinstance(mapping[key][0], list):
                    agg_res = []
                    for val in mapping[key]:
                        aggregator = methodcaller(agg_fcn, val)
                        agg_res.append(aggregator(np))
                    mapping[key] = agg_res
                else:
                    aggregator = methodcaller(agg_fcn, value)
                    mapping[key] = aggregator(np)


def load_json(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    return data
