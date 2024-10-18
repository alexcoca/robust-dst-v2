"""Collect the turns from the schemas in [our paper citation] to a table.

This is used to ground the the testing prompts into the same turns as
[Coca et al] (but note that when this table is used, for each example the
prompt will be different as we do not control the choice of KST and the
concatenation order).
"""

from __future__ import annotations

import json
import logging
import re
import string
import sys
from collections import defaultdict
from typing import Literal

import click
from datasets import Dataset, load_dataset
from tqdm import tqdm

from robust_dst.utils import default_to_regular, nested_defaultdict

logger = logging.getLogger(__name__)


NameToKST = dict[str, list]
DomainToNameMapping = dict[str, NameToKST]
KSTTable = dict[Literal["slots", "intents"], DomainToNameMapping]


def init_kst_table(schema_paths: list[str]) -> KSTTable:
    kst_table = nested_defaultdict(list, depth=3)

    for schema_path in schema_paths:
        with open(schema_path, "r") as f:
            schema = json.load(f)

        for service in schema:
            service_name = service["service_name"][:-1]
            for slot_info in service["slots"]:
                slot_intent_name = slot_info["name"]
                description = slot_info["description"]
                kst_table["slots"][service_name][slot_intent_name].append(description)

            for intent_info in service["intents"]:
                intent_name = intent_info["name"]
                description = intent_info["description"]
                kst_table["intents"][service_name][intent_name].append(description)

    return default_to_regular(kst_table)


def validate_kst_table(num_variants: int, kst_table: KSTTable) -> None:
    for i in ("slots", "intents"):
        for service, slot_intent_ksts in kst_table[i].items():
            for slot_intent, ksts in slot_intent_ksts.items():
                assert len(ksts) == num_variants, f"{service}:{slot_intent} error"


def generate_sgdx_kst_tables(
    raw_data_root: str, output_path: str, kst_table: KSTTable
) -> None:
    """Generate KST mapping for SGD-X."""

    with open(f"{raw_data_root}/original/test/schema.json", "r") as f:
        original_schema = json.load(f)

    for version in ("v1", "v2", "v3", "v4", "v5"):
        versioned_kst_table = nested_defaultdict(list, depth=3)

        new_schema_path = f"{raw_data_root}/{version}/test/schema.json"
        with open(new_schema_path, "r") as f:
            new_schema = json.load(f)

        mapping = defaultdict(dict)

        for original_service, new_service in zip(original_schema, new_schema):
            service_name = original_service["service_name"]
            for schema_elements in ("slots", "intents"):
                for orig_slot_intent_info, new_slot_intent_info in zip(
                    original_service[schema_elements], new_service[schema_elements]
                ):
                    orig_slot_name = orig_slot_intent_info["name"]
                    new_slot_name = new_slot_intent_info["name"]
                    mapping[service_name][orig_slot_name] = new_slot_name

        n = version[-1]
        for schema_elements in ("slots", "intents"):
            for service, slot_intent_ksts in kst_table[schema_elements].items():
                for slot_intent_name, ksts in slot_intent_ksts.items():
                    versioned_kst_table[schema_elements][f"{service}{n}"][
                        mapping[service][slot_intent_name]
                    ] = list(map(lambda s: s.lower(), ksts))

        versioned_kst_table = default_to_regular(versioned_kst_table)
        with open(f"{output_path}/kst_table_{version}.json", "w") as f:
            json.dump(versioned_kst_table, f, indent=4)


def filter_sgd_seen_kst_table(dataset, kst_table: KSTTable) -> KSTTable:
    filtered_kst_table = nested_defaultdict(list, depth=3)
    seen_slots_intents, sampled_ksts = _examine_dataset(dataset)

    for i in ("slots", "intents"):
        for service, slot_intent_ksts in kst_table[i].items():
            for slot_intent_name, ksts in slot_intent_ksts.items():
                for kst in ksts:
                    slot_intent_is_seen = (
                        service in seen_slots_intents[i]
                        and slot_intent_name in seen_slots_intents[i][service]
                    )
                    if slot_intent_is_seen:
                        corpus_has_ksts = (
                            service in sampled_ksts[i]
                            and slot_intent_name in sampled_ksts[i][service]
                        )
                        if corpus_has_ksts:
                            # if the slot/intent is seen in the training set with valid
                            # ksts to sample from
                            for sampled_kst in sampled_ksts[i][service][
                                slot_intent_name
                            ]:
                                # if the provided description in the provided schema
                                # variants matches at least one of the valid ksts from
                                # the training set

                                if kst.lower() in _extract_sentence_body(
                                    sampled_kst
                                ) or kst.lower() in _extract_question_text(sampled_kst):
                                    filtered_kst_table[i][service][
                                        slot_intent_name
                                    ].append(kst.lower().strip())
                                    break
                    else:
                        filtered_kst_table[i][service][slot_intent_name].append(
                            kst.lower().strip()
                        )

    return filtered_kst_table


def _extract_sentence_body(sentence: str) -> str:
    # Remove any leading or trailing whitespace
    sentence = sentence.strip()

    # Remove any common ending punctuation characters
    if sentence.endswith(tuple(string.punctuation)):
        sentence = sentence[:-1]

    return sentence


def _extract_question_text(utterance: str) -> str:
    """Extract the question only.

    Example of an utterance from the SGD dataset:
        yes, that's right, what is the address of the restaurant?
            -> what is the address of the restaurant
    """
    match = re.search(r"(.*[.,?!]\s)*(.*)[?.]?", utterance)

    if match is None:
        raise RuntimeError(f"Cannot extract question from {utterance}")
    return match.group(2).strip(" .?")


def _examine_dataset(dataset):
    """Get seen slots and intents and KSTs available to be sampled from."""
    seen_slots_intents, sampled_kst_idxs = _build_acts_cache(
        dataset, omit_confirmation_turns=True
    )

    n_slots = 0
    for service, slots in seen_slots_intents["slots"].items():
        n_slots += len(slots)
    logger.debug(f"Number of slots seen: {n_slots}")

    col_idx_to_col_name = {1: "user_utt", 2: "sys_utt"}
    sampled_ksts = nested_defaultdict(set, depth=3)
    for i in ("slots", "intents"):
        for service, slot_intent_kst_idxs in sampled_kst_idxs[i].items():
            for slot_intent_name, kst_idxs in slot_intent_kst_idxs.items():
                for kst_idx, col_idx in kst_idxs:
                    sampled_ksts[i][service][slot_intent_name].add(
                        dataset[kst_idx][col_idx_to_col_name[col_idx]]
                    )

    return seen_slots_intents, default_to_regular(sampled_ksts)


def _cache_speaker_acts(
    acts_cache,
    idx: int,
    act_info: dict,
    domain: str,
    speaker: int,
    omit_confirmation_turns: bool,
) -> None:
    for act, act_params in act_info.items():
        # requesting/informing several slots/intents in a single utterance
        if len(act_params) > 1:
            continue
        act_param = act_params.pop()
        if act == "REQUEST":
            # find brackets in action parameters, which shows the action has
            # target value
            if "(" in act_param:
                if omit_confirmation_turns:
                    continue
                else:
                    # do not skip these confirmations, extract the slot name
                    act_param = re.search(r"(.*?)\(.*\)", act_param).group(1)
            acts_cache["slots"][domain][act_param].append((idx, speaker))
        elif act in ("INFORM_INTENT", "OFFER_INTENT"):
            acts_cache["intents"][domain][act_param].append((idx, speaker))


def _build_acts_cache(dataset: Dataset, omit_confirmation_turns: bool = True):
    seen_slots_intents = nested_defaultdict(set, depth=2)
    acts_cache = nested_defaultdict(list, depth=3)

    for idx, example in tqdm(enumerate(dataset)):
        domain = example["turn_domain"]
        sys_acts = json.loads(example["sys_acts"])
        user_acts = json.loads(example["user_acts"])

        seen_slots_intents["slots"][domain] |= set(
            json.loads(example["slot_mapping"]).values()
        )
        seen_slots_intents["intents"][domain] |= set(
            json.loads(example["intent_mapping"]).values()
        )

        if len(sys_acts) == 1:
            _cache_speaker_acts(
                acts_cache=acts_cache,
                idx=idx,
                act_info=sys_acts,
                domain=domain,
                speaker=2,
                omit_confirmation_turns=omit_confirmation_turns,
            )

        if len(user_acts) == 1:
            _cache_speaker_acts(
                acts_cache=acts_cache,
                idx=idx,
                act_info=user_acts,
                domain=domain,
                speaker=1,
                omit_confirmation_turns=omit_confirmation_turns,
            )

    return default_to_regular(seen_slots_intents), default_to_regular(acts_cache)


@click.command()
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option(
    "-o",
    "--output_path",
    "output_path",
    required=False,
    default="data/external",
    type=click.Path(exists=False),
    help="Directory where the kst tables are output.",
)
@click.option(
    "-r",
    "--raw_data_root",
    "raw_data_root",
    required=False,
    default="data/raw",
    type=click.Path(exists=False),
    help="Directory containing the raw SGD and SGD-X datasets.",
)
@click.option(
    "-d",
    "--processed_train_data_path",
    "processed_train_data_path",
    required=True,
    type=click.Path(exists=False),
    help="Path to the preprocessed SGD train dataset.",
)
@click.option(
    "-s",
    "--schema_variant",
    "schema_variant_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=False),
    help=".",
)
def main(
    log_level,
    processed_train_data_path,
    raw_data_root,
    output_path,
    schema_variant_paths,
):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    kst_table = init_kst_table(schema_variant_paths)
    validate_kst_table(len(schema_variant_paths), kst_table)

    logger.info("Generating KST tables for SGD-X")
    generate_sgdx_kst_tables(raw_data_root, output_path, kst_table)

    logger.info("Generating KST table for SGD")
    dataset = load_dataset(
        "json",
        data_files=processed_train_data_path,
        field="data",
        split="train",
    )
    filtered_kst_table = filter_sgd_seen_kst_table(dataset, kst_table)
    with open(f"{output_path}/kst_table_original.json", "w") as f:
        json.dump(filtered_kst_table, f, indent=4)


if __name__ == "__main__":
    main()
