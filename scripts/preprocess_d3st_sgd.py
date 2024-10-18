# Based on code released by Google Research, the original copyright notice
# is included below:
# Copyright 2021 Google Research.
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

from __future__ import annotations

import copy
import dataclasses
import json
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
from typing import Any, Optional

import click
from omegaconf import DictConfig, OmegaConf

from robust_dst.utils import (
    get_datetime,
    infer_schema_variant_from_path,
    save_data,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TurnInfo:
    """Information extracted from dialog turns.

    This is used to keep state across turns and frames in a dialogue.
    """

    out_ctx_str: str = ""
    out_ctx_with_desc_str: str = ""
    out_state_str: str = ""
    out_act_str: str = ""
    prev_state_str: str = ""
    delta_state_str: str = ""
    history_states: str = ""
    out_intent_str: str = ""
    new_states: str = ""
    new_intents: str = ""
    curr_utt: str = ""
    user_turn: bool = False
    turn_domain: str = ""
    dialogue_id: str = ""
    turn_id: str = ""
    frame_id: str = ""

    # see PrecessedTurn class for descriptions of these attributes
    desc_mapping: dict = dataclasses.field(default_factory=dict)
    slot_mapping: dict = dataclasses.field(default_factory=dict)
    cat_values_mapping: defaultdict[dict] = dataclasses.field(
        default_factory=lambda: defaultdict(dict)
    )
    intent_mapping: dict = dataclasses.field(default_factory=dict)
    file_name: str = ""
    sys_utt: str = ""
    sys_acts: defaultdict[list] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
    )
    """mapping from action to list of action parameters of the previous system turn
    - For actions on slots, the action maps to a list of slot names
        E.g. `{"REQUEST": ["restaurant_name", "location"]}`
    - For actions on slots with values, the values are also included with the slot
      names, inside brackets.
        E.g. `{"INFORM": ["location(San Jose)", "restaurant_name(Sino)"]}`
    - For actions on intents, the action maps to a list of intent names
        E.g. `{"OFFER_INTENT": ["ReserveCar"]}`
    """
    user_utt: str = ""
    user_acts: defaultdict[list] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
    )
    """mapping from action to list of action parameters of the current user turn
    - For actions on slots, the action maps to a list of slot names
        E.g. `{"REQUEST": ["restaurant_name", "location"]}`
    - For actions on slots with values, the values are also included with the slot
      names, inside brackets.
        E.g. `{"INFORM": ["location(San Jose)", "restaurant_name(Sino)"]}`
    - For actions on intents, the action maps to a list of intent names
        E.g. `{"OFFER_INTENT": ["ReserveCar"]}`
    """


@dataclasses.dataclass
class ProcessedTurn:
    """Processed data is output as JSON Lines format, where each row is a dict whose
    keys are attributes of this class.
    """

    # mapping from slot index to slot description, e.g. "{\"0\": \"name of the event\"}"
    description_mapping: str
    # string of the target state, [states] ... [intents] ... [req_slots]
    state: str
    # dialogue context, [user] ... [system] ...
    dialogue_context: str
    # mapping from slot index to slot name
    slot_mapping: str
    # mapping [slot_name][cat_value][cat_slot_index]
    cat_values_mapping: str
    # mapping from intent index to intent name
    intent_mapping: str
    turn_domain: str
    turn_idx: str
    dialogue_id: str
    file_name: str
    # sytem utterance of the previous turn
    sys_utt: str
    # mapping from act to list of slots of the previous turn
    # e.g. "{\"REQUEST\": [\"restaurant_name\", \"location\"]}"
    sys_acts: str
    # user utterance of the current turn
    user_utt: str
    # mapping from act to list of slots of the current turn, same form as prev_sys_acts
    user_acts: str

    def from_turn_info(turn_info: TurnInfo, target: str) -> ProcessedTurn:
        return ProcessedTurn(
            description_mapping=json.dumps(turn_info.desc_mapping),
            state=target,
            dialogue_context=turn_info.out_ctx_str,
            slot_mapping=json.dumps(turn_info.slot_mapping),
            cat_values_mapping=json.dumps(turn_info.cat_values_mapping),
            intent_mapping=json.dumps(turn_info.intent_mapping),
            turn_domain=turn_info.turn_domain,
            turn_idx=turn_info.turn_id,
            dialogue_id=turn_info.dialogue_id,
            file_name=turn_info.file_name,
            sys_utt=turn_info.sys_utt,
            sys_acts=json.dumps(turn_info.sys_acts),
            user_utt=turn_info.user_utt,
            user_acts=json.dumps(turn_info.user_acts),
        )


def _merge_domain_slot(domain: str, slot_name: str):
    return f"{domain}-{slot_name}"


SchemaInfo = dict[str, dict]


def load_schema(
    preproc_config: DictConfig, schemas: list[dict]
) -> tuple[OrderedDict, SchemaInfo]:
    """Loads schema items and descriptions.

    Args:
        preproc_config: Preprocessing configuration.
        shemas: List of service schemas.

    Returns:
        A tuple, including an ordered dictionary whose keys are slot names and
        values are placeholder values, and a dictionary whose keys are slot names
        and values are descriptions.
    """
    # We need to preserve state orders since we hope the model learns to generate
    # states in consistent order.
    # TODO(yuancao): We might need to do this for intents/actions as well (in case
    # multiple intents/actions turn out to be a problem).
    # TODO(jeffreyzhao): Clean up how we store schema information by using a
    # dataclass.
    slots = OrderedDict()
    item_desc = {
        "slots": {},
        "intents": {},
        "is_categorical": {},
        "possible_values": {},
        "slots_rand_name": {},
        "intents_rand_name": {},
    }
    for schema in schemas:
        domain = schema["service_name"]
        slots.update(
            {_merge_domain_slot(domain, slot["name"]): "" for slot in schema["slots"]}
        )
        item_desc["slots"].update(
            {
                _merge_domain_slot(domain, slot["name"]): slot["description"]
                for slot in schema["slots"]
            }
        )

        for slot in schema["slots"]:
            name = _merge_domain_slot(domain, slot["name"])
            is_cat = slot["is_categorical"]
            poss_vals = slot["possible_values"]

            # If this is a categorical slot but the possible value are all numeric,
            # consider this as a noncat slot if all_numeric_noncat is True.
            if is_cat and (
                preproc_config.all_numeric_noncat
                and all([v.isdigit() for v in poss_vals])
            ):
                poss_vals = []
                is_cat = False

            item_desc["is_categorical"][name] = is_cat
            item_desc["possible_values"][name] = poss_vals

        item_desc["intents"].update(
            {
                _merge_domain_slot(domain, intent["name"]): intent["description"]
                for intent in schema["intents"]
            }
        )

        if preproc_config.data_format == "rand_name":
            item_desc["slots_rand_name"].update(
                {
                    _merge_domain_slot(domain, slot["name"]): "".join(
                        random.sample(list(slot["name"]), len(slot["name"]))
                    )
                    for slot in schema["slots"]
                }
            )
            # pylint: disable=g-complex-comprehension
            item_desc["intents_rand_name"].update(
                {
                    _merge_domain_slot(domain, intent["name"]): "".join(
                        random.sample(list(intent["name"]), len(intent["name"]))
                    )
                    for intent in schema["intents"]
                }
            )
            # pylint: enable=g-complex-comprehension
    return slots, item_desc


def _process_user_turn(
    preproc_config: DictConfig,
    state: dict[str, Any],
    turn_info: TurnInfo,
    cumu_slots: OrderedDict,
    domain: str,
    item_desc: SchemaInfo,
    state_dict: dict[str, list[str]],
) -> dict[str, int]:
    """Updates turn_info and cumu_slots based on user turn input.

    Args:
        preproc_config: Preporcessing configuration.
        state: A dictionary containing state info.
        turn_info: A TurnInfo object accmulating essential info from each turn.
        cumu_slots: An OrderedDict containing cmumulative slot information.
        domain: A string, domain (service) of the turn.
        item_desc: A dictionary of items and their descriptions.
        state_dict: A dictionary of states from the current turn.

    Returns:
        A dictionary that maps slot descriptions to ids.
    """
    turn_info.desc_mapping = {}
    turn_info.slot_mapping = {}
    turn_info.cat_values_mapping = defaultdict(dict)
    turn_info.intent_mapping = {}

    slot_values = state["slot_values"]
    domain_slot_values = {}
    for slot, value in slot_values.items():
        domain_slot_values[_merge_domain_slot(domain, slot)] = value
    slot_values = domain_slot_values

    # Order of slots is preserved. Meanwhile new values of the same
    # slots will overwrite existing ones.
    for slot, value in slot_values.items():
        if slot not in cumu_slots:
            raise ValueError(f"Unknown slot: {slot}.")
        cumu_slots.update({slot: preproc_config.value_separator.join(value)})

    # Clean up.
    desc_to_slot_id = {}
    slots = list(item_desc["slots"].keys())
    if preproc_config.randomize_items:
        random.shuffle(slots)
    # In multi-domain turn case, desc_prefix already contains desc from the
    # previous domain.
    slot_id = len(state_dict["slot_desc"])
    for slot in slots:
        if preproc_config.data_format == "full_desc":
            desc = item_desc["slots"][slot]
        elif preproc_config.data_format == "item_name":
            desc = slot
        elif preproc_config.data_format == "rand_name":
            desc = item_desc["slots_rand_name"][slot]

        # If we are generating with multiple choice, append this prompt.
        if (
            preproc_config.multiple_choice != "none"
            and item_desc["is_categorical"][slot]
        ):
            possible_values = item_desc["possible_values"][slot]
            if preproc_config.randomize_items:
                random.shuffle(possible_values)
            assert len(possible_values) < len(string.ascii_lowercase)
            letters = list(string.ascii_lowercase)

            possible_values_pieces = []
            for letter, value in zip(letters, possible_values):
                if preproc_config.multiple_choice == "1a":
                    possible_values_pieces.append(f"{slot_id}{letter}) {value}")
                    if domain in slot.split("-")[0]:
                        turn_info.cat_values_mapping[slot.split("-")[1]][
                            str(value)
                        ] = f"{slot_id}{letter}"

                elif preproc_config.multiple_choice == "a":
                    possible_values_pieces.append(f"{letter}) {value}")
                    if domain in slot.split("-")[0]:
                        turn_info.cat_values_mapping[slot.split("-")[1]][
                            str(value)
                        ] = letter

            desc += " " + " ".join(possible_values_pieces)

        # Only consider slots in the utterance domain.
        if domain in slot.split("-")[0]:
            # Description prefix to be included in each turn.
            t = f"{slot_id}{preproc_config.delimiter}"
            desc_to_slot_id[slot] = slot_id
            state_dict["slot_desc"].append(
                t + desc.lower() if preproc_config.lowercase else t + desc
            )

            turn_info.slot_mapping[str(slot_id)] = slot.split("-")[1]
            turn_info.desc_mapping[str(slot_id)] = (
                desc.lower() if preproc_config.lowercase else desc
            )

            state_str = ""
            # Corresponding values for active slots.
            if cumu_slots[slot]:
                value = cumu_slots[slot]
                if (
                    preproc_config.multiple_choice != "none"
                    and item_desc["is_categorical"][slot]
                    and value != "dontcare"
                ):
                    # Convert to multiple choice for categorical slots.
                    assert value in possible_values
                    state_str = t + str(slot_id) + letters[possible_values.index(value)]
                else:
                    state_str = t + value

            if state_str:
                if turn_info.out_state_str != "[states] ":
                    turn_info.out_state_str += " "
                turn_info.out_state_str += (
                    state_str.lower() if preproc_config.lowercase else state_str
                )
            turn_info.turn_domain = domain
            slot_id += 1

    # Handle intents.
    # In multi-domain turn case, intent list already contains intents from the
    # previous domain.
    intents = list(item_desc["intents"].keys())
    if preproc_config.randomize_items:
        random.shuffle(intents)
    intent_id = len(state_dict["intent_desc"])
    for intent in intents:
        if preproc_config.data_format == "full_desc":
            desc = item_desc["intents"][intent]
        if preproc_config.data_format == "item_name":
            desc = intent
        elif preproc_config.data_format == "rand_name":
            desc = item_desc["intents_rand_name"][intent]

        # Only consider slots in the utterance domain.
        if domain in intent:
            active_intent = domain + "-" + state["active_intent"]
            # Description prefix to be included in each turn.
            t = f"i{intent_id}{preproc_config.delimiter}"
            intent_str = ""
            if active_intent == intent:
                intent_str = t[:-1]

            state_dict["intent_desc"].append(
                t + desc.lower() if preproc_config.lowercase else t + desc
            )

            turn_info.intent_mapping[f"i{intent_id}"] = intent.split("-")[1]
            turn_info.desc_mapping[f"i{intent_id}"] = (
                desc.lower() if preproc_config.lowercase else desc
            )

            if intent_str:
                state_dict["intent_ids"].append(intent_str)

            intent_id += 1

    # Handle requested slots.
    for req_slot in state["requested_slots"]:
        slot_name = domain + "-" + req_slot
        assert slot_name in desc_to_slot_id, "Requested slots must be in the slot list!"
        req_slot_id = desc_to_slot_id[slot_name]
        # Note the order of requested slots is totally determined by the user's
        # utterance, and is not guaranteed to be sorted.
        state_dict["req_slots"].append(str(req_slot_id))

    return desc_to_slot_id


def _process_agent_turn(
    preproc_config: DictConfig,
    actions: list[dict[str, Any]],
    turn_info: TurnInfo,
    domain: str,
    desc_to_slot_id: dict[str, int],
) -> None:
    """Updates turn_info based on the system actions.

    Args:
        preproc_config: Preporcessing configuration.
        actions: A list of strings for system actions.
        turn_info: A Turninfo object accmulating essential info from each turn.
        domain: A string, domain (service) of the current turn.
        desc_to_slot_id: A dictionary that maps descriptions to slot ids.
    """
    turn_info.prev_state_str = turn_info.out_state_str
    turn_info.out_act_str += " [actions] "
    acts = {}
    for action in actions:
        act = action["act"]
        slot = action["slot"]
        # Note that we don't include api function values but only names, as these
        # values are supposed to be delexicalized and retrieved from db.
        # values = action['values']
        if act not in acts:
            acts[act] = ""
        if slot:
            act_slot = _merge_domain_slot(domain, slot)
            if act_slot in desc_to_slot_id:
                slot_id = desc_to_slot_id[act_slot]
                acts[act] += str(slot_id) + ";"
        else:
            acts[act] += "none;"

    turn_info.out_act_str += " ".join(
        [f"{action}({params})" for action, params in acts.items()]
    )
    if preproc_config.lowercase:
        turn_info.out_act_str = turn_info.out_act_str.lower()


def _get_acts(
    preproc_config: DictConfig, actions: list[dict[str, Any]]
) -> defaultdict[str, list[str]]:
    """Get a mapping of actions to action parameters.

    - For actions on slots, the action maps to a list of slot names
        E.g. {"REQUEST": ["restaurant_name", "location"]}
    - For actions on slots with values, the values are also included with the slot
    names, inside brackets.
        E.g. {"INFORM": ["location(San Jose)", "restaurant_name(Sino)"]}
    - For actions on intents, the action maps to a list of intent names
        E.g. {"OFFER_INTENT": ["ReserveCar"]}

    Args:
        preproc_config: Preporcessing configuration.
        actions: A list of action dictionaries following SGD format

    Returns:
        dict[str, list[str]]: Mapping from action to list of action parameters.
    """
    acts = defaultdict(list)
    for action in actions:
        act = action["act"]
        slot = action["slot"]
        values = action["values"]
        if slot == "intent":
            for value in values:
                acts[act].append(value)
        elif len(values) > 0:
            acts[act].append(f"{slot}({preproc_config.value_separator.join(values)})")
        else:
            acts[act].append(slot)
    return acts


def process_turn(
    preproc_config: DictConfig,
    turn: dict[str, Any],
    turn_info: TurnInfo,
    cumu_slots: OrderedDict,
    item_desc: SchemaInfo,
    prefix: str,
    turn_id: int,
) -> tuple[str, list[TurnInfo]]:
    """Collects information from a single turn.

    Args:
        preproc_config: Preporcessing configuration.
        turn: A dictionary containing original turn structure.
        turn_info: A dictionary accmulating essential info from each turn.
        cumu_slots: An OrderedDict containing cumumulative slot information.
        item_desc: A dictionary of scheam items and their descriptions.
        prefix: A string of the schema item description prefix.
        turn_id: Integer index of turn in dialogue.

    Returns:
        Prefix string (item descriptions) from the current turn and per-frame
        TurnInfo objects.
    """
    speaker = turn["speaker"].lower()
    user_turn = speaker == "user"
    turn_info.user_turn = user_turn
    utt = turn["utterance"]
    turn_info.curr_utt = f"[{speaker}] {utt} "
    turn_info.out_ctx_str += f"[{speaker}] {utt} "
    turn_info.turn_id = str(turn_id)
    if preproc_config.lowercase:
        turn_info.curr_utt = turn_info.curr_utt.lower()
        turn_info.out_ctx_str = turn_info.out_ctx_str.lower()
    # Intent and act strings are not accumulative.
    turn_info.out_act_str = ""
    if user_turn:
        turn_info.out_state_str = "[states] "
        turn_info.out_intent_str = "[intents] "

    desc_to_slot_id = {}
    turn_info_per_frame = []
    for frame_id, frames in enumerate(turn["frames"]):
        domain = frames["service"]
        turn_info.frame_id = str(frame_id)
        state_dict = {
            "slot_desc": [],
            "intent_desc": [],
            "intent_ids": [],
            "req_slots": [],
        }

        if user_turn:
            # Multi-service turns are possible, each frame corresponds to one
            # service (domain).

            # Note: frames['slots'] is not used for generation.
            turn_info.out_state_str = "[states] "
            turn_info.out_intent_str = "[intents] "
            desc_to_slot_id = _process_user_turn(
                preproc_config,
                frames["state"],
                turn_info,
                cumu_slots,
                domain,
                item_desc,
                state_dict,
            )
            turn_info.out_intent_str += " ".join(state_dict["intent_ids"])
            turn_info.out_intent_str += " [req_slots] "
            turn_info.out_intent_str += " ".join(state_dict["req_slots"])

            turn_info.user_utt = utt
            if preproc_config.lowercase:
                turn_info.user_utt = turn_info.user_utt.lower()
            turn_info.user_acts = _get_acts(preproc_config, frames["actions"])

        else:
            _process_agent_turn(
                preproc_config, frames["actions"], turn_info, domain, desc_to_slot_id
            )
            if turn_info.out_intent_str == "":
                turn_info.out_intent_str = "[intents]  [req_slots] "

            turn_info.sys_utt = utt
            if preproc_config.lowercase:
                turn_info.sys_utt = turn_info.sys_utt.lower()
            turn_info.sys_acts = _get_acts(preproc_config, frames["actions"])

        # Add item description prefixes and states to outputs (coming from user
        # turns).
        user_turn_prefix = " ".join(state_dict["slot_desc"] + state_dict["intent_desc"])
        if user_turn:
            turn_info.out_ctx_with_desc_str = (
                user_turn_prefix + " " + turn_info.out_ctx_str
            )
        else:
            # Prefix from the previous user turn.
            turn_info.out_ctx_with_desc_str = prefix + " " + turn_info.out_ctx_str

        turn_info_per_frame.append(copy.deepcopy(turn_info))

    return user_turn_prefix, turn_info_per_frame


def preprocess_file(
    preproc_config: DictConfig,
    ordered_slots: OrderedDict,
    item_desc: SchemaInfo,
    file_name: str,
    raw_dialogues: list[dict],
) -> list[TurnInfo]:
    """Convert an SGD dialogue file to TurnInfo objects, which hold the state at each
    frame of each dialogue turn.

    Args:
        preproc_config: Preporcessing configuration.
        ordered_slots: An ordered dictionary containing slot names.
        item_desc: A dictionary containing items and thier descriptions.
        file_name: Name of the SGD dialogue file.
        raw_dialogues: List of dictionaries containing raw dialogues.

    Returns:
        A list of TurnInfo objects.
    """
    turn_infos = []
    for dlg in raw_dialogues:
        # cumulative states throughout this dialog.
        cumu_slots = copy.deepcopy(ordered_slots)
        turn_info = TurnInfo()
        turn_info.dialogue_id = dlg["dialogue_id"]
        turn_info.file_name = file_name
        prefix = ""
        for turn_idx, turn in enumerate(dlg["turns"]):
            _, per_frame_turn_info = process_turn(
                preproc_config,
                turn,
                turn_info,
                cumu_slots,
                item_desc,
                prefix,
                turn_idx,
            )
            turn_infos.extend(per_frame_turn_info)

            if not preproc_config.cumulate_slots:
                cumu_slots = copy.deepcopy(ordered_slots)

    return turn_infos


def serialize(preproc_config: DictConfig, turn_infos: list[TurnInfo]) -> list[dict]:
    """Serialize internal TurnInfo objects into rows of the output JSON Lines file.

    Args:
        preproc_config (DictConfig): Preprocessing configuration.
        turn_infos (List[TurnInfo]): List of TurnInfo objects.

    Returns:
        List of dictionaries, each corresponding to one row of the output.
    """
    processed_turns = []

    for turn_info in turn_infos:
        # Write samples to file. Each example is divided into two parts
        # separated by \t, the first part being inputs to the model, and the
        # second part are labels for prediction.
        tgt = ""
        if preproc_config.level == "dst":
            if turn_info.user_turn:
                # Only output at user turns if dst
                tgt = turn_info.out_state_str
        elif preproc_config.level == "dst_intent":
            if turn_info.user_turn:
                # Only output at user turns if dst_intent
                tgt = " ".join([turn_info.out_state_str, turn_info.out_intent_str])
        elif preproc_config.level == "dst_intent_act":
            if not turn_info.user_turn:
                # Only output at system turns, including:
                # state + action + responses
                turn_info.curr_utt = turn_info.curr_utt.replace(
                    "[system]", "[response]"
                )
                tgt = " ".join(
                    [
                        turn_info.out_state_str,
                        turn_info.out_intent_str,
                        turn_info.out_act_str,
                        turn_info.curr_utt,
                    ]
                )

        if tgt:
            processed_turns.append(
                dataclasses.asdict(ProcessedTurn.from_turn_info(turn_info, tgt))
            )

    return processed_turns


def example_filter(
    preproc_config: DictConfig, turn_list: list[TurnInfo]
) -> list[TurnInfo]:
    """Extract specified percentage of examples.
    And ensure uniform domain distribution if specified.

    Args:
        preproc_config: Preprocessing configuration.
        turn_list: A list of TurnInfo containing all examples.

    Returns:
        Specified percentage of examples, with uniform domain distribution if
        needed.
    """
    if preproc_config.data_percent == 0.0:
        return turn_list

    out_sample_num = int(len(turn_list) * preproc_config.data_percent)
    if not preproc_config.uniform_domain_distribution:
        if preproc_config.randomize_items:
            random.shuffle(turn_list)
        return turn_list[:out_sample_num]
    else:
        domain_examples = {}
        domain_id = {}
        domain_count = 0
        for turn in turn_list:
            if turn.turn_domain in domain_id:
                domain_examples[domain_id[turn.turn_domain]].append(turn)
            else:
                domain_examples[domain_count] = [turn]
                domain_id[turn.turn_domain] = domain_count
                domain_count += 1

        # How many examples from each domain has been added to the final list.
        consumed_examples = {d: 0 for d in range(domain_count)}
        uniform_turn_list = []
        for s in range(out_sample_num):
            # Find first domain that still has unused examples.
            domain_id = s % domain_count
            for d in range(domain_count):
                cand_domain = (domain_id + d) % domain_count
                if len(domain_examples[cand_domain]) > consumed_examples[cand_domain]:
                    domain_id = cand_domain
                    break

            uniform_turn_list.append(
                domain_examples[domain_id][consumed_examples[domain_id]]
            )
            consumed_examples[domain_id] += 1

        if preproc_config.randomize_items:
            random.shuffle(uniform_turn_list)

        return uniform_turn_list


def process_shard(
    config: DictConfig,
    version: str,
    override: bool,
    output_path: pathlib.Path,
    shard_path: pathlib.Path,
    split: str,
) -> None:
    preproc_config = config.preprocessing
    config.metadata.split = split

    if hasattr(config.preprocessing, "descriptions"):
        config.preprocessing.descriptions.split = split

    logger.info(f"Preprocessing {split} set for shard {shard_path}")
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
        schemas = json.load(f)
    slots, item_desc = load_schema(preproc_config, schemas)

    preprocessed_dialogues = {"data": []}
    pattern = re.compile(r"dialogues_[0-9]+\.json")
    this_shard_files = sorted(
        [f for f in this_shard_data_dir.iterdir() if pattern.match(f.name)],
        key=lambda fpath: int((fpath.name.split("_")[1]).split(".")[0]),
    )
    for file in this_shard_files:
        if pattern.match(file.name):
            logger.info(
                f"Processing file {file.name} (shard: {shard_path}, split: {split})"
            )
            with open(file, "r") as f:
                raw_dialogues = json.load(f)

            preproc_file = preprocess_file(
                preproc_config,
                slots,
                item_desc,
                file.name,
                raw_dialogues,
            )
            preprocessed_dialogues["data"].extend(preproc_file)

    preprocessed_dialogues["data"] = serialize(
        preproc_config,
        example_filter(preproc_config, preprocessed_dialogues["data"]),
    )

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
@click.option("--all", "split", flag_value="all")
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
    config.metadata.output_path = output_path
    output_path = pathlib.Path(output_path)
    data_paths = [pathlib.Path(p) for p in data_paths]
    splits = ["train", "test", "dev"] if split == "all" else [split]

    with Pool() as pool:
        pool.starmap(
            partial(process_shard, config, version, override, output_path),
            product(data_paths, splits),
        )

    logger.info("Done")


if __name__ == "__main__":
    main()
