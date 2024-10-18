# Based on code released by Google Research, the original copyright notice
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

"""Create MultiWOZ schemaless training data for T5x models.

Does not process TRADE pre-processed data.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import random
import string
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Literal, Optional

import click
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from robust_dst.multiwoz_utils import (
    DialogueActs,
    MultiwozDataclassData,
    MultiwozDialog,
    SchemaInfo,
    extract_domains,
    get_domain,
    load_data_as_dataclasses,
    load_dialogue_acts,
    load_schema,
)
from robust_dst.utils import get_datetime, save_data, set_seed

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TextToTextExample:
    """A single text-to-text dialogue example.
    Attributes:
        src: Input text for the model.
        tgt: Target text for the model.
        dialog_id: Id of dialog this example was generated from.
        turn: Turn of dialog this example was generated from.
        metadata: Any other key-value pairs to be included in the output TF Example.
        frame: Frame of the dialog this example was generated from.
    """

    src: str
    tgt: str
    dialog_id: str
    turn: str
    metadata: dict[str, str] = dataclasses.field(default_factory=dict)
    frame: int = 0

    # see PrecessedTurn class for descriptions of these attributes
    dialogue_context: str = ""
    turn_domain: str = ""
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
    """mapping from action to list of action parameters of the current system turn
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
    """mapping from action to list of action parameters of the previous user turn
    - For actions on slots, the action maps to a list of slot names
        E.g. `{"REQUEST": ["restaurant_name", "location"]}`
    - For actions on slots with values, the values are also included with the slot
      names, inside brackets.
      names, inside brackets.
        E.g. `{"INFORM": ["location(San Jose)", "restaurant_name(Sino)"]}`
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
    # sytem utterance of the current turn
    sys_utt: str
    # mapping from act to list of slots of the current turn
    # e.g. "{\"REQUEST\": [\"restaurant_name\", \"location\"]}"
    sys_acts: str
    # user utterance of the previous turn
    user_utt: str
    # mapping from act to list of slots of the previous turn, same form as prev_sys_acts
    user_acts: str

    def from_example(
        example: TextToTextExample,
    ) -> ProcessedTurn:
        return ProcessedTurn(
            description_mapping=json.dumps(example.desc_mapping),
            state=example.tgt,
            dialogue_context=example.dialogue_context,
            slot_mapping=json.dumps(example.slot_mapping),
            cat_values_mapping=json.dumps(example.cat_values_mapping),
            intent_mapping=json.dumps(example.intent_mapping),
            turn_domain=example.turn_domain,
            turn_idx=example.turn,
            dialogue_id=example.dialog_id,
            file_name=example.file_name,
            sys_utt=example.sys_utt,
            sys_acts=json.dumps(example.sys_acts),
            user_utt=example.user_utt,
            user_acts=json.dumps(example.user_acts),
        )


def create_schemaless_data(
    dialogs_by_id: dict[str, MultiwozDialog],
    schema_info: SchemaInfo,
    slot_descriptions: dict[str, list[str]],
    dialogue_acts: DialogueActs,
    options: DictConfig,
) -> list[TextToTextExample]:
    """Converts raw MultiWOZ data into schemaless examples."""

    def _multiple_choice_answer(
        slot_id: int,
        letters: list[str],
        possible_values_shuffled: list[str],
        value: str,
    ):
        """Get answer for multiple choice prompt."""
        if value == "none":
            return "none"
        if value == "dontcare":
            return "dontcare"
        # Often we have have "guest house" when the categorical
        # value is "guesthouse".
        if value == "guest house":
            value = "guesthouse"

        if value not in possible_values_shuffled:
            logging.warning(
                'Value "%s" not in possible values %s', value, possible_values_shuffled
            )
            value_nospaces = value.replace(" ", "")
            if value_nospaces in possible_values_shuffled:
                letter = letters[possible_values_shuffled.index(value_nospaces)]
            else:
                # Give up and return unknown as the value.
                logging.warning(
                    'Value "%s" not in possible values %s',
                    value,
                    possible_values_shuffled,
                )
                return "unknown"
        else:
            letter = letters[possible_values_shuffled.index(value)]

        if options.multiple_choice == "1a":
            return f"{slot_id}{letter}"
        elif options.multiple_choice == "a":
            return letter

    def _process_one_turn(
        dialog_id: str,
        turn_num: int,
        belief_state: dict[str, str],
        history_str: str,
        active_domains: set[str],
        slot_descriptions: dict[str, list[str]],
    ) -> TextToTextExample:
        """Creates a `TextToTextExample` from a turn in the dialogue.

        Args:
            dialog_id:
            turn_num:
            belief_state: belief state of the current turn
            history_str: the dialogue context
            active_domains: set of domain names active in the turn belief state
            slot_descriptions: mapping between slot name and a list of descriptions of
                that slot
        """
        # Generate a random mapping from slot name to index.
        # slot_names[i] will translate to "i:slot_names[i]".
        # slot_names is the entire MultiWOZ slot ontology
        slot_names = list(slot_descriptions.keys())
        if options.use_active_domains_only:
            domains = active_domains
            slot_names = list(
                filter(
                    lambda name: get_domain(name) in domains,
                    slot_names,
                )
            )
        else:
            domains = {get_domain(slot_name) for slot_name in slot_names}

        # Remove descriptions of slots from the blocked domains for leave-one-out
        # experiments
        if options.blocked_domains:
            domains = domains - set(options.blocked_domains)
            slot_names = list(
                filter(
                    lambda name: get_domain(name) not in options.blocked_domains,
                    slot_names,
                )
            )

        random.shuffle(slot_names)

        cat_values_mapping = defaultdict(dict)
        prefix_pieces = []
        state_pieces = []
        for i, slot_name in enumerate(slot_names):
            domain = get_domain(slot_name)

            # Decide description for this slot.
            # slot_descriptions.json has multiple descriptions for each slot,
            # for now only use the first one.
            full_desc = slot_descriptions[slot_name][0]
            if options.description_type == "full_desc":
                desc = f"{i}{options.delimiter}{full_desc}"
            elif options.description_type == "full_desc_with_domain":
                desc = f"{i}{options.delimiter}{domain}-{full_desc}"
            elif options.description_type == "item_name":
                desc = f"{i}{options.delimiter}{slot_name}"
            elif options.description_type == "shuffled_item_name":
                # Make a copy of the slot name and shuffle it
                slot_name_shuffled = list(slot_name)
                random.shuffle(slot_name_shuffled)
                slot_name_shuffled = "".join(slot_name_shuffled)
                desc = f"{i}{options.delimiter}{slot_name_shuffled}"

            letters = list(string.ascii_lowercase)
            possible_values_shuffled = []
            slot = schema_info.slots_by_domain[domain][slot_name]
            # Optionally append multiple choice prompt for this slot's description.
            if options.multiple_choice != "none" and slot.is_categorical:
                possible_values_shuffled = slot.possible_values.copy()
                random.shuffle(possible_values_shuffled)
                assert len(possible_values_shuffled) < len(letters)

                if options.multiple_choice == "a":
                    idx_format_str = "{letter}"
                elif options.multiple_choice == "1a":
                    idx_format_str = "{slot_id}{letter}"

                possible_values_pieces = []
                for letter, value in zip(letters, possible_values_shuffled):
                    if options.description_type == "shuffled_item_name":
                        value_list = list(value)
                        random.shuffle(value_list)
                        value = "".join(value_list)
                    cat_value_idx = idx_format_str.format(slot_id=i, letter=letter)
                    possible_values_pieces.append(f"{cat_value_idx}) {value}")

                    cat_values_mapping[slot_name][value] = cat_value_idx

                desc += " " + " ".join(possible_values_pieces)
            prefix_pieces.append(desc)

            # Generate target state string for this slot.
            if slot_name in belief_state:
                values = belief_state[slot_name]
                if "|" in values:
                    values = values.split("|")
                elif ">" in values:
                    values = values.split(">")
                elif "<" in values:
                    values = values.split("<")
                elif options.multiwoz_version != "2.2":
                    # In 2.2, multiple possible values are given. Consider a list of
                    # values to accommodate.
                    values = [values]

                # Convert this target value to categorical if required.
                if options.multiple_choice != "none" and slot.is_categorical:
                    values = [
                        _multiple_choice_answer(
                            i, letters, possible_values_shuffled, val
                        )
                        for val in values
                    ]

                values_str = options.value_separator.join(values)
                state_pieces.append(f"{i}{options.delimiter}{values_str}")

        # Make sure all slots in the belief state end up in the target.
        if len(state_pieces) != len(belief_state):
            raise ValueError(
                "Len of state_pieces must equal len of domain belief state."
                f"len(state_pieces): {len(state_pieces)}. "
                f"len(belief_state): {len(belief_state)}."
            )

        prefix_str = " ".join(prefix_pieces)
        state_separator = " ; " if options.use_target_separators else " "
        state_str = "[states] " + state_separator.join(state_pieces)

        user_acts = {}
        for d, domain_act_info in dialogue_acts[dialog_id][str(turn_num - 1)].items():
            if d in domains:
                user_acts.update(domain_act_info)
        sys_acts = {}
        for d, domain_act_info in dialogue_acts[dialog_id][str(turn_num)].items():
            if d in domains:
                sys_acts.update(domain_act_info)

        return TextToTextExample(
            src=f"{prefix_str} {history_str.strip()}".strip(),
            # TODO(jeffreyzhao): Support intents, requested slots from
            # MultiWOZ 2.2.
            # For now add empty "[intents] [req_slots]" to be consistent with
            # SGD.
            tgt=f"{state_str.strip()} [intents] [req_slots]",
            dialog_id=dialog_id,
            turn=str(turn_num),
            metadata={
                "slot_ordering": ", ".join(slot_names),
            },
            #
            dialogue_context=history_str.strip(),
            desc_mapping=dict(
                prefix_piece.split(options.delimiter) for prefix_piece in prefix_pieces
            ),
            slot_mapping={
                str(idx): slot_name for idx, slot_name in enumerate(slot_names)
            },
            cat_values_mapping=cat_values_mapping,
            intent_mapping={},
            turn_domain="all",
            user_acts=user_acts,
            sys_acts=sys_acts,
        )

    examples = []
    for dialog_id, dialog in tqdm(dialogs_by_id.items()):
        history_str = ""

        for turn_num, turn in enumerate(dialog.turns):
            is_system = turn_num % 2 == 1
            speaker = "system" if is_system else "user"
            utterance = turn.utterance.strip().replace("\t", " ")
            belief_state = turn.belief_state

            # State, action, response only appear at system turns.
            domains_in_turn = extract_domains(belief_state)
            if not is_system:
                prev_utterance = utterance
            else:
                if domains_in_turn & set(options.blocked_domains):
                    continue
                turn_info = _process_one_turn(
                    dialog_id,
                    turn_num,
                    belief_state,
                    history_str,
                    domains_in_turn,
                    slot_descriptions,
                )
                turn_info.file_name = "data.json"
                turn_info.user_utt = prev_utterance
                turn_info.sys_utt = utterance

                examples.append(
                    dataclasses.asdict(ProcessedTurn.from_example(turn_info))
                )

            history_str += f"[{speaker}] {utterance} "

    return examples


def process_split(
    config: DictConfig,
    version: str,
    override: bool,
    output_path: pathlib.Path,
    multiwoz_data: MultiwozDataclassData,
    schema_info: SchemaInfo,
    dialogue_acts: DialogueActs,
    split: Literal["train", "test", "dev"],
):
    split_to_attr_name = {
        "train": "train_dialogs",
        "dev": "dev_dialogs",
        "test": "test_dialogs",
    }
    preproc_config = config.preprocessing

    examples = create_schemaless_data(
        getattr(multiwoz_data, split_to_attr_name[split]),
        schema_info,
        multiwoz_data.slot_descriptions,
        dialogue_acts,
        preproc_config,
    )

    save_data(
        {"data": examples},
        output_path.joinpath("multiwoz", split),
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
    "--data_path",
    "data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the original MultiWOZ datasets.",
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
    "-s",
    "--schema_file",
    "schema_file",
    required=True,
    type=click.Path(exists=False),
    help="MultiWOZ schema file in 2.2/SGD format.",
)
@click.option(
    "-a",
    "--dialogue_acts_file",
    "dialogue_acts_file",
    required=True,
    type=click.Path(exists=False),
    help="MultiWOZ 2.2 dialogue acts file.",
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
    data_path: tuple[str],
    output_path: pathlib.Path,
    schema_file: pathlib.Path,
    dialogue_acts_file: pathlib.Path,
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
    config.metadata.raw_data_path = data_path
    config.metadata.output_path = output_path
    output_path = pathlib.Path(output_path)
    splits = ["train", "test", "dev"] if split == "all" else [split]

    multiwoz_data = load_data_as_dataclasses(
        data_path=data_path,
        multiwoz_version=config.preprocessing.multiwoz_version,
        is_trade=False,
    )
    schema_info = load_schema(schema_file)
    dialogue_acts = load_dialogue_acts(dialogue_acts_file, schema_info)

    with Pool() as pool:
        pool.map(
            partial(
                process_split,
                config,
                version,
                override,
                output_path,
                multiwoz_data,
                schema_info,
                dialogue_acts,
            ),
            splits,
        )

    logger.info("Done")


if __name__ == "__main__":
    main()
