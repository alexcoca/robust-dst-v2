from __future__ import annotations

import functools
import json
import logging
import pathlib
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional, Union

from robust_dst.parser_metadata import ParserMetaMixin
from robust_dst.utils import default_to_regular, nested_defaultdict

logger = logging.getLogger(__name__)


class Parser(ABC, ParserMetaMixin):
    def __init__(
        self,
        schema_path: Union[str, pathlib.Path],
        template_dir: Optional[Union[str, pathlib.Path]] = None,
        files_to_parse: Optional[list[str]] = None,
        dialogues_to_parse: Optional[list[str]] = None,
    ) -> None:
        """Instantiate a parser object.

        Args:
            schema_path: Path to the schemas
            template_dir: Path to the directory containing the blank dialogue templates
                to be filled, required for SGD parsing
            files_to_parse: List of file names to parse a subset of files
            dialogues_to_parse: List of dialogue ids to parse a subset of dialogues
        """
        self.template_dir = template_dir
        self.files_to_parse = files_to_parse
        self.dialogues_to_parse = dialogues_to_parse
        self.schema_slot_metadata = self._get_schema_slot_metadata(schema_path)

    @staticmethod
    def _get_schema_slot_metadata(
        schema_path: Union[str, pathlib.Path]
    ) -> dict[str, dict[str, dict]]:
        """Obtain slot metadata in schema in a hierarchical dict
        {service: slot}, {slot: slot metadata}.

        Args:
            schema_path: Path to the schema file

        Returns:
            A mapping from service names to slots to slot information extracted from
            the schema. Used to differentiate categorical/non-categorical slots.
        """
        with open(schema_path, "r") as f:
            schema = json.load(f)

        schema_slot_metadata = defaultdict(dict)
        for service_metadata in schema:
            service_name = service_metadata["service_name"]
            for slot_metadata in service_metadata["slots"]:
                slot_name = slot_metadata["name"]
                schema_slot_metadata[service_name][slot_name] = slot_metadata

        return schema_slot_metadata

    def restore_case(
        self, value: str, service: str, restore_categorical_case: bool = True
    ) -> str:
        """Restore the case of a given categorical slot `value` to the original
        schema casing to ensure scoring is correct."""
        if (
            not restore_categorical_case
            or value not in self.LOWER_TO_SCHEMA_CASE_MAPPING
        ):
            return value
        if value in self.LOWER_TO_SCHEMA_CASE_MAPPING:
            recased_data = self.LOWER_TO_SCHEMA_CASE_MAPPING[value]
            if isinstance(recased_data, str):
                return recased_data
            else:
                assert isinstance(recased_data, dict)
                return recased_data[service]

    @abstractmethod
    def _map_values_to_slots(
        self,
        preprocessed_refs: list[dict],
        predictions: list[str],
    ) -> dict[str, dict[str, dict[str, dict[str, dict[str, list[str]]]]]]:
        """Extract slot-values in the prediction strings to a dictionary.

        Args:
            preprocessed_refs: A list of dictionaries with the preprocessed data, which
                should follow the same ordering as the predicted values (i.e.,
                predictions[i] corresponds to the example in preprocessed_refs[i].
            predictions: A list of values predicted by the model. For each frame,
                a prediction is made for each slot in that service.

        Returns:
            Dictionary with keys [dial_file_name][dial_id][turn_idx][service][slot],
            and values are list[str]
        """
        pass

    @staticmethod
    def _load_sgd_templates(
        template_dir: Union[str, pathlib.Path],
        files_to_parse: Optional[list[str]] = None,
        dialogues_to_parse: Optional[list[str]] = None,
    ) -> tuple[dict[str, Path], dict[str, list[dict]]]:
        """Load blank dialogue files as templates.

        Args:
            template_dir: Path to the directory containing the blank dialogue templates
                to be filled
            files_to_parse: Optional list of file names to parse a subset of files
            dialogues_to_parse: Optional list of dialogue ids to parse a subset of
                dialogues

        Returns:
            A mapping from file name to file path, and a mapping from SGD file names to
            list of SGD-formatted dialogue templates without annotations.
        """
        logger.info("Loading SGD dialogue templates")
        if not any((files_to_parse is None, dialogues_to_parse is None)):
            logger.info(f"There are {len(files_to_parse)} files to parse")
            logger.info(f"There are {len(dialogues_to_parse)} dialogues to parse")
        pattern = re.compile(r"dialogues_[0-9]+\.json")
        sgd_files_paths = list(Path(template_dir).glob("*.json"))
        sgd_files_paths = sorted(
            [p for p in sgd_files_paths if pattern.match(p.name)],
            key=lambda p: int((p.name.split("_")[1]).split(".")[0]),
        )
        sgd_dialogue_templates = defaultdict(list)
        name_to_fpath = {}
        for fpath in sgd_files_paths:
            fname = fpath.name
            if files_to_parse is not None and fname not in files_to_parse:
                logger.info(
                    f"Skipping {fpath.name} as it is not amongst files to parse. "
                    "This is expected if you are testing with few samples."
                )
                continue
            with open(fpath, "r") as f:
                raw_dialogues = json.load(f)
            # exclude files/dialogues to support testing/debugging
            if dialogues_to_parse is not None:
                logger.info(
                    "Some dialogues will be skipped during template loading as only a"
                    " subset have been decoded"
                )
                raw_dialogues = [
                    dial
                    for dial in raw_dialogues
                    if dial["dialogue_id"] in dialogues_to_parse
                ]
            sgd_dialogue_templates[fname] = raw_dialogues
            name_to_fpath[fname] = fpath

        return name_to_fpath, dict(sgd_dialogue_templates)

    @staticmethod
    def _copy_predictions_to_sgd_format(
        dial_file_name: str,
        raw_sgd_dialogues: list[dict],
        slot_values_predictions: dict[
            str, dict[str, dict[str, dict[str, dict[str, list[str]]]]]
        ],
    ) -> None:
        """Copies predictions in SGD frames.

        If predictions are not found, dialogues are ignored and dialogues removed from
        the input list and references.

        Args:
            dial_file_name: Name of the dialogue file
            raw_sgd_dialogues: List of the blank sgd dialogue templates
            slot_values_predictions: Dictionary with keys
                [dial_file_name][dial_id][turn_idx][service][slot],
                and values are lists of slot values
        """

        # in testing with few dialogues, we might not predict the entire
        # dialogue state so we abort filling in the templates the first
        # time this occurs
        stop_copying = False
        for dial in raw_sgd_dialogues:
            dial_id = dial["dialogue_id"]  # dialogue unique id (name), e.g., "1_00000"
            try:
                # user+sys if only user turn predictions
                num_turn_predictions = (
                    len(slot_values_predictions[dial_file_name].get(dial_id, [])) * 2
                )
                assert len(dial["turns"]) == num_turn_predictions
            except AssertionError:
                logger.warning(
                    f"For dialogue {dial_id}, there were {len(dial['turns'])} turns"
                    f" but only {num_turn_predictions} predictions were made."
                    " Ignore this warning if you are testing the code with few examples"
                    " as this does not guarantee a prediction is made for every turn"
                    " in a dialogue."
                )
            if stop_copying:
                break
            for turn_idx, turn in enumerate(dial["turns"]):
                if stop_copying or turn["speaker"] == "SYSTEM":
                    continue
                turn_idx = str(turn_idx)
                try:
                    assert len(turn["frames"]) == len(
                        slot_values_predictions[dial_file_name][dial_id][turn_idx]
                    )
                except AssertionError:
                    logger.warning(
                        f"For dialogue {dial_id}, there were {len(turn['frames'])}"
                        f" frames in turn {turn_idx}. "
                        f"{len(slot_values_predictions[dial_file_name][dial_id][turn_idx])}"  # noqa: E501
                        " predictions were made for those frames. Ignore this warning"
                        " if you are testing the code with few examples as this does"
                        " not guarantee a prediction is made for every frame in a turn."
                    )
                except KeyError:
                    logger.warning(
                        f"{dial_id}: Could not find turn {turn_idx} in slot value"
                        " predictions. Only"
                        f" {list(slot_values_predictions[dial_file_name].get(dial_id, {}).keys())}"  # noqa: E501
                        " turns had predictions"
                    )
                for frame_idx, frame in enumerate(turn["frames"]):
                    service = frame["service"]
                    try:
                        slot_values_prediction = slot_values_predictions[
                            dial_file_name
                        ][dial_id][turn_idx][service]
                    except KeyError:
                        logger.warning(
                            f"{dial_id}-{turn_idx} Could not find predictions for"
                            f" service {service} in dialogue"
                        )
                        stop_copying = True
                        break
                    frame["state"]["slot_values"] = slot_values_prediction

    def convert_to_sgd_format(
        self,
        preprocessed_refs: list[dict],
        predictions: list[str],
        write_to_disk: bool = False,
    ) -> dict[str, list[dict]]:
        """Replace the annotations of `sgd_dialogue_templates` with model predictions.

        This method requires template_dir attribute to be set on the Parser instance.

        Args:
            preprocessed_refs: A list of dictionaries with the preprocessed data, which
                should follow the same ordering as the predicted values (i.e.,
                predictions[i] corresponds to the example in preprocessed_refs[i].
            predictions: A list of values predicted by the model in the D3ST format.
                For each frame, a prediction is made for each slot in that service.
            write_do_disk: Whether to write the SGD formatted dialogues to disk by
                overwirting the templates.

        Notes:
            Each string in the predicitons is mapped to slot-value pairs using
            preprocessed_refs by the _map_values_to_slots method.

        Returns:
            Dictionary of SGD dialogue file name to dialogues filled with model
            predictions
        """
        if self.template_dir is None:
            raise RuntimeError("template_dir is required for SGD parsing")
        name_to_fpath, sgd_dialogue_templates = self._load_sgd_templates(
            template_dir=self.template_dir,
            files_to_parse=self.files_to_parse,
            dialogues_to_parse=self.dialogues_to_parse,
        )

        # a mapping from SGD file names to list of SGD-formatted dialogue templates
        # without annotations
        sgd_format_predictions = deepcopy(sgd_dialogue_templates)
        # slot_value_predictions has keys
        # [dial_file_name][dial_id][turn_idx][service][slot] & vals are list[str]
        slot_value_predictions = self._map_values_to_slots(
            preprocessed_refs=preprocessed_refs,
            predictions=predictions,
        )

        logger.info("Parsing predictions to SGD format...")
        for fname, raw_dialogues in sgd_format_predictions.items():
            logger.info(f"Converting file {fname}")
            try:
                assert len(raw_dialogues) == len(slot_value_predictions[fname])
            except AssertionError:
                logger.warning(
                    f"For file {fname} only"
                    f" {len(slot_value_predictions[fname])} dialogues hadpredictions"
                    f" but there were {len(raw_dialogues)} in total. Ignore this"
                    " warningif you are testing the code with few examples as this"
                    " does not guarantee a predictionis made for every dialogue in a"
                    " file."
                )
            self._copy_predictions_to_sgd_format(
                fname, raw_dialogues, slot_value_predictions
            )
            logger.info(f"Parsed file {fname}")

        if write_to_disk:
            self._write_sgd_formatted_dialogues(sgd_format_predictions, name_to_fpath)

        return sgd_format_predictions

    def _write_sgd_formatted_dialogues(
        self,
        sgd_format_predictions: dict[str, list[dict]],
        name_to_fpath: dict[str, Path],
    ) -> None:
        """Write the sgd dialogues filled with predictions.

        Args:
            sgd_format_predictions: Dictionary of SGD dialogue file name to dialogues
                filled with model predictions
            name_to_fpath: A mapping from file name to file path
        """
        logger.info("Writing SGD-formatted dialogues...")
        for fname in sgd_format_predictions:
            with open(name_to_fpath[fname], "w") as f:
                json.dump(sgd_format_predictions[fname], f, indent=2)
        logger.info("Completed parsing!")

    def convert_to_multiwoz_format(
        self,
        preprocessed_refs: list[dict],
        predictions: list[str],
    ) -> dict[str, list[dict]]:
        """Convert the model predictions into format of the Shades of BLEU evaluator.

        The evaluator can be found at https://github.com/Tomiinek/MultiWOZ_Evaluation.

        Args:
            preprocessed_refs: A list of dictionaries with the preprocessed data, which
                should follow the same ordering as the predicted values (i.e.,
                predictions[i] corresponds to the example in preprocessed_refs[i].
            predictions: A list of values predicted by the model in the D3ST format.
                For each frame, a prediction is made for each slot in that service.

        Notes:
            Each string in the predicitons is mapped to slot-value pairs using
            preprocessed_refs by the _map_values_to_slots method.

        Returns:
            Dictionary of the following format
                {
                    "xxx0000" : [
                        {
                            "response": "Your generated delexicalized response.",
                            "state": {
                                "restaurant" : {
                                    "food" : "eatable"
                                }, ...
                            },
                            "active_domains": ["restaurant"]
                        }, ...
                    ], ...
                }
        """
        responses = defaultdict(dict)
        for preprocessed_ref in preprocessed_refs:
            dial_id = preprocessed_ref["dialogue_id"]
            turn_idx = preprocessed_ref["turn_idx"]
            response = preprocessed_ref["sys_utt"]
            responses[dial_id.replace(".json", "")][turn_idx] = response
        responses = dict(responses)

        # slot_value_predictions has keys
        # [dial_id][turn_idx]["all"][slot] & vals are list[str]
        slot_value_predictions = self._map_values_to_slots(
            preprocessed_refs=preprocessed_refs,
            predictions=predictions,
        )["data.json"]

        responses = defaultdict(dict)
        for preprocessed_ref in preprocessed_refs:
            dial_id = preprocessed_ref["dialogue_id"]
            turn_idx = preprocessed_ref["turn_idx"]
            response = preprocessed_ref["sys_utt"]
            responses[dial_id.replace(".json", "")][turn_idx] = response
        responses = dict(responses)

        # slot_value_predictions has keys
        # [dial_id][turn_idx]["all"][slot] & vals are list[str]
        slot_value_predictions = self._map_values_to_slots(
            preprocessed_refs=preprocessed_refs,
            predictions=predictions,
        )["data.json"]

        multiwoz_data = defaultdict(list)
        for dial_id, dial_preds in slot_value_predictions.items():
            dial_id = dial_id.replace(".json", "")
            prev_turn_id = -1
            # the MultiWOZ evaluator does not handle skipped turns, add placeholder
            # for skipped turns
            for turn_id, turn_preds in sorted(
                dial_preds.items(), key=lambda kv: int(kv[0])
            ):
                skipped_turn: bool = int(turn_id) > (prev_turn_id + 2)
                if skipped_turn:
                    for prev_id in range(prev_turn_id + 2, int(turn_id), 2):
                        multiwoz_data[dial_id].append(
                            {
                                "response": responses[dial_id].get(str(prev_id), ""),
                                "state": dict(),
                            }
                        )
                prev_turn_id = int(turn_id)

                state_dict = defaultdict(dict)
                for domain_slot, slot_preds in turn_preds["all"].items():
                    domain, slot_name = domain_slot.split("-")
                    # if multiple values predicted, MultiWOZ evaluation only uses the
                    # first one
                    state_dict[domain][slot_name] = slot_preds[0]
                multiwoz_data[dial_id].append(
                    {"response": responses[dial_id][turn_id], "state": dict(state_dict)}
                )

        return dict(multiwoz_data)


class T5DSTParser(Parser):
    NONE_TOKENS = ["NONE", "none"]
    DONTCARE_TOKEN = "dontcare"
    MULTI_VALUE_TOKEN = " || "
    NULL_VALUE = "NULL"

    def __init__(
        self,
        template_dir: Union[str, pathlib.Path],
        schema_path: Union[str, pathlib.Path],
        data_format: Literal["T5DST", "T5DST2", "google"],
        files_to_parse: Optional[list[str]] = None,
        dialogues_to_parse: Optional[list[str]] = None,
    ):
        super().__init__(
            template_dir=template_dir,
            schema_path=schema_path,
            files_to_parse=files_to_parse,
            dialogues_to_parse=dialogues_to_parse,
        )
        self.data_format = data_format

    def _map_values_to_slots(
        self,
        preprocessed_refs: list[dict],
        predictions: list[str],
    ) -> dict[str, dict[str, dict[str, dict[str, dict[str, list[str]]]]]]:
        logger.info("Mapping predicted values to slot names...")
        slot_value_predictions = nested_defaultdict(dict, depth=4)
        for line_idx, (prediction, preprocessed_ref) in enumerate(
            zip(predictions, preprocessed_refs)
        ):
            # get example info
            dial_file_name = preprocessed_ref["file_name"]
            dial_id = preprocessed_ref["dialogue_id"]
            turn_idx = preprocessed_ref["turn_idx"]
            service = preprocessed_ref["service"]
            slot = preprocessed_ref["slot"]
            assert (
                slot
                not in slot_value_predictions[dial_file_name][dial_id][turn_idx][
                    service
                ]
            )
            # inactive slot
            if prediction in self.NONE_TOKENS:
                # nb. this excludes "None",
                # which is a value in dev/Media_2/subtitles_language
                continue
            # map back values that are ambiguous to corpus values
            if (
                self.data_format in ["T5DST", "T5DST2"]
                and service in self.SPECIAL_SLOT_VALUES
                and slot in self.SPECIAL_SLOT_VALUES[service]
            ):
                if prediction in self.SPECIAL_SLOT_VALUES[service][slot]:
                    slot_value_predictions[dial_file_name][dial_id][turn_idx][service][
                        slot
                    ] = ["None"]
                else:
                    slot_value_predictions[dial_file_name][dial_id][turn_idx][service][
                        slot
                    ] = [prediction]
                continue
            if self.data_format in ["T5DST", "T5DST2"]:
                values = [prediction]
            else:  # google format
                is_categorical = self.schema_slot_metadata[service][slot][
                    "is_categorical"
                ]
                if prediction == self.DONTCARE_TOKEN:
                    values = [prediction]
                elif is_categorical:  # deal with multiple choices example
                    # model could generate value string or invalid options
                    possible_values = json.loads(preprocessed_ref["option2value"])
                    try:
                        values = [possible_values[prediction]]
                    except KeyError:
                        logger.warning(
                            f"Predicted option {prediction} for slot {slot} in service"
                            f" {service} but only"
                            f" {preprocessed_ref['option2value']} are valid options"
                        )
                        values = [self.NULL_VALUE]
                else:  # split multiple values
                    values = prediction.split(self.MULTI_VALUE_TOKEN)
                # restore cases
                values = [
                    self.restore_case(value, service, is_categorical)
                    for value in values
                ]
            assert isinstance(values, list)
            slot_value_predictions[dial_file_name][dial_id][turn_idx][service][
                slot
            ] = values
        return default_to_regular(slot_value_predictions)

class SDTParser(Parser):
    NONE_TOKENS = ["NONE", "none"]
    DONTCARE_TOKEN = "dontcare"

    def __init__(
            self,
            template_dir: Union[str, pathlib.Path],
            schema_path: Union[str, pathlib.Path],
            data_format: Literal["google"],
            files_to_parse: Optional[list[str]] = None,
            dialogues_to_parse: Optional[list[str]] = None,
    ):
        super().__init__(
            template_dir=template_dir,
            schema_path=schema_path,
            files_to_parse=files_to_parse,
            dialogues_to_parse=dialogues_to_parse,
        )
        self.data_format = data_format

    def _dialogue_id_to_file(self, dialogue_id: str) -> str:
        """Infer the file name from the dialogue id

        dialogue_id is of format: '<file_number>_<dialogue_number>'
        file name is of format: 'dialogues_{file_number}.json'
        Note that file_number in file_name has to be a three digit number"""
        file_number = dialogue_id.split("_")[0] # This is a string
        number_of_zeros = 3 - len(file_number)
        assert number_of_zeros in (0, 1, 2)
        file_number = "0" * number_of_zeros + file_number
        file_name = f"dialogues_{file_number}.json"
        return file_name

    def _drop_none_slots(self, slots: dict[str, str]) -> dict[str, str]:
        """Drop the slots whose values are NONE_TOKENS
        """
        return {slot: value for slot, value in slots.items() if value not in self.NONE_TOKENS}

    def _handle_categorical_slots(self, slots: dict[str, str],
                                  categorical_mapping: str,
                                  dialogue_id: str) -> dict[str, str]:
        """Handle categorical slots present in model output
        """
        # Categorical slots mapping is a dictionary but is stored as a string
        # So we need to convert it to a dictionary
        try:
            mapping_dict = eval(categorical_mapping)
        except SyntaxError:
            logger.error(f"Syntax error in categorical slots mapping: {categorical_mapping}"
                         f"for dialogue id: {dialogue_id}"
                         "Cannot convert it to a dictionary and hence cannot handle categorical slots")
            return slots
        # Now that categorical slots mapping is a dictionary, we can handle the categorical slots
        for slot, value in slots.items():
            if slot not in mapping_dict:
                continue # Ignore non-categorical slots
            if value == self.DONTCARE_TOKEN:
                continue
            mapping = mapping_dict[slot].get(value, None)
            if mapping is None:
                # So the output is diffent from the options present so just raise a warning
                logger.warning(f"Value: {value} for slot: {slot} in dialogue id: {dialogue_id}"
                                    "is not present in the options for the slot"
                                    f"Options for the slot: {mapping_dict[slot]}")
            # So the model output is present in the schema and isn't None
            slots[slot] = mapping
        return slots


    def _extract_slots_from_output(self, output: str,
                                   dialogue_id: str) -> dict[str, str]:
        """Extract the slot-value pairs from the output of the model

        In ideal case, the output of the model will be a string of the following format:
        '[state] <slot_name_1>=<value_1> <slot_name_2>=<value_2> ...'
        The extraction will be done without looking at the schema but just from the output string
        """
        slots = {}
        # If output starts with '[state] ', remove it else return empty dict
        start_format = r'\[state\](.*)'
        start_match = re.search(start_format, output)
        if not start_match:
            return slots # If the output does not start with '[state]' then ignore it
        output = start_match.group(1).strip() # Remove the '[state]' from the output

        # Now that [state] is removed this regex will extract the slot-value pairs
        # Slot name and values will be separated by '='
        # slot names only inlcude alphanumeric characters and underscores
        # value names include all characters except '='
        # There is a space after the value until the next slot name
        slot_value_pattern = r'(\w+)=((?:(?!\w+=).)+)'
        matches = re.findall(slot_value_pattern, output)
        slots = {slot.strip(): value.strip() for slot, value in matches}
        return slots

    def _map_values_to_slots(
            self,
            preprocessed_refs: list[dict],
            predictions: list[str]) -> dict[str, dict[str, dict[str, dict[str, dict[str, list[str]]]]]]:

        #logger.info("Mapping predicted values to slot names...")
        slot_value_predictions = nested_defaultdict(dict, depth=4)
        for line_idx, (prediction, preprocessed_ref) in enumerate(
            zip(predictions, preprocessed_refs)):
            categorical_slots_mapping = preprocessed_ref["categorical_slots_mapping"]
            dialogue_id = preprocessed_ref["dialogue_id"]
            turn_id = preprocessed_ref["turn_id"]
            frame_id = preprocessed_ref["frame_id"]
            service = preprocessed_ref["service"]
            file = self._dialogue_id_to_file(dialogue_id)
            slots = self._extract_slots_from_output(prediction, dialogue_id)
            slots = self._drop_none_slots(slots)
            if categorical_slots_mapping: # i.e. Categorical slots isn't equal to None
                slots = self._handle_categorical_slots(slots, categorical_slots_mapping, dialogue_id)
            if len(slots) == 0:
                slot_value_predictions[file][dialogue_id][turn_id][service] = {}
                continue
            for slot, value in slots.items():
                assert (
                    slot not in slot_value_predictions[file][dialogue_id][turn_id][service]
                )
                slot_value_predictions[file][dialogue_id][turn_id][service][slot] = [value]
        return default_to_regular(slot_value_predictions)

class D3STParser(Parser):
    MAX_PARSED_SUBSTRINGS = 30
    MAX_SUBSTRING_LEN = 150
    DONTCARE_TOKEN = "dontcare"

    def __init__(
        self,
        template_dir: Union[str, pathlib.Path],
        schema_path: Union[str, pathlib.Path],
        files_to_parse: Optional[list[str]] = None,
        dialogues_to_parse: Optional[list[str]] = None,
        value_separator: Optional[str] = " || ",
        target_slot_index_separator: str = "=",
        restore_categorical_case: bool = False,
    ):
        super().__init__(
            template_dir=template_dir,
            schema_path=schema_path,
            files_to_parse=files_to_parse,
            dialogues_to_parse=dialogues_to_parse,
        )

        self.value_separator = value_separator
        self.target_slot_index_separator = target_slot_index_separator
        self.restore_categorical_case = restore_categorical_case
        if self.value_separator is not None and self.target_slot_index_separator == ":":
            logger.warning(
                "Parser algorithm is not correct when multiple values are in the target"
                " sequence.Use at your own risk!"
            )

        self.dialogue_id: str = ""
        self.turn_index: str = ""

    def _parse_predicted_string(
        self,
        service: str,
        predicted_str: str,
        slot_mapping: dict,
        cat_values_mapping: dict,
        intent_mapping: dict,
        context: str,
    ) -> dict:
        """Convert predicted string to a SGD state dictionary of the form:

        {
            'slot_values': dict[str, list[str]], mapping slot names to lists of values,
            'active_intent': str, the current turn active intent
            'requested_slots': list[str] of names of information requested by the user.
        }
        """
        state = {"slot_values": {}, "active_intent": "NONE", "requested_slots": []}
        # Expect [states] 0:value 1:1a ... [intents] i1 [req_slots] 2 ...
        match = re.search(r"\[states](.*)\[intents](.*)\[req_slots](.*)", predicted_str)
        if match is None:
            # String was not in expected format
            logger.warning(
                f"Could not parse predicted string {predicted_str} in"
                f" {self.dialogue_id}_{self.turn_index}."
            )
            return state

        # Parse slot values
        if match.group(1).strip():
            pattern = rf"(?<!^)\s+(?=[0-9]+{self.target_slot_index_separator})"

            if self.value_separator not in self.UNAMBIGUOUS_VALUE_SEPARATORS:
                raise ValueError(f"Ambiguous value separator {self.value_separator}")
            if self.value_separator == " || ":
                pattern = (
                    rf"(?<!^)\s+(?=[0-9]+{self.target_slot_index_separator})(?<!\|\| )"
                )
            elif self.value_separator is not None:
                logger.error(
                    "State splitting pattern undefined for value separator"
                    f" {self.value_separator}"
                )

            substrings = re.compile(pattern).split(match.group(1).strip())
            if self.target_slot_index_separator in self.UNAMBIGUOUS_TARGET_SEPARATORS:
                self._parse_without_context(
                    state,
                    substrings,
                    service,
                    predicted_str,
                    slot_mapping,
                    cat_values_mapping,
                    context,
                )
            else:
                raise ValueError(
                    "Unknown target slot index separator"
                    f" {self.target_slot_index_separator}"
                )
        # Parse intent
        intent = match.group(2).strip()
        if intent:
            try:
                state["active_intent"] = intent_mapping[intent]
            except KeyError:
                logger.warning(
                    f"Could not extract intent {intent} in {predicted_str} in"
                    f" {self.dialogue_id}_{self.turn_index}."
                    f" Intent mapping: {intent_mapping}"
                )

        # Parse requested slots
        requested = match.group(3).strip().split()
        for index in requested:
            try:
                state["requested_slots"].append(slot_mapping[index.strip()])
            except KeyError:
                logger.warning(
                    f"Could not extract requested slot {index.strip()} in"
                    f" {predicted_str} in {self.dialogue_id}_{self.turn_index}."
                )
        return state

    def _parse_categorical_slot(
        self,
        state: dict,
        service: str,
        slot_name: str,
        slot_index_value_pair: list[str, str],
        predicted_str: str,
        cat_values_mapping: dict,
    ):
        # Categorical
        recase = functools.partial(
            self.restore_case, restore_categorical_case=self.restore_categorical_case
        )
        # Invert the mapping to get the categorical value
        if slot_name not in cat_values_mapping:
            logger.warning(
                f"{self.dialogue_id}({self.turn_index}): Could not find slot"
                f" {slot_name} in categorical mapping. \nCategorical slots for service"
                f" {service} are {cat_values_mapping.keys()}."
            )
            return
        predicted_value = slot_index_value_pair[1].strip()
        if self.value_separator is not None and self.value_separator in predicted_value:
            value_list = predicted_value.split(self.value_separator)
            value_list = [v.strip() for v in value_list]
        else:
            value_list = [predicted_value]

        cat_values_reverse_mapping = {
            categorical_value_idx: categorical_value
            for categorical_value, categorical_value_idx in cat_values_mapping[
                slot_name
            ].items()
        }
        for value in value_list:
            if value in cat_values_reverse_mapping:
                recased_value = recase(
                    value=cat_values_reverse_mapping[value], service=service
                )
                assert isinstance(recased_value, str)
                if slot_name not in state["slot_values"]:
                    state["slot_values"][slot_name] = [recased_value]
                else:
                    state["slot_values"][slot_name].append(recased_value)

        if slot_name not in state["slot_values"]:
            if predicted_value == self.DONTCARE_TOKEN:
                state["slot_values"][slot_name] = [self.DONTCARE_TOKEN]
            else:
                logger.warning(
                    f"{self.dialogue_id}({self.turn_index}): Could not lookup"
                    f" categorical value {slot_index_value_pair[1].strip()} for slot"
                    f" {slot_index_value_pair[0].strip()} in {predicted_str}. \nValues"
                    f" defined for this slot were {cat_values_mapping[slot_name]}"
                )

    def _parse_without_context(
        self,
        state: dict,
        substrings: list[str],
        service: str,
        predicted_str: str,
        slot_mapping: dict,
        cat_values_mapping: dict,
        context: str,
    ):
        for i, pair in enumerate(substrings):
            pair = pair.strip().split(
                f"{self.target_slot_index_separator}", 1
            )  # slot value pair
            if len(pair) != 2:
                # String was not in expected format
                logger.warning(
                    f"Could not extract slot values in {predicted_str} in"
                    f" {self.dialogue_id}_{self.turn_index}."
                )
                continue
            try:
                slot = slot_mapping[pair[0].strip()]
                if slot in cat_values_mapping:
                    self._parse_categorical_slot(
                        state,
                        service,
                        slot,
                        pair,
                        predicted_str,
                        cat_values_mapping,
                    )
                else:
                    value = pair[1]
                    if (
                        self.value_separator is not None
                        and self.value_separator in value
                    ):
                        value_list = value.split(self.value_separator)
                        value_list = [v.strip() for v in value_list]
                    else:
                        value_list = [value.strip()]
                    state["slot_values"][slot] = value_list
                    for value in value_list:
                        if (
                            value.replace(" ", "").lower()
                            not in context.replace(" ", "").lower()
                        ):
                            # Replace spaces to avoid issues with whitespace
                            if value.strip() not in self.CATEGORICAL_SPECIAL_VALUES:
                                logger.warning(
                                    f"Predicted value {value.strip()} for slot"
                                    f" {pair[0].strip()} not in context in"
                                    f" {self.dialogue_id}_{self.turn_index}."
                                )
            except KeyError:
                logger.warning(
                    f"Could not extract slot {pair[0].strip()} in {predicted_str} in"
                    f" {self.dialogue_id}_{self.turn_index}."
                )

    def _map_values_to_slots(
        self,
        preprocessed_refs: list[dict],
        predictions: list[str],
    ) -> dict[str, dict[str, dict[str, dict[str, dict[str, list[str]]]]]]:
        logger.info("Mapping predicted values to slot names...")
        slot_value_predictions = nested_defaultdict(dict, depth=4)
        for line_idx, (prediction, preprocessed_ref) in enumerate(
            zip(predictions, preprocessed_refs)
        ):
            # get example info
            dial_file_name = preprocessed_ref["file_name"]
            dial_id = preprocessed_ref["dialogue_id"]
            turn_idx = preprocessed_ref["turn_idx"]
            service = preprocessed_ref["turn_domain"]
            slot_mapping = json.loads(preprocessed_ref["slot_mapping"])
            cat_values_mapping = json.loads(preprocessed_ref["cat_values_mapping"])
            intent_mapping = json.loads(preprocessed_ref["intent_mapping"])
            context = preprocessed_ref["dialogue_context"]

            self.dialogue_id = dial_id
            self.turn_index = turn_idx

            # Some checks
            assert service in self.schema_slot_metadata or service == "all"

            state = self._parse_predicted_string(
                service=service,
                predicted_str=prediction.strip(),
                slot_mapping=slot_mapping,
                cat_values_mapping=cat_values_mapping,
                intent_mapping=intent_mapping,
                context=context,
            )

            # TODO required slots and intents

            if len(state["slot_values"]) == 0:
                slot_value_predictions[dial_file_name][dial_id][turn_idx][service] = {}
                continue

            for slot, values in state["slot_values"].items():
                assert (
                    slot
                    not in slot_value_predictions[dial_file_name][dial_id][turn_idx][
                        service
                    ]
                )

                assert isinstance(values, list)
                slot_value_predictions[dial_file_name][dial_id][turn_idx][service][
                    slot
                ] = values

        return default_to_regular(slot_value_predictions)
