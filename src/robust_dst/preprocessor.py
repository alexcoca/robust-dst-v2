from __future__ import annotations

import copy
import json
import logging
import random
import re
from collections import defaultdict
from typing import Optional

from datasets import Dataset, load_dataset
from robust_dst.pipeline import PipelineMixin
from tqdm import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


Batch = dict[str, list]


class ValueType:
    SLOT = 1
    INTENT = 2


class Speaker:
    USER = 1
    SYSTEM = 2


class Preprocessor(PipelineMixin):
    """Preprocessor class build around the HuggingFace Dataset.map method.

    Processing methods can be registered using the @runs_before decorator,
    with the name of the next method in the pipeline specified. These methods
    should form a directed acyclic graph with a unique starting node.

    Such methods should accept a datasets.arrow_dataset.Batch (or a dictionary)
    and return a processed Batch (or a dictionary). This is the same input and
    output behaviour as processing callbacks allowed in the HuggingFace
    Dataset.map method with batch processing enabled, for more information
    please refer to the HuggingFace docs.
    """

    def __init__(
        self,
        tokenize: bool = True,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_source_length: int = 1024,
        max_target_length: int = 64,
        padding: bool | str = False,
        ignore_pad_token_for_loss: bool = True,
        num_proc: Optional[int] = None,
        load_from_cache_file: bool = True,
        source_column: str = "dialogues",
        target_column: str = "state",
        input_prefix: str = "",
    ) -> None:
        """Instantiate a preprocessor object.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to use
            max_source_length (int, optional): The maximum total input sequence length
                after tokenization. Defaults to 1024.
            max_target_length (int, optional): The maximum total sequence length for
                target text after tokenization. Defaults to 64.
            padding (Union[bool, str], optional): Activates and controls padding.
                Accepts the following values:
                    - True or 'longest'
                    - 'max_length'
                    - False or 'do_not_pad' (default)
            ignore_pad_token_for_loss (bool, optional): Whether to ignore the tokens
                corresponding to padded labels in the loss computation.
                Defaults to True.
            num_proc (Optional[int], optional): The number of processes to use for the
                preprocessing. Defaults to None.
            load_from_cache_file (bool, optional): If a cache file storing the current
                computation from function can be identified, use it instead of
                recomputing. Defaults to True.
            source_column (str, optional): The name of the source column.
                Defaults to "dialogues".
            target_column (str, optional): The name of the target column.
                Defaults to "state".
            input_prefix (str, optional): The prefix to be added to the inputs before
                tokenization. Defaults to "".
        """
        self.tokenize = tokenize
        if self.tokenize and not isinstance(tokenizer, PreTrainedTokenizer):
            raise RuntimeError(
                "A tokenizer must be specified if tokenization is desired"
            )
        self.tokenizer = tokenizer

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file
        self.source_column = source_column
        self.target_column = target_column
        self.input_prefix = input_prefix

        self._dataset = None

    def _extract_and_tokenize(
        self,
        examples: Batch,
        truncation: bool,
        discard_truncated_examples: bool,
    ) -> Batch:
        """Tokenize the input and target columns.

        Args:
            examples (Batch): a batch of preprocessed dataset

        Returns:
            Tokenized dataset batch with columns: input_ids, attention_mask, labels
        """
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[self.source_column])):
            if (
                examples[self.source_column][i] is not None
                and examples[self.target_column][i] is not None
            ):
                inputs.append(examples[self.source_column][i])
                targets.append(examples[self.target_column][i])

        inputs = [self.input_prefix + inp for inp in inputs]

        model_inputs = self.tokenizer(
            text=inputs,
            max_length=self.max_source_length,
            padding=self.padding,
            truncation=truncation,
            return_overflowing_tokens=truncation and discard_truncated_examples,
        )
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_target_length,
            padding=self.padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels
        # by -100 when we want to ignore padding in the loss.
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [
                    (token_id if token_id != self.tokenizer.pad_token_id else -100)
                    for token_id in label
                ]
                for label in labels["input_ids"]
            ]

        if not discard_truncated_examples:
            model_inputs["labels"] = labels["input_ids"]
        else:
            new_model_inputs = defaultdict(list)
            # overflow_to_sample_mapping is a list where each element is the original
            # index of the overflown batch sample at that position
            # currently this is not documented
            # https://github.com/huggingface/transformers/blob/6ce6d62b6f20040129ec9831e7c4f6576402ea42/src/transformers/tokenization_utils_fast.py#L469  # noqa
            overflow_to_sample_mapping = model_inputs["overflow_to_sample_mapping"]
            overflow_sample_idx = set()
            for idx in range(1, len(overflow_to_sample_mapping)):
                if (
                    overflow_to_sample_mapping[idx - 1]
                    == overflow_to_sample_mapping[idx]
                ):
                    overflow_sample_idx.add(overflow_to_sample_mapping[idx])
            for idx, sample_idx in enumerate(overflow_to_sample_mapping):
                if sample_idx not in overflow_sample_idx:
                    new_model_inputs["input_ids"].append(model_inputs["input_ids"][idx])
                    new_model_inputs["attention_mask"].append(
                        model_inputs["attention_mask"][idx]
                    )
                    new_model_inputs["labels"].append(labels["input_ids"][sample_idx])
            model_inputs = new_model_inputs

        return model_inputs

    def _preprocess_pipeline(
        self, truncation: bool, discard_truncated_examples: bool, **pipeline_kwargs
    ):
        def wrapped(examples: Batch) -> Batch:
            """The preprocess pipeline, calls all registered preprocess methods.

            Args:
                examples (Batch): a raw dataset batch

            Returns:
                Processed batch
            """
            # deepcopy to ensure the pipeline does not mutate the input examples
            # any mutation on the input exampels in place will be reflected on the
            # processed dataset regardless of the returned batch of this function
            # this will incur a memory penalty, but the pipeline will have no
            # side-effects
            examples_intermediate = copy.deepcopy(examples)
            transformed = self.execute_pipeline(
                examples_intermediate, **pipeline_kwargs
            )
            if self.tokenize:
                return self._extract_and_tokenize(
                    transformed,
                    truncation,
                    discard_truncated_examples,
                )
            return transformed

        return wrapped

    def process(
        self,
        dataset: Dataset,
        *,
        desc: Optional[str] = None,
        truncation: bool = True,
        discard_truncated_examples: bool = False,
        **process_kwargs,
    ) -> Dataset:
        """Process and tokenize a dataset.

        Args:
            dataset (Dataset): Raw dataset to be processed.
            desc (Optional[str]): Meaningful description to be displayed alongside with
                the progress bar while processing.
            truncation (bool): Whether to truncate the ids after tokenization.
            discard_tuncated_examples (bool): Whether to discard examples with inputs
                exceeding the max_source_length when tokenized.

        Returns:
            Processed dataset with features ['input_ids', 'attention_mask', 'labels']
        """
        self._dataset = dataset

        column_names = dataset.column_names
        return dataset.map(
            self._preprocess_pipeline(
                truncation=truncation,
                discard_truncated_examples=discard_truncated_examples,
                **process_kwargs,
            ),
            batched=True,
            num_proc=self.num_proc,
            remove_columns=column_names,
            load_from_cache_file=self.load_from_cache_file,
            desc=desc if desc else "Processing and running tokenizer on dataset",
        )


class T5DSTPreprocessor(Preprocessor):
    def process(
        self,
        dataset: Dataset,
        desc: Optional[str] = None,
        truncation: bool = True,
        **process_kwargs,
    ) -> Dataset:
        if process_kwargs.get("sample_dialogue", False):
            raise NotImplementedError(
                "Sampling from dialogue is not implemented for T5DST"
            )
        return super().process(
            dataset, desc=desc, truncation=truncation, **process_kwargs
        )


class D3STPreprocessor(Preprocessor):
    class AugmentStyle:
        """How the sampled knowledge-seeking turns are added to the prompts."""

        NONE = "NONE"
        """Do not sample or add KSTs."""
        REPLACE = "REPLACE"
        """Use sampled KSTs instead of schema descriptions, fallback if no KST
            available. KSTRandom (Figure 7) in
            [zhangGroundingDescriptionDrivenDialogue2023]"""
        DA = "DA"
        """Data augmentation, where the dataset is augmented with prompts containing
            only the KSTs, forming a dataset twice as large. KSTRandomConcat (Figure 7)
            in [zhangGroundingDescriptionDrivenDialogue2023])"""
        TURN = "TURN"
        """For each turn, the KSTs are concatenated to schema descriptions in random
            order, forming a dataset of the same size.
        """
        TURNSLOT = "TURNSLOT"
        """For each turn, the KSTs and slot names are concatenated to the schema
            descriptions in random order, forming a dataset of the same size.
        """

    NameToKST = dict[str, list]
    DomainToNameMapping = dict[str, NameToKST]
    KSTMapping = dict[int, DomainToNameMapping]
    # key: 1 denotes SLOT, and 2 denotes INTENT, see ValueType

    def __init__(
        self,
        max_source_length: int = 1024,
        max_target_length: int = 512,
        padding: bool | str = False,
        ignore_pad_token_for_loss: bool = True,
        num_proc: Optional[int] = None,
        load_from_cache_file: bool = True,
        source_column: str = "dialogues",
        target_column: str = "state",
        input_prefix: str = "",
        #
        delimiter: str = "=",
        domain_in_desc: bool = False,
        desc_mapping_column: str = "description_mapping",
        diag_context_column: str = "dialogue_context",
        turn_domain_column: str = "turn_domain",
        slot_mapping_column: str = "slot_mapping",
        intent_mapping_column: str = "intent_mapping",
        user_utt_column: str = "user_utt",
        user_act_column: str = "user_acts",
        sys_utt_column: str = "sys_utt",
        sys_act_column: str = "sys_acts",
        **kwargs,
    ) -> None:
        """D3ST runtime preprocessor.

        Args:
            Parameters to the parent Preprocessor class.

            delimiter (str, optional): Delimiter to separate slot/intent IDs from their
                descriptions (input) or values (target). Defaults to "=".
            domain_in_desc (bool, optional): Whether the domain name in included in the
                descriptions (input), e.g. "train-arrival time of the train". Defaults
                to False.

            Other optional parameters are the column names:
                desc_mapping_column (str, optional): Defaults to "description_mapping".
                diag_context_column (str, optional): Defaults to "dialogue_context".
                turn_domain_column (str, optional): Defaults to "turn_domain".
                slot_mapping_column (str, optional): Defaults to "slot_mapping".
                intent_mapping_column (str, optional): Defaults to "intent_mapping".
                user_utt_column (str, optional): Defaults to "user_utt".
                user_act_column (str, optional): Defaults to "user_acts".
                sys_utt_column (str, optional): Defaults to "sys_utt".
                sys_act_column (str, optional): Defaults to "sys_acts".
        """
        super().__init__(
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            ignore_pad_token_for_loss=ignore_pad_token_for_loss,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
            source_column=source_column,
            target_column=target_column,
            input_prefix=input_prefix,
            **kwargs,
        )

        self.delimiter: str = delimiter
        self.domain_in_desc: bool = domain_in_desc
        self.desc_mapping_column: str = desc_mapping_column
        self.diag_context_column: str = diag_context_column
        self.turn_domain_column: str = turn_domain_column
        self.slot_mapping_column: str = slot_mapping_column
        self.intent_mapping_column: str = intent_mapping_column
        self.user_utt_column: str = user_utt_column
        self.user_act_column: str = user_act_column
        self.sys_utt_column: str = sys_utt_column
        self.sys_act_column: str = sys_act_column

        self._acts_cache: Optional[D3STPreprocessor.KSTMapping] = None
        self._kst_table: Optional[D3STPreprocessor.KSTMapping] = None
        # whether to sample KSTs from the corpus
        self._sample_corpus: bool = False
        # whether to sample KSTs from a user provided table,
        # have precedence over self._sample_corpus
        self._sample_table: bool = False
        self._augment_style: str = D3STPreprocessor.AugmentStyle.DA
        self._iterative_decoding: bool = False

    @staticmethod
    def _extract_cat_choices(description: str) -> str:
        """Extract the description substring containing the possible values."""
        match = re.search(r".*?(\s?\d+\w\).*)", description)

        if match is not None:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_domain(description: str) -> str:
        """Extact the domain from description.

        Domain could be prepended to MultiWOZ descriptions, e.g.
            restaurant-number of people booking the restaurant
            train-arrival time of the train
        """
        # the regex matches against some word characters followed by -, at the start of
        # the string, which is then followed by some other word charactors
        # e.g. (hotel-)the name of the hotel
        match = re.search(r"^\w*-(?=\w*)", description)

        if match is None:
            raise RuntimeError(
                "Domain cannot be extracted, check if domains are in descriptions."
            )

        return match.group(0)

    @staticmethod
    def _extract_question_text(utterance: str) -> str:
        """Extract the question only.

        Example of an utterance from the SGD dataset:
            yes, that's right, what is the address of the restaurant?
                -> what is the address of the restaurant
        """
        match = re.search(r"(.*[.,?!]\s)*(and\s)?(.*)[?.]?", utterance)

        if match is None:
            raise RuntimeError(f"Cannot extract question from {utterance}")
        return match.group(3).strip(" .?")

    def _sample_turn_from_corpus(
        self,
        domain: str,
        value_type: int,
        slot_or_intent_name: str,
    ) -> str:
        """Sample a KST from the corpus for the given slot or intent.

        Notes:
            The random sampling uses the _acts_cache. See self._build_acts_cache method.
        """
        sampled_example_idx, sampled_example_speaker = random.choice(
            self._acts_cache[value_type][domain][slot_or_intent_name]
        )
        utt_col_name = (
            self.sys_utt_column
            if sampled_example_speaker == Speaker.SYSTEM
            else self.user_utt_column
        )
        return self._extract_question_text(
            self._dataset[sampled_example_idx][utt_col_name]
        )

    @PipelineMixin.op()
    def _do_json_loads(self, examples: Batch) -> Batch:
        examples[self.desc_mapping_column] = list(
            map(json.loads, examples[self.desc_mapping_column])
        )
        examples[self.slot_mapping_column] = list(
            map(json.loads, examples[self.slot_mapping_column])
        )
        examples[self.intent_mapping_column] = list(
            map(json.loads, examples[self.intent_mapping_column])
        )
        return examples

    @PipelineMixin.op(condition="_sample_corpus")
    def _do_sample_corpus(
        self, results_from__do_json_loads: Batch
    ) -> list[dict[str, str]]:
        """Replace the descriptions in description_mapping with sampled
        knowledge-seeking turns from the corpus."""
        if self._acts_cache is None:
            raise RuntimeError("No acts cache")

        new_desc_mappings = []
        for desc_mapping, slot_mapping, intent_mapping, domain in zip(
            results_from__do_json_loads[self.desc_mapping_column],
            results_from__do_json_loads[self.slot_mapping_column],
            results_from__do_json_loads[self.intent_mapping_column],
            results_from__do_json_loads[self.turn_domain_column],
        ):
            new_desc_mapping = {}
            for desc_idx, desc in desc_mapping.items():
                if desc_idx in slot_mapping:
                    slot_or_intent_name = slot_mapping[desc_idx]
                    value_type = ValueType.SLOT
                else:
                    slot_or_intent_name = intent_mapping[desc_idx]
                    value_type = ValueType.INTENT

                # knowledge-seeking turns available to use as alternative
                # descriptions
                if slot_or_intent_name in self._acts_cache[value_type][domain]:
                    new_desc_mapping[desc_idx] = self._sample_turn_from_corpus(
                        domain=domain,
                        value_type=value_type,
                        slot_or_intent_name=slot_or_intent_name,
                    )
            new_desc_mappings.append(new_desc_mapping)
        return new_desc_mappings

    @PipelineMixin.op(condition="_sample_table")
    def _do_sample_table(
        self, results_from__do_json_loads: Batch
    ) -> list[dict[str, str]]:
        if self._kst_table is None:
            raise RuntimeError("No KST table")

        new_desc_mappings = []
        for desc_mapping, slot_mapping, intent_mapping, domain in zip(
            results_from__do_json_loads[self.desc_mapping_column],
            results_from__do_json_loads[self.slot_mapping_column],
            results_from__do_json_loads[self.intent_mapping_column],
            results_from__do_json_loads[self.turn_domain_column],
        ):
            new_desc_mapping = {}
            for desc_idx, desc in desc_mapping.items():
                if desc_idx in slot_mapping:
                    slot_or_intent_name = slot_mapping[desc_idx]
                    value_type = ValueType.SLOT
                else:
                    slot_or_intent_name = intent_mapping[desc_idx]
                    value_type = ValueType.INTENT

                if (
                    value_type in self._kst_table
                    and domain in self._kst_table[value_type]
                    and slot_or_intent_name in self._kst_table[value_type][domain]
                ):
                    new_desc_mapping[desc_idx] = random.choice(
                        self._kst_table[value_type][domain][slot_or_intent_name]
                    )
            new_desc_mappings.append(new_desc_mapping)
        return new_desc_mappings

    @PipelineMixin.op()
    def _do_combine_kst(
        self,
        results_from__do_json_loads: Batch,
        results_from__do_sample_corpus: list[dict] | PipelineMixin.StepSkipped,
        results_from__do_sample_table: list[dict] | PipelineMixin.StepSkipped,
    ) -> Batch:
        if (
            results_from__do_sample_corpus is PipelineMixin.StepSkipped
            and results_from__do_sample_table is PipelineMixin.StepSkipped
        ):  # not adding KSTs
            return results_from__do_json_loads

        sampled_kst_mappings = (
            results_from__do_sample_corpus
            if results_from__do_sample_corpus is not PipelineMixin.StepSkipped
            else results_from__do_sample_table
        )

        new_desc_mappings = []
        for desc_mapping, sampled_kst_mapping, slot_mapping, intent_mapping in zip(
            results_from__do_json_loads[self.desc_mapping_column],
            sampled_kst_mappings,
            results_from__do_json_loads[self.slot_mapping_column],
            results_from__do_json_loads[self.intent_mapping_column],
        ):
            new_desc_mapping = {}
            for desc_idx, desc in desc_mapping.items():
                if desc_idx in slot_mapping:
                    slot_or_intent_name = slot_mapping[desc_idx]
                else:
                    slot_or_intent_name = intent_mapping[desc_idx]

                if self.domain_in_desc:
                    domain_name = self._extract_domain(desc)
                else:
                    domain_name = ""

                cat_choices = self._extract_cat_choices(desc)
                if len(cat_choices) > 0:
                    cat_choices = f" {cat_choices}"

                if self._augment_style in (
                    D3STPreprocessor.AugmentStyle.REPLACE,
                    D3STPreprocessor.AugmentStyle.DA,
                ):
                    if desc_idx in sampled_kst_mapping:
                        new_desc_mapping[desc_idx] = (
                            domain_name + sampled_kst_mapping[desc_idx] + cat_choices
                        )
                    else:
                        # default to schema descriptions if no KSTs were sampled
                        new_desc_mapping[desc_idx] = (
                            domain_name + desc_mapping[desc_idx] + cat_choices
                        )
                    continue

                # when doing grounding with turns or turns and slots, categorical
                # options always come last and domain name (only relavent for multiwoz)
                # always come first
                # they are only shown once in the description string
                # temporarily remove categorical option from description
                desc = desc.replace(cat_choices, "", 1)
                new_desc_args = [
                    # temporarily remove domain name from description
                    desc.replace(domain_name, "", 1)
                    if self.domain_in_desc
                    else desc
                ]

                kst_avialable = desc_idx in sampled_kst_mapping
                if self._augment_style in (
                    D3STPreprocessor.AugmentStyle.TURN,
                    D3STPreprocessor.AugmentStyle.TURNSLOT,
                ):
                    if kst_avialable:
                        new_desc_args.append(sampled_kst_mapping[desc_idx])

                if self._augment_style == D3STPreprocessor.AugmentStyle.TURNSLOT:
                    # transform slot/intent name into natural language
                    new_desc_args.append(
                        re.sub(
                            r"([A-Z])", r" \1", slot_or_intent_name.replace("_", " ")
                        )
                        .strip()
                        .lower()
                    )

                random.shuffle(new_desc_args)
                new_desc_mapping[desc_idx] = (
                    domain_name + "; ".join(new_desc_args) + cat_choices
                )
                if not kst_avialable:
                    new_desc_mapping[desc_idx] += ";"

            new_desc_mappings.append(new_desc_mapping)

        if self._augment_style == D3STPreprocessor.AugmentStyle.DA:
            # if using data augmentation, augment the batch with new description
            # mappings, and appropriate feature values
            for column_name, feature_value in results_from__do_json_loads.items():
                if column_name == self.desc_mapping_column:
                    results_from__do_json_loads[self.desc_mapping_column].extend(
                        new_desc_mappings
                    )
                else:
                    results_from__do_json_loads[column_name].extend(feature_value)
        else:
            # if not using data augmentation, replace the description mapping
            results_from__do_json_loads[self.desc_mapping_column] = new_desc_mappings
        return results_from__do_json_loads

    @PipelineMixin.op(condition="_iterative_decoding")
    def _do_split_examples(self, results_from__do_combine_kst: Batch) -> Batch:
        """Split the examples into multiple, where each contains only description to
        one slot/intent.

        Used for iterative decoding.
        """
        split_examples = defaultdict(list)
        for idx in range(len(results_from__do_combine_kst[self.desc_mapping_column])):
            desc_mappings = json.loads(
                results_from__do_combine_kst[self.desc_mapping_column][idx]
            )
            for desc_idx, desc in desc_mappings.items():
                split_examples[self.desc_mapping_column].append(
                    json.dumps(
                        {
                            # reset slot index to "0" and intent index to "i0"
                            "0"
                            if "i" not in desc_idx
                            else "i0": re.sub(r"\d+(?=\w\))", "0", desc)
                            # the regex matches the categorical value indices, e.g. 3a)
                            # these indices are reset to 0a), 0b), etc
                        }
                    )
                )
                for key in results_from__do_combine_kst.keys():
                    if key == self.desc_mapping_column:
                        continue
                    split_examples[key].append(results_from__do_combine_kst[key][idx])
        return split_examples

    @PipelineMixin.op(condition="tokenize")
    def _do_join_description_mapping(
        self,
        results_from__do_combine_kst: Batch,
        results_from__do_split_examples: Batch | PipelineMixin.StepSkipped,
    ) -> Batch:
        """Join the slot/intent indices with descriptions, and concatenate the
        dialogue context to form the prompt."""
        if results_from__do_split_examples is not PipelineMixin.StepSkipped:
            # using iterative decoding
            processed_examples = results_from__do_split_examples
        else:
            # not using iterative decoding
            processed_examples = results_from__do_combine_kst

        descs = map(
            lambda desc_mapping: " ".join(
                [f"{idx}{self.delimiter}{desc}" for idx, desc in desc_mapping.items()]
            ),
            processed_examples[self.desc_mapping_column],
        )

        return {
            self.source_column: list(
                map(
                    lambda desc_diag_context: " ".join(desc_diag_context),
                    zip(descs, processed_examples[self.diag_context_column]),
                )
            ),
            self.target_column: processed_examples[self.target_column],
        }

    @PipelineMixin.op()
    def _do_json_dumps(
        self,
        results_from__do_combine_kst: Batch,
        results_from__do_split_examples: Batch | PipelineMixin.StepSkipped,
        results_from__do_join_description_mapping: Batch | PipelineMixin.StepSkipped,
    ):
        if results_from__do_join_description_mapping is not PipelineMixin.StepSkipped:
            return results_from__do_join_description_mapping

        if results_from__do_split_examples is not PipelineMixin.StepSkipped:
            examples = results_from__do_split_examples
        else:
            examples = results_from__do_combine_kst

        examples[self.desc_mapping_column] = list(
            map(json.dumps, examples[self.desc_mapping_column])
        )
        examples[self.slot_mapping_column] = list(
            map(json.dumps, examples[self.slot_mapping_column])
        )
        examples[self.intent_mapping_column] = list(
            map(json.dumps, examples[self.intent_mapping_column])
        )
        return examples

    def _cache_speaker_acts(
        self,
        idx: int,
        act_info: dict,
        domain: str,
        speaker: int,
        omit_confirmation_turns: bool,
    ) -> None:
        """Cache speaker action info.

        act_info is a mapping from action to list of action parameters:
            - For actions on slots, the action maps to a list of slot names
                E.g. `{"REQUEST": ["restaurant_name", "location"]}`
            - For actions on slots with values, the values are also included with the
              slot names, inside brackets.
                E.g. `{"INFORM": ["location(San Jose)", "restaurant_name(Sino)"]}`
            - For actions on intents, the action maps to a list of intent names
                E.g. `{"OFFER_INTENT": ["ReserveCar"]}`

        Turns annotated with REQUEST(slot=value) are confirmations, these turns are not
        sampled if omit_confirmation_turns is True.
        """
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
                self._acts_cache[ValueType.SLOT][domain][act_param].append(
                    (idx, speaker)
                )
            elif act in ("INFORM_INTENT", "OFFER_INTENT"):
                self._acts_cache[ValueType.INTENT][domain][act_param].append(
                    (idx, speaker)
                )

    def _build_acts_cache(
        self, dataset: Dataset, omit_confirmation_turns: bool = True
    ) -> None:
        """Build a hashmap for actions.

        The cache has the form:
            cache[slot_or_intent][domain][slot_intent_name]
                -> list [tuple[row_index, speaker]]
        slot_or_intent is an integer denoting slot or intent, see ValueType class.

        Args:
            dataset (Dataset): the dataset to process
            omit_confirmation_turns (bool): Whether to skip sampling KSTs that are
                confirmations. Defaults to True.
        """
        # don't use Dataset.map to prevent potential GIL issue when having side-effects
        # (i.e. mutating state of a single hash table)
        self._acts_cache = {
            ValueType.SLOT: defaultdict(lambda: defaultdict(list)),
            ValueType.INTENT: defaultdict(lambda: defaultdict(list)),
        }

        logger.info("Building dialogue acts cache...")
        for idx, example in tqdm(enumerate(dataset)):
            domain = example[self.turn_domain_column]
            sys_acts = json.loads(example[self.sys_act_column])
            user_acts = json.loads(example[self.user_act_column])

            if len(sys_acts) == 1:
                self._cache_speaker_acts(
                    idx=idx,
                    act_info=sys_acts,
                    domain=domain,
                    speaker=Speaker.SYSTEM,
                    omit_confirmation_turns=omit_confirmation_turns,
                )

            if len(user_acts) == 1:
                self._cache_speaker_acts(
                    idx=idx,
                    act_info=user_acts,
                    domain=domain,
                    speaker=Speaker.USER,
                    omit_confirmation_turns=omit_confirmation_turns,
                )

    def process(
        self,
        dataset: Dataset,
        *,
        augment_style: str = AugmentStyle.NONE,
        kst_table: Optional[KSTMapping] = None,
        iterative_decoding: bool = False,
        omit_confirmation_turns: bool = True,
        **process_kwargs,
    ) -> Dataset:
        """

        Args:
            dataset (Dataset): Raw dataset to be processed.
            augment_style (str, optional): How the sampled knowledge-seeking
                turns are added to the prompt. Default to 0, i.e. data augmentation.
            kst_table (KSTMapping, optional): User specified KST table to sample from
                if not sampling from the corpus.
            iterative_decoding (bool, optional): Whether to split the examples so each
                contains only description to one slot/intent. Defaults to False.
            omit_confirmation_turns (bool, optional): Whether to skip sampling KSTs
                that are confirmations. Defaults to True.

            Other keyword arguments accepted by the process method of the base
            Preprocessor class.

        Raises:
            ValueError: At least one of sample_schema and sample_dialogue must be True.

        Returns:
            Dataset: Processed dataset.
        """
        self._iterative_decoding = iterative_decoding
        self._augment_style = augment_style
        if kst_table is not None:
            self._kst_table = {
                ValueType.SLOT: kst_table["slots"],
                ValueType.INTENT: kst_table["intents"],
            }
        self._sample_corpus = (
            self._augment_style != D3STPreprocessor.AugmentStyle.NONE
            and kst_table is None
        )
        self._sample_table = (
            self._augment_style != D3STPreprocessor.AugmentStyle.NONE
            and kst_table is not None
        )

        if augment_style == D3STPreprocessor.AugmentStyle.NONE:
            logging.info("Using schema prompts only")
        else:
            logging.info(
                "Sampling knowledge-seeking turns from the"
                f" {'corpus' if self._sample_corpus else 'given table'},"
                f" combining with schema prompts using: {self._augment_style}"
            )
            if self._sample_corpus:
                self._build_acts_cache(
                    dataset, omit_confirmation_turns=omit_confirmation_turns
                )

        processed_dataset = super().process(dataset, **process_kwargs)
        return processed_dataset


if __name__ == "__main__":
    # Below are code for testing

    from transformers import AutoTokenizer

    logging.getLogger().setLevel(logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
    preprocessor = D3STPreprocessor(
        tokenizer=tokenizer,
        load_from_cache_file=False,
        domain_in_desc=False,
        max_source_length=1024,
    )

    #  with open('data/external/kst_table.json', 'r') as f:
    #  kst_table = json.load(f)

    processed_dataset = preprocessor.process(
        load_dataset(
            "json",
            # data_files="data/processed/turn/original/test/version_1/data.json",
            data_files="data/processed/original/dev/version_9/data.json",
            # data_files="data/processed/multiwoz/train/version_9/data.json",
            field="data",
            split="train",
        ),
        augment_style=D3STPreprocessor.AugmentStyle.DA,
        #  kst_table=kst_table,
        omit_confirmation_turns=True,
        discard_truncated_examples=True,
    )

    processed_dataset.to_json("runtime_preprocessed_dataset.json")
