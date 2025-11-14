# Work by Ankit Bhattarai as part of Conversational AI Development project
import json
import logging
import pathlib
import re
from collections import OrderedDict

import click
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


## Note that this file will only work with the delimiters in place
example_slots_regex = r"\[slots\](.*)\[context\]"
individual_slots_regex = r"(\w+)=([^;]+)"
mwoz_individual_slots_regex = r"(\w+-\w+)=(\S.*?(?= \w+-|$))"


def load_data(data_path, num_samples=-1, start_index=0):
    with open(data_path, "r") as f:
        data = json.load(f)
    data = data["data"]
    if num_samples == -1:
        return data
    else:
        return data[start_index : start_index + num_samples]


def load_model_tokenizer(model_name_or_path):
    model = T5ForConditionalGeneration.from_pretrained(
        model_name_or_path, cache_dir="cache"
    )
    tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path, cache_dir="cache")
    return model, tokenizer


def tokenize_slots(slot_text, tokenizer, not_first_slot=True):
    if not_first_slot:
        slot_text = "=" + slot_text
    tokenized = tokenizer(slot_text, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].flatten()
    if not_first_slot:
        tokenized = tokenized[1:]
    return tokenized


def tokenize_constraints(constraint_text, tokenizer):
    constraint_text = "=" + constraint_text
    tokenized = tokenizer(
        constraint_text, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].flatten()
    return tokenized[1:].item()


def get_constraints(slots, categorical_slots_mapping, tokenizer):
    logger.info("Tokenizing constraints")
    all_constraints = []  # Constraints for each example stored as a list
    tokenized_dict = {}  # A lookup table of slot / categorical values already tokenized
    for turn_slots, mapping in tqdm(zip(slots, categorical_slots_mapping)):
        constraint = OrderedDict()
        mapping = eval(mapping) if mapping else {}
        for i, slot in enumerate(turn_slots):
            not_first = bool(i)
            not_first = False  # Check this out!
            slot_tokenized = tokenized_dict.setdefault(
                (slot, not_first), tokenize_slots(slot, tokenizer, not_first)
            )
            if slot in mapping:
                slot_constraints = []
                options = list(mapping[slot].keys())
                for option in options:
                    option_tokenized = tokenized_dict.setdefault(
                        option, tokenize_constraints(option, tokenizer)
                    )
                    slot_constraints.append(option_tokenized)
            else:
                slot_constraints = None
            constraint[slot_tokenized] = slot_constraints
        all_constraints.append(constraint)
    logger.info("Tokenizing constraints done")
    return all_constraints


def get_sorted_data(data, tokenizer):
    source = [item["source"] for item in data]
    source_ids = tokenizer(source)["input_ids"]
    len_source = np.array([len(src) for src in source_ids])
    sorting_index = np.argsort(len_source)[::-1]  # Sorting by reverse order
    source = [source[i] for i in sorting_index]
    categorical_slots_mapping = [
        data[i]["categorical_slots_mapping"] for i in sorting_index
    ]
    reference_slot_values = [re.findall(example_slots_regex, src)[0] for src in source]

    slots_regex = individual_slots_regex
    slots = [
        [slot for slot, _ in re.findall(slots_regex, ref.replace("[SEP]", ";"))]
        for ref in reference_slot_values
    ]
    source_ids = [source_ids[i] for i in sorting_index]
    return source, source_ids, categorical_slots_mapping, slots, sorting_index


def dynamic_batches(
    source,
    tokenizer,
    start_batch_size=128,
    max_target_length=140,
    safety_factor=0.95,
    fixed_batch_size=False,
):
    batches = []
    len_source = len(tokenizer(source[0])["input_ids"])
    start_max_source_size = np.max(len_source)
    current_max_length = start_max_source_size
    batch_size = start_batch_size
    i = 0
    while True:
        if len(source) == 0:
            break
        if i != 0:
            n = start_batch_size * (start_max_source_size + max_target_length)
            start_item = source[0]
            current_max_length = len(tokenizer(start_item)["input_ids"])
            n /= current_max_length + max_target_length
            batch_size = (
                np.maximum(int(n * safety_factor), 1)
                if not fixed_batch_size
                else start_batch_size
            )
        log_text = f"Batch: {i}, Batch size: {batch_size}, Current max length: {current_max_length}"
        logger.info(log_text)
        current_slice = source[:batch_size]
        tokenized = tokenizer(current_slice, padding=True, return_tensors="pt")
        batches.append(tokenized)
        source = source[batch_size:]
        i += 1
    return batches



@torch.no_grad()
def decode(
    model, tokenizer, batches, device, max_target_length=140
):
    outputs = []
    model = model.to(device)
    model.eval()
    for batch in tqdm(batches):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        model_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_target_length,
        )
        decoded = tokenizer.batch_decode(model_output, skip_special_tokens=False)
        outputs.extend(decoded)
    return outputs


def reorder_outputs(outputs, sorting_index):
    reverse_mapping = {v: i for i, v in enumerate(sorting_index)}
    sorted_outputs = [outputs[reverse_mapping[i]] for i in range(len(sorting_index))]
    sorted_outputs = [
        item.replace("<pad>", "").replace("</s>", "").strip() for item in sorted_outputs
    ]
    return sorted_outputs


@click.command()
@click.option("--test_file", type=click.Path(exists=True), required=True)
@click.option("--model_name_or_path", type=str, required=True)
@click.option("--output_dir", type=click.Path(exists=True), required=True)
@click.option("--start_batch_size", type=int, default=128, required=False)
@click.option("--max_target_length", type=int, default=140, required=False)
@click.option("--safety_factor", type=float, default=0.97, required=False)
@click.option(
    "--num_samples",
    type=int,
    default=-1,
    required=False,
    help="Number of samples to run the inference on. If left as -1, it will run on the entire dataset",
)
@click.option(
    "--start_index",
    type=int,
    default=0,
    required=False,
    help="Start index for the dataset - good for debugging",
)
@click.option(
    "--fixed_batch_size",
    type=bool,
    default=False,
    required=False,
    help="If true, the batch size will be fixed at the start_batch_size",
)
def main(
    test_file: pathlib.Path,
    model_name_or_path: str,
    output_dir: pathlib.Path,
    start_batch_size: int,
    max_target_length: int,
    safety_factor: float,
    num_samples: int,
    start_index: int,
    fixed_batch_size: bool,
):
    logger.info(f"Running inference on {test_file} using model {model_name_or_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(test_file, num_samples, start_index)
    model, tokenizer = load_model_tokenizer(model_name_or_path)
    (
        source,
        source_ids,
        categorical_slots_mapping,
        slots,
        sorting_index,
    ) = get_sorted_data(data, tokenizer)
    batches = dynamic_batches(
        source,
        tokenizer,
        start_batch_size,
        max_target_length,
        safety_factor,
        fixed_batch_size,
    )
    decoded_outputs = decode(
        model, tokenizer, batches, device, max_target_length
    )
    reordered_outputs = reorder_outputs(decoded_outputs, sorting_index)
    output_file = pathlib.Path(output_dir).joinpath("generated_predictions.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(reordered_outputs))
    logger.info(f"Predictions saved at {output_file}")


if __name__ == '__main__':
    main()