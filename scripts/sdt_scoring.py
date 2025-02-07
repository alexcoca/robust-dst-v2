# Work by Ankit Bhattarai as part of Conversational AI Development project, modified by Alex Coca
import json
import logging
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import click
from datasets import load_dataset
from omegaconf import OmegaConf
from robust_dst.evaluation import get_metrics
from robust_dst.parser import SDTParser
from robust_dst.scoring_utils import flatten_metrics_dict, setup_sgd_evaluation

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--demonstration_id",
    type=click.Choice(["v0", "v1", "v2", "v3", "v4"]),
    default=None,
)
@click.option("--data_type", type=click.Choice(["dev", "test"]), default="dev")
@click.option("--predictions_file", type=click.Path(exists=True))
@click.option("--refs_file", type=click.Path(exists=True))
@click.option("--data_format", default="google")
@click.option("--output_file", required=False, type=click.Path(exists=False))
@click.option("--output_file_name", default="metrics.json", required=False)
@click.option("--save_files", default=False, required=False)
@click.option("--hyp_dir", type=click.Path(exists=True), required=False)
@click.option("--max_samples", default=None, required=False)
@click.option("--version", default=1, required=False)
def main(
    demonstration_id,
    data_type,
    predictions_file,
    refs_file,
    data_format,
    output_file,
    output_file_name,
    save_files,
    hyp_dir,
    max_samples,
    version,
):
    data_args = SimpleNamespace(
        max_eval_samples=max_samples,
        max_predict_samples=max_samples,
        validation_file=f"data/processed/SGD_SDT/{demonstration_id}/dev/version_{version}/data.json",
        test_file=f"data/processed/SGD_SDT/{demonstration_id}/test/version_{version}/data.json",
        validation_ref_dir="data/raw/original/dev",
        test_ref_dir="data/raw/original/test",
        validation_template_dir="data/interim/blank_dialogue_templates/original/dev",
        test_template_dir="data/interim/blank_dialogue_templates/original/test",

    )
    preprocessing_configs = {}  # type: dict[str, DictConfig]
    data_files = {}  # type: dict[str, str]
    extension = "json"
    val_file, test_file = data_args.validation_file, data_args.test_file
    preprocessing_configs["validation"] = OmegaConf.load(
        Path(val_file).parent.joinpath("preprocessing_config.yaml")
    )
    preprocessing_configs["test"] = OmegaConf.load(
        Path(test_file).parent.joinpath("preprocessing_config.yaml")
    )

    if val_file is not None:
        data_files["validation"] = val_file
        extension = val_file.split(".")[-1]
        preprocessing_configs["validation"] = OmegaConf.load(
            Path(val_file).parent.joinpath("preprocessing_config.yaml")
        )
    if test_file is not None:
        data_files["test"] = test_file
        extension = test_file.split(".")[-1]
        preprocessing_configs["test"] = OmegaConf.load(
            Path(test_file).parent.joinpath("preprocessing_config.yaml")
        )

    raw_datasets = load_dataset(
        extension, data_files=data_files, cache_dir="cache", field="data"
    )

    raw_preprocessed_refs = {}  # type: dict[str, list[dict]]
    for split in raw_datasets:
        if split == "train":
            continue
        raw_preprocessed_refs[split] = raw_datasets.data[split].table.to_pylist()
        if split == "test" and max_samples is not None:
            raw_preprocessed_refs[split] = raw_preprocessed_refs[split][
                : max_samples
            ]
        if split == "validation" and max_samples is not None:
            raw_preprocessed_refs[split] = raw_preprocessed_refs[split][
                : max_samples
            ]
    input_dtype = "validation" if data_type == "dev" else data_type
    parser_inputs, sgd_evaluator_inputs = setup_sgd_evaluation(
        data_args, preprocessing_configs, raw_preprocessed_refs, input_dtype
    )

    with open(refs_file, "r") as f:
        gt = json.load(f)
    gt = gt["data"]
    with open(predictions_file, "r") as f:
        hyps = f.readlines()
    hyps = [item.replace(";", " ").strip() for item in hyps]

    parser = SDTParser(
        data_format=data_format,
        template_dir=f"data/interim/blank_dialogue_templates/original/{data_type}",
        schema_path=f"data/raw/original/{data_type}/schema.json",
        files_to_parse=None,
        dialogues_to_parse=None,
    )
    file_to_hyp_dials = parser.convert_to_sgd_format(
        preprocessed_refs=gt,
        predictions=hyps,
    )
    dataset_hyp = {
        dial["dialogue_id"]: dial for dial in chain(*file_to_hyp_dials.values())
    }

    all_metrics_aggregate, frame_metrics = get_metrics(
        dataset_ref=sgd_evaluator_inputs["dataset_ref"],
        dataset_hyp=dataset_hyp,
        service_schemas=sgd_evaluator_inputs["eval_services"],
        in_domain_services=sgd_evaluator_inputs["in_domain_services"],
    )
    all_metrics_aggregate = flatten_metrics_dict(all_metrics_aggregate)
    if output_file is None:
        output_file = Path(predictions_file).parent.joinpath(output_file_name)
    with open(output_file, "w") as f:
        json.dump(all_metrics_aggregate, f, indent=4)
    if save_files:
        hyp_dir = Path(hyp_dir)
        # for fname, this_file_dials in file_to_hyp_dials.items():
        #     current_step_sgd_format_predictions_pth = hyp_dir.joinpath(fname)
        #     logger.info(
        #         "Saving formatted predictions at"
        #         f" {current_step_sgd_format_predictions_pth}"
        #     )
        #     with open(current_step_sgd_format_predictions_pth, "w") as f:
        #         json.dump(this_file_dials, f, indent=2)
        with open(hyp_dir.joinpath("metrics_and_dialogues.json")) as f:
            json.dump(frame_metrics, f, indent=2)

if __name__ == "__main__":
    main()