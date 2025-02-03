from __future__ import annotations

import logging
import math
import time
from typing import Optional

from datasets import Dataset
from transformers import Seq2SeqTrainer, is_torch_tpu_available
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import PredictionOutput, speed_metrics

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm  # noqa
    import torch_xla.debug.metrics as met  # noqa

from robust_dst.callbacks import CustomWandbCallback
from robust_dst.scoring_utils import (
    infer_step,
    save_evaluator_outputs,
    setup_evaluator_output_dirs,
)

logger = logging.getLogger(__name__)


class CustomTrainer(Seq2SeqTrainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> dict[str, float]:
        """Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute
        metrics, as they are task-dependent (pass it to the init `compute_metrics`
        argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it
                is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. It must implement
                the `__len__` method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary)
                that should be ignored when gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix.
                For example the metrics "bleu" will be named "eval_bleu" if the
                prefix is "eval" (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate
                method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting
                with the generate method. 1 means no beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.arch that will be used when
                predicting with the generate method. 1 means no beam search.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics
            computed from the predictions. The dictionary also contains the epoch
            number which comes from the training state.
        """
        gen_kwargs = gen_kwargs.copy()
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we
            # defer to self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # save evaluation artefacts
        current_step = self.state.global_step
        if self.args.predict_with_generate:
            hyps_dir, metrics_dir = setup_evaluator_output_dirs(
                self.args, "dev", current_step
            )
            outputs_to_save = (
                output.metrics.pop(f"{metric_key_prefix}_all_metrics_aggregate"),
                output.metrics.pop(f"{metric_key_prefix}_file_to_hyp_dials"),
                output.metrics.pop(f"{metric_key_prefix}_raw_predictions"),
            )
            save_evaluator_outputs(hyps_dir, metrics_dir, *outputs_to_save, current_step)
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile,
            # execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain
        labels. In that case, this method will also return metrics, like in
        `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`,
                columns not accepted by the `model.forward()` method are automatically
                removed. Has to implement the method `__len__`
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that
                should be ignored when gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example
                the metrics "bleu" will be named "test_bleu" if the prefix is
                "test" (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate
                method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with
                the generate method. 1 means no beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence length (for instance
        because you're doing dynamic padding in a token classification task) the
        predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:
            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset
              contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of
              metrics (if the dataset contained labels).
        """
        gen_kwargs = gen_kwargs.copy()
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            test_dataloader,
            description="Prediction",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        # save evaluation artefacts
        hyps_dir, metrics_dir = setup_evaluator_output_dirs(self.args, "test")
        outputs_to_save = (
            output.metrics.pop(f"{metric_key_prefix}_all_metrics_aggregate"),
            output.metrics.pop(f"{metric_key_prefix}_file_to_hyp_dials"),
            output.metrics.pop(f"{metric_key_prefix}_raw_predictions"),
        )  # type: tuple[dict, dict[str, list[dict]], list[str]]
        save_evaluator_outputs(
            hyps_dir, metrics_dir, *outputs_to_save, infer_step(self.args)
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        self.log(output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return PredictionOutput(
            predictions=output.predictions,
            label_ids=output.label_ids,
            metrics=output.metrics,
        )

    def log_config_to_wandb(self, config: dict):
        if self.state.is_world_process_zero:
            for c in self.callback_handler.callbacks:
                if isinstance(c, CustomWandbCallback):
                    break
            else:
                logger.info(
                    "``wandb`` logging is not enabled so can't log experiment"
                    " configuration!"
                )
                return
            # training args are saved by huggingface automatically
            to_log = {
                "data": config.pop("data", {}),
                "data_args": config.pop("training_args", {}),
                "model_args": config.pop("model_args", {}),
            }
            wandb = c._wandb  # noqa
            wandb.config.update(to_log, allow_val_change=True)
