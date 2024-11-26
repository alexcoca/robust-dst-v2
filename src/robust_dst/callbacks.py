import logging
import os

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    is_torch_tpu_available,
    EarlyStoppingCallback,
)
from transformers.integrations import WandbCallback

from robust_dst.cli import CustomSeq2SeqTrainingArguments

logger = logging.getLogger(__name__)


class CacheManagerCallback(TrainerCallback):
    def on_evaluate(
        self,
        args: CustomSeq2SeqTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        torch.cuda.empty_cache()


class CustomWandbCallback(WandbCallback):
    """Allows logging of the learning rate more frequently early on
    in training to check the scheduling is correct"""

    def __init__(self):
        super().__init__()
        self.lr_scheduler = None

    def setup(self, args: CustomSeq2SeqTrainingArguments, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.

        One can subclass and override this method to customize the setup if needed.
        Find more information [here](https://docs.wandb.ai/integrations/huggingface).
        You can also override the following environment variables:

        Environment:
            WANDB_LOG_MODEL (`bool`, *optional*, defaults to `False`):
                Whether or not to log model as artifact at the end of training.
                Use along with *TrainingArguments.load_best_model_at_end* to upload
                best model.
            WANDB_WATCH (`str`, *optional* defaults to `"gradients"`):
                Can be `"gradients"`, `"all"` or `"false"`. Set to `"false"` to
                disable gradient logging or `"all"` to log gradients and parameters.
            WANDB_PROJECT (`str`, *optional*, defaults to `"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (`bool`, *optional*, defaults to `False`):
                Whether or not to disable wandb entirely.
                Set *WANDB_DISABLED=true* to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                "Automatic Weights & Biases logging enabled, to disable set"
                ' os.environ["WANDB_DISABLED"] = "true"'
            )
            logger.info(
                f"The following tags will be added to this run: {args.wandb_tags}"
            )
            logger.info(
                f"The run will be logged to the following entity: {args.wandb_entity}"
            )
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                run_name = trial_name
                init_args["group"] = args.run_name
            else:
                run_name = args.run_name
            if args.wandb_project is not None:
                init_args["entity"] = args.wandb_entity
            if args.wandb_tags is not None:
                init_args["tags"] = args.wandb_tags
            wandb_init_args = {**init_args, "name": run_name}
            if args.resume_from_checkpoint:
                wandb_api = self._wandb.Api()
                runs = wandb_api.runs(
                    path=f"{args.wandb_entity}/{args.wandb_project}",
                    filters={
                        "config.run_name": f"{args.run_name}",
                        "tags": {"$in": args.wandb_tags},
                    },
                )
                for r in runs:
                    logger.info(f"Run: {r.id}, {r.name}")
                if runs:
                    run = runs[0]
                    wandb_init_args["resume"] = "must"
                    wandb_init_args["id"] = run.id
                else:
                    logger.info("No runs with the specified tags were found!")
            logger.info(f"wandb init args: {wandb_init_args}")
            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"), **wandb_init_args
                )
            else:
                self._wandb.finish()
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"), **wandb_init_args
                )

            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric(
                    "*", step_metric="train/global_step", step_sync=True
                )

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model,
                    log=os.getenv("WANDB_WATCH", "gradients"),
                    log_freq=max(100, args.logging_steps),
                )
        self.lr_scheduler = kwargs.get("lr_scheduler", None)

    def on_step_end(
        self,
        args: CustomSeq2SeqTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if args.do_predict or self.lr_scheduler is None:
            logger.warning(
                "Learning rate will not be logged as you are either running in"
                " inference mode or there is no scheduler."
            )
            return
        if not self._initialized:
            logger.info("Initializing LR logger callback")
            self.setup(args, state, control, **kwargs)
        if state.is_world_process_zero:
            lr_log_freq = args.lr_log_freq
            lr_freq_log_until = args.log_lr_freq_limit
            current_step = state.global_step
            if current_step % lr_log_freq == 0 and current_step < lr_freq_log_until:
                try:
                    last_lr = self.lr_scheduler.get_last_lr()[0]
                except AssertionError as e:
                    if "need to call step" in str(e):
                        logger.warning(
                            "tried to get lr value before scheduler/optimizer started"
                            " stepping, returning lr=0"
                        )
                        last_lr = 0
                    else:
                        raise
                self._wandb.log(
                    {
                        "learning_rate": last_lr,
                        "step": current_step,
                        "train/global_step": current_step,
                    }
                )


class JGAEarlyStoppingCallback(EarlyStoppingCallback):

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return
        logger.info(f"{metric_to_check} at step {state.global_step}: {metric_value:4f}")
        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            logger.info(
                f"Stopping training as {metric_to_check} has not improved "
                f"in {self.early_stopping_patience} evaluations."
            )
            control.should_training_stop = True

