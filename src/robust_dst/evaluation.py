import collections

import numpy as np
from absl import logging

from robust_dst import metrics
from robust_dst.utils import linear_thresholding

ALL_SERVICES = "#ALL_SERVICES"
SEEN_SERVICES = "#SEEN_SERVICES"
UNSEEN_SERVICES = "#UNSEEN_SERVICES"
PER_FRAME_OUTPUT_FILENAME = "metrics_and_dialogues.json"


def get_metrics(
    dataset_ref,
    dataset_hyp,
    service_schemas,
    in_domain_services,
    use_fuzzy_match=True,
    joint_acc_across_turn=False,
    fuzzy_threshold=0.8
):
    """Calculate the DSTC8 metrics.
    Args:
      joint_acc_across_turn:
      use_fuzzy_match:
      dataset_ref: The ground truth dataset represented as a dict mapping dialogue
        id to the corresponding dialogue.
      dataset_hyp: The predictions in the same format as `dataset_ref`.
      service_schemas: A dict mapping service name to the schema for the service.
      in_domain_services: The set of services which are present in the training
        set.
    Returns:
      A dict mapping a metric collection name to a dict containing the values
      for various metrics. Each metric collection aggregates the metrics across
      a specific set of frames in the dialogues.
    """
    # Metrics can be aggregated in various ways, eg over all dialogues, only for
    # dialogues containing unseen services or for dialogues corresponding to a
    # single service. This aggregation is done through metric_collections, which
    # is a dict mapping a collection name to a dict, which maps a metric to a list
    # of values for that metric. Each value in this list is the value taken by
    # the metric on a frame.
    metric_collections = collections.defaultdict(lambda: collections.defaultdict(list))

    # Ensure the dialogs in dataset_hyp also occur in dataset_ref.
    assert set(dataset_hyp.keys()).issubset(set(dataset_ref.keys()))
    logging.info(
        "len(dataset_hyp)=%d, len(dataset_ref)=%d", len(dataset_hyp), len(dataset_ref)
    )

    # Store metrics for every frame for debugging.
    per_frame_metric = {}
    # Store metrics for every intent in each dialogue for internal use
    per_intent_metrics = {}
    for dial_id, dial_hyp in dataset_hyp.items():
        dial_ref = dataset_ref[dial_id]

        if set(dial_ref["services"]) != set(dial_hyp["services"]):
            raise ValueError(
                "Set of services present in ground truth and predictions don't match "
                "for dialogue with id {}".format(dial_id)
            )
        joint_metrics = [
            metrics.JOINT_GOAL_ACCURACY,
            metrics.JOINT_CAT_ACCURACY,
            metrics.JOINT_NONCAT_ACCURACY,
        ]
        for turn_id, (turn_ref, turn_hyp) in enumerate(
            zip(dial_ref["turns"], dial_hyp["turns"])
        ):
            metric_collections_per_turn = collections.defaultdict(
                lambda: collections.defaultdict(lambda: 1.0)
            )
            if turn_ref["speaker"] != turn_hyp["speaker"]:
                raise ValueError(
                    "Speakers don't match in dialogue with id {}".format(dial_id)
                )

            # Skip system turns because metrics are only computed for user turns.
            if turn_ref["speaker"] != "USER":
                continue

            if turn_ref["utterance"] != turn_hyp["utterance"]:
                logging.info("Ref utt: %s", turn_ref["utterance"])
                logging.info("Hyp utt: %s", turn_hyp["utterance"])
                raise ValueError(
                    "Utterances don't match for dialogue with id {}".format(dial_id)
                )

            hyp_frames_by_service = {
                frame["service"]: frame for frame in turn_hyp["frames"]
            }

            # Calculate metrics for each frame in each user turn.
            for frame_ref in turn_ref["frames"]:
                service_name = frame_ref["service"]
                intent = frame_ref['state']['active_intent']
                if service_name not in hyp_frames_by_service:
                    raise ValueError(
                        "Frame for service {} not found in dialogue with id {}".format(
                            service_name, dial_id
                        )
                    )
                service = service_schemas[service_name]
                frame_hyp = hyp_frames_by_service[service_name]

                active_intent_acc = metrics.get_active_intent_accuracy(
                    frame_ref, frame_hyp
                )
                slot_tagging_f1_scores = metrics.get_slot_tagging_f1(
                    frame_ref, frame_hyp, turn_ref["utterance"], service
                )
                requested_slots_f1_scores = metrics.get_requested_slots_f1(
                    frame_ref, frame_hyp
                )
                goal_accuracy_dict = metrics.get_average_and_joint_goal_accuracy(
                    frame_ref, frame_hyp, service, use_fuzzy_match
                )

                frame_metric = {
                    metrics.ACTIVE_INTENT_ACCURACY: active_intent_acc,
                    metrics.REQUESTED_SLOTS_F1: requested_slots_f1_scores.f1,
                    metrics.REQUESTED_SLOTS_PRECISION: requested_slots_f1_scores.precision,  # noqa: E501
                    metrics.REQUESTED_SLOTS_RECALL: requested_slots_f1_scores.recall,
                }
                if slot_tagging_f1_scores is not None:
                    frame_metric[metrics.SLOT_TAGGING_F1] = slot_tagging_f1_scores.f1
                    frame_metric[
                        metrics.SLOT_TAGGING_PRECISION
                    ] = slot_tagging_f1_scores.precision
                    frame_metric[
                        metrics.SLOT_TAGGING_RECALL
                    ] = slot_tagging_f1_scores.recall
                frame_metric.update(goal_accuracy_dict)

                if intent != "NONE": # ignore None intents for the following calculations
                    thresholded_jga = linear_thresholding(goal_accuracy_dict[metrics.JOINT_GOAL_ACCURACY],
                                                          fuzzy_threshold)
                    intent_correct = thresholded_jga > 0
                    intent_id = f"{dial_id}-{intent}"
                    if intent_id not in per_intent_metrics: # new intent introduced for this dialogue
                        consistency_adjusted_jga = thresholded_jga
                        correct_turns = 1 if intent_correct else 0
                    else: # intent from previous turns in this dialogue
                        intent_dict = per_intent_metrics[intent_id]
                        prev_consistency_adjusted_jga = intent_dict[metrics.CONSISTENCY_ADJUSTED_JOINT_GOAL_ACCURACY]
                        prev_correct_turns = intent_dict[metrics.CORRECT_TURNS]
                        if prev_consistency_adjusted_jga == 0.0: #   wrong in the past, keep it wrong
                            if thresholded_jga > 0.0:
                                logging.info(f"Recovery has been made in {intent_id} in turn {turn_id}")
                            consistency_adjusted_jga = 0
                            correct_turns = prev_correct_turns
                        else: # was correct
                            consistency_adjusted_jga = thresholded_jga
                            correct_turns = prev_correct_turns + 2 if intent_correct else prev_correct_turns
                    update_intent_dict = {metrics.CORRECT_TURNS: correct_turns,
                                          metrics.CONSISTENCY_ADJUSTED_JOINT_GOAL_ACCURACY: consistency_adjusted_jga}
                    per_intent_metrics[intent_id] = update_intent_dict
                    frame_metric.update(update_intent_dict)
                    frame_metric[f'{service_name}/{intent}/{metrics.CORRECT_TURNS}'] = correct_turns
                    frame_metric[f'{service_name}/{intent}/{metrics.CONSISTENCY_ADJUSTED_JOINT_GOAL_ACCURACY}'] = consistency_adjusted_jga
                # Code for computing consistency adjusted JGA ends

                frame_id = "{:s}-{:03d}-{:s}".format(
                    dial_id, turn_id, frame_hyp["service"]
                )
                per_frame_metric[frame_id] = frame_metric
                # Add the frame-level metric result back to dialogues.
                frame_hyp["metrics"] = frame_metric

                # Get the domain name of the service.
                domain_name = frame_hyp["service"].split("_")[0]
                domain_keys = [ALL_SERVICES, frame_hyp["service"], domain_name]
                if frame_hyp["service"] in in_domain_services:
                    domain_keys.append(SEEN_SERVICES)
                else:
                    domain_keys.append(UNSEEN_SERVICES)
                for domain_key in domain_keys:
                    for metric_key, metric_value in frame_metric.items():
                        if metric_value != metrics.NAN_VAL:
                            if joint_acc_across_turn and metric_key in joint_metrics:
                                metric_collections_per_turn[domain_key][
                                    metric_key
                                ] *= metric_value
                            else:
                                metric_collections[domain_key][metric_key].append(
                                    metric_value
                                )
            if joint_acc_across_turn:
                # Conduct multiwoz style evaluation that computes joint goal accuracy
                # across all the slot values of all the domains for each turn.
                for domain_key in metric_collections_per_turn:
                    for metric_key, metric_value in metric_collections_per_turn[
                        domain_key
                    ].items():
                        metric_collections[domain_key][metric_key].append(metric_value)
    all_metric_aggregate = {}
    for domain_key, domain_metric_vals in metric_collections.items():
        domain_metric_aggregate = {}
        for metric_key, value_list in domain_metric_vals.items():
            if value_list:
                # Metrics are macro-averaged across all frames.
                domain_metric_aggregate[metric_key] = float(np.mean(value_list))
            else:
                domain_metric_aggregate[metric_key] = metrics.NAN_VAL
        all_metric_aggregate[domain_key] = domain_metric_aggregate
    return all_metric_aggregate, per_frame_metric
