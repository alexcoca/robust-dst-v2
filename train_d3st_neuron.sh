#!/bin/bash
# run 1
python scripts/run_dialogue_state_tracking.py configs/train_d3st_finetuned_pegasus_sbert_neuron.json
# run 2
python scripts/run_dialogue_state_tracking.py configs/train_d3st_finetuned_pegasus_sbert_neuron_2.json