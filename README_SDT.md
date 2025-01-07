# Replicating Show, Don't Tell

## Downloading and Pre-Processing SGD Data
The following command will:
1. Download SGD
2. Create Data for SDT
3. Complete preprocessing for SDT and create a ```.json``` file containing the data and ```.yaml``` containing the data processing configuration
```bash
bash scripts/preprocess_sdt/prepare_sgd_sdt_data.sh
```

Check the script file for detailed implementation of the process.
```bash
python scripts/run_dialogue_state_tracking.py "configs/replicate_sdt_v0.json"
```