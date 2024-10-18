[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/robust-dst.svg?branch=main)](https://cirrus-ci.com/github/<USER>/robust-dst)
[![ReadTheDocs](https://readthedocs.org/projects/robust-dst/badge/?version=latest)](https://robust-dst.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/robust-dst/main.svg)](https://coveralls.io/r/<USER>/robust-dst)
[![PyPI-Server](https://img.shields.io/pypi/v/robust-dst.svg)](https://pypi.org/project/robust-dst/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/robust-dst.svg)](https://anaconda.org/conda-forge/robust-dst)
[![Monthly Downloads](https://pepy.tech/badge/robust-dst/month)](https://pepy.tech/project/robust-dst)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/robust-dst)
-->

# robust-dst

> Add a short description here!

A longer description of your project goes here...

## Installation

In order to set up the necessary environment:

1. review and uncomment what you need in `environment.yml` and create an environment `robust-dst` with the help of [conda]:
   ```bash
   conda env create -f environment.yml
   ```
2. activate the new environment with:
   ```bash
   conda activate robust-dst
   ```
3. install development dependencies:
   ```bash
   conda env update -f dev_environment.yaml
   ```
4. install PyTorch
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
   ```
5. Install MultiWOZ evaluator
   From the project root directory,
   ```bash
   git clone git@github.com:WeixuanZ/MultiWOZ_Evaluation.git ./src
   pip install -e ./src/MultiWOZ_Evaluation
   ```
6. Install Methodflow
   From the project root directory,
   ```bash
   git clone --branch 0.0.1a1 git@github.com:WeixuanZ/methodflow.git ./src
   pip install -e ./src/methodflow
   ```

> **_NOTE:_**  The conda environment will have robust-dst installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.


Optional and needed only once after `git clone`:

7. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

8. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.


Then take a look into the `scripts` and `src` folders.


## Preparing and Preprocessing Data

Download the SGD and MultiWOZ datasets by running
```bash
chmod +x scripts/prepare_datasets.sh
./scripts/prepare_datasets
```

### D3ST Format

SGD and MultiWOZ datasets should be first preprocessed into a dataset-agnostic format,
which follows the baseline D3ST format and contains the necessary metadata.

Preprocess SGD dataset using
```bash
python -m scripts.preprocess_d3st_sgd \
  -d data/raw/original/ \
  -d data/raw/v1/ \
  -d data/raw/v2/ \
  -d data/raw/v3/ \
  -d data/raw/v4/ \
  -d data/raw/v5/ \
  -o data/processed/ \
  -c configs/data_processing_d3st_sgd.yaml \
  --all \
  -vv
```

Preprocess MultiWOZ dataset using
```bash
python -m scripts.preprocess_d3st_multiwoz \
  -d data/raw/multiwoz/ \
  --schema_file data/raw/multiwoz/schema.json \
  --dialogue_acts_file data/raw/multiwoz/dialog_acts.json \
  -o data/processed/ \
  -c configs/data_processing_d3st_multiwoz.yaml \
  --all \
  -vv
```

### T5DST Format

Preprocess SGD dataset using
```bash
declare -a versions=("original" "v1" "v2" "v3" "v4" "v5")
for i in "${versions[@]}"
do
  mkdir -p data/preprocessed/
  python -m scripts.preprocess_t5dst -d data/raw/"$i"/ -o data/preprocessed/ -c configs/data_processing_t5dst.yaml --train
  python -m scripts.preprocess_t5dst -d data/raw/"$i"/ -o data/preprocessed/ -c configs/data_processing_t5dst.yaml --dev
  python -m scripts.preprocess_t5dst -d data/raw/"$i"/ -o data/preprocessed/ -c configs/data_processing_t5dst.yaml --test
done
```


## Reproducing Baselines

1. Install the dependencies, prepare and preprocess the datasets as in previous sections.
2. Complete the relevant configuration file with:
   * Paths to the processed dataset
   * wandb account details
3. Use the follwing commands

### D3ST on SGD

```python

```

### D3ST on MultiWOZ

```python
```


## Citation

```
PLACEHOLDER FOR SIGDAIL PAPER
@article{cocaGroundingDescriptionDrivenDialogue2023,
   title={Grounding Description-Driven Dialogue State Trackers with Knowledge-Seeking Turns},
   author={Coca, Alexandru},
   year={2023}
}

@mastersthesis{zhangGroundingDescriptionDrivenDialogue2023,
   type={Master’s thesis},
   title={Grounding Description-Driven Dialogue State Tracking},
   school={University of Cambridge},
   author={Zhang, Weixuan},
   year={2023},
   month={May}
}
```


## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n robust-dst -f environment.lock.yml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:
   ```bash
   conda env update -f environment.lock.yml --prune
   ```


## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── robust_dst          <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.3.1 and the [dsproject extension] 0.7.2.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
