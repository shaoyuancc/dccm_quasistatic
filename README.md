# dccm_quasistatic

## Installation

Clone the repo and execute the following commands from the repository's root.

Install the `dccm_quasistatic` package in development mode:
```
pip install -e .
```

Install `pre-commit` for automatic black formatting:
```
pre-commit install
```

## Runing a single experiment

Create a config file specifying the experiment in `config` and run it using the following command:

```
python3 scripts/run_dccm_sim.py --config-name basic
```

where `basic` should be replaced with your config name.

## Running multiple experiments

Create a bash script in `scripts` and make it executable with `chmod +x run_multiple_experiments.sh`.
Then run it with `run_multiple_experiments.sh`