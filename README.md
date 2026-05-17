# EPH Income Modeling

This repository is the thesis experiment layer for a reproducible EPH income-prediction baseline.

## Data boundary

The input CSV files in `data/annual_preprocessed_inputs/` are annual preprocessed EPH-derived artifacts produced by an upstream preprocessing repository. They are not raw INDEC microdata and they are not the train subset of the train/test/validation split.

The upstream preprocessing layer owns raw-data ingestion, household-individual merging, variable harmonization, regional assignment, monetary deflation, rank generation, and construction of training-level indicators. This repository treats those files as fixed inputs and owns the modeling-specific target, feature contract, leakage exclusions, split registry, model training, metrics, and thesis artifacts.

Expected input artifacts:

- `data/annual_preprocessed_inputs/EPHARG_annual_input_22.csv`
- `data/annual_preprocessed_inputs/EPHARG_annual_input_23.csv`
- `data/annual_preprocessed_inputs/EPHARG_annual_input_24.csv`
- `data/annual_preprocessed_inputs/EPHARG_annual_input_25.csv`

## Current command surface

```bash
make test
python scripts/01_build_dataset.py --check-only
make build-dataset
make run-debug
python scripts/02_run_baseline_experiment.py --config configs/experiment_baseline.yaml --allow-full-run
make run-baseline
```

The debug runtime intentionally uses a small sample and minimal model set. Full baseline training is enabled behind an explicit guard because the configured HistGradientBoostingRegressor and MLPRegressor grids can be expensive on the full processed dataset.
