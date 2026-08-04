# EPH Income Modeling

This repository is the preprocessing authority and thesis experiment layer for a reproducible EPH
income-prediction baseline.

## Data boundary

The input CSV files in `data/annual_preprocessed_inputs/` are the repository-owned release
`artifact:research.eph-annual-preprocessed@1`. They are annual preprocessed EPH-derived artifacts, not raw INDEC
microdata and not the train subset of the train/test/validation split. The historical files were produced by the
former authority, `encuestador-de-hogares`, and renamed from `EPHARG_train*` without content changes.

This repository now owns acquisition-to-analysis preprocessing after receipt of a versioned EPH source release,
household/person merge policy, harmonization, regional assignment, monetary normalization, rank and indicator
derivation, annual releases, and the downstream modeling system. It does not own official EPH publication, raw
archive acquisition/DBF conversion, official geography, or official poverty statistics. The current materialized
files can be validated and manifested locally; their original raw source hashes and monetary reference remain
explicitly provisional pending upstream reconciliation. See
[`docs/PREPROCESSING_CHARACTERIZATION.md`](docs/PREPROCESSING_CHARACTERIZATION.md).

Expected input artifacts:

- `data/annual_preprocessed_inputs/EPHARG_annual_input_22.csv`
- `data/annual_preprocessed_inputs/EPHARG_annual_input_23.csv`
- `data/annual_preprocessed_inputs/EPHARG_annual_input_24.csv`
- `data/annual_preprocessed_inputs/EPHARG_annual_input_25.csv`

## Current command surface

```bash
make validate
make preprocessing-smoke
make preprocessing-release-fixture
make preprocessing-manifests
make test
make build-dataset
make run-debug
python scripts/02_run_baseline_experiment.py --config configs/experiment_baseline.yaml --allow-full-run
make run-baseline
```

The debug runtime intentionally uses a small sample and minimal model set. Full baseline training is enabled behind an explicit guard because the configured HistGradientBoostingRegressor and MLPRegressor grids can be expensive on the full processed dataset.

The machine-readable annual column lineage is `configs/annual_input_lineage.yaml`; per-release manifests are in
`data/annual_preprocessed_manifests/`; and the Batch 2 contract is
`configs/annual_input_consumer_contract.yaml`. Ordinary validation does not mutate committed annual CSVs.
