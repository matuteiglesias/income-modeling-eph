# EPH Income Modeling

Scientific workspace for reproducible EPH income-prediction experiments, controlled model comparison, training evidence and promotion of model candidates.

## Authority boundary

This repository owns EPH-side preprocessing after receipt of a versioned source release, modeling-dataset construction, targets/features/leakage policy, splits, training, diagnostics, experiment comparison and evidence used to decide whether a model is suitable for promotion.

It does **not** own execution of a promoted model over an exact Census sample. Census sample identity and downstream scoring orchestration belong outside this thesis/experiment workspace; semantic EPH↔Censo mappings belong to `eph-censo-aligner`.

A useful invariant for future promotion is:

> A Census-deployable model candidate must expose an input contract that can be constructed from an approved EPH↔Censo deployment feature frame without importing this research repository at scoring time.

## Data boundary

The CSV files in `data/annual_preprocessed_inputs/` are repository-owned `artifact:research.eph-annual-preprocessed@1` artifacts. They are EPH-derived annual inputs, not raw INDEC microdata and not train subsets. Historical files came from the former `encuestador-de-hogares` authority and were renamed without content changes.

Current preprocessing authority includes household/person merge policy, harmonization, regional assignment, monetary normalization, ranks/indicators and annual release evidence after receipt of an upstream EPH source release. It does not own official EPH publication, raw archive acquisition/DBF conversion, official geography or official poverty statistics. See `docs/PREPROCESSING_CHARACTERIZATION.md`.

Expected input artifacts:

- `data/annual_preprocessed_inputs/EPHARG_annual_input_22.csv`
- `data/annual_preprocessed_inputs/EPHARG_annual_input_23.csv`
- `data/annual_preprocessed_inputs/EPHARG_annual_input_24.csv`
- `data/annual_preprocessed_inputs/EPHARG_annual_input_25.csv`

## Research and deployment are different surfaces

The frozen HGB flagship is an EPH research/model-release candidate and explicitly rejects Census-shaped inference. It should remain a scientific benchmark rather than being retrofitted by silently substituting Census-derived columns.

The repository also contains the newer staged Census experiment/packaging path (`src/eph_income/census_income.py`, `docs/STAGED_CENSUS_INFERENCE_DESIGN.md`). That code is preserved as a **provisional bridge and research prototype**: in particular its out-of-fold intermediate-prediction design is scientifically useful. Its current presence does not make this repository the long-term owner of Census sample execution.

No code is removed by this boundary declaration. A later implementation can extract/promote deployment-safe model bundles and move scoring orchestration downstream only after exact contracts are proven.

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

The debug runtime intentionally uses a small sample and minimal model set. Full baseline training remains explicitly guarded.

The machine-readable annual column lineage is `configs/annual_input_lineage.yaml`; per-release manifests are under `data/annual_preprocessed_manifests/`; and the annual-input consumer contract is `configs/annual_input_consumer_contract.yaml`. Ordinary validation does not mutate committed annual CSVs.
