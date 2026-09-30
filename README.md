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

## Longitudinal EPH evidence lane

The modern source-backed path now also has a separate governed longitudinal lane for
`2017-Q1..2026-Q1`. It consumes exactly 37 pinned `publicdata.eph-microdata@1`
parents plus an immutable `research.argentina-monetary-conversion/v1` release and
emits `research.eph-longitudinal-analysis-frame/v1`.

This lane retains every valid source person, including zero-income observations,
keeps period-qualified observation identity separate from repeated-wave linkage
candidates, preserves both nominal and common-reference real `P47T`, and marks
2020-Q2 plus 2024-Q1/Q2 as exceptional periods without deleting them. It does not
modify the frozen 2022–2025 flagship or train the downstream welfare model.

The first real L2 materialization is now complete:

- release: `eph-longitudinal-2017q1-2026q1-c155bb8f847a2f39`;
- 37/37 exact quarterly parents;
- 1,869,620 person-period rows, including 704,167 zero-income observations;
- 1,172,374 candidate repeated-person links, of which 1,073,713 are demographically consistent;
- midpoint monetary timing is the bounded-commissioning centerline: Q1→February, Q2→May, Q3→August, Q4→November.

The monetary conversion parent remains `candidate`, so this release is authorized for bounded longitudinal commissioning but not yet promoted to approved-mode scientific freeze.

See `docs/LONGITUDINAL_EPH_2017_2026_IMPLEMENTATION.md` and `docs/L2_REAL_LONGITUDINAL_EPH_COMMISSIONING_GATE.md` for the exact parent, schema, panel, monetary and L2 contracts.

## Research and deployment are different surfaces

The frozen HGB flagship is an EPH research/model-release candidate and explicitly rejects Census-shaped inference. It should remain a scientific benchmark rather than being retrofitted by silently substituting Census-derived columns.

The repository also contains the newer staged Census experiment/packaging path (`src/eph_income/census_income.py`, `docs/STAGED_CENSUS_INFERENCE_DESIGN.md`). That code is preserved as a **provisional bridge and research prototype**: in particular its out-of-fold intermediate-prediction design is scientifically useful. Its current presence does not make this repository the long-term owner of Census sample execution.

No code is removed by this boundary declaration. A later implementation can extract/promote deployment-safe model bundles and move scoring orchestration downstream only after exact contracts are proven.

## Historical 2024-Q3 labor-bridge result

The bounded **2024-Q3** labor bridge is closed under the commissioning registry owned by
`matuteiglesias/indice-pobreza-UBA`. This is historical/anchor evidence and must not be confused with the newer 2017-Q1..2026-Q1 longitudinal L10/L11/L12 program owned by `encuestador-de-hogares`:

- L2 corrected reconstruction: `closed_pass`; unresolved `H06` is absent from the governed feature contract.
- L3 marginal calibration: `closed_pass` downstream of the exact corrected L2 artifact.
- L4 welfare adjudication: `closed_negative`; true labor improves welfare prediction, while the transportable labor probabilities do not.

That negative result remains valid for the declared 2024-Q3 reconstruction hypothesis. It does **not** answer the newer longitudinal questions, which separate current aggregate labor context (L10), genuinely stale observed labor state on repeated EPH waves (L11), and donor-informed current-state probabilities (L12).

The longitudinal program consumes this repository's real C2/L2 frame but does not move Census transport or L10/L11/L12 ownership back into this repository. See `docs/LABOR_BRIDGE_SHADOW.md` for the historical anchor and the Poverty commissioning registry for its rerun triggers.

## Current command surface

```bash
make validate
make preprocessing-smoke
make preprocessing-release-fixture
make preprocessing-manifests
make longitudinal-fixture
make test
make build-dataset
make run-debug
python scripts/02_run_baseline_experiment.py --config configs/experiment_baseline.yaml --allow-full-run
make run-baseline
```

The debug runtime intentionally uses a small sample and minimal model set. Full baseline training remains explicitly guarded.

The machine-readable annual column lineage is `configs/annual_input_lineage.yaml`; per-release manifests are under `data/annual_preprocessed_manifests/`; and the annual-input consumer contract is `configs/annual_input_consumer_contract.yaml`. Ordinary validation does not mutate committed annual CSVs.
