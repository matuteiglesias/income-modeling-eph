# Experiment Governance for `income-modeling-eph`

## 0. Purpose

This repository is no longer just a place to run models. It is becoming a small scientific machine for studying income prediction over EPH-derived datasets.

The goal of this governance document is to keep that machine coherent while we add:

- baseline/debug experiments,
- HGB sweeps,
- regularization sweeps,
- training-frame audit artifacts,
- feature views,
- geography leakage probes,
- fixed effects,
- run comparisons,
- and thesis-ready diagnostics.

The main risk is not lack of code. The main risk is contradictory experiment configs, silent leakage, duplicated logic, and unclear ownership of artifacts.

This document defines the shared patterns.

---

# 1. Ownership boundaries

## 1.1 Upstream preprocessing owns raw/annual preparation

The upstream preprocessing pipeline owns:

- raw EPH ingestion,
- annual harmonization,
- deflation,
- income preprocessing,
- construction of annual preprocessed input CSVs,
- any externally maintained ranks or indicators.

This repo should treat upstream annual inputs as fixed input artifacts.

Current expected boundary:

```text
data/annual_preprocessed_inputs/EPHARG_annual_input_22.csv
data/annual_preprocessed_inputs/EPHARG_annual_input_23.csv
data/annual_preprocessed_inputs/EPHARG_annual_input_24.csv
data/annual_preprocessed_inputs/EPHARG_annual_input_25.csv
````

This repo should not silently become the owner of raw EPH preprocessing.

## 1.2 Dataset builder owns modeling dataset construction

The dataset builder owns:

* loading annual preprocessed inputs,
* harmonizing columns across years,
* applying inclusion criteria,
* creating target `logP47T`,
* applying modeling-specific feature engineering,
* excluding forbidden predictors,
* writing `modeling_dataset.parquet`,
* writing `dataset_metadata.json`,
* creating/reusing split assignments.

Canonical outputs:

```text
data/processed/modeling_dataset.parquet
data/processed/dataset_metadata.json
data/processed/split_assignments.csv
```

## 1.3 Feature views own experiment-specific feature variants

A feature view is an experiment-level transformation of the final modeling frame.

It should not modify the base processed dataset.

Feature views may:

* drop columns,
* permute controlled columns,
* create fixed-effect design columns,
* define clean geography variants,
* define leakage probes.

They should be applied after:

```text
dataset + split assignments + runtime sample
```

and before:

```text
train/test/validation separation and model fitting
```

Feature views must be reflected in:

```text
feature_columns.json
training_frame_sample.csv
run_manifest.json
experiment_card.json
```

## 1.4 Experiment runner owns training and run-local artifacts

The experiment runner owns:

* loading processed dataset,
* loading split assignments,
* applying runtime sample,
* applying feature views,
* constructing model pipelines,
* running GridSearchCV,
* writing predictions,
* writing metrics,
* writing CV results,
* writing run-local artifacts,
* writing run manifest.

Every experiment run should have a canonical owner:

```text
reports/runs/<run_id>/
```

Convenience copies may exist under:

```text
reports/tables/
reports/experiment_cards/
```

but the run directory is canonical.

## 1.5 Diagnostics own post-run interpretation

Diagnostics should read saved artifacts.

They should not retrain models.

They may produce:

* residual summaries,
* error by income decile,
* prediction distribution summaries,
* hyperparameter plots,
* regularization plots,
* runtime efficiency summaries,
* scientific summary markdowns,
* comparison tables.

Diagnostics should not mutate canonical predictions or training artifacts.

---

# 2. Runtime modes

The project should keep runtime modes simple.

## 2.1 Allowed runtime modes

Use only:

```yaml
runtime:
  mode: debug
```

```yaml
runtime:
  mode: sweep
```

```yaml
runtime:
  mode: full
```

Do not proliferate new runtime modes such as `pilot`, `benchmark`, `study`, `probe`, etc. Those are semantic experiment kinds, not execution modes.

## 2.2 Semantic experiment kind

Use:

```yaml
experiment:
  id: hgb_quick_benchmark_v1
  kind: benchmark
```

or:

```yaml
experiment:
  id: hgb_lr_iter_sweep_v1
  kind: hgb_param_sweep
```

Possible `experiment.kind` values:

```text
debug
benchmark
baseline
regularization_sweep
hgb_param_sweep
geography_leakage_probe
fixed_effects_study
comparison_suite
```

This keeps execution modes simple while still documenting scientific intent.

## 2.3 Sampling semantics

Sampling is controlled by:

```yaml
runtime:
  sample_n: 100000
```

Rules:

* `sample_n: null` means use all available rows.
* `sample_n: 5000` means sampled experiment.
* Sampling must preserve train/test/validation split labels.
* Training-frame sample artifacts must be drawn after runtime sampling.

## 2.4 Guardrails

Expensive modes should be guarded:

```yaml
runtime:
  allow_full_run: true
```

or CLI:

```bash
--allow-full-run
```

Rules:

* `debug` should run without explicit full-run permission.
* `sweep` and `full` may require permission, depending on config.
* This avoids accidental long runs.

---

# 3. Artifact standards

## 3.1 Run directory is canonical

Every run writes:

```text
reports/runs/<run_id>/
  config_used.yaml
  feature_contract_used.yaml
  dataset_card.json
  feature_columns.json
  run_manifest.json
  metrics/
  predictions/
  cv_results/
  diagnostics/
  artifacts/
  plots/
```

The manifest must list canonical paths.

## 3.2 Training-frame sample

Every normal experiment should write exactly one small training-frame sample unless disabled.

Config:

```yaml
artifacts:
  training_frame_sample_n: 10
```

Disable:

```yaml
artifacts:
  training_frame_sample_n: 0
```

Output:

```text
artifacts/training_frame_sample.csv
artifacts/training_frame_sample_metadata.json
```

Rules:

* one file per run, not per model,
* drawn from train split,
* after runtime sampling,
* before model fitting,
* includes `row_id`, `split`, target, and final feature columns,
* must reflect feature views and dropped columns.

## 3.3 Predictions

Prediction artifacts must include:

```text
run_id
model
split
row_id
y_true
y_pred
residual
abs_error
squared_error
income_decile
```

Predictions are used for residual and decile diagnostics.

## 3.4 CV results

Every model should write:

```text
cv_results/<ModelName>.csv
```

This is the canonical source for:

* hyperparameter plots,
* regularization curves,
* overfit gaps,
* runtime efficiency summaries.

---

# 4. Feature contract and leakage rules

## 4.1 Direct forbidden predictors

The following classes must never enter `feature_columns`:

* target source,
* target transform,
* income components,
* transformed income components,
* sample inclusion flags,
* identifiers,
* temporal derived columns that duplicate retained time variables.

Examples:

```text
P47T
logP47T as feature
P21
PP08D1
TOT_P12
T_VI
V2_M
V3_M
V5_M
logP21
logPP08D1
logTOT_P12
INGRESO
CODUSU
Q
source_year
```

The target `logP47T` may appear in training-frame sample only as explicit target, not as a feature.

## 4.2 Target-derived features

Some features are not direct leakage but are target-derived. They require special handling.

Current examples:

```text
AGLO_rk
Reg_rk
```

These are geographic ranks based on mean income by agglomerate or region.

They must be treated as:

```text
leakage-suspect target-derived geographic encodings
```

They should not be part of the clean benchmark unless they are computed fold-safely, train-only, historically, or externally.

## 4.3 Substantive covariates

Variables such as:

```text
CONDACT
CAT_INAC
CAT_OCUP
PP07G_59
```

are not automatically leakage. They are contemporaneous labor-market covariates.

They are allowed if the experiment is framed as:

```text
predicting/simulating income using observed person, household, housing,
geographic, and labor-status information
```

The thesis must state this scope clearly.

---

# 5. Feature views

## 5.1 Purpose

Feature views define controlled variants of the same modeling dataset.

They prevent ad hoc code paths.

## 5.2 Config pattern

```yaml
feature_view:
  name: clean_geo
  drop_columns:
    - AGLO_rk
    - Reg_rk
```

```yaml
feature_view:
  name: shuffled_geo_ranks
  permute_group_values:
    - column: AGLO_rk
      group_by: AGLOMERADO
      random_state: 42
    - column: Reg_rk
      group_by: Region
      random_state: 42
```

```yaml
feature_view:
  name: no_geo
  drop_columns:
    - AGLOMERADO
    - Region
    - AGLO_rk
    - Reg_rk
```

## 5.3 Rules

Feature views must:

* run after dataset/split/runtime sampling,
* modify only the experiment frame,
* update final feature columns,
* be reflected in training-frame sample,
* be reflected in manifest,
* re-run forbidden predictor checks.

## 5.4 Permutation rule

For geographic ranks, permutation should be by group mapping, not row-wise.

Good:

```text
permute rank assignments across AGLOMERADO categories
```

Bad:

```text
shuffle rank values row by row
```

Group-level permutation preserves the structure of “one rank per geography” while breaking the correct association between geography and rank.

---

# 6. Fixed effects

## 6.1 Purpose

Fixed effects provide a clean way for linear models to use geography or grouped intercepts without relying on precomputed target ranks.

Examples:

```text
Region FE
AGLOMERADO FE
AGLOMERADO × ANO4 FE
```

## 6.2 Config pattern

```yaml
model_design:
  fixed_effects:
    - name: region_fe
      columns: [Region]
      max_levels: 20
      drop_first: true
```

```yaml
model_design:
  fixed_effects:
    - name: aglo_fe
      columns: [AGLOMERADO]
      max_levels: 100
      drop_first: true
```

```yaml
model_design:
  fixed_effects:
    - name: aglo_year_fe
      columns: [AGLOMERADO, ANO4]
      max_levels: 500
      drop_first: true
```

## 6.3 Rules

Allowed:

* one-column FE,
* two-column interaction FE.

Disallowed for now:

* FE with more than two columns,
* FE exceeding `max_levels`,
* FE silently created with huge cardinality.

If FE cardinality is too large, fail clearly.

## 6.4 Coefficient export

For linear, Ridge, and Lasso models with FE, export:

```text
diagnostics/fixed_effect_coefficients.csv
```

Columns:

```text
run_id
model
fe_name
level
feature
coefficient
abs_coefficient
sign
rank
```

No FE coefficient extraction is required for HGB.

---

# 7. Comparison suites

## 7.1 Purpose

Comparison suites compare several runs or variants that answer one scientific question.

They should avoid ad hoc manual copy-paste comparison.

## 7.2 Minimal comparison script

A simple script should read multiple run directories:

```bash
python scripts/05_compare_runs.py \
  --comparison-id geo_rank_leakage_probe_v1 \
  --run-dirs reports/runs/run_a reports/runs/run_b reports/runs/run_c
```

Outputs:

```text
reports/comparisons/geo_rank_leakage_probe_v1.csv
reports/comparisons/geo_rank_leakage_probe_v1.md
```

Minimum table columns:

```text
comparison_id
run_id
experiment_id
model
feature_count
cv_r2_mean
test_r2
validation_r2
test_mae
validation_mae
feature_view_summary
```

## 7.3 Initial comparison suites

### Geography rank leakage probe

Variants:

```text
clean_geo:
  AGLOMERADO + Region
  no AGLO_rk / Reg_rk

with_geo_ranks:
  AGLOMERADO + Region + AGLO_rk + Reg_rk

shuffled_geo_ranks:
  AGLOMERADO + Region + shuffled AGLO_rk + shuffled Reg_rk
```

Interpretation:

```text
with_geo_ranks >> clean_geo and shuffled ≈ clean_geo:
  ranks contain target-derived signal

with_geo_ranks ≈ clean_geo:
  ranks are unnecessary

shuffled ≈ with_geo_ranks:
  suspect bug or spurious effect

clean_geo >> no_geo:
  geography matters even without target-derived ranks
```

### Fixed-effect linear study

Variants:

```text
linear_no_geo
linear_region_fe
linear_aglo_fe
linear_with_geo_ranks
```

Purpose:

* measure geography in linear models,
* inspect FE coefficients,
* compare FE coefficients with target-derived ranks.

---

# 8. HGB benchmark governance

## 8.1 What we learned

From sweeps:

* Ridge/Lasso did not materially improve over OLS.
* HGB strongly improved predictive performance.
* HGB reaches a plateau around CV R² roughly 0.545–0.55 in current feature space.
* More capacity increases train R² and overfit gap.
* `l2_regularization` was not an important lever in tested region.
* `max_leaf_nodes` around 31–63 is the useful region.
* `learning_rate × max_iter` is a coupled budget.

## 8.2 Fast benchmark

Use this as fast stable benchmark:

```yaml
hist_gradient_boosting:
  grid:
    reg__loss: ["squared_error"]
    reg__learning_rate: [0.075]
    reg__max_iter: [200]
    reg__max_leaf_nodes: [31]
    reg__min_samples_leaf: [100]
    reg__l2_regularization: [0.0]
    reg__early_stopping: [false]
    reg__random_state: [42]
```

This is not the absolute best observed point. It is the preferred quick benchmark because it is:

* fast,
* stable,
* close to the HGB performance plateau,
* lower overfit than higher-capacity alternatives,
* easy to explain.

## 8.3 Costly candidate

A more expensive candidate is:

```yaml
reg__learning_rate: [0.03]
reg__max_iter: [600]
reg__max_leaf_nodes: [31]
reg__min_samples_leaf: [100]
reg__l2_regularization: [0.0]
reg__early_stopping: [false]
```

Use this only for confirmatory runs, not every iteration.

## 8.4 Clean benchmark principle

The thesis benchmark should be conservative.

Recommended clean HGB benchmark:

```text
use AGLOMERADO and Region
drop AGLO_rk and Reg_rk
```

Target-derived ranks should appear in a methodological leakage probe, not the main clean benchmark, unless fold-safe encoding is implemented.

---

# 9. Current config issues to resolve

## 9.1 Quick benchmark sample size ambiguity

If `experiment_hgb_quick_benchmark.yaml` uses:

```yaml
runtime:
  sample_n: 5000
```

then it is not a 100k benchmark. It is a smoke/fast debug benchmark.

Decide one of:

### Option A: keep as smoke

Rename or document:

```text
hgb_quick_smoke_v1
```

### Option B: make it actual quick benchmark

Set:

```yaml
runtime:
  sample_n: 100000
```

## 9.2 Baseline HGB grid is stale

If `experiment_baseline.yaml` still includes aggressive values such as:

```yaml
learning_rate: [0.25]
max_leaf_nodes: [200, 400]
min_samples_leaf: [15, 30]
```

then it contradicts current sweep evidence.

Baseline should be migrated to:

```yaml
learning_rate: [0.075]
max_iter: [200]
max_leaf_nodes: [31]
min_samples_leaf: [100]
l2_regularization: [0.0]
```

or to a very small confirmatory grid:

```yaml
learning_rate: [0.05, 0.075]
max_iter: [200, 400]
max_leaf_nodes: [31, 63]
min_samples_leaf: [100]
l2_regularization: [0.0]
```

Do not leave old aggressive HGB grid as final baseline.

## 9.3 Diagnostics config style is inconsistent

Prefer:

```yaml
diagnostics:
  sweep_type: hgb_single_param
  primary_param: max_leaf_nodes
```

or:

```yaml
diagnostics:
  sweep_type: hgb_lr_iter
  primary_param: max_iter
  group_param: learning_rate
```

Avoid mixing with older flags like:

```yaml
diagnostics:
  hist_gradient_boosting: true
```

unless that flag has a clearly separate meaning.

## 9.4 Observability should live under observability

Preferred:

```yaml
observability:
  heartbeat_seconds: 10
  sklearn_verbose: 2
```

Avoid introducing competing fields under `runtime` unless already required for compatibility.

## 9.5 Artifacts should live under artifacts

Preferred:

```yaml
artifacts:
  training_frame_sample_n: 10
```

---

# 10. Feature audit governance

## 10.1 Required audits

Before thesis-level claims, generate a feature audit.

Artifacts:

```text
diagnostics/feature_audit.csv
diagnostics/feature_audit_summary.md
diagnostics/household_age_composition_check.csv
```

## 10.2 Household age composition invariant

If features exist:

```text
Personas_0-3
Personas_4-8
Personas_9-12
Personas_13-17
Personas_18-24
Personas_25-64
Personas_65+
```

then compute:

```text
age_bin_sum = sum(Personas_*)
```

Check against:

```text
IX_TOT
```

If this does not match at high rate, investigate feature engineering before using these features in final thesis models.

Do not silently trust the model if engineered count features are inconsistent.

## 10.3 Special code audit

Numeric columns with special codes should be audited.

Examples:

```text
H15 / H16:
  31 = MISSING

P03:
  111 = NOTAPPLICABLE
  112 = MISSING
```

If special codes enter as numeric values, consider recoding to `NaN` or explicit categorical/sentinel handling.

## 10.4 High-cardinality audit

Columns such as:

```text
AGLOMERADO
```

should report:

* number of levels,
* minimum rows per level,
* rare levels,
* whether used as categorical or FE.

---

# 11. Naming conventions

## 11.1 Experiment IDs

Use:

```text
<family>_<question>_<version>
```

Examples:

```text
hgb_quick_clean_geo_v1
hgb_quick_with_geo_ranks_v1
hgb_quick_shuffled_geo_ranks_v1
hgb_lr_iter_sweep_v1
linear_region_fe_v1
linear_aglo_fe_v1
```

Avoid vague names like:

```text
test2
new_baseline
experiment_final
```

## 11.2 Config file names

Use:

```text
configs/experiment_<experiment_id>.yaml
```

Examples:

```text
configs/experiment_hgb_quick_clean_geo_v1.yaml
configs/experiment_linear_region_fe_v1.yaml
```

## 11.3 Make targets

Use:

```text
run-<experiment-id-with-dashes>
```

Examples:

```text
run-hgb-quick-clean-geo
run-hgb-quick-with-geo-ranks
run-hgb-quick-shuffled-geo-ranks
run-linear-region-fe
run-linear-aglo-fe
```

---

# 12. Minimal near-term roadmap

## Step 1: stabilize benchmark config

Choose whether quick benchmark is:

```text
5000-row smoke
```

or:

```text
100000-row quick benchmark
```

Rename or update accordingly.

## Step 2: clean HGB geography benchmark

Add:

```text
hgb_quick_clean_geo_v1
```

Drops:

```text
AGLO_rk
Reg_rk
```

Keeps:

```text
AGLOMERADO
Region
```

## Step 3: geo-rank leakage probe

Add:

```text
hgb_quick_with_geo_ranks_v1
hgb_quick_shuffled_geo_ranks_v1
```

Compare:

```text
clean_geo vs with_geo_ranks vs shuffled_geo_ranks
```

## Step 4: fixed-effect linear baselines

Add:

```text
linear_region_fe_v1
linear_aglo_fe_v1
```

Export FE coefficients.

## Step 5: feature audit

Add feature audit diagnostics.

Prioritize:

```text
Personas_* vs IX_TOT
special numeric codes
AGLO_rk / Reg_rk presence
forbidden predictors
constant / near-constant columns
```

## Step 6: migrate baseline config

Update `experiment_baseline.yaml` to reflect learned HGB region and clean geography choice.

---

# 13. Decision log

## Decision 1: HGB is primary nonlinear benchmark

Reason:

* HGB moved predictive performance substantially.
* Linear regularization did not.

## Decision 2: Ridge/Lasso remain interpretability/regularization studies

Reason:

* useful for coefficient paths and sparsity tradeoff,
* not main predictive winner.

## Decision 3: AGLO_rk / Reg_rk are leakage-suspect

Reason:

* target-derived geographic rankings,
* should not be in clean benchmark without train-only/fold-safe construction.

## Decision 4: AGLOMERADO / Region remain legitimate geography

Reason:

* geography is substantively relevant,
* categorical geography is not target-derived by itself.

## Decision 5: fixed effects are the linear middle ground

Reason:

* allows geography in regressions,
* estimable only from train,
* coefficients can be inspected,
* avoids precomputed target-derived ranks.

## Decision 6: do not build fold-safe target encoding yet

Reason:

* useful but more invasive,
* first run simpler A/B leakage probes.

---

# 14. Final operating principle

Do not add a new script or config style for every question.

Use the same small set of patterns:

```text
dataset builder
feature view
model design
experiment runner
run directory
diagnostics
comparison suite
```

If a new idea does not fit one of those patterns, pause before implementing it.



