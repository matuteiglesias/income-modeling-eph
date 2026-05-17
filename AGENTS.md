# AGENTS.md

## Project context

This repository is a spec-driven migration and hardening of an EPH income prediction experiment.

The original work exists as legacy student scripts. Those scripts produced a first-pass machine learning experiment for a thesis about predicting individual income in Argentina’s Encuesta Permanente de Hogares (EPH). The scientific idea, upstream data-preparation direction, modeling questions, and final research interpretation are owned by the project lead. The legacy scripts were written by a student and should be treated as reference material, not production code.

The goal of this repo is to turn that prototype into a reproducible, auditable research pipeline.

This is not a generic ML playground. It is a scientific modeling repo.

The central objective is:

> Build a reliable baseline and controlled experiment system for predicting individual total income in EPH-derived annual preprocessed data, using a common target, common feature contract, common split, common metrics, and thesis-ready evidence artifacts.

The repo now includes more than a baseline comparison. It also supports, or is expected to support:

- model comparison experiments,
- debug/smoke runs,
- HGB hyperparameter sweeps,
- regularization sweeps,
- run-local diagnostics,
- training-frame sample artifacts,
- feature views,
- geography leakage probes,
- fixed-effect linear baselines,
- run comparison suites,
- and feature audits.

The main risk is not lack of code. The main risk is contradictory experiment configs, silent leakage, duplicated logic, and unclear ownership of artifacts.

Do not drag the project back to earlier overturned assumptions. Use this document, the specs, and the current configs as the governing source of truth.

---

## Development philosophy

This repo follows a lightweight spec-driven development approach.

That means:

- Specs define the scientific and engineering contract.
- Configs define experiment choices.
- Code implements the spec.
- Tests enforce dangerous assumptions.
- Reports and diagnostics are generated from saved outputs, not from ad hoc reruns.
- Legacy scripts are retrieval/reference material during migration.
- Run directories are canonical owners of generated experiment artifacts.

Do not improvise scientific changes unless explicitly asked.

Do not add new models, new targets, new validation schemes, or new data sources just because they seem useful.

The preferred pattern is:

```text
scientific question
→ config
→ reusable implementation path
→ run-local artifact
→ diagnostic summary
→ comparison suite if needed
````

Avoid one-off scripts unless explicitly requested.

---

## Core scientific question

The baseline experiment asks:

> How well can supervised regression models predict individual total income, measured as `log10(P47T)`, from observed demographic, educational, labor, household, temporal, and geographic characteristics in EPH-derived annual preprocessed input files?

The initial comparison includes:

* LinearRegression
* Ridge
* Lasso
* HistGradientBoostingRegressor
* MLPRegressor

The goal is not to find the globally best possible model. The goal is to produce controlled, reproducible, leakage-audited evidence.

The current scientific position, based on experiments already run, is:

* Ridge and Lasso are useful as regularization and interpretability studies, but they do not materially improve over the basic linear model in predictive performance.
* HistGradientBoostingRegressor is the primary nonlinear predictive benchmark.
* HGB performance appears to plateau around the current feature information frontier.
* More HGB capacity often increases train fit and overfit gap more than CV performance.
* The clean benchmark should be conservative about target-derived geographic rank features.

---

## Non-goals

Do not add these unless explicitly requested:

* XGBoost
* LightGBM
* CatBoost
* SHAP
* dashboards
* Docker
* CI/CD
* temporal validation
* grouped household split
* deep neural architecture search
* poverty simulation
* imputation of missing income
* causal interpretation
* new targets
* fold-safe target encoding
* bootstrap resampling framework
* custom CV/grid-search engine

These may be future specs. They do not belong in the current controlled baseline or leakage-audit work unless requested.

---

## Ownership boundaries

### Upstream preprocessing owns raw/annual preparation

The upstream preprocessing pipeline owns:

* raw EPH ingestion,
* annual harmonization before this repo,
* deflation or real-income treatment if any,
* construction of annual preprocessed input CSVs,
* externally maintained indicators or ranks,
* upstream documentation about how variables were created.

This repo should treat upstream annual inputs as fixed artifacts.

Expected input files:

```text
data/annual_preprocessed_inputs/EPHARG_annual_input_22.csv
data/annual_preprocessed_inputs/EPHARG_annual_input_23.csv
data/annual_preprocessed_inputs/EPHARG_annual_input_24.csv
data/annual_preprocessed_inputs/EPHARG_annual_input_25.csv
```

These are annual preprocessed EPH-derived inputs, not raw INDEC microdata and not the train subset of a train/test/validation split. The older term “training files” is historical and should not be used in new docs unless clearly explained.

### Dataset builder owns modeling dataset construction

The dataset builder owns:

* loading annual preprocessed inputs,
* harmonizing columns across years,
* applying inclusion criteria,
* constructing `logP47T`,
* applying modeling-specific feature engineering,
* excluding forbidden predictors,
* writing `modeling_dataset.parquet`,
* writing `dataset_metadata.json`,
* creating or reusing split assignments.

Canonical outputs:

```text
data/processed/modeling_dataset.parquet
data/processed/dataset_metadata.json
data/processed/split_assignments.csv
```

### Feature views own experiment-specific feature variants

A feature view is an experiment-level transformation of the final modeling frame.

It should not modify the base processed dataset.

Feature views may:

* drop columns,
* permute controlled columns,
* define clean geography variants,
* define leakage probes,
* add fixed-effect design columns if specified by model design.

Feature views must be applied after the modeling dataset, split assignments, and runtime sampling are established, but before model fitting.

Feature views must be reflected in:

```text
feature_columns.json
training_frame_sample.csv
run_manifest.json
experiment_card.json
```

### Experiment runner owns training and run-local artifacts

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

### Diagnostics own post-run interpretation

Diagnostics read saved artifacts.

They must not retrain models.

They may produce:

* residual summaries,
* error by income decile,
* prediction distribution summaries,
* hyperparameter plots,
* regularization plots,
* runtime efficiency summaries,
* scientific summary markdowns,
* comparison tables,
* feature audits.

Diagnostics should not mutate canonical predictions, training samples, or CV results.

---

## Runtime modes and experiment kinds

Keep runtime modes simple.

Allowed runtime modes:

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

Do not proliferate new runtime modes such as `pilot`, `benchmark`, `study`, or `probe`.

Use `experiment.kind` for semantic intent:

```yaml
experiment:
  id: hgb_quick_benchmark_v1
  kind: benchmark
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
feature_audit
```

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

Expensive runs should be guarded through config or CLI:

```yaml
runtime:
  allow_full_run: true
```

or:

```bash
--allow-full-run
```

---

## Data context

Expected input files:

```text
data/annual_preprocessed_inputs/EPHARG_annual_input_22.csv
data/annual_preprocessed_inputs/EPHARG_annual_input_23.csv
data/annual_preprocessed_inputs/EPHARG_annual_input_24.csv
data/annual_preprocessed_inputs/EPHARG_annual_input_25.csv
```

The legacy scripts load these four annual files, harmonize 2024 and 2025 by dropping some columns not aligned with prior years, concatenate all years, construct `logP47T`, engineer features, and train models.

Do not reintroduce `data/training/` terminology in new configs or docs unless explicitly describing historical naming.

---

## Legacy scripts

Legacy scripts live in:

```text
legacy/student_scripts/
```

Expected files may include:

```text
data_ingestion_analysis_legacy.py
feature_engineering_legacy.py
deeper_grid_search_legacy.py
neural_network_legacy.py
```

They correspond to the original uploaded files:

* `Data_Ingestion_and_Analysis(4).py`
* `Feature_Engineering(4).py`
* `Deeper_Grid_Search(3).py`
* `Neural_Network(2).py`

Legacy policy:

* Do not modify legacy scripts unless explicitly instructed.
* Use them to understand current behavior.
* Migrate behavior into `src/eph_income/`.
* Preserve scientific intent first.
* Improve comparability and reproducibility where the spec requires it.

---

## Target and inclusion contract

Target:

```text
logP47T = log10(P47T)
```

Inclusion criteria:

```text
INGRESO == 1
P47T > 0
PROP not missing
```

The dataset builder must apply these criteria before producing the modeling dataset.

The thesis and reports must describe the target as `log10(P47T)`, not natural log.

A difference of `0.05` in log base 10 is not approximately 5 percent. It corresponds to:

```text
10^0.05 - 1 ≈ 12.2%
```

Use base-10 interpretation when explaining errors.

---

## Forbidden predictors

The following variables must never enter the feature matrix `X`.

### Target and direct transformation

```text
P47T
logP47T
```

`logP47T` may appear in training-frame sample artifacts only as the explicit target column, not as a feature.

### Identifiers

```text
CODUSU
```

### Derived temporal/date variables excluded from X

```text
Q
source_year
```

`Q` is a derived quarter/provenance variable. It is excluded because `ANO4` and `TRIMESTRE` already represent time for the baseline.

### Direct income components

```text
P21
T_VI
V12_M
V2_M
V3_M
V5_M
TOT_P12
PP08D1
```

### Transformed income components

```text
logP21
logT_VI
logV12_M
logV2_M
logV3_M
logV5_M
logTOT_P12
logPP08D1
```

### Income source indicators

```text
INGRESO_NLB
INGRESO_JUB
INGRESO_SBS
```

### Sample indicators

```text
INGRESO
```

Any script constructing `X` must fail loudly if any forbidden predictor is present.

---

## Target-derived and leakage-suspect features

Some features are not direct leakage but are target-derived.

Current examples:

```text
AGLO_rk
Reg_rk
```

These are geographic ranks between 0 and 1 based on mean income by agglomerate or region.

Treat them as:

```text
leakage-suspect target-derived geographic encodings
```

They should not be part of the clean benchmark unless they are computed fold-safely, train-only, historically, or externally.

Do not silently treat `AGLO_rk` and `Reg_rk` as ordinary numeric geography variables.

Recommended clean benchmark policy:

```text
keep AGLOMERADO and Region
drop AGLO_rk and Reg_rk
```

Use target-derived ranks only in explicit geography leakage probes.

---

## Substantive covariates

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
predicting or simulating income using observed person, household, housing,
geographic, temporal, and labor-status information
```

The thesis must state this scope clearly.

Do not confuse substantive labor covariates with target-derived geographic ranks.

---

## Main feature engineering behavior

The legacy feature engineering script does the following:

* loads 2022, 2023, 2024, 2025 annual preprocessed input files,
* drops columns from 2024 and 2025:

```text
V2_01_M
V2_02_M
V2_03_M
V5_01_M
V5_02_M
V5_03_M
```

* concatenates annual files,
* constructs `logP47T`,
* recodes several categorical variables,
* uses `PP07G_59` as summarized labor-benefit indicator,
* drops detailed variables:

```text
PP07G1
PP07G2
PP07G3
PP07G4
PP07H
PP07I
PP07J
PP07K
```

* drops `P07` and `P08`, keeping more detailed education variables,
* constructs `sanitacion_nivel` from bathroom/sanitation variables,
* drops `H10`, `H11`, `H12` after constructing `sanitacion_nivel`,
* constructs household age composition variables:

```text
Personas_0-3
Personas_4-8
Personas_9-12
Personas_13-17
Personas_18-24
Personas_25-64
Personas_65+
```

* constructs `Max_Nivel_Educativo` as maximum educational level within household.

Important: household age composition features must be audited. If they are counts, their sum should be checked against `IX_TOT` or the correct household-size variable.

---

## Split design

The repo uses a random split:

```text
70% train
20% test
10% validation
random_state = 42
```

The split is done by individual row.

Known limitation:

* Since household-derived variables are used, members of the same household may appear in different splits.
* This is acceptable for the baseline if documented.
* Do not implement grouped household splitting unless a future spec requests it.

All models in a run must use the same split assignments.

---

## Preprocessing policy

The baseline should use scikit-learn pipelines.

General principle:

* Fit preprocessing only on training folds/data.
* Never preprocess the full dataset outside the pipeline except for deterministic feature engineering.
* Use the same declared feature set for all models in the same experiment unless the config explicitly defines a variant.
* Prefer explicit imputers inside the pipeline over mysterious failure.

### LinearRegression, Ridge, Lasso, MLP

Numeric variables:

* impute if needed,
* standardize with `StandardScaler`.

Categorical variables:

* encode with `OneHotEncoder(handle_unknown="ignore")`.

For linear models, using `drop="first"` is acceptable if specified and tested. Consistency is more important than reproducing every legacy detail.

### HistGradientBoostingRegressor

Numeric variables:

* passthrough or impute if needed,
* no scaling required.

Categorical variables:

* currently one-hot encode with `handle_unknown="ignore"` unless a future spec moves to native categorical handling.

Do not change preprocessing strategy silently.

---

## HGB benchmark governance

Current experiments indicate:

* HGB is the main nonlinear benchmark.
* Linear regularization did not materially improve prediction.
* HGB reaches a practical plateau around the current feature information frontier.
* `learning_rate × max_iter` is a coupled budget.
* `max_leaf_nodes` around 31 to 63 is the useful region.
* `min_samples_leaf` around 50 to 150 is reasonable.
* `l2_regularization` was not an important lever in the tested region.
* More capacity can increase overfit and runtime without meaningful CV gain.

### Fast stable HGB benchmark

Use this as the fast stable HGB benchmark unless explicitly overridden:

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

This is not the absolute best observed point. It is preferred because it is:

* fast,
* stable,
* close to the HGB performance plateau,
* lower overfit than higher-capacity alternatives,
* easy to explain.

### Costlier candidate

A more expensive candidate is:

```yaml
reg__learning_rate: [0.03]
reg__max_iter: [600]
reg__max_leaf_nodes: [31]
reg__min_samples_leaf: [100]
reg__l2_regularization: [0.0]
reg__early_stopping: [false]
```

Use this only for confirmatory runs, not routine iteration.

### Stale HGB grids

Do not use old aggressive HGB grids such as:

```yaml
learning_rate: [0.25]
max_leaf_nodes: [200, 400]
min_samples_leaf: [15, 30]
```

unless the experiment is explicitly marked as historical or stress-test. These values contradict the current sweep evidence and should not be used as the final baseline.

---

## Feature views

### Purpose

Feature views define controlled variants of the same modeling dataset.

They prevent ad hoc code paths.

### Config pattern

Clean geography:

```yaml
feature_view:
  name: clean_geo
  drop_columns:
    - AGLO_rk
    - Reg_rk
```

No geography:

```yaml
feature_view:
  name: no_geo
  drop_columns:
    - AGLOMERADO
    - Region
    - AGLO_rk
    - Reg_rk
```

Shuffled geographic ranks:

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

### Rules

Feature views must:

* run after dataset/split/runtime sampling,
* modify only the experiment frame,
* update final feature columns,
* be reflected in training-frame sample,
* be reflected in manifest,
* re-run forbidden predictor checks.

### Permutation rule

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

## Fixed effects

### Purpose

Fixed effects provide a clean way for linear models to use geography or grouped intercepts without relying on precomputed target ranks.

Examples:

```text
Region FE
AGLOMERADO FE
AGLOMERADO × ANO4 FE
```

### Config pattern

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

### Rules

Allowed:

* one-column FE,
* two-column interaction FE.

Disallowed for now:

* FE with more than two columns,
* FE exceeding `max_levels`,
* FE silently created with huge cardinality.

If FE cardinality is too large, fail clearly.

### Coefficient export

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

## Geography leakage probe

The immediate geography study should compare:

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
with_geo_ranks > clean_geo and shuffled ≈ clean_geo:
  ranks contain target-derived signal

with_geo_ranks ≈ clean_geo:
  ranks are unnecessary

shuffled ≈ with_geo_ranks:
  suspect bug or spurious effect

clean_geo > no_geo:
  geography matters even without target-derived ranks
```

Do not implement fold-safe target encoding yet. First run simple A/B leakage probes.

---

## Comparison suites

Comparison suites compare several runs or variants that answer one scientific question.

Minimal comparison script pattern:

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

Do not create a new comparison script for every question.

---

## Training-frame sample artifact

Every normal run should write exactly one small training-frame sample unless disabled.

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

Use this artifact to visually audit what is being sent into training.

---

## Feature audit governance

Before thesis-level claims, generate a feature audit.

Recommended artifacts:

```text
diagnostics/feature_audit.csv
diagnostics/feature_audit_summary.md
diagnostics/household_age_composition_check.csv
```

### Household age composition invariant

If these features exist:

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

or the correct household-size variable.

If this does not match at high rate, investigate feature engineering before using these features in final thesis models.

### Special numeric codes

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

### High-cardinality audit

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

## Required outputs

The baseline pipeline should produce:

```text
data/processed/modeling_dataset.parquet
data/processed/dataset_metadata.json
data/processed/split_assignments.csv
reports/tables/model_comparison.csv
reports/tables/model_comparison.tex
reports/tables/dataset_summary.tex
reports/experiment_cards/baseline_income_prediction.json
reports/experiment_cards/baseline_income_prediction.md
```

Run directories should produce:

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

The main model comparison table must include:

```text
model
best_params
feature_count
cv_r2_mean
cv_r2_std
test_r2
test_mae
test_mse
validation_r2
validation_mae
validation_mse
fit_time_seconds
```

---

## Reporting policy

Reporting scripts must not train models.

Reporting scripts should read saved outputs only:

```text
reports/tables/model_comparison.csv
data/processed/dataset_metadata.json
reports/experiment_cards/baseline_income_prediction.json
reports/runs/<run_id>/
```

Then they should produce thesis-ready tables, markdown cards, and diagnostics.

---

## Repository architecture

Expected minimal structure:

```text
income-modeling-eph/
  AGENTS.md
  README.md
  Makefile
  pyproject.toml
  .gitignore

  specs/
    001_baseline_income_prediction.md

  configs/
    feature_contract.yaml
    experiment_baseline.yaml
    experiment_debug.yaml
    experiment_hgb_quick_benchmark.yaml
    experiment_hgb_quick_clean_geo_v1.yaml
    experiment_hgb_quick_with_geo_ranks_v1.yaml
    experiment_hgb_quick_shuffled_geo_ranks_v1.yaml

  legacy/
    student_scripts/

  data/
    annual_preprocessed_inputs/
    processed/

  src/
    eph_income/
      __init__.py
      config.py
      contracts.py
      dataset.py
      features.py
      splits.py
      pipelines.py
      experiments.py
      diagnostics.py
      reporting.py

  scripts/
    01_build_dataset.py
    02_run_baseline_experiment.py
    03_build_report_artifacts.py
    04_build_diagnostics.py
    05_compare_runs.py

  reports/
    tables/
    figures/
    experiment_cards/
    runs/
    comparisons/

  tests/
    test_feature_contract.py
    test_dataset_build.py
    test_split_integrity.py
    test_experiment_outputs.py
    test_diagnostics.py
```

Do not create many additional modules unless necessary.

---

## Expected commands

The repo should support:

```bash
make install
make test
make lint
make build-dataset
make run-debug
make run-baseline
make run-hgb-quick-benchmark
make run-hgb-quick-clean-geo
make run-hgb-quick-with-geo-ranks
make run-hgb-quick-shuffled-geo-ranks
make build-diagnostics RUN_DIR=...
make report
make all
```

Suggested Makefile behavior:

```text
make build-dataset
  builds modeling dataset and split assignments

make run-debug
  runs a cheap debug experiment

make run-hgb-quick-benchmark
  runs the fast stable HGB benchmark

make run-baseline
  trains baseline model families and writes model_comparison.csv

make build-diagnostics
  builds diagnostics from a saved run directory

make report
  generates thesis-ready report artifacts from saved outputs
```

---

## Testing priorities

Tests should focus on scientific failure modes.

### Feature contract

* Forbidden predictors are correctly loaded from YAML.
* A feature list containing `P47T` or `logP47T` fails.
* A feature list containing income components fails.
* Target column is allowed only as explicit target in training-frame sample.

### Dataset build

* Processed dataset is created.
* Target exists.
* Inclusion criteria are applied.
* Forbidden predictors are absent from candidate features.
* Metadata contains row counts and feature count.

### Split integrity

* Every row has exactly one split.
* Splits are train/test/validation only.
* Proportions are approximately 70/20/10.
* Split file can be reused.

### Experiment outputs

* `model_comparison.csv` exists after running experiment.
* It has one row per enabled model.
* It includes CV, test, and validation metrics.
* All models in the same experiment report the expected feature count.
* No forbidden predictor appears in stored feature columns.
* Training-frame sample is written once per run.

### Feature views

* `drop_columns` removes requested columns.
* Clean geography drops `AGLO_rk` and `Reg_rk` but keeps `AGLOMERADO` and `Region`.
* No-geo view removes all geography columns.
* Shuffled rank view preserves rank value distribution but changes mapping.
* Training-frame sample reflects final feature set.

### Fixed effects

* FE supports one-column and two-column interaction specs.
* FE rejects more than two columns.
* FE rejects excessive cardinality.
* FE coefficient export works on synthetic linear pipelines.

### Comparison suites

* Compare script reads run-local model comparison files.
* Outputs CSV and markdown.
* Includes feature view summary where available.

---

## Scientific claim boundaries

Safe claims:

* This repo implements a reproducible baseline for predicting positive observed individual income in EPH-derived data.
* The comparison is predictive, not causal.
* Metrics are computed on log10 income scale.
* The baseline random split evaluates generalization within the pooled sample distribution.
* HGB outperforms linear-family models under the current feature contract and split design.
* Target-derived geographic ranks are treated separately as leakage-suspect.

Unsafe claims:

* Do not claim the model replaces official poverty or income methodology.
* Do not claim causal effects of features on income.
* Do not claim missing-income imputation validity unless a future spec addresses missingness.
* Do not claim temporal generalization unless a temporal validation experiment is implemented.
* Do not claim household-level poverty prediction unless household aggregation and poverty-line logic are implemented.
* Do not claim `AGLO_rk` or `Reg_rk` are legitimate clean predictors unless their provenance is train-only, fold-safe, historical, or external.

---

## Important methodological notes

### Log scale

The target is base-10 log income.

If translating log errors to relative income differences, use:

```text
relative error factor = 10^error
relative percent difference = 10^error - 1
```

Do not use natural-log approximations unless the target is changed.

### Income total vs labor income

`P47T` is total individual income, not strictly labor income.

Use “ingreso total individual” unless the code uses a specifically labor-income variable.

### Deflation / real income

Do not assert that `P47T` is deflated unless the pipeline explicitly does that or the input dataset documentation proves it.

If the annual preprocessed input files already contain deflated `P47T`, document the source.

If not verified, say:

> The annual preprocessed input files provide the income variable used in the experiment; whether it is nominal or real must be documented from the upstream data-generation pipeline.

### Temporal variables

The baseline should explicitly decide whether `ANO4` and `TRIMESTRE` are included.

Current baseline decision:

```text
Include ANO4 and TRIMESTRE for all baseline models.
```

Future robustness specs may exclude them or use temporal validation.

---

## Naming conventions

### Experiment IDs

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

Avoid vague names:

```text
test2
new_baseline
experiment_final
```

### Config file names

Use:

```text
configs/experiment_<experiment_id>.yaml
```

Examples:

```text
configs/experiment_hgb_quick_clean_geo_v1.yaml
configs/experiment_linear_region_fe_v1.yaml
```

### Make targets

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

## Near-term roadmap

### Step 1: Stabilize benchmark config

Decide whether `experiment_hgb_quick_benchmark.yaml` is:

```text
5000-row smoke/debug
```

or:

```text
100000-row quick benchmark
```

Rename or update accordingly.

### Step 2: Clean HGB geography benchmark

Add or maintain:

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

### Step 3: Geo-rank leakage probe

Add:

```text
hgb_quick_with_geo_ranks_v1
hgb_quick_shuffled_geo_ranks_v1
```

Compare:

```text
clean_geo vs with_geo_ranks vs shuffled_geo_ranks
```

### Step 4: Fixed-effect linear baselines

Add:

```text
linear_region_fe_v1
linear_aglo_fe_v1
```

Export FE coefficients.

### Step 5: Feature audit

Add feature audit diagnostics.

Prioritize:

```text
Personas_* vs IX_TOT
special numeric codes
AGLO_rk / Reg_rk presence
forbidden predictors
constant / near-constant columns
```

### Step 6: Migrate baseline config

Update `experiment_baseline.yaml` to reflect learned HGB region and clean geography choice.

---

## Decision log

### Decision 1: HGB is primary nonlinear benchmark

Reason:

* HGB moved predictive performance substantially.
* Linear regularization did not.

### Decision 2: Ridge/Lasso remain interpretability/regularization studies

Reason:

* useful for coefficient paths and sparsity tradeoff,
* not main predictive winner.

### Decision 3: AGLO_rk / Reg_rk are leakage-suspect

Reason:

* target-derived geographic rankings,
* should not be in clean benchmark without train-only/fold-safe construction.

### Decision 4: AGLOMERADO / Region remain legitimate geography

Reason:

* geography is substantively relevant,
* categorical geography is not target-derived by itself.

### Decision 5: fixed effects are the linear middle ground

Reason:

* allows geography in regressions,
* estimable only from train,
* coefficients can be inspected,
* avoids precomputed target-derived ranks.

### Decision 6: do not build fold-safe target encoding yet

Reason:

* useful but more invasive,
* first run simpler A/B leakage probes.

---

## How to work on this repo

Before coding:

1. Read `specs/001_baseline_income_prediction.md`.
2. Read `configs/feature_contract.yaml`.
3. Read the relevant experiment config.
4. Read this `AGENTS.md`.
5. Inspect relevant legacy scripts only if needed.
6. Identify the smallest patch that satisfies the task.

When coding:

* Prefer clarity over clever abstraction.
* Keep functions pure when practical.
* Avoid import-time side effects.
* Do not load data when importing a module.
* Do not train models when importing a module.
* Put executable behavior in scripts, not module import bodies.
* Save metadata for every generated artifact.
* Keep generated artifacts run-local unless they are documented convenience copies.
* Do not modify legacy scripts unless explicitly instructed.

Before finishing:

Run:

```bash
pytest -q
```

When relevant, also run:

```bash
ruff check src tests scripts
make build-dataset
make run-debug
make report
```

If a command cannot be run because data is unavailable, state that clearly and still ensure tests that do not require real data pass.

---

## Final operating principle

Do not improve the science by invention.

Improve the infrastructure so the science can be trusted.

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

The near-term milestone is:

> one reproducible, leakage-audited, comparable model table, plus controlled evidence about whether target-derived geographic ranks should be excluded from the clean benchmark.



