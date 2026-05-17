# AGENTS.md

## Project context

This repository is a spec-driven migration of an EPH income prediction experiment.

The original work exists as a set of legacy student scripts. Those scripts produced a first-pass machine learning experiment for a thesis about predicting individual income in Argentina’s Encuesta Permanente de Hogares (EPH). The scientific idea and training datasets are owned by the project lead. The legacy scripts were written by a student and should be treated as reference material, not production code.

The goal of this repo is to turn that prototype into a reproducible, auditable research pipeline.

This is not a generic ML playground. It is a scientific modeling repo.

The central objective is:

> Build a reliable baseline experiment comparing supervised regression models for predicting individual total income in EPH-derived training data, using a common target, common feature contract, common split, common metrics, and thesis-ready evidence artifacts.

## Development philosophy

This repo follows a lightweight spec-driven development approach.

That means:

- Specs define the scientific and engineering contract.
- Configs define experiment choices.
- Code implements the spec.
- Tests enforce dangerous assumptions.
- Reports are generated from saved outputs, not from ad hoc reruns.
- Legacy scripts are retrieval/reference material during migration.

Do not improvise scientific changes unless explicitly asked.

Current spec-driven development tools and discussions emphasize using specs to guide AI coding agents before implementation. GitHub’s Spec Kit describes SDD as a workflow where specs drive agent implementation, and recent discussion around agent workflows stresses grounding agents in repository context to avoid hallucinated APIs and architectural drift. This repo uses that principle in a minimal way: specs and configs are the source of truth, while the legacy scripts are historical evidence and migration reference.

## Core scientific question

The baseline experiment asks:

> How well can supervised regression models predict individual total income, measured as `log10(P47T)`, from observed demographic, educational, labor, household, temporal, and geographic characteristics in EPH-derived training files?

The initial comparison includes:

- LinearRegression
- Ridge
- Lasso
- HistGradientBoostingRegressor
- MLPRegressor

The goal is not to find the globally best possible model. The goal is to produce a controlled, reproducible baseline with clear assumptions.

## Non-goals

Do not add these unless explicitly requested:

- XGBoost
- LightGBM
- CatBoost
- SHAP
- dashboards
- Docker
- CI/CD
- temporal validation
- grouped household split
- deep neural architecture search
- poverty simulation
- imputation of missing income
- causal interpretation
- new targets

These may be future specs. They do not belong in the baseline unless requested.

## Data context

Expected input files:

```text
data/training/EPHARG_train_22.csv
data/training/EPHARG_train_23.csv
data/training/EPHARG_train_24.csv
data/training/EPHARG_train_25.csv
```

These files should be treated as processed EPH-derived training inputs, not necessarily raw INDEC microdata.

The legacy scripts load these four annual files, harmonize 2024 and 2025 by dropping some columns not aligned with prior years, concatenate all years, construct `logP47T`, engineer features, and train models.

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

## Important findings from the legacy scripts

The legacy experiment currently has useful logic but also methodological risks.

### Target construction

The target is:

```python
logP47T = np.log10(P47T) when P47T > 0
```

The regression sample is restricted to:

```python
INGRESO == 1
PROP not missing
P47T > 0
```

The thesis and reports must describe the target as `log10(P47T)`, not natural log.

A difference of `0.05` in log base 10 is not approximately 5 percent. It corresponds to:

```text
10^0.05 - 1 ≈ 12.2%
```

### Main feature engineering behavior

The legacy feature engineering script does the following:

* loads 2022, 2023, 2024, 2025 training files
* drops columns from 2024 and 2025:

  * `V2_01_M`
  * `V2_02_M`
  * `V2_03_M`
  * `V5_01_M`
  * `V5_02_M`
  * `V5_03_M`
* concatenates annual files
* constructs `logP47T`
* recodes several categorical variables
* uses `PP07G_59` as summarized labor-benefit indicator
* drops detailed variables:

  * `PP07G1`
  * `PP07G2`
  * `PP07G3`
  * `PP07G4`
  * `PP07H`
  * `PP07I`
  * `PP07J`
  * `PP07K`
* drops `P07` and `P08`, keeping more detailed education variables
* constructs `sanitacion_nivel` from bathroom/sanitation variables
* drops `H10`, `H11`, `H12` after constructing `sanitacion_nivel`
* constructs household age composition variables:

  * `Personas_0-3`
  * `Personas_4-8`
  * `Personas_9-12`
  * `Personas_13-17`
  * `Personas_18-24`
  * `Personas_25-64`
  * `Personas_65+`
* constructs `Max_Nivel_Educativo` as maximum educational level within household

### Existing split design

The legacy scripts use a random split:

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

### Existing model comparison

The legacy main grid search compares:

* LinearRegression
* Ridge
* Lasso
* HistGradientBoostingRegressor

The MLP is trained in a separate script.

Known comparability problem:

* The main grid search and the MLP script use slightly different feature exclusion sets.
* In particular, the MLP legacy script excludes `ANO4` and `TRIMESTRE`, while the main grid search keeps them.
* The MLP also uses imputers and a different one-hot encoding strategy.

Baseline repo requirement:

> All baseline models must use the same declared feature set unless a config explicitly marks the model as a separate experiment.

For the baseline, include `ANO4` and `TRIMESTRE` for all models unless the config says otherwise.

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

## Forbidden predictors

The following variables must never enter the feature matrix `X`:

### Target and direct transformation

```text
P47T
logP47T
```

### Identifiers

```text
CODUSU
Q
```

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

Any script constructing `X` must fail loudly if any forbidden predictor is present.

## Baseline model grids

Use the legacy grids unless explicitly changed by config.

### LinearRegression

```yaml
reg__fit_intercept: [true]
```

### Ridge

```yaml
reg__alpha: [2, 5, 10]
reg__fit_intercept: [true]
```

### Lasso

```yaml
reg__alpha: [0.0025, 0.005, 0.0075]
reg__fit_intercept: [true]
```

### HistGradientBoostingRegressor

```yaml
reg__max_iter: [100, 200]
reg__learning_rate: [0.25, 0.1, 0.05]
reg__max_leaf_nodes: [200, 400]
reg__min_samples_leaf: [15, 30]
reg__l2_regularization: [0]
```

### MLPRegressor

```yaml
reg__hidden_layer_sizes:
  - [64, 32]
  - [128, 64]
  - [64, 64, 32]
  - [128, 128]
reg__batch_size: [64, 128]
reg__max_iter: [100, 200]
```

## Preprocessing policy

The baseline should use scikit-learn pipelines.

General principle:

* Fit preprocessing only on training folds/data.
* Never preprocess the full dataset outside the pipeline except for deterministic feature engineering.
* Use the same feature set for all models.

Expected preprocessing:

### LinearRegression, Ridge, Lasso, MLP

* Numeric variables:

  * impute if needed
  * standardize with `StandardScaler`
* Categorical variables:

  * encode with `OneHotEncoder(handle_unknown="ignore")`
* For linear models, using `drop="first"` is acceptable.
* For MLP, consistency is more important than reproducing every legacy detail. Follow the config/spec.

### HistGradientBoostingRegressor

* Numeric variables:

  * passthrough or impute if needed
  * no scaling required
* Categorical variables:

  * one-hot encode with `handle_unknown="ignore"`

If missingness exists, prefer explicit imputers inside the pipeline over failing mysteriously.

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

The main model comparison table must include one row per model and at least:

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
fit_time_seconds if available
```

## Reporting policy

Reporting scripts must not train models.

Reporting scripts should read saved outputs only:

```text
reports/tables/model_comparison.csv
data/processed/dataset_metadata.json
reports/experiment_cards/baseline_income_prediction.json
```

Then they should produce thesis-ready tables.

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

  legacy/
    student_scripts/

  data/
    training/
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
      reporting.py

  scripts/
    01_build_dataset.py
    02_run_baseline_experiment.py
    03_build_report_artifacts.py

  reports/
    tables/
    figures/
    experiment_cards/

  tests/
    test_feature_contract.py
    test_dataset_build.py
    test_split_integrity.py
    test_experiment_outputs.py
```

Do not create many additional modules unless necessary.

## Expected commands

The repo should support:

```bash
make install
make test
make build-dataset
make run-baseline
make report
make all
```

Suggested Makefile behavior:

```text
make build-dataset
  builds modeling dataset and split assignments

make run-baseline
  trains baseline models and writes model_comparison.csv

make report
  generates thesis-ready report artifacts from saved outputs

make test
  runs pytest
```

## Testing priorities

Tests should focus on scientific failure modes.

Minimum tests:

### Feature contract

* Forbidden predictors are correctly loaded from YAML.
* A feature list containing `P47T` or `logP47T` fails.
* A feature list containing income components fails.

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

* `model_comparison.csv` exists after running baseline.
* It has one row per enabled model.
* It includes CV, test, and validation metrics.
* All models report the same feature count.
* No forbidden predictor appears in stored feature columns.

## Scientific claim boundaries

Safe claims:

* This repo implements a reproducible baseline for predicting positive observed individual income in EPH-derived data.
* The comparison is predictive, not causal.
* Metrics are computed on log10 income scale.
* The baseline random split evaluates generalization within the pooled sample distribution.

Unsafe claims:

* Do not claim the model replaces official poverty or income methodology.
* Do not claim causal effects of features on income.
* Do not claim missing-income imputation validity unless a future spec addresses missingness.
* Do not claim temporal generalization unless a temporal validation experiment is implemented.
* Do not claim household-level poverty prediction unless household aggregation and poverty-line logic are implemented.

## Important methodological notes

### Log scale

The target is base-10 log income.

If translating log errors to relative income differences, use:

```text
relative error factor = 10^error
relative percent difference = 10^error - 1
```

Do not use natural-log approximations unless the target is changed.

### Income total vs income labor

`P47T` is total individual income, not strictly labor income.

Use “ingreso total individual” unless the code uses a specifically labor-income variable.

### Deflation / real income

Do not assert that `P47T` is deflated unless the pipeline explicitly does that or the input dataset documentation proves it.

If the processed training files already contain deflated `P47T`, document the source.

If not verified, say:

> The training files provide the income variable used in the experiment; whether it is nominal or real must be documented from the data-generation pipeline.

### Temporal variables

The baseline should explicitly decide whether `ANO4` and `TRIMESTRE` are included.

Current baseline decision:

```text
Include ANO4 and TRIMESTRE for all baseline models.
```

Future robustness specs may exclude them or use temporal validation.

## How to work on this repo

Before coding:

1. Read `specs/001_baseline_income_prediction.md`.
2. Read `configs/feature_contract.yaml`.
3. Read `configs/experiment_baseline.yaml`.
4. Inspect relevant legacy scripts.
5. Identify the smallest patch that satisfies the task.

When coding:

* Prefer clarity over clever abstraction.
* Keep functions pure when practical.
* Avoid import-time side effects.
* Do not load data when importing a module.
* Do not train models when importing a module.
* Put executable behavior in scripts, not module import bodies.
* Save metadata for every generated artifact.

Before finishing:

Run:

```bash
pytest -q
```

When relevant, also run:

```bash
make build-dataset
make run-baseline
make report
```

If a command cannot be run because data is unavailable, state that clearly and still ensure tests that do not require real data pass.

## Current migration milestones

### Milestone 1: Contracts and scaffold

Goal:

* Load YAML configs.
* Expose forbidden predictor checks.
* Add minimal tests.

### Milestone 2: Dataset builder

Goal:

* Migrate feature engineering into clean functions.
* Produce `modeling_dataset.parquet`.
* Produce `dataset_metadata.json`.

### Milestone 3: Split registry

Goal:

* Save reproducible train/test/validation assignment.
* Ensure every model uses the same split.

### Milestone 4: Baseline experiment runner

Goal:

* Train all baseline model families.
* Use consistent feature set.
* Write one model comparison table.

### Milestone 5: Reporting

Goal:

* Generate thesis-ready LaTeX and Markdown artifacts from saved outputs only.

### Milestone 6: Audit

Goal:

* Verify implementation against spec.
* Produce an audit card.
* Fix inconsistencies without adding scope.

## Final reminder

Do not improve the science by invention.

Improve the infrastructure so the science can be trusted.

The first real milestone is:

> one reproducible, leakage-audited, comparable model table.

