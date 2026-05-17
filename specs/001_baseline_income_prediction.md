# Spec 001: Baseline Income Prediction on EPH Annual Preprocessed Inputs

## Purpose

Build a reproducible baseline experiment for predicting individual income using EPH-derived annual preprocessed input files.

The goal is not to produce the most optimized model. The goal is to establish a scientifically auditable baseline that reproduces and hardens the legacy student experiment.

## Scientific question

How well do linear, regularized, tree-boosting, and neural-network models predict log individual income under a common data split, common feature contract, and common evaluation protocol?

## Input data

Expected files:

- `data/annual_preprocessed_inputs/EPHARG_annual_input_22.csv`
- `data/annual_preprocessed_inputs/EPHARG_annual_input_23.csv`
- `data/annual_preprocessed_inputs/EPHARG_annual_input_24.csv`
- `data/annual_preprocessed_inputs/EPHARG_annual_input_25.csv`

These are annual preprocessed EPH-derived inputs produced by an upstream preprocessing pipeline, not raw INDEC microdata and not the train subset of a train/test split. The historical upstream filenames used the word "train"; this repo treats the files as fixed upstream artifacts.

## Target

Target variable:

- `logP47T = log10(P47T)`

Inclusion criteria:

- `INGRESO == 1`
- `P47T > 0`
- `PROP` not missing

## Forbidden predictors

The following variables must not enter the feature matrix:

- target and direct transformations: `P47T`, `logP47T`
- identifiers: `CODUSU`
- derived temporal/date variables excluded from X: `Q`
- income components: `P21`, `T_VI`, `V12_M`, `V2_M`, `V3_M`, `V5_M`, `TOT_P12`, `PP08D1`
- transformed income components: `logP21`, `logT_VI`, `logV12_M`, `logV2_M`, `logV3_M`, `logV5_M`, `logTOT_P12`, `logPP08D1`
- income-source indicators: `INGRESO_NLB`, `INGRESO_JUB`, `INGRESO_SBS`

## Feature engineering

The baseline dataset builder must implement the feature transformations found in the legacy feature engineering script, including:

- annual file harmonization
- construction of `logP47T`
- categorical recodings
- use of `PP07G_59` as summarized labor-benefit indicator
- dropping detailed PP07G/PP07H/PP07I/PP07J/PP07K variables
- dropping `P07` and `P08`
- construction of `sanitacion_nivel`
- construction of household age composition variables
- construction of `Max_Nivel_Educativo`

## Splits

Use a random split with fixed seed.

- train: 70%
- test: 20%
- validation: 10%

The split assignment must be saved so every model uses exactly the same rows.

## Models

Baseline model families:

- LinearRegression
- Ridge
- Lasso
- HistGradientBoostingRegressor
- MLPRegressor

All models must use the same feature columns unless the experiment config explicitly declares otherwise.

## Metrics

For every model, report:

- CV mean R2
- CV std R2
- test R2
- test MAE
- test MSE
- validation R2
- validation MAE
- validation MSE
- fit time if available

## Required outputs

- `data/processed/modeling_dataset.parquet`
- `data/processed/split_assignments.csv`
- `reports/tables/model_comparison.csv`
- `reports/experiment_cards/baseline_income_prediction.json`
- thesis-ready LaTeX table generated from the CSV

## Non-goals

- Do not optimize neural network architecture deeply.
- Do not claim causal interpretation.
- Do not claim replacement of official income or poverty methodology.
- Do not introduce new model families before the baseline is reproducible.