# Diagnostics governance

Metrics compare.
Diagnostics explain.
Guards protect.
Artifacts remember.

This document governs `diagnostics` as a first-class experiment concept in the EPH income prediction framework. It is intentionally separate from guardrail work: diagnostics explain model behavior after a run has produced archived outputs, while guards protect validity and may warn or stop a run before fitting or reporting.

## Concept boundaries

### Metrics compare

Metrics are comparable performance quantities used to compare models or runs. They belong in `reports/runs/<run_id>/metrics/` or comparison tables.

Examples:

- R², MAE, MSE, RMSE, and bias.
- Cross-validation score means and standard deviations.
- Train-vs-test or train-vs-validation gaps when summarized as performance quantities.
- Fit time and other comparable runtime quantities.

### Diagnostics explain

Diagnostics are scientific artifacts used to understand how a model behaves. They belong in `reports/runs/<run_id>/diagnostics/` for tables and `reports/runs/<run_id>/plots/` for figures.

Examples:

- Residual summaries.
- Error by income decile.
- Prediction distribution summaries.
- Observed-vs-predicted plots.
- Residual distribution plots.
- Model pairwise error comparisons.
- Coefficient summaries for coefficient-bearing models.
- Fixed-effect coefficient exports.
- Regularization paths and alpha curves.
- HGB CV summaries, sweep summaries, and train-vs-CV gap plots.

### Guards protect

Guards are validity checks that may warn or fail before fitting or reporting. Guard behavior must not live under top-level `diagnostics:`. Guard behavior belongs under a separate top-level `guards:` namespace in a separate PR.

Examples of guard concepts include forbidden predictor checks, split integrity, fixed-effect cardinality limits, VIF, multicollinearity checks, and rank-deficiency checks. This PR does not implement VIF, multicollinearity checks, or any new guardrail behavior.

### Artifacts remember

Artifacts are persisted evidence that allow runs to be audited without ad hoc reruns. They are not inherently interpretive, though diagnostics may read them.

Examples:

- Predictions.
- `cv_results` exports.
- Feature columns.
- Config snapshots.
- Dataset cards.
- Training-frame samples.
- Run manifests.
- The run-local `diagnostics/diagnostics_plan.json` file.

## Current standard diagnostics

Standard diagnostics are expected for ordinary model-comparison runs where archived predictions and metrics exist:

- `residual_summary.csv`
- `error_by_income_decile.csv`
- `prediction_distribution_summary.csv`
- `model_pairwise_error_comparison.csv`
- `metric_gaps.csv`
- observed-vs-predicted plots
- prediction distribution plots
- residual distribution plots
- MAE by income decile plots
- mean residual by income decile plots

The experiment runner currently writes the first three run-time tables from saved predictions. The post-run diagnostics builder can regenerate and extend diagnostics from archived run artifacts.

## Model-specific diagnostics

Coefficient-bearing models such as LinearRegression, Ridge, and Lasso may produce coefficient summaries. Fixed-effect coefficient exports are only meaningful when fixed effects are used and a coefficient-bearing model is enabled.

HistGradientBoostingRegressor may produce HGB-specific CV result summaries, top-configuration summaries, overfit-gap summaries, and HGB plots.

## Sweep-specific diagnostics

Sweep diagnostics explain controlled hyperparameter studies. Current examples include:

- HGB sweep CV-result summaries.
- HGB single-parameter sweep plots.
- HGB learning-rate by iteration-budget sweep plots.
- Regularization path summaries and alpha-curve plots for Ridge/Lasso.

Sweep diagnostics should explain the configured scientific question; they should not introduce a new model, target, data source, or validation scheme.

## Run-time diagnostics vs post-run diagnostics

Some diagnostics are written during `run_experiment()` because the required fitted estimator or in-memory `GridSearchCV` result is already available. Examples include HGB CV summaries, regularization path summaries, and fixed-effect coefficient exports.

`src/eph_income/diagnostics.py` should remain post-run and archived-artifact based. It should read saved predictions, metrics, CV result CSVs, config snapshots, and run manifests. It should not train models, fit estimators, or call prediction APIs. This keeps diagnostics auditable and reproducible from a completed run directory.

## Diagnostics config and registry

Top-level `diagnostics:` config declares which explanatory artifacts are planned for a run. The registry normalizes old and new YAML shapes into `diagnostics_v1` and writes `diagnostics/diagnostics_plan.json` into each run directory.

Supported profiles are:

- `none`: disables optional diagnostics planning.
- `minimal`: plans only core run-time tables.
- `standard`: plans residuals, deciles, prediction distributions, metric gaps, and pairwise comparisons where possible.
- `sweep`: standard diagnostics plus sweep-specific diagnostics where applicable.
- `smoke_minimal`: cheap runtime sanity checks that should not be treated as thesis evidence.
- `thesis_core`: main thesis evidence for predictive comparison, distribution compression, and decile-error claims.
- `hgb_capacity_sweep`: HGB tuning evidence for operating regions, cost-performance tradeoffs, overfit gaps, and plateau behavior.
- `regularization_interpretation`: Ridge/Lasso interpretation of the linear frontier, alpha curves, coefficient norms, and sparsity.
- `geo_leakage_probe`: controlled geography variants for clean geography, no geography, target-derived ranks, and shuffled-rank placebo checks.
- `linear_fe_interpretation`: fixed-effect linear interpretation with coefficient and reference-level diagnostics.

## Why guards are separate

Diagnostics should not decide whether a run is valid enough to fit. Anything that can stop a fit or reporting step is a guard and belongs under top-level `guards:`. Keeping `diagnostics:` explanatory avoids mixing scientific interpretation with safety policy.

## Backward compatibility

Existing configs using old top-level sweep keys under `diagnostics:` remain supported. For example:

```yaml
diagnostics:
  sweep_type: hgb_single_param
  primary_param: max_leaf_nodes
  group_param: learning_rate
```

is normalized internally to a sweep profile with `diagnostics.sweeps` enabled. Likewise:

```yaml
diagnostics:
  regularization_sweep: true
```

is normalized internally to a sweep profile with `diagnostics.regularization.enabled: true`.

Configs do not need a mass migration. New or edited configs should prefer explicit `diagnostics_v1`-style sections for readability.
