# Guards governance

This repository now treats **guards** as a first-class execution governance layer, without
renaming or replacing `contracts.py`.

## Concepts

- **Metrics compare** model performance across models, runs, and feature views.
- **Diagnostics explain** model behavior using saved outputs; they do not retrain models.
- **Guards protect** the pipeline at execution time and may pass, warn, fail, or skip.
- **Artifacts remember** run-local evidence so a run can be audited after completion.

## Contracts vs. guards

Contracts are pure scientific invariants. They define facts such as the target being
`log10(P47T)`, the forbidden predictor set, and the temporal feature decision. Contracts
remain in `src/eph_income/contracts.py`.

Guards are execution-layer checks. A guard applies a contract or another validation at a
specific pipeline stage, records a decision, and decides whether the run may continue.
Guards live in `src/eph_income/guards.py` and write run-local evidence under
`reports/runs/<run_id>/guards/`.

Rules:

1. If a check can stop a run because the experiment would be invalid or misleading, it
   belongs in guards.
2. If a check is a pure scientific invariant, keep it in `contracts.py` and call it from
   guards.
3. Do not put guards under diagnostics. Diagnostics explain; guards protect.
4. Do not migrate every existing `ValueError` at once. Governance should grow through
   small, auditable seams.

## Guard stages

Guard stages describe where the check protects the pipeline:

- `config`
- `contract`
- `data_input`
- `feature_view`
- `fixed_effects`
- `pre_fit`
- `post_fit`
- `output`
- `runtime_cost`

The initial implementation governs or documents these checks:

| Guard | Stage | Current behavior |
| --- | --- | --- |
| `target_contract` | `contract` | Calls `validate_target_contract`; fails the run for an invalid target contract. |
| `forbidden_predictors` | `contract/pre_fit` | Calls `get_forbidden_predictors` and `assert_no_forbidden_predictors`; fails the run if leakage columns enter the feature set. |
| `runtime_mode` | `runtime_cost` | Records the result of existing runtime-mode and full-run validation. |
| `fixed_effect_design` | `fixed_effects` | Records the result of existing fixed-effect design validation. |
| `multicollinearity` | `pre_fit` | Optional raw-feature guard for constants, exact duplicates, numeric rank deficiency, and VIF. |

## Guard statuses and actions

Statuses:

- `pass`: the guard condition was satisfied.
- `warn`: the guard found a concern, but the configured policy permits continuation.
- `fail`: the guard found a violation.
- `skip`: the guard was disabled or not applicable.

Actions:

- `continue`: proceed silently.
- `warn_continue`: proceed while surfacing the concern.
- `fail_run`: stop the run.

YAML may use `action: fail` as a shorthand for `fail_run`, and `action: warn` as a
shorthand for `warn_continue`.

## YAML configuration

Top-level guard configuration is optional. Defaults preserve current behavior for
contract and runtime guards, while leaving unimplemented guards disabled.

```yaml
guards:
  enabled: true
  target_contract:
    enabled: true
    action: fail
  forbidden_predictors:
    enabled: true
    action: fail
  runtime_mode:
    enabled: true
    action: fail
  fixed_effect_design:
    enabled: true
    action: fail
  multicollinearity:
    enabled: true
    stage: pre_fit
    sample_n: 5000
    warn_vif: 5.0
    fail_vif: 10.0
    max_vif_features: 200
    fail_rank_deficiency: true
    fail_duplicate_columns: true
    fail_constant_columns: false
    fail_models:
      - linear_regression
    warn_only_models:
      - ridge
      - lasso
      - hist_gradient_boosting
      - mlp
```

Unknown guard names fail config validation. A misspelled guard should not silently create
an unaudited run.

## Run-local artifacts

Each run directory includes:

```text
reports/runs/<run_id>/guards/guard_plan.json
reports/runs/<run_id>/guards/guard_decisions.json
```

`guard_plan.json` records:

- `run_id`
- `schema_version: guards_v1`
- enabled guards
- stage
- default action
- notes

`guard_decisions.json` records:

- `run_id`
- `schema_version: guards_v1`
- decisions list
- decision `name`, `stage`, `status`, `action`, `reasons`, and `details`

The run manifest includes paths to both guard artifacts.

## Preliminary audit note

The preliminary audit commands requested for this change were run against `src`, `tests`,
`configs`, `scripts`, and `Makefile`. Existing implicit guards found during the audit:

| Existing implicit guard | File/function | Stage | Current behavior | Governance decision |
| --- | --- | --- | --- | --- |
| Target definition (`logP47T` from `P47T` using `log10`) | `contracts.py::validate_target_contract` | `contract` | Raises `ValueError` on invalid target contract. | Keep in contracts and call from `target_contract` guard. |
| Temporal feature contract | `contracts.py::validate_temporal_features` | `contract` | Raises `ValueError` on missing `ANO4`/`TRIMESTRE` or wrong decision. | Stay local for now; future governed contract guard candidate. |
| Forbidden predictor flattening and assertion | `contracts.py::get_forbidden_predictors`, `contracts.py::assert_no_forbidden_predictors` | `contract/pre_fit` | Raises `TypeError`/`ValueError` on malformed contract or leakage columns. | Keep in contracts and call from `forbidden_predictors` guard. |
| Dataset input existence | `dataset.py::validate_dataset_inputs`, `experiments.py::load_processed_dataset` | `data_input` | Fails if required files are missing. | Stay local initially; future `data_input` guard candidate. |
| Split integrity | `splits.py::validate_split_assignments`, `experiments.py::load_split_assignments` | `data_input/pre_fit` | Fails on missing/invalid split rows or labels. | Stay local initially; future governed guard candidate. |
| Runtime mode and full-run safety | `experiments.py::_validate_runtime_mode` | `runtime_cost` | Fails on unsupported modes or unapproved full runs. | Represent in `runtime_mode` guard decisions; keep logic local. |
| Feature-view schema and permutation safety | `experiments.py::apply_feature_view` | `feature_view` | Fails on malformed `drop_columns`/`permute_group_values` and forbidden predictors after view. | Stay local initially; future `feature_view` guard candidate. |
| Fixed-effect design limits | `experiments.py::apply_fixed_effects` | `fixed_effects` | Fails on malformed specs, missing source columns, excessive levels, conflicts, and forbidden predictors. | Represent in `fixed_effect_design` guard decisions; keep logic local. |
| Non-empty train/test/validation splits | `experiments.py::run_experiment` | `pre_fit` | Fails if any split is empty. | Stay local initially; future `pre_fit` guard candidate. |
| Model config support | `pipelines.py::enabled_model_configs` and pipeline construction | `config/pre_fit` | Fails on unsupported model keys or invalid enabled model configs. | Stay local initially; future `config` guard candidate. |
| Output feature-count consistency | `experiments.py::_validate_feature_count_consistency` | `output` | Fails if comparison rows disagree on feature count. | Stay local initially; future `output` guard candidate. |
| Raw-feature multicollinearity | `guards.py::build_multicollinearity_audit`, `guards.py::decide_multicollinearity_action` | `pre_fit` | When enabled, writes raw-feature multicollinearity artifacts and warns/fails according to model policy. | Governed guard. |

## Multicollinearity guard

The `multicollinearity` guard is a statistical `pre_fit` guard. It audits raw final
feature columns after sampling, feature views, fixed effects, split joining, and forbidden
predictor checks. It intentionally does not audit the post-one-hot sklearn design matrix
yet, because transformed matrices can become large and fragile.

Implemented behavior:

- detect constant raw columns,
- detect exact duplicate raw columns,
- detect numeric rank deficiency on sampled numeric raw features,
- compute numeric raw-feature VIF when the numeric feature count is at or below
  `max_vif_features`,
- fail `LinearRegression` by default on duplicate columns, rank deficiency, or
  `VIF > fail_vif`,
- warn `Ridge`/`Lasso` by default,
- warn non-linear benchmarks such as `HistGradientBoostingRegressor` and `MLPRegressor`
  by default without blocking them.

Run-local artifacts when enabled:

```text
reports/runs/<run_id>/guards/multicollinearity_raw_features.csv
reports/runs/<run_id>/guards/multicollinearity_raw_summary.json
```

TODO: add a future transformed-design audit for post-preprocessing one-hot matrices.
