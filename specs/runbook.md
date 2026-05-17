# EPH Income Modeling Runbook

This runbook explains how to run the repository end to end with the real annual EPH-derived input files, train the baseline models, archive a local run, and build the scientific diagnostics and plots from that archived run.

The workflow is intentionally evidence-first:

```text
annual_preprocessed_inputs
→ data/processed/modeling_dataset.parquet
→ data/processed/split_assignments.csv
→ reports/runs/<run_id>/
→ diagnostics tables and plots derived from archived run artifacts
```

The run directory is the canonical owner of experiment evidence. Do not treat notebooks, copied CSVs, or plot files as the primary source of truth.

---

## 0. Repository assumptions

Run every command from the repository root:

```bash
cd /workspace/income-modeling-eph
```

The expected project command surface is:

```bash
make install
make test
make build-dataset
make run-debug
make run-baseline
make build-diagnostics RUN_DIR=reports/runs/<run_id>
```

The `report` target currently exists, but the active run diagnostics and plots are produced by `scripts/04_build_diagnostics.py` / `make build-diagnostics`.

---

## 1. Create and install the Python environment

Recommended: use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install
```

This installs the package in editable mode with development dependencies.

Quick smoke test:

```bash
pytest -q
```

Optional lint check for maintained code:

```bash
ruff check src tests scripts
```

Avoid running `ruff check .` as the legacy student scripts may contain historical lint issues and are not part of the maintained migration surface.

---

## 2. Put the real input data in place

This repository expects fixed annual preprocessed EPH-derived CSV files from the upstream preprocessing pipeline. These are not raw INDEC microdata and are not split-specific train/test files.

Create the input directory if needed:

```bash
mkdir -p data/annual_preprocessed_inputs
```

Copy or symlink the four real annual input files into exactly these paths:

```text
data/annual_preprocessed_inputs/EPHARG_annual_input_22.csv
data/annual_preprocessed_inputs/EPHARG_annual_input_23.csv
data/annual_preprocessed_inputs/EPHARG_annual_input_24.csv
data/annual_preprocessed_inputs/EPHARG_annual_input_25.csv
```

Example using copies from an upstream artifact directory:

```bash
cp /path/to/upstream/EPHARG_annual_input_22.csv data/annual_preprocessed_inputs/
cp /path/to/upstream/EPHARG_annual_input_23.csv data/annual_preprocessed_inputs/
cp /path/to/upstream/EPHARG_annual_input_24.csv data/annual_preprocessed_inputs/
cp /path/to/upstream/EPHARG_annual_input_25.csv data/annual_preprocessed_inputs/
```

Verify the files exist:

```bash
python scripts/01_build_dataset.py --check-only
```

Expected behavior: the command prints JSON with the configured input files, processed dataset path, metadata path, and split assignment path.

---

## 3. Build the modeling dataset and split registry

Build the processed modeling dataset from the real annual files:

```bash
make build-dataset
```

This writes:

```text
data/processed/modeling_dataset.parquet
data/processed/dataset_metadata.json
data/processed/split_assignments.csv
```

The dataset builder applies the baseline inclusion and target contract:

```text
INGRESO == 1
P47T > 0
PROP not missing
logP47T = log10(P47T)
```

It also removes forbidden predictors from the candidate feature matrix.

If you intentionally need to recreate the split assignments, run:

```bash
python scripts/01_build_dataset.py --overwrite-splits
```

Do this only when you explicitly want a refreshed split file. Normal reruns should reuse the existing split registry.

Sanity-check the processed outputs:

```bash
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

metadata = json.loads(Path('data/processed/dataset_metadata.json').read_text())
dataset = pd.read_parquet('data/processed/modeling_dataset.parquet')
splits = pd.read_csv('data/processed/split_assignments.csv')

print('rows:', len(dataset))
print('feature_count:', metadata['feature_count'])
print('target:', metadata['target_definition'])
print('split_counts:')
print(splits['split'].value_counts())
PY
```

---

## 4. Run a debug experiment first

Before launching the full baseline, run the debug config. This verifies that the dataset, splits, model pipelines, run archive, prediction files, CV results, and diagnostics scaffolding work.

```bash
make run-debug
```

The command prints JSON including a canonical `run_dir`, for example:

```json
{
  "model_comparison": "/workspace/income-modeling-eph/reports/tables/model_comparison_debug.csv",
  "models": ["LinearRegression", "Ridge"],
  "rows": 5000,
  "run_dir": "/workspace/income-modeling-eph/reports/runs/debug_experiment_YYYYMMDDTHHMMSSZ"
}
```

Capture the latest debug run directory:

```bash
RUN_DIR="$(find reports/runs -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
echo "$RUN_DIR"
```

Expected archived run structure:

```text
reports/runs/<run_id>/
  run_manifest.json
  config_used.yaml
  feature_contract_used.yaml
  dataset_card.json
  feature_columns.json
  metrics/
    model_comparison.csv
    metrics_long.csv
  predictions/
    test_predictions.parquet
    validation_predictions.parquet
  cv_results/
    LinearRegression.csv
    Ridge.csv
  diagnostics/
  artifacts/
```

Verify the archived run exists:

```bash
find "$RUN_DIR" -maxdepth 2 -type f | sort
```

---

## 5. Build diagnostics and plots for a run

Diagnostics are derived from archived artifacts. They read saved predictions, metrics, CV result CSVs, and the run manifest. They do not retrain models and do not call `predict`.

For the default `test` split:

```bash
python scripts/04_build_diagnostics.py --run-dir "$RUN_DIR"
```

Equivalent Makefile target:

```bash
make build-diagnostics RUN_DIR="$RUN_DIR"
```

For validation-split plots:

```bash
python scripts/04_build_diagnostics.py --run-dir "$RUN_DIR" --split validation
```

or:

```bash
make build-diagnostics RUN_DIR="$RUN_DIR" SPLIT=validation
```

Diagnostics refresh or create:

```text
reports/runs/<run_id>/diagnostics/residual_summary.csv
reports/runs/<run_id>/diagnostics/error_by_income_decile.csv
reports/runs/<run_id>/diagnostics/model_pairwise_error_comparison.csv
reports/runs/<run_id>/diagnostics/metric_gaps.csv
reports/runs/<run_id>/diagnostics/diagnostics_build_notes.md
```

Plots are written to:

```text
reports/runs/<run_id>/plots/prediction_distribution_by_model.png
reports/runs/<run_id>/plots/observed_vs_predicted_best_model.png
reports/runs/<run_id>/plots/mae_by_income_decile.png
reports/runs/<run_id>/plots/mean_residual_by_income_decile.png
reports/runs/<run_id>/plots/residual_distribution_by_model.png
reports/runs/<run_id>/plots/ridge_alpha_curve.png      # if Ridge CV alpha data exists
reports/runs/<run_id>/plots/lasso_alpha_curve.png      # if Lasso CV alpha data exists
```

If a plot cannot be generated, the diagnostics builder skips it and records the reason in:

```text
reports/runs/<run_id>/diagnostics/diagnostics_build_notes.md
```

For example, a debug run with only LinearRegression and Ridge will usually skip `lasso_alpha_curve.png`.

Quick inspection commands:

```bash
cat "$RUN_DIR/diagnostics/diagnostics_build_notes.md"
find "$RUN_DIR/plots" -maxdepth 1 -type f | sort
```

---

## 6. Open the actual plots

On a local machine with a graphical file browser, open the PNG files directly from:

```text
reports/runs/<run_id>/plots/
```

From a terminal on Linux, common options are:

```bash
xdg-open "$RUN_DIR/plots/observed_vs_predicted_best_model.png"
xdg-open "$RUN_DIR/plots/mae_by_income_decile.png"
xdg-open "$RUN_DIR/plots/mean_residual_by_income_decile.png"
```

In a remote/container environment, copy the plots out of the container or archive them:

```bash
tar -czf diagnostics_plots.tgz -C "$RUN_DIR" plots diagnostics
```

Then download `diagnostics_plots.tgz` and inspect locally.

---

## 7. Run the full baseline on the real data

The full baseline is guarded because the configured model grids can be expensive, especially HistGradientBoostingRegressor and MLPRegressor.

After the debug run succeeds, launch the full baseline:

```bash
make run-baseline
```

Equivalent explicit command:

```bash
python scripts/02_run_baseline_experiment.py \
  --config configs/experiment_baseline.yaml \
  --allow-full-run
```

The command writes a full run directory under:

```text
reports/runs/<run_id>/
```

It also writes a convenience copy of model comparison metrics to:

```text
reports/tables/model_comparison.csv
```

Capture the full run directory from the command output. If needed, use:

```bash
RUN_DIR="$(find reports/runs -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
echo "$RUN_DIR"
```

Then build diagnostics and plots for the full run:

```bash
make build-diagnostics RUN_DIR="$RUN_DIR"
```

Optionally also render validation-split plots:

```bash
make build-diagnostics RUN_DIR="$RUN_DIR" SPLIT=validation
```

---

## 8. End-to-end command sequence

Use this sequence for a full real-data run from a clean environment after placing the four annual input CSVs:

```bash
cd /workspace/income-modeling-eph
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install
pytest -q
python scripts/01_build_dataset.py --check-only
make build-dataset
make run-debug
DEBUG_RUN_DIR="$(find reports/runs -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
make build-diagnostics RUN_DIR="$DEBUG_RUN_DIR"
make run-baseline
FULL_RUN_DIR="$(find reports/runs -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
make build-diagnostics RUN_DIR="$FULL_RUN_DIR"
make build-diagnostics RUN_DIR="$FULL_RUN_DIR" SPLIT=validation
tar -czf full_run_diagnostics.tgz -C "$FULL_RUN_DIR" plots diagnostics metrics cv_results run_manifest.json dataset_card.json feature_columns.json
```

After this, the main scientific evidence for the full run is in:

```text
$FULL_RUN_DIR
```

The plot bundle is:

```text
full_run_diagnostics.tgz
```

---

## 9. What each major artifact means

### `data/processed/modeling_dataset.parquet`

The target-filtered modeling dataset with stable `row_id`, `logP47T`, and allowed feature columns.

### `data/processed/split_assignments.csv`

The reusable random individual-level train/test/validation assignment. All baseline models use the same split file.

### `reports/runs/<run_id>/run_manifest.json`

The top-level manifest for a run. It identifies the run, mode, models, feature count, row count, and canonical artifact paths.

### `reports/runs/<run_id>/dataset_card.json`

Dataset lineage for the run, including source file configuration, processed dataset path, split assignment path, target, row count, feature count, digest information, and split counts.

### `reports/runs/<run_id>/metrics/model_comparison.csv`

Canonical model comparison table for the run. The top-level `reports/tables/model_comparison.csv` is only a convenience copy.

### `reports/runs/<run_id>/predictions/*.parquet`

Canonical archived predictions and residuals for test and validation splits. Diagnostics and plots should derive from these files, not from ad-hoc model prediction calls.

### `reports/runs/<run_id>/cv_results/*.csv`

Archived GridSearchCV results per model. Ridge/Lasso alpha curve plots are built from these files when available.

### `reports/runs/<run_id>/diagnostics/*.csv`

Derived scientific diagnostics from archived predictions and metrics.

### `reports/runs/<run_id>/plots/*.png`

Derived visual diagnostics from archived predictions, metrics, and CV results.

---

## 10. Troubleshooting

### Missing input files

Symptom:

```text
Configured input files do not exist
```

Fix: place the four real annual CSVs in `data/annual_preprocessed_inputs/` with the exact expected names and rerun:

```bash
python scripts/01_build_dataset.py --check-only
```

### Full training refuses to run

Symptom:

```text
Full baseline training is guarded because it can be expensive
```

Fix: use the guarded full command:

```bash
python scripts/02_run_baseline_experiment.py --config configs/experiment_baseline.yaml --allow-full-run
```

or:

```bash
make run-baseline
```

### Diagnostics cannot find a run

Symptom:

```text
No archived prediction files found
```

Fix: pass the actual run directory, not `reports/runs` itself:

```bash
make build-diagnostics RUN_DIR=reports/runs/<run_id>
```

### A plot is missing

Check:

```bash
cat "$RUN_DIR/diagnostics/diagnostics_build_notes.md"
```

Some plots are conditional. For example, `lasso_alpha_curve.png` requires an archived Lasso CV results file with an alpha column and `mean_test_score`.

### Need a clean rerun

Generated data and report artifacts are ignored by git. You can remove generated outputs and rebuild:

```bash
rm -rf data/processed reports/runs reports/tables reports/experiment_cards
make build-dataset
make run-debug
```

Be careful: deleting `data/processed/split_assignments.csv` changes which split file is reused. If you need strict reproducibility, preserve the split assignment file.

---

## 11. Scientific guardrails

- The target is `log10(P47T)`, not natural log.
- Metrics and residuals are on the base-10 log income scale.
- The baseline split is random individual-level train/test/validation.
- This is a predictive baseline, not a causal analysis.
- Do not modify legacy scripts to make the modern pipeline run.
- Do not add new model families or change model grids unless a new spec explicitly requests it.
- Do not treat plots as canonical data. Plots are derived artifacts; predictions, metrics, CV results, and manifests are the evidence base.

