"""Experiment runner for debug and baseline model comparisons."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from eph_income.contracts import assert_no_forbidden_predictors, get_forbidden_predictors, validate_target_contract
from eph_income.dataset import ROW_ID_COLUMN, get_metadata_path, resolve_project_path
from eph_income.pipelines import MODEL_DISPLAY_NAMES, enabled_model_configs, make_model_pipeline
from eph_income.splits import get_split_path, validate_split_assignments

PREDICTION_COLUMNS = [
    "run_id",
    "model",
    "split",
    ROW_ID_COLUMN,
    "y_true",
    "y_pred",
    "residual",
    "abs_error",
    "squared_error",
]


def _resolve_output_path(path: str | Path) -> Path:
    return resolve_project_path(path)


def load_processed_dataset(experiment_config: Mapping[str, Any]) -> pd.DataFrame:
    path = _resolve_output_path(experiment_config["data"]["processed_dataset"])
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset does not exist: {path}")
    return pd.read_parquet(path)


def load_split_assignments(experiment_config: Mapping[str, Any], dataset: pd.DataFrame) -> pd.DataFrame:
    path = get_split_path(experiment_config)
    if not path.exists():
        raise FileNotFoundError(f"Split assignments do not exist: {path}")
    assignments = pd.read_csv(path)
    validate_split_assignments(dataset, assignments)
    return assignments


def apply_debug_sample(
    joined: pd.DataFrame, *, sample_n: int | None, random_state: int, min_train_rows: int
) -> pd.DataFrame:
    """Sample rows for debug mode while retaining all split labels."""

    if sample_n is None or sample_n >= len(joined):
        return joined.copy()
    fractions = joined["split"].value_counts(normalize=True).to_dict()
    sampled_parts = []
    remaining = sample_n
    for split in ["train", "test", "validation"]:
        split_frame = joined[joined["split"] == split]
        if split_frame.empty:
            continue
        desired = int(round(sample_n * fractions.get(split, 0.0)))
        if split == "train":
            desired = max(desired, min_train_rows)
        else:
            desired = max(desired, 2)
        desired = min(desired, len(split_frame), remaining if remaining > 0 else len(split_frame))
        sampled_parts.append(split_frame.sample(n=desired, random_state=random_state))
        remaining -= desired
    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) < sample_n:
        missing = sample_n - len(sampled)
        extra_pool = joined[~joined[ROW_ID_COLUMN].isin(sampled[ROW_ID_COLUMN])]
        if not extra_pool.empty:
            sampled = pd.concat(
                [
                    sampled,
                    extra_pool.sample(n=min(missing, len(extra_pool)), random_state=random_state),
                ],
                ignore_index=True,
            )
    return sampled.sort_values(ROW_ID_COLUMN).reset_index(drop=True)


def prepare_experiment_frame(
    experiment_config: Mapping[str, Any], feature_contract: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[str], str]:
    """Load dataset/splits, optionally sample debug rows, and return feature columns."""

    dataset = load_processed_dataset(experiment_config)
    assignments = load_split_assignments(experiment_config, dataset)
    joined = dataset.merge(assignments, on=ROW_ID_COLUMN, how="inner", validate="one_to_one")

    cv_folds = int(experiment_config.get("cv", {}).get("folds", 5))
    runtime = _runtime_mapping(experiment_config)
    sample_n = runtime.get("sample_n")
    if sample_n is not None:
        sample_n = int(sample_n)
    random_seed = int(experiment_config.get("experiment", {}).get("random_seed", 42))
    joined = apply_debug_sample(
        joined, sample_n=sample_n, random_state=random_seed, min_train_rows=max(cv_folds, 2)
    )

    target = validate_target_contract(feature_contract)["name"]
    forbidden = get_forbidden_predictors(feature_contract)
    feature_columns = [column for column in dataset.columns if column not in {ROW_ID_COLUMN, target}]
    assert_no_forbidden_predictors(feature_columns, forbidden)
    return joined, feature_columns, target


def _evaluate(model: Any, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "r2": float(r2_score(target, predictions)),
        "mae": float(mean_absolute_error(target, predictions)),
        "mse": float(mean_squared_error(target, predictions)),
    }


def _runtime_mapping(experiment_config: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = experiment_config.get("runtime", {})
    return runtime if isinstance(runtime, Mapping) else {}


def _validate_runtime_mode(experiment_config: Mapping[str, Any], *, allow_full_run: bool) -> str:
    """Validate runtime guardrails and return the configured runtime mode."""

    runtime = _runtime_mapping(experiment_config)
    mode = str(runtime.get("mode", "full"))
    if mode == "debug":
        return mode
    if mode != "full":
        raise ValueError(f"Unsupported runtime.mode: {mode}")

    config_allows_full = bool(runtime.get("allow_full_run", False))
    if not allow_full_run and not config_allows_full:
        raise ValueError(
            "Full baseline training is guarded because it can be expensive. "
            "Pass --allow-full-run or set runtime.allow_full_run: true in the config."
        )
    return mode


def _validate_feature_count_consistency(comparison: pd.DataFrame) -> None:
    """Fail loudly if model rows do not report a common feature count."""

    if comparison.empty:
        raise ValueError("Experiment produced no model result rows.")
    if "feature_count" not in comparison.columns:
        raise ValueError("Experiment results must include feature_count.")
    if comparison["feature_count"].nunique(dropna=False) != 1:
        raise ValueError("All enabled models must report the same feature_count.")


def _safe_run_component(value: str) -> str:
    """Return a filesystem-safe run ID component."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def _make_run_id(experiment_config: Mapping[str, Any]) -> str:
    experiment_id = str(experiment_config.get("experiment", {}).get("id", "experiment"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_safe_run_component(experiment_id)}_{timestamp}"


def _prepare_run_directory(experiment_config: Mapping[str, Any], run_id: str) -> Path:
    outputs = experiment_config.get("outputs", {})
    base_dir = outputs.get("runs_dir", "reports/runs")
    run_dir = _resolve_output_path(base_dir) / run_id
    if run_dir.exists():
        suffix = datetime.now(timezone.utc).strftime("%f")
        run_dir = run_dir.with_name(f"{run_dir.name}_{suffix}")
    for child in ["metrics", "predictions", "cv_results", "diagnostics", "artifacts"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_yaml_snapshot(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True), encoding="utf-8")


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset_metadata(experiment_config: Mapping[str, Any]) -> dict[str, Any]:
    processed_path = _resolve_output_path(experiment_config["data"]["processed_dataset"])
    metadata_path = get_metadata_path(processed_path)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _make_dataset_card(
    experiment_config: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    joined: pd.DataFrame,
    feature_columns: list[str],
    target: str,
) -> dict[str, Any]:
    processed_path = _resolve_output_path(experiment_config["data"]["processed_dataset"])
    split_path = get_split_path(experiment_config)
    metadata = _load_dataset_metadata(experiment_config)
    split_counts = joined["split"].value_counts().reindex(["train", "test", "validation"], fill_value=0)
    data_config = experiment_config.get("data", {})
    source = data_config.get("input_files") or metadata.get("input_files") or metadata.get("upstream_input_files")
    return {
        "dataset_name": "modeling_dataset",
        "source": source,
        "processed_dataset": str(processed_path),
        "split_assignments": str(split_path),
        "target": target,
        "target_definition": feature_contract.get("target", {}),
        "row_count": int(len(joined)),
        "feature_count": int(len(feature_columns)),
        "digest": _sha256_file(processed_path),
        "split_digest": _sha256_file(split_path),
        "splits": {split: int(count) for split, count in split_counts.items()},
        "metadata": metadata,
    }


def _prediction_frame(
    *,
    run_id: str,
    model_name: str,
    split_name: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    estimator: Any,
) -> pd.DataFrame:
    y_true = frame[target].to_numpy()
    y_pred = estimator.predict(frame[feature_columns])
    predictions = pd.DataFrame(
        {
            "run_id": run_id,
            "model": model_name,
            "split": split_name,
            ROW_ID_COLUMN: frame[ROW_ID_COLUMN].to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    predictions["residual"] = predictions["y_true"] - predictions["y_pred"]
    predictions["abs_error"] = predictions["residual"].abs()
    predictions["squared_error"] = predictions["residual"] ** 2
    try:
        predictions["income_decile"] = pd.qcut(
            predictions["y_true"], q=10, labels=False, duplicates="drop"
        )
        predictions["income_decile"] = predictions["income_decile"].astype("Int64") + 1
    except ValueError:
        predictions["income_decile"] = pd.Series([pd.NA] * len(predictions), dtype="Int64")
    return predictions


def _metrics_long(comparison: pd.DataFrame, run_id: str) -> pd.DataFrame:
    metric_columns = [
        column
        for column in comparison.columns
        if column not in {"model", "best_params"} and pd.api.types.is_numeric_dtype(comparison[column])
    ]
    long = comparison.melt(
        id_vars=["model"], value_vars=metric_columns, var_name="metric", value_name="value"
    )
    long.insert(0, "run_id", run_id)
    return long


def _write_predictions(predictions_by_split: dict[str, list[pd.DataFrame]], run_dir: Path) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for split_name in ["test", "validation"]:
        predictions = pd.concat(predictions_by_split[split_name], ignore_index=True)
        missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
        if missing:
            raise ValueError("Prediction artifact is missing required columns: " + ", ".join(missing))
        path = run_dir / "predictions" / f"{split_name}_predictions.parquet"
        predictions.to_parquet(path, index=False)
        written[split_name] = path
    return written


def _read_saved_predictions(run_dir: Path) -> pd.DataFrame:
    frames = []
    for split_name in ["test", "validation"]:
        path = run_dir / "predictions" / f"{split_name}_predictions.parquet"
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def _write_diagnostics_from_predictions(run_dir: Path) -> dict[str, Path]:
    predictions = _read_saved_predictions(run_dir)
    residual_summary = (
        predictions.groupby(["run_id", "model", "split"], dropna=False)
        .agg(
            row_count=(ROW_ID_COLUMN, "count"),
            residual_mean=("residual", "mean"),
            residual_std=("residual", "std"),
            residual_min=("residual", "min"),
            residual_max=("residual", "max"),
            abs_error_mean=("abs_error", "mean"),
            squared_error_mean=("squared_error", "mean"),
        )
        .reset_index()
    )
    error_by_decile = (
        predictions.groupby(["run_id", "model", "split", "income_decile"], dropna=False)
        .agg(
            row_count=(ROW_ID_COLUMN, "count"),
            y_true_mean=("y_true", "mean"),
            y_pred_mean=("y_pred", "mean"),
            residual_mean=("residual", "mean"),
            abs_error_mean=("abs_error", "mean"),
            squared_error_mean=("squared_error", "mean"),
        )
        .reset_index()
    )
    distribution_summary = (
        predictions.groupby(["run_id", "model", "split"], dropna=False)
        .agg(
            row_count=(ROW_ID_COLUMN, "count"),
            y_true_mean=("y_true", "mean"),
            y_true_std=("y_true", "std"),
            y_true_min=("y_true", "min"),
            y_true_p25=("y_true", lambda s: s.quantile(0.25)),
            y_true_median=("y_true", "median"),
            y_true_p75=("y_true", lambda s: s.quantile(0.75)),
            y_true_max=("y_true", "max"),
            y_pred_mean=("y_pred", "mean"),
            y_pred_std=("y_pred", "std"),
            y_pred_min=("y_pred", "min"),
            y_pred_p25=("y_pred", lambda s: s.quantile(0.25)),
            y_pred_median=("y_pred", "median"),
            y_pred_p75=("y_pred", lambda s: s.quantile(0.75)),
            y_pred_max=("y_pred", "max"),
        )
        .reset_index()
    )
    paths = {
        "residual_summary": run_dir / "diagnostics" / "residual_summary.csv",
        "error_by_income_decile": run_dir / "diagnostics" / "error_by_income_decile.csv",
        "prediction_distribution_summary": run_dir
        / "diagnostics"
        / "prediction_distribution_summary.csv",
    }
    residual_summary.to_csv(paths["residual_summary"], index=False)
    error_by_decile.to_csv(paths["error_by_income_decile"], index=False)
    distribution_summary.to_csv(paths["prediction_distribution_summary"], index=False)
    return paths


def run_experiment(
    experiment_config: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    *,
    allow_full_run: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a debug or guarded full baseline model comparison and write outputs."""

    mode = _validate_runtime_mode(experiment_config, allow_full_run=allow_full_run)
    run_id = _make_run_id(experiment_config)
    run_dir = _prepare_run_directory(experiment_config, run_id)

    joined, feature_columns, target = prepare_experiment_frame(experiment_config, feature_contract)
    train = joined[joined["split"] == "train"]
    test = joined[joined["split"] == "test"]
    validation = joined[joined["split"] == "validation"]
    if train.empty or test.empty or validation.empty:
        raise ValueError("Train, test, and validation splits must all be non-empty.")

    cv_folds = int(experiment_config.get("cv", {}).get("folds", 5))
    scoring = str(experiment_config.get("cv", {}).get("scoring", "r2"))
    random_seed = int(experiment_config.get("experiment", {}).get("random_seed", 42))

    _write_yaml_snapshot(run_dir / "config_used.yaml", experiment_config)
    _write_yaml_snapshot(run_dir / "feature_contract_used.yaml", feature_contract)
    dataset_card = _make_dataset_card(
        experiment_config, feature_contract, joined, feature_columns, target
    )
    (run_dir / "dataset_card.json").write_text(
        json.dumps(dataset_card, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2, sort_keys=True), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    predictions_by_split: dict[str, list[pd.DataFrame]] = {"test": [], "validation": []}
    for model_key, model_config in enabled_model_configs(experiment_config).items():
        model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        pipeline = make_model_pipeline(model_key, train[feature_columns], random_seed)
        grid = model_config.get("grid", {})
        search = GridSearchCV(
            pipeline,
            grid,
            cv=cv_folds,
            scoring=scoring,
            refit=True,
            n_jobs=None,
        )
        start = time.perf_counter()
        search.fit(train[feature_columns], train[target])
        fit_time = time.perf_counter() - start

        cv_results = pd.DataFrame(search.cv_results_)
        cv_results.insert(0, "run_id", run_id)
        cv_results.insert(1, "model", model_name)
        cv_results.to_csv(run_dir / "cv_results" / f"{model_name}.csv", index=False)

        test_metrics = _evaluate(search.best_estimator_, test[feature_columns], test[target])
        validation_metrics = _evaluate(
            search.best_estimator_, validation[feature_columns], validation[target]
        )
        for split_name, split_frame in [("test", test), ("validation", validation)]:
            predictions_by_split[split_name].append(
                _prediction_frame(
                    run_id=run_id,
                    model_name=model_name,
                    split_name=split_name,
                    frame=split_frame,
                    feature_columns=feature_columns,
                    target=target,
                    estimator=search.best_estimator_,
                )
            )
        best_index = int(search.best_index_)
        rows.append(
            {
                "model": model_name,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "feature_count": int(len(feature_columns)),
                "cv_r2_mean": float(search.cv_results_["mean_test_score"][best_index]),
                "cv_r2_std": float(search.cv_results_["std_test_score"][best_index]),
                "test_r2": test_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_mse": test_metrics["mse"],
                "validation_r2": validation_metrics["r2"],
                "validation_mae": validation_metrics["mae"],
                "validation_mse": validation_metrics["mse"],
                "fit_time_seconds": float(fit_time),
            }
        )

    comparison = pd.DataFrame(rows)
    _validate_feature_count_consistency(comparison)
    outputs = experiment_config.get("outputs", {})
    output_path = _resolve_output_path(outputs["model_comparison"])
    card_path = _resolve_output_path(outputs["experiment_card"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.parent.mkdir(parents=True, exist_ok=True)

    run_model_comparison_path = run_dir / "metrics" / "model_comparison.csv"
    comparison.to_csv(run_model_comparison_path, index=False)
    _metrics_long(comparison, run_id).to_csv(run_dir / "metrics" / "metrics_long.csv", index=False)
    shutil.copyfile(run_model_comparison_path, output_path)
    prediction_paths = _write_predictions(predictions_by_split, run_dir)
    diagnostic_paths = _write_diagnostics_from_predictions(run_dir)

    manifest = {
        "run_id": run_id,
        "experiment_id": experiment_config.get("experiment", {}).get("id"),
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_standard": "mlflow_inspired_local_run_v1",
        "canonical_owner": "run_directory",
        "run_dir": str(run_dir),
        "feature_count": int(len(feature_columns)),
        "rows_used": int(len(joined)),
        "models": comparison["model"].tolist(),
        "paths": {
            "config_used": str(run_dir / "config_used.yaml"),
            "feature_contract_used": str(run_dir / "feature_contract_used.yaml"),
            "dataset_card": str(run_dir / "dataset_card.json"),
            "feature_columns": str(run_dir / "feature_columns.json"),
            "model_comparison": str(run_model_comparison_path),
            "metrics_long": str(run_dir / "metrics" / "metrics_long.csv"),
            "test_predictions": str(prediction_paths["test"]),
            "validation_predictions": str(prediction_paths["validation"]),
            "residual_summary": str(diagnostic_paths["residual_summary"]),
            "error_by_income_decile": str(diagnostic_paths["error_by_income_decile"]),
            "prediction_distribution_summary": str(
                diagnostic_paths["prediction_distribution_summary"]
            ),
            "model_comparison_convenience_copy": str(output_path),
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    card = {
        "experiment_id": experiment_config.get("experiment", {}).get("id"),
        "run_id": run_id,
        "mode": mode,
        "rows_used": int(len(joined)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "models": comparison["model"].tolist(),
        "model_comparison": str(output_path),
        "canonical_run_dir": str(run_dir),
        "canonical_model_comparison": str(run_model_comparison_path),
        "run_manifest": str(run_dir / "run_manifest.json"),
    }
    card_path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    return comparison, card
