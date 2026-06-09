"""Run-local artifacts for EPH income experiments."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from eph_income.contracts import assert_no_forbidden_predictors, get_forbidden_predictors
from eph_income.dataset import ROW_ID_COLUMN, get_metadata_path, resolve_project_path
from eph_income.splits import get_split_path

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


def _artifacts_mapping(experiment_config: Mapping[str, Any]) -> Mapping[str, Any]:
    artifacts = experiment_config.get("artifacts", {})
    return artifacts if isinstance(artifacts, Mapping) else {}


def _write_training_frame_sample(
    *,
    run_dir: Path,
    run_id: str,
    train: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    experiment_config: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
) -> dict[str, Path]:
    """Write a deterministic training-frame sample for scientific inspection."""

    artifacts_config = _artifacts_mapping(experiment_config)
    configured_sample_n = artifacts_config.get("training_frame_sample_n", 10)
    if configured_sample_n is None:
        return {}

    sample_n_requested = int(configured_sample_n)
    if sample_n_requested <= 0:
        return {}

    forbidden = get_forbidden_predictors(feature_contract)
    assert_no_forbidden_predictors(feature_columns, forbidden)

    columns_written = [ROW_ID_COLUMN, "split", target, *feature_columns]
    forbidden_sample_columns = sorted((set(columns_written) & forbidden) - {target})
    if forbidden_sample_columns:
        joined = ", ".join(forbidden_sample_columns)
        raise ValueError(f"Forbidden predictors present in training-frame sample columns: {joined}")

    missing_columns = [column for column in columns_written if column not in train.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"Training-frame sample columns are missing from train split: {joined}")

    random_seed = int(experiment_config.get("experiment", {}).get("random_seed", 42))
    sample_n_written = min(sample_n_requested, len(train))
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sample_path = artifacts_dir / "training_frame_sample.csv"
    metadata_path = artifacts_dir / "training_frame_sample_metadata.json"

    sample = train[columns_written].sample(n=sample_n_written, random_state=random_seed)
    sample = sample.sort_values(ROW_ID_COLUMN).reset_index(drop=True)
    sample.to_csv(sample_path, index=False)

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_n_requested": sample_n_requested,
        "sample_n_written": int(sample_n_written),
        "source_split": "train",
        "target": target,
        "feature_count": int(len(feature_columns)),
        "columns_written": columns_written,
        "random_state": random_seed,
        "note": "Sample drawn from the train split after runtime sampling and before model fitting.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "training_frame_sample": sample_path,
        "training_frame_sample_metadata": metadata_path,
    }


def _evaluate(model: Any, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "r2": float(r2_score(target, predictions)),
        "mae": float(mean_absolute_error(target, predictions)),
        "mse": float(mean_squared_error(target, predictions)),
    }


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
    for child in [
        "metrics",
        "predictions",
        "cv_results",
        "diagnostics",
        "artifacts",
        "plots",
        "guards",
    ]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_yaml_snapshot(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


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
    fixed_effects_used: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    processed_path = _resolve_output_path(experiment_config["data"]["processed_dataset"])
    split_path = get_split_path(experiment_config)
    metadata = _load_dataset_metadata(experiment_config)
    split_counts = (
        joined["split"].value_counts().reindex(["train", "test", "validation"], fill_value=0)
    )
    data_config = experiment_config.get("data", {})
    source = (
        data_config.get("input_files")
        or metadata.get("input_files")
        or metadata.get("upstream_input_files")
    )
    return {
        "dataset_name": "modeling_dataset",
        "source": source,
        "processed_dataset": str(processed_path),
        "split_assignments": str(split_path),
        "target": target,
        "target_definition": feature_contract.get("target", {}),
        "row_count": int(len(joined)),
        "feature_count": int(len(feature_columns)),
        "fixed_effects_used": fixed_effects_used or [],
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
        if column not in {"model", "best_params"}
        and pd.api.types.is_numeric_dtype(comparison[column])
    ]
    long = comparison.melt(
        id_vars=["model"], value_vars=metric_columns, var_name="metric", value_name="value"
    )
    long.insert(0, "run_id", run_id)
    return long


def _write_predictions(
    predictions_by_split: dict[str, list[pd.DataFrame]], run_dir: Path
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for split_name in ["test", "validation"]:
        predictions = pd.concat(predictions_by_split[split_name], ignore_index=True)
        missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
        if missing:
            raise ValueError(
                "Prediction artifact is missing required columns: " + ", ".join(missing)
            )
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




def _coefficient_table(estimator: Any) -> pd.DataFrame | None:
    """Return fitted linear-model coefficients from a pipeline when available."""

    if not hasattr(estimator, "named_steps") or "reg" not in estimator.named_steps:
        return None
    regressor = estimator.named_steps["reg"]
    if not hasattr(regressor, "coef_"):
        return None
    coefficients = np.ravel(regressor.coef_).astype(float)
    preprocessor = estimator.named_steps.get("preproc")
    try:
        feature_names = (
            list(preprocessor.get_feature_names_out()) if preprocessor is not None else []
        )
    except Exception:  # noqa: BLE001 - coefficient diagnostics should not fail a completed fit.
        feature_names = []
    if len(feature_names) != len(coefficients):
        feature_names = [f"feature_{index}" for index in range(len(coefficients))]
    table = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    table["abs_coefficient"] = table["coefficient"].abs()
    table = table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    table.insert(0, "abs_rank", range(1, len(table) + 1))
    return table




def _coefficient_norms(estimator: Any) -> dict[str, float | int] | None:
    table = _coefficient_table(estimator)
    if table is None:
        return None
    coefficients = table["coefficient"].to_numpy(dtype=float)
    return {
        "coefficient_l1_norm": float(np.sum(np.abs(coefficients))),
        "coefficient_l2_norm": float(np.sqrt(np.sum(coefficients**2))),
        "nonzero_coefficients": int(np.count_nonzero(np.abs(coefficients) > 1e-10)),
    }



def _write_best_coefficient_summary(
    *, run_dir: Path, model_name: str, run_id: str, estimator: Any
) -> Path | None:
    table = _coefficient_table(estimator)
    if table is None:
        return None
    table.insert(0, "run_id", run_id)
    table.insert(1, "model", model_name)
    path = run_dir / "diagnostics" / f"{model_name}_best_coefficients.csv"
    table.to_csv(path, index=False)
    return path


def _fixed_effect_coefficient_rows(
    *,
    run_id: str,
    model_name: str,
    estimator: Any,
    fixed_effects_used: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract coefficient rows corresponding to generated fixed-effect columns."""

    if model_name not in {"LinearRegression", "Ridge", "Lasso"} or not fixed_effects_used:
        return []
    coefficient_table = _coefficient_table(estimator)
    if coefficient_table is None:
        return []

    rows: list[dict[str, Any]] = []
    for fe_spec in fixed_effects_used:
        fe_name = str(fe_spec["fe_name"])
        generated_column = str(fe_spec["generated_column"])
        prefixes = [
            f"cat__{generated_column}_",
            f"cat_drop_first__{generated_column}_",
        ]
        fe_rows = []
        for record in coefficient_table[["feature", "coefficient", "abs_coefficient"]].to_dict(
            orient="records"
        ):
            feature = str(record["feature"])
            level = None
            for prefix in prefixes:
                if feature.startswith(prefix):
                    level = feature.removeprefix(prefix)
                    break
            if level is None:
                continue
            coefficient = float(record["coefficient"])
            fe_rows.append(
                {
                    "run_id": run_id,
                    "model": model_name,
                    "fe_name": fe_name,
                    "level": level,
                    "feature": feature,
                    "coefficient": coefficient,
                    "abs_coefficient": float(record["abs_coefficient"]),
                    "sign": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
                }
            )
        fe_rows = sorted(fe_rows, key=lambda row: row["abs_coefficient"], reverse=True)
        for rank, row in enumerate(fe_rows, start=1):
            row["rank"] = rank
            rows.append(row)
    return rows


def _write_fixed_effect_coefficients(
    *, run_dir: Path, rows: list[dict[str, Any]]
) -> Path | None:
    """Write the combined fixed-effect coefficient export when linear rows exist."""

    if not rows:
        return None
    columns = [
        "run_id",
        "model",
        "fe_name",
        "level",
        "feature",
        "coefficient",
        "abs_coefficient",
        "sign",
        "rank",
    ]
    path = run_dir / "diagnostics" / "fixed_effect_coefficients.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path
