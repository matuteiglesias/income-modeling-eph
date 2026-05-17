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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
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
    if mode not in {"full", "sweep"}:
        raise ValueError(f"Unsupported runtime.mode: {mode}")

    config_allows_full = bool(runtime.get("allow_full_run", False))
    if not allow_full_run and not config_allows_full:
        guarded_label = "Full baseline" if mode == "full" else mode.title()
        raise ValueError(
            f"{guarded_label} training is guarded because it can be expensive. "
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
    for child in ["metrics", "predictions", "cv_results", "diagnostics", "artifacts", "plots"]:
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
        feature_names = list(preprocessor.get_feature_names_out()) if preprocessor is not None else []
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


def _cv_param_rows(cv_results: pd.DataFrame) -> list[dict[str, Any]]:
    """Extract one fitted-parameter dictionary per GridSearchCV row."""

    if "params" in cv_results.columns:
        return [dict(params) for params in cv_results["params"].tolist()]
    param_columns = [column for column in cv_results.columns if column.startswith("param_")]
    rows: list[dict[str, Any]] = []
    for _, row in cv_results.iterrows():
        params = {}
        for column in param_columns:
            value = row[column]
            if pd.notna(value):
                params[column.removeprefix("param_")] = value
        rows.append(params)
    return rows


def _regularization_path_summary(
    *,
    model_key: str,
    model_name: str,
    run_id: str,
    base_pipeline: Any,
    cv_results: pd.DataFrame,
    train: pd.DataFrame,
    feature_columns: list[str],
    target: str,
) -> pd.DataFrame | None:
    """Fit one full-training estimator per alpha to summarize shrinkage/sparsity."""

    if model_key not in {"ridge", "lasso"} or "param_reg__alpha" not in cv_results.columns:
        return None

    params_by_row = _cv_param_rows(cv_results)
    rows: list[dict[str, Any]] = []
    seen_alphas: set[float] = set()
    for row_index, params in enumerate(params_by_row):
        alpha = float(params.get("reg__alpha", cv_results.loc[row_index, "param_reg__alpha"]))
        if alpha in seen_alphas:
            continue
        seen_alphas.add(alpha)
        estimator = clone(base_pipeline)
        estimator.set_params(**params)
        estimator.fit(train[feature_columns], train[target])
        norms = _coefficient_norms(estimator)
        if norms is None:
            continue
        train_score = cv_results.loc[row_index, "mean_train_score"] if "mean_train_score" in cv_results.columns else np.nan
        rows.append(
            {
                "run_id": run_id,
                "model": model_name,
                "alpha": alpha,
                "mean_train_r2": float(train_score) if pd.notna(train_score) else np.nan,
                "mean_cv_r2": float(cv_results.loc[row_index, "mean_test_score"]),
                "std_cv_r2": float(cv_results.loc[row_index, "std_test_score"]),
                "mean_fit_time": float(cv_results.loc[row_index, "mean_fit_time"]),
                **norms,
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def _plot_regularization_line(
    path_summary: pd.DataFrame,
    *,
    x_column: str,
    y_columns: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for y_column, label in zip(y_columns, labels, strict=True):
        ax.plot(path_summary[x_column], path_summary[y_column], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(y_columns) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_regularization_plots(run_dir: Path, model_name: str, path_summary: pd.DataFrame) -> dict[str, Path]:
    model_slug = model_name.lower()
    plots_dir = run_dir / "plots" / "regularization"
    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    cv_path = plots_dir / f"{model_slug}_cv_r2_vs_alpha.png"
    _plot_regularization_line(
        path_summary,
        x_column="alpha",
        y_columns=["mean_cv_r2"],
        labels=["CV R2"],
        title=f"{model_name} CV R2 vs alpha",
        ylabel="Mean CV R2",
        output_path=cv_path,
    )
    outputs[cv_path.name] = cv_path

    train_cv_path = plots_dir / f"{model_slug}_train_vs_cv_r2_alpha.png"
    _plot_regularization_line(
        path_summary,
        x_column="alpha",
        y_columns=["mean_train_r2", "mean_cv_r2"],
        labels=["Train R2", "CV R2"],
        title=f"{model_name} train vs CV R2 by alpha",
        ylabel="R2",
        output_path=train_cv_path,
    )
    outputs[train_cv_path.name] = train_cv_path

    if model_name == "Ridge":
        l2_path = plots_dir / "ridge_coef_l2_norm_vs_alpha.png"
        _plot_regularization_line(
            path_summary,
            x_column="alpha",
            y_columns=["coefficient_l2_norm"],
            labels=["Coefficient L2 norm"],
            title="Ridge coefficient L2 norm vs alpha",
            ylabel="Coefficient L2 norm",
            output_path=l2_path,
        )
        outputs[l2_path.name] = l2_path
    elif model_name == "Lasso":
        nonzero_path = plots_dir / "lasso_nonzero_coefficients_vs_alpha.png"
        _plot_regularization_line(
            path_summary,
            x_column="alpha",
            y_columns=["nonzero_coefficients"],
            labels=["Nonzero coefficients"],
            title="Lasso nonzero coefficients vs alpha",
            ylabel="Nonzero coefficients",
            output_path=nonzero_path,
        )
        outputs[nonzero_path.name] = nonzero_path
        l1_path = plots_dir / "lasso_coef_l1_norm_vs_alpha.png"
        _plot_regularization_line(
            path_summary,
            x_column="alpha",
            y_columns=["coefficient_l1_norm"],
            labels=["Coefficient L1 norm"],
            title="Lasso coefficient L1 norm vs alpha",
            ylabel="Coefficient L1 norm",
            output_path=l1_path,
        )
        outputs[l1_path.name] = l1_path
    return outputs


def _hgb_param_value(row: pd.Series, param_name: str) -> Any:
    """Return a HistGradientBoosting parameter from a cv_results row or NA if absent."""

    column = f"param_reg__{param_name}"
    if column not in row.index or pd.isna(row[column]):
        return pd.NA
    return row[column]


def _hgb_cv_results_summary(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize HGB GridSearchCV rows in scientifically interpretable columns."""

    rows: list[dict[str, Any]] = []
    for _, row in cv_results.iterrows():
        mean_train_score = row.get("mean_train_score", np.nan)
        mean_test_score = row.get("mean_test_score", np.nan)
        rows.append(
            {
                "run_id": row.get("run_id", pd.NA),
                "model": row.get("model", "HistGradientBoostingRegressor"),
                "learning_rate": _hgb_param_value(row, "learning_rate"),
                "max_iter": _hgb_param_value(row, "max_iter"),
                "max_leaf_nodes": _hgb_param_value(row, "max_leaf_nodes"),
                "max_depth": _hgb_param_value(row, "max_depth"),
                "min_samples_leaf": _hgb_param_value(row, "min_samples_leaf"),
                "l2_regularization": _hgb_param_value(row, "l2_regularization"),
                "early_stopping": _hgb_param_value(row, "early_stopping"),
                "mean_train_score": float(mean_train_score) if pd.notna(mean_train_score) else np.nan,
                "mean_test_score": float(mean_test_score) if pd.notna(mean_test_score) else np.nan,
                "std_test_score": float(row.get("std_test_score", np.nan)),
                "overfit_gap": (
                    float(mean_train_score - mean_test_score)
                    if pd.notna(mean_train_score) and pd.notna(mean_test_score)
                    else np.nan
                ),
                "mean_fit_time": float(row.get("mean_fit_time", np.nan)),
                "rank_test_score": int(row.get("rank_test_score", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("rank_test_score").reset_index(drop=True)


def _numeric_plot_frame(summary: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    plot_frame = summary.copy()
    for column in columns:
        if column in plot_frame.columns:
            plot_frame[column] = pd.to_numeric(plot_frame[column], errors="coerce")
    return plot_frame.dropna(subset=[column for column in columns if column in plot_frame.columns])


def _plot_hgb_learning_rate_vs_max_iter(summary: pd.DataFrame, output_path: Path) -> None:
    plot_frame = _numeric_plot_frame(summary, ["learning_rate", "max_iter", "mean_test_score"])
    if plot_frame.empty:
        return
    grouped = (
        plot_frame.groupby(["learning_rate", "max_iter"], as_index=False)["mean_test_score"].mean()
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    for learning_rate, group in grouped.groupby("learning_rate"):
        group = group.sort_values("max_iter")
        ax.plot(group["max_iter"], group["mean_test_score"], marker="o", label=str(learning_rate))
    ax.set_xlabel("max_iter")
    ax.set_ylabel("Mean CV R2")
    ax.set_title("HGB CV R2 by learning_rate and max_iter")
    ax.legend(title="learning_rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_hgb_param_vs_cv_r2(summary: pd.DataFrame, param: str, output_path: Path) -> None:
    plot_frame = _numeric_plot_frame(summary, [param, "mean_test_score"])
    if plot_frame.empty:
        return
    grouped = plot_frame.groupby(param, as_index=False).agg(
        mean_cv_r2=("mean_test_score", "mean"),
        std_cv_r2=("mean_test_score", "std"),
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        grouped[param],
        grouped["mean_cv_r2"],
        yerr=grouped["std_cv_r2"].fillna(0.0),
        marker="o",
        capsize=3,
    )
    ax.set_xlabel(param)
    ax.set_ylabel("Mean CV R2")
    ax.set_title(f"HGB CV R2 by {param}")
    if param == "l2_regularization" and (grouped[param] > 0).all():
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_hgb_train_vs_cv_top_configs(summary: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    plot_frame = _numeric_plot_frame(summary, ["mean_train_score", "mean_test_score"])
    if plot_frame.empty:
        return
    plot_frame = plot_frame.sort_values("rank_test_score").head(top_n).copy()
    plot_frame["config_rank"] = range(1, len(plot_frame) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(plot_frame["config_rank"], plot_frame["mean_train_score"], marker="o", label="Train R2")
    ax.plot(plot_frame["config_rank"], plot_frame["mean_test_score"], marker="o", label="CV R2")
    ax.set_xlabel("CV rank among top configurations")
    ax.set_ylabel("R2")
    ax.set_title("HGB train vs CV R2 for top configurations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_hgb_diagnostics(run_dir: Path, cv_results: pd.DataFrame) -> dict[str, Path]:
    """Write HGB-specific tables and plots from archived GridSearchCV results."""

    summary = _hgb_cv_results_summary(cv_results)
    diagnostics_dir = run_dir / "diagnostics"
    plots_dir = run_dir / "plots" / "hgb"
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "hgb_cv_results_summary": diagnostics_dir / "hgb_cv_results_summary.csv",
        "hgb_top_configs": diagnostics_dir / "hgb_top_configs.csv",
        "hgb_overfit_gap_by_config": diagnostics_dir / "hgb_overfit_gap_by_config.csv",
    }
    summary.to_csv(paths["hgb_cv_results_summary"], index=False)
    summary.head(20).to_csv(paths["hgb_top_configs"], index=False)
    summary.sort_values("overfit_gap", ascending=False, na_position="last").to_csv(
        paths["hgb_overfit_gap_by_config"], index=False
    )

    plot_paths = {
        "hgb_learning_rate_vs_max_iter": plots_dir / "hgb_learning_rate_vs_max_iter.png",
        "hgb_max_leaf_nodes_vs_cv_r2": plots_dir / "hgb_max_leaf_nodes_vs_cv_r2.png",
        "hgb_min_samples_leaf_vs_cv_r2": plots_dir / "hgb_min_samples_leaf_vs_cv_r2.png",
        "hgb_l2_regularization_vs_cv_r2": plots_dir / "hgb_l2_regularization_vs_cv_r2.png",
        "hgb_train_vs_cv_r2_top_configs": plots_dir / "hgb_train_vs_cv_r2_top_configs.png",
    }
    _plot_hgb_learning_rate_vs_max_iter(summary, plot_paths["hgb_learning_rate_vs_max_iter"])
    _plot_hgb_param_vs_cv_r2(summary, "max_leaf_nodes", plot_paths["hgb_max_leaf_nodes_vs_cv_r2"])
    _plot_hgb_param_vs_cv_r2(
        summary, "min_samples_leaf", plot_paths["hgb_min_samples_leaf_vs_cv_r2"]
    )
    _plot_hgb_param_vs_cv_r2(
        summary, "l2_regularization", plot_paths["hgb_l2_regularization_vs_cv_r2"]
    )
    _plot_hgb_train_vs_cv_top_configs(summary, plot_paths["hgb_train_vs_cv_r2_top_configs"])
    paths.update({key: path for key, path in plot_paths.items() if path.exists()})
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

    diagnostics_config = experiment_config.get("diagnostics", {})
    write_coefficient_diagnostics = bool(
        diagnostics_config.get("regularization_sweep", False)
        if isinstance(diagnostics_config, Mapping)
        else False
    )

    rows: list[dict[str, Any]] = []
    predictions_by_split: dict[str, list[pd.DataFrame]] = {"test": [], "validation": []}
    for model_key, model_config in enabled_model_configs(experiment_config).items():
        model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        pipeline = make_model_pipeline(model_key, train[feature_columns], random_seed)
        grid = model_config.get("grid", {})
        return_train_score = bool(experiment_config.get("cv", {}).get("return_train_score", False))
        search = GridSearchCV(
            pipeline,
            grid,
            cv=cv_folds,
            scoring=scoring,
            refit=True,
            n_jobs=None,
            return_train_score=return_train_score,
        )
        start = time.perf_counter()
        search.fit(train[feature_columns], train[target])
        fit_time = time.perf_counter() - start

        cv_results = pd.DataFrame(search.cv_results_)
        cv_results.insert(0, "run_id", run_id)
        cv_results.insert(1, "model", model_name)
        cv_results.to_csv(run_dir / "cv_results" / f"{model_name}.csv", index=False)

        if model_key == "hist_gradient_boosting":
            _write_hgb_diagnostics(run_dir, cv_results)

        if write_coefficient_diagnostics:
            _write_best_coefficient_summary(
                run_dir=run_dir,
                model_name=model_name,
                run_id=run_id,
                estimator=search.best_estimator_,
            )
            path_summary = _regularization_path_summary(
                model_key=model_key,
                model_name=model_name,
                run_id=run_id,
                base_pipeline=pipeline,
                cv_results=cv_results,
                train=train,
                feature_columns=feature_columns,
                target=target,
            )
            if path_summary is not None:
                path_summary.to_csv(
                    run_dir / "diagnostics" / f"{model_name}_regularization_path_summary.csv",
                    index=False,
                )
                _write_regularization_plots(run_dir, model_name, path_summary)

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
    output_path = (
        _resolve_output_path(outputs["model_comparison"]) if "model_comparison" in outputs else None
    )
    card_path = _resolve_output_path(outputs["experiment_card"]) if "experiment_card" in outputs else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if card_path is not None:
        card_path.parent.mkdir(parents=True, exist_ok=True)

    run_model_comparison_path = run_dir / "metrics" / "model_comparison.csv"
    comparison.to_csv(run_model_comparison_path, index=False)
    _metrics_long(comparison, run_id).to_csv(run_dir / "metrics" / "metrics_long.csv", index=False)
    if output_path is not None:
        shutil.copyfile(run_model_comparison_path, output_path)
    prediction_paths = _write_predictions(predictions_by_split, run_dir)
    diagnostic_paths = _write_diagnostics_from_predictions(run_dir)

    manifest_paths = {
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
        "prediction_distribution_summary": str(diagnostic_paths["prediction_distribution_summary"]),
    }
    hgb_diagnostics = {
        path.stem: str(path) for path in sorted((run_dir / "diagnostics").glob("hgb_*.csv"))
    }
    hgb_plots = {
        path.stem: str(path) for path in sorted((run_dir / "plots" / "hgb").glob("hgb_*.png"))
    }
    manifest_paths.update(hgb_diagnostics)
    manifest_paths.update(hgb_plots)
    if output_path is not None:
        manifest_paths["model_comparison_convenience_copy"] = str(output_path)

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
        "paths": manifest_paths,
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
        "model_comparison": str(output_path or run_model_comparison_path),
        "canonical_run_dir": str(run_dir),
        "canonical_model_comparison": str(run_model_comparison_path),
        "run_manifest": str(run_dir / "run_manifest.json"),
    }
    run_card_path = run_dir / "experiment_card.json"
    run_card_path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    if card_path is not None:
        card_path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    return comparison, card
