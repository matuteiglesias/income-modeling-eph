"""Scientific diagnostics built from archived experiment run artifacts.

This module intentionally reads only saved run outputs: prediction parquet files,
model-comparison metrics, CV result CSVs, and the run manifest. It does not import
training pipelines, fit estimators, or call model prediction APIs.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import yaml

REQUIRED_PREDICTION_COLUMNS = {
    "run_id",
    "model",
    "split",
    "row_id",
    "y_true",
    "y_pred",
    "residual",
    "abs_error",
    "squared_error",
}

PREDICTION_FILES = {
    "test": "test_predictions.parquet",
    "validation": "validation_predictions.parquet",
}


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _quantile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q))


def _rmse(series: pd.Series) -> float:
    return float(np.sqrt(series.mean()))


def _r2_from_group(group: pd.DataFrame) -> float:
    denominator = float(((group["y_true"] - group["y_true"].mean()) ** 2).sum())
    if denominator == 0.0:
        return float("nan")
    numerator = float(((group["y_true"] - group["y_pred"]) ** 2).sum())
    return float(1.0 - numerator / denominator)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_income_decile(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return predictions with income deciles, without mutating archived files."""

    if "income_decile" in predictions.columns:
        return predictions.copy()

    with_decile = predictions.copy()
    with_decile["income_decile"] = pd.NA
    group_keys = ["run_id", "model", "split"]
    for _, index in with_decile.groupby(group_keys, dropna=False).groups.items():
        y_true = with_decile.loc[index, "y_true"]
        try:
            deciles = pd.qcut(y_true, q=10, labels=False, duplicates="drop").astype("Int64") + 1
        except ValueError:
            deciles = pd.Series([pd.NA] * len(y_true), index=index, dtype="Int64")
        with_decile.loc[index, "income_decile"] = deciles.to_numpy()
    with_decile["income_decile"] = with_decile["income_decile"].astype("Int64")
    return with_decile


def read_run_manifest(run_dir: str | Path) -> dict[str, Any]:
    """Read a run manifest if present."""

    return _read_json(_as_path(run_dir) / "run_manifest.json")


def read_predictions(run_dir: str | Path) -> pd.DataFrame:
    """Read archived test/validation prediction files from a run directory."""

    run_path = _as_path(run_dir)
    frames: list[pd.DataFrame] = []
    missing_files: list[str] = []
    for split, filename in PREDICTION_FILES.items():
        path = run_path / "predictions" / filename
        if not path.exists():
            missing_files.append(str(path))
            continue
        frame = pd.read_parquet(path)
        missing_columns = sorted(REQUIRED_PREDICTION_COLUMNS - set(frame.columns))
        if missing_columns:
            raise ValueError(
                f"Prediction file {path} is missing required columns: "
                + ", ".join(missing_columns)
            )
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No archived prediction files found. Missing expected files: " + ", ".join(missing_files)
        )
    return _ensure_income_decile(pd.concat(frames, ignore_index=True))


def read_model_comparison(run_dir: str | Path) -> pd.DataFrame:
    """Read archived model-comparison metrics from a run directory."""

    path = _as_path(run_dir) / "metrics" / "model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Model comparison file does not exist: {path}")
    return pd.read_csv(path)


def compute_residual_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute residual and predictive-metric summaries by run/model/split."""

    rows: list[dict[str, Any]] = []
    for (run_id, model, split), group in predictions.groupby(["run_id", "model", "split"], dropna=False):
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "split": split,
                "n": int(len(group)),
                "residual_mean": float(group["residual"].mean()),
                "residual_std": float(group["residual"].std()),
                "residual_median": float(group["residual"].median()),
                "residual_p05": _quantile(group["residual"], 0.05),
                "residual_p25": _quantile(group["residual"], 0.25),
                "residual_p75": _quantile(group["residual"], 0.75),
                "residual_p95": _quantile(group["residual"], 0.95),
                "mae": float(group["abs_error"].mean()),
                "mse": float(group["squared_error"].mean()),
                "rmse": _rmse(group["squared_error"]),
                "r2": _r2_from_group(group),
                "y_true_mean": float(group["y_true"].mean()),
                "y_true_std": float(group["y_true"].std()),
                "y_pred_mean": float(group["y_pred"].mean()),
                "y_pred_std": float(group["y_pred"].std()),
            }
        )
    return pd.DataFrame(rows).sort_values(["run_id", "model", "split"]).reset_index(drop=True)


def compute_error_by_income_decile(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute error summaries by income decile."""

    rows: list[dict[str, Any]] = []
    group_keys = ["run_id", "model", "split", "income_decile"]
    for (run_id, model, split, decile), group in predictions.groupby(group_keys, dropna=False):
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "split": split,
                "income_decile": decile,
                "n": int(len(group)),
                "y_true_min": float(group["y_true"].min()),
                "y_true_max": float(group["y_true"].max()),
                "y_true_mean": float(group["y_true"].mean()),
                "mae": float(group["abs_error"].mean()),
                "mse": float(group["squared_error"].mean()),
                "rmse": _rmse(group["squared_error"]),
                "residual_mean": float(group["residual"].mean()),
                "residual_median": float(group["residual"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(group_keys).reset_index(drop=True)


def compute_pairwise_error_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare absolute errors between model pairs on common rows."""

    rows: list[dict[str, Any]] = []
    for (run_id, split), split_frame in predictions.groupby(["run_id", "split"], dropna=False):
        models = sorted(split_frame["model"].dropna().unique().tolist())
        for model_a, model_b in itertools.combinations(models, 2):
            a = split_frame.loc[split_frame["model"] == model_a, ["row_id", "abs_error"]].rename(
                columns={"abs_error": "abs_error_a"}
            )
            b = split_frame.loc[split_frame["model"] == model_b, ["row_id", "abs_error"]].rename(
                columns={"abs_error": "abs_error_b"}
            )
            merged = a.merge(b, on="row_id", how="inner", validate="one_to_one")
            if merged.empty:
                continue
            a_better = merged["abs_error_a"] < merged["abs_error_b"]
            b_better = merged["abs_error_b"] < merged["abs_error_a"]
            tied = merged["abs_error_a"] == merged["abs_error_b"]
            rows.append(
                {
                    "run_id": run_id,
                    "split": split,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_common_rows": int(len(merged)),
                    "share_rows_a_better": float(a_better.mean()),
                    "share_rows_b_better": float(b_better.mean()),
                    "share_rows_tied": float(tied.mean()),
                    "mean_abs_error_a": float(merged["abs_error_a"].mean()),
                    "mean_abs_error_b": float(merged["abs_error_b"].mean()),
                    "mean_abs_error_difference_a_minus_b": float(
                        merged["abs_error_a"].mean() - merged["abs_error_b"].mean()
                    ),
                }
            )
    columns = [
        "run_id",
        "split",
        "model_a",
        "model_b",
        "n_common_rows",
        "share_rows_a_better",
        "share_rows_b_better",
        "share_rows_tied",
        "mean_abs_error_a",
        "mean_abs_error_b",
        "mean_abs_error_difference_a_minus_b",
    ]
    return pd.DataFrame(rows, columns=columns)


def compute_metric_gaps(model_comparison: pd.DataFrame, run_id: str | None) -> pd.DataFrame:
    """Compute validation-minus-test metric gaps from archived comparison metrics."""

    rows: list[dict[str, Any]] = []
    test_metrics = {
        column.removeprefix("test_"): column
        for column in model_comparison.columns
        if column.startswith("test_")
    }
    validation_metrics = {
        column.removeprefix("validation_"): column
        for column in model_comparison.columns
        if column.startswith("validation_")
    }
    for metric in sorted(set(test_metrics) & set(validation_metrics)):
        for _, row in model_comparison.iterrows():
            test_value = row[test_metrics[metric]]
            validation_value = row[validation_metrics[metric]]
            rows.append(
                {
                    "run_id": run_id,
                    "model": row["model"],
                    "metric": metric,
                    "test_value": float(test_value),
                    "validation_value": float(validation_value),
                    "validation_minus_test": float(validation_value - test_value),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "model",
            "metric",
            "test_value",
            "validation_value",
            "validation_minus_test",
        ],
    )


def _selected_split(predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = predictions[predictions["split"] == split]
    if selected.empty:
        available = sorted(predictions["split"].dropna().unique().tolist())
        raise ValueError(f"Split {split!r} is not available. Available splits: {available}")
    return selected


def _subplot_grid(n_items: int) -> tuple[plt.Figure, np.ndarray]:
    n_cols = min(3, max(1, n_items))
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)
    return fig, axes.ravel()


def _save_prediction_distribution(predictions: pd.DataFrame, split: str, output_path: Path) -> None:
    selected = _selected_split(predictions, split)
    models = sorted(selected["model"].unique())
    fig, axes = _subplot_grid(len(models))
    for ax, model in zip(axes, models, strict=False):
        data = selected[selected["model"] == model]
        ax.hist(data["y_true"], bins=30, alpha=0.45, density=True, label="y_true")
        ax.hist(data["y_pred"], bins=30, alpha=0.45, density=True, label="y_pred")
        ax.set_title(model)
        ax.set_xlabel("log10(P47T)")
        ax.set_ylabel("Density")
        ax.legend()
    for ax in axes[len(models) :]:
        ax.axis("off")
    fig.suptitle(f"Observed and predicted distributions ({split})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _best_model_by_validation_r2(model_comparison: pd.DataFrame) -> str | None:
    if "validation_r2" not in model_comparison.columns or model_comparison.empty:
        return None
    best_index = model_comparison["validation_r2"].astype(float).idxmax()
    return str(model_comparison.loc[best_index, "model"])


def _save_observed_vs_predicted(
    predictions: pd.DataFrame, model_comparison: pd.DataFrame, split: str, output_path: Path
) -> str | None:
    best_model = _best_model_by_validation_r2(model_comparison)
    if best_model is None:
        return "Skipped observed_vs_predicted_best_model.png: validation_r2 is missing."
    selected = _selected_split(predictions, split)
    data = selected[selected["model"] == best_model]
    if data.empty:
        return f"Skipped observed_vs_predicted_best_model.png: best model {best_model!r} missing."
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(data) > 5000:
        ax.hexbin(data["y_true"], data["y_pred"], gridsize=45, mincnt=1)
    else:
        ax.scatter(data["y_true"], data["y_pred"], s=12, alpha=0.5)
    axis_min = float(min(data["y_true"].min(), data["y_pred"].min()))
    axis_max = float(max(data["y_true"].max(), data["y_pred"].max()))
    ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", linewidth=1)
    ax.set_title(f"Observed vs predicted: {best_model} ({split})")
    ax.set_xlabel("Observed log10(P47T)")
    ax.set_ylabel("Predicted log10(P47T)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return None


def _save_decile_line_plot(
    decile_table: pd.DataFrame,
    split: str,
    y_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
    *,
    zero_line: bool = False,
) -> None:
    selected = decile_table[decile_table["split"] == split]
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in sorted(selected["model"].dropna().unique()):
        data = selected[selected["model"] == model].sort_values("income_decile")
        ax.plot(data["income_decile"], data[y_column], marker="o", label=model)
    if zero_line:
        ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Income decile")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_residual_distribution(predictions: pd.DataFrame, split: str, output_path: Path) -> None:
    selected = _selected_split(predictions, split)
    models = sorted(selected["model"].unique())
    fig, axes = _subplot_grid(len(models))
    for ax, model in zip(axes, models, strict=False):
        data = selected[selected["model"] == model]
        ax.hist(data["residual"], bins=30, alpha=0.7, density=True)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel("Residual (y_true - y_pred)")
        ax.set_ylabel("Density")
    for ax in axes[len(models) :]:
        ax.axis("off")
    fig.suptitle(f"Residual distributions ({split})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _find_cv_file(run_dir: Path, model_name: str) -> Path | None:
    candidates = [run_dir / "cv_results" / f"{model_name}.csv"]
    candidates.extend((run_dir / "cv_results").glob(f"*{model_name}*.csv"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _alpha_column(columns: Iterable[str]) -> str | None:
    for column in ["param_reg__alpha", "reg__alpha", "alpha"]:
        if column in columns:
            return column
    return None


def _save_alpha_curve(run_dir: Path, model_name: str, output_path: Path) -> str | None:
    cv_file = _find_cv_file(run_dir, model_name)
    if cv_file is None:
        return f"Skipped {output_path.name}: no {model_name} cv_results CSV found."
    cv_results = pd.read_csv(cv_file)
    alpha_col = _alpha_column(cv_results.columns)
    if alpha_col is None or "mean_test_score" not in cv_results.columns:
        return f"Skipped {output_path.name}: missing alpha or mean_test_score columns in {cv_file}."
    plot_frame = cv_results[[alpha_col, "mean_test_score"]].copy()
    plot_frame[alpha_col] = pd.to_numeric(plot_frame[alpha_col], errors="coerce")
    plot_frame["mean_test_score"] = pd.to_numeric(plot_frame["mean_test_score"], errors="coerce")
    plot_frame = plot_frame.dropna().sort_values(alpha_col)
    if plot_frame.empty:
        return f"Skipped {output_path.name}: no numeric alpha/score values in {cv_file}."
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(plot_frame[alpha_col], plot_frame["mean_test_score"], marker="o")
    ax.set_title(f"{model_name} CV alpha curve")
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean CV score")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return None


HGB_SWEEP_PARAM_COLUMNS = [
    "learning_rate",
    "max_iter",
    "max_leaf_nodes",
    "max_depth",
    "min_samples_leaf",
    "l2_regularization",
    "early_stopping",
]


def _read_config_used(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config_used.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _hgb_cv_file(run_dir: Path) -> Path | None:
    return _find_cv_file(run_dir, "HistGradientBoostingRegressor")


def _cv_param_value(row: pd.Series, param_name: str) -> Any:
    for column in [f"param_reg__{param_name}", f"reg__{param_name}", param_name]:
        if column in row.index and pd.notna(row[column]):
            return row[column]
    return pd.NA


def _hgb_sweep_summary(cv_results: pd.DataFrame, *, run_id: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in cv_results.iterrows():
        mean_train_score = row.get("mean_train_score", np.nan)
        mean_test_score = row.get("mean_test_score", np.nan)
        result = {
            "run_id": row.get("run_id", run_id),
            "model": row.get("model", "HistGradientBoostingRegressor"),
        }
        for param in HGB_SWEEP_PARAM_COLUMNS:
            result[param] = _cv_param_value(row, param)
        result.update(
            {
                "mean_train_score": float(mean_train_score) if pd.notna(mean_train_score) else np.nan,
                "mean_test_score": float(mean_test_score) if pd.notna(mean_test_score) else np.nan,
                "std_test_score": float(row.get("std_test_score", np.nan)),
                "overfit_gap": (
                    float(mean_train_score - mean_test_score)
                    if pd.notna(mean_train_score) and pd.notna(mean_test_score)
                    else np.nan
                ),
                "mean_fit_time": float(row.get("mean_fit_time", np.nan)),
                "rank_test_score": int(row.get("rank_test_score", 0)) if pd.notna(row.get("rank_test_score", np.nan)) else pd.NA,
            }
        )
        rows.append(result)
    summary = pd.DataFrame(rows)
    if "rank_test_score" in summary.columns:
        summary = summary.sort_values("rank_test_score", na_position="last")
    return summary.reset_index(drop=True)


def _to_numeric_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _aggregate_sweep_curve(summary: pd.DataFrame, primary_param: str, group_param: str | None = None) -> pd.DataFrame:
    required = [primary_param, "mean_test_score"]
    if group_param:
        required.append(group_param)
    optional = ["mean_train_score", "overfit_gap", "mean_fit_time", "std_test_score"]
    numeric = _to_numeric_frame(summary, [*required, *optional])
    numeric = numeric.dropna(subset=required)
    if numeric.empty:
        return numeric
    group_cols = [group_param, primary_param] if group_param else [primary_param]
    agg = numeric.groupby(group_cols, as_index=False).agg(
        mean_cv_r2=("mean_test_score", "mean"),
        std_cv_r2=("std_test_score", "mean"),
        mean_train_r2=("mean_train_score", "mean"),
        mean_overfit_gap=("overfit_gap", "mean"),
        mean_fit_time=("mean_fit_time", "mean"),
    )
    return agg.sort_values(group_cols).reset_index(drop=True)


def _use_log_scale_for_param(frame: pd.DataFrame, param: str) -> bool:
    if param != "l2_regularization" or param not in frame.columns:
        return False
    values = pd.to_numeric(frame[param], errors="coerce").dropna()
    return bool(not values.empty and (values > 0).all())


def _plot_grouped_errorbar(
    curve: pd.DataFrame,
    *,
    primary_param: str,
    group_param: str | None,
    y_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
    error_column: str | None = None,
    zero_line: bool = False,
) -> None:
    if curve.empty or y_column not in curve.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = curve.groupby(group_param, dropna=False) if group_param else [(None, curve)]
    for group_value, group in groups:
        group = group.sort_values(primary_param)
        yerr = None
        if error_column and error_column in group.columns:
            yerr = pd.to_numeric(group[error_column], errors="coerce").fillna(0.0)
        label = None if group_param is None else f"{group_param}={group_value}"
        ax.errorbar(
            group[primary_param],
            group[y_column],
            yerr=yerr,
            marker="o",
            capsize=3 if yerr is not None else 0,
            label=label,
        )
    if zero_line:
        ax.axhline(0, linestyle="--", linewidth=1)
    if _use_log_scale_for_param(curve, primary_param):
        ax.set_xscale("log")
    ax.set_xlabel(primary_param)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if group_param:
        ax.legend(title=group_param)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_train_vs_cv(
    curve: pd.DataFrame, *, primary_param: str, group_param: str | None, output_path: Path
) -> None:
    if curve.empty or "mean_train_r2" not in curve.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = curve.groupby(group_param, dropna=False) if group_param else [(None, curve)]
    for group_value, group in groups:
        group = group.sort_values(primary_param)
        suffix = "" if group_param is None else f" ({group_param}={group_value})"
        ax.plot(group[primary_param], group["mean_train_r2"], marker="o", linestyle="--", label=f"Train R²{suffix}")
        ax.errorbar(
            group[primary_param],
            group["mean_cv_r2"],
            yerr=group.get("std_cv_r2", pd.Series(index=group.index, dtype=float)).fillna(0.0),
            marker="o",
            capsize=3,
            label=f"CV R²{suffix}",
        )
    if _use_log_scale_for_param(curve, primary_param):
        ax.set_xscale("log")
    ax.set_xlabel(primary_param)
    ax.set_ylabel("R²")
    ax.set_title("HGB train vs CV R²")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _format_config(row: pd.Series) -> str:
    parts = []
    for param in HGB_SWEEP_PARAM_COLUMNS:
        if param in row.index and pd.notna(row[param]):
            parts.append(f"{param}={row[param]}")
    return ", ".join(parts) if parts else "unavailable"


def _simplest_within(summary: pd.DataFrame, best_score: float, tolerance: float = 0.002) -> pd.Series:
    candidates = summary[pd.to_numeric(summary["mean_test_score"], errors="coerce") >= best_score - tolerance].copy()
    if candidates.empty:
        return summary.iloc[0]
    for column, default in [("max_iter", np.inf), ("max_leaf_nodes", np.inf), ("min_samples_leaf", -np.inf), ("l2_regularization", np.inf)]:
        if column in candidates.columns:
            candidates[f"_sort_{column}"] = pd.to_numeric(candidates[column], errors="coerce").fillna(default)
    ascending = []
    sort_cols = []
    for column in ["_sort_max_iter", "_sort_max_leaf_nodes", "_sort_min_samples_leaf", "_sort_l2_regularization"]:
        if column in candidates.columns:
            sort_cols.append(column)
            ascending.append(False if column == "_sort_min_samples_leaf" else True)
    sort_cols.append("mean_test_score")
    ascending.append(False)
    return candidates.sort_values(sort_cols, ascending=ascending).iloc[0]


def _recommended_region(summary: pd.DataFrame, best_score: float, tolerance: float = 0.002) -> str:
    near = summary[pd.to_numeric(summary["mean_test_score"], errors="coerce") >= best_score - tolerance].copy()
    if near.empty:
        near = summary.head(1).copy()
    pieces: list[str] = []
    for param in ["learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"]:
        if param not in near.columns:
            continue
        values = pd.to_numeric(near[param], errors="coerce").dropna().sort_values().unique()
        if len(values) == 0:
            continue
        if len(values) == 1:
            pieces.append(f"{param}={values[0]:g}")
        else:
            pieces.append(f"{param}≈{values[0]:g}–{values[-1]:g}")
    return "; ".join(pieces) if pieces else "No numeric parameter region could be inferred."


def _write_hgb_sweep_summary_markdown(
    *,
    summary: pd.DataFrame,
    experiment_id: str | None,
    run_id: str | None,
    sweep_type: str,
    primary_param: str,
    group_param: str | None,
    output_path: Path,
) -> None:
    numeric = _to_numeric_frame(summary, ["mean_test_score", "mean_train_score", "overfit_gap"])
    numeric = numeric.dropna(subset=["mean_test_score"])
    if numeric.empty:
        output_path.write_text("# HGB sweep scientific summary\n\nNo numeric CV results were available.\n", encoding="utf-8")
        return
    best = numeric.sort_values("mean_test_score", ascending=False).iloc[0]
    best_score = float(best["mean_test_score"])
    simplest = _simplest_within(numeric, best_score)
    gap = float(best.get("overfit_gap", np.nan))
    warning = ""
    if pd.notna(gap) and gap > 0.10:
        warning = "\n\n> **Warning:** Best configuration overfit gap exceeds 0.10 R²."
    interpretation = (
        f"This {sweep_type} sweep varies `{primary_param}`"
        + (f" within `{group_param}` groups" if group_param else "")
        + ". The recommended region is based on configurations within 0.002 mean CV R² of the best row, favoring simpler capacity when performance is statistically close."
    )
    lines = [
        "# HGB sweep scientific summary",
        "",
        f"- Experiment id: `{experiment_id or 'unknown'}`",
        f"- Run id: `{run_id or 'unknown'}`",
        f"- Sweep type: `{sweep_type}`",
        f"- Primary parameter: `{primary_param}`",
    ]
    if group_param:
        lines.append(f"- Group parameter: `{group_param}`")
    lines.extend(
        [
            "",
            "## Best configuration by mean CV R²",
            "",
            f"- Mean CV R²: `{best_score:.6f}`",
            f"- CV std: `{float(best.get('std_test_score', np.nan)):.6f}`",
            f"- Parameters: {_format_config(best)}",
            "",
            "## Simplest configuration within 0.002 CV R² of best",
            "",
            f"- Mean CV R²: `{float(simplest['mean_test_score']):.6f}`",
            f"- Parameters: {_format_config(simplest)}",
            "",
            "## Overfit gap",
            "",
            f"- Best configuration train-minus-CV R² gap: `{gap:.6f}`" if pd.notna(gap) else "- Best configuration overfit gap unavailable because train scores were not archived.",
            warning,
            "",
            "## Recommended parameter region",
            "",
            f"- {_recommended_region(numeric, best_score)}",
            "",
            "## Interpretation",
            "",
            interpretation,
        ]
    )
    output_path.write_text("\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8")


def _write_hgb_sweep_diagnostics(
    run_dir: Path,
    *,
    config: Mapping[str, Any],
    run_id: str | None,
) -> tuple[dict[str, Path], list[str]]:
    diagnostics_config = config.get("diagnostics", {}) if isinstance(config, Mapping) else {}
    if not isinstance(diagnostics_config, Mapping):
        return {}, []
    sweep_type = diagnostics_config.get("sweep_type")
    primary_param = diagnostics_config.get("primary_param")
    group_param = diagnostics_config.get("group_param")
    if sweep_type not in {"hgb_lr_iter", "hgb_single_param"} or not primary_param:
        return {}, []
    cv_file = _hgb_cv_file(run_dir)
    if cv_file is None:
        return {}, ["Skipped HGB sweep plots: no HistGradientBoostingRegressor cv_results CSV found."]
    cv_results = pd.read_csv(cv_file)
    summary = _hgb_sweep_summary(cv_results, run_id=run_id)
    diagnostics_dir = run_dir / "diagnostics"
    plots_dir = run_dir / "plots" / "hgb_sweeps"
    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    summary_path = diagnostics_dir / "hgb_sweep_cv_results_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs["hgb_sweep_cv_results_summary"] = summary_path
    primary = str(primary_param)
    group = str(group_param) if group_param else None
    curve = _aggregate_sweep_curve(summary, primary, group)
    if sweep_type == "hgb_lr_iter":
        prefix = "hgb_lr_iter"
    else:
        prefix = f"hgb_{primary}"
    plot_specs = {
        f"{prefix}_cv_r2": plots_dir / f"{prefix}_cv_r2.png",
        f"{prefix}_train_vs_cv": plots_dir / f"{prefix}_train_vs_cv.png",
        f"{prefix}_overfit_gap": plots_dir / f"{prefix}_overfit_gap.png",
        f"{prefix}_fit_time": plots_dir / f"{prefix}_fit_time.png",
    }
    _plot_grouped_errorbar(
        curve,
        primary_param=primary,
        group_param=group,
        y_column="mean_cv_r2",
        ylabel="Mean CV R²",
        title=f"HGB CV R² by {primary}" + (f" and {group}" if group else ""),
        output_path=plot_specs[f"{prefix}_cv_r2"],
        error_column="std_cv_r2",
    )
    _plot_train_vs_cv(curve, primary_param=primary, group_param=group, output_path=plot_specs[f"{prefix}_train_vs_cv"])
    _plot_grouped_errorbar(
        curve,
        primary_param=primary,
        group_param=group,
        y_column="mean_overfit_gap",
        ylabel="Train-minus-CV R²",
        title=f"HGB overfit gap by {primary}",
        output_path=plot_specs[f"{prefix}_overfit_gap"],
        zero_line=True,
    )
    _plot_grouped_errorbar(
        curve,
        primary_param=primary,
        group_param=group,
        y_column="mean_fit_time",
        ylabel="Mean fit time (seconds)",
        title=f"HGB fit time by {primary}",
        output_path=plot_specs[f"{prefix}_fit_time"],
    )
    outputs.update({key: path for key, path in plot_specs.items() if path.exists()})
    experiment_id = config.get("experiment", {}).get("id") if isinstance(config.get("experiment", {}), Mapping) else None
    summary_md = diagnostics_dir / "hgb_sweep_scientific_summary.md"
    _write_hgb_sweep_summary_markdown(
        summary=summary,
        experiment_id=experiment_id,
        run_id=run_id,
        sweep_type=str(sweep_type),
        primary_param=primary,
        group_param=group,
        output_path=summary_md,
    )
    outputs["hgb_sweep_scientific_summary"] = summary_md
    return outputs, []


def _write_notes(run_dir: Path, notes: list[str], generated: list[Path]) -> Path:
    notes_path = run_dir / "diagnostics" / "diagnostics_build_notes.md"
    lines = ["# Diagnostics build notes", "", "## Generated artifacts", ""]
    lines.extend(f"- `{path.relative_to(run_dir)}`" for path in generated)
    if notes:
        lines.extend(["", "## Skipped or notable items", ""])
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.extend(["", "## Skipped or notable items", "", "- None"])
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return notes_path


def build_diagnostics(run_dir: str | Path, *, split: str = "test") -> dict[str, Any]:
    """Build diagnostic tables and plots from archived run artifacts."""

    run_path = _as_path(run_dir)
    if split not in PREDICTION_FILES:
        raise ValueError(f"Unsupported split {split!r}. Expected one of: {sorted(PREDICTION_FILES)}")

    diagnostics_dir = run_path / "diagnostics"
    plots_dir = run_path / "plots" / split
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_run_manifest(run_path)
    run_id = manifest.get("run_id")
    config_used = _read_config_used(run_path)
    predictions = read_predictions(run_path)
    if run_id is None and not predictions.empty:
        run_id = str(predictions["run_id"].iloc[0])
    model_comparison = read_model_comparison(run_path)

    residual_summary = compute_residual_summary(predictions)
    error_by_decile = compute_error_by_income_decile(predictions)
    pairwise = compute_pairwise_error_comparison(predictions)
    metric_gaps = compute_metric_gaps(model_comparison, run_id=str(run_id) if run_id else None)

    outputs: dict[str, Path] = {
        "residual_summary": diagnostics_dir / "residual_summary.csv",
        "error_by_income_decile": diagnostics_dir / "error_by_income_decile.csv",
        "model_pairwise_error_comparison": diagnostics_dir
        / "model_pairwise_error_comparison.csv",
        "metric_gaps": diagnostics_dir / "metric_gaps.csv",
    }
    residual_summary.to_csv(outputs["residual_summary"], index=False)
    error_by_decile.to_csv(outputs["error_by_income_decile"], index=False)
    pairwise.to_csv(outputs["model_pairwise_error_comparison"], index=False)
    metric_gaps.to_csv(outputs["metric_gaps"], index=False)

    notes: list[str] = []
    plot_outputs = {
        "prediction_distribution_by_model": plots_dir / "prediction_distribution_by_model.png",
        "observed_vs_predicted_best_model": plots_dir / "observed_vs_predicted_best_model.png",
        "mae_by_income_decile": plots_dir / "mae_by_income_decile.png",
        "mean_residual_by_income_decile": plots_dir / "mean_residual_by_income_decile.png",
        "residual_distribution_by_model": plots_dir / "residual_distribution_by_model.png",
        "ridge_alpha_curve": plots_dir / "ridge_alpha_curve.png",
        "lasso_alpha_curve": plots_dir / "lasso_alpha_curve.png",
    }
    generated_plots: list[Path] = []

    for plotter, output_path in [
        (lambda: _save_prediction_distribution(predictions, split, output_path), plot_outputs["prediction_distribution_by_model"]),
        (
            lambda: _save_observed_vs_predicted(
                predictions, model_comparison, split, output_path
            ),
            plot_outputs["observed_vs_predicted_best_model"],
        ),
        (
            lambda: _save_decile_line_plot(
                error_by_decile,
                split,
                "mae",
                "MAE",
                f"MAE by income decile ({split})",
                output_path,
            ),
            plot_outputs["mae_by_income_decile"],
        ),
        (
            lambda: _save_decile_line_plot(
                error_by_decile,
                split,
                "residual_mean",
                "Mean residual",
                f"Mean residual by income decile ({split})",
                output_path,
                zero_line=True,
            ),
            plot_outputs["mean_residual_by_income_decile"],
        ),
        (lambda: _save_residual_distribution(predictions, split, output_path), plot_outputs["residual_distribution_by_model"]),
    ]:
        try:
            note = plotter()
        except Exception as exc:  # noqa: BLE001 - diagnostics should skip gracefully and record notes.
            notes.append(f"Skipped {output_path.name}: {exc}")
            continue
        if note:
            notes.append(note)
        elif output_path.exists():
            generated_plots.append(output_path)

    for model_name, key in [("Ridge", "ridge_alpha_curve"), ("Lasso", "lasso_alpha_curve")]:
        output_path = plot_outputs[key]
        note = _save_alpha_curve(run_path, model_name, output_path)
        if note:
            notes.append(note)
        elif output_path.exists():
            generated_plots.append(output_path)

    sweep_outputs, sweep_notes = _write_hgb_sweep_diagnostics(
        run_path, config=config_used, run_id=str(run_id) if run_id else None
    )
    notes.extend(sweep_notes)
    outputs.update(sweep_outputs)
    generated_sweep_artifacts = list(sweep_outputs.values())

    generated_tables = list(outputs.values())
    notes_path = _write_notes(run_path, notes, [*generated_tables, *generated_plots, *generated_sweep_artifacts])
    outputs["diagnostics_build_notes"] = notes_path
    outputs.update(plot_outputs)

    return {
        "run_dir": str(run_path),
        "split": split,
        "run_id": run_id,
        "tables": {key: str(path) for key, path in outputs.items() if path.suffix == ".csv"},
        "plots": {
            **{key: str(path) for key, path in plot_outputs.items() if path.exists()},
            **{key: str(path) for key, path in sweep_outputs.items() if path.suffix == ".png" and path.exists()},
        },
        "notes": notes,
        "notes_path": str(notes_path),
    }
