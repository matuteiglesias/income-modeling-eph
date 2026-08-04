"""Experiment runner for debug and baseline model comparisons."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from datetime import datetime, timezone

from eph_income.diagnostics import _write_hgb_sweep_diagnostics
from eph_income.diagnostics_registry import build_diagnostics_plan
from eph_income.experiment_artifacts import (
    _coefficient_norms,
    _evaluate,
    _fixed_effect_coefficient_rows,
    _make_dataset_card,
    _make_run_id,
    _metrics_long,
    _prediction_frame,
    _prepare_run_directory,
    _resolve_output_path,
    _write_best_coefficient_summary,
    _write_diagnostics_from_predictions,
    _write_fixed_effect_coefficients,
    _write_predictions,
    _transformed_coefficient_rows,
    _write_transformed_coefficients,
    _write_training_frame_sample,
    _write_yaml_snapshot,
)
from eph_income.experiment_frame import (
    _fixed_effect_drop_first_columns,
    apply_feature_view,
    apply_fixed_effects,
    build_feature_type_spec,
    prepare_experiment_frame,
)
from eph_income.guards import (
    GuardDecision,
    apply_guard_decision,
    build_guard_plan,
    build_multicollinearity_audit,
    decide_multicollinearity_action,
    fixed_effect_design_guard,
    forbidden_predictors_guard,
    multicollinearity_guard_enabled,
    runtime_mode_guard,
    target_contract_guard,
    write_guard_decisions,
    write_guard_plan,
)
from eph_income.observability import (
    grid_configuration_count,
    heartbeat,
    log_model_fit_end,
    log_model_fit_start,
)
from eph_income.pipelines import MODEL_DISPLAY_NAMES, enabled_model_configs, make_model_pipeline
from eph_income.dataset import ROW_ID_COLUMN

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


def _heartbeat_interval_seconds(experiment_config: Mapping[str, Any]) -> float:
    """Return configured heartbeat interval, defaulting to 10 seconds."""

    runtime = _runtime_mapping(experiment_config)
    value = runtime.get("heartbeat_interval_seconds", 10)
    if value is None:
        return 10.0
    interval = float(value)
    if interval < 0:
        raise ValueError("runtime.heartbeat_interval_seconds must be non-negative.")
    return interval


def _diagnostics_context_columns(experiment_config: Mapping[str, Any]) -> list[str]:
    """Return row-level context columns to preserve in prediction artifacts."""

    diagnostics = experiment_config.get("diagnostics", {})
    artifacts = experiment_config.get("artifacts", {})
    candidates = []
    if isinstance(diagnostics, Mapping):
        candidates = diagnostics.get("context_columns", [])
    if not candidates and isinstance(artifacts, Mapping):
        candidates = artifacts.get("prediction_context_columns", [])
    if candidates is None:
        return []
    if not isinstance(candidates, list):
        raise TypeError("diagnostics.context_columns must be a list when provided.")
    return list(dict.fromkeys(str(column) for column in candidates))


def _validate_feature_count_consistency(comparison: pd.DataFrame) -> None:
    """Fail loudly if model rows do not report a common feature count."""

    if comparison.empty:
        raise ValueError("Experiment produced no model result rows.")
    if "feature_count" not in comparison.columns:
        raise ValueError("Experiment results must include feature_count.")
    if comparison["feature_count"].nunique(dropna=False) != 1:
        raise ValueError("All enabled models must report the same feature_count.")


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
        train_score = (
            cv_results.loc[row_index, "mean_train_score"]
            if "mean_train_score" in cv_results.columns
            else np.nan
        )
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



MIN_INFORMATIVE_PLOT_POINTS = 3


def _has_informative_points(
    frame: pd.DataFrame,
    *,
    x_column: str | None = None,
    min_points: int = MIN_INFORMATIVE_PLOT_POINTS,
) -> bool:
    """Return whether a line/sweep plot has enough information to be useful."""

    if len(frame) < min_points:
        return False
    if x_column is not None:
        if x_column not in frame.columns:
            return False
        return int(frame[x_column].nunique(dropna=True)) >= min_points
    return True


def _write_plot_skip_reasons(path: Path, reasons: dict[str, str]) -> Path | None:
    """Write plot skip reasons as run-local evidence."""

    if not reasons:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reasons, indent=2, sort_keys=True), encoding="utf-8")
    return path


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



def _write_regularization_plots(
    run_dir: Path, model_name: str, path_summary: pd.DataFrame
) -> dict[str, Path]:
    """Write regularization plots only when alpha path has enough points."""

    model_slug = model_name.lower()
    plots_dir = run_dir / "plots" / "regularization"
    diagnostics_dir = run_dir / "diagnostics"
    plots_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    skipped: dict[str, str] = {}

    if not _has_informative_points(path_summary, x_column="alpha"):
        skipped["regularization_plots"] = (
            "Skipped regularization plots because the alpha path has fewer than "
            f"{MIN_INFORMATIVE_PLOT_POINTS} distinct alpha values."
        )
        skip_path = _write_plot_skip_reasons(
            diagnostics_dir / f"{model_name}_regularization_plot_skip_reasons.json",
            skipped,
        )
        if skip_path is not None:
            outputs[skip_path.name] = skip_path
        return outputs

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
                "mean_train_score": float(mean_train_score)
                if pd.notna(mean_train_score)
                else np.nan,
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
    grouped = plot_frame.groupby(["learning_rate", "max_iter"], as_index=False)[
        "mean_test_score"
    ].mean()
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


def _plot_hgb_train_vs_cv_top_configs(
    summary: pd.DataFrame, output_path: Path, top_n: int = 20
) -> None:
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
    """Write HGB tables and only produce sweep plots when they are informative."""

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

    skipped: dict[str, str] = {}

    # Global guard: a one- or two-configuration HGB run is a benchmark, not a sweep.
    if len(summary) < MIN_INFORMATIVE_PLOT_POINTS:
        skipped["hgb_sweep_plots"] = (
            "Skipped HGB sweep plots because cv_results contains fewer than "
            f"{MIN_INFORMATIVE_PLOT_POINTS} configurations. Tables were still written."
        )
        skip_path = _write_plot_skip_reasons(
            diagnostics_dir / "hgb_plot_skip_reasons.json",
            skipped,
        )
        if skip_path is not None:
            paths["hgb_plot_skip_reasons"] = skip_path
        return paths

    plot_paths = {
        "hgb_learning_rate_vs_max_iter": plots_dir / "hgb_learning_rate_vs_max_iter.png",
        "hgb_max_leaf_nodes_vs_cv_r2": plots_dir / "hgb_max_leaf_nodes_vs_cv_r2.png",
        "hgb_min_samples_leaf_vs_cv_r2": plots_dir / "hgb_min_samples_leaf_vs_cv_r2.png",
        "hgb_l2_regularization_vs_cv_r2": plots_dir / "hgb_l2_regularization_vs_cv_r2.png",
        "hgb_train_vs_cv_r2_top_configs": plots_dir / "hgb_train_vs_cv_r2_top_configs.png",
    }

    # learning_rate vs max_iter needs at least 3 observed combinations.
    lr_iter_frame = _numeric_plot_frame(summary, ["learning_rate", "max_iter", "mean_test_score"])
    lr_iter_points = (
        lr_iter_frame.groupby(["learning_rate", "max_iter"], as_index=False)["mean_test_score"]
        .mean()
        if not lr_iter_frame.empty
        else lr_iter_frame
    )
    if len(lr_iter_points) >= MIN_INFORMATIVE_PLOT_POINTS:
        _plot_hgb_learning_rate_vs_max_iter(
            summary,
            plot_paths["hgb_learning_rate_vs_max_iter"],
        )
    else:
        skipped["hgb_learning_rate_vs_max_iter"] = (
            "Skipped because fewer than "
            f"{MIN_INFORMATIVE_PLOT_POINTS} learning_rate/max_iter combinations are available."
        )

    for param, key in [
        ("max_leaf_nodes", "hgb_max_leaf_nodes_vs_cv_r2"),
        ("min_samples_leaf", "hgb_min_samples_leaf_vs_cv_r2"),
        ("l2_regularization", "hgb_l2_regularization_vs_cv_r2"),
    ]:
        plot_frame = _numeric_plot_frame(summary, [param, "mean_test_score"])
        grouped = (
            plot_frame.groupby(param, as_index=False)["mean_test_score"].mean()
            if not plot_frame.empty
            else plot_frame
        )
        if len(grouped) >= MIN_INFORMATIVE_PLOT_POINTS and grouped[param].nunique(dropna=True) >= 2:
            _plot_hgb_param_vs_cv_r2(summary, param, plot_paths[key])
        else:
            skipped[key] = (
                f"Skipped because {param} has fewer than "
                f"{MIN_INFORMATIVE_PLOT_POINTS} plotted points."
            )

    if _has_informative_points(summary):
        _plot_hgb_train_vs_cv_top_configs(
            summary,
            plot_paths["hgb_train_vs_cv_r2_top_configs"],
        )
    else:
        skipped["hgb_train_vs_cv_r2_top_configs"] = (
            "Skipped because fewer than "
            f"{MIN_INFORMATIVE_PLOT_POINTS} ranked configurations are available."
        )

    paths.update({key: path for key, path in plot_paths.items() if path.exists()})

    skip_path = _write_plot_skip_reasons(
        diagnostics_dir / "hgb_plot_skip_reasons.json",
        skipped,
    )
    if skip_path is not None:
        paths["hgb_plot_skip_reasons"] = skip_path

    return paths


def run_experiment(
    experiment_config: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    *,
    allow_full_run: bool = False,
    freeze_estimator: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a debug or guarded full baseline model comparison and write outputs."""

    run_id = _make_run_id(experiment_config)
    run_dir = _prepare_run_directory(experiment_config, run_id)
    guard_decisions: list[GuardDecision] = []
    guard_plan = build_guard_plan(experiment_config)
    guard_plan_path = write_guard_plan(run_dir, run_id, guard_plan)
    guard_decisions_path = run_dir / "guards" / "guard_decisions.json"

    try:
        mode = _validate_runtime_mode(experiment_config, allow_full_run=allow_full_run)
        runtime_decision = runtime_mode_guard(mode, guard_plan)
        guard_decisions.append(runtime_decision)
        apply_guard_decision(runtime_decision)

        target_decision = target_contract_guard(feature_contract, guard_plan)
        guard_decisions.append(target_decision)
        apply_guard_decision(target_decision)

        joined, feature_columns, target = prepare_experiment_frame(experiment_config, feature_contract)
        forbidden_decision = forbidden_predictors_guard(
            feature_contract, feature_columns, guard_plan
        )
        guard_decisions.append(forbidden_decision)
        apply_guard_decision(forbidden_decision)
    except Exception:
        write_guard_decisions(run_dir, run_id, guard_decisions)
        raise
    joined, feature_columns, feature_view_metadata = apply_feature_view(
        experiment_config=experiment_config,
        feature_contract=feature_contract,
        joined=joined,
        feature_columns=feature_columns,
    )
    try:
        joined, feature_columns, fixed_effects_used = apply_fixed_effects(
            experiment_config=experiment_config,
            feature_contract=feature_contract,
            joined=joined,
            feature_columns=feature_columns,
        )
        fixed_effect_decision = fixed_effect_design_guard(fixed_effects_used, guard_plan)
        guard_decisions.append(fixed_effect_decision)
        apply_guard_decision(fixed_effect_decision)
        final_forbidden_decision = forbidden_predictors_guard(
            feature_contract, feature_columns, guard_plan
        )
        guard_decisions.append(final_forbidden_decision)
        apply_guard_decision(final_forbidden_decision)
    except Exception:
        write_guard_decisions(run_dir, run_id, guard_decisions)
        raise
    fixed_effect_drop_first_columns = _fixed_effect_drop_first_columns(fixed_effects_used)
    feature_type_spec = build_feature_type_spec(feature_contract, feature_columns, fixed_effects_used)
    feature_metadata = list(feature_type_spec.get("feature_metadata", []))
    prediction_context_columns = _diagnostics_context_columns(experiment_config)

    feature_metadata_path = run_dir / "artifacts" / "feature_metadata.json"
    feature_metadata_path.write_text(
        json.dumps(feature_metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    train = joined[joined["split"] == "train"]
    test = joined[joined["split"] == "test"]
    validation = joined[joined["split"] == "validation"]
    if train.empty or test.empty or validation.empty:
        raise ValueError("Train, test, and validation splits must all be non-empty.")

    cv_folds = int(experiment_config.get("cv", {}).get("folds", 5))
    scoring = str(experiment_config.get("cv", {}).get("scoring", "r2"))
    heartbeat_interval = _heartbeat_interval_seconds(experiment_config)
    random_seed = int(experiment_config.get("experiment", {}).get("random_seed", 42))

    _write_yaml_snapshot(run_dir / "config_used.yaml", experiment_config)
    _write_yaml_snapshot(run_dir / "feature_contract_used.yaml", feature_contract)
    dataset_card = _make_dataset_card(
        experiment_config, feature_contract, joined, feature_columns, target, fixed_effects_used
    )
    (run_dir / "dataset_card.json").write_text(
        json.dumps(dataset_card, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2, sort_keys=True), encoding="utf-8"
    )
    training_frame_sample_paths = _write_training_frame_sample(
        run_dir=run_dir,
        run_id=run_id,
        train=train,
        feature_columns=feature_columns,
        target=target,
        experiment_config=experiment_config,
        feature_contract=feature_contract,
    )

    enabled_models = list(enabled_model_configs(experiment_config).keys())
    multicollinearity_artifact_paths: dict[str, Path] = {}
    multicollinearity_summary: dict[str, Any] | None = None
    multicollinearity_config = guard_plan["guards"].get("multicollinearity", {})
    if multicollinearity_guard_enabled(guard_plan):
        _, multicollinearity_summary, multicollinearity_artifact_paths = (
            build_multicollinearity_audit(
                train_frame=train,
                feature_columns=feature_columns,
                run_id=run_id,
                guard_config=multicollinearity_config,
                run_dir=run_dir,
                random_seed=random_seed,
            )
        )

    diagnostics_plan = build_diagnostics_plan(
        experiment_config=experiment_config,
        enabled_models=enabled_models,
        fixed_effects_used=fixed_effects_used,
    )
    diagnostics_plan_payload = {
        "run_id": run_id,
        **diagnostics_plan,
        "enabled_model_display_names": {
            model_key: MODEL_DISPLAY_NAMES.get(model_key, model_key)
            for model_key in enabled_models
        },
    }
    diagnostics_plan_path = run_dir / "diagnostics" / "diagnostics_plan.json"
    diagnostics_plan_path.write_text(
        json.dumps(diagnostics_plan_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    resolved_diagnostics = diagnostics_plan.get("resolved", diagnostics_plan)
    coefficients_plan = resolved_diagnostics.get("coefficients", {})
    regularization_plan = resolved_diagnostics.get("regularization", {})
    write_coefficient_diagnostics = bool(
        coefficients_plan.get("best_coefficients", False)
        or coefficients_plan.get("standardized_coefficients", False)
        or coefficients_plan.get("fixed_effect_coefficients", False)
    )
    write_regularization_diagnostics = bool(
        regularization_plan.get("coefficient_paths", False)
    )

    rows: list[dict[str, Any]] = []
    fixed_effect_coefficient_rows: list[dict[str, Any]] = []
    transformed_coefficient_rows: list[dict[str, Any]] = []
    predictions_by_split: dict[str, list[pd.DataFrame]] = {"test": [], "validation": []}
    frozen_estimator_path: Path | None = None

    for model_key, model_config in enabled_model_configs(experiment_config).items():
        if multicollinearity_summary is not None:
            multicollinearity_decision = decide_multicollinearity_action(
                audit_summary=multicollinearity_summary,
                model_key=model_key,
                guard_config=multicollinearity_config,
            )
            guard_decisions.append(multicollinearity_decision)
            try:
                apply_guard_decision(multicollinearity_decision)
            except Exception:
                write_guard_decisions(run_dir, run_id, guard_decisions)
                raise

        model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        pipeline = make_model_pipeline(
            model_key,
            train[feature_columns],
            random_seed,
            drop_first_categorical_columns=fixed_effect_drop_first_columns,
            feature_type_spec=feature_type_spec,
        )
        grid = model_config.get("grid", {})

        return_train_score = bool(experiment_config.get("cv", {}).get("return_train_score", False))
        observability = experiment_config.get("observability", {})
        sklearn_verbose = int(observability.get("sklearn_verbose", 0))
        heartbeat_interval = float(observability.get("heartbeat_seconds", 10))

        search = GridSearchCV(
            pipeline,
            grid,
            cv=cv_folds,
            scoring=scoring,
            refit=True,
            n_jobs=None,
            return_train_score=return_train_score,
            verbose=sklearn_verbose,
        )
        grid_configurations = grid_configuration_count(grid)
        log_model_fit_start(
            run_id=run_id,
            model_name=model_name,
            grid_configurations=grid_configurations,
            cv_folds=cv_folds,
            train_rows=len(train),
            feature_count=len(feature_columns),
            sklearn_verbose=sklearn_verbose,
            heartbeat_interval=heartbeat_interval,
        )
        start = time.perf_counter()
        with heartbeat(
            f"GridSearchCV model={model_name} run_id={run_id}",
            interval_seconds=heartbeat_interval,
        ):
            search.fit(train[feature_columns], train[target])
        if freeze_estimator:
            if experiment_config.get("experiment", {}).get("id") != "hgb_quick_benchmark_v1":
                raise ValueError("Freeze-only estimator persistence is limited to hgb_quick_benchmark_v1.")
            if frozen_estimator_path is not None:
                raise ValueError("Freeze path requires exactly one enabled model.")
            import joblib

            frozen_estimator_path = run_dir / "artifacts" / "model_pipeline.joblib"
            joblib.dump(search.best_estimator_, frozen_estimator_path)
        fit_time = time.perf_counter() - start

        cv_results = pd.DataFrame(search.cv_results_)
        cv_results.insert(0, "run_id", run_id)
        cv_results.insert(1, "model", model_name)
        cv_path = run_dir / "cv_results" / f"{model_name}.csv"
        cv_results.to_csv(cv_path, index=False)
        model_artifact_paths: list[Path] = [cv_path]

        if model_key == "hist_gradient_boosting":
            model_artifact_paths.extend(_write_hgb_diagnostics(run_dir, cv_results).values())
            sweep_diagnostic_paths, _ = _write_hgb_sweep_diagnostics(
                run_dir, config=experiment_config, run_id=run_id
            )
            model_artifact_paths.extend(sweep_diagnostic_paths.values())

        fixed_effect_coefficient_rows.extend(
            _fixed_effect_coefficient_rows(
                run_id=run_id,
                model_name=model_name,
                estimator=search.best_estimator_,
                fixed_effects_used=fixed_effects_used,
            )
        )

        if write_coefficient_diagnostics:
            transformed_coefficient_rows.extend(
                _transformed_coefficient_rows(
                    run_id=run_id,
                    experiment_id=experiment_config.get("experiment", {}).get("id"),
                    model_name=model_name,
                    estimator=search.best_estimator_,
                    feature_metadata=feature_metadata,
                )
            )

        if write_coefficient_diagnostics:
            coefficient_path = _write_best_coefficient_summary(
                run_dir=run_dir,
                model_name=model_name,
                run_id=run_id,
                estimator=search.best_estimator_,
            )
            if coefficient_path is not None:
                model_artifact_paths.append(coefficient_path)
            if write_regularization_diagnostics:
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
                    path_summary_path = (
                        run_dir / "diagnostics" / f"{model_name}_regularization_path_summary.csv"
                    )
                    path_summary.to_csv(
                        path_summary_path,
                        index=False,
                    )
                    model_artifact_paths.append(path_summary_path)
                    model_artifact_paths.extend(
                        _write_regularization_plots(run_dir, model_name, path_summary).values()
                    )

        best_index = int(search.best_index_)
        best_cv_score = float(search.cv_results_["mean_test_score"][best_index])
        log_model_fit_end(
            run_id=run_id,
            model_name=model_name,
            elapsed_seconds=fit_time,
            best_params=search.best_params_,
            best_cv_score=best_cv_score,
            artifact_paths=model_artifact_paths,
        )

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
                    context_columns=prediction_context_columns,
                )
            )
        rows.append(
            {
                "model": model_name,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "feature_count": int(len(feature_columns)),
                "cv_r2_mean": best_cv_score,
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

    fixed_effect_coefficients_path = _write_fixed_effect_coefficients(
        run_dir=run_dir, rows=fixed_effect_coefficient_rows
    )
    transformed_coefficient_paths = _write_transformed_coefficients(
        run_dir=run_dir, rows=transformed_coefficient_rows
    )

    comparison = pd.DataFrame(rows)
    _validate_feature_count_consistency(comparison)
    outputs = experiment_config.get("outputs", {})
    output_path = (
        _resolve_output_path(outputs["model_comparison"]) if "model_comparison" in outputs else None
    )
    card_path = (
        _resolve_output_path(outputs["experiment_card"]) if "experiment_card" in outputs else None
    )
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
        "feature_metadata": str(feature_metadata_path),
        "model_comparison": str(run_model_comparison_path),
        "metrics_long": str(run_dir / "metrics" / "metrics_long.csv"),
        "test_predictions": str(prediction_paths["test"]),
        "validation_predictions": str(prediction_paths["validation"]),
        "residual_summary": str(diagnostic_paths["residual_summary"]),
        "error_by_income_decile": str(diagnostic_paths["error_by_income_decile"]),
        "prediction_distribution_summary": str(diagnostic_paths["prediction_distribution_summary"]),
        "diagnostics_plan": str(diagnostics_plan_path),
        "guard_plan": str(guard_plan_path),
        "guard_decisions": str(guard_decisions_path),
    }
    hgb_diagnostics = {
        path.stem: str(path) for path in sorted((run_dir / "diagnostics").glob("hgb_*.csv"))
    }
    hgb_plots = {
        path.stem: str(path) for path in sorted((run_dir / "plots" / "hgb").glob("hgb_*.png"))
    }
    hgb_sweep_plots = {
        path.stem: str(path)
        for path in sorted((run_dir / "plots" / "hgb_sweeps").glob("hgb_*.png"))
    }
    hgb_sweep_markdown = {
        path.stem: str(path) for path in sorted((run_dir / "diagnostics").glob("hgb_sweep*.md"))
    }
    manifest_paths.update(hgb_diagnostics)
    manifest_paths.update(hgb_plots)
    manifest_paths.update(hgb_sweep_plots)
    manifest_paths.update(hgb_sweep_markdown)
    if fixed_effect_coefficients_path is not None:
        manifest_paths["fixed_effect_coefficients"] = str(fixed_effect_coefficients_path)
    manifest_paths.update({key: str(path) for key, path in transformed_coefficient_paths.items()})
    manifest_paths.update({key: str(path) for key, path in multicollinearity_artifact_paths.items()})
    manifest_paths.update({key: str(path) for key, path in training_frame_sample_paths.items()})
    if output_path is not None:
        manifest_paths["model_comparison_convenience_copy"] = str(output_path)

    guard_decisions_path = write_guard_decisions(run_dir, run_id, guard_decisions)

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
        "feature_view": feature_view_metadata,
        "feature_metadata": feature_metadata,
        "prediction_context_columns": prediction_context_columns,
        "fixed_effects_used": fixed_effects_used,
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
        "feature_metadata": feature_metadata,
        "feature_metadata_path": str(feature_metadata_path),
        "feature_view": feature_view_metadata,
        "fixed_effects_used": fixed_effects_used,
        "models": comparison["model"].tolist(),
        "model_comparison": str(output_path or run_model_comparison_path),
        "canonical_run_dir": str(run_dir),
        "canonical_model_comparison": str(run_model_comparison_path),
        "run_manifest": str(run_dir / "run_manifest.json"),
        "training_frame_sample": (
            str(training_frame_sample_paths["training_frame_sample"])
            if "training_frame_sample" in training_frame_sample_paths
            else None
        ),
        "training_frame_sample_metadata": (
            str(training_frame_sample_paths["training_frame_sample_metadata"])
            if "training_frame_sample_metadata" in training_frame_sample_paths
            else None
        ),
    }
    run_card_path = run_dir / "experiment_card.json"
    run_card_path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    if card_path is not None:
        card_path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    return comparison, card
