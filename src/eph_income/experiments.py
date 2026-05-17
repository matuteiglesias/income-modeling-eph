"""Experiment runner for debug and baseline model comparisons."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from eph_income.contracts import assert_no_forbidden_predictors, get_forbidden_predictors, validate_target_contract
from eph_income.dataset import ROW_ID_COLUMN, resolve_project_path
from eph_income.pipelines import MODEL_DISPLAY_NAMES, enabled_model_configs, make_model_pipeline
from eph_income.splits import get_split_path, validate_split_assignments


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


def run_experiment(
    experiment_config: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    *,
    allow_full_run: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a debug or guarded full baseline model comparison and write outputs."""

    mode = _validate_runtime_mode(experiment_config, allow_full_run=allow_full_run)

    joined, feature_columns, target = prepare_experiment_frame(experiment_config, feature_contract)
    train = joined[joined["split"] == "train"]
    test = joined[joined["split"] == "test"]
    validation = joined[joined["split"] == "validation"]
    if train.empty or test.empty or validation.empty:
        raise ValueError("Train, test, and validation splits must all be non-empty.")

    cv_folds = int(experiment_config.get("cv", {}).get("folds", 5))
    scoring = str(experiment_config.get("cv", {}).get("scoring", "r2"))
    random_seed = int(experiment_config.get("experiment", {}).get("random_seed", 42))

    rows: list[dict[str, Any]] = []
    for model_key, model_config in enabled_model_configs(experiment_config).items():
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

        test_metrics = _evaluate(search.best_estimator_, test[feature_columns], test[target])
        validation_metrics = _evaluate(
            search.best_estimator_, validation[feature_columns], validation[target]
        )
        best_index = int(search.best_index_)
        rows.append(
            {
                "model": MODEL_DISPLAY_NAMES.get(model_key, model_key),
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
    comparison.to_csv(output_path, index=False)

    card = {
        "experiment_id": experiment_config.get("experiment", {}).get("id"),
        "mode": mode,
        "rows_used": int(len(joined)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "models": comparison["model"].tolist(),
        "model_comparison": str(output_path),
    }
    card_path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    return comparison, card
