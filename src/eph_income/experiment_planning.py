"""Pre-flight planning for EPH income experiments.

This module inspects experiment YAML configuration only. It does not load datasets,
build sklearn pipelines, fit estimators, read predictions, or write diagnostics.
Its purpose is to make experiment cost visible before training starts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eph_income.observability import grid_configuration_count
from eph_income.pipelines import MODEL_DISPLAY_NAMES, enabled_model_configs

SCHEMA_VERSION = "experiment_fit_plan_v1"

COST_ORDER = {
    "cheap": 0,
    "medium": 1,
    "expensive": 2,
    "very_expensive": 3,
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _runtime_mapping(experiment_config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(experiment_config.get("runtime", {}))


def _cv_mapping(experiment_config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(experiment_config.get("cv", {}))


def _observability_mapping(experiment_config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(experiment_config.get("observability", {}))


def _sample_n(value: Any) -> int | None:
    """Normalize runtime.sample_n.

    None means the config does not impose an explicit row cap. Strings such as
    "full", "none", and "" are also treated as no explicit cap.
    """

    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "full"}:
        return None
    return int(value)


def _max_cost(first: str, second: str) -> str:
    return first if COST_ORDER[first] >= COST_ORDER[second] else second


def model_grid_size(model_config: Mapping[str, Any]) -> int:
    """Return the number of concrete parameter configurations for one model."""

    grid = model_config.get("grid", {})
    if grid is None:
        grid = {}
    if not isinstance(grid, Mapping) and not isinstance(grid, list):
        raise TypeError("model grid must be a mapping or a list of mappings")
    return int(grid_configuration_count(grid))


def build_model_fit_plan(
    model_key: str,
    model_config: Mapping[str, Any],
    *,
    cv_folds: int,
) -> dict[str, Any]:
    """Build the pre-fit plan block for one enabled model."""

    grid = model_config.get("grid", {}) or {}
    grid_keys = sorted(str(key) for key in grid.keys()) if isinstance(grid, Mapping) else []
    grid_configurations = model_grid_size(model_config)
    return {
        "model_key": model_key,
        "display_name": MODEL_DISPLAY_NAMES.get(model_key, model_key),
        "grid_configurations": int(grid_configurations),
        "expected_cv_fits": int(grid_configurations * cv_folds),
        "grid_keys": grid_keys,
    }


def classify_experiment_cost(plan: Mapping[str, Any]) -> str:
    """Classify planned fit cost using deterministic thresholds.

    This is not a wall-time estimate. It is a bounded-work classification based
    on expected CV fits plus a small sample-size modifier.
    """

    total_fits = int(plan["total_expected_cv_fits"])
    sample_n = plan.get("sample_n")

    if total_fits <= 10:
        cost = "cheap"
    elif total_fits <= 60:
        cost = "medium"
    elif total_fits <= 200:
        cost = "expensive"
    else:
        cost = "very_expensive"

    if sample_n is None:
        if total_fits > 100:
            cost = "very_expensive"
        elif total_fits > 50:
            cost = _max_cost(cost, "expensive")
    elif isinstance(sample_n, int) and sample_n >= 100_000 and total_fits > 50:
        cost = _max_cost(cost, "expensive")
        if total_fits > 200:
            cost = "very_expensive"

    return cost


def requires_full_run(plan: Mapping[str, Any]) -> bool:
    """Return whether this plan needs --allow-full-run to execute."""

    return str(plan["runtime_mode"]) in {"full", "sweep"} and not bool(
        plan.get("config_allows_full_run", False)
    )


def requires_expensive_run(plan: Mapping[str, Any]) -> bool:
    """Return whether this plan needs explicit expensive-run permission."""

    return COST_ORDER[str(plan["cost_class"])] >= COST_ORDER["expensive"]


def build_experiment_fit_plan(experiment_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build a JSON-serializable fit plan from an experiment config."""

    runtime = _runtime_mapping(experiment_config)
    cv = _cv_mapping(experiment_config)
    observability = _observability_mapping(experiment_config)

    cv_folds = int(cv.get("folds", 5))
    sample_n = _sample_n(runtime.get("sample_n"))
    enabled_models = enabled_model_configs(experiment_config)

    model_plans = [
        build_model_fit_plan(model_key, model_config, cv_folds=cv_folds)
        for model_key, model_config in enabled_models.items()
    ]

    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": _mapping(experiment_config.get("experiment", {})).get("id"),
        "runtime_mode": str(runtime.get("mode", "full")),
        "sample_n": sample_n,
        "config_allows_full_run": bool(runtime.get("allow_full_run", False)),
        "cv_folds": cv_folds,
        "scoring": str(cv.get("scoring", "r2")),
        "return_train_score": bool(cv.get("return_train_score", False)),
        "observability": {
            "sklearn_verbose": int(observability.get("sklearn_verbose", 0)),
            "heartbeat_seconds": float(observability.get("heartbeat_seconds", 10)),
        },
        "enabled_models": list(enabled_models.keys()),
        "models": model_plans,
        "enabled_model_count": int(len(model_plans)),
        "total_grid_configurations": int(
            sum(model["grid_configurations"] for model in model_plans)
        ),
        "total_expected_cv_fits": int(sum(model["expected_cv_fits"] for model in model_plans)),
        "notes": [],
    }
    plan["cost_class"] = classify_experiment_cost(plan)
    plan["requires_allow_full_run"] = requires_full_run(plan)
    plan["requires_allow_expensive_run"] = requires_expensive_run(plan)
    return plan


def write_experiment_fit_plan(run_dir: str | Path, run_id: str, plan: Mapping[str, Any]) -> Path:
    """Write the fit plan as run-local evidence."""

    path = Path(run_dir) / "artifacts" / "experiment_fit_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, **dict(plan)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def format_experiment_fit_plan(plan: Mapping[str, Any]) -> str:
    """Format a fit plan for terminal reading."""

    lines = [
        "Experiment fit plan",
        "===================",
        "",
        f"experiment_id: {plan.get('experiment_id')}",
        f"runtime_mode: {plan.get('runtime_mode')}",
        f"sample_n: {plan.get('sample_n') if plan.get('sample_n') is not None else 'full'}",
        f"cv_folds: {plan.get('cv_folds')}",
        f"scoring: {plan.get('scoring')}",
        f"return_train_score: {plan.get('return_train_score')}",
        "",
        "Models",
        "------",
    ]

    for model in plan.get("models", []):
        lines.append(
            f"{model['display_name']:<34} "
            f"grid={model['grid_configurations']:<6} "
            f"fits={model['expected_cv_fits']:<6}"
        )
        grid_keys = model.get("grid_keys") or []
        if grid_keys:
            lines.append("  grid_keys: " + ", ".join(grid_keys))

    lines.extend(
        [
            "",
            "Totals",
            "------",
            f"enabled_model_count: {plan.get('enabled_model_count')}",
            f"total_grid_configurations: {plan.get('total_grid_configurations')}",
            f"total_expected_cv_fits: {plan.get('total_expected_cv_fits')}",
            f"cost_class: {plan.get('cost_class')}",
            f"requires_allow_full_run: {plan.get('requires_allow_full_run')}",
            f"requires_allow_expensive_run: {plan.get('requires_allow_expensive_run')}",
        ]
    )

    notes = plan.get("notes") or []
    if notes:
        lines.extend(["", "Notes", "-----"])
        lines.extend(f"- {note}" for note in notes)

    return "\n".join(lines)
