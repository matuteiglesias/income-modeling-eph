from __future__ import annotations

import json
from pathlib import Path

from eph_income.diagnostics_registry import (
    build_diagnostics_plan,
    diagnostics_enabled,
    normalize_diagnostics_config,
)
from eph_income.experiments import run_experiment


def test_missing_diagnostics_block_defaults_to_standard_plan() -> None:
    config = {"models": {"hist_gradient_boosting": {"enabled": True, "grid": {}}}}

    normalized = normalize_diagnostics_config(config)
    plan = build_diagnostics_plan(
        experiment_config=config, enabled_models=["hist_gradient_boosting"]
    )

    assert diagnostics_enabled(config)
    assert normalized["profile"] == "standard"
    assert normalized["standard"]["residual_summary"] is True
    assert plan["resolved"]["standard"]["residual_summary"] is True


def test_profile_none_disables_optional_diagnostics() -> None:
    config = {"diagnostics": {"profile": "none"}}

    normalized = normalize_diagnostics_config(config)
    plan = build_diagnostics_plan(experiment_config=config, enabled_models=[])

    assert normalized["enabled"] is False
    assert diagnostics_enabled(config) is False
    assert all(enabled is False for enabled in plan["resolved"]["standard"].values())
    assert plan["resolved"]["hgb"]["enabled"] is False


def test_old_hgb_sweep_compatibility() -> None:
    config = {
        "diagnostics": {
            "sweep_type": "hgb_single_param",
            "primary_param": "max_leaf_nodes",
        }
    }

    normalized = normalize_diagnostics_config(config)

    assert normalized["profile"] == "sweep"
    assert normalized["compatibility"]["used_legacy_sweep_type_key"] is True
    assert normalized["sweeps"]["enabled"] is True
    assert normalized["sweeps"]["sweep_type"] == "hgb_single_param"
    assert normalized["sweeps"]["primary_param"] == "max_leaf_nodes"


def test_old_regularization_compatibility() -> None:
    config = {"diagnostics": {"regularization_sweep": True}}

    normalized = normalize_diagnostics_config(config)

    assert normalized["profile"] == "sweep"
    assert normalized["compatibility"]["used_legacy_regularization_sweep_key"] is True
    assert normalized["regularization"]["enabled"] is True


def test_new_explicit_style_keeps_unspecified_standard_defaults() -> None:
    config = {"diagnostics": {"enabled": True, "standard": {"residual_summary": True}}}

    normalized = normalize_diagnostics_config(config)

    assert normalized["standard"]["residual_summary"] is True
    assert normalized["standard"]["error_by_income_decile"] is True
    assert normalized["standard"]["prediction_distribution_summary"] is True


def test_hgb_only_plan_enables_hgb_and_auto_disables_coefficients() -> None:
    config = {"diagnostics": {"hgb": {"enabled": "auto"}}}

    plan = build_diagnostics_plan(
        experiment_config=config, enabled_models=["hist_gradient_boosting"]
    )

    assert plan["resolved"]["hgb"]["enabled"] is True
    assert plan["resolved"]["coefficients"]["enabled"] is False
    assert (
        "Coefficient diagnostics disabled because no coefficient-bearing model is enabled."
        in plan["notes"]
    )


def test_linear_fixed_effect_plan_enables_coefficient_exports() -> None:
    plan = build_diagnostics_plan(
        experiment_config={"diagnostics": {"coefficients": {"enabled": "auto"}}},
        enabled_models=["linear_regression", "ridge"],
        fixed_effects_used=[{"name": "region_fe", "columns": ["Region"]}],
    )

    assert plan["resolved"]["coefficients"]["enabled"] is True
    assert plan["resolved"]["coefficients"]["fixed_effect_coefficients"] is True


def test_regularization_plan_enables_when_alpha_grid_exists() -> None:
    config = {
        "diagnostics": {"regularization": {"enabled": "auto"}},
        "models": {
            "ridge": {"enabled": True, "grid": {"reg__alpha": [0.1, 1.0]}},
            "lasso": {"enabled": False, "grid": {}},
        },
    }

    plan = build_diagnostics_plan(experiment_config=config, enabled_models=["ridge"])

    assert plan["resolved"]["regularization"]["enabled"] is True
    assert plan["resolved"]["regularization"]["alpha_curves"] is True


def test_debug_run_writes_diagnostics_plan_and_manifest_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "modeling_dataset.parquet"
    split_path = tmp_path / "split_assignments.csv"
    runs_dir = tmp_path / "runs"

    import pandas as pd

    pd.DataFrame(
        {
            "row_id": range(30),
            "logP47T": [1.0 + i * 0.01 for i in range(30)],
            "ANO4": [2022, 2023, 2024] * 10,
            "TRIMESTRE": [1, 2, 3, 4, 1] * 6,
            "feature_num": range(30),
            "feature_cat": ["a", "b", "c"] * 10,
        }
    ).to_parquet(dataset_path, index=False)
    pd.DataFrame(
        {"row_id": range(30), "split": ["train"] * 21 + ["test"] * 6 + ["validation"] * 3}
    ).to_csv(split_path, index=False)

    experiment_config = {
        "experiment": {"id": "diagnostics_plan_run_test", "random_seed": 42},
        "runtime": {"mode": "debug", "sample_n": None},
        "data": {
            "processed_dataset": str(dataset_path),
            "split_assignments": str(split_path),
        },
        "cv": {"folds": 2, "scoring": "r2"},
        "diagnostics": {"enabled": True, "profile": "standard"},
        "models": {
            "linear_regression": {"enabled": True, "grid": {"reg__fit_intercept": [True]}},
        },
        "outputs": {"runs_dir": str(runs_dir)},
    }
    feature_contract = {
        "target": {"name": "logP47T", "source": "P47T", "transform": "log10"},
        "forbidden_predictors": {"target": ["P47T", "logP47T"], "identifiers": ["CODUSU"]},
    }

    _, card = run_experiment(experiment_config, feature_contract)

    run_dir = Path(card["canonical_run_dir"])
    plan_path = run_dir / "diagnostics" / "diagnostics_plan.json"
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["run_id"] == card["run_id"]
    assert plan["diagnostics_schema_version"] == "diagnostics_v1"
    assert plan["resolved"]["standard"]["residual_summary"] is True
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["paths"]["diagnostics_plan"] == str(plan_path)
