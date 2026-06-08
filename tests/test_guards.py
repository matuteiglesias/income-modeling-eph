from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from eph_income.experiments import run_experiment
from eph_income.guards import (
    GuardDecision,
    apply_guard_decision,
    build_guard_plan,
    build_multicollinearity_audit,
    decide_multicollinearity_action,
    detect_constant_columns,
    detect_duplicate_columns,
    forbidden_predictors_guard,
    multicollinearity_guard_enabled,
    target_contract_guard,
    write_guard_decisions,
)


def _feature_contract() -> dict[str, object]:
    return {
        "target": {"name": "logP47T", "source": "P47T", "transform": "log10"},
        "forbidden_predictors": {
            "target": ["P47T", "logP47T"],
            "income_components": ["P21"],
            "identifiers": ["CODUSU"],
        },
    }


def _debug_experiment_config(tmp_path: Path) -> dict[str, object]:
    dataset_path = tmp_path / "modeling_dataset.parquet"
    split_path = tmp_path / "split_assignments.csv"
    dataset = pd.DataFrame(
        {
            "row_id": range(30),
            "logP47T": [1.0 + i * 0.01 for i in range(30)],
            "ANO4": [2022, 2023, 2024] * 10,
            "TRIMESTRE": [1, 2, 3, 4, 1] * 6,
            "feature_num": range(30),
            "feature_cat": ["a", "b", "c"] * 10,
        }
    )
    dataset.to_parquet(dataset_path, index=False)
    pd.DataFrame(
        {
            "row_id": range(30),
            "split": ["train"] * 21 + ["test"] * 6 + ["validation"] * 3,
        }
    ).to_csv(split_path, index=False)
    return {
        "experiment": {"id": "guards_debug_test", "random_seed": 42},
        "runtime": {"mode": "debug", "sample_n": None},
        "data": {
            "processed_dataset": str(dataset_path),
            "split_assignments": str(split_path),
        },
        "cv": {"folds": 2, "scoring": "r2"},
        "models": {
            "linear_regression": {"enabled": True, "grid": {"reg__fit_intercept": [True]}},
        },
        "outputs": {
            "model_comparison": str(tmp_path / "model_comparison_debug.csv"),
            "experiment_card": str(tmp_path / "debug_experiment.json"),
            "runs_dir": str(tmp_path / "runs"),
        },
    }


def test_build_guard_plan_missing_guards_uses_defaults() -> None:
    plan = build_guard_plan({})

    assert plan["schema_version"] == "guards_v1"
    assert plan["enabled"] is True
    assert plan["guards"]["target_contract"]["enabled"] is True
    assert plan["guards"]["forbidden_predictors"]["action"] == "fail_run"
    assert plan["guards"]["multicollinearity"]["enabled"] is False


def test_build_guard_plan_rejects_unknown_guard_name() -> None:
    with pytest.raises(ValueError, match="Unknown guard names"):
        build_guard_plan({"guards": {"typo_guard": {"enabled": True}}})


def test_target_contract_guard_passes_valid_contract() -> None:
    decision = target_contract_guard(_feature_contract(), build_guard_plan({}))

    assert decision.status == "pass"
    assert decision.action == "continue"
    assert decision.details["target"]["name"] == "logP47T"


def test_target_contract_guard_fails_invalid_contract() -> None:
    contract = _feature_contract()
    contract["target"] = {"name": "lnP47T", "source": "P47T", "transform": "log"}

    decision = target_contract_guard(contract, build_guard_plan({}))

    assert decision.status == "fail"
    assert decision.action == "fail_run"
    assert "Invalid target contract" in decision.reasons[0]


def test_forbidden_predictors_guard_fails_for_leakage_columns() -> None:
    decision = forbidden_predictors_guard(
        _feature_contract(), ["ANO4", "P47T", "logP47T", "P21"], build_guard_plan({})
    )

    assert decision.status == "fail"
    assert "P47T" in decision.reasons[0]
    assert "logP47T" in decision.reasons[0]
    assert "P21" in decision.reasons[0]


def test_forbidden_predictors_guard_passes_safe_feature_columns() -> None:
    decision = forbidden_predictors_guard(
        _feature_contract(), ["ANO4", "TRIMESTRE", "CH04"], build_guard_plan({})
    )

    assert decision.status == "pass"
    assert decision.details["feature_count"] == 3


def test_guard_decision_serializes_to_json(tmp_path) -> None:
    decision = GuardDecision(
        name="example",
        stage="config",
        status="pass",
        action="continue",
        reasons=["ok"],
        details={"x": 1},
    )

    path = write_guard_decisions(tmp_path, "run_1", [decision])
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert asdict(decision) == payload["decisions"][0]


def test_apply_guard_decision_raises_on_fail_run() -> None:
    decision = GuardDecision(
        name="example",
        stage="contract",
        status="fail",
        action="fail_run",
        reasons=["bad"],
    )

    with pytest.raises(ValueError, match="Guard example failed"):
        apply_guard_decision(decision)


def test_run_experiment_writes_guard_plan_json(tmp_path) -> None:
    _, card = run_experiment(_debug_experiment_config(tmp_path), _feature_contract())

    run_dir = Path(card["canonical_run_dir"])
    guard_plan = run_dir / "guards" / "guard_plan.json"

    assert guard_plan.exists()
    payload = json.loads(guard_plan.read_text(encoding="utf-8"))
    assert payload["run_id"] == card["run_id"]
    assert payload["schema_version"] == "guards_v1"


def test_successful_run_writes_guard_decisions_and_manifest_paths(tmp_path) -> None:
    _, card = run_experiment(_debug_experiment_config(tmp_path), _feature_contract())

    run_dir = Path(card["canonical_run_dir"])
    decisions_path = run_dir / "guards" / "guard_decisions.json"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))

    assert decisions_path.exists()
    assert manifest["paths"]["guard_plan"] == str(run_dir / "guards" / "guard_plan.json")
    assert manifest["paths"]["guard_decisions"] == str(decisions_path)
    assert {decision["name"] for decision in decisions["decisions"]} >= {
        "runtime_mode",
        "target_contract",
        "forbidden_predictors",
        "fixed_effect_design",
    }


def _multicollinearity_guard_config(**overrides: object) -> dict[str, object]:
    config = {
        "enabled": True,
        "stage": "pre_fit",
        "sample_n": 5000,
        "warn_vif": 5.0,
        "fail_vif": 10.0,
        "max_vif_features": 200,
        "fail_rank_deficiency": True,
        "fail_duplicate_columns": True,
        "fail_constant_columns": False,
        "fail_models": ["linear_regression"],
        "warn_only_models": ["ridge", "lasso", "hist_gradient_boosting", "mlp"],
    }
    config.update(overrides)
    return config


def _collinear_frame() -> pd.DataFrame:
    x1 = pd.Series(range(1, 41), dtype="float64")
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x1 * 2.0,
            "x3": [float(i % 7) for i in range(40)],
            "cat": ["a", "b"] * 20,
        }
    )


def test_missing_multicollinearity_guard_preserves_current_behavior(tmp_path) -> None:
    plan = build_guard_plan({})
    _, card = run_experiment(_debug_experiment_config(tmp_path), _feature_contract())
    run_dir = Path(card["canonical_run_dir"])

    assert multicollinearity_guard_enabled(plan) is False
    assert not (run_dir / "guards" / "multicollinearity_raw_summary.json").exists()


def test_enabled_multicollinearity_guard_writes_raw_artifacts(tmp_path) -> None:
    experiment_config = _debug_experiment_config(tmp_path)
    experiment_config["guards"] = {
        "multicollinearity": {
            "enabled": True,
            "fail_models": [],
            "warn_only_models": ["linear_regression"],
        }
    }

    _, card = run_experiment(experiment_config, _feature_contract())
    run_dir = Path(card["canonical_run_dir"])
    summary_path = run_dir / "guards" / "multicollinearity_raw_summary.json"
    features_path = run_dir / "guards" / "multicollinearity_raw_features.csv"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert summary_path.exists()
    assert features_path.exists()
    assert manifest["paths"]["multicollinearity_raw_summary"] == str(summary_path)
    assert manifest["paths"]["multicollinearity_raw_features"] == str(features_path)


def test_multicollinearity_detects_constant_columns() -> None:
    frame = pd.DataFrame({"x": [1, 1, 1], "y": [1, 2, 3], "z": [None, None, None]})

    assert detect_constant_columns(frame) == ["x", "z"]


def test_multicollinearity_detects_exact_duplicate_columns() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3], "x_copy": [1, 2, 3], "y": [3, 2, 1]})

    assert detect_duplicate_columns(frame) == {"x_copy": "x"}


def test_synthetic_strong_collinearity_produces_vif_above_fail(tmp_path) -> None:
    _, summary, _ = build_multicollinearity_audit(
        train_frame=_collinear_frame(),
        feature_columns=["x1", "x2", "x3", "cat"],
        run_id="run_collinear",
        guard_config=_multicollinearity_guard_config(),
        run_dir=tmp_path,
        random_seed=42,
    )

    assert summary["vif_computed"] is True
    assert summary["max_vif"] > summary["fail_vif"]
    assert summary["rank_deficient"] is True


def test_linear_regression_fails_when_vif_exceeds_threshold(tmp_path) -> None:
    guard_config = _multicollinearity_guard_config()
    _, summary, _ = build_multicollinearity_audit(
        train_frame=_collinear_frame(),
        feature_columns=["x1", "x2", "x3", "cat"],
        run_id="run_linear_fail",
        guard_config=guard_config,
        run_dir=tmp_path,
        random_seed=42,
    )

    decision = decide_multicollinearity_action(
        audit_summary=summary, model_key="linear_regression", guard_config=guard_config
    )

    assert decision.status == "fail"
    assert decision.action == "fail_run"
    assert "max_vif_above_fail" in decision.reasons


def test_ridge_warns_and_continues_when_vif_exceeds_threshold(tmp_path) -> None:
    guard_config = _multicollinearity_guard_config()
    _, summary, _ = build_multicollinearity_audit(
        train_frame=_collinear_frame(),
        feature_columns=["x1", "x2", "x3", "cat"],
        run_id="run_ridge_warn",
        guard_config=guard_config,
        run_dir=tmp_path,
        random_seed=42,
    )

    decision = decide_multicollinearity_action(
        audit_summary=summary, model_key="ridge", guard_config=guard_config
    )

    assert decision.status == "warn"
    assert decision.action == "warn_continue"


def test_lasso_warns_and_continues_when_vif_exceeds_threshold(tmp_path) -> None:
    guard_config = _multicollinearity_guard_config()
    _, summary, _ = build_multicollinearity_audit(
        train_frame=_collinear_frame(),
        feature_columns=["x1", "x2", "x3", "cat"],
        run_id="run_lasso_warn",
        guard_config=guard_config,
        run_dir=tmp_path,
        random_seed=42,
    )

    decision = decide_multicollinearity_action(
        audit_summary=summary, model_key="lasso", guard_config=guard_config
    )

    assert decision.status == "warn"
    assert decision.action == "warn_continue"


def test_hgb_is_not_blocked_by_vif_by_default(tmp_path) -> None:
    guard_config = _multicollinearity_guard_config()
    _, summary, _ = build_multicollinearity_audit(
        train_frame=_collinear_frame(),
        feature_columns=["x1", "x2", "x3", "cat"],
        run_id="run_hgb_warn",
        guard_config=guard_config,
        run_dir=tmp_path,
        random_seed=42,
    )

    decision = decide_multicollinearity_action(
        audit_summary=summary, model_key="hist_gradient_boosting", guard_config=guard_config
    )

    assert decision.status == "warn"
    assert decision.action == "warn_continue"


def test_vif_skips_cleanly_when_numeric_feature_count_exceeds_limit(tmp_path) -> None:
    frame = pd.DataFrame({f"x{i}": range(20) for i in range(3)})
    guard_config = _multicollinearity_guard_config(max_vif_features=2)

    _, summary, _ = build_multicollinearity_audit(
        train_frame=frame,
        feature_columns=["x0", "x1", "x2"],
        run_id="run_skip_vif",
        guard_config=guard_config,
        run_dir=tmp_path,
        random_seed=42,
    )

    assert summary["vif_computed"] is False
    assert "exceeds max_vif_features=2" in summary["vif_skipped_reason"]


def test_multicollinearity_decision_serialized_to_guard_decisions(tmp_path) -> None:
    experiment_config = _debug_experiment_config(tmp_path)
    experiment_config["guards"] = {
        "multicollinearity": {
            "enabled": True,
            "fail_models": [],
            "warn_only_models": ["linear_regression"],
        }
    }

    _, card = run_experiment(experiment_config, _feature_contract())
    run_dir = Path(card["canonical_run_dir"])
    decisions = json.loads(
        (run_dir / "guards" / "guard_decisions.json").read_text(encoding="utf-8")
    )

    assert "multicollinearity" in {decision["name"] for decision in decisions["decisions"]}
