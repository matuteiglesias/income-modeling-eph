from __future__ import annotations

import json

from eph_income.experiment_planning import (
    build_experiment_fit_plan,
    classify_experiment_cost,
    model_grid_size,
    requires_expensive_run,
    write_experiment_fit_plan,
)


def _config(*, sample_n=5000, folds=2, models=None):
    return {
        "experiment": {"id": "planning_test"},
        "runtime": {"mode": "debug", "sample_n": sample_n},
        "cv": {"folds": folds, "scoring": "r2", "return_train_score": True},
        "models": models
        or {
            "ridge": {
                "enabled": True,
                "grid": {"reg__alpha": [1, 2], "reg__fit_intercept": [True]},
            }
        },
    }


def test_model_grid_size_counts_cartesian_product():
    model_config = {"grid": {"a": [1, 2, 3], "b": [True, False]}}
    assert model_grid_size(model_config) == 6


def test_model_grid_size_counts_empty_grid_as_single_candidate():
    assert model_grid_size({"grid": {}}) == 1


def test_fit_plan_counts_enabled_models_and_expected_fits():
    plan = build_experiment_fit_plan(
        _config(
            folds=3,
            models={
                "linear_regression": {"enabled": True, "grid": {"reg__fit_intercept": [True]}},
                "ridge": {
                    "enabled": True,
                    "grid": {"reg__alpha": [1, 2], "reg__fit_intercept": [True, False]},
                },
                "lasso": {"enabled": False, "grid": {"reg__alpha": [1, 2, 3]}},
            },
        )
    )

    assert plan["enabled_models"] == ["linear_regression", "ridge"]
    assert plan["total_grid_configurations"] == 5
    assert plan["total_expected_cv_fits"] == 15
    assert plan["cost_class"] == "medium"


def test_full_mode_requires_allow_full_run_when_config_does_not_allow_it():
    config = _config(sample_n=1000)
    config["runtime"] = {"mode": "full", "sample_n": 1000}
    plan = build_experiment_fit_plan(config)
    assert plan["requires_allow_full_run"] is True


def test_full_mode_does_not_require_allow_full_run_when_config_allows_it():
    config = _config(sample_n=1000)
    config["runtime"] = {"mode": "full", "sample_n": 1000, "allow_full_run": True}
    plan = build_experiment_fit_plan(config)
    assert plan["requires_allow_full_run"] is False


def test_expensive_plan_requires_expensive_permission():
    plan = build_experiment_fit_plan(
        _config(
            sample_n=100_000,
            folds=3,
            models={
                "hist_gradient_boosting": {
                    "enabled": True,
                    "grid": {
                        "reg__learning_rate": [0.03, 0.075, 0.1],
                        "reg__max_iter": [100, 200, 400],
                        "reg__max_leaf_nodes": [15, 31, 63],
                    },
                }
            },
        )
    )

    assert plan["total_expected_cv_fits"] == 81
    assert plan["cost_class"] == "expensive"
    assert requires_expensive_run(plan) is True
    assert plan["requires_allow_expensive_run"] is True


def test_very_expensive_full_uncapped_plan():
    plan = build_experiment_fit_plan(
        _config(
            sample_n=None,
            folds=3,
            models={
                "hist_gradient_boosting": {
                    "enabled": True,
                    "grid": {
                        "a": list(range(10)),
                        "b": list(range(10)),
                    },
                }
            },
        )
    )

    assert plan["total_expected_cv_fits"] == 300
    assert plan["cost_class"] == "very_expensive"
    assert plan["requires_allow_expensive_run"] is True


def test_plan_is_json_serializable():
    plan = build_experiment_fit_plan(_config())
    json.dumps(plan)


def test_write_experiment_fit_plan(tmp_path):
    plan = build_experiment_fit_plan(_config())
    path = write_experiment_fit_plan(tmp_path, "run_123", plan)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "experiment_fit_plan.json"
    assert payload["run_id"] == "run_123"
    assert payload["schema_version"] == "experiment_fit_plan_v1"


def test_classify_experiment_cost_thresholds():
    assert classify_experiment_cost({"total_expected_cv_fits": 10, "sample_n": 5000}) == "cheap"
    assert classify_experiment_cost({"total_expected_cv_fits": 60, "sample_n": 5000}) == "medium"
    assert classify_experiment_cost({"total_expected_cv_fits": 200, "sample_n": 5000}) == "expensive"
    assert (
        classify_experiment_cost({"total_expected_cv_fits": 201, "sample_n": 5000})
        == "very_expensive"
    )
