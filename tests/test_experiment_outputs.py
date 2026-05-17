from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from eph_income.config import load_experiment_config
from eph_income.experiments import _validate_feature_count_consistency, run_experiment
from eph_income.pipelines import enabled_model_configs


def test_debug_experiment_writes_expected_outputs(tmp_path) -> None:
    dataset_path = tmp_path / "modeling_dataset.parquet"
    split_path = tmp_path / "split_assignments.csv"
    output_csv = tmp_path / "model_comparison_debug.csv"
    card_path = tmp_path / "debug_experiment.json"
    runs_dir = tmp_path / "runs"

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
    splits = pd.DataFrame(
        {
            "row_id": range(30),
            "split": ["train"] * 21 + ["test"] * 6 + ["validation"] * 3,
        }
    )
    splits.to_csv(split_path, index=False)

    experiment_config = {
        "experiment": {"id": "debug_test", "random_seed": 42},
        "runtime": {"mode": "debug", "sample_n": None, "enabled_models": ["linear_regression", "ridge"]},
        "data": {
            "processed_dataset": str(dataset_path),
            "split_assignments": str(split_path),
        },
        "cv": {"folds": 2, "scoring": "r2"},
        "models": {
            "linear_regression": {"enabled": True, "grid": {"reg__fit_intercept": [True]}},
            "ridge": {"enabled": True, "grid": {"reg__alpha": [1.0], "reg__fit_intercept": [True]}},
        },
        "outputs": {
            "model_comparison": str(output_csv),
            "experiment_card": str(card_path),
            "runs_dir": str(runs_dir),
        },
    }
    feature_contract = {
        "target": {"name": "logP47T", "source": "P47T", "transform": "log10"},
        "forbidden_predictors": {"target": ["P47T", "logP47T"], "identifiers": ["CODUSU"]},
    }

    comparison, card = run_experiment(experiment_config, feature_contract)

    assert output_csv.exists()
    assert card_path.exists()
    assert comparison["model"].tolist() == ["LinearRegression", "Ridge"]
    expected_columns = {
        "model",
        "best_params",
        "feature_count",
        "cv_r2_mean",
        "cv_r2_std",
        "test_r2",
        "test_mae",
        "test_mse",
        "validation_r2",
        "validation_mae",
        "validation_mse",
        "fit_time_seconds",
    }
    assert expected_columns.issubset(comparison.columns)
    assert comparison["feature_count"].nunique() == 1
    assert card["feature_columns"] == ["ANO4", "TRIMESTRE", "feature_num", "feature_cat"]
    leakage_columns = {
        "P47T",
        "logP47T",
        "P21",
        "T_VI",
        "V12_M",
        "V2_M",
        "V3_M",
        "V5_M",
        "TOT_P12",
        "PP08D1",
        "logP21",
        "logT_VI",
        "logV12_M",
        "logV2_M",
        "logV3_M",
        "logV5_M",
        "logTOT_P12",
        "logPP08D1",
        "CODUSU",
        "Q",
        "source_year",
        "INGRESO",
    }
    assert leakage_columns.isdisjoint(card["feature_columns"])
    written_card = json.loads(card_path.read_text(encoding="utf-8"))
    assert written_card["mode"] == "debug"


def test_debug_experiment_writes_local_run_artifacts(tmp_path) -> None:
    dataset_path = tmp_path / "modeling_dataset.parquet"
    split_path = tmp_path / "split_assignments.csv"
    output_csv = tmp_path / "tables" / "model_comparison_debug.csv"
    card_path = tmp_path / "cards" / "debug_experiment.json"
    runs_dir = tmp_path / "runs"

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

    experiment_config = {
        "experiment": {"id": "debug_test", "random_seed": 42},
        "runtime": {"mode": "debug", "sample_n": None},
        "data": {
            "processed_dataset": str(dataset_path),
            "split_assignments": str(split_path),
        },
        "cv": {"folds": 2, "scoring": "r2"},
        "models": {
            "linear_regression": {"enabled": True, "grid": {"reg__fit_intercept": [True]}},
            "ridge": {"enabled": True, "grid": {"reg__alpha": [1.0], "reg__fit_intercept": [True]}},
        },
        "outputs": {
            "model_comparison": str(output_csv),
            "experiment_card": str(card_path),
            "runs_dir": str(runs_dir),
        },
    }
    feature_contract = {
        "target": {"name": "logP47T", "source": "P47T", "transform": "log10"},
        "forbidden_predictors": {"target": ["P47T", "logP47T"], "identifiers": ["CODUSU"]},
    }

    comparison, card = run_experiment(experiment_config, feature_contract)

    run_dir = Path(card["canonical_run_dir"])
    assert run_dir.is_dir()
    expected_artifacts = [
        "run_manifest.json",
        "config_used.yaml",
        "feature_contract_used.yaml",
        "dataset_card.json",
        "feature_columns.json",
        "metrics/model_comparison.csv",
        "metrics/metrics_long.csv",
        "predictions/test_predictions.parquet",
        "predictions/validation_predictions.parquet",
        "cv_results/LinearRegression.csv",
        "cv_results/Ridge.csv",
        "diagnostics/residual_summary.csv",
        "diagnostics/error_by_income_decile.csv",
        "diagnostics/prediction_distribution_summary.csv",
    ]
    for relative_path in expected_artifacts:
        assert (run_dir / relative_path).exists(), relative_path

    pd.testing.assert_frame_equal(pd.read_csv(run_dir / "metrics/model_comparison.csv"), comparison)
    pd.testing.assert_frame_equal(pd.read_csv(output_csv), comparison)

    test_predictions = pd.read_parquet(run_dir / "predictions/test_predictions.parquet")
    required_prediction_columns = {
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
    assert required_prediction_columns.issubset(test_predictions.columns)
    assert set(test_predictions["model"]) == {"LinearRegression", "Ridge"}
    assert set(test_predictions["split"]) == {"test"}

    residual_summary = pd.read_csv(run_dir / "diagnostics/residual_summary.csv")
    assert {"run_id", "model", "split", "residual_mean", "abs_error_mean"}.issubset(
        residual_summary.columns
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_owner"] == "run_directory"
    assert manifest["paths"]["model_comparison_convenience_copy"] == str(output_csv)


def test_baseline_config_enables_all_state6_models() -> None:
    experiment_config = load_experiment_config("configs/experiment_baseline.yaml")

    assert experiment_config["runtime"] == {
        "mode": "full",
        "sample_n": None,
        "allow_full_run": False,
    }
    assert set(enabled_model_configs(experiment_config)) == {
        "linear_regression",
        "ridge",
        "lasso",
        "hist_gradient_boosting",
        "mlp",
    }


def test_full_runtime_requires_explicit_allowance(tmp_path) -> None:
    dataset_path = tmp_path / "modeling_dataset.parquet"
    split_path = tmp_path / "split_assignments.csv"
    output_csv = tmp_path / "model_comparison.csv"
    card_path = tmp_path / "baseline_experiment.json"

    pd.DataFrame(
        {
            "row_id": range(10),
            "logP47T": [1.0 + i * 0.01 for i in range(10)],
            "feature_num": range(10),
        }
    ).to_parquet(dataset_path, index=False)
    pd.DataFrame(
        {
            "row_id": range(10),
            "split": ["train"] * 7 + ["test"] * 2 + ["validation"],
        }
    ).to_csv(split_path, index=False)

    experiment_config = {
        "experiment": {"id": "guard_test", "random_seed": 42},
        "runtime": {"mode": "full", "sample_n": None, "allow_full_run": False},
        "data": {
            "processed_dataset": str(dataset_path),
            "split_assignments": str(split_path),
        },
        "cv": {"folds": 2, "scoring": "r2"},
        "models": {
            "linear_regression": {"enabled": True, "grid": {"reg__fit_intercept": [True]}},
        },
        "outputs": {"model_comparison": str(output_csv), "experiment_card": str(card_path)},
    }
    feature_contract = {
        "target": {"name": "logP47T", "source": "P47T", "transform": "log10"},
        "forbidden_predictors": {"target": ["P47T", "logP47T"]},
    }

    with pytest.raises(ValueError, match="Full baseline training is guarded"):
        run_experiment(experiment_config, feature_contract)


def test_feature_count_consistency_is_enforced() -> None:
    consistent = pd.DataFrame({"model": ["a", "b"], "feature_count": [3, 3]})
    _validate_feature_count_consistency(consistent)

    inconsistent = pd.DataFrame({"model": ["a", "b"], "feature_count": [3, 4]})
    with pytest.raises(ValueError, match="same feature_count"):
        _validate_feature_count_consistency(inconsistent)
