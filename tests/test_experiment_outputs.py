from __future__ import annotations

import json

import pandas as pd

from eph_income.experiments import run_experiment


def test_debug_experiment_writes_expected_outputs(tmp_path) -> None:
    dataset_path = tmp_path / "modeling_dataset.parquet"
    split_path = tmp_path / "split_assignments.csv"
    output_csv = tmp_path / "model_comparison_debug.csv"
    card_path = tmp_path / "debug_experiment.json"

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
        "outputs": {"model_comparison": str(output_csv), "experiment_card": str(card_path)},
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
