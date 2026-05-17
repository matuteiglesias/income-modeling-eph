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


def test_regularization_sweep_writes_path_diagnostics_and_plots(tmp_path) -> None:
    dataset_path = tmp_path / "modeling_dataset.parquet"
    split_path = tmp_path / "split_assignments.csv"
    runs_dir = tmp_path / "runs"

    dataset = pd.DataFrame(
        {
            "row_id": range(45),
            "logP47T": [1.0 + i * 0.02 for i in range(45)],
            "ANO4": [2022, 2023, 2024] * 15,
            "TRIMESTRE": [1, 2, 3, 4, 1] * 9,
            "feature_num": [float(i % 11) for i in range(45)],
            "feature_cat": ["a", "b", "c"] * 15,
        }
    )
    dataset.to_parquet(dataset_path, index=False)
    pd.DataFrame(
        {
            "row_id": range(45),
            "split": ["train"] * 30 + ["test"] * 10 + ["validation"] * 5,
        }
    ).to_csv(split_path, index=False)

    experiment_config = {
        "experiment": {"id": "regularization_sweep_test", "random_seed": 42},
        "runtime": {"mode": "debug", "sample_n": None},
        "data": {
            "processed_dataset": str(dataset_path),
            "split_assignments": str(split_path),
        },
        "cv": {"folds": 2, "scoring": "r2", "return_train_score": True},
        "diagnostics": {"regularization_sweep": True},
        "models": {
            "linear_regression": {"enabled": True, "grid": {"reg__fit_intercept": [True]}},
            "ridge": {
                "enabled": True,
                "grid": {"reg__alpha": [0.1, 1.0, 10.0], "reg__fit_intercept": [True]},
            },
            "lasso": {
                "enabled": True,
                "grid": {
                    "reg__alpha": [0.001, 0.01, 0.1],
                    "reg__fit_intercept": [True],
                    "reg__max_iter": [10000],
                },
            },
        },
        "outputs": {"runs_dir": str(runs_dir)},
    }
    feature_contract = {
        "target": {"name": "logP47T", "source": "P47T", "transform": "log10"},
        "forbidden_predictors": {"target": ["P47T", "logP47T"], "identifiers": ["CODUSU"]},
    }

    comparison, card = run_experiment(experiment_config, feature_contract)
    run_dir = Path(card["canonical_run_dir"])

    assert comparison["model"].tolist() == ["LinearRegression", "Ridge", "Lasso"]
    assert (run_dir / "cv_results" / "Ridge.csv").exists()
    assert (run_dir / "cv_results" / "Lasso.csv").exists()
    assert "mean_train_score" in pd.read_csv(run_dir / "cv_results" / "Ridge.csv").columns

    ridge_summary = pd.read_csv(run_dir / "diagnostics" / "Ridge_regularization_path_summary.csv")
    lasso_summary = pd.read_csv(run_dir / "diagnostics" / "Lasso_regularization_path_summary.csv")
    expected_columns = {
        "mean_train_r2",
        "mean_cv_r2",
        "std_cv_r2",
        "mean_fit_time",
        "coefficient_l1_norm",
        "coefficient_l2_norm",
        "nonzero_coefficients",
    }
    assert expected_columns.issubset(ridge_summary.columns)
    assert expected_columns.issubset(lasso_summary.columns)
    assert ridge_summary["alpha"].tolist() == [0.1, 1.0, 10.0]
    assert lasso_summary["alpha"].tolist() == [0.001, 0.01, 0.1]

    assert (run_dir / "diagnostics" / "LinearRegression_best_coefficients.csv").exists()
    assert (run_dir / "diagnostics" / "Ridge_best_coefficients.csv").exists()
    assert (run_dir / "diagnostics" / "Lasso_best_coefficients.csv").exists()

    plot_names = {path.name for path in (run_dir / "plots" / "regularization").glob("*.png")}
    assert {
        "ridge_cv_r2_vs_alpha.png",
        "ridge_train_vs_cv_r2_alpha.png",
        "ridge_coef_l2_norm_vs_alpha.png",
        "lasso_cv_r2_vs_alpha.png",
        "lasso_train_vs_cv_r2_alpha.png",
        "lasso_nonzero_coefficients_vs_alpha.png",
        "lasso_coef_l1_norm_vs_alpha.png",
    }.issubset(plot_names)

    assert not (tmp_path / "model_comparison.csv").exists()


def test_hgb_debug_config_declares_reference_and_hgb_grid() -> None:
    experiment_config = load_experiment_config("configs/experiment_hgb_debug.yaml")

    assert experiment_config["experiment"]["id"] == "hgb_debug"
    assert experiment_config["runtime"]["sample_n"] == 10000
    assert experiment_config["cv"] == {"folds": 2, "scoring": "r2", "return_train_score": True}
    assert set(enabled_model_configs(experiment_config)) == {"ridge", "hist_gradient_boosting"}
    hgb_grid = experiment_config["models"]["hist_gradient_boosting"]["grid"]
    assert hgb_grid == {
        "reg__learning_rate": [0.1],
        "reg__max_iter": [100],
        "reg__max_leaf_nodes": [31, 127],
        "reg__min_samples_leaf": [20, 100],
        "reg__l2_regularization": [0.0],
    }


def test_hgb_sweep_config_disables_early_stopping_for_interpretable_max_iter() -> None:
    experiment_config = load_experiment_config("configs/experiment_hgb_sweep.yaml")

    assert experiment_config["experiment"]["id"] == "hgb_sweep_v1"
    assert experiment_config["runtime"]["mode"] == "sweep"
    assert experiment_config["cv"] == {"folds": 3, "scoring": "r2", "return_train_score": True}
    assert set(enabled_model_configs(experiment_config)) == {"hist_gradient_boosting"}
    hgb_grid = experiment_config["models"]["hist_gradient_boosting"]["grid"]
    assert hgb_grid["reg__learning_rate"] == [0.03, 0.05, 0.075, 0.1]
    assert hgb_grid["reg__max_iter"] == [100, 200, 400]
    assert hgb_grid["reg__max_leaf_nodes"] == [15, 31, 63]
    assert hgb_grid["reg__min_samples_leaf"] == [50, 100, 200]
    assert hgb_grid["reg__l2_regularization"] == [0.0, 0.01]
    assert hgb_grid["reg__early_stopping"] == [False]
    assert "exact number of boosting iterations" in experiment_config["scientific_notes"]["early_stopping_decision"]


def test_hgb_experiment_writes_hgb_specific_diagnostics_and_plots(tmp_path) -> None:
    dataset_path = tmp_path / "modeling_dataset.parquet"
    split_path = tmp_path / "split_assignments.csv"
    runs_dir = tmp_path / "runs"

    dataset = pd.DataFrame(
        {
            "row_id": range(60),
            "logP47T": [1.0 + (i % 15) * 0.03 + (i // 15) * 0.01 for i in range(60)],
            "ANO4": [2022, 2023, 2024, 2025] * 15,
            "TRIMESTRE": [1, 2, 3, 4, 1] * 12,
            "feature_num": [float(i % 13) for i in range(60)],
            "feature_cat": ["a", "b", "c", "d"] * 15,
        }
    )
    dataset.to_parquet(dataset_path, index=False)
    pd.DataFrame(
        {
            "row_id": range(60),
            "split": ["train"] * 42 + ["test"] * 12 + ["validation"] * 6,
        }
    ).to_csv(split_path, index=False)

    experiment_config = {
        "experiment": {"id": "hgb_diagnostics_test", "random_seed": 42},
        "runtime": {"mode": "debug", "sample_n": None, "enabled_models": ["hist_gradient_boosting"]},
        "data": {
            "processed_dataset": str(dataset_path),
            "split_assignments": str(split_path),
        },
        "cv": {"folds": 2, "scoring": "r2", "return_train_score": True},
        "models": {
            "hist_gradient_boosting": {
                "enabled": True,
                "grid": {
                    "reg__learning_rate": [0.1],
                    "reg__max_iter": [5],
                    "reg__max_leaf_nodes": [3, 5],
                    "reg__min_samples_leaf": [5],
                    "reg__l2_regularization": [0.0],
                    "reg__early_stopping": [False],
                },
            },
        },
        "outputs": {"runs_dir": str(runs_dir)},
    }
    feature_contract = {
        "target": {"name": "logP47T", "source": "P47T", "transform": "log10"},
        "forbidden_predictors": {"target": ["P47T", "logP47T"], "identifiers": ["CODUSU"]},
    }

    _, card = run_experiment(experiment_config, feature_contract)

    run_dir = Path(card["canonical_run_dir"])
    summary = pd.read_csv(run_dir / "diagnostics" / "hgb_cv_results_summary.csv")
    assert {
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "max_depth",
        "min_samples_leaf",
        "l2_regularization",
        "mean_train_score",
        "mean_test_score",
        "std_test_score",
        "overfit_gap",
        "mean_fit_time",
        "rank_test_score",
    }.issubset(summary.columns)
    assert len(summary) == 2
    assert (run_dir / "diagnostics" / "hgb_top_configs.csv").exists()
    assert (run_dir / "diagnostics" / "hgb_overfit_gap_by_config.csv").exists()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert "hgb_cv_results_summary" in manifest["paths"]
    assert "hgb_train_vs_cv_r2_top_configs" in manifest["paths"]

    expected_plots = {
        "hgb_learning_rate_vs_max_iter.png",
        "hgb_max_leaf_nodes_vs_cv_r2.png",
        "hgb_min_samples_leaf_vs_cv_r2.png",
        "hgb_l2_regularization_vs_cv_r2.png",
        "hgb_train_vs_cv_r2_top_configs.png",
    }
    assert expected_plots.issubset({path.name for path in (run_dir / "plots" / "hgb").glob("*.png")})



def test_targeted_hgb_sweep_configs_declare_single_question_grids() -> None:
    lr_iter = load_experiment_config("configs/experiment_hgb_lr_iter_sweep.yaml")
    assert lr_iter["experiment"]["id"] == "hgb_lr_iter_sweep_v1"
    assert lr_iter["runtime"]["sample_n"] == 100000
    assert lr_iter["cv"] == {"folds": 3, "scoring": "r2", "return_train_score": True}
    assert lr_iter["diagnostics"] == {
        "sweep_type": "hgb_lr_iter",
        "primary_param": "max_iter",
        "group_param": "learning_rate",
    }
    assert set(enabled_model_configs(lr_iter)) == {"hist_gradient_boosting"}
    lr_grid = lr_iter["models"]["hist_gradient_boosting"]["grid"]
    assert lr_grid["reg__learning_rate"] == [0.02, 0.03, 0.05, 0.075, 0.1]
    assert lr_grid["reg__max_iter"] == [100, 200, 400, 600]
    assert lr_grid["reg__max_leaf_nodes"] == [31]
    assert lr_grid["reg__min_samples_leaf"] == [100]
    assert lr_grid["reg__l2_regularization"] == [0.0]
    assert lr_grid["reg__early_stopping"] == [False]

    leaf = load_experiment_config("configs/experiment_hgb_leaf_capacity_sweep.yaml")
    leaf_grid = leaf["models"]["hist_gradient_boosting"]["grid"]
    assert leaf["experiment"]["id"] == "hgb_leaf_capacity_sweep_v1"
    assert leaf["diagnostics"] == {"sweep_type": "hgb_single_param", "primary_param": "max_leaf_nodes"}
    assert leaf_grid["reg__max_leaf_nodes"] == [7, 15, 23, 31, 47, 63, 95, 127]
    assert leaf_grid["reg__learning_rate"] == [0.05]
    assert leaf_grid["reg__max_iter"] == [200]

    min_leaf = load_experiment_config("configs/experiment_hgb_min_leaf_sweep.yaml")
    min_leaf_grid = min_leaf["models"]["hist_gradient_boosting"]["grid"]
    assert min_leaf["experiment"]["id"] == "hgb_min_leaf_sweep_v1"
    assert min_leaf["diagnostics"] == {"sweep_type": "hgb_single_param", "primary_param": "min_samples_leaf"}
    assert min_leaf_grid["reg__min_samples_leaf"] == [20, 30, 50, 75, 100, 150, 200, 300]

    l2 = load_experiment_config("configs/experiment_hgb_l2_sweep.yaml")
    l2_grid = l2["models"]["hist_gradient_boosting"]["grid"]
    assert l2["experiment"]["id"] == "hgb_l2_sweep_v1"
    assert l2["diagnostics"] == {"sweep_type": "hgb_single_param", "primary_param": "l2_regularization"}
    assert l2_grid["reg__l2_regularization"] == [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
