from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from eph_income.diagnostics import build_diagnostics


def _prediction_rows(run_id: str, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    split_offset = 0 if split == "test" else 100
    for model, error in [("LinearRegression", 0.20), ("Ridge", 0.05)]:
        for row_id in range(20):
            y_true = 1.0 + row_id * 0.05
            signed_error = error if row_id % 2 == 0 else -error
            y_pred = y_true - signed_error
            residual = y_true - y_pred
            rows.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "split": split,
                    "row_id": split_offset + row_id,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "residual": residual,
                    "abs_error": abs(residual),
                    "squared_error": residual**2,
                }
            )
    return rows


def _write_archived_run(tmp_path: Path) -> Path:
    run_id = "synthetic_run"
    run_dir = tmp_path / "reports" / "runs" / run_id
    for child in ["predictions", "metrics", "cv_results", "diagnostics", "artifacts"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": run_id, "canonical_owner": "run_directory"}), encoding="utf-8"
    )
    pd.DataFrame(_prediction_rows(run_id, "test")).to_parquet(
        run_dir / "predictions" / "test_predictions.parquet", index=False
    )
    pd.DataFrame(_prediction_rows(run_id, "validation")).to_parquet(
        run_dir / "predictions" / "validation_predictions.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "model": "LinearRegression",
                "test_r2": 0.70,
                "test_mae": 0.20,
                "test_mse": 0.04,
                "validation_r2": 0.68,
                "validation_mae": 0.22,
                "validation_mse": 0.05,
            },
            {
                "model": "Ridge",
                "test_r2": 0.90,
                "test_mae": 0.05,
                "test_mse": 0.0025,
                "validation_r2": 0.91,
                "validation_mae": 0.04,
                "validation_mse": 0.0020,
            },
        ]
    ).to_csv(run_dir / "metrics" / "model_comparison.csv", index=False)
    pd.DataFrame(
        {
            "param_reg__alpha": [1.0, 2.0, 5.0],
            "mean_test_score": [0.80, 0.85, 0.83],
            "std_test_score": [0.01, 0.02, 0.01],
        }
    ).to_csv(run_dir / "cv_results" / "Ridge.csv", index=False)
    return run_dir


def test_build_diagnostics_from_archived_run(tmp_path: Path) -> None:
    run_dir = _write_archived_run(tmp_path)
    original_test_predictions = pd.read_parquet(run_dir / "predictions" / "test_predictions.parquet")

    result = build_diagnostics(run_dir, split="test")

    residual_summary = pd.read_csv(run_dir / "diagnostics" / "residual_summary.csv")
    expected_residual_columns = {
        "run_id",
        "model",
        "split",
        "n",
        "residual_mean",
        "residual_std",
        "residual_median",
        "residual_p05",
        "residual_p25",
        "residual_p75",
        "residual_p95",
        "mae",
        "mse",
        "rmse",
        "r2",
        "y_true_mean",
        "y_true_std",
        "y_pred_mean",
        "y_pred_std",
    }
    assert expected_residual_columns.issubset(residual_summary.columns)
    assert set(residual_summary["split"]) == {"test", "validation"}

    error_by_decile = pd.read_csv(run_dir / "diagnostics" / "error_by_income_decile.csv")
    assert {"income_decile", "mae", "rmse", "residual_median"}.issubset(
        error_by_decile.columns
    )

    pairwise = pd.read_csv(run_dir / "diagnostics" / "model_pairwise_error_comparison.csv")
    assert pairwise.loc[0, "model_a"] == "LinearRegression"
    assert pairwise.loc[0, "model_b"] == "Ridge"
    assert pairwise.loc[0, "share_rows_b_better"] == 1.0

    metric_gaps = pd.read_csv(run_dir / "diagnostics" / "metric_gaps.csv")
    assert set(metric_gaps["metric"]) == {"mae", "mse", "r2"}

    expected_plots = {
        "prediction_distribution_by_model.png",
        "observed_vs_predicted_best_model.png",
        "mae_by_income_decile.png",
        "mean_residual_by_income_decile.png",
        "residual_distribution_by_model.png",
        "ridge_alpha_curve.png",
    }
    assert expected_plots.issubset({path.name for path in (run_dir / "plots" / "test").glob("*.png")})
    assert not (run_dir / "plots" / "test" / "lasso_alpha_curve.png").exists()

    notes = (run_dir / "diagnostics" / "diagnostics_build_notes.md").read_text(encoding="utf-8")
    assert "Skipped lasso_alpha_curve.png" in notes
    assert result["split"] == "test"

    after_test_predictions = pd.read_parquet(run_dir / "predictions" / "test_predictions.parquet")
    pd.testing.assert_frame_equal(after_test_predictions, original_test_predictions)
    assert "income_decile" not in after_test_predictions.columns


def test_diagnostics_cli_accepts_run_dir_and_split(tmp_path: Path) -> None:
    run_dir = _write_archived_run(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/04_build_diagnostics.py",
            "--run-dir",
            str(run_dir),
            "--split",
            "validation",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["split"] == "validation"
    assert Path(payload["notes_path"]).exists()
    assert (run_dir / "plots" / "validation" / "observed_vs_predicted_best_model.png").exists()
    assert not (run_dir / "plots" / "test" / "observed_vs_predicted_best_model.png").exists()



def _write_synthetic_hgb_sweep_run(tmp_path: Path, *, sweep_type: str, primary_param: str, group_param: str | None = None) -> Path:
    run_dir = _write_archived_run(tmp_path)
    (run_dir / "plots" / "hgb_sweeps").mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": {"id": "hgb_synthetic_sweep"},
        "diagnostics": {"sweep_type": sweep_type, "primary_param": primary_param},
    }
    if group_param:
        config["diagnostics"]["group_param"] = group_param
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    rows = []
    rank = 1
    learning_rates = [0.03, 0.05] if group_param == "learning_rate" else [0.05]
    primary_values = [100, 200, 400] if primary_param == "max_iter" else [15, 31, 63]
    for learning_rate in learning_rates:
        for value in primary_values:
            score = 0.80 + (0.02 if value == primary_values[1] else 0.01) + (0.005 if learning_rate == 0.05 else 0.0)
            rows.append(
                {
                    "run_id": "synthetic_run",
                    "model": "HistGradientBoostingRegressor",
                    "param_reg__learning_rate": learning_rate,
                    "param_reg__max_iter": value if primary_param == "max_iter" else 200,
                    "param_reg__max_leaf_nodes": value if primary_param == "max_leaf_nodes" else 31,
                    "param_reg__min_samples_leaf": 100,
                    "param_reg__l2_regularization": 0.0,
                    "param_reg__early_stopping": False,
                    "mean_train_score": score + 0.04,
                    "mean_test_score": score,
                    "std_test_score": 0.01,
                    "mean_fit_time": float(value) / 100.0,
                    "rank_test_score": rank,
                }
            )
            rank += 1
    pd.DataFrame(rows).to_csv(run_dir / "cv_results" / "HistGradientBoostingRegressor.csv", index=False)
    return run_dir


def test_build_diagnostics_writes_hgb_lr_iter_sweep_plots_and_summary(tmp_path: Path) -> None:
    run_dir = _write_synthetic_hgb_sweep_run(
        tmp_path, sweep_type="hgb_lr_iter", primary_param="max_iter", group_param="learning_rate"
    )

    result = build_diagnostics(run_dir, split="test")

    expected = {
        "hgb_lr_iter_cv_r2.png",
        "hgb_lr_iter_train_vs_cv.png",
        "hgb_lr_iter_overfit_gap.png",
        "hgb_lr_iter_fit_time.png",
    }
    assert expected.issubset({path.name for path in (run_dir / "plots" / "hgb_sweeps").glob("*.png")})
    summary = (run_dir / "diagnostics" / "hgb_sweep_scientific_summary.md").read_text(encoding="utf-8")
    assert "Experiment id: `hgb_synthetic_sweep`" in summary
    assert "Best configuration by mean CV R²" in summary
    assert "Simplest configuration within 0.002 CV R² of best" in summary
    assert "Recommended parameter region" in summary
    assert "hgb_lr_iter_cv_r2" in result["plots"]


def test_build_diagnostics_writes_hgb_single_param_sweep_plots(tmp_path: Path) -> None:
    run_dir = _write_synthetic_hgb_sweep_run(
        tmp_path, sweep_type="hgb_single_param", primary_param="max_leaf_nodes"
    )

    build_diagnostics(run_dir, split="test")

    expected = {
        "hgb_max_leaf_nodes_cv_r2.png",
        "hgb_max_leaf_nodes_train_vs_cv.png",
        "hgb_max_leaf_nodes_overfit_gap.png",
        "hgb_max_leaf_nodes_fit_time.png",
    }
    assert expected.issubset({path.name for path in (run_dir / "plots" / "hgb_sweeps").glob("*.png")})
    sweep_summary = pd.read_csv(run_dir / "diagnostics" / "hgb_sweep_cv_results_summary.csv")
    assert {"max_leaf_nodes", "mean_test_score", "overfit_gap"}.issubset(sweep_summary.columns)
