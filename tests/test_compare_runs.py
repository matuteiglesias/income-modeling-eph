from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_run(
    base: Path,
    run_id: str,
    *,
    experiment_id: str,
    models: list[dict[str, object]],
    feature_count: int,
    rows_used: int,
    feature_view=None,
    fixed_effects_used=None,
    write_card: bool = True,
) -> Path:
    run_dir = base / run_id
    (run_dir / "metrics").mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "mode": "sweep",
        "feature_count": feature_count,
        "rows_used": rows_used,
        "run_dir": str(run_dir),
    }
    if feature_view is not None:
        manifest["feature_view"] = feature_view
    if fixed_effects_used is not None:
        manifest["fixed_effects_used"] = fixed_effects_used
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if write_card:
        card = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "mode": "sweep",
            "feature_count": feature_count,
            "rows_used": rows_used,
            "feature_view": feature_view or {},
            "fixed_effects_used": fixed_effects_used or [],
            "canonical_run_dir": str(run_dir),
        }
        (run_dir / "experiment_card.json").write_text(
            json.dumps(card, indent=2, sort_keys=True), encoding="utf-8"
        )
    pd.DataFrame(models).to_csv(run_dir / "metrics" / "model_comparison.csv", index=False)
    return run_dir


def test_compare_runs_script_writes_csv_and_markdown_for_multiple_models(tmp_path) -> None:
    runs_dir = tmp_path / "reports" / "runs"
    run_a = _write_run(
        runs_dir,
        "run_a",
        experiment_id="linear_region_fe_v1",
        feature_count=12,
        rows_used=5000,
        feature_view={"name": "linear_region_fe_clean", "drop_columns": ["AGLO_rk", "Reg_rk"]},
        fixed_effects_used=[
            {
                "fe_name": "region_fe",
                "source_columns": ["Region"],
                "generated_column": "fe_region_fe",
                "n_levels": 6,
                "max_levels": 20,
                "drop_first": True,
            }
        ],
        models=[
            {
                "model": "LinearRegression",
                "feature_count": 12,
                "cv_r2_mean": 0.4,
                "cv_r2_std": 0.01,
                "test_r2": 0.39,
                "validation_r2": 0.38,
                "test_mae": 0.2,
                "validation_mae": 0.21,
                "fit_time_seconds": 1.1,
            },
            {
                "model": "Ridge",
                "feature_count": 12,
                "cv_r2_mean": 0.42,
                "cv_r2_std": 0.02,
                "test_r2": 0.4,
                "validation_r2": 0.43,
                "test_mae": 0.19,
                "validation_mae": 0.18,
                "fit_time_seconds": 1.3,
            },
        ],
    )
    run_b = _write_run(
        runs_dir,
        "run_b",
        experiment_id="linear_year_fe_v1",
        feature_count=10,
        rows_used=5000,
        write_card=False,
        models=[
            {
                "model": "Ridge",
                "feature_count": 10,
                "cv_r2_mean": 0.45,
                "cv_r2_std": 0.03,
                "test_r2": 0.41,
                "validation_r2": 0.37,
                "test_mae": 0.22,
                "validation_mae": 0.23,
                "fit_time_seconds": 0.9,
            }
        ],
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "05_compare_runs.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--comparison-id",
            "synthetic_compare_v1",
            "--run-dirs",
            str(run_a),
            str(run_b),
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    csv_path = tmp_path / payload["csv"]
    markdown_path = tmp_path / payload["markdown"]
    assert csv_path == tmp_path / "reports" / "comparisons" / "synthetic_compare_v1.csv"
    assert markdown_path == tmp_path / "reports" / "comparisons" / "synthetic_compare_v1.md"
    assert csv_path.exists()
    assert markdown_path.exists()

    comparison = pd.read_csv(csv_path)
    assert comparison.columns.tolist() == [
        "comparison_id",
        "run_id",
        "experiment_id",
        "mode",
        "model",
        "feature_count",
        "rows_used",
        "cv_r2_mean",
        "cv_r2_std",
        "test_r2",
        "validation_r2",
        "test_mae",
        "validation_mae",
        "fit_time_seconds",
        "feature_view_summary",
        "fixed_effects_summary",
        "canonical_run_dir",
    ]
    assert comparison["comparison_id"].tolist() == ["synthetic_compare_v1"] * 3
    assert comparison["run_id"].tolist() == ["run_a", "run_a", "run_b"]
    assert comparison["model"].tolist() == ["LinearRegression", "Ridge", "Ridge"]
    assert comparison.loc[0, "feature_view_summary"] == "name=linear_region_fe_clean; drop=AGLO_rk,Reg_rk"
    assert "region_fe(columns=Region" in comparison.loc[0, "fixed_effects_summary"]
    assert pd.isna(comparison.loc[2, "feature_view_summary"])
    assert pd.isna(comparison.loc[2, "fixed_effects_summary"])

    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Run comparison: synthetic_compare_v1")
    assert "| comparison_id | run_id | experiment_id |" in markdown
    assert "Best validation R2: 0.43 (run_a / Ridge)." in markdown
    assert "Best CV R2: 0.45 (run_b / Ridge)." in markdown
    assert "Lowest validation MAE: 0.18 (run_a / Ridge)." in markdown
    assert "Warning: feature_count differs across compared runs (10, 12)." in markdown
