#!/usr/bin/env python
"""Compare saved run-local model-comparison artifacts without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUTPUT_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-id",
        required=True,
        help="Identifier used for reports/comparisons/<comparison_id> outputs.",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more reports/runs/<run_id>/ directories to compare.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def _optional_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _feature_view_summary(feature_view: Any) -> str:
    if not isinstance(feature_view, Mapping) or not feature_view:
        return ""

    parts: list[str] = []
    name = feature_view.get("name")
    if name:
        parts.append(f"name={name}")
    drop_columns = feature_view.get("drop_columns")
    if isinstance(drop_columns, list) and drop_columns:
        parts.append("drop=" + ",".join(str(column) for column in drop_columns))
    permuted_columns = feature_view.get("permuted_columns")
    if isinstance(permuted_columns, list) and permuted_columns:
        parts.append("permuted=" + ",".join(str(column) for column in permuted_columns))
    if parts:
        return "; ".join(parts)
    return json.dumps(dict(feature_view), sort_keys=True)


def _fixed_effects_summary(fixed_effects: Any) -> str:
    if not isinstance(fixed_effects, list) or not fixed_effects:
        return ""

    summaries = []
    for spec in fixed_effects:
        if not isinstance(spec, Mapping):
            continue
        name = spec.get("fe_name") or spec.get("name") or "fixed_effect"
        columns = spec.get("source_columns") or spec.get("columns") or []
        if isinstance(columns, list):
            column_text = "+".join(str(column) for column in columns)
        else:
            column_text = str(columns)
        generated = spec.get("generated_column")
        n_levels = spec.get("n_levels")
        max_levels = spec.get("max_levels")
        drop_first = spec.get("drop_first")
        detail_parts = [f"columns={column_text}"]
        if generated:
            detail_parts.append(f"generated={generated}")
        if n_levels is not None:
            detail_parts.append(f"n_levels={n_levels}")
        if max_levels is not None:
            detail_parts.append(f"max_levels={max_levels}")
        if drop_first is not None:
            detail_parts.append(f"drop_first={drop_first}")
        summaries.append(f"{name}(" + ", ".join(detail_parts) + ")")
    return "; ".join(summaries)


def _metadata_value(
    key: str, manifest: Mapping[str, Any], card: Mapping[str, Any], default: Any = None
) -> Any:
    manifest_value = manifest.get(key)
    if manifest_value not in (None, {}, []):
        return manifest_value
    card_value = card.get(key)
    if card_value not in (None, {}, []):
        return card_value
    return default


def _run_rows(comparison_id: str, run_dir: Path) -> list[dict[str, Any]]:
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics" / "model_comparison.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest does not exist: {manifest_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Model comparison does not exist: {metrics_path}")

    manifest = _read_json(manifest_path)
    card = _optional_json(run_dir / "experiment_card.json")
    model_comparison = pd.read_csv(metrics_path)

    feature_view = _metadata_value("feature_view", manifest, card, {})
    fixed_effects = _metadata_value("fixed_effects_used", manifest, card, [])
    run_id = _metadata_value("run_id", manifest, card, run_dir.name)
    experiment_id = _metadata_value("experiment_id", manifest, card)
    mode = _metadata_value("mode", manifest, card)
    rows_used = _metadata_value("rows_used", manifest, card)
    canonical_run_dir = manifest.get("run_dir") or card.get("canonical_run_dir") or str(run_dir)

    rows = []
    for _, model_row in model_comparison.iterrows():
        rows.append(
            {
                "comparison_id": comparison_id,
                "run_id": run_id,
                "experiment_id": experiment_id,
                "mode": mode,
                "model": model_row.get("model"),
                "feature_count": model_row.get(
                    "feature_count", _metadata_value("feature_count", manifest, card)
                ),
                "rows_used": rows_used,
                "cv_r2_mean": model_row.get("cv_r2_mean"),
                "cv_r2_std": model_row.get("cv_r2_std"),
                "test_r2": model_row.get("test_r2"),
                "validation_r2": model_row.get("validation_r2"),
                "test_mae": model_row.get("test_mae"),
                "validation_mae": model_row.get("validation_mae"),
                "fit_time_seconds": model_row.get("fit_time_seconds"),
                "feature_view_summary": _feature_view_summary(feature_view),
                "fixed_effects_summary": _fixed_effects_summary(fixed_effects),
                "canonical_run_dir": canonical_run_dir,
            }
        )
    return rows


def build_comparison(comparison_id: str, run_dirs: Sequence[str | Path]) -> pd.DataFrame:
    """Build the run-comparison table from saved run directories."""

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.extend(_run_rows(comparison_id, Path(run_dir)))
    comparison = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return comparison


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = OUTPUT_COLUMNS
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _best_note(frame: pd.DataFrame, metric: str, *, highest: bool, label: str) -> str:
    values = pd.to_numeric(frame[metric], errors="coerce")
    if values.isna().all():
        return f"- {label}: unavailable."
    index = values.idxmax() if highest else values.idxmin()
    row = frame.loc[index]
    return (
        f"- {label}: {values.loc[index]:.6g} "
        f"({row['run_id']} / {row['model']})."
    )


def write_markdown(comparison: pd.DataFrame, comparison_id: str, path: Path) -> None:
    """Write a compact Markdown summary sorted by validation R2 descending."""

    sorted_comparison = comparison.assign(
        __validation_r2_sort=pd.to_numeric(comparison["validation_r2"], errors="coerce")
    ).sort_values("__validation_r2_sort", ascending=False, na_position="last")
    sorted_comparison = sorted_comparison.drop(columns=["__validation_r2_sort"])

    notes = [
        _best_note(comparison, "validation_r2", highest=True, label="Best validation R2"),
        _best_note(comparison, "cv_r2_mean", highest=True, label="Best CV R2"),
        _best_note(comparison, "validation_mae", highest=False, label="Lowest validation MAE"),
    ]
    feature_counts = comparison["feature_count"].dropna().unique()
    if len(feature_counts) > 1:
        notes.append(
            "- Warning: feature_count differs across compared runs "
            f"({', '.join(str(value) for value in sorted(feature_counts, key=str))})."
        )

    markdown = "\n\n".join(
        [
            f"# Run comparison: {comparison_id}",
            _markdown_table(sorted_comparison),
            "## Notes\n" + "\n".join(notes),
        ]
    )
    path.write_text(markdown + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path("reports") / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison(args.comparison_id, args.run_dirs)
    csv_path = output_dir / f"{args.comparison_id}.csv"
    markdown_path = output_dir / f"{args.comparison_id}.md"
    comparison.to_csv(csv_path, index=False)
    write_markdown(comparison, args.comparison_id, markdown_path)
    print(json.dumps({"csv": str(csv_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
