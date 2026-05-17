"""First-stage dataset builder for the baseline EPH income experiment.

This module implements only the State 3a dataset work: loading configured annual
files, applying the known 2024/2025 harmonization drops, filtering the target
sample, constructing ``logP47T``, creating a stable ``row_id``, and writing
metadata. It intentionally does not implement full legacy feature engineering,
split assignment, or model training.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eph_income.config import PROJECT_ROOT
from eph_income.contracts import (
    assert_no_forbidden_predictors,
    get_forbidden_predictors,
    validate_target_contract,
)

KNOWN_2024_2025_COLUMN_DROPS = (
    "V2_01_M",
    "V2_02_M",
    "V2_03_M",
    "V5_01_M",
    "V5_02_M",
    "V5_03_M",
)
ROW_ID_COLUMN = "row_id"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve repo-relative paths against the project root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def get_metadata_path(processed_dataset_path: str | Path) -> Path:
    """Return the standard metadata path next to the processed dataset."""

    return resolve_project_path(processed_dataset_path).with_name("dataset_metadata.json")


def _configured_input_files(experiment_config: Mapping[str, Any]) -> dict[str, Path]:
    data_config = experiment_config.get("data")
    if not isinstance(data_config, Mapping):
        raise ValueError("Experiment config must define a 'data' mapping.")

    input_files = data_config.get("input_files")
    if not isinstance(input_files, Mapping):
        raise ValueError("Experiment config data section must define an 'input_files' mapping.")

    return {str(year): resolve_project_path(path) for year, path in input_files.items()}


def validate_dataset_inputs(experiment_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate configured input and output paths without building the dataset."""

    input_files = _configured_input_files(experiment_config)
    missing = [str(path) for path in input_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Configured input files do not exist: " + ", ".join(missing))

    data_config = experiment_config.get("data", {})
    processed_dataset = resolve_project_path(data_config.get("processed_dataset"))
    metadata_path = get_metadata_path(processed_dataset)
    return {
        "input_files": {year: str(path) for year, path in input_files.items()},
        "processed_dataset": str(processed_dataset),
        "metadata": str(metadata_path),
    }


def load_configured_annual_files(
    experiment_config: Mapping[str, Any], *, nrows: int | None = None
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Load annual CSV files declared by the experiment config."""

    frames: dict[str, pd.DataFrame] = {}
    row_counts: list[dict[str, Any]] = []
    for year, path in _configured_input_files(experiment_config).items():
        frame = pd.read_csv(path, nrows=nrows)
        frames[year] = frame
        row_counts.append(
            {
                "year": year,
                "file": str(path),
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
            }
        )
    return frames, row_counts


def harmonize_annual_files(
    annual_frames: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Apply known 2024/2025 column drops before concatenation."""

    harmonized: dict[str, pd.DataFrame] = {}
    dropped_columns: dict[str, list[str]] = {}
    for year, frame in annual_frames.items():
        columns_to_drop = []
        if year in {"2024", "2025"}:
            columns_to_drop = [col for col in KNOWN_2024_2025_COLUMN_DROPS if col in frame.columns]
        harmonized[year] = frame.drop(columns=columns_to_drop)
        dropped_columns[year] = columns_to_drop
    return harmonized, dropped_columns


def concatenate_annual_files(annual_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate annual frames in configured year order."""

    frames = []
    for year, frame in annual_frames.items():
        with_year = frame.copy()
        with_year["source_year"] = year
        frames.append(with_year)
    if not frames:
        raise ValueError("No annual files were provided for concatenation.")
    return pd.concat(frames, axis=0, ignore_index=True, sort=False)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError("Required columns are missing from the dataset: " + ", ".join(missing))


def apply_inclusion_criteria(
    frame: pd.DataFrame, feature_contract: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the baseline target-sample inclusion criteria."""

    inclusion = feature_contract.get("inclusion")
    if not isinstance(inclusion, Mapping):
        raise ValueError("Feature contract must define an 'inclusion' mapping.")

    ingreso_value = inclusion.get("INGRESO")
    positive_columns = [str(col) for col in inclusion.get("require_positive", [])]
    not_missing_columns = [str(col) for col in inclusion.get("require_not_missing", [])]
    required = ["INGRESO", *positive_columns, *not_missing_columns]
    _require_columns(frame, required)

    mask = frame["INGRESO"].eq(ingreso_value)
    for column in positive_columns:
        mask &= frame[column].gt(0)
    for column in not_missing_columns:
        mask &= frame[column].notna()

    filtered = frame.loc[mask].copy()
    criteria = {
        "INGRESO": ingreso_value,
        "require_positive": positive_columns,
        "require_not_missing": not_missing_columns,
    }
    return filtered, criteria


def construct_target(frame: pd.DataFrame, feature_contract: Mapping[str, Any]) -> pd.DataFrame:
    """Construct ``logP47T`` according to the validated target contract."""

    target = validate_target_contract(feature_contract)
    source = target["source"]
    name = target["name"]
    _require_columns(frame, [source])

    result = frame.copy()
    if (result[source] <= 0).any():
        raise ValueError(f"Cannot construct {name}: {source} contains non-positive values.")
    result[name] = np.log10(result[source])
    return result


def add_stable_row_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a deterministic row ID based on post-filter row order."""

    result = frame.reset_index(drop=True).copy()
    result.insert(0, ROW_ID_COLUMN, np.arange(len(result), dtype="int64"))
    return result


def select_modeling_columns(
    frame: pd.DataFrame, feature_contract: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[str], bool]:
    """Remove forbidden predictors from feature columns while retaining the target."""

    target = validate_target_contract(feature_contract)
    target_name = target["name"]
    forbidden = get_forbidden_predictors(feature_contract)
    protected = {ROW_ID_COLUMN, target_name}
    excluded_forbidden = sorted((set(frame.columns) & forbidden) - {target_name})
    feature_columns = [
        column for column in frame.columns if column not in protected and column not in forbidden
    ]
    assert_no_forbidden_predictors(feature_columns, forbidden)
    output_columns = [ROW_ID_COLUMN, target_name, *feature_columns]
    return frame.loc[:, output_columns].copy(), excluded_forbidden, True


def build_modeling_dataset(
    experiment_config: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    *,
    output_dataset_path: str | Path | None = None,
    output_metadata_path: str | Path | None = None,
    write: bool = True,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build and optionally write the State 3a target-filtered modeling dataset."""

    target = validate_target_contract(feature_contract)
    annual_frames, row_counts_by_file = load_configured_annual_files(experiment_config, nrows=nrows)
    harmonized, harmonization_drops = harmonize_annual_files(annual_frames)
    concatenated = concatenate_annual_files(harmonized)
    row_count_before_filters = int(len(concatenated))
    filtered, inclusion_criteria = apply_inclusion_criteria(concatenated, feature_contract)
    with_target = construct_target(filtered, feature_contract)
    with_row_id = add_stable_row_id(with_target)
    modeling_dataset, excluded_forbidden, forbidden_check_passed = select_modeling_columns(
        with_row_id, feature_contract
    )

    data_config = experiment_config.get("data", {})
    default_output = data_config.get("processed_dataset")
    if output_dataset_path is None:
        output_dataset_path = default_output
    if output_metadata_path is None:
        output_metadata_path = get_metadata_path(output_dataset_path)

    dataset_path = resolve_project_path(output_dataset_path)
    metadata_path = resolve_project_path(output_metadata_path)
    metadata: dict[str, Any] = {
        "input_files": {year: str(path) for year, path in _configured_input_files(experiment_config).items()},
        "row_counts_by_file": row_counts_by_file,
        "row_count_before_filters": row_count_before_filters,
        "row_count_after_filters": int(len(modeling_dataset)),
        "target_definition": target,
        "inclusion_criteria_applied": inclusion_criteria,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "harmonization": {
            "known_2024_2025_column_drops": list(KNOWN_2024_2025_COLUMN_DROPS),
            "dropped_columns_by_year": harmonization_drops,
        },
        "excluded_forbidden_predictors": excluded_forbidden,
        "forbidden_predictor_check_passed": forbidden_check_passed,
        "processed_dataset": str(dataset_path),
        "metadata": str(metadata_path),
    }

    if write:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        modeling_dataset.to_parquet(dataset_path, index=False)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return modeling_dataset, metadata
