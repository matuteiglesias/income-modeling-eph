"""Stable train/test/validation split registry."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eph_income.dataset import ROW_ID_COLUMN, resolve_project_path

VALID_SPLITS = {"train", "test", "validation"}


def get_split_path(experiment_config: Mapping[str, Any]) -> Path:
    data_config = experiment_config.get("data")
    if not isinstance(data_config, Mapping):
        raise ValueError("Experiment config must define a 'data' mapping.")
    return resolve_project_path(data_config.get("split_assignments"))


def create_split_assignments(dataset: pd.DataFrame, split_config: Mapping[str, Any]) -> pd.DataFrame:
    """Create random individual split assignments for every row_id."""

    if ROW_ID_COLUMN not in dataset.columns:
        raise KeyError(f"Dataset must contain {ROW_ID_COLUMN!r} before splitting.")
    row_ids = dataset[ROW_ID_COLUMN].to_numpy()
    if len(row_ids) != len(set(row_ids.tolist())):
        raise ValueError("row_id values must be unique before splitting.")

    train_size = float(split_config.get("train_size", 0.70))
    test_size = float(split_config.get("test_size", 0.20))
    validation_size = float(split_config.get("validation_size", 0.10))
    if not np.isclose(train_size + test_size + validation_size, 1.0):
        raise ValueError("Split sizes must sum to 1.0.")
    random_state = int(split_config.get("random_state", 42))

    rng = np.random.default_rng(random_state)
    shuffled = row_ids.copy()
    rng.shuffle(shuffled)

    n_rows = len(shuffled)
    train_end = int(round(n_rows * train_size))
    test_end = train_end + int(round(n_rows * test_size))
    test_end = min(test_end, n_rows)

    assignments = pd.DataFrame({ROW_ID_COLUMN: shuffled})
    assignments["split"] = "validation"
    assignments.loc[: train_end - 1, "split"] = "train"
    assignments.loc[train_end : test_end - 1, "split"] = "test"
    assignments = assignments.sort_values(ROW_ID_COLUMN).reset_index(drop=True)
    return assignments


def validate_split_assignments(dataset: pd.DataFrame, assignments: pd.DataFrame) -> None:
    """Validate that split assignments exactly cover dataset row IDs once."""

    if ROW_ID_COLUMN not in dataset.columns or ROW_ID_COLUMN not in assignments.columns:
        raise KeyError(f"Dataset and assignments must contain {ROW_ID_COLUMN!r}.")
    if "split" not in assignments.columns:
        raise KeyError("Split assignments must contain a 'split' column.")
    if assignments[ROW_ID_COLUMN].duplicated().any():
        raise ValueError("Split assignments contain duplicated row_id values.")

    dataset_ids = set(dataset[ROW_ID_COLUMN].tolist())
    assignment_ids = set(assignments[ROW_ID_COLUMN].tolist())
    if dataset_ids != assignment_ids:
        missing = sorted(dataset_ids - assignment_ids)[:5]
        extra = sorted(assignment_ids - dataset_ids)[:5]
        raise ValueError(
            "Split assignments do not match dataset row_ids; "
            f"missing examples={missing}, extra examples={extra}."
        )
    invalid = sorted(set(assignments["split"].dropna().astype(str)) - VALID_SPLITS)
    if invalid:
        raise ValueError("Invalid split labels: " + ", ".join(invalid))


def load_or_create_split_assignments(
    dataset: pd.DataFrame,
    experiment_config: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Load existing split assignments or create and persist them."""

    split_path = get_split_path(experiment_config)
    if split_path.exists() and not overwrite:
        assignments = pd.read_csv(split_path)
        validate_split_assignments(dataset, assignments)
        return assignments

    split_config = experiment_config.get("split", {})
    if not isinstance(split_config, Mapping):
        raise ValueError("Experiment config 'split' section must be a mapping.")
    assignments = create_split_assignments(dataset, split_config)
    validate_split_assignments(dataset, assignments)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(split_path, index=False)
    return assignments


def split_proportions(assignments: pd.DataFrame) -> dict[str, float]:
    counts = assignments["split"].value_counts(normalize=True).to_dict()
    return {split: float(counts.get(split, 0.0)) for split in ["train", "test", "validation"]}
