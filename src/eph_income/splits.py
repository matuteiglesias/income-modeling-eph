"""Stable train/test/validation split registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eph_income.dataset import ROW_ID_COLUMN, resolve_project_path

VALID_SPLITS = {"train", "test", "validation"}
RANDOM_PERSON_STRATEGY = "random_person_v1"
HOUSEHOLD_GROUPED_STRATEGY = "household_grouped_v1"
DEFAULT_HOUSEHOLD_GROUP_COLUMNS = ("CODUSU", "NRO_HOGAR")


def get_split_path(experiment_config: Mapping[str, Any]) -> Path:
    data_config = experiment_config.get("data")
    if not isinstance(data_config, Mapping):
        raise TypeError("Experiment config must define a 'data' mapping.")
    return resolve_project_path(data_config.get("split_assignments"))


def _split_sizes(split_config: Mapping[str, Any]) -> tuple[float, float, float, int]:
    train_size = float(split_config.get("train_size", 0.70))
    test_size = float(split_config.get("test_size", 0.20))
    validation_size = float(split_config.get("validation_size", 0.10))
    if not np.isclose(train_size + test_size + validation_size, 1.0):
        raise ValueError("Split sizes must sum to 1.0.")
    if min(train_size, test_size, validation_size) < 0:
        raise ValueError("Split sizes must be non-negative.")
    random_state = int(split_config.get("random_state", 42))
    return train_size, test_size, validation_size, random_state


def _validate_row_ids(dataset: pd.DataFrame) -> np.ndarray:
    if ROW_ID_COLUMN not in dataset.columns:
        raise KeyError(f"Dataset must contain {ROW_ID_COLUMN!r} before splitting.")
    row_ids = dataset[ROW_ID_COLUMN].to_numpy()
    if len(row_ids) != len(set(row_ids.tolist())):
        raise ValueError("row_id values must be unique before splitting.")
    return row_ids


def _random_person_assignments(
    dataset: pd.DataFrame, split_config: Mapping[str, Any]
) -> pd.DataFrame:
    row_ids = _validate_row_ids(dataset)
    train_size, test_size, _, random_state = _split_sizes(split_config)

    rng = np.random.default_rng(random_state)
    shuffled = row_ids.copy()
    rng.shuffle(shuffled)

    n_rows = len(shuffled)
    train_end = round(n_rows * train_size)
    test_end = min(train_end + round(n_rows * test_size), n_rows)

    assignments = pd.DataFrame({ROW_ID_COLUMN: shuffled})
    assignments["split"] = "validation"
    assignments.loc[: train_end - 1, "split"] = "train"
    assignments.loc[train_end : test_end - 1, "split"] = "test"
    return assignments.sort_values(ROW_ID_COLUMN).reset_index(drop=True)


def _group_columns(split_config: Mapping[str, Any]) -> tuple[str, ...]:
    configured = split_config.get("group_columns", DEFAULT_HOUSEHOLD_GROUP_COLUMNS)
    if isinstance(configured, str):
        configured = [configured]
    if not isinstance(configured, Sequence) or not configured:
        raise ValueError("Household-grouped splitting requires non-empty group_columns.")
    return tuple(str(column) for column in configured)


def _household_grouped_assignments(
    dataset: pd.DataFrame, split_config: Mapping[str, Any]
) -> pd.DataFrame:
    _validate_row_ids(dataset)
    train_size, test_size, _, random_state = _split_sizes(split_config)
    group_columns = _group_columns(split_config)
    missing = [column for column in group_columns if column not in dataset.columns]
    if missing:
        raise KeyError(
            "Household-grouped splitting requires columns: " + ", ".join(missing)
        )
    if dataset.loc[:, group_columns].isna().any().any():
        raise ValueError("Household group columns cannot contain missing values.")

    work = dataset.loc[:, [ROW_ID_COLUMN, *group_columns]].copy()
    work["_group_key"] = list(map(tuple, work.loc[:, group_columns].astype(str).to_numpy()))
    group_sizes = work.groupby("_group_key", sort=False).size()
    groups = list(group_sizes.index)
    if len(groups) < 3:
        raise ValueError("Household-grouped splitting requires at least three groups.")

    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(groups))
    shuffled_groups = [groups[index] for index in order]
    sizes = np.array([int(group_sizes[group]) for group in shuffled_groups], dtype=int)
    cumulative = np.cumsum(sizes)
    total = int(cumulative[-1])

    # Pick group boundaries nearest the requested row-mass targets. This keeps
    # households intact while usually staying closer to 70/20/10 than cutting
    # by group count.
    train_target = total * train_size
    test_target = total * (train_size + test_size)
    train_cut = int(np.argmin(np.abs(cumulative - train_target))) + 1
    train_cut = min(max(train_cut, 1), len(groups) - 2)
    candidate = cumulative[train_cut:]
    test_cut = train_cut + int(np.argmin(np.abs(candidate - test_target))) + 1
    test_cut = min(max(test_cut, train_cut + 1), len(groups) - 1)

    labels: dict[tuple[str, ...], str] = {}
    for group in shuffled_groups[:train_cut]:
        labels[group] = "train"
    for group in shuffled_groups[train_cut:test_cut]:
        labels[group] = "test"
    for group in shuffled_groups[test_cut:]:
        labels[group] = "validation"

    assignments = work.loc[:, [ROW_ID_COLUMN]].copy()
    assignments["split"] = work["_group_key"].map(labels)
    return assignments.sort_values(ROW_ID_COLUMN).reset_index(drop=True)


def create_split_assignments(dataset: pd.DataFrame, split_config: Mapping[str, Any]) -> pd.DataFrame:
    """Create deterministic split assignments according to the named strategy."""
    strategy = str(split_config.get("strategy", RANDOM_PERSON_STRATEGY))
    if strategy == RANDOM_PERSON_STRATEGY:
        return _random_person_assignments(dataset, split_config)
    if strategy == HOUSEHOLD_GROUPED_STRATEGY:
        return _household_grouped_assignments(dataset, split_config)
    raise ValueError(f"Unsupported split strategy: {strategy}")


def validate_split_assignments(
    dataset: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    group_columns: Sequence[str] | None = None,
) -> None:
    """Validate row coverage and, when requested, group isolation."""
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

    if group_columns:
        group_columns = tuple(str(column) for column in group_columns)
        missing = [column for column in group_columns if column not in dataset.columns]
        if missing:
            raise KeyError(
                "Cannot validate grouped split without columns: " + ", ".join(missing)
            )
        joined = dataset.loc[:, [ROW_ID_COLUMN, *group_columns]].merge(
            assignments.loc[:, [ROW_ID_COLUMN, "split"]],
            on=ROW_ID_COLUMN,
            validate="one_to_one",
        )
        split_counts = joined.groupby(list(group_columns), dropna=False)["split"].nunique()
        if (split_counts > 1).any():
            raise ValueError(
                "Grouped split integrity failure: at least one household crosses splits."
            )


def load_or_create_split_assignments(
    dataset: pd.DataFrame,
    experiment_config: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Load existing split assignments or create and persist them."""
    split_path = get_split_path(experiment_config)
    split_config = experiment_config.get("split", {})
    if not isinstance(split_config, Mapping):
        raise TypeError("Experiment config 'split' section must be a mapping.")
    strategy = str(split_config.get("strategy", RANDOM_PERSON_STRATEGY))
    group_columns = _group_columns(split_config) if strategy == HOUSEHOLD_GROUPED_STRATEGY else None

    if split_path.exists() and not overwrite:
        assignments = pd.read_csv(split_path)
        validate_split_assignments(dataset, assignments, group_columns=group_columns)
        return assignments

    assignments = create_split_assignments(dataset, split_config)
    validate_split_assignments(dataset, assignments, group_columns=group_columns)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(split_path, index=False)
    return assignments


def split_proportions(assignments: pd.DataFrame) -> dict[str, float]:
    counts = assignments["split"].value_counts(normalize=True).to_dict()
    return {
        split: float(counts.get(split, 0.0))
        for split in ["train", "test", "validation"]
    }
