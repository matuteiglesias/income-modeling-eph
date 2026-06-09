"""Experiment-frame construction for EPH income experiments."""

from __future__ import annotations

import re
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from eph_income.contracts import (
    assert_no_forbidden_predictors,
    get_forbidden_predictors,
    validate_target_contract,
)
from eph_income.dataset import ROW_ID_COLUMN, resolve_project_path
from eph_income.splits import get_split_path, validate_split_assignments


def _resolve_output_path(path: str | Path) -> Path:
    """Resolve repo-relative output/input paths against the project root."""
    return resolve_project_path(path)


def _runtime_mapping(experiment_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the runtime config mapping, or an empty mapping."""
    runtime = experiment_config.get("runtime", {})
    return runtime if isinstance(runtime, Mapping) else {}


def _safe_run_component(value: str) -> str:
    """Return a filesystem-safe component for generated experiment columns."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"




def load_processed_dataset(experiment_config: Mapping[str, Any]) -> pd.DataFrame:
    path = _resolve_output_path(experiment_config["data"]["processed_dataset"])
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset does not exist: {path}")
    return pd.read_parquet(path)


def load_split_assignments(
    experiment_config: Mapping[str, Any], dataset: pd.DataFrame
) -> pd.DataFrame:
    path = get_split_path(experiment_config)
    if not path.exists():
        raise FileNotFoundError(f"Split assignments do not exist: {path}")
    assignments = pd.read_csv(path)
    validate_split_assignments(dataset, assignments)
    return assignments


def apply_debug_sample(
    joined: pd.DataFrame, *, sample_n: int | None, random_state: int, min_train_rows: int
) -> pd.DataFrame:
    """Sample rows for debug mode while retaining all split labels."""

    if sample_n is None or sample_n >= len(joined):
        return joined.copy()
    fractions = joined["split"].value_counts(normalize=True).to_dict()
    sampled_parts = []
    remaining = sample_n
    for split in ["train", "test", "validation"]:
        split_frame = joined[joined["split"] == split]
        if split_frame.empty:
            continue
        desired = int(round(sample_n * fractions.get(split, 0.0)))
        if split == "train":
            desired = max(desired, min_train_rows)
        else:
            desired = max(desired, 2)
        desired = min(desired, len(split_frame), remaining if remaining > 0 else len(split_frame))
        sampled_parts.append(split_frame.sample(n=desired, random_state=random_state))
        remaining -= desired
    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) < sample_n:
        missing = sample_n - len(sampled)
        extra_pool = joined[~joined[ROW_ID_COLUMN].isin(sampled[ROW_ID_COLUMN])]
        if not extra_pool.empty:
            sampled = pd.concat(
                [
                    sampled,
                    extra_pool.sample(n=min(missing, len(extra_pool)), random_state=random_state),
                ],
                ignore_index=True,
            )
    return sampled.sort_values(ROW_ID_COLUMN).reset_index(drop=True)


def prepare_experiment_frame(
    experiment_config: Mapping[str, Any], feature_contract: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[str], str]:
    """Load dataset/splits, optionally sample debug rows, and return feature columns."""

    dataset = load_processed_dataset(experiment_config)
    assignments = load_split_assignments(experiment_config, dataset)
    joined = dataset.merge(assignments, on=ROW_ID_COLUMN, how="inner", validate="one_to_one")

    cv_folds = int(experiment_config.get("cv", {}).get("folds", 5))
    runtime = _runtime_mapping(experiment_config)
    sample_n = runtime.get("sample_n")
    if sample_n is not None:
        sample_n = int(sample_n)
    random_seed = int(experiment_config.get("experiment", {}).get("random_seed", 42))
    joined = apply_debug_sample(
        joined, sample_n=sample_n, random_state=random_seed, min_train_rows=max(cv_folds, 2)
    )

    target = validate_target_contract(feature_contract)["name"]
    feature_columns = [
        column for column in dataset.columns if column not in {ROW_ID_COLUMN, target}
    ]
    return joined, feature_columns, target


def _model_design_mapping(experiment_config: Mapping[str, Any]) -> Mapping[str, Any]:
    model_design = experiment_config.get("model_design", {})
    return model_design if isinstance(model_design, Mapping) else {}


def _fixed_effect_specs(experiment_config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    fixed_effects = _model_design_mapping(experiment_config).get("fixed_effects", [])
    if fixed_effects is None:
        return []
    if not isinstance(fixed_effects, list):
        raise TypeError("model_design.fixed_effects must be a list when provided.")
    for spec in fixed_effects:
        if not isinstance(spec, Mapping):
            raise TypeError("Each model_design.fixed_effects entry must be a mapping.")
    return fixed_effects


def _fixed_effect_column_name(fe_name: str) -> str:
    return f"fe_{_safe_run_component(fe_name)}"


def _string_level_values(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].astype("string").fillna("<NA>")
    if len(columns) == 1:
        return values[columns[0]]
    return values[columns[0]].str.cat(values[columns[1]], sep="__")


def apply_fixed_effects(
    *,
    experiment_config: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    joined: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    """Generate configured one-way or two-way fixed-effect categorical columns."""

    fixed_effect_metadata: list[dict[str, Any]] = []
    fixed_effect_specs = _fixed_effect_specs(experiment_config)
    if not fixed_effect_specs:
        return joined, feature_columns, fixed_effect_metadata

    frame = joined.copy()
    final_feature_columns = list(feature_columns)
    for index, spec in enumerate(fixed_effect_specs):
        fe_name = str(spec.get("name") or f"fixed_effect_{index + 1}")
        columns_raw = spec.get("columns")
        if not isinstance(columns_raw, list) or not columns_raw:
            raise ValueError(f"Fixed effect {fe_name} must declare a non-empty columns list.")
        source_columns = [str(column) for column in columns_raw]
        if len(source_columns) > 2:
            raise ValueError(
                f"Fixed effect {fe_name} declares {len(source_columns)} columns; "
                "only one-column and two-column fixed effects are supported."
            )
        missing_columns = [column for column in source_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Fixed effect {fe_name} source columns are missing from the experiment frame: "
                + ", ".join(missing_columns)
            )
        max_levels = int(spec.get("max_levels", 0))
        if max_levels <= 0:
            raise ValueError(f"Fixed effect {fe_name} must declare positive max_levels.")

        generated_column = _fixed_effect_column_name(fe_name)
        if generated_column in frame.columns and generated_column not in final_feature_columns:
            raise ValueError(
                f"Fixed effect {fe_name} generated column {generated_column} already exists "
                "outside the feature list."
            )
        frame[generated_column] = _string_level_values(frame, source_columns)
        n_levels = int(frame[generated_column].nunique(dropna=False))
        if n_levels > max_levels:
            raise ValueError(
                f"Fixed effect {fe_name} has {n_levels} levels, exceeding max_levels={max_levels}."
            )
        if generated_column not in final_feature_columns:
            final_feature_columns.append(generated_column)

        fixed_effect_metadata.append(
            {
                "fe_name": fe_name,
                "source_columns": source_columns,
                "generated_column": generated_column,
                "n_levels": n_levels,
                "max_levels": max_levels,
                "drop_first": bool(spec.get("drop_first", False)),
            }
        )

    forbidden = get_forbidden_predictors(feature_contract)
    assert_no_forbidden_predictors(final_feature_columns, forbidden)
    return frame, final_feature_columns, fixed_effect_metadata


def _fixed_effect_drop_first_columns(fixed_effects_used: list[dict[str, Any]]) -> set[str]:
    return {
        str(spec["generated_column"])
        for spec in fixed_effects_used
        if bool(spec.get("drop_first", False))
    }


def _feature_view_mapping(experiment_config: Mapping[str, Any]) -> Mapping[str, Any]:
    feature_view = experiment_config.get("feature_view", {})
    return feature_view if isinstance(feature_view, Mapping) else {}


def _deterministic_group_value(values: pd.Series) -> Any:
    """Return a deterministic representative value for one group."""

    counts = values.value_counts(dropna=False)
    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()
    return sorted(candidates, key=lambda value: (str(type(value)), str(value)))[0]


def _apply_group_value_permutation(
    frame: pd.DataFrame, spec: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Permute one feature column's group-level values across group categories."""

    column = str(spec.get("column"))
    group_by = str(spec.get("group_by"))
    random_state = int(spec.get("random_state", 42))
    metadata: dict[str, Any] = {
        "column": column,
        "group_by": group_by,
        "random_state": random_state,
        "number_of_groups": 0,
        "group_to_value_mapping_one_to_one": True,
        "warnings": [],
    }

    missing_columns = [name for name in [column, group_by] if name not in frame.columns]
    if missing_columns:
        metadata["warnings"].append(
            f"Skipping permutation because columns are missing: {', '.join(missing_columns)}"
        )
        metadata["missing_columns"] = missing_columns
        return frame, metadata

    group_values = []
    warning_groups = []
    grouped = frame.groupby(group_by, sort=True, dropna=False)[column]
    for group, values in grouped:
        unique_count = int(values.nunique(dropna=False))
        if unique_count > 1:
            warning_groups.append(group)
        group_values.append(
            {
                group_by: group,
                "__feature_view_value": _deterministic_group_value(values),
            }
        )

    value_frame = pd.DataFrame(group_values)
    metadata["number_of_groups"] = int(len(value_frame))
    metadata["group_to_value_mapping_one_to_one"] = not warning_groups
    if warning_groups:
        metadata["warnings"].append(
            f"Column {column} has multiple values within {len(warning_groups)} {group_by} groups; "
            "using a deterministic modal value before permutation."
        )

    if value_frame.empty:
        return frame.copy(), metadata

    permutation = np.random.default_rng(random_state).permutation(len(value_frame))
    if len(value_frame) > 1 and np.array_equal(permutation, np.arange(len(value_frame))):
        permutation = np.roll(permutation, 1)
    value_frame["__feature_view_permuted_value"] = value_frame["__feature_view_value"].to_numpy()[
        permutation
    ]

    permuted = frame.merge(
        value_frame[[group_by, "__feature_view_permuted_value"]],
        on=group_by,
        how="left",
        sort=False,
        validate="many_to_one",
    )
    permuted[column] = permuted["__feature_view_permuted_value"]
    permuted = permuted.drop(columns=["__feature_view_permuted_value"])
    return permuted, metadata


def apply_feature_view(
    *,
    experiment_config: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    joined: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Apply experiment-level feature-view transformations before model fitting.

    Feature views are applied after the processed dataset, split assignments, and runtime
    sample have been established. The processed dataset is not mutated; the returned
    experiment frame and feature list are the final inputs used for training and
    artifacts.
    """

    feature_view = _feature_view_mapping(experiment_config)
    drop_columns = feature_view.get("drop_columns", [])
    if drop_columns is None:
        drop_columns = []
    if not isinstance(drop_columns, list):
        raise TypeError("feature_view.drop_columns must be a list when provided.")

    permutation_specs = feature_view.get("permute_group_values", [])
    if permutation_specs is None:
        permutation_specs = []
    if not isinstance(permutation_specs, list):
        raise TypeError("feature_view.permute_group_values must be a list when provided.")

    viewed = joined.copy()
    permutation_metadata = []
    for spec in permutation_specs:
        if not isinstance(spec, Mapping):
            raise TypeError("Each feature_view.permute_group_values entry must be a mapping.")
        viewed, metadata = _apply_group_value_permutation(viewed, spec)
        permutation_metadata.append(metadata)

    requested_drop_columns = [str(column) for column in drop_columns]
    feature_column_set = set(feature_columns)
    dropped_columns_present = [
        column for column in requested_drop_columns if column in feature_column_set
    ]
    dropped_columns_missing = [
        column for column in requested_drop_columns if column not in feature_column_set
    ]
    dropped_column_set = set(dropped_columns_present)
    final_feature_columns = [
        column for column in feature_columns if column not in dropped_column_set
    ]

    forbidden = get_forbidden_predictors(feature_contract)
    assert_no_forbidden_predictors(final_feature_columns, forbidden)

    metadata = {
        "name": feature_view.get("name"),
        "drop_columns": requested_drop_columns,
        "dropped_columns_present": dropped_columns_present,
        "dropped_columns_missing": dropped_columns_missing,
        "permute_group_values": permutation_metadata,
        "permuted_columns": [item["column"] for item in permutation_metadata],
        "group_by_columns": [item["group_by"] for item in permutation_metadata],
    }
    return viewed, final_feature_columns, metadata
