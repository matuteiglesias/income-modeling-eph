"""Executable scientific contract checks for the baseline experiment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

EXPECTED_TARGET_NAME = "logP47T"
EXPECTED_TARGET_SOURCE = "P47T"
EXPECTED_TARGET_TRANSFORM = "log10"
EXPECTED_TEMPORAL_FEATURES = frozenset({"ANO4", "TRIMESTRE"})
EXPECTED_TEMPORAL_DECISION = "include_in_baseline"


def get_forbidden_predictors(feature_contract: Mapping[str, Any]) -> set[str]:
    """Flatten all forbidden predictor groups from the feature contract.

    The YAML stores forbidden columns by scientific category. Modeling code needs
    a single set so it can fail loudly if any leakage variable is included in X.
    """

    groups = feature_contract.get("forbidden_predictors", {})
    if not isinstance(groups, Mapping):
        raise TypeError("feature_contract['forbidden_predictors'] must be a mapping of groups.")

    forbidden: set[str] = set()
    for group_name, columns in groups.items():
        if columns is None:
            continue
        if not isinstance(columns, list):
            raise TypeError(
                "Each forbidden predictor group must be a list of column names; "
                f"group {group_name!r} is {type(columns).__name__}."
            )
        forbidden.update(str(column) for column in columns)
    return forbidden


def assert_no_forbidden_predictors(
    feature_columns: Iterable[str], forbidden_columns: Iterable[str]
) -> None:
    """Raise ``ValueError`` if any forbidden predictor appears in ``feature_columns``.

    The error message lists offending columns in sorted order so leakage failures
    are easy to diagnose in dataset-building and experiment-running contexts.
    """

    feature_set = {str(column) for column in feature_columns}
    forbidden_set = {str(column) for column in forbidden_columns}
    offending = sorted(feature_set & forbidden_set)
    if offending:
        joined = ", ".join(offending)
        raise ValueError(f"Forbidden predictors present in feature columns: {joined}")


def validate_target_contract(feature_contract: Mapping[str, Any]) -> dict[str, str]:
    """Validate and return the baseline target contract."""

    target = feature_contract.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("Feature contract must define a 'target' mapping.")

    expected = {
        "name": EXPECTED_TARGET_NAME,
        "source": EXPECTED_TARGET_SOURCE,
        "transform": EXPECTED_TARGET_TRANSFORM,
    }
    errors = [
        f"target.{key} expected {expected_value!r}, got {target.get(key)!r}"
        for key, expected_value in expected.items()
        if target.get(key) != expected_value
    ]
    if errors:
        raise ValueError("Invalid target contract: " + "; ".join(errors))

    return {key: str(target[key]) for key in expected}


def validate_temporal_features(feature_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the baseline decision to include ANO4 and TRIMESTRE."""

    temporal = feature_contract.get("temporal_features")
    if not isinstance(temporal, Mapping):
        raise ValueError("Feature contract must define a 'temporal_features' mapping.")

    included = temporal.get("include")
    if not isinstance(included, list):
        raise ValueError("temporal_features.include must be a list.")

    included_set = {str(column) for column in included}
    missing = sorted(EXPECTED_TEMPORAL_FEATURES - included_set)
    decision = temporal.get("decision")
    errors: list[str] = []
    if missing:
        errors.append("missing temporal features: " + ", ".join(missing))
    if decision != EXPECTED_TEMPORAL_DECISION:
        errors.append(
            f"temporal_features.decision expected {EXPECTED_TEMPORAL_DECISION!r}, got {decision!r}"
        )
    if errors:
        raise ValueError("Invalid temporal feature contract: " + "; ".join(errors))

    return {"include": included, "decision": str(decision)}
