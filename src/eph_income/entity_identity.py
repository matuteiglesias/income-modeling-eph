"""Canonical entity-identity contracts shared across EPH and Census surfaces.

Identity columns are lineage metadata, never model predictors. Contracts retain
raw keys rather than reconstructing missing identities from row order or feature
signatures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


EPH_HOUSEHOLD_KEY = ("CODUSU", "NRO_HOGAR")
EPH_PERSON_KEY = ("CODUSU", "NRO_HOGAR", "COMPONENTE")
EPH_PERIOD_KEY = ("ANO4", "TRIMESTRE")
EPH_GLOBAL_HOUSEHOLD_KEY = (*EPH_PERIOD_KEY, *EPH_HOUSEHOLD_KEY)
EPH_GLOBAL_PERSON_KEY = (*EPH_PERIOD_KEY, *EPH_PERSON_KEY)

CENSUS_FRAME_DWELLING_KEY = ("frame_dwelling_id",)
CENSUS_FRAME_HOUSEHOLD_KEY = ("frame_household_id",)
CENSUS_FRAME_PERSON_KEY = ("frame_person_id",)
CENSUS_SAMPLE_HOUSEHOLD_KEY = ("sample_household_id",)
CENSUS_SAMPLE_PERSON_KEY = ("sample_person_id",)

CENSUS_IDENTITY_COLUMNS = (
    "sample_person_id",
    "frame_person_id",
    "sample_household_id",
    "frame_household_id",
    "frame_dwelling_id",
)


class EntityIdentityError(ValueError):
    """Raised when entity identity is missing, non-unique, or inconsistent."""


@dataclass(frozen=True)
class EntityIdentityAudit:
    rows: int
    key: tuple[str, ...]
    unique: bool
    missing_rows: int


def _require(frame: "pd.DataFrame", columns: Sequence[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise EntityIdentityError(f"{context} missing identity columns: {missing}")


def audit_unique_key(
    frame: "pd.DataFrame",
    key: Sequence[str],
    *,
    context: str,
) -> EntityIdentityAudit:
    import pandas as pd  # lazy: neutral frame producers only need identity constants

    key = tuple(key)
    _require(frame, key, context)
    values = frame.loc[:, list(key)]
    missing_mask = values.isna().any(axis=1)
    object_columns = values.select_dtypes(include=["object", "string"]).columns
    if len(object_columns):
        string_values = values.loc[:, object_columns].astype("string")
        missing_mask |= string_values.apply(lambda col: col.str.strip().eq("")).any(axis=1)
    missing_rows = int(missing_mask.sum())
    if missing_rows:
        raise EntityIdentityError(f"{context} identity has missing rows: {missing_rows}")
    duplicate_rows = int(values.duplicated(keep=False).sum())
    if duplicate_rows:
        raise EntityIdentityError(
            f"{context} identity is not unique: duplicate_rows={duplicate_rows}"
        )
    return EntityIdentityAudit(rows=int(len(frame)), key=key, unique=True, missing_rows=0)


def validate_eph_person_identity(
    frame: "pd.DataFrame", *, require_period: bool, context: str
) -> EntityIdentityAudit:
    key = EPH_GLOBAL_PERSON_KEY if require_period else EPH_PERSON_KEY
    return audit_unique_key(frame, key, context=context)


def validate_eph_household_identity(
    frame: "pd.DataFrame", *, require_period: bool, context: str
) -> EntityIdentityAudit:
    key = EPH_GLOBAL_HOUSEHOLD_KEY if require_period else EPH_HOUSEHOLD_KEY
    return audit_unique_key(frame, key, context=context)


def census_identity_columns(frame: "pd.DataFrame") -> list[str]:
    """Return recognized Census identity columns in canonical hierarchy order."""
    return [column for column in CENSUS_IDENTITY_COLUMNS if column in frame.columns]


def validate_census_person_identity(
    frame: "pd.DataFrame", *, context: str
) -> dict[str, object]:
    """Validate Census person identity and available parent relationships."""
    _require(frame, CENSUS_SAMPLE_PERSON_KEY, context)
    audit_unique_key(frame, CENSUS_SAMPLE_PERSON_KEY, context=context)

    if "frame_person_id" in frame:
        audit_unique_key(frame, CENSUS_FRAME_PERSON_KEY, context=f"{context}:frame-person")

    relationships: dict[str, object] = {}
    pairs = (
        ("sample_person_id", "sample_household_id", "person_to_sample_household"),
        ("frame_person_id", "frame_household_id", "frame_person_to_household"),
        ("frame_household_id", "frame_dwelling_id", "frame_household_to_dwelling"),
        ("sample_household_id", "frame_household_id", "sample_household_to_frame_household"),
    )
    for child, parent, label in pairs:
        if child not in frame or parent not in frame:
            continue
        pair = frame[[child, parent]].drop_duplicates()
        parent_counts = pair.groupby(child, dropna=False)[parent].nunique(dropna=False)
        violating = int((parent_counts != 1).sum())
        if violating:
            raise EntityIdentityError(
                f"{context} {label} is not many-to-one: violating_children={violating}"
            )
        relationships[label] = {
            "status": "validated",
            "child": child,
            "parent": parent,
            "children": int(pair[child].nunique(dropna=False)),
        }

    return {
        "rows": int(len(frame)),
        "identity_columns": census_identity_columns(frame),
        "person_key": list(CENSUS_SAMPLE_PERSON_KEY),
        "relationships": relationships,
    }
