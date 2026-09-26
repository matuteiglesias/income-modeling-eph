from __future__ import annotations

import pandas as pd
import pytest

from eph_income.entity_identity import (
    EPH_GLOBAL_HOUSEHOLD_KEY,
    EPH_GLOBAL_PERSON_KEY,
    EntityIdentityError,
    census_identity_columns,
    validate_census_person_identity,
    validate_eph_person_identity,
)


def test_period_qualified_eph_person_identity_is_unique() -> None:
    frame = pd.DataFrame(
        {
            "ANO4": [2024, 2024, 2024],
            "TRIMESTRE": [3, 3, 4],
            "CODUSU": ["A", "A", "A"],
            "NRO_HOGAR": [1, 1, 1],
            "COMPONENTE": [1, 2, 1],
        }
    )
    audit = validate_eph_person_identity(
        frame, require_period=True, context="fixture"
    )
    assert audit.key == EPH_GLOBAL_PERSON_KEY
    assert audit.unique is True
    assert EPH_GLOBAL_HOUSEHOLD_KEY == (
        "ANO4", "TRIMESTRE", "CODUSU", "NRO_HOGAR"
    )


def test_within_period_person_key_is_not_enough_for_pooled_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "ANO4": [2024, 2024],
            "TRIMESTRE": [3, 4],
            "CODUSU": ["A", "A"],
            "NRO_HOGAR": [1, 1],
            "COMPONENTE": [1, 1],
        }
    )
    validate_eph_person_identity(frame, require_period=True, context="pooled")
    with pytest.raises(EntityIdentityError, match="not unique"):
        validate_eph_person_identity(frame, require_period=False, context="pooled-local-key")


def test_census_identity_validates_full_entity_hierarchy() -> None:
    frame = pd.DataFrame(
        {
            "sample_person_id": ["sp1", "sp2", "sp3"],
            "frame_person_id": ["fp1", "fp2", "fp3"],
            "sample_household_id": ["sh1", "sh1", "sh2"],
            "frame_household_id": ["fh1", "fh1", "fh2"],
            "frame_dwelling_id": ["fd1", "fd1", "fd2"],
        }
    )
    audit = validate_census_person_identity(frame, context="census fixture")
    assert audit["identity_columns"] == census_identity_columns(frame)
    assert set(audit["relationships"]) == {
        "person_to_sample_household",
        "frame_person_to_household",
        "frame_household_to_dwelling",
        "sample_household_to_frame_household",
    }


def test_census_identity_rejects_one_person_mapped_to_two_households() -> None:
    frame = pd.DataFrame(
        {
            "sample_person_id": ["sp1", "sp2"],
            "sample_household_id": ["sh1", "sh2"],
            "frame_person_id": ["fp1", "fp2"],
            "frame_household_id": ["fh1", "fh2"],
            "frame_dwelling_id": ["fd1", "fd2"],
        }
    )
    frame.loc[1, "sample_person_id"] = "sp1"
    with pytest.raises(EntityIdentityError):
        validate_census_person_identity(frame, context="bad census")
