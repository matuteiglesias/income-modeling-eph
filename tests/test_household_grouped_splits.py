from __future__ import annotations

import pandas as pd
import pytest

from eph_income.splits import (
    HOUSEHOLD_GROUPED_STRATEGY,
    create_split_assignments,
    validate_split_assignments,
)


def _household_dataset() -> pd.DataFrame:
    rows = []
    row_id = 0
    for household in range(20):
        members = 1 + (household % 4)
        for component in range(1, members + 1):
            rows.append(
                {
                    "row_id": row_id,
                    "CODUSU": f"U{household // 3}",
                    "NRO_HOGAR": str(household),
                    "COMPONENTE": component,
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


def _config() -> dict:
    return {
        "strategy": HOUSEHOLD_GROUPED_STRATEGY,
        "group_columns": ["CODUSU", "NRO_HOGAR"],
        "train_size": 0.70,
        "test_size": 0.20,
        "validation_size": 0.10,
        "random_state": 42,
    }


def test_household_grouped_split_is_deterministic_and_has_no_household_leakage() -> None:
    dataset = _household_dataset()
    first = create_split_assignments(dataset, _config())
    second = create_split_assignments(dataset, _config())

    pd.testing.assert_frame_equal(first, second)
    validate_split_assignments(
        dataset,
        first,
        group_columns=["CODUSU", "NRO_HOGAR"],
    )

    joined = dataset.merge(first, on="row_id", validate="one_to_one")
    assert joined.groupby(["CODUSU", "NRO_HOGAR"])["split"].nunique().max() == 1
    assert set(joined["split"]) == {"train", "test", "validation"}


def test_household_grouped_split_fails_when_household_identity_is_missing() -> None:
    dataset = _household_dataset().drop(columns=["NRO_HOGAR"])
    with pytest.raises(KeyError, match="NRO_HOGAR"):
        create_split_assignments(dataset, _config())


def test_group_integrity_validator_rejects_cross_split_household() -> None:
    dataset = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "CODUSU": ["A", "A", "B", "C"],
            "NRO_HOGAR": ["1", "1", "1", "1"],
        }
    )
    bad = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "split": ["train", "test", "test", "validation"],
        }
    )
    with pytest.raises(ValueError, match="household crosses splits"):
        validate_split_assignments(
            dataset,
            bad,
            group_columns=["CODUSU", "NRO_HOGAR"],
        )
