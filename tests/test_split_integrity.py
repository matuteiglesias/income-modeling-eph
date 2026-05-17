from __future__ import annotations

import pandas as pd
import pytest

from eph_income.splits import (
    create_split_assignments,
    load_or_create_split_assignments,
    split_proportions,
    validate_split_assignments,
)


def _dataset(n: int = 1000) -> pd.DataFrame:
    return pd.DataFrame({"row_id": range(n), "logP47T": [1.0] * n})


def _config(path) -> dict:
    return {
        "data": {"split_assignments": str(path)},
        "split": {
            "train_size": 0.70,
            "test_size": 0.20,
            "validation_size": 0.10,
            "random_state": 42,
        },
    }


def test_create_split_assignments_has_one_valid_split_per_row() -> None:
    assignments = create_split_assignments(_dataset(), _config("unused")["split"])

    assert len(assignments) == 1000
    assert assignments["row_id"].is_unique
    assert set(assignments["split"]) == {"train", "test", "validation"}
    props = split_proportions(assignments)
    assert props["train"] == pytest.approx(0.70, abs=0.01)
    assert props["test"] == pytest.approx(0.20, abs=0.01)
    assert props["validation"] == pytest.approx(0.10, abs=0.01)


def test_split_file_is_reused_unless_overwritten(tmp_path) -> None:
    dataset = _dataset(100)
    config = _config(tmp_path / "split_assignments.csv")

    first = load_or_create_split_assignments(dataset, config)
    second = load_or_create_split_assignments(dataset, config)

    pd.testing.assert_frame_equal(first, second)


def test_split_validation_rejects_missing_row_ids() -> None:
    dataset = _dataset(3)
    bad = pd.DataFrame({"row_id": [0, 1], "split": ["train", "test"]})

    with pytest.raises(ValueError, match="do not match"):
        validate_split_assignments(dataset, bad)
