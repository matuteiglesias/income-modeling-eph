from __future__ import annotations

import pandas as pd

from eph_income.dataset import (
    ROW_ID_COLUMN,
    SOURCE_PERSON_IDENTITY,
    build_source_identity_sidecar,
)


def test_source_identity_sidecar_is_exact_and_row_id_keyed() -> None:
    frame = pd.DataFrame(
        {
            ROW_ID_COLUMN: [0, 1],
            "CODUSU": ["A", "A"],
            "NRO_HOGAR": [1, 1],
            "COMPONENTE": [1, 2],
            "ANO4": [2024, 2024],
            "TRIMESTRE": [3, 3],
            "x": [10, 20],
        }
    )
    sidecar, status = build_source_identity_sidecar(frame)
    assert sidecar is not None
    assert status["status"] == "exact"
    assert status["exact_source_identity_available"] is True
    assert list(sidecar.columns) == [ROW_ID_COLUMN, *SOURCE_PERSON_IDENTITY]
    assert sidecar[ROW_ID_COLUMN].tolist() == [0, 1]


def test_legacy_identity_gap_fails_closed_without_nro_hogar_component() -> None:
    frame = pd.DataFrame(
        {
            ROW_ID_COLUMN: [0, 1],
            "CODUSU": ["A", "A"],
            "ANO4": [2024, 2024],
            "TRIMESTRE": [3, 3],
        }
    )
    sidecar, status = build_source_identity_sidecar(frame)
    assert sidecar is None
    assert status["status"] == "unavailable"
    assert status["exact_source_identity_available"] is False
    assert status["missing_columns"] == ["NRO_HOGAR", "COMPONENTE"]


def test_non_unique_exact_identity_is_not_published() -> None:
    frame = pd.DataFrame(
        {
            ROW_ID_COLUMN: [0, 1],
            "CODUSU": ["A", "A"],
            "NRO_HOGAR": [1, 1],
            "COMPONENTE": [1, 1],
            "ANO4": [2024, 2024],
            "TRIMESTRE": [3, 3],
        }
    )
    sidecar, status = build_source_identity_sidecar(frame)
    assert sidecar is None
    assert status["status"] == "non_unique"
    assert status["duplicate_identity_rows"] == 2
