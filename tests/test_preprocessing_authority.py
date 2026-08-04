import json
from pathlib import Path

import pandas as pd
import pytest

from eph_income.preprocessing_authority import build_manifest, run_fixture, validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_fixture_exercises_merge_deflation_mapping_and_inclusion() -> None:
    spec = json.loads((ROOT / "tests/fixtures/preprocessing_fixture.json").read_text())
    result = run_fixture(spec)
    assert result["person_id"].tolist() == [1, 2, 1, 1]
    assert result.loc[result["CODUSU"].eq("h2"), "P47T"].item() == 1100
    assert result["merge_valid"].sum() == 3
    assert result["included"].sum() == 1
    assert result.loc[result["CODUSU"].eq("missing"), "Region"].isna().all()


def test_fixture_rejects_nonunique_households() -> None:
    spec = json.loads((ROOT / "tests/fixtures/preprocessing_fixture.json").read_text())
    spec["households"].append(dict(spec["households"][0]))
    with pytest.raises(ValueError, match="many-to-one"):
        run_fixture(spec)


def test_manifest_round_trip(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame(
        {"ANO4": [2024], "TRIMESTRE": [1], "CODUSU": ["x"], "P02": [1], "P03": [30]}
    ).to_csv(csv_path, index=False)
    lineage = tmp_path / "lineage.yaml"
    lineage.write_text(
        "schema_version: 1\ncolumns:\n"
        + "".join(f"  {c}: {{reviewer_status: reviewed}}\n" for c in ["ANO4", "TRIMESTRE", "CODUSU", "P02", "P03"])
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "preprocessing_release.yaml").write_text("schema_version: 1\n")
    manifest = build_manifest(csv_path, lineage, tmp_path)
    validate_manifest(manifest, csv_path, lineage)
    csv_path.write_text(csv_path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        validate_manifest(manifest, csv_path, lineage)
