import json
from pathlib import Path

import pytest

from eph_income.analysis_frame import AnalysisFrameError, build_analysis_frame


def _parent(root: Path, *, orphan: bool = False) -> Path:
    parent = root / "eph-2026-q1-fixture"
    (parent / "household").mkdir(parents=True)
    (parent / "individual").mkdir()
    (parent / "household" / "hogar.txt").write_text(
        "CODUSU;NRO_HOGAR;PONDERA;PONDIH;IV1\nA;1;10;11;1\nA;2;20;21;2\n",
        encoding="utf-8",
    )
    second_household = "9" if orphan else "1"
    (parent / "individual" / "individual.txt").write_text(
        "CODUSU;NRO_HOGAR;COMPONENTE;PONDERA;PONDIIO;CH04;CH06;P47T\n"
        "A;1;1;10;12;1;34;100\n"
        f"A;{second_household};2;10;13;2;30;200\n",
        encoding="utf-8",
    )
    manifest = {
        "release_id": parent.name,
        "files": [
            {"role": "household", "file": "household/hogar.txt"},
            {"role": "individual", "file": "individual/individual.txt"},
        ],
    }
    lock = {
        "schema": "eph-upstream-parent-lock/v1",
        "release_id": parent.name,
        "period": {"year": 2026, "quarter": "Q1"},
        "source": {"source_archive_sha256": "fixture-source-sha"},
    }
    (parent / "output-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (parent / "parent_lock.json").write_text(json.dumps(lock), encoding="utf-8")
    return parent


def test_neutral_frame_preserves_native_identity_and_design_fields(tmp_path):
    parent = _parent(tmp_path)
    release = build_analysis_frame(parent, tmp_path / "releases")
    manifest = json.loads((release / "manifest.json").read_text())
    qa = json.loads((release / "qa.json").read_text())

    assert manifest["contract"] == "research.eph-analysis-frame@1"
    assert manifest["identity"]["household_key"] == ["CODUSU", "NRO_HOGAR"]
    assert manifest["identity"]["person_key"] == ["CODUSU", "NRO_HOGAR", "COMPONENTE"]
    assert manifest["field_policy"]["source_semantics"] == "native_eph"
    assert manifest["field_policy"]["survey_weights_applied"] is False
    assert manifest["field_policy"]["census_shaped_aliases_added"] is False
    assert manifest["field_policy"]["target_or_cohort_fields_added"] is False
    assert manifest["monetary_lineage"]["deflation_or_rebasing_applied"] is False
    assert qa["person_to_household_cardinality"] == "many_to_one_validated"
    assert "PONDERA" in qa["survey_design_fields"]["individual"]
    assert "PONDIH" in qa["survey_design_fields"]["household"]

    person_header = (release / "persons.csv").read_text().splitlines()[0]
    assert "CODUSU" in person_header
    assert "NRO_HOGAR" in person_header
    assert "COMPONENTE" in person_header
    assert "P47T" in person_header
    assert "AGLO_rk" not in person_header
    assert "Reg_rk" not in person_header
    assert "logP47T" not in person_header


def test_orphan_person_household_fails_closed(tmp_path):
    parent = _parent(tmp_path, orphan=True)
    with pytest.raises(AnalysisFrameError, match="person_household_cardinality_failure"):
        build_analysis_frame(parent, tmp_path / "releases")
