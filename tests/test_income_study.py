import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from eph_income.analysis_frame import build_analysis_frame
from eph_income.income_study import IncomeStudyError, build_income_study_cohort


def _analysis_frame(tmp_path: Path) -> Path:
    parent = tmp_path / "eph-2026-q1-fixture"
    (parent / "household").mkdir(parents=True)
    (parent / "individual").mkdir()
    (parent / "household" / "hogar.txt").write_text(
        "CODUSU;NRO_HOGAR;PONDERA;PONDIH;II7\n"
        "A;1;10;11;7\n"
        "A;2;20;21;2\n",
        encoding="utf-8",
    )
    (parent / "individual" / "individual.txt").write_text(
        "CODUSU;NRO_HOGAR;COMPONENTE;ANO4;TRIMESTRE;PONDERA;PONDIIO;PONDII;PONDIH;P47T\n"
        "A;1;1;2026;1;10;12;13;14;300\n"
        "A;1;2;2026;1;10;15;16;17;199\n"
        "A;2;1;2026;1;20;22;23;24;400\n",
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
    return build_analysis_frame(parent, tmp_path / "analysis-frames")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _conversion_release(tmp_path: Path, *, status: str = "approved") -> Path:
    root = tmp_path / f"conversion-{status}"
    root.mkdir()
    factors = root / "monthly_conversion_factors.csv"
    factors.write_text(
        "period,reference_period,consensus_index,factor_period_to_reference,"
        "factor_reference_to_period,coverage_class,approved_mode_eligible\n"
        "2026-02-01,2016-01-01,200,0.5,2,acceptable_coverage,true\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "research-artifact-manifest/v1",
        "artifact_type": "research.argentina-monetary-conversion/v1",
        "release_id": f"conversion-{status}",
        "status": status,
        "monetary_reference_id": (
            "research.argentina-price-consensus/curated-official-panel-v2@2016-01=100"
        ),
        "files": [
            {
                "path": factors.name,
                "sha256": _sha256(factors),
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_income_study_rebuilds_historical_cohort_from_neutral_frame(tmp_path):
    frame = _analysis_frame(tmp_path)
    conversion = _conversion_release(tmp_path)

    release = build_income_study_cohort(frame, conversion, tmp_path / "cohorts")
    cohort = pd.read_csv(release / "cohort.csv")
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    qa = json.loads((release / "qa.json").read_text(encoding="utf-8"))

    assert manifest["contract"] == "research.eph-income-study-cohort@1"
    assert manifest["identity"]["person_key"] == ["CODUSU", "NRO_HOGAR", "COMPONENTE"]
    assert manifest["identity"]["household_group_key"] == ["CODUSU", "NRO_HOGAR"]
    assert manifest["weight_policy"]["claim_boundary"] == "sample_conditional"
    assert manifest["weight_policy"]["fitting"] is None
    assert manifest["weight_policy"]["evaluation"] is None
    assert manifest["weight_policy"]["calibration"] is None

    assert cohort[["CODUSU", "NRO_HOGAR", "COMPONENTE"]].values.tolist() == [
        ["A", 1, 1],
        ["A", 2, 1],
    ]
    assert cohort["P47T_native"].tolist() == [300.0, 400.0]
    assert cohort["P47T"].tolist() == [150.0, 200.0]
    assert cohort["INGRESO"].tolist() == [1, 1]
    assert cohort["PROP"].tolist() == [6, 2]
    assert pytest.approx(cohort["logP47T"].tolist()) == [
        2.1760912590556813,
        2.3010299956639813,
    ]
    assert {"PONDERA", "PONDIIO", "PONDII", "PONDIH"}.issubset(cohort.columns)
    assert qa["source_persons"] == 3
    assert qa["eligible_persons"] == 2
    assert qa["household_group_count"] == 2
    assert qa["identity_unique"] is True
    assert qa["survey_weights_used_for_cohort"] is False
    assert qa["survey_weights_used_for_target"] is False


def test_candidate_conversion_release_fails_closed(tmp_path):
    frame = _analysis_frame(tmp_path)
    conversion = _conversion_release(tmp_path, status="candidate")

    with pytest.raises(IncomeStudyError, match="conversion_release_not_approved:candidate"):
        build_income_study_cohort(frame, conversion, tmp_path / "cohorts")


def test_non_eligible_conversion_period_fails_closed(tmp_path):
    frame = _analysis_frame(tmp_path)
    conversion = _conversion_release(tmp_path)
    factors = conversion / "monthly_conversion_factors.csv"
    factors.write_text(
        "period,reference_period,consensus_index,factor_period_to_reference,"
        "factor_reference_to_period,coverage_class,approved_mode_eligible\n"
        "2026-02-01,2016-01-01,200,0.5,2,thin_coverage,false\n",
        encoding="utf-8",
    )
    manifest_path = conversion / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = _sha256(factors)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        IncomeStudyError,
        match="conversion_period_not_approved_mode_eligible:2026-02-01",
    ):
        build_income_study_cohort(frame, conversion, tmp_path / "cohorts")
