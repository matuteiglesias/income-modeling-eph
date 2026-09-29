import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from eph_income.longitudinal_frame import (
    LongitudinalFrameError,
    build_longitudinal_frame,
    build_parent_ledger,
    expected_periods,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _period_parts(period: str) -> tuple[int, int]:
    year, quarter = period.split("-Q")
    return int(year), int(quarter)


def _write_parent(root: Path, period: str, *, remove_estado: bool = False) -> None:
    year, quarter = _period_parts(period)
    release_id = f"eph-{year}-q{quarter}-fixture"
    release = root / release_id
    (release / "household").mkdir(parents=True)
    (release / "individual").mkdir()

    household = release / "household" / "hogar.txt"
    household.write_text(
        "CODUSU;NRO_HOGAR;ANO4;TRIMESTRE;AGLOMERADO;REGION;PONDERA;PONDIH;II7\n"
        f"A{year}{quarter};1;{year};{quarter};32;1;10;11;2\n"
        f"R{year};1;{year};{quarter};32;1;20;21;7\n",
        encoding="utf-8",
    )

    fields = [
        "CODUSU",
        "NRO_HOGAR",
        "COMPONENTE",
        "ANO4",
        "TRIMESTRE",
        "AGLOMERADO",
        "REGION",
        "PONDERA",
        "PONDIIO",
        "PONDII",
        "P47T",
        "CH04",
        "CH06",
    ]
    # Meaningful schema transitions: ESTADO is canonical throughout, CONDACT appears
    # later, and an optional labor-detail field appears only from 2024 onward.
    if not remove_estado:
        fields.append("ESTADO")
    if year >= 2020:
        fields.append("CONDACT")
    if year >= 2024:
        fields.append("PP04A")

    individual = release / "individual" / "individual.txt"
    with individual.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";", lineterminator="\n")
        writer.writeheader()
        row1 = {
            "CODUSU": f"A{year}{quarter}",
            "NRO_HOGAR": "1",
            "COMPONENTE": "1",
            "ANO4": str(year),
            "TRIMESTRE": str(quarter),
            "AGLOMERADO": "32",
            "REGION": "1",
            "PONDERA": "10",
            "PONDIIO": "12",
            "PONDII": "13",
            "P47T": "0" if period == "2017-Q1" else "100",
            "CH04": "1",
            "CH06": "30",
            "ESTADO": "1",
            "CONDACT": "1",
            "PP04A": "1",
        }
        # R2024 is deliberately repeated across quarters. Component 1 remains
        # demographically consistent except Q4, which changes sex and lets the
        # panel audit demonstrate that COMPONENTE is not blindly trusted.
        age = year - 1990
        repeated = {
            "CODUSU": f"R{year}",
            "NRO_HOGAR": "1",
            "COMPONENTE": "1",
            "ANO4": str(year),
            "TRIMESTRE": str(quarter),
            "AGLOMERADO": "32",
            "REGION": "1",
            "PONDERA": "20",
            "PONDIIO": "22",
            "PONDII": "23",
            "P47T": "200",
            "CH04": "2" if not (year == 2024 and quarter == 4) else "1",
            "CH06": str(age),
            "ESTADO": "2",
            "CONDACT": "2",
            "PP04A": "2",
        }
        writer.writerow({field: row1.get(field, "") for field in fields})
        writer.writerow({field: repeated.get(field, "") for field in fields})

    manifest = {
        "release_id": release_id,
        "requested_year": year,
        "requested_quarter": f"Q{quarter}",
        "files": [
            {"role": "household", "file": "household/hogar.txt", "sha256": _sha(household)},
            {
                "role": "individual",
                "file": "individual/individual.txt",
                "sha256": _sha(individual),
            },
        ],
    }
    manifest_path = release / "output-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    lock = {
        "schema": "eph-upstream-parent-lock/v1",
        "artifact_type": "publicdata.eph-microdata@1",
        "release_id": release_id,
        "period": {"year": year, "quarter": f"Q{quarter}"},
        "producer_manifest_sha256": _sha(manifest_path),
        "source": {
            "source_manifest_sha256": hashlib.sha256(f"manifest-{period}".encode()).hexdigest(),
            "source_archive_sha256": hashlib.sha256(f"archive-{period}".encode()).hexdigest(),
        },
    }
    (release / "parent_lock.json").write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")


def _parents(tmp_path: Path) -> Path:
    root = tmp_path / "parents"
    root.mkdir()
    for period in expected_periods():
        _write_parent(root, period)
    return root


def _conversion(tmp_path: Path, *, status: str = "approved") -> Path:
    root = tmp_path / "conversion"
    root.mkdir()
    factor_path = root / "monthly_conversion_factors.csv"
    rows = []
    # Every quarter midpoint plus one explicit common real reference month.
    for index, period in enumerate(expected_periods(), start=1):
        year, quarter = _period_parts(period)
        month = {1: 2, 2: 5, 3: 8, 4: 11}[quarter]
        rows.append(
            {
                "period": f"{year:04d}-{month:02d}-01",
                "reference_period": "2016-01-01",
                "consensus_index": str(100 + index),
                "factor_period_to_reference": str(Decimal(100) / Decimal(100 + index)),
                "factor_reference_to_period": str(Decimal(100 + index) / Decimal(100)),
                "coverage_class": "full",
                "approved_mode_eligible": "true",
            }
        )
    with factor_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "research-artifact-manifest/v1",
        "artifact_type": "research.argentina-monetary-conversion/v1",
        "release_id": "arg-monetary-conversion-fixture",
        "status": status,
        "monetary_reference_id": "fixture-price-consensus-v2",
        "files": [{"path": factor_path.name, "sha256": _sha(factor_path)}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "longitudinal_eph_harmonization_v1.json"
    )


def test_parent_ledger_requires_all_37_exact_quarters(tmp_path):
    parents = _parents(tmp_path)
    missing = parents / "eph-2019-q3-fixture"
    for child in sorted(missing.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        else:
            child.rmdir()
    missing.rmdir()

    with pytest.raises(LongitudinalFrameError, match="missing_parent_periods:2019-Q3"):
        build_parent_ledger(parents, tmp_path / "parent-lock.json")


def test_longitudinal_release_preserves_zeroes_drift_exceptions_and_panel_audit(tmp_path):
    parents = _parents(tmp_path)
    ledger = build_parent_ledger(parents, tmp_path / "parent-lock.json")
    conversion = _conversion(tmp_path)
    target_reference = "2025-11-01"

    release = build_longitudinal_frame(
        ledger,
        parents,
        conversion,
        _policy_path(),
        tmp_path / "releases",
        monetary_reference_period=target_reference,
    )
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    qa = json.loads((release / "qa.json").read_text(encoding="utf-8"))
    with (release / "persons.csv").open("r", encoding="utf-8", newline="") as stream:
        persons = list(csv.DictReader(stream))
    with (release / "coverage.csv").open("r", encoding="utf-8", newline="") as stream:
        coverage = {row["period"]: row for row in csv.DictReader(stream)}
    with (release / "schema_inventory.csv").open("r", encoding="utf-8", newline="") as stream:
        schema = list(csv.DictReader(stream))
    with (release / "panel_links.csv").open("r", encoding="utf-8", newline="") as stream:
        links = list(csv.DictReader(stream))

    assert manifest["contract"] == "research.eph-longitudinal-analysis-frame/v1"
    assert manifest["coverage"]["period_count"] == 37
    assert manifest["identity"]["permanent_person_identity_claim"] is False
    assert manifest["monetary_lineage"]["common_reference_period"] == target_reference
    assert manifest["monetary_lineage"]["common_reference_period"] != "2016-01-01"
    assert manifest["field_policy"]["zero_income_rows_retained"] is True
    assert manifest["field_policy"]["target_derived_geography_ranks_added"] is False
    assert qa["zero_income_persons_retained"] == 1
    assert qa["source_person_rows_accounted"] is True
    assert len(persons) == 74
    zero = next(row for row in persons if row["period"] == "2017-Q1" and row["P47T_nominal"] == "0")
    assert zero["P47T_real"] == "0"
    assert zero["p47t_value_status"] == "zero"
    assert zero["monetary_reference_period"] == target_reference
    assert "PP04A" in persons[0]
    assert next(row for row in persons if row["period"] == "2017-Q1")["PP04A"] == ""
    assert any(
        row["period"] == "2024-Q1"
        and row["role"] == "person"
        and "PP04A" in row["added_columns"]
        for row in schema
    )
    assert coverage["2020-Q2"]["exception_flags"] == "pandemic_fieldwork_regime"
    assert coverage["2024-Q1"]["ordinary_structural_time_eligible"] == "false"
    assert coverage["2024-Q2"]["ordinary_seasonality_eligible"] == "false"
    assert coverage["2024-Q3"]["exception_flags"] == ""
    assert any(link["person_linkage_status"] == "demographic_conflict" for link in links)
    assert any(
        link["person_linkage_status"] == "demographically_consistent_component_candidate"
        for link in links
    )
    assert (release / "checksums.sha256").is_file()


def test_required_schema_transition_fails_closed(tmp_path):
    parents = _parents(tmp_path)
    broken = parents / "eph-2021-q2-fixture" / "individual" / "individual.txt"
    rows = broken.read_text(encoding="utf-8").splitlines()
    header = rows[0].split(";")
    remove_indices = sorted(
        [header.index("ESTADO"), header.index("CONDACT")], reverse=True
    )
    for idx in remove_indices:
        header.pop(idx)
    rewritten = [";".join(header)]
    for line in rows[1:]:
        values = line.split(";")
        for idx in remove_indices:
            values.pop(idx)
        rewritten.append(";".join(values))
    broken.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    manifest_path = parents / "eph-2021-q2-fixture" / "output-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][1]["sha256"] = _sha(broken)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    lock_path = parents / "eph-2021-q2-fixture" / "parent_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["producer_manifest_sha256"] = _sha(manifest_path)
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")

    ledger = build_parent_ledger(parents, tmp_path / "parent-lock.json")
    conversion = _conversion(tmp_path)
    with pytest.raises(
        LongitudinalFrameError,
        match=r"2021-Q2:person:missing_required_any_of:ESTADO\|CONDACT",
    ):
        build_longitudinal_frame(
            ledger,
            parents,
            conversion,
            _policy_path(),
            tmp_path / "releases",
            monetary_reference_period="2025-11-01",
        )


def test_candidate_conversion_requires_explicit_override(tmp_path):
    parents = _parents(tmp_path)
    ledger = build_parent_ledger(parents, tmp_path / "parent-lock.json")
    conversion = _conversion(tmp_path, status="candidate")

    with pytest.raises(LongitudinalFrameError, match="conversion_release_not_approved:candidate"):
        build_longitudinal_frame(
            ledger,
            parents,
            conversion,
            _policy_path(),
            tmp_path / "releases",
            monetary_reference_period="2025-11-01",
        )
