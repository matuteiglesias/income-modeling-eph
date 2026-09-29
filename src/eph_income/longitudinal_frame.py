"""Build the governed 2017-Q1..2026-Q1 longitudinal EPH analysis frame."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from eph_income.entity_identity import EPH_HOUSEHOLD_KEY, EPH_PERSON_KEY
from eph_income.longitudinal_inputs import (
    CONVERSION_CONTRACT,
    END_PERIOD,
    EXPECTED_PERIOD_COUNT,
    HARMONIZATION_SCHEMA,
    LONGITUDINAL_CONTRACT,
    PARENT_LOCK_SCHEMA,
    START_PERIOD,
    UPSTREAM_ARTIFACT,
    LongitudinalFrameError,
    build_monetary_plan,
    build_parent_ledger,
    canonical_json,
    decimal_text,
    expected_periods,
    load_harmonization,
    parse_period,
    period_index,
    read_rows,
    sha256,
    validate_parent_ledger,
)

EXCEPTION_PERIODS = {
    "2020-Q2": "pandemic_fieldwork_regime",
    "2024-Q1": "2024_h1_macroeconomic_shock",
    "2024-Q2": "2024_h1_macroeconomic_shock",
}
REGION_IDS = {
    "1": "gran_buenos_aires",
    "40": "noroeste",
    "41": "noreste",
    "42": "cuyo",
    "43": "pampeana",
    "44": "patagonia",
}
WEIGHT_FIELDS = ("PONDERA", "PONDIIO", "PONDII", "PONDIH")
EXTRA_COLUMNS = (
    "row_id",
    "household_observation_id",
    "panel_dwelling_id",
    "panel_household_id",
    "person_linkage_candidate_id",
    "person_key_within_household",
    "period",
    "region_id",
    "P47T_nominal",
    "P47T_real",
    "p47t_value_status",
    "monetary_source_period",
    "monetary_reference_period",
    "monetary_factor_period_to_reference",
    "source_release_id",
    "source_row_identity",
    "weight_pondera",
    "weight_pondiio",
    "weight_pondii",
    "weight_pondih",
    "exception_flags",
    "ordinary_seasonality_eligible",
    "ordinary_structural_time_eligible",
    "shock_effect_group",
)


def _unique(values: Iterable[str]) -> list[str]:
    output, seen = [], set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _parent_table(pinned_root: Path, entry: dict[str, Any], role: str) -> Path:
    relative = PurePosixPath(entry[role]["file"])
    path = Path(pinned_root) / entry["source_release_id"] / Path(*relative.parts)
    if not path.is_file() or sha256(path) != entry[role]["sha256"]:
        raise LongitudinalFrameError(f"parent_table_drift:{entry['period']}:{role}")
    return path


def _check_period(row: dict[str, str], period: str, role: str) -> None:
    year, quarter = parse_period(period)
    try:
        observed = int(row["ANO4"]), int(row["TRIMESTRE"])
    except (KeyError, ValueError) as exc:
        raise LongitudinalFrameError(f"{period}:{role}:invalid_row_period") from exc
    if observed != (year, quarter):
        raise LongitudinalFrameError(f"{period}:{role}:row_period_mismatch")


def _region(raw: str, period: str) -> str:
    code = raw.strip().removesuffix(".0")
    try:
        return REGION_IDS[code]
    except KeyError as exc:
        raise LongitudinalFrameError(f"{period}:unknown_region_code:{raw}") from exc


def _exception(period: str) -> dict[str, str]:
    shock = EXCEPTION_PERIODS.get(period, "")
    ordinary = "false" if shock else "true"
    return {
        "exception_flags": shock,
        "ordinary_seasonality_eligible": ordinary,
        "ordinary_structural_time_eligible": ordinary,
        "shock_effect_group": shock,
    }


def _schema_rows(
    entries: list[dict[str, Any]], policy: dict[str, Any]
) -> list[dict[str, str]]:
    output = []
    previous: dict[str, list[str] | None] = {"individual": None, "household": None}
    for entry in entries:
        for source_role, role in (("individual", "person"), ("household", "household")):
            fields = list(entry[source_role]["columns"])
            before = previous[source_role]
            added = fields if before is None else [field for field in fields if field not in before]
            removed = [] if before is None else [field for field in before if field not in fields]
            reordered = before is not None and not added and not removed and fields != before
            if before is None:
                drift = "initial"
            elif added or removed:
                drift = "optional_column_set_change"
            elif reordered:
                drift = "column_order_change"
            else:
                drift = "identical"
            required = set(policy.get(f"required_{role}_fields") or [])
            output.append(
                {
                    "period": entry["period"],
                    "role": role,
                    "column_count": str(len(fields)),
                    "schema_fingerprint": entry[source_role]["schema_fingerprint"],
                    "drift_kind": drift,
                    "added_columns": ";".join(added),
                    "removed_columns": ";".join(removed),
                    "missing_required_columns": ";".join(sorted(required - set(fields))),
                    "harmonization_action": (
                        "preserve_source_union_blank_when_absent"
                        if drift == "optional_column_set_change"
                        else "identity_preserve"
                    ),
                }
            )
            previous[source_role] = fields
    return output


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _income(nominal_text: str, factor: Decimal) -> tuple[str, str, Decimal]:
    if not nominal_text:
        return "", "missing", Decimal(0)
    try:
        nominal = Decimal(nominal_text)
    except InvalidOperation:
        return "", "non_numeric_source_value", Decimal(0)
    if not nominal.is_finite():
        return "", "non_finite_source_value", Decimal(0)
    if nominal < 0:
        return "", "negative_source_code_or_value", Decimal(0)
    with localcontext() as context:
        context.prec = 40
        real = nominal * factor
        recovered = real / factor
    status = "zero" if nominal == 0 else "positive"
    return decimal_text(real), status, abs(recovered - nominal)


def _age(value: str) -> int | None:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if 0 <= parsed <= 120 else None


def _classify_link(
    previous_sex: str,
    current_sex: str,
    previous_age_raw: str,
    current_age_raw: str,
    elapsed: int,
) -> tuple[str, str, str]:
    cadence = "expected_2_2_2_adjacent_gap" if elapsed in {1, 3} else "irregular_or_attrition_gap"
    if elapsed <= 0:
        return "invalid_time_order", "conflict", cadence
    if elapsed > 5:
        return "outside_expected_panel_horizon", "not_supported", cadence
    if previous_sex and current_sex and previous_sex != current_sex:
        return "demographic_conflict", "conflict", cadence
    previous_age, current_age = _age(previous_age_raw), _age(current_age_raw)
    if previous_age is not None and current_age is not None:
        delta = current_age - previous_age
        if delta < -1 or delta > ((elapsed + 3) // 4) + 1:
            return "demographic_conflict", "conflict", cadence
    if previous_sex and current_sex and previous_age is not None and current_age is not None:
        return (
            "demographically_consistent_component_candidate",
            "supporting_demographic_consistency",
            cadence,
        )
    return "component_key_only_candidate", "key_only", cadence


def _panel_audit(database: sqlite3.Connection, staging: Path) -> dict[str, Any]:
    database.execute("CREATE INDEX person_candidate_idx ON person_obs(candidate_id, period_index)")
    database.execute(
        "CREATE INDEX household_candidate_idx ON household_obs(panel_household_id, period_index)"
    )
    link_fields = [
        "person_linkage_candidate_id",
        "previous_row_id",
        "current_row_id",
        "previous_period",
        "current_period",
        "elapsed_quarters",
        "previous_component",
        "current_component",
        "person_linkage_status",
        "person_linkage_confidence",
        "rotation_gap_status",
    ]
    counts: Counter[str] = Counter()
    query = """
      WITH ordered AS (
        SELECT candidate_id,row_id,period,period_index,component,sex,age,
          LAG(row_id) OVER w previous_row_id,
          LAG(period) OVER w previous_period,
          LAG(period_index) OVER w previous_period_index,
          LAG(component) OVER w previous_component,
          LAG(sex) OVER w previous_sex,
          LAG(age) OVER w previous_age
        FROM person_obs
        WINDOW w AS (PARTITION BY candidate_id ORDER BY period_index,row_id)
      )
      SELECT * FROM ordered WHERE previous_row_id IS NOT NULL
      ORDER BY candidate_id,period_index,row_id
    """
    with (staging / "panel_links.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=link_fields, lineterminator="\n")
        writer.writeheader()
        for row in database.execute(query):
            (
                candidate,
                current_row,
                current_period,
                current_index,
                component,
                sex,
                age,
                previous_row,
                previous_period,
                previous_index,
                previous_component,
                previous_sex,
                previous_age,
            ) = row
            elapsed = int(current_index) - int(previous_index)
            status, confidence, cadence = _classify_link(
                previous_sex or "", sex or "", previous_age or "", age or "", elapsed
            )
            counts[status] += 1
            writer.writerow(
                {
                    "person_linkage_candidate_id": candidate,
                    "previous_row_id": previous_row,
                    "current_row_id": current_row,
                    "previous_period": previous_period,
                    "current_period": current_period,
                    "elapsed_quarters": elapsed,
                    "previous_component": previous_component,
                    "current_component": component,
                    "person_linkage_status": status,
                    "person_linkage_confidence": confidence,
                    "rotation_gap_status": cadence,
                }
            )

    household_fields = [
        "panel_household_id",
        "panel_dwelling_id",
        "observation_count",
        "first_period",
        "last_period",
        "observed_periods",
    ]
    query = """
      SELECT panel_household_id,panel_dwelling_id,COUNT(*),MIN(period),MAX(period),
             GROUP_CONCAT(period,';')
      FROM household_obs
      GROUP BY panel_household_id,panel_dwelling_id
      ORDER BY panel_household_id
    """
    households, repeated = 0, 0
    with (staging / "panel_households.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=household_fields, lineterminator="\n")
        writer.writeheader()
        for panel_household, dwelling, count, first, last, periods in database.execute(query):
            households += 1
            repeated += int(count > 1)
            writer.writerow(
                {
                    "panel_household_id": panel_household,
                    "panel_dwelling_id": dwelling,
                    "observation_count": count,
                    "first_period": first,
                    "last_period": last,
                    "observed_periods": periods,
                }
            )
    return {
        "candidate_link_rows": sum(counts.values()),
        "candidate_link_status_counts": dict(sorted(counts.items())),
        "panel_household_candidates": households,
        "repeated_panel_household_candidates": repeated,
        "person_identity_interpretation": (
            "CODUSU+NRO_HOGAR+COMPONENTE is a short-horizon linkage candidate only; "
            "it is not permanent person identity."
        ),
    }


def _inventory(staging: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": sha256(staging / name), "size_bytes": (staging / name).stat().st_size}
        for name in names
    }


def _checksums(staging: Path) -> None:
    names = sorted(
        path.name
        for path in staging.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (staging / "checksums.sha256").write_text(
        "".join(f"{sha256(staging / name)}  {name}\n" for name in names), encoding="utf-8"
    )


def build_longitudinal_frame(
    parent_ledger_path: Path,
    pinned_root: Path,
    conversion_root: Path,
    harmonization_path: Path,
    output_root: Path,
    *,
    monetary_reference_period: str,
    require_approved_conversion: bool = True,
) -> Path:
    """Build one immutable neutral longitudinal EPH frame from 37 exact parents."""
    parent_ledger_path = Path(parent_ledger_path).resolve()
    pinned_root = Path(pinned_root).resolve()
    conversion_root = Path(conversion_root).resolve()
    harmonization_path = Path(harmonization_path).resolve()
    output_root = Path(output_root).resolve()
    policy = load_harmonization(harmonization_path)
    ledger, entries = validate_parent_ledger(parent_ledger_path, pinned_root, policy)
    conversion_manifest, monetary = build_monetary_plan(
        conversion_root,
        monetary_reference_period,
        require_approved=require_approved_conversion,
    )

    person_union = _unique(
        field for entry in entries for field in entry["individual"]["columns"]
    )
    household_union = _unique(
        field for entry in entries for field in entry["household"]["columns"]
    )
    collisions = sorted(set(person_union) & set(EXTRA_COLUMNS))
    if collisions:
        raise LongitudinalFrameError(
            "source_columns_collide_with_longitudinal_contract:" + ",".join(collisions)
        )
    household_context = [
        field
        for field in household_union
        if field not in {*EPH_HOUSEHOLD_KEY, "ANO4", "TRIMESTRE"}
    ]
    output_fields = [*EXTRA_COLUMNS, *person_union]
    output_fields.extend(f"HH__{field}" for field in household_context)

    parent_lock_sha = sha256(parent_ledger_path)
    harmonization_sha = sha256(harmonization_path)
    conversion_manifest_sha = sha256(conversion_root / "manifest.json")
    seed = {
        "contract": LONGITUDINAL_CONTRACT,
        "parent_lock_sha256": parent_lock_sha,
        "harmonization_sha256": harmonization_sha,
        "conversion_release_id": conversion_manifest.get("release_id"),
        "conversion_manifest_sha256": conversion_manifest_sha,
        "monetary_reference_period": monetary_reference_period,
    }
    digest = hashlib.sha256(
        json.dumps(seed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    release_id = f"eph-longitudinal-2017q1-2026q1-{digest[:16]}"
    destination = output_root / release_id
    if destination.exists():
        raise LongitudinalFrameError(f"immutable_release_exists:{destination}")

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release_id}.", dir=output_root))
    database = sqlite3.connect(staging / "_panel.sqlite")
    database.execute(
        "CREATE TABLE person_obs(candidate_id,row_id,panel_household_id,period,"
        "period_index,component,sex,age)"
    )
    database.execute(
        "CREATE TABLE household_obs(panel_household_id,panel_dwelling_id,period,"
        "period_index,household_observation_id UNIQUE)"
    )
    try:
        (staging / "parent_lock.json").write_text(canonical_json(ledger), encoding="utf-8")
        (staging / "harmonization.json").write_text(canonical_json(policy), encoding="utf-8")
        _write_csv(
            staging / "schema_inventory.csv",
            [
                "period",
                "role",
                "column_count",
                "schema_fingerprint",
                "drift_kind",
                "added_columns",
                "removed_columns",
                "missing_required_columns",
                "harmonization_action",
            ],
            _schema_rows(entries, policy),
        )
        _write_csv(
            staging / "monetary_lineage.csv",
            [
                "period",
                "source_period",
                "reference_period",
                "producer_base_reference_period",
                "factor_period_to_reference",
                "source_coverage_class",
                "reference_coverage_class",
                "approved_mode_eligible",
            ],
            ({"period": period, **monetary[period]} for period in expected_periods()),
        )

        coverage, total_persons = [], 0
        status_counts: Counter[str] = Counter()
        roundtrip_error = Decimal(0)
        with (staging / "persons.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=output_fields, lineterminator="\n")
            writer.writeheader()
            for entry in entries:
                period = entry["period"]
                households: dict[tuple[str, str], dict[str, str]] = {}
                household_rows = []
                for household in read_rows(_parent_table(pinned_root, entry, "household")):
                    _check_period(household, period, "household")
                    key = tuple(household[field] for field in EPH_HOUSEHOLD_KEY)
                    if any(not part for part in key) or key in households:
                        raise LongitudinalFrameError(f"{period}:household:invalid_identity")
                    households[key] = household
                    panel_household = f"{key[0]}|{key[1]}"
                    household_rows.append(
                        (
                            panel_household,
                            key[0],
                            period,
                            period_index(period),
                            f"{period}|{key[0]}|{key[1]}",
                        )
                    )
                database.executemany("INSERT INTO household_obs VALUES(?,?,?,?,?)", household_rows)

                factor = Decimal(monetary[period]["factor_period_to_reference"])
                exception = _exception(period)
                seen_people: set[tuple[str, str, str]] = set()
                panel_rows, period_status = [], Counter()
                person_count = 0
                for person in read_rows(_parent_table(pinned_root, entry, "individual")):
                    _check_period(person, period, "person")
                    key = tuple(person[field] for field in EPH_PERSON_KEY)
                    if any(not part for part in key) or key in seen_people:
                        raise LongitudinalFrameError(f"{period}:person:invalid_identity")
                    seen_people.add(key)
                    household = households.get(key[:2])
                    if household is None:
                        raise LongitudinalFrameError(f"{period}:person_without_household:{key[:2]}")
                    row_id = f"{period}|{key[0]}|{key[1]}|{key[2]}"
                    panel_household = f"{key[0]}|{key[1]}"
                    candidate = f"{key[0]}|{key[1]}|{key[2]}"
                    nominal = person.get("P47T", "").strip()
                    real, value_status, error = _income(nominal, factor)
                    roundtrip_error = max(roundtrip_error, error)
                    status_counts[value_status] += 1
                    period_status[value_status] += 1
                    region_raw = person.get("REGION") or household.get("REGION", "")
                    output: dict[str, Any] = {
                        "row_id": row_id,
                        "household_observation_id": f"{period}|{key[0]}|{key[1]}",
                        "panel_dwelling_id": key[0],
                        "panel_household_id": panel_household,
                        "person_linkage_candidate_id": candidate,
                        "person_key_within_household": key[2],
                        "period": period,
                        "region_id": _region(region_raw, period),
                        "P47T_nominal": nominal,
                        "P47T_real": real,
                        "p47t_value_status": value_status,
                        "monetary_source_period": monetary[period]["source_period"],
                        "monetary_reference_period": monetary_reference_period,
                        "monetary_factor_period_to_reference": monetary[period][
                            "factor_period_to_reference"
                        ],
                        "source_release_id": entry["source_release_id"],
                        "source_row_identity": (
                            f"{entry['source_release_id']}|{key[0]}|{key[1]}|{key[2]}"
                        ),
                        "weight_pondera": person.get("PONDERA")
                        or household.get("PONDERA", ""),
                        "weight_pondiio": person.get("PONDIIO", ""),
                        "weight_pondii": person.get("PONDII", ""),
                        "weight_pondih": person.get("PONDIH") or household.get("PONDIH", ""),
                        **exception,
                    }
                    output.update({field: person.get(field, "") for field in person_union})
                    output.update(
                        {f"HH__{field}": household.get(field, "") for field in household_context}
                    )
                    writer.writerow(output)
                    panel_rows.append(
                        (
                            candidate,
                            row_id,
                            panel_household,
                            period,
                            period_index(period),
                            key[2],
                            person.get("CH04", ""),
                            person.get("CH06", ""),
                        )
                    )
                    person_count += 1
                    total_persons += 1
                if person_count != entry["individual"]["rows"]:
                    raise LongitudinalFrameError(f"{period}:person_row_accounting_mismatch")
                if len(households) != entry["household"]["rows"]:
                    raise LongitudinalFrameError(f"{period}:household_row_accounting_mismatch")
                database.executemany("INSERT INTO person_obs VALUES(?,?,?,?,?,?,?,?)", panel_rows)
                database.commit()
                coverage.append(
                    {
                        "period": period,
                        "source_release_id": entry["source_release_id"],
                        "households": len(households),
                        "persons": person_count,
                        "individual_schema_fingerprint": entry["individual"]["schema_fingerprint"],
                        "household_schema_fingerprint": entry["household"]["schema_fingerprint"],
                        "p47t_zero": period_status.get("zero", 0),
                        "p47t_positive": period_status.get("positive", 0),
                        "p47t_negative_source_code_or_value": period_status.get(
                            "negative_source_code_or_value", 0
                        ),
                        "p47t_missing_or_non_numeric": sum(
                            period_status.get(name, 0)
                            for name in (
                                "missing",
                                "non_numeric_source_value",
                                "non_finite_source_value",
                            )
                        ),
                        **{key: exception[key] for key in (
                            "exception_flags",
                            "ordinary_seasonality_eligible",
                            "ordinary_structural_time_eligible",
                        )},
                    }
                )

        _write_csv(
            staging / "coverage.csv",
            [
                "period",
                "source_release_id",
                "households",
                "persons",
                "individual_schema_fingerprint",
                "household_schema_fingerprint",
                "p47t_zero",
                "p47t_positive",
                "p47t_negative_source_code_or_value",
                "p47t_missing_or_non_numeric",
                "exception_flags",
                "ordinary_seasonality_eligible",
                "ordinary_structural_time_eligible",
            ],
            coverage,
        )
        panel_qa = _panel_audit(database, staging)
        database.close()
        (staging / "_panel.sqlite").unlink(missing_ok=True)
        qa = {
            "result": "pass",
            "period_start": START_PERIOD,
            "period_end": END_PERIOD,
            "period_count": EXPECTED_PERIOD_COUNT,
            "persons": total_persons,
            "source_person_rows_accounted": total_persons
            == sum(entry["individual"]["rows"] for entry in entries),
            "zero_income_persons_retained": status_counts.get("zero", 0),
            "p47t_value_status_counts": dict(sorted(status_counts.items())),
            "monetary_roundtrip_max_absolute_error": decimal_text(roundtrip_error),
            "forbidden_derived_fields_present": [],
            "exception_periods": EXCEPTION_PERIODS,
            "panel_audit": panel_qa,
        }
        (staging / "qa.json").write_text(canonical_json(qa), encoding="utf-8")

        payload = [
            "persons.csv",
            "coverage.csv",
            "schema_inventory.csv",
            "monetary_lineage.csv",
            "panel_links.csv",
            "panel_households.csv",
            "parent_lock.json",
            "harmonization.json",
            "qa.json",
        ]
        manifest = {
            "contract": LONGITUDINAL_CONTRACT,
            "release_id": release_id,
            "status": "source_backed_longitudinal_analysis_frame",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "coverage": {
                "period_start": START_PERIOD,
                "period_end": END_PERIOD,
                "period_count": EXPECTED_PERIOD_COUNT,
                "all_expected_periods_present": True,
            },
            "parents": {
                "eph_microdata": {
                    "artifact_type": UPSTREAM_ARTIFACT,
                    "parent_lock_schema": PARENT_LOCK_SCHEMA,
                    "parent_lock_sha256": parent_lock_sha,
                    "period_count": EXPECTED_PERIOD_COUNT,
                },
                "monetary_conversion": {
                    "artifact_type": CONVERSION_CONTRACT,
                    "release_id": conversion_manifest.get("release_id"),
                    "manifest_sha256": conversion_manifest_sha,
                    "status": conversion_manifest.get("status"),
                    "monetary_reference_id": conversion_manifest.get("monetary_reference_id"),
                },
            },
            "identity": {
                "row_id": "period|CODUSU|NRO_HOGAR|COMPONENTE",
                "household_observation_id": "period|CODUSU|NRO_HOGAR",
                "panel_dwelling_id": "CODUSU; repeated-wave dwelling candidate",
                "panel_household_id": "CODUSU|NRO_HOGAR; repeated-wave household candidate",
                "person_key_within_household": "COMPONENTE",
                "person_linkage_candidate_id": (
                    "CODUSU|NRO_HOGAR|COMPONENTE; candidate only, audited separately"
                ),
                "permanent_person_identity_claim": False,
            },
            "schema_harmonization": {
                "policy_schema": HARMONIZATION_SCHEMA,
                "policy_sha256": harmonization_sha,
                "semantic_renames": "none_without_explicit_proof",
                "optional_field_policy": "union_preserve_blank_when_absent",
                "schema_inventory": "schema_inventory.csv",
            },
            "monetary_lineage": {
                "source_field": "P47T",
                "nominal_output": "P47T_nominal",
                "real_output": "P47T_real",
                "quarter_period_mapping": "Q1->Feb,Q2->May,Q3->Aug,Q4->Nov",
                "common_reference_period": monetary_reference_period,
                "factor_composition": (
                    "factor(source_quarter_month->producer_base) / "
                    "factor(target_reference_month->producer_base)"
                ),
                "rounding": "none; decimal arithmetic",
                "zero_state_preserved": True,
                "negative_source_values": "preserved nominal; real value left blank",
                "lineage_table": "monetary_lineage.csv",
            },
            "exception_policy": {
                "periods": EXCEPTION_PERIODS,
                "ordinary_seasonality_fit": "exclude_flagged_periods",
                "ordinary_structural_time_fit": "exclude_flagged_periods",
                "shock_terms": "dedicated_effects_allowed_downstream",
                "observations_retained": True,
                "sensitivity_requirement": (
                    "report ordinary-time structural sensitivities with and without "
                    "exceptional quarters"
                ),
            },
            "field_policy": {
                "all_source_person_rows_retained": True,
                "zero_income_rows_retained": True,
                "source_person_columns_union": person_union,
                "source_household_context_columns": household_context,
                "household_context_prefix": "HH__",
                "survey_weight_fields_preserved": list(WEIGHT_FIELDS),
                "survey_weights_applied": False,
                "target_derived_geography_ranks_added": False,
                "final_welfare_model_trained": False,
            },
            "panel_audit": panel_qa,
            "artifacts": _inventory(staging, payload),
            "qa": qa,
            "limitations": [
                (
                    "Repeated-wave person links are short-horizon candidates, not permanent "
                    "person identities."
                ),
                (
                    "The observed EPH rotation does not identify a 2010-to-2026 "
                    "person-state transition."
                ),
                "Optional source fields absent in a quarter are blank rather than imputed.",
                "This EPH evidence release performs no Census scoring or welfare-model training.",
            ],
        }
        (staging / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
        _checksums(staging)
        staging.replace(destination)
        return destination
    except Exception:
        try:
            database.close()
        except sqlite3.Error:
            pass
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "LongitudinalFrameError",
    "build_longitudinal_frame",
    "build_parent_ledger",
    "expected_periods",
]
