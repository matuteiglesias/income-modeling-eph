"""Build a neutral, source-backed EPH analysis-frame release.

The producer consumes one exact parent pinned by ``eph_release_intake`` and
publishes source-faithful household and person tables. It deliberately does not
apply a modeling cohort, construct an income target, create Census-shaped
aliases, rank geography by income, or apply survey weights.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CONTRACT = "research.eph-analysis-frame@1"
PERSON_KEY = ("CODUSU", "NRO_HOGAR", "COMPONENTE")
HOUSEHOLD_KEY = ("CODUSU", "NRO_HOGAR")
SURVEY_DESIGN_FIELDS = ("PONDERA", "PONDIIO", "PONDII", "PONDIH")
FORBIDDEN_DERIVED_FIELDS = {"logP47T", "AGLO_rk", "Reg_rk"}


class AnalysisFrameError(ValueError):
    """Raised when a neutral frame cannot be published safely."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, reason: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisFrameError(reason) from exc
    if not isinstance(value, dict):
        raise AnalysisFrameError(reason)
    return value


def _dialect(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return csv.excel


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, dialect=_dialect(path))
        fields = list(reader.fieldnames or [])
        rows = [
            {field: (row.get(field) or "").strip() for field in fields}
            for row in reader
        ]
    if not fields:
        raise AnalysisFrameError(f"empty_schema:{path.name}")
    return fields, rows


def _write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _role_path(parent_root: Path, output_manifest: dict, role: str) -> Path:
    matches = [
        item.get("file")
        for item in output_manifest.get("files", [])
        if isinstance(item, dict) and item.get("role") == role
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise AnalysisFrameError(f"expected_exactly_one_{role}_file")
    path = parent_root / matches[0]
    if not path.is_file():
        raise AnalysisFrameError(f"missing_{role}_file")
    return path


def _require(fields: list[str], required: tuple[str, ...], role: str) -> None:
    missing = [field for field in required if field not in fields]
    if missing:
        raise AnalysisFrameError(
            f"{role}:missing_required_fields:{','.join(missing)}"
        )


def _assert_unique(rows: list[dict[str, str]], key: tuple[str, ...], role: str) -> None:
    keys = [tuple(row[field] for field in key) for row in rows]
    if any(any(part == "" for part in value) for value in keys):
        raise AnalysisFrameError(f"{role}:empty_identity")
    duplicates = [value for value, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise AnalysisFrameError(f"{role}:duplicate_identity:{duplicates[:3]}")


def build_analysis_frame(parent_root: Path, output_root: Path) -> Path:
    """Publish one immutable neutral EPH frame from one exact pinned quarter."""
    parent_root = Path(parent_root).resolve()
    output_root = Path(output_root).resolve()
    lock_path = parent_root / "parent_lock.json"
    manifest_path = parent_root / "output-manifest.json"
    parent_lock = _read_json(lock_path, "missing_or_invalid_parent_lock")
    output_manifest = _read_json(
        manifest_path, "missing_or_invalid_parent_output_manifest"
    )

    if parent_lock.get("schema") != "eph-upstream-parent-lock/v1":
        raise AnalysisFrameError("unexpected_parent_lock_schema")
    release_id = parent_lock.get("release_id")
    if not isinstance(release_id, str) or output_manifest.get("release_id") != release_id:
        raise AnalysisFrameError("parent_release_identity_mismatch")

    household_path = _role_path(parent_root, output_manifest, "household")
    person_path = _role_path(parent_root, output_manifest, "individual")
    household_fields, households = _read_rows(household_path)
    person_fields, persons = _read_rows(person_path)

    _require(household_fields, HOUSEHOLD_KEY, "household")
    _require(person_fields, PERSON_KEY, "individual")
    _assert_unique(households, HOUSEHOLD_KEY, "household")
    _assert_unique(persons, PERSON_KEY, "individual")

    household_keys = {(row["CODUSU"], row["NRO_HOGAR"]) for row in households}
    orphan_person_keys = sorted(
        {
            (row["CODUSU"], row["NRO_HOGAR"])
            for row in persons
            if (row["CODUSU"], row["NRO_HOGAR"]) not in household_keys
        }
    )
    if orphan_person_keys:
        raise AnalysisFrameError(
            f"person_household_cardinality_failure:{orphan_person_keys[:3]}"
        )

    forbidden_present = sorted(
        (set(household_fields) | set(person_fields)) & FORBIDDEN_DERIVED_FIELDS
    )
    if forbidden_present:
        raise AnalysisFrameError(
            "neutral_frame_contains_derived_fields:" + ",".join(forbidden_present)
        )

    parent_identity = {
        "release_id": release_id,
        "parent_lock_sha256": _sha256(lock_path),
        "producer_manifest_sha256": _sha256(manifest_path),
        "source_archive_sha256": parent_lock.get("source", {}).get(
            "source_archive_sha256"
        ),
        "period": parent_lock.get("period"),
    }
    identity_payload = {
        "contract": CONTRACT,
        "parent": parent_identity,
        "household_source_sha256": _sha256(household_path),
        "person_source_sha256": _sha256(person_path),
    }
    config_hash = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    frame_release_id = f"eph-analysis-frame-{release_id}-{config_hash[:12]}"
    destination = output_root / frame_release_id
    if destination.exists():
        raise AnalysisFrameError(f"immutable_release_exists:{destination}")

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{frame_release_id}.", dir=output_root))
    try:
        _write_rows(staging / "households.csv", household_fields, households)
        _write_rows(staging / "persons.csv", person_fields, persons)
        qa = {
            "households": len(households),
            "persons": len(persons),
            "household_key": list(HOUSEHOLD_KEY),
            "person_key": list(PERSON_KEY),
            "household_key_unique": True,
            "person_key_unique": True,
            "person_to_household_cardinality": "many_to_one_validated",
            "orphan_person_households": 0,
            "survey_design_fields": {
                "household": [
                    field for field in SURVEY_DESIGN_FIELDS if field in household_fields
                ],
                "individual": [
                    field for field in SURVEY_DESIGN_FIELDS if field in person_fields
                ],
            },
            "forbidden_derived_fields_present": [],
        }
        (staging / "qa.json").write_text(_canonical_json(qa), encoding="utf-8")

        artifacts = {}
        for filename in ("households.csv", "persons.csv", "qa.json"):
            path = staging / filename
            artifacts[filename] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        release_manifest = {
            "contract": CONTRACT,
            "release_id": frame_release_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent": parent_identity,
            "artifacts": artifacts,
            "identity": {
                "household_key": list(HOUSEHOLD_KEY),
                "person_key": list(PERSON_KEY),
                "person_to_household_cardinality": "many_to_one",
            },
            "field_policy": {
                "source_semantics": "native_eph",
                "household_columns": household_fields,
                "person_columns": person_fields,
                "survey_design_fields_preserved": list(SURVEY_DESIGN_FIELDS),
                "survey_weights_applied": False,
                "census_shaped_aliases_added": False,
                "target_or_cohort_fields_added": False,
                "target_derived_geography_ranks_added": False,
            },
            "monetary_lineage": {
                "mode": "source_native_values_no_transform",
                "ipc_release": None,
                "deflation_or_rebasing_applied": False,
            },
            "scientific_scope": {
                "population_claim": "none",
                "note": (
                    "This is a neutral observed-EPH frame. Downstream starter studies "
                    "may preserve but not use EPH expansion fields and must describe "
                    "unweighted results as sample-conditional."
                ),
            },
        }
        (staging / "manifest.json").write_text(
            _canonical_json(release_manifest), encoding="utf-8"
        )
        staging.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
