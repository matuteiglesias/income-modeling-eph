"""Exact upstream locks and monetary inputs for the longitudinal EPH lane."""
from __future__ import annotations

import csv
import hashlib
import json
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path, PurePosixPath
from typing import Any

LONGITUDINAL_CONTRACT = "research.eph-longitudinal-analysis-frame/v1"
PARENT_LOCK_SCHEMA = "eph-longitudinal-parent-lock/v1"
UPSTREAM_PARENT_SCHEMA = "eph-upstream-parent-lock/v1"
UPSTREAM_ARTIFACT = "publicdata.eph-microdata@1"
CONVERSION_CONTRACT = "research.argentina-monetary-conversion/v1"
HARMONIZATION_SCHEMA = "eph-longitudinal-harmonization/v1"
START_PERIOD = "2017-Q1"
END_PERIOD = "2026-Q1"
EXPECTED_PERIOD_COUNT = 37
QUARTER_REFERENCE_MONTH = {1: 2, 2: 5, 3: 8, 4: 11}
FORBIDDEN_SOURCE_FIELDS = {"AGLO_rk", "Reg_rk", "logP47T"}


class LongitudinalFrameError(ValueError):
    """Raised when exact longitudinal inputs or output invariants fail."""


def canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LongitudinalFrameError(reason) from exc
    if not isinstance(value, dict):
        raise LongitudinalFrameError(reason)
    return value


def parse_period(period: str) -> tuple[int, int]:
    try:
        year_text, quarter_text = period.split("-Q", 1)
        year, quarter = int(year_text), int(quarter_text)
    except (ValueError, AttributeError) as exc:
        raise LongitudinalFrameError(f"invalid_period:{period}") from exc
    if quarter not in {1, 2, 3, 4}:
        raise LongitudinalFrameError(f"invalid_period:{period}")
    return year, quarter


def period_index(period: str) -> int:
    year, quarter = parse_period(period)
    return year * 4 + quarter - 1


def expected_periods(start: str = START_PERIOD, end: str = END_PERIOD) -> list[str]:
    first, last = period_index(start), period_index(end)
    if last < first:
        raise LongitudinalFrameError("period_range_reversed")
    output = []
    for index in range(first, last + 1):
        year, q0 = divmod(index, 4)
        output.append(f"{year:04d}-Q{q0 + 1}")
    return output


def _dialect(path: Path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return csv.excel


def table_profile(path: Path) -> tuple[list[str], int, str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, dialect=_dialect(path))
        try:
            fields = [cell.strip() for cell in next(reader)]
        except StopIteration as exc:
            raise LongitudinalFrameError(f"empty_table:{path.name}") from exc
        rows = sum(1 for _ in reader)
    if not fields or any(not field for field in fields) or len(fields) != len(set(fields)):
        raise LongitudinalFrameError(f"invalid_table_header:{path.name}")
    fingerprint = hashlib.sha256(
        json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return fields, rows, fingerprint


def read_rows(path: Path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, dialect=_dialect(path))
        for row in reader:
            yield {key: (value or "").strip() for key, value in row.items()}


def _role_item(manifest: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [
        item
        for item in manifest.get("files", [])
        if isinstance(item, dict) and item.get("role") == role
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("file"), str):
        raise LongitudinalFrameError(f"expected_exactly_one_{role}_file")
    return matches[0]


def _role_path(root: Path, item: dict[str, Any], role: str) -> Path:
    relative = PurePosixPath(item["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise LongitudinalFrameError(f"unsafe_{role}_file_path")
    path = root.joinpath(*relative.parts)
    declared = item.get("sha256")
    if not path.is_file() or not isinstance(declared, str) or not declared:
        raise LongitudinalFrameError(f"missing_{role}_file_or_checksum")
    if sha256(path) != declared:
        raise LongitudinalFrameError(f"{role}_file_checksum_mismatch")
    return path


def inspect_parent(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    lock_path, manifest_path = root / "parent_lock.json", root / "output-manifest.json"
    lock = read_json(lock_path, "missing_or_invalid_parent_lock")
    manifest = read_json(manifest_path, "missing_or_invalid_parent_output_manifest")
    if lock.get("schema") != UPSTREAM_PARENT_SCHEMA:
        raise LongitudinalFrameError("unexpected_upstream_parent_lock_schema")
    if lock.get("artifact_type") not in {None, UPSTREAM_ARTIFACT}:
        raise LongitudinalFrameError("unexpected_upstream_artifact_type")
    release_id = lock.get("release_id")
    if not isinstance(release_id, str) or manifest.get("release_id") != release_id:
        raise LongitudinalFrameError("upstream_release_identity_mismatch")
    declared_manifest = lock.get("producer_manifest_sha256")
    if not isinstance(declared_manifest, str) or sha256(manifest_path) != declared_manifest:
        raise LongitudinalFrameError("upstream_manifest_checksum_mismatch")
    meta = lock.get("period") or {}
    try:
        year = int(meta["year"])
        quarter = int(str(meta["quarter"]).upper().removeprefix("Q"))
    except (KeyError, TypeError, ValueError) as exc:
        raise LongitudinalFrameError("invalid_upstream_period") from exc
    period = f"{year:04d}-Q{quarter}"
    if period not in expected_periods():
        parse_period(period)
    source = lock.get("source") or {}
    source_manifest = source.get("source_manifest_sha256")
    source_archive = source.get("source_archive_sha256")
    if not all(isinstance(value, str) and value for value in (source_manifest, source_archive)):
        raise LongitudinalFrameError("incomplete_source_checksums")
    tables: dict[str, Any] = {}
    for role in ("household", "individual"):
        item = _role_item(manifest, role)
        path = _role_path(root, item, role)
        columns, rows, fingerprint = table_profile(path)
        tables[role] = {
            "file": item["file"],
            "sha256": sha256(path),
            "rows": rows,
            "schema_fingerprint": fingerprint,
            "columns": columns,
        }
    return {
        "period": period,
        "year": year,
        "quarter": quarter,
        "source_release_id": release_id,
        "parent_lock_sha256": sha256(lock_path),
        "producer_manifest_sha256": sha256(manifest_path),
        "source_manifest_sha256": source_manifest,
        "source_archive_sha256": source_archive,
        **tables,
        "status": "verified",
    }


def build_parent_ledger(pinned_root: Path, output_path: Path) -> Path:
    """Lock exactly the 37 pinned parents; never substitute a nearby quarter."""
    pinned_root, output_path = Path(pinned_root).resolve(), Path(output_path).resolve()
    wanted = expected_periods()
    by_period: dict[str, dict[str, Any]] = {}
    if not pinned_root.is_dir():
        raise LongitudinalFrameError("pinned_parent_root_missing")
    for candidate in sorted(path for path in pinned_root.iterdir() if path.is_dir()):
        if not (candidate / "parent_lock.json").is_file():
            continue
        entry = inspect_parent(candidate)
        if entry["period"] not in wanted:
            continue
        if entry["period"] in by_period:
            raise LongitudinalFrameError(f"duplicate_parent_period:{entry['period']}")
        by_period[entry["period"]] = entry
    missing = [period for period in wanted if period not in by_period]
    if missing:
        raise LongitudinalFrameError("missing_parent_periods:" + ",".join(missing))
    ledger = {
        "schema": PARENT_LOCK_SCHEMA,
        "artifact_type": UPSTREAM_ARTIFACT,
        "period_start": START_PERIOD,
        "period_end": END_PERIOD,
        "period_count": EXPECTED_PERIOD_COUNT,
        "selection_policy": "exact_period_only_no_nearby_substitution",
        "entries": [by_period[period] for period in wanted],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if read_json(output_path, "invalid_existing_parent_lock") == ledger:
            return output_path
        raise LongitudinalFrameError("immutable_parent_lock_exists_with_different_content")
    output_path.write_text(canonical_json(ledger), encoding="utf-8")
    return output_path


def load_harmonization(path: Path) -> dict[str, Any]:
    policy = read_json(path, "missing_or_invalid_harmonization_policy")
    if policy.get("schema") != HARMONIZATION_SCHEMA:
        raise LongitudinalFrameError("unexpected_harmonization_schema")
    if policy.get("semantic_rename_policy") != "no_unproven_renames":
        raise LongitudinalFrameError("unsafe_harmonization_rename_policy")
    if policy.get("optional_field_policy") != "union_preserve_blank_when_absent":
        raise LongitudinalFrameError("unsafe_optional_field_policy")
    return policy


def _validate_fields(fields: list[str], policy: dict[str, Any], role: str, period: str) -> None:
    required = set(policy.get(f"required_{role}_fields") or [])
    missing = sorted(required - set(fields))
    if missing:
        raise LongitudinalFrameError(
            f"{period}:{role}:missing_required_fields:{','.join(missing)}"
        )
    if role == "person":
        for group in policy.get("required_person_any_of") or []:
            if not any(field in fields for field in group):
                raise LongitudinalFrameError(
                    f"{period}:person:missing_required_any_of:{'|'.join(group)}"
                )
    forbidden = sorted(set(fields) & FORBIDDEN_SOURCE_FIELDS)
    if forbidden:
        raise LongitudinalFrameError(
            f"{period}:{role}:forbidden_derived_fields:{','.join(forbidden)}"
        )


def validate_parent_ledger(
    ledger_path: Path, pinned_root: Path, policy: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ledger = read_json(ledger_path, "missing_or_invalid_longitudinal_parent_lock")
    entries = ledger.get("entries")
    if (
        ledger.get("schema") != PARENT_LOCK_SCHEMA
        or ledger.get("artifact_type") != UPSTREAM_ARTIFACT
        or ledger.get("period_start") != START_PERIOD
        or ledger.get("period_end") != END_PERIOD
        or ledger.get("period_count") != EXPECTED_PERIOD_COUNT
        or not isinstance(entries, list)
        or len(entries) != EXPECTED_PERIOD_COUNT
        or [entry.get("period") for entry in entries] != expected_periods()
    ):
        raise LongitudinalFrameError("invalid_longitudinal_parent_lock_contract")
    verified = []
    for locked in entries:
        actual = inspect_parent(Path(pinned_root) / locked["source_release_id"])
        if actual != locked:
            raise LongitudinalFrameError(f"parent_lock_drift:{locked.get('period')}")
        _validate_fields(actual["individual"]["columns"], policy, "person", actual["period"])
        _validate_fields(actual["household"]["columns"], policy, "household", actual["period"])
        verified.append(actual)
    return ledger, verified


def _manifest_hash(manifest: dict[str, Any], filename: str) -> str:
    for item in manifest.get("files", []):
        if isinstance(item, dict) and (item.get("path") or item.get("file")) == filename:
            value = item.get("sha256")
            if isinstance(value, str) and value:
                return value
    raise LongitudinalFrameError(f"conversion_manifest_missing_file_hash:{filename}")


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _decimal(value: str, reason: str) -> Decimal:
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise LongitudinalFrameError(reason) from exc
    if not result.is_finite() or result <= 0:
        raise LongitudinalFrameError(reason)
    return result


def decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def build_monetary_plan(
    conversion_root: Path,
    reference_period: str,
    *,
    require_approved: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    root = Path(conversion_root).resolve()
    manifest = read_json(root / "manifest.json", "missing_or_invalid_conversion_manifest")
    if manifest.get("schema") != "research-artifact-manifest/v1":
        raise LongitudinalFrameError("unexpected_conversion_manifest_schema")
    if manifest.get("artifact_type") != CONVERSION_CONTRACT:
        raise LongitudinalFrameError("unexpected_conversion_artifact_type")
    status = manifest.get("status")
    if require_approved and status != "approved":
        raise LongitudinalFrameError(f"conversion_release_not_approved:{status}")
    if status not in {"candidate", "reviewed", "approved"}:
        raise LongitudinalFrameError(f"invalid_conversion_release_status:{status}")
    factors_path = root / "monthly_conversion_factors.csv"
    if not factors_path.is_file() or sha256(factors_path) != _manifest_hash(
        manifest, factors_path.name
    ):
        raise LongitudinalFrameError("conversion_factor_hash_mismatch")
    with factors_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_period = {row["period"]: row for row in rows}
    if len(by_period) != len(rows):
        raise LongitudinalFrameError("duplicate_conversion_period")
    target = by_period.get(reference_period)
    if target is None:
        raise LongitudinalFrameError(f"conversion_reference_period_missing:{reference_period}")
    if not _truthy(target.get("approved_mode_eligible")):
        raise LongitudinalFrameError(
            f"conversion_reference_period_not_eligible:{reference_period}"
        )
    target_factor = _decimal(
        target.get("factor_period_to_reference", ""), "invalid_reference_conversion_factor"
    )
    producer_base = target.get("reference_period")
    plan = {}
    for period in expected_periods():
        year, quarter = parse_period(period)
        source_period = f"{year:04d}-{QUARTER_REFERENCE_MONTH[quarter]:02d}-01"
        row = by_period.get(source_period)
        if row is None:
            raise LongitudinalFrameError(f"conversion_period_missing:{source_period}")
        if row.get("reference_period") != producer_base:
            raise LongitudinalFrameError("conversion_factor_base_reference_mismatch")
        if not _truthy(row.get("approved_mode_eligible")):
            raise LongitudinalFrameError(
                f"conversion_period_not_approved_mode_eligible:{source_period}"
            )
        source_factor = _decimal(
            row.get("factor_period_to_reference", ""),
            f"invalid_period_conversion_factor:{source_period}",
        )
        with localcontext() as context:
            context.prec = 40
            factor = source_factor / target_factor
        plan[period] = {
            "source_period": source_period,
            "reference_period": reference_period,
            "producer_base_reference_period": str(producer_base),
            "factor_period_to_reference": decimal_text(factor),
            "source_coverage_class": str(row.get("coverage_class", "")),
            "reference_coverage_class": str(target.get("coverage_class", "")),
            "approved_mode_eligible": "true",
        }
    return manifest, plan
