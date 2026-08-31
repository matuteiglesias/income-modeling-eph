"""Verify and pin exact durable EPH source releases.

This module intentionally stops at the transport boundary.  It does not decide
how quarter releases should be transformed into a neutral annual EPH analysis
frame; that scientific work remains governed by issues #24/#29.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

DISCOVERY_SCHEMA = "ecosystem-release-discovery/v1"
EXPECTED_PRODUCER = "matuteiglesias/microdatos-EPH-INDEC"
EXPECTED_ARTIFACT = "publicdata.eph-microdata@1"


class IntakeError(ValueError):
    """Raised when an upstream EPH release cannot be safely pinned."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _safe_members(zf: zipfile.ZipFile, release_id: str):
    expected_root = release_id + "/"
    seen: set[str] = set()
    for info in zf.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise IntakeError(f"unsafe_archive_member:{info.filename}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode not in (0, 0o100000):
            raise IntakeError(f"unsupported_archive_member:{info.filename}")
        if info.is_dir():
            continue
        if not info.filename.startswith(expected_root):
            raise IntakeError(f"unexpected_archive_root:{info.filename}")
        if info.filename in seen:
            raise IntakeError(f"duplicate_archive_member:{info.filename}")
        seen.add(info.filename)
        yield info


def _read_json(path: Path, reason: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntakeError(reason) from exc
    if not isinstance(value, dict):
        raise IntakeError(reason)
    return value


def verify_and_pin(
    discovery_path: Path,
    asset_path: Path,
    output_root: Path,
    *,
    selection_mode: str = "convergence",
    transport_tag: str | None = None,
) -> Path:
    """Copy one exact producer release into consumer custody and write its parent lock."""
    discovery_path = Path(discovery_path).resolve()
    asset_path = Path(asset_path).resolve()
    output_root = Path(output_root).resolve()
    discovery = _read_json(discovery_path, "invalid_discovery_json")

    if discovery.get("schema") != DISCOVERY_SCHEMA:
        raise IntakeError("unexpected_discovery_schema")
    if discovery.get("producer") != EXPECTED_PRODUCER:
        raise IntakeError("unexpected_producer")
    if discovery.get("artifact_type") != EXPECTED_ARTIFACT:
        raise IntakeError("unexpected_artifact_type")
    if discovery.get("status") != "candidate":
        raise IntakeError("unexpected_release_status")
    release_id = discovery.get("release_id")
    if not isinstance(release_id, str) or not release_id.startswith("eph-"):
        raise IntakeError("invalid_release_id")
    period = discovery.get("period") or {}
    year, quarter = period.get("year"), period.get("quarter")
    if not isinstance(year, int) or quarter not in {"Q1", "Q2", "Q3", "Q4"}:
        raise IntakeError("invalid_release_period")

    github_release = discovery.get("github_release") or {}
    expected_asset = github_release.get("asset_name")
    expected_asset_sha = github_release.get("asset_sha256")
    expected_manifest_sha = github_release.get("manifest_sha256")
    declared_tag = github_release.get("tag")
    if not all(
        isinstance(value, str) and value
        for value in (expected_asset, expected_asset_sha, expected_manifest_sha, declared_tag)
    ):
        raise IntakeError("incomplete_github_release_locator")
    if asset_path.name != expected_asset:
        raise IntakeError("asset_name_mismatch")
    if transport_tag is not None and transport_tag != declared_tag:
        raise IntakeError("transport_tag_mismatch")
    if sha256_file(asset_path) != expected_asset_sha:
        raise IntakeError("asset_checksum_mismatch")

    destination = output_root / release_id
    if destination.exists():
        existing_lock = destination / "parent_lock.json"
        if not existing_lock.is_file():
            raise IntakeError("existing_parent_missing_lock")
        existing = _read_json(existing_lock, "invalid_existing_parent_lock")
        if (
            existing.get("release_id") == release_id
            and existing.get("transport", {}).get("asset_sha256") == expected_asset_sha
            and existing.get("producer_manifest_sha256") == expected_manifest_sha
        ):
            return destination
        raise IntakeError("existing_parent_identity_mismatch")

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release_id}.", dir=output_root))
    try:
        payload_root = staging / release_id
        with zipfile.ZipFile(asset_path) as zf:
            members = list(_safe_members(zf, release_id))
            if not members:
                raise IntakeError("empty_release_archive")
            for info in members:
                relative = PurePosixPath(info.filename).relative_to(release_id)
                target = payload_root.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)

        output_manifest_path = payload_root / "output-manifest.json"
        source_manifest_path = payload_root / "_source" / "source-manifest.json"
        output_manifest = _read_json(output_manifest_path, "missing_or_invalid_output_manifest")
        source_manifest = _read_json(source_manifest_path, "missing_or_invalid_source_manifest")

        if output_manifest.get("release_id") != release_id:
            raise IntakeError("inner_release_id_mismatch")
        if (
            output_manifest.get("requested_year") != year
            or output_manifest.get("requested_quarter") != quarter
        ):
            raise IntakeError("inner_period_mismatch")
        if sha256_file(output_manifest_path) != expected_manifest_sha:
            raise IntakeError("producer_manifest_checksum_mismatch")

        source_discovery = discovery.get("source") or {}
        source_manifest_sha = source_discovery.get("source_manifest_sha256")
        source_archive_sha = source_discovery.get("source_archive_sha256")
        source_name = source_discovery.get("original_filename")
        if not all(
            isinstance(value, str) and value
            for value in (source_manifest_sha, source_archive_sha, source_name)
        ):
            raise IntakeError("incomplete_source_identity")
        if Path(source_name).name != source_name:
            raise IntakeError("invalid_source_filename")
        if sha256_file(source_manifest_path) != source_manifest_sha:
            raise IntakeError("source_manifest_checksum_mismatch")
        if output_manifest.get("source_manifest_sha256") != source_manifest_sha:
            raise IntakeError("output_source_manifest_mismatch")
        if output_manifest.get("source_archive_sha256") != source_archive_sha:
            raise IntakeError("output_source_archive_mismatch")
        if source_manifest.get("sha256") != source_archive_sha:
            raise IntakeError("source_manifest_archive_mismatch")
        if (
            source_manifest.get("requested_year") != year
            or source_manifest.get("requested_quarter") != quarter
        ):
            raise IntakeError("source_period_mismatch")
        if source_manifest.get("original_filename") != source_name:
            raise IntakeError("source_filename_mismatch")
        source_archive = payload_root / "_source" / source_name
        if not source_archive.is_file() or sha256_file(source_archive) != source_archive_sha:
            raise IntakeError("source_archive_checksum_mismatch")

        for item in output_manifest.get("files", []):
            if not isinstance(item, dict) or not isinstance(item.get("file"), str):
                raise IntakeError("invalid_output_file_inventory")
            relative = PurePosixPath(item["file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise IntakeError("unsafe_output_file_path")
            path = payload_root.joinpath(*relative.parts)
            if not path.is_file():
                raise IntakeError(f"missing_output_file:{item['file']}")
            if sha256_file(path) != item.get("sha256"):
                raise IntakeError(f"output_file_checksum_mismatch:{item['file']}")

        lock = {
            "schema": "eph-upstream-parent-lock/v1",
            "producer": EXPECTED_PRODUCER,
            "artifact_type": EXPECTED_ARTIFACT,
            "release_id": release_id,
            "upstream_status": discovery["status"],
            "period": {"year": year, "quarter": quarter},
            "selection": {
                "mode": selection_mode,
                "selected_tag": declared_tag,
            },
            "transport": {
                "locator_family": "github_release",
                "tag": declared_tag,
                "asset_name": expected_asset,
                "asset_sha256": expected_asset_sha,
                "discovery_sha256": sha256_file(discovery_path),
            },
            "producer_manifest_sha256": expected_manifest_sha,
            "source": {
                "original_filename": source_name,
                "source_archive_sha256": source_archive_sha,
                "source_manifest_sha256": source_manifest_sha,
            },
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "local_release_root": release_id,
            "limitations": [
                "This lock proves exact upstream EPH transport/custody only.",
                (
                    "It does not approve an annual preprocessing method, modeling cohort, "
                    "monetary transform, or downstream scientific use."
                ),
            ],
        }
        (payload_root / "parent_lock.json").write_text(
            canonical_json(lock), encoding="utf-8"
        )
        os.replace(payload_root, destination)
        staging.rmdir()
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
