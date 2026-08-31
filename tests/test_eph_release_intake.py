import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import pytest

from eph_income.eph_release_intake import IntakeError, verify_and_pin


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def make_fixture(root: Path, *, unsafe_member: str | None = None):
    release_id = "eph-2026-q1-fixture123456"
    source_name = "EPH_usu_1_Trim_2026_txt.zip"
    source_bytes = b"exact-official-source-fixture"
    source_manifest = {
        "schema_version": 2,
        "publisher": "INDEC",
        "dataset_family": "EPH",
        "requested_year": 2026,
        "requested_quarter": "Q1",
        "resolved_source_url": "https://example.invalid/EPH_usu_1_Trim_2026_txt.zip",
        "original_filename": source_name,
        "bytes": len(source_bytes),
        "sha256": sha256_bytes(source_bytes),
        "retrieval_status": "success",
        "candidate_selection": {
            "rule": "preferred_format_class_then_exactly_one",
            "format_preference": ["text", "dbf", "generic"],
            "selected": "https://example.invalid/EPH_usu_1_Trim_2026_txt.zip",
            "selected_format_class": "text",
        },
        "tool_version": "1.0.0",
    }
    source_manifest_bytes = canonical(source_manifest)
    household_bytes = b"CODUSU;NRO_HOGAR\nA;1\n"
    individual_bytes = b"CODUSU;NRO_HOGAR;COMPONENTE\nA;1;1\n"
    output_manifest = {
        "schema_version": 1,
        "release_id": release_id,
        "extraction_contract_version": "eph-zip-v2",
        "source_archive_sha256": sha256_bytes(source_bytes),
        "requested_year": 2026,
        "requested_quarter": "Q1",
        "source_manifest_sha256": sha256_bytes(source_manifest_bytes),
        "files": [
            {
                "role": "household",
                "period": "2026-Q1",
                "file": "household/usu_hogar_fixture.txt",
                "sha256": sha256_bytes(household_bytes),
            },
            {
                "role": "individual",
                "period": "2026-Q1",
                "file": "individual/usu_individual_fixture.txt",
                "sha256": sha256_bytes(individual_bytes),
            },
        ],
    }
    output_manifest_bytes = canonical(output_manifest)
    asset = root / f"{release_id}.zip"
    with zipfile.ZipFile(asset, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{release_id}/output-manifest.json", output_manifest_bytes)
        zf.writestr(f"{release_id}/household/usu_hogar_fixture.txt", household_bytes)
        zf.writestr(f"{release_id}/individual/usu_individual_fixture.txt", individual_bytes)
        zf.writestr(f"{release_id}/_source/source-manifest.json", source_manifest_bytes)
        zf.writestr(f"{release_id}/_source/{source_name}", source_bytes)
        if unsafe_member:
            zf.writestr(unsafe_member, b"escape")
    tag = f"candidate-{release_id}"
    discovery = {
        "schema": "ecosystem-release-discovery/v1",
        "producer": "matuteiglesias/microdatos-EPH-INDEC",
        "artifact_type": "publicdata.eph-microdata@1",
        "release_id": release_id,
        "status": "candidate",
        "period": {"year": 2026, "quarter": "Q1"},
        "extraction_contract_version": "eph-zip-v2",
        "source": {
            "publisher": "INDEC",
            "dataset_family": "EPH",
            "original_filename": source_name,
            "source_archive_sha256": sha256_bytes(source_bytes),
            "source_manifest_sha256": sha256_bytes(source_manifest_bytes),
        },
        "github_release": {
            "tag": tag,
            "asset_name": asset.name,
            "asset_sha256": sha256_file(asset),
            "manifest_sha256": sha256_bytes(output_manifest_bytes),
        },
    }
    discovery_path = root / "discovery.json"
    discovery_path.write_bytes(canonical(discovery))
    return discovery_path, asset, release_id, tag


def test_verify_and_pin_copies_exact_parent_without_preprocessing():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        discovery, asset, release_id, tag = make_fixture(root)
        pinned = verify_and_pin(discovery, asset, root / "parents", transport_tag=tag)
        assert pinned.name == release_id
        assert (pinned / "output-manifest.json").is_file()
        assert (pinned / "_source" / "source-manifest.json").is_file()
        lock = json.loads((pinned / "parent_lock.json").read_text())
        assert lock["release_id"] == release_id
        assert lock["period"] == {"year": 2026, "quarter": "Q1"}
        assert lock["selection"]["mode"] == "convergence"
        assert "annual_release" not in lock
        assert "transformation" not in lock
        assert any(
            "does not approve an annual preprocessing method" in limitation.lower()
            for limitation in lock["limitations"]
        )


def test_same_exact_parent_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        discovery, asset, _, tag = make_fixture(root)
        first = verify_and_pin(discovery, asset, root / "parents", transport_tag=tag)
        lock_before = (first / "parent_lock.json").read_bytes()
        second = verify_and_pin(discovery, asset, root / "parents", transport_tag=tag)
        assert first == second
        assert (second / "parent_lock.json").read_bytes() == lock_before


def test_asset_checksum_mismatch_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        discovery, asset, _, tag = make_fixture(root)
        asset.write_bytes(asset.read_bytes() + b"drift")
        with pytest.raises(IntakeError, match="asset_checksum_mismatch"):
            verify_and_pin(discovery, asset, root / "parents", transport_tag=tag)
        assert not (root / "parents").exists()


def test_archive_path_escape_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        discovery, asset, _, tag = make_fixture(root, unsafe_member="../escape.txt")
        # Rebind discovery to the exact bytes so this specifically tests archive safety.
        meta = json.loads(discovery.read_text())
        meta["github_release"]["asset_sha256"] = sha256_file(asset)
        discovery.write_bytes(canonical(meta))
        with pytest.raises(IntakeError, match="unsafe_archive_member"):
            verify_and_pin(discovery, asset, root / "parents", transport_tag=tag)
        assert not (root / "escape.txt").exists()


def test_inner_manifest_identity_mismatch_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        discovery, asset, release_id, tag = make_fixture(root)
        rebuilt = root / "rebuilt.zip"
        with zipfile.ZipFile(asset) as source, zipfile.ZipFile(rebuilt, "w") as target:
            for info in source.infolist():
                data = source.read(info)
                if info.filename == f"{release_id}/output-manifest.json":
                    manifest = json.loads(data)
                    manifest["release_id"] = "eph-other"
                    data = canonical(manifest)
                target.writestr(info.filename, data)
        rebuilt.replace(asset)
        meta = json.loads(discovery.read_text())
        meta["github_release"]["asset_sha256"] = sha256_file(asset)
        meta["github_release"]["manifest_sha256"] = "0" * 64
        discovery.write_bytes(canonical(meta))
        with pytest.raises(IntakeError, match="inner_release_id_mismatch"):
            verify_and_pin(discovery, asset, root / "parents", transport_tag=tag)
