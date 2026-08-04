import hashlib
import json
from pathlib import Path

import pytest

from eph_income.flagship_freeze import (
    canonical_hash,
    validate_annual_inputs,
    validate_inference_rows,
    validate_release,
)


ROOT = Path(__file__).resolve().parents[1]


def test_shared_annual_envelope_and_provenance_semantics():
    releases = validate_annual_inputs(ROOT)
    assert [item["year"] for item in releases] == [2022, 2023, 2024, 2025]
    for item in releases:
        manifest = item["manifest"]
        assert manifest["manifest_schema"] == "research-artifact-manifest/v1"
        assert manifest["release_status"] == "candidate"
        assert manifest["artifact_materialization_commit"] == (
            "unresolved-historical-materialization"
        )
        assert (
            manifest["manifest_attestation_commit"] != manifest["artifact_materialization_commit"]
        )


def test_approved_mode_rejects_unresolved_historical_inputs():
    with pytest.raises(ValueError, match="Approved mode rejects"):
        validate_annual_inputs(ROOT, approved=True)


def test_inference_schema_and_census_rejection():
    validate_inference_rows(["row_id", "a", "b"], ["a", "b"], mode="eph")
    with pytest.raises(ValueError, match="Census-mode"):
        validate_inference_rows(["row_id", "a", "b"], ["a", "b"], mode="census")
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_inference_rows(["row_id", "a", "unexpected"], ["a", "b"], mode="eph")


def test_release_rejects_tampered_file(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    contract = {
        "target": {"scale": "log10_ars", "plus_one": False},
        "census_sample_compatible": False,
    }
    contract_path = release / "inference_contract.json"
    contract_path.write_text(json.dumps(contract))
    manifest = {
        "manifest_schema": "research-artifact-manifest/v1",
        "artifact_type": "research.eph-income-model/v1",
        "release_status": "candidate",
        "files": [
            {
                "path": contract_path.name,
                "bytes": contract_path.stat().st_size,
                "sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            }
        ],
    }
    (release / "manifest.json").write_text(json.dumps(manifest))
    validate_release(ROOT, release)
    contract_path.write_text("tampered")
    with pytest.raises(ValueError, match="Tampered"):
        validate_release(ROOT, release)


def test_canonical_identity_ignores_mapping_order():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
