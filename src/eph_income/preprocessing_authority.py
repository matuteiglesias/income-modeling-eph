"""Release validation for the repository-owned annual EPH input boundary.

This module does not reconstruct unavailable raw EPH releases.  It makes the
currently materialized annual inputs auditable and provides a bounded synthetic
preprocessing fixture for the merge/normalization contract.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

MANIFEST_SCHEMA_VERSION = "1.0"
SHARED_MANIFEST_SCHEMA = "research-artifact-manifest/v1"
ANNUAL_RELEASE_ID = "artifact:research.eph-annual-preprocessed@1"
PROVISIONAL_SOURCE_RELEASE = "artifact:publicdata.eph-microdata@1-provisional"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_lineage(path: Path) -> dict[str, Any]:
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or not isinstance(registry.get("columns"), dict):
        raise ValueError("Lineage registry must use schema_version 1 and define columns.")
    return registry


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _git_commit_timestamp(root: Path, commit: str) -> str:
    result = subprocess.run(
        ["git", "show", "-s", "--format=%cI", commit],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def build_manifest(csv_path: Path, lineage_path: Path, root: Path) -> dict[str, Any]:
    """Describe an annual CSV without embedding a machine-local absolute path."""

    frame = pd.read_csv(csv_path)
    lineage = load_lineage(lineage_path)
    missing_lineage = sorted(set(frame.columns) - set(lineage["columns"]))
    if missing_lineage:
        raise ValueError(f"Annual input has columns absent from lineage: {missing_lineage}")
    relpath = csv_path.resolve().relative_to(root.resolve()).as_posix()
    years = [
        int(value)
        for value in sorted(pd.to_numeric(frame["ANO4"], errors="coerce").dropna().unique())
    ]
    quarters = [
        int(value)
        for value in sorted(pd.to_numeric(frame["TRIMESTRE"], errors="coerce").dropna().unique())
    ]
    schema = [{"name": c, "dtype": str(frame[c].dtype)} for c in frame.columns]
    duplicate_rows = int(frame.duplicated().sum())
    key = [c for c in ("CODUSU", "ANO4", "TRIMESTRE", "P02", "P03") if c in frame]
    commit = _git_commit(root)
    configuration_path = root / "configs" / "preprocessing_release.yaml"
    consumer_path = root / "configs" / "annual_input_consumer_contract.yaml"
    artifact_file = {
        "path": relpath,
        "role": "annual_preprocessed_input",
        "bytes": csv_path.stat().st_size,
        "sha256": sha256_file(csv_path),
    }
    report_file = {
        "path": lineage_path.resolve().relative_to(root.resolve()).as_posix(),
        "role": "column_lineage_registry",
        "bytes": lineage_path.stat().st_size,
        "sha256": sha256_file(lineage_path),
    }
    return {
        "manifest_schema": SHARED_MANIFEST_SCHEMA,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "release_id": f"{ANNUAL_RELEASE_ID}+{years[0] if years else 'unknown'}",
        "release_status": "candidate",
        "artifact_type": "research.eph-annual-preprocessed",
        "producer_repository": "matuteiglesias/income-modeling-eph",
        "manifest_attestation_commit": commit,
        "artifact_materialization_commit": "unresolved-historical-materialization",
        "created_at": _git_commit_timestamp(root, commit),
        "method_contract_version": "research.eph-annual-preprocessed/v1",
        "data_vintage": years[0] if years else None,
        "inputs": [
            {"release_id": PROVISIONAL_SOURCE_RELEASE, "sha256": "unresolved-upstream-hash"}
        ],
        "files": [artifact_file],
        "reports": [report_file],
        "limitations": [
            "Historical materialization commit is unresolved.",
            "Upstream source hashes and monetary reference are unresolved.",
            "A true unique person key is unavailable.",
            "This artifact is not declared compatible with a Census sample.",
        ],
        "artifact": artifact_file,
        "source_releases": [
            {"release_id": PROVISIONAL_SOURCE_RELEASE, "sha256": "unresolved-upstream-hash"}
        ],
        "preprocessing_code_commit": commit,
        "configuration_sha256": sha256_file(configuration_path),
        "consumer_contract_sha256": sha256_file(consumer_path) if consumer_path.exists() else None,
        "lineage_registry_sha256": sha256_file(lineage_path),
        "coverage": {"years": years, "quarters": quarters},
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_inventory": list(frame.columns),
        "schema": schema,
        "schema_sha256": canonical_hash(schema),
        "entity_checks": {
            "candidate_person_key": key,
            "candidate_person_key_unique": bool(not frame.duplicated(key).any()) if key else None,
            "duplicate_complete_rows": duplicate_rows,
            "note": "A true person sequence key is not present; candidate uniqueness is diagnostic only.",
        },
        "missingness": {c: int(frame[c].isna().sum()) for c in frame.columns},
        "monetary_reference": {
            "status": "unresolved_historical_materialization",
            "price_series_id": "provisional:legacy-price-series-unidentified",
        },
        "geography": {
            "crosswalk_release_id": None,
            "rank_sources": ["data/info/AGLO_rk", "data/info/Reg_rk"],
        },
        "warnings": [
            "Source release hashes and monetary reference require upstream reconciliation.",
            "This manifest characterizes a materialized artifact; it does not prove raw-data reproduction.",
        ],
        "unresolved_lineage": sorted(
            c
            for c, entry in lineage["columns"].items()
            if entry.get("reviewer_status") != "reviewed"
        ),
        "producing_command": "python scripts/11_preprocessing_authority.py manifests",
        "environment": {"python": platform.python_version(), "pandas": pd.__version__},
    }


def validate_manifest(manifest: dict[str, Any], csv_path: Path, lineage_path: Path) -> None:
    if manifest["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported annual-input manifest version.")
    if manifest["artifact"]["sha256"] != sha256_file(csv_path):
        raise ValueError("Manifest file hash does not match annual input.")
    if manifest["artifact"]["bytes"] != csv_path.stat().st_size:
        raise ValueError("Manifest file size does not match annual input.")
    if manifest["lineage_registry_sha256"] != sha256_file(lineage_path):
        raise ValueError("Manifest lineage hash does not match registry.")


def run_fixture(spec: dict[str, Any]) -> pd.DataFrame:
    """Execute the synthetic merge, mapping, deflation, and inclusion boundary."""

    households = pd.DataFrame(spec["households"])
    persons = pd.DataFrame(spec["persons"])
    keys = ["CODUSU", "ANO4", "TRIMESTRE"]
    if households.duplicated(keys).any():
        raise ValueError("Household fixture violates many-to-one merge cardinality.")
    result = persons.merge(households, on=keys, how="left", validate="many_to_one", indicator=True)
    result["merge_valid"] = result.pop("_merge").eq("both")
    result["Region"] = result["AGLOMERADO"].astype(str).map(spec["region_by_agglomerate"])
    factors = {int(k): float(v) for k, v in spec["monetary_factor_by_year"].items()}
    result["P47T"] = result["P47T_nominal"] / result["ANO4"].map(factors)
    result["AGLO_rk"] = result["AGLOMERADO"].astype(str).map(spec["agglomerate_rank_stub"])
    result["INGRESO"] = (result["P47T"] > 0).astype("int64")
    result["included"] = result["merge_valid"] & result["PROP"].notna() & result["INGRESO"].eq(1)
    return result.sort_values(keys + ["person_id"], kind="stable").reset_index(drop=True)
