"""Deterministic flagship input/release envelopes and standard-library validation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

ANNUAL_SCHEMA = "research-artifact-manifest/v1"
COMPATIBILITY_SCHEMA = "research-artifact-compatibility/v1"
LOCK_SCHEMA = "research.eph-model-input-lock/v1"
RELEASE_SCHEMA = "research-artifact-manifest/v1"
FLAGSHIP_EXPERIMENT = "hgb_quick_benchmark_v1"
REQUIRED_YEARS = (2022, 2023, 2024, 2025)
ALLOWED_STATUS = {"candidate", "reviewed", "approved"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def yaml_hash(path: Path) -> str:
    import yaml

    return canonical_hash(yaml.safe_load(path.read_text(encoding="utf-8")))


def _safe_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Absolute artifact path rejected: {value}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes repository: {value}") from exc
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate_annual_inputs(root: Path, *, approved: bool = False) -> list[dict[str, Any]]:
    """Validate the complete annual envelope without importing pandas or model code."""

    compatibility_path = root / "configs/model_training_compatibility.json"
    compatibility = _read_json(compatibility_path)
    if compatibility.get("schema") != COMPATIBILITY_SCHEMA:
        raise ValueError("Unsupported compatibility schema.")
    if compatibility.get("census_sample_compatible") is not False:
        raise ValueError("Compatibility contract must reject Census samples.")
    results = []
    for year in REQUIRED_YEARS:
        short = str(year)[2:]
        manifest_path = (
            root / f"data/annual_preprocessed_manifests/EPHARG_annual_input_{short}.manifest.json"
        )
        manifest = _read_json(manifest_path)
        if manifest.get("manifest_schema") not in compatibility["supported_manifest_schemas"]:
            raise ValueError("Unsupported annual manifest schema.")
        if (
            manifest.get("manifest_schema_version")
            not in compatibility["supported_manifest_schema_versions"]
        ):
            raise ValueError("Unsupported annual manifest schema version.")
        if manifest.get("artifact_type") != "research.eph-annual-preprocessed":
            raise ValueError("Wrong annual artifact type.")
        status = manifest.get("release_status")
        if status not in ALLOWED_STATUS:
            raise ValueError(f"Unknown release status: {status}")
        if manifest.get("data_vintage") != year or manifest.get("coverage", {}).get("years") != [
            year
        ]:
            raise ValueError(f"Wrong annual vintage for {manifest_path.name}")
        if not set(compatibility["minimum_required_columns"]).issubset(
            manifest.get("column_inventory", [])
        ):
            raise ValueError(f"Required columns absent from {manifest_path.name}")
        files = manifest.get("files", [])
        if len(files) != 1:
            raise ValueError("Annual manifest must identify exactly one data file.")
        entry = files[0]
        expected_rel = compatibility["required_files"][REQUIRED_YEARS.index(year)]
        if entry.get("path") != expected_rel:
            raise ValueError(f"Wrong annual file for vintage {year}")
        artifact_path = _safe_path(root, entry["path"])
        if not artifact_path.is_file():
            raise ValueError(f"Missing annual file: {entry['path']}")
        if entry.get("bytes") != artifact_path.stat().st_size:
            raise ValueError(f"Stale annual file size: {entry['path']}")
        if entry.get("sha256") != sha256_file(artifact_path):
            raise ValueError(f"Stale annual file hash: {entry['path']}")
        expected_hashes = {
            "configuration_sha256": sha256_file(root / "configs/preprocessing_release.yaml"),
            "lineage_registry_sha256": sha256_file(root / "configs/annual_input_lineage.yaml"),
            "consumer_contract_sha256": sha256_file(
                root / "configs/annual_input_consumer_contract.yaml"
            ),
            "schema_sha256": canonical_hash(manifest["schema"]),
        }
        for field, expected in expected_hashes.items():
            if manifest.get(field) != expected:
                raise ValueError(f"Stale {field} in {manifest_path.name}")
        if (
            manifest.get("artifact_materialization_commit")
            != "unresolved-historical-materialization"
        ):
            raise ValueError("Historical materialization provenance must not be fabricated.")
        unresolved = (
            manifest.get("source_releases", [{}])[0].get("sha256") == "unresolved-upstream-hash"
            or manifest.get("monetary_reference", {}).get("status") != "resolved"
            or manifest.get("artifact_materialization_commit", "").startswith("unresolved")
        )
        if approved and (status != "approved" or unresolved):
            raise ValueError("Approved mode rejects candidate or unresolved annual artifacts.")
        results.append(
            {
                "year": year,
                "manifest_path": manifest_path,
                "manifest": manifest,
                "manifest_sha256": sha256_file(manifest_path),
                "csv_sha256": entry["sha256"],
            }
        )
    return results


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _governing_source_commit(root: Path) -> str:
    """Return the newest commit affecting inputs/code, excluding generated lock commits."""

    return _git(
        root,
        "log",
        "-1",
        "--format=%H",
        "--",
        "src",
        "scripts",
        "configs",
        "Makefile",
        "pyproject.toml",
    )


def build_input_lock(root: Path) -> dict[str, Any]:
    annual = validate_annual_inputs(root)
    processed = root / "data/processed/modeling_dataset.parquet"
    metadata = root / "data/processed/dataset_metadata.json"
    splits = root / "data/processed/split_assignments.csv"
    for path in (processed, metadata, splits):
        if not path.is_file():
            raise ValueError(f"Canonical processed artifact missing: {path.relative_to(root)}")
    registry = _read_registry_entry(root)
    payload = {
        "schema": LOCK_SCHEMA,
        "annual_artifacts": [
            {
                "year": item["year"],
                "release_id": item["manifest"]["release_id"],
                "manifest_sha256": item["manifest_sha256"],
                "csv_sha256": item["csv_sha256"],
            }
            for item in annual
        ],
        "contracts": {
            "annual_lineage_registry_sha256": sha256_file(
                root / "configs/annual_input_lineage.yaml"
            ),
            "annual_consumer_contract_sha256": sha256_file(
                root / "configs/annual_input_consumer_contract.yaml"
            ),
            "preprocessing_release_config_sha256": sha256_file(
                root / "configs/preprocessing_release.yaml"
            ),
            "compatibility_sha256": sha256_file(root / "configs/model_training_compatibility.json"),
            "dataset_builder_sha256": sha256_file(root / "src/eph_income/dataset.py"),
            "dataset_builder_contract_version": "research.eph-modeling-dataset/v1",
            "feature_contract_sha256": yaml_hash(root / "configs/feature_contract.yaml"),
            "experiment_config_sha256": yaml_hash(
                root / "configs/experiment_hgb_quick_benchmark.yaml"
            ),
            "experiment_registry_entry_sha256": canonical_hash(registry),
        },
        "processed": {
            "modeling_dataset_sha256": sha256_file(processed),
            "dataset_metadata_sha256": sha256_file(metadata),
            "split_assignments_sha256": sha256_file(splits),
        },
        "split": {
            "strategy": "random_individual",
            "random_state": 42,
            "train_size": 0.7,
            "test_size": 0.2,
            "validation_size": 0.1,
        },
        "target": {
            "name": "logP47T",
            "source": "P47T",
            "transform": "log10",
            "plus_one": False,
            "output_scale": "log10_ars",
        },
        "feature_view": {"name": "clean_geo", "drop_columns": ["AGLO_rk", "Reg_rk"]},
        "environment": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "repository": {
            "name": "matuteiglesias/income-modeling-eph",
            "commit": _governing_source_commit(root),
        },
    }
    payload["lock_id"] = f"model-input-lock-{canonical_hash(payload)[:16]}"
    return payload


def _read_registry_entry(root: Path) -> dict[str, Any]:
    # Avoid a YAML dependency in preflight: extract the normalized authority fields used by the lock.
    import yaml

    registry = yaml.safe_load((root / "configs/experiment_registry.yaml").read_text())
    return next(
        item for item in registry["experiments"] if item["experiment_id"] == FLAGSHIP_EXPERIMENT
    )


def write_or_check_lock(root: Path, *, check: bool = False) -> dict[str, Any]:
    lock = build_input_lock(root)
    path = root / "artifacts/model_freeze/model_input_lock.json"
    if check:
        if not path.is_file() or _read_json(path) != lock:
            raise ValueError("Model input lock is missing or stale.")
    else:
        if path.exists() and _read_json(path) != lock:
            raise ValueError("Refusing to overwrite a different model input lock.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return lock


def inventory_runs(root: Path, lock: dict[str, Any]) -> tuple[list[dict[str, Any]], Path | None]:
    candidates: list[dict[str, Any]] = []
    selected = None
    required = _read_registry_entry(root).get("required_artifacts", [])
    for run_dir in (
        sorted((root / "reports/runs").glob("*")) if (root / "reports/runs").exists() else []
    ):
        if not run_dir.is_dir() or not (run_dir / "config_used.yaml").exists():
            continue
        checks = {
            "experiment_id": run_dir.name.startswith(f"{FLAGSHIP_EXPERIMENT}_"),
            "config_hash": yaml_hash(run_dir / "config_used.yaml")
            == lock["contracts"]["experiment_config_sha256"],
            "feature_contract_hash": (run_dir / "feature_contract_used.yaml").is_file()
            and yaml_hash(run_dir / "feature_contract_used.yaml")
            == lock["contracts"]["feature_contract_sha256"],
            "required_artifacts": all((run_dir / rel).is_file() for rel in required),
            "core_artifacts": all(
                (run_dir / rel).is_file()
                for rel in [
                    "run_manifest.json",
                    "experiment_card.json",
                    "predictions/test_predictions.parquet",
                    "predictions/validation_predictions.parquet",
                    "metrics/model_comparison.csv",
                ]
            ),
            "comparison_convenience_copy": (
                root / "reports/tables/model_comparison_hgb_quick_benchmark.csv"
            ).is_file()
            and sha256_file(root / "reports/tables/model_comparison_hgb_quick_benchmark.csv")
            == sha256_file(run_dir / "metrics/model_comparison.csv"),
            "card_convenience_copy": (
                root / "reports/experiment_cards/hgb_quick_benchmark.json"
            ).is_file()
            and sha256_file(root / "reports/experiment_cards/hgb_quick_benchmark.json")
            == sha256_file(run_dir / "experiment_card.json"),
        }
        if (run_dir / "dataset_card.json").exists():
            card = _read_json(run_dir / "dataset_card.json")
            checks["processed_hash"] = (
                card.get("digest") == lock["processed"]["modeling_dataset_sha256"]
            )
            checks["split_hash"] = (
                card.get("split_digest") == lock["processed"]["split_assignments_sha256"]
            )
        else:
            checks["processed_hash"] = checks["split_hash"] = False
        item = {
            "run_dir": run_dir.relative_to(root).as_posix(),
            "checks": checks,
            "compatible": all(checks.values()),
        }
        candidates.append(item)
        if item["compatible"]:
            selected = run_dir
            break
    return candidates, selected


def validate_inference_rows(columns: list[str], required: list[str], *, mode: str) -> None:
    if mode.lower() == "census":
        raise ValueError("Census-mode inference is not compatible with this candidate release.")
    missing = sorted(set(required) - set(columns))
    extra = sorted(set(columns) - set(required) - {"row_id"})
    if missing or extra:
        raise ValueError(f"Inference schema mismatch; missing={missing}, extra={extra}")


def validate_release(root: Path, release_dir: Path, *, load_model: bool = False) -> dict[str, Any]:
    manifest = _read_json(release_dir / "manifest.json")
    if (
        manifest.get("manifest_schema") != RELEASE_SCHEMA
        or manifest.get("artifact_type") != "research.eph-income-model/v1"
        or manifest.get("release_status") != "candidate"
    ):
        raise ValueError("Unsupported model release envelope.")
    for entry in manifest.get("files", []):
        path = _safe_path(release_dir, entry["path"])
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError(f"Tampered release artifact: {entry['path']}")
    contract = _read_json(release_dir / "inference_contract.json")
    if (
        contract.get("target", {}).get("scale") != "log10_ars"
        or contract["target"].get("plus_one") is not False
    ):
        raise ValueError("Invalid target inference contract.")
    if contract.get("census_sample_compatible") is not False:
        raise ValueError("Census compatibility must be rejected.")
    if load_model:
        import joblib

        joblib.load(release_dir / "model_pipeline.joblib")
    return manifest


def smoke_release(root: Path, release_dir: Path) -> Path:
    smoke_root = Path(os.environ.get("TMPDIR", "/tmp")) / "eph-model-release-smoke"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    shutil.copytree(release_dir, smoke_root)
    validate_release(root, smoke_root, load_model=True)
    contract = _read_json(smoke_root / "inference_contract.json")
    fixture = smoke_root / "prediction_fixture.csv"
    with fixture.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        columns = list(rows[0]) if rows else []
    validate_inference_rows(columns, contract["required_input_columns"], mode="eph")
    if [row["row_id"] for row in rows] != contract["fixture_row_ids"]:
        raise ValueError("Prediction fixture row identity/order mismatch.")
    try:
        validate_inference_rows(columns, contract["required_input_columns"], mode="census")
    except ValueError:
        pass
    else:
        raise ValueError("Census mode was not rejected.")
    return smoke_root
