#!/usr/bin/env python
"""Preflight, lock, resolve, release, and validate the governed flagship candidate."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eph_income.flagship_freeze import (  # noqa: E402
    canonical_hash,
    inventory_runs,
    sha256_file,
    smoke_release,
    validate_annual_inputs,
    validate_release,
    write_or_check_lock,
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(write_doc: bool = True) -> Path | None:
    lock = write_or_check_lock(ROOT, check=True)
    candidates, selected = inventory_runs(ROOT, lock)
    if write_doc:
        lines = [
            "# Flagship run resolution",
            "",
            f"Input lock: `{lock['lock_id']}`",
            "",
            "## Candidates inspected",
            "",
        ]
        if not candidates:
            lines.append("No canonical run directories were present.")
        for item in candidates:
            lines += [f"### `{item['run_dir']}`", "", f"Compatible: **{item['compatible']}**", ""]
            lines += [f"- `{key}`: {value}" for key, value in item["checks"].items()]
        lines += ["", "## Resolution", ""]
        lines.append(
            f"Selected `{selected.relative_to(ROOT)}` by complete identity checks; no recency heuristic was used."
            if selected
            else "No run matches the deterministic lock. Exact retraining with `make run-hgb-quick-benchmark-freeze` is required."
        )
        lines += [
            "",
            "Missing/stale artifacts are represented by failed checks above. Training must not begin before annual preflight and lock checks pass.",
            "",
        ]
        (ROOT / "docs/FLAGSHIP_RUN_RESOLUTION.md").write_text("\n".join(lines), encoding="utf-8")
    return selected


def _release() -> Path:
    lock = write_or_check_lock(ROOT, check=True)
    selected = _resolve()
    if selected is None:
        raise ValueError("No compatible flagship run; run the exact freeze benchmark first.")
    model_source = selected / "artifacts/model_pipeline.joblib"
    if not model_source.is_file():
        raise ValueError(
            "Compatible run lacks the freeze-only fitted pipeline; exact retraining is required."
        )
    release_id = f"eph-income-model-{lock['lock_id'].removeprefix('model-input-lock-')}"
    release_dir = ROOT / "artifacts/model_releases" / release_id
    if release_dir.exists():
        validate_release(ROOT, release_dir)
        return release_dir
    release_dir.mkdir(parents=True)
    shutil.copy2(model_source, release_dir / "model_pipeline.joblib")
    shutil.copy2(
        ROOT / "artifacts/model_freeze/model_input_lock.json", release_dir / "model_input_lock.json"
    )
    feature_columns = _json(selected / "feature_columns.json")
    sample = selected / "artifacts/training_frame_sample.csv"
    with sample.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))[:3]
    fixture_columns = ["row_id", *feature_columns]
    with (release_dir / "prediction_fixture.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fixture_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in fixture_columns})
    contract = {
        "schema": "research.eph-income-inference-contract/v1",
        "required_input_columns": feature_columns,
        "feature_order": feature_columns,
        "missing_columns": "reject",
        "extra_columns": "reject except row_id",
        "output_row_identity": "row_id preserved in input order",
        "target": {
            "name": "predicted_logP47T",
            "scale": "log10_ars",
            "definition": "log10(P47T)",
            "plus_one": False,
            "primary": True,
            "mechanical_inverse": "Not emitted; 10**prediction would be a biased mechanical retransformation.",
        },
        "census_sample_compatible": False,
        "provenance_fields": ["model_release_id", "input_lock_id", "source_row_id"],
        "fixture_row_ids": [row["row_id"] for row in rows],
        "trust_boundary": "Validate the envelope and hash before loading. Joblib/pickle is trusted-local only and must never be loaded from an untrusted source.",
    }
    (release_dir / "inference_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
        stdout=(release_dir / "requirements.lock.txt").open("w"),
        check=True,
    )
    comparison = next(csv.DictReader((selected / "metrics/model_comparison.csv").open()))
    dataset_card = _json(selected / "dataset_card.json")
    copied = [
        "metrics/model_comparison.csv",
        "predictions/test_predictions.parquet",
        "predictions/validation_predictions.parquet",
        "diagnostics/residual_summary.csv",
        "diagnostics/error_by_income_decile.csv",
        "diagnostics/prediction_distribution_summary.csv",
        "diagnostics/distribution_compression_summary.csv",
    ]
    for rel in copied:
        destination = release_dir / "run_evidence" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected / rel, destination)
    files = []
    for path in sorted(p for p in release_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(release_dir).as_posix()
        files.append(
            {
                "path": rel,
                "role": "fitted_pipeline" if rel == "model_pipeline.joblib" else "release_evidence",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest_core = {
        "manifest_schema": "research-artifact-manifest/v1",
        "artifact_type": "research.eph-income-model/v1",
        "release_id": release_id,
        "release_status": "candidate",
        "producer_repository": "matuteiglesias/income-modeling-eph",
        "producer_commit": lock["repository"]["commit"],
        "created_at": subprocess.run(
            ["git", "show", "-s", "--format=%cI", lock["repository"]["commit"]],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip(),
        "input_lock": {
            "id": lock["lock_id"],
            "sha256": sha256_file(ROOT / "artifacts/model_freeze/model_input_lock.json"),
        },
        "experiment": {
            "id": "hgb_quick_benchmark_v1",
            "config_sha256": lock["contracts"]["experiment_config_sha256"],
        },
        "target": lock["target"],
        "feature_view": lock["feature_view"],
        "model": {
            "class": "HistGradientBoostingRegressor",
            "parameters": json.loads(comparison["best_params"]),
        },
        "splits": dataset_card["splits"],
        "split_identity": lock["processed"]["split_assignments_sha256"],
        "environment": {
            "python": platform.python_version(),
            "serialization": "joblib",
            "joblib": __import__("joblib").__version__,
        },
        "metrics": {
            key: comparison[key]
            for key in comparison
            if key.endswith(("r2", "mae", "mse")) or key.startswith("cv_r2")
        },
        "training_frame_sample_sha256": sha256_file(sample),
        "compatibility": "inference_contract.json",
        "files": files,
        "limitations": [
            "Candidate only; not reviewed or approved.",
            "Raw EPH reproduction, historical monetary reference, and true person key remain unresolved.",
            "EPH-derived rows only; Census inference is incompatible and rejected.",
            "Predictions are predictive, not causal or official income estimates.",
        ],
    }
    manifest_core["release_content_sha256"] = canonical_hash(manifest_core)
    (release_dir / "manifest.json").write_text(
        json.dumps(manifest_core, indent=2, sort_keys=True) + "\n"
    )
    return release_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "preflight",
            "lock",
            "lock-check",
            "resolve",
            "release",
            "release-check",
            "release-smoke",
        ],
    )
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--release-dir")
    args = parser.parse_args()
    if args.action == "preflight":
        result = validate_annual_inputs(ROOT, approved=args.approved)
        print(
            json.dumps(
                {
                    "validated_years": [x["year"] for x in result],
                    "mode": "approved" if args.approved else "candidate",
                },
                indent=2,
            )
        )
    elif args.action == "lock":
        print(json.dumps(write_or_check_lock(ROOT), indent=2))
    elif args.action == "lock-check":
        print(write_or_check_lock(ROOT, check=True)["lock_id"])
    elif args.action == "resolve":
        print(_resolve() or "no-compatible-run")
    else:
        release_dir = (
            Path(args.release_dir)
            if args.release_dir
            else (_release() if args.action == "release" else None)
        )
        if release_dir is None:
            releases = sorted((ROOT / "artifacts/model_releases").glob("*"))
            if len(releases) != 1:
                raise ValueError("Specify --release-dir when exactly one release is not present.")
            release_dir = releases[0]
        if args.action == "release":
            print(release_dir)
        elif args.action == "release-check":
            print(validate_release(ROOT, release_dir)["release_id"])
        else:
            print(smoke_release(ROOT, release_dir))


if __name__ == "__main__":
    main()
