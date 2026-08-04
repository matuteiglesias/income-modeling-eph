#!/usr/bin/env python3
"""Inventory, fixture, package, and validate Census person-income releases."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eph_income.census_income import (  # noqa: E402
    StageSpec,
    StagedClassifierRegressor,
    load_census_release,
    save_staged_model,
    sha256,
    validate_predictions,
    write_release,
)


def _csv_facts(path: Path) -> dict:
    try:
        frame = pd.read_csv(path, nrows=0)
        rows = sum(1 for _ in path.open(errors="replace")) - 1
        error = None
    except Exception as exc:  # artifact inventory must record, not hide, unreadable candidates
        frame, rows, error = pd.DataFrame(), None, str(exc)
    return {"row_count": rows, "columns": list(frame.columns), "read_error": error}


def inventory(roots: list[Path], output: Path) -> int:
    patterns = ("RFC*.csv", "*census*.csv", "*censo*.csv", "*.joblib", "*.pkl", "*.pickle")
    candidates: set[Path] = set()
    inspected = []
    for root in roots:
        exists = root.exists()
        inspected.append({"path": str(root.absolute()), "exists": exists})
        if exists:
            for pattern in patterns:
                candidates.update(path for path in root.rglob(pattern) if path.is_file())
    artifacts = []
    for path in sorted(candidates):
        item = {
            "absolute_path": str(path.absolute()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "kind": "csv" if path.suffix.lower() == ".csv" else "model",
            "period": None,
            "sample_fraction": None,
            "person_id_evidence": "not_demonstrated",
            "transform_evidence": "not_demonstrated",
            "monetary_reference_evidence": "not_demonstrated",
            "stage_identity": path.stem,
            "compatible_census_release": None,
            "package_without_recomputation": False,
        }
        if item["kind"] == "csv":
            item.update(_csv_facts(path))
        artifacts.append(item)
    result = {
        "inventory_version": 1,
        "roots_inspected": inspected,
        "artifacts": artifacts,
        "matching_final_output_found": False,
        "inference_attempted": False,
        "release_produced": False,
        "stop_reason": (
            "No exactly matched Census sample and RFC4/final prediction pair was discovered; "
            "required historical model/feature artifacts are absent or unverified."
        ),
        "next_step": (
            "Mount or copy the historical indice-pobreza-UBA/encuestador-de-hogares workspaces, "
            "Census release, RFC files, fitted_RF/modelos, rank tables, and run metadata; rerun inventory."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


def _fixture_frames():
    train = pd.DataFrame({
        "age": [18, 24, 33, 45, 52, 61, 29, 38, 49, 70, 41, 56],
        "sex": ["F", "M"] * 6,
        "department": ["A", "A", "B", "B", "C", "C"] * 2,
        "activity": [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0],
        "occupation": [0, 1, 2, 2, 0, 0, 1, 2, 2, 0, 1, 0],
        "education": [1, 2, 3, 2, 1, 1, 3, 2, 2, 1, 3, 1],
        "income": [-0.2, 3.0, 3.5, 3.8, 2.2, 2.0, 3.7, 3.6, 3.9, 1.8, 3.8, 2.1],
    })
    census = pd.DataFrame({
        "sample_person_id": ["hh1:p1", "hh1:p2", "hh2:p1", "hh3:p1"],
        "period": ["2010"] * 4,
        "age": [19, 35, 48, 64],
        "sex": ["F", "M", None, "F"],
        "department": ["A", "B", "C", "C"],
    })
    return train, census


def _pipeline(columns):
    prep = ColumnTransformer([(
        "features", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                   OneHotEncoder(handle_unknown="ignore")), columns
    )])
    return prep


def run_fixture(output_root: Path) -> Path:
    train, census = _fixture_frames()
    base = ["age", "sex", "department"]
    stage1 = make_pipeline(_pipeline(base), RandomForestClassifier(n_estimators=12, random_state=42))
    wave2_features = [*base, "pred_activity"]
    stage2 = make_pipeline(_pipeline(wave2_features), RandomForestClassifier(n_estimators=12, random_state=42))
    wave3_features = [*wave2_features, "pred_occupation"]
    stage3 = make_pipeline(_pipeline(wave3_features), RandomForestClassifier(n_estimators=12, random_state=42))
    regression_features = [*wave3_features, "pred_education"]
    regressor = make_pipeline(_pipeline(regression_features), RandomForestRegressor(n_estimators=16, random_state=42))
    model = StagedClassifierRegressor([
        (StageSpec("activity", "activity", "pred_activity", tuple(base)), stage1),
        (StageSpec("occupation", "occupation", "pred_occupation", tuple(wave2_features)), stage2),
        (StageSpec("education", "education", "pred_education", tuple(wave3_features)), stage3),
    ], regressor, regression_features).fit(train, "income")

    output_root.mkdir(parents=True, exist_ok=True)
    census_dir = output_root / "census-fixture-v1"
    census_dir.mkdir(exist_ok=False)
    census_path = census_dir / "census_sample.csv"
    census.to_csv(census_path, index=False)
    census_manifest = {
        "contract": "research.census-sample/v1", "release_id": "census-fixture-v1",
        "sample_id_namespace": "fixture:household-person", "data_file": census_path.name,
        "data_sha256": sha256(census_path),
    }
    census_manifest_path = census_dir / "manifest.json"
    census_manifest_path.write_text(json.dumps(census_manifest, indent=2) + "\n")
    model_path = output_root / "staged_fixture.joblib"
    save_staged_model(model, model_path)
    values = model.predict(census)
    predictions = pd.DataFrame({
        "sample_person_id": census.sample_person_id, "period": census.period,
        "prediction_value": values, "prediction_transform": "log10_ars",
        "monetary_reference": "fixture-unresolved", "classification": "projected",
        "model_release_id": "staged-rf-fixture-v1",
    })
    release_dir = output_root / "person-income-fixture-v1"
    manifest = write_release(
        census, predictions, census_manifest, census_manifest_path, release_dir,
        source_paths=[model_path],
        stage_identities=[{"name": name, "type": "RandomForestClassifier"}
                          for name in ("activity", "occupation", "education")]
                         + [{"name": "income", "type": "RandomForestRegressor"}],
        monetary_status="unresolved", warnings=["Synthetic fixture; no scientific claim."],
    )
    print(manifest)
    return release_dir


def check_release(release_dir: Path) -> int:
    manifest_path = release_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    output = release_dir / manifest["output_file"]
    if sha256(output) != manifest["output_sha256"]:
        raise ValueError("Prediction output checksum mismatch")
    predictions = pd.read_csv(output, dtype={"sample_person_id": str})
    if len(predictions) != manifest["row_count"]:
        raise ValueError("Prediction row count differs from manifest")
    if manifest.get("contract") != "research.person-income-predictions/v1":
        raise ValueError("Wrong release contract")
    expected_order = predictions.sort_values(["period", "sample_person_id"], kind="stable")
    if not predictions.reset_index(drop=True).equals(expected_order.reset_index(drop=True)):
        raise ValueError("Prediction output ordering is not deterministic")
    print(json.dumps({"release": str(release_dir), "valid": True, "rows": len(predictions)}))
    return 0


def package(args) -> int:
    sample, census_manifest, _ = load_census_release(args.census_release)
    source = pd.read_csv(args.predictions, dtype={"sample_person_id": str})
    validate_predictions(sample, source)
    write_release(
        sample, source, census_manifest, args.census_release / "manifest.json", args.output_dir,
        source_paths=[args.predictions], stage_identities=json.loads(args.stage_identities),
        monetary_status=args.monetary_status, warnings=args.warning,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--root", action="append", type=Path, default=[])
    inv.add_argument("--output", type=Path, default=Path("reports/census_income/artifact_inventory.json"))
    fixture = sub.add_parser("fixture")
    fixture.add_argument("--output-root", type=Path)
    check = sub.add_parser("check")
    check.add_argument("--release-dir", type=Path, required=True)
    package_parser = sub.add_parser("package")
    package_parser.add_argument("--census-release", type=Path, required=True)
    package_parser.add_argument("--predictions", type=Path, required=True)
    package_parser.add_argument("--output-dir", type=Path, required=True)
    package_parser.add_argument("--stage-identities", default="[]")
    package_parser.add_argument("--monetary-status", choices=["verified", "probable_by_code_lineage", "unresolved", "incompatible"], required=True)
    package_parser.add_argument("--warning", action="append", default=[])
    args = parser.parse_args()
    if args.command == "inventory":
        roots = args.root or [Path.cwd(), Path("/media/matias/Elements/suite"), Path.home()]
        return inventory(roots, args.output)
    if args.command == "fixture":
        if args.output_root:
            run_fixture(args.output_root)
        else:
            with tempfile.TemporaryDirectory() as directory:
                release = run_fixture(Path(directory))
                check_release(release)
        return 0
    if args.command == "check":
        return check_release(args.release_dir)
    return package(args)


if __name__ == "__main__":
    raise SystemExit(main())
