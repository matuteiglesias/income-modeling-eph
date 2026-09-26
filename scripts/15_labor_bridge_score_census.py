#!/usr/bin/env python3
"""Score a harmonized Census person frame with a fitted EPH labor bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from eph_income.entity_identity import (  # noqa: E402
    census_identity_columns,
    validate_census_person_identity,
)
from eph_income.labor_bridge import TwoStageLaborBridge  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def run(
    model_path: Path,
    census_path: Path,
    output: Path,
    *,
    person_id_column: str,
    calibration_domain_column: str,
) -> dict[str, object]:
    frame = load_frame(census_path)
    for column in (person_id_column, calibration_domain_column):
        if column not in frame:
            raise ValueError(f"Census labor scoring input missing {column}")
    if frame[person_id_column].astype(str).duplicated().any():
        raise ValueError("Census labor scoring person IDs must be unique")

    identity_frame = frame.copy()
    normalized_person_id = frame[person_id_column].astype(str)
    if "sample_person_id" in identity_frame and person_id_column != "sample_person_id":
        if not identity_frame["sample_person_id"].astype(str).equals(normalized_person_id):
            raise ValueError(
                "configured Census person identity disagrees with sample_person_id"
            )
    identity_frame["sample_person_id"] = normalized_person_id
    identity_audit = validate_census_person_identity(
        identity_frame, context="Census labor scoring"
    )
    identity_columns = census_identity_columns(identity_frame)

    bridge = TwoStageLaborBridge.load(model_path)
    scored = bridge.predict_probabilities(frame)
    out = identity_frame.loc[:, identity_columns].copy()
    out["calibration_domain_id"] = frame[calibration_domain_column].to_numpy()
    out["p_active_raw"] = scored["labor_p_active"].to_numpy()
    out["p_unemployed_given_active_raw"] = scored[
        "labor_p_unemployed_given_active"
    ].to_numpy()
    out["p_employed_raw"] = scored["labor_p_employed"].to_numpy()
    out["p_unemployed_raw"] = scored["labor_p_unemployed"].to_numpy()
    out["p_inactive_raw"] = scored["labor_p_inactive"].to_numpy()

    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "census_labor_probabilities_raw.parquet"
    out.to_parquet(raw_path, index=False)
    manifest = {
        "contract": "research.census-labor-probabilities-raw/v1",
        "release_id": output.name,
        "model": {"path": str(model_path.resolve()), "sha256": sha256(model_path)},
        "census_frame": {
            "path": str(census_path.resolve()),
            "sha256": sha256(census_path),
            "rows": len(frame),
        },
        "identity": identity_audit,
        "person_id_column": person_id_column,
        "calibration_domain_column": calibration_domain_column,
        "artifact": {
            "path": raw_path.name,
            "sha256": sha256(raw_path),
            "bytes": raw_path.stat().st_size,
            "rows": len(out),
        },
        "next_contract": (
            "research.census-labor-probabilities/v1 after official-domain calibration"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--model", type=Path, required=True)
    out.add_argument("--census", type=Path, required=True)
    out.add_argument("--output", type=Path, required=True)
    out.add_argument("--person-id-column", default="sample_person_id")
    out.add_argument("--calibration-domain-column", default="eph_agglomerate_id")
    return out


def main() -> int:
    args = parser().parse_args()
    manifest = run(
        args.model,
        args.census,
        args.output,
        person_id_column=args.person_id_column,
        calibration_domain_column=args.calibration_domain_column,
    )
    print(
        json.dumps(
            {"release": manifest["release_id"], "rows": manifest["artifact"]["rows"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
