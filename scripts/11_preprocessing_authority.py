#!/usr/bin/env python
"""Validate and manifest repository-owned annual preprocessed EPH inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eph_income.preprocessing_authority import (  # noqa: E402
    build_manifest,
    load_lineage,
    run_fixture,
    validate_manifest,
)

LINEAGE = ROOT / "configs" / "annual_input_lineage.yaml"
INPUT_DIR = ROOT / "data" / "annual_preprocessed_inputs"
MANIFEST_DIR = ROOT / "data" / "annual_preprocessed_manifests"


def inputs() -> list[Path]:
    return sorted(INPUT_DIR.glob("EPHARG_annual_input_*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate", "smoke", "fixture", "manifests"])
    parser.add_argument("--check", action="store_true", help="Compare generated manifests with disk.")
    args = parser.parse_args()
    load_lineage(LINEAGE)
    if args.action == "smoke":
        for path in inputs():
            manifest = build_manifest(path, LINEAGE, ROOT)
            print(path.name, manifest["rows"], manifest["columns"])
        return
    if args.action == "fixture":
        spec = json.loads((ROOT / "tests" / "fixtures" / "preprocessing_fixture.json").read_text())
        result = run_fixture(spec)
        expected = spec["expected"]
        assert len(result) == expected["rows"]
        assert int(result["included"].sum()) == expected["included_rows"]
        assert int(result["merge_valid"].sum()) == expected["valid_merge_rows"]
        print(json.dumps(expected, sort_keys=True))
        return
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for path in inputs():
        output = MANIFEST_DIR / f"{path.stem}.manifest.json"
        generated = build_manifest(path, LINEAGE, ROOT)
        if args.check:
            existing = json.loads(output.read_text())
            # Commit/environment fields are provenance snapshots, not content checks.
            for volatile in ("preprocessing_code_commit", "environment"):
                generated[volatile] = existing.get(volatile)
            if generated != existing:
                raise ValueError(f"Stale manifest: {output.relative_to(ROOT)}")
            validate_manifest(existing, path, LINEAGE)
        else:
            output.write_text(json.dumps(generated, indent=2, sort_keys=True) + "\n")
        print(output.relative_to(ROOT))
    if args.action == "validate":
        print("annual preprocessing releases validated")


if __name__ == "__main__":
    main()
