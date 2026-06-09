#!/usr/bin/env python3
"""Build diagnostics for thesis-relevant experiment runs.

This script is intentionally orchestration-only: it does not train models and it
should not import experiment-training code. It reads an experiment registry,
finds the latest run directory for selected experiments, calls the existing
`04_build_diagnostics.py` script for each requested split, and writes a manifest
under reports/thesis_evidence/.

Expected registry shape, minimally:

experiments:
  - experiment_id: hgb_quick_clean_geo_v1
    priority: required
    family: thesis_core
    diagnostics_profile: thesis_core

Optional fields such as role, thesis_section, required_artifacts are preserved
in the output manifest when present.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyYAML is required. Install project dependencies first.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "experiment_registry.yaml"
DEFAULT_RUNS_DIR = REPO_ROOT / "reports" / "runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "thesis_evidence"
DIAGNOSTICS_SCRIPT = REPO_ROOT / "scripts" / "04_build_diagnostics.py"


@dataclass(frozen=True)
class ExperimentEntry:
    experiment_id: str
    raw: dict[str, Any]

    @property
    def priority(self) -> str | None:
        return self.raw.get("priority")

    @property
    def family(self) -> str | None:
        return self.raw.get("family")

    @property
    def diagnostics_profile(self) -> str | None:
        return self.raw.get("diagnostics_profile") or self.raw.get("profile")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_list_arg(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    out: set[str] = set()
    for value in values:
        for part in value.replace(",", " ").split():
            if part:
                out.add(part)
    return out or None


def load_registry(path: Path) -> list[ExperimentEntry]:
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment registry not found: {path}. "
            "Create configs/experiment_registry.yaml before using thesis orchestration."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    experiments = data.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError(f"Registry must contain a top-level experiments list: {path}")

    entries: list[ExperimentEntry] = []
    seen: set[str] = set()
    for idx, item in enumerate(experiments):
        if not isinstance(item, dict):
            raise ValueError(f"Registry experiment at index {idx} is not a mapping")
        experiment_id = item.get("experiment_id") or item.get("id")
        if not experiment_id:
            raise ValueError(f"Registry experiment at index {idx} lacks experiment_id")
        if experiment_id in seen:
            raise ValueError(f"Duplicate experiment_id in registry: {experiment_id}")
        seen.add(experiment_id)
        entries.append(ExperimentEntry(str(experiment_id), item))
    return entries


def select_entries(
    entries: list[ExperimentEntry],
    *,
    priorities: set[str] | None,
    families: set[str] | None,
    profiles: set[str] | None,
) -> list[ExperimentEntry]:
    selected: list[ExperimentEntry] = []
    for entry in entries:
        if priorities is not None and entry.priority not in priorities:
            continue
        if families is not None and entry.family not in families:
            continue
        if profiles is not None and entry.diagnostics_profile not in profiles:
            continue
        selected.append(entry)
    return selected


def find_latest_run_dir(runs_dir: Path, experiment_id: str) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith(f"{experiment_id}_")]
    if not candidates:
        candidates = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith(experiment_id)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def run_diagnostics(run_dir: Path, split: str, *, max_scatter_points: int) -> dict[str, Any]:
    if not DIAGNOSTICS_SCRIPT.exists():
        raise FileNotFoundError(f"Diagnostics script not found: {DIAGNOSTICS_SCRIPT}")
    cmd = [
        sys.executable,
        str(DIAGNOSTICS_SCRIPT),
        "--run-dir",
        str(run_dir),
        "--split",
        split,
        "--max-scatter-points", str(max_scatter_points),

    ]

    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    result: dict[str, Any] = {
        "split": split,
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode == 0 and completed.stdout.strip():
        try:
            result["parsed_stdout"] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            result["parsed_stdout"] = None
    return result

def build_manifest(
    *,
    registry_path: Path,
    selected: list[ExperimentEntry],
    runs_dir: Path,
    splits: list[str],
    allow_missing: bool,
    dry_run: bool,
    max_scatter_points: int,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at": utc_now(),
        "registry": str(registry_path),
        "runs_dir": str(runs_dir),
        "splits": splits,
        "allow_missing": allow_missing,
        "dry_run": dry_run,
        "experiments": [],
    }

    missing: list[str] = []
    failures: list[dict[str, Any]] = []
    missing_required_artifacts: list[dict[str, Any]] = []

    for entry in selected:
        run_dir = find_latest_run_dir(runs_dir, entry.experiment_id)
        required_artifacts_raw = entry.raw.get("required_artifacts", [])
        required_artifacts = (
            [str(path) for path in required_artifacts_raw]
            if isinstance(required_artifacts_raw, list)
            else []
        )

        item: dict[str, Any] = {
            "experiment_id": entry.experiment_id,
            "family": entry.family,
            "priority": entry.priority,
            "diagnostics_profile": entry.diagnostics_profile,
            "registry_entry": entry.raw,
            "run_dir": str(run_dir) if run_dir else None,
            "status": "available" if run_dir else "missing_run",
            "diagnostics": [],
            "required_artifacts": required_artifacts,
            "missing_required_artifacts": [],
        }

        if run_dir is None:
            missing.append(entry.experiment_id)
            if required_artifacts:
                item["missing_required_artifacts"] = required_artifacts
                missing_required_artifacts.append(
                    {
                        "experiment_id": entry.experiment_id,
                        "run_dir": None,
                        "missing": required_artifacts,
                    }
                )
            manifest["experiments"].append(item)
            continue

        if not dry_run:
            for split in splits:
                result = run_diagnostics(
                    run_dir,
                    split,
                    max_scatter_points=max_scatter_points,
                )
                item["diagnostics"].append(result)
                if result["returncode"] != 0:
                    failures.append(
                        {
                            "experiment_id": entry.experiment_id,
                            "run_dir": str(run_dir),
                            "split": split,
                            "returncode": result["returncode"],
                            "stderr": result["stderr"],
                        }
                    )

        missing_artifacts = [
            rel_path for rel_path in required_artifacts if not (run_dir / rel_path).exists()
        ]
        item["missing_required_artifacts"] = missing_artifacts

        if missing_artifacts:
            missing_required_artifacts.append(
                {
                    "experiment_id": entry.experiment_id,
                    "run_dir": str(run_dir),
                    "missing": missing_artifacts,
                }
            )

        manifest["experiments"].append(item)

    manifest["summary"] = {
        "selected_experiments": len(selected),
        "missing_runs": missing,
        "diagnostics_failures": failures,
        "missing_required_artifacts": missing_required_artifacts,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--priority", action="append", help="Priority filter, can repeat or pass space/comma-separated values")
    parser.add_argument("--family", action="append", help="Family filter, can repeat or pass space/comma-separated values")
    parser.add_argument("--profile", action="append", help="Diagnostics profile filter, can repeat or pass space/comma-separated values")
    parser.add_argument("--splits", default="test validation", help="Space-separated splits to build, default: 'test validation'")
    parser.add_argument("--max-scatter-points", type=int, default=20_000, help="Maximum points used by scatter diagnostics.")
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail if selected experiments have no run dir")
    parser.add_argument("--dry-run", action="store_true", help="Select runs and write manifest without invoking diagnostics builder")
    args = parser.parse_args()

    registry_path = args.registry.resolve()
    entries = load_registry(registry_path)
    selected = select_entries(
        entries,
        priorities=parse_list_arg(args.priority),
        families=parse_list_arg(args.family),
        profiles=parse_list_arg(args.profile),
    )
    if not selected:
        raise SystemExit("No experiments selected. Check --priority/--family/--profile filters.")

    splits = [s for s in args.splits.replace(",", " ").split() if s]
    if not splits:
        raise SystemExit("At least one split is required.")

    manifest = build_manifest(
        registry_path=registry_path,
        selected=selected,
        runs_dir=args.runs_dir.resolve(),
        splits=splits,
        allow_missing=args.allow_missing,
        dry_run=args.dry_run,
        max_scatter_points=args.max_scatter_points,
    )  # diagnostics run in here

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "diagnostics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary = manifest["summary"]
    print(json.dumps({"manifest": str(manifest_path), **summary}, indent=2, sort_keys=True))

    if summary["missing_runs"] and not args.allow_missing:
        raise SystemExit(f"Missing required run dirs: {summary['missing_runs']}")
    if summary["diagnostics_failures"]:
        raise SystemExit("One or more diagnostics builds failed; see manifest for details.")
    if summary["missing_required_artifacts"]:
        raise SystemExit("One or more required artifacts are missing; see manifest for details.")
    
    
if __name__ == "__main__":
    main()
