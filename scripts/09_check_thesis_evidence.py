#!/usr/bin/env python3
"""Check that thesis-required runs and artifacts exist.

This script is intentionally read-only. It checks the experiment registry, finds
latest run directories, verifies required artifacts, and prints a concise table.
It exits nonzero if required evidence is missing unless --allow-missing is used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install project dependencies first.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "experiment_registry.yaml"
DEFAULT_RUNS_DIR = REPO_ROOT / "reports" / "runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "thesis_evidence"

DEFAULT_REQUIRED_ARTIFACTS = [
    "metrics/model_comparison.csv",
    "diagnostics/diagnostics_plan.json",
    "diagnostics/residual_summary.csv",
    "diagnostics/error_by_income_decile.csv",
    "diagnostics/prediction_distribution_summary.csv",
    "diagnostics/distribution_compression_summary.csv",
    "plots/validation/observed_vs_predicted_best_model.png",
    "plots/validation/prediction_distribution_by_model.png",
    "plots/validation/mae_by_income_decile.png",
    "plots/validation/mean_residual_by_income_decile.png",
]


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

    @property
    def required_artifacts(self) -> list[str]:
        artifacts = self.raw.get("required_artifacts")
        if isinstance(artifacts, list) and artifacts:
            return [str(a) for a in artifacts]
        return DEFAULT_REQUIRED_ARTIFACTS


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
        raise FileNotFoundError(f"Experiment registry not found: {path}")
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
        experiment_id = str(experiment_id)
        if experiment_id in seen:
            raise ValueError(f"Duplicate experiment_id in registry: {experiment_id}")
        seen.add(experiment_id)
        entries.append(ExperimentEntry(experiment_id, item))
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
    return sorted(candidates, key=lambda p: p.name)[-1] if candidates else None


def status_icon(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def check_entries(entries: list[ExperimentEntry], runs_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for entry in entries:
        run_dir = find_latest_run_dir(runs_dir, entry.experiment_id)
        run_exists = run_dir is not None
        if not run_exists:
            missing.append({"experiment_id": entry.experiment_id, "path": "<run_dir>"})
        artifact_checks: list[dict[str, Any]] = []
        for rel_artifact in entry.required_artifacts:
            exists = bool(run_dir and (run_dir / rel_artifact).exists())
            if not exists:
                missing.append(
                    {
                        "experiment_id": entry.experiment_id,
                        "path": str((run_dir / rel_artifact) if run_dir else rel_artifact),
                    }
                )
            artifact_checks.append({"path": rel_artifact, "exists": exists})
        checks.append(
            {
                "experiment_id": entry.experiment_id,
                "family": entry.family,
                "priority": entry.priority,
                "diagnostics_profile": entry.diagnostics_profile,
                "run_dir": str(run_dir) if run_dir else None,
                "run_exists": run_exists,
                "artifacts": artifact_checks,
            }
        )
    return {"created_at": utc_now(), "checks": checks, "missing": missing}


def print_table(report: dict[str, Any]) -> None:
    rows = []
    for item in report["checks"]:
        total = len(item["artifacts"])
        present = sum(1 for a in item["artifacts"] if a["exists"])
        rows.append(
            [
                item["experiment_id"],
                item.get("family") or "",
                item.get("priority") or "",
                status_icon(item["run_exists"]),
                f"{present}/{total}",
            ]
        )
    headers = ["experiment", "family", "priority", "run", "artifacts"]
    widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--priority", action="append")
    parser.add_argument("--family", action="append")
    parser.add_argument("--profile", action="append")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON report instead of table")
    args = parser.parse_args()

    entries = load_registry(args.registry.resolve())
    selected = select_entries(
        entries,
        priorities=parse_list_arg(args.priority),
        families=parse_list_arg(args.family),
        profiles=parse_list_arg(args.profile),
    )
    if not selected:
        raise SystemExit("No experiments selected. Check --priority/--family/--profile filters.")

    report = check_entries(selected, args.runs_dir.resolve())
    output_dir = args.output_dir.resolve()
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "manifests" / "thesis_evidence_check.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if args.json:
        print(json.dumps({"report": str(report_path), **report}, indent=2, sort_keys=True))
    else:
        print_table(report)
        print(f"\nReport: {report_path}")

    if report["missing"] and not args.allow_missing:
        raise SystemExit("Missing required thesis evidence; see report above.")


if __name__ == "__main__":
    main()
