#!/usr/bin/env python3
"""Collect thesis-facing artifacts from latest experiment runs.

This script copies selected artifacts from run-local directories into
`reports/thesis_evidence/` so the thesis can consume stable paths. It does not
train models or build diagnostics. Run `07_build_thesis_diagnostics.py` first if
expected diagnostics/plots have not yet been generated.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    "metrics/metrics_long.csv",
    "diagnostics/diagnostics_plan.json",
    "diagnostics/residual_summary.csv",
    "diagnostics/error_by_income_decile.csv",
    "diagnostics/prediction_distribution_summary.csv",
    "diagnostics/distribution_compression_summary.csv",
    "plots/validation/observed_vs_predicted_best_model.png",
    "plots/validation/prediction_distribution_by_model.png",
    "plots/validation/mae_by_income_decile.png",
    "plots/validation/mean_residual_by_income_decile.png",
    "plots/validation/distribution_compression_by_model.png",
]

TABLE_EXTENSIONS = {".csv", ".tex", ".json", ".md"}
FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}


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


def safe_name(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


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
        if experiment_id in seen:
            raise ValueError(f"Duplicate experiment_id in registry: {experiment_id}")
        seen.add(str(experiment_id))
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
    return sorted(candidates, key=lambda p: p.name)[-1] if candidates else None


def destination_for(output_dir: Path, experiment_id: str, rel_path: str) -> Path:
    suffix = Path(rel_path).suffix.lower()
    stem = safe_name(f"{experiment_id}__{Path(rel_path).parent.as_posix().replace('/', '__')}__{Path(rel_path).stem}")
    filename = f"{stem}{suffix}"
    if suffix in FIGURE_EXTENSIONS:
        return output_dir / "figures" / filename
    if suffix in TABLE_EXTENSIONS:
        return output_dir / "tables" / filename
    return output_dir / "artifacts" / filename


def collect_artifacts(
    *,
    entries: list[ExperimentEntry],
    runs_dir: Path,
    output_dir: Path,
    allow_missing: bool,
    overwrite: bool,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at": utc_now(),
        "runs_dir": str(runs_dir),
        "output_dir": str(output_dir),
        "experiments": [],
    }
    missing: list[dict[str, str]] = []
    copied: list[dict[str, str]] = []

    for subdir in ["tables", "figures", "artifacts", "manifests"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    for entry in entries:
        run_dir = find_latest_run_dir(runs_dir, entry.experiment_id)
        exp_record: dict[str, Any] = {
            "experiment_id": entry.experiment_id,
            "family": entry.family,
            "priority": entry.priority,
            "diagnostics_profile": entry.diagnostics_profile,
            "run_dir": str(run_dir) if run_dir else None,
            "artifacts": [],
            "missing_artifacts": [],
        }
        if run_dir is None:
            missing.append({"experiment_id": entry.experiment_id, "path": "<run_dir>"})
            exp_record["status"] = "missing_run"
            manifest["experiments"].append(exp_record)
            continue

        exp_record["status"] = "available"
        for rel_artifact in entry.required_artifacts:
            source = run_dir / rel_artifact
            if not source.exists():
                missing.append({"experiment_id": entry.experiment_id, "path": str(source)})
                exp_record["missing_artifacts"].append(rel_artifact)
                continue
            dest = destination_for(output_dir, entry.experiment_id, rel_artifact)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and not overwrite:
                status = "exists"
            else:
                shutil.copy2(source, dest)
                status = "copied"
            artifact_record = {
                "source": str(source),
                "destination": str(dest),
                "relative_artifact": rel_artifact,
                "status": status,
            }
            copied.append(artifact_record)
            exp_record["artifacts"].append(artifact_record)
        manifest["experiments"].append(exp_record)

    manifest["summary"] = {
        "selected_experiments": len(entries),
        "copied_or_existing_artifacts": len(copied),
        "missing": missing,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--priority", action="append")
    parser.add_argument("--family", action="append")
    parser.add_argument("--profile", action="append")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing collected files")
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

    output_dir = args.output_dir.resolve()
    manifest = collect_artifacts(
        entries=selected,
        runs_dir=args.runs_dir.resolve(),
        output_dir=output_dir,
        allow_missing=args.allow_missing,
        overwrite=args.overwrite,
    )
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "thesis_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary = manifest["summary"]
    print(json.dumps({"manifest": str(manifest_path), **summary}, indent=2, sort_keys=True))

    if summary["missing"] and not args.allow_missing:
        raise SystemExit("Missing required runs/artifacts; see thesis evidence manifest.")


if __name__ == "__main__":
    main()
