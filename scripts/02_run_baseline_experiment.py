#!/usr/bin/env python
"""Run a configured baseline/debug model-comparison experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eph_income.config import (  # noqa: E402
    DEFAULT_FEATURE_CONTRACT_PATH,
    load_experiment_config,
    load_feature_contract,
)
from eph_income.experiments import run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=REPO_ROOT / "configs" / "experiment_baseline.yaml",
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--feature-contract",
        default=DEFAULT_FEATURE_CONTRACT_PATH,
        help="Path to feature contract YAML config.",
    )
    parser.add_argument(
        "--allow-full-run",
        action="store_true",
        help="Allow runtime.mode=full experiments; required unless the config explicitly allows them.",
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=float,
        default=None,
        help="Seconds between long-fit heartbeat messages (default: config value or 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = load_experiment_config(args.config)
    if args.heartbeat_interval_seconds is not None:
        experiment_config = dict(experiment_config)
        runtime = dict(experiment_config.get("runtime", {}))
        runtime["heartbeat_interval_seconds"] = args.heartbeat_interval_seconds
        experiment_config["runtime"] = runtime
    feature_contract = load_feature_contract(args.feature_contract)
    comparison, card = run_experiment(
        experiment_config, feature_contract, allow_full_run=args.allow_full_run
    )
    print(
        json.dumps(
            {
                "rows": int(card["rows_used"]),
                "models": comparison["model"].tolist(),
                "model_comparison": card["model_comparison"],
                "run_dir": card["canonical_run_dir"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
