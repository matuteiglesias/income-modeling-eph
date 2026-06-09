#!/usr/bin/env python3
"""Print a pre-flight fit plan for an EPH income experiment config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eph_income.config import load_experiment_config
from eph_income.experiment_planning import (
    build_experiment_fit_plan,
    format_experiment_fit_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Experiment YAML config path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the plan as JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--fail-if-expensive",
        action="store_true",
        help="Exit with status 3 if the plan requires --allow-expensive-run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_config = load_experiment_config(args.config)
    plan = build_experiment_fit_plan(experiment_config)

    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(format_experiment_fit_plan(plan))

    if args.fail_if_expensive and plan["requires_allow_expensive_run"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
