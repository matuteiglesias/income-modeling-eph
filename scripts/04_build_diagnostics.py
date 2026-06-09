#!/usr/bin/env python
"""Build scientific diagnostics and first plots from an archived run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eph_income.diagnostics import build_diagnostics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to an existing reports/runs/<run_id>/ directory.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to use for split-specific plots. Default: test.",
    )
    parser.add_argument("--max-scatter-points", type=int, default=20_000)
    parser.add_argument("--min-distribution-rows", type=int, default=100)
    parser.add_argument("--min-decile-points", type=int, default=3)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_diagnostics(
    args.run_dir,
    split=args.split,
    max_scatter_points=args.max_scatter_points,
    min_distribution_rows=args.min_distribution_rows,
    min_decile_points=args.min_decile_points,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
