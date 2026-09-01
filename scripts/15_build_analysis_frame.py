#!/usr/bin/env python3
"""Build one neutral EPH analysis-frame release from one pinned parent."""
from __future__ import annotations

import argparse
from pathlib import Path

from eph_income.analysis_frame import build_analysis_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-root",
        required=True,
        type=Path,
        help="Pinned EPH parent release directory containing parent_lock.json.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory in which the immutable neutral release will be created.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    release = build_analysis_frame(args.parent_root, args.output_root)
    print(release)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
