"""Build the exact 2017-Q1..2026-Q1 EPH parent lock from pinned releases."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eph_income.longitudinal_frame import LongitudinalFrameError, build_parent_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pinned-root",
        type=Path,
        required=True,
        help="Directory containing exact pinned EPH parent release directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the immutable 37-quarter parent lock JSON.",
    )
    args = parser.parse_args()
    try:
        result = build_parent_ledger(args.pinned_root, args.output)
    except (LongitudinalFrameError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
