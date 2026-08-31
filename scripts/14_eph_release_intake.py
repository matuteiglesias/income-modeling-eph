#!/usr/bin/env python
"""Verify and pin one exact durable EPH parent release without preprocessing it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eph_income.eph_release_intake import IntakeError, verify_and_pin  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/upstream/eph"))
    parser.add_argument("--selection-mode", choices=["convergence", "reproduction"], default="convergence")
    parser.add_argument("--transport-tag")
    args = parser.parse_args()
    try:
        pinned = verify_and_pin(
            args.discovery,
            args.asset,
            args.output_root,
            selection_mode=args.selection_mode,
            transport_tag=args.transport_tag,
        )
    except (IntakeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    lock = json.loads((pinned / "parent_lock.json").read_text(encoding="utf-8"))
    print(json.dumps({"pinned_release": str(pinned), "parent_lock": lock}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
