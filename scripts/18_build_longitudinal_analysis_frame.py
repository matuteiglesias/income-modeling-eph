"""Build the governed 2017-Q1..2026-Q1 longitudinal EPH analysis frame."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eph_income.longitudinal_frame import LongitudinalFrameError, build_longitudinal_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-lock", type=Path, required=True)
    parser.add_argument("--pinned-root", type=Path, required=True)
    parser.add_argument("--monetary-conversion", type=Path, required=True)
    parser.add_argument(
        "--harmonization",
        type=Path,
        default=ROOT / "configs" / "longitudinal_eph_harmonization_v1.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--monetary-reference-period",
        required=True,
        help="Common real-income reference month, YYYY-MM-01, present in the conversion release.",
    )
    parser.add_argument(
        "--allow-candidate-conversion",
        action="store_true",
        help=(
            "Permit a candidate/reviewed IPC conversion release for bounded evidence only. "
            "Approved conversion is required by default."
        ),
    )
    args = parser.parse_args()
    try:
        result = build_longitudinal_frame(
            args.parent_lock,
            args.pinned_root,
            args.monetary_conversion,
            args.harmonization,
            args.output_root,
            monetary_reference_period=args.monetary_reference_period,
            require_approved_conversion=not args.allow_candidate_conversion,
        )
    except (LongitudinalFrameError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
