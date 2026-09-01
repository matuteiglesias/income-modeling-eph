"""Build the governed EPH income-study cohort from a neutral frame."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eph_income.income_study import IncomeStudyError, build_income_study_cohort


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-frame", type=Path, required=True)
    parser.add_argument("--monetary-conversion", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/income-study-cohorts"),
    )
    args = parser.parse_args()

    try:
        release = build_income_study_cohort(
            args.analysis_frame,
            args.monetary_conversion,
            args.output_root,
        )
    except (IncomeStudyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(release)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
