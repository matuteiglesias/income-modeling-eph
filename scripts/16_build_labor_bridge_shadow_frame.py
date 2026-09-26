#!/usr/bin/env python3
"""Attach exact OOF labor probability features to an EPH income-modeling frame."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from eph_income.labor_bridge import attach_probability_features  # noqa: E402


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--key-columns",
        type=csv_list,
        default=csv_list("CODUSU,NRO_HOGAR,COMPONENTE,ANO4,TRIMESTRE"),
    )
    args = parser.parse_args()

    frame = load_frame(args.frame)
    probabilities = load_frame(args.probabilities)
    joined = attach_probability_features(
        frame, probabilities, key_columns=args.key_columns
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(args.output, index=False)
    print(json.dumps({"output": str(args.output), "rows": len(joined)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
