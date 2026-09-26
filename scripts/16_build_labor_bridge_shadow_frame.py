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

from eph_income.entity_identity import (  # noqa: E402
    EPH_GLOBAL_PERSON_KEY,
    validate_eph_person_identity,
)
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
    parser.add_argument(
        "--identity-sidecar",
        type=Path,
        help=(
            "Exact modeling row_id -> EPH source identity sidecar. Required when "
            "the estimator-facing frame intentionally excludes source identifiers."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--key-columns",
        type=csv_list,
        default=list(EPH_GLOBAL_PERSON_KEY),
    )
    args = parser.parse_args()

    frame = load_frame(args.frame)
    probabilities = load_frame(args.probabilities)

    missing_frame_keys = [column for column in args.key_columns if column not in frame.columns]
    temporary_keys: list[str] = []
    if missing_frame_keys:
        if args.identity_sidecar is None:
            raise ValueError(
                "income frame lacks exact source join keys "
                f"{missing_frame_keys}; provide --identity-sidecar. "
                "Row-order and fuzzy joins are forbidden."
            )
        identity = load_frame(args.identity_sidecar)
        required_identity = {"row_id", *args.key_columns}
        missing_identity = sorted(required_identity - set(identity.columns))
        if missing_identity:
            raise ValueError(
                f"identity sidecar missing required columns: {missing_identity}"
            )
        if "row_id" not in frame.columns:
            raise ValueError("income frame lacks row_id required for identity sidecar")
        if frame["row_id"].duplicated().any() or identity["row_id"].duplicated().any():
            raise ValueError("row_id must be unique in frame and identity sidecar")
        left = set(frame["row_id"].astype(str))
        right = set(identity["row_id"].astype(str))
        if left != right:
            raise ValueError(
                "identity sidecar row_id coverage mismatch: "
                f"missing={len(left-right)} extra={len(right-left)}"
            )
        identity = identity[["row_id", *args.key_columns]].copy()
        validate_eph_person_identity(
            identity, require_period=True, context="modeling identity sidecar"
        )
        existing_keys = [column for column in args.key_columns if column in frame.columns]
        if existing_keys:
            check = frame[["row_id", *existing_keys]].merge(
                identity[["row_id", *existing_keys]],
                on="row_id",
                how="inner",
                suffixes=("_frame", "_identity"),
                validate="one_to_one",
            )
            for column in existing_keys:
                if not check[f"{column}_frame"].astype(str).equals(
                    check[f"{column}_identity"].astype(str)
                ):
                    raise ValueError(
                        f"income frame identity column disagrees with sidecar: {column}"
                    )
        temporary_keys = [column for column in args.key_columns if column not in frame.columns]
        frame = frame.merge(
            identity[["row_id", *temporary_keys]],
            on="row_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )

    validate_eph_person_identity(
        frame, require_period=True, context="L4 income frame"
    )
    joined = attach_probability_features(
        frame, probabilities, key_columns=args.key_columns
    )
    if temporary_keys:
        joined = joined.drop(columns=temporary_keys)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(args.output, index=False)
    print(json.dumps({"output": str(args.output), "rows": len(joined)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
