#!/usr/bin/env python3
"""Audit legacy EPH annual identity loss against source-backed person microdata.

The historical annual producer joined household/person data without NRO_HOGAR
and omitted COMPONENTE from its output. This audit quantifies the raw-person
surface where that reduced key is ambiguous. It never reconstructs identities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXACT_PERSON_KEY = ["CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE"]
LEGACY_JOIN_KEY = ["CODUSU", "ANO4", "TRIMESTRE", "AGLOMERADO"]


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, sep=";", low_memory=False)
    except Exception:
        return pd.read_csv(path, low_memory=False)


def audit(paths: list[Path]) -> tuple[dict[str, object], pd.DataFrame]:
    frames = []
    for path in paths:
        frame = load_table(path)
        required = set(EXACT_PERSON_KEY + ["AGLOMERADO"])
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing required EPH identity columns: {missing}")
        work = frame[EXACT_PERSON_KEY + ["AGLOMERADO"]].copy()
        work["source_file"] = str(path)
        frames.append(work)

    if not frames:
        raise ValueError("at least one --individual source is required")

    persons = pd.concat(frames, ignore_index=True, sort=False)
    if persons.duplicated(EXACT_PERSON_KEY).any():
        duplicates = int(persons.duplicated(EXACT_PERSON_KEY, keep=False).sum())
        raise ValueError(
            f"source person identity is not unique across inputs: duplicate_rows={duplicates}"
        )

    household_counts = (
        persons.groupby(LEGACY_JOIN_KEY, dropna=False)["NRO_HOGAR"]
        .nunique(dropna=False)
        .rename("distinct_nro_hogar")
        .reset_index()
    )
    ambiguous = household_counts[household_counts["distinct_nro_hogar"] > 1].copy()
    persons_with_flag = persons.merge(
        ambiguous[LEGACY_JOIN_KEY],
        on=LEGACY_JOIN_KEY,
        how="inner",
        validate="many_to_one",
    )

    exact_households = persons[
        ["CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", "AGLOMERADO"]
    ].drop_duplicates()
    legacy_groups = persons[LEGACY_JOIN_KEY].drop_duplicates()

    summary = {
        "contract": "research.eph-legacy-identity-gap-audit/v1",
        "person_rows": int(len(persons)),
        "exact_households": int(len(exact_households)),
        "legacy_join_groups": int(len(legacy_groups)),
        "ambiguous_legacy_join_groups": int(len(ambiguous)),
        "persons_in_ambiguous_legacy_join_groups": int(len(persons_with_flag)),
        "share_persons_in_ambiguous_groups": (
            float(len(persons_with_flag) / len(persons)) if len(persons) else 0.0
        ),
        "max_distinct_nro_hogar_per_legacy_group": (
            int(household_counts["distinct_nro_hogar"].max())
            if len(household_counts)
            else 0
        ),
        "exact_person_key": EXACT_PERSON_KEY,
        "legacy_join_key": LEGACY_JOIN_KEY,
        "interpretation": (
            "Any ambiguous legacy join group contains more than one true EPH household "
            "under the same reduced legacy key. The old annual artifact cannot prove "
            "which household row supplied covariates to each person in those groups."
        ),
    }
    return summary, ambiguous.sort_values(LEGACY_JOIN_KEY).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--individual",
        type=Path,
        action="append",
        required=True,
        help="Raw/source-backed EPH individual file; repeat for multiple quarters.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-ambiguous-csv", type=Path, required=True)
    args = parser.parse_args()

    summary, ambiguous = audit(args.individual)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_ambiguous_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ambiguous.to_csv(args.output_ambiguous_csv, index=False)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
