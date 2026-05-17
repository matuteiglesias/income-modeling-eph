#!/usr/bin/env python
"""Build the State 3a target-filtered EPH modeling dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eph_income.config import (  # noqa: E402
    DEFAULT_EXPERIMENT_CONFIG_PATH,
    DEFAULT_FEATURE_CONTRACT_PATH,
    load_experiment_config,
    load_feature_contract,
)
from eph_income.contracts import validate_target_contract, validate_temporal_features  # noqa: E402
from eph_income.dataset import build_modeling_dataset, validate_dataset_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_EXPERIMENT_CONFIG_PATH,
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--feature-contract",
        default=DEFAULT_FEATURE_CONTRACT_PATH,
        help="Path to feature contract YAML config.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate configs and paths without reading full data or writing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = load_experiment_config(args.config)
    feature_contract = load_feature_contract(args.feature_contract)

    validate_target_contract(feature_contract)
    validate_temporal_features(feature_contract)

    if args.check_only:
        summary = validate_dataset_inputs(experiment_config)
        print(json.dumps({"check_only": True, **summary}, indent=2, sort_keys=True))
        return

    dataset, metadata = build_modeling_dataset(experiment_config, feature_contract)
    print(
        json.dumps(
            {
                "processed_dataset": metadata["processed_dataset"],
                "rows": int(len(dataset)),
                "metadata": metadata["metadata"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
