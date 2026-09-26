#!/usr/bin/env python3
"""Fit/cross-fit the two-stage EPH labor bridge and emit an OOF probability release."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from eph_income.entity_identity import (  # noqa: E402
    EPH_GLOBAL_PERSON_KEY,
    validate_eph_person_identity,
)
from eph_income.labor_bridge import (  # noqa: E402
    LaborFeatureContract,
    TwoStageLaborBridge,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def load_contract(path: Path) -> LaborFeatureContract:
    data = yaml.safe_load(path.read_text())
    return LaborFeatureContract(
        categorical=tuple(data.get("categorical", [])),
        numeric=tuple(data.get("numeric", [])),
    )


def run(
    data_path: Path,
    output: Path,
    contract_path: Path,
    *,
    classifier_kind: str,
    weight_mode: str,
    key_columns: list[str],
    group_columns: list[str] | None,
    cv: int,
) -> dict[str, object]:
    frame = load_frame(data_path)
    contract = load_contract(contract_path)
    weight_col = None if weight_mode == "none" else "PONDERA"

    bridge = TwoStageLaborBridge(
        contract,
        classifier_kind=classifier_kind,
        cv=cv,
        random_state=42,
    )
    oof = bridge.fit_oof(
        frame,
        state_col="ESTADO",
        weight_col=weight_col,
        group_columns=group_columns,
    )
    missing_keys = sorted(set(key_columns) - set(frame.columns))
    if missing_keys:
        raise ValueError(f"input missing output identity columns: {missing_keys}")
    release = frame.loc[oof.index, key_columns].copy()
    release = release.join(oof)
    identity_audit = validate_eph_person_identity(
        release, require_period=True, context="EPH labor OOF release"
    )
    output.mkdir(parents=True, exist_ok=False)

    probabilities_path = output / "eph_labor_probabilities_oof.parquet"
    model_path = output / "labor_bridge_model.joblib"
    release.to_parquet(probabilities_path, index=False)
    bridge.save(model_path)

    active_target = release["observed_active"].to_numpy(dtype=int)
    unemployment_target = release["observed_unemployed"].to_numpy(dtype=int)
    active_mask = active_target == 1
    qa = {
        "contract": "research.eph-labor-probabilities-oof/v1",
        "classifier_kind": classifier_kind,
        "weight_mode": weight_mode,
        "cv": cv,
        "rows": len(release),
        "active_brier": float(
            brier_score_loss(active_target, release["labor_p_active"])
        ),
        "conditional_unemployment_brier": float(
            brier_score_loss(
                unemployment_target[active_mask],
                release.loc[active_mask, "labor_p_unemployed_given_active"],
            )
        ),
        "observed_activity_rate_unweighted": float(active_target.mean()),
        "oof_activity_rate_unweighted": float(release["labor_p_active"].mean()),
        "observed_unemployment_rate_active_unweighted": float(
            unemployment_target[active_mask].mean()
        ),
        "oof_unemployment_rate_active_weighted_by_p_active": float(
            release["labor_p_unemployed"].sum() / release["labor_p_active"].sum()
        ),
        "probability_sum_max_abs_error": float(
            (
                release[
                    ["labor_p_employed", "labor_p_unemployed", "labor_p_inactive"]
                ].sum(axis=1)
                - 1.0
            ).abs().max()
        ),
    }
    (output / "qa.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n")

    manifest = {
        "contract": "research.eph-labor-probabilities-oof/v1",
        "release_id": output.name,
        "input": {
            "path": str(data_path.resolve()),
            "sha256": sha256(data_path),
            "rows": len(frame),
        },
        "feature_contract": {
            "path": str(contract_path.resolve()),
            "sha256": sha256(contract_path),
            "categorical": list(contract.categorical),
            "numeric": list(contract.numeric),
        },
        "classifier_kind": classifier_kind,
        "weight_mode": weight_mode,
        "cv": cv,
        "identity": {
            "person_key": list(EPH_GLOBAL_PERSON_KEY),
            "rows": identity_audit.rows,
            "unique": identity_audit.unique,
            "period_qualified": True,
        },
        "key_columns": key_columns,
        "group_columns": group_columns,
        "artifacts": {
            probabilities_path.name: {
                "sha256": sha256(probabilities_path),
                "bytes": probabilities_path.stat().st_size,
                "rows": len(release),
            },
            model_path.name: {
                "sha256": sha256(model_path),
                "bytes": model_path.stat().st_size,
            },
        },
        "qa": qa,
        "scientific_invariants": [
            "ESTADO=0 is excluded from supervised targets",
            "stage 1 models active versus non-active",
            "stage 2 models unemployment conditional on activity",
            "OOF probabilities are used for EPH rows",
            "observed labor labels are not predictor features",
            "no hard labor states are sampled",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--data", type=Path, required=True)
    out.add_argument("--output", type=Path, required=True)
    out.add_argument(
        "--feature-contract",
        type=Path,
        default=ROOT / "configs" / "labor_bridge_feature_contract.yaml",
    )
    out.add_argument(
        "--classifier-kind",
        choices=["logistic", "hist_gradient_boosting"],
        default="logistic",
    )
    out.add_argument("--weight-mode", choices=["none", "pondera"], default="none")
    out.add_argument(
        "--key-columns",
        type=csv_list,
        default=list(EPH_GLOBAL_PERSON_KEY),
    )
    out.add_argument(
        "--group-columns",
        type=csv_list,
        default=csv_list("CODUSU,NRO_HOGAR"),
    )
    out.add_argument("--cv", type=int, default=5)
    return out


def main() -> int:
    args = parser().parse_args()
    manifest = run(
        args.data,
        args.output,
        args.feature_contract,
        classifier_kind=args.classifier_kind,
        weight_mode=args.weight_mode,
        key_columns=args.key_columns,
        group_columns=args.group_columns,
        cv=args.cv,
    )
    print(json.dumps({"release": manifest["release_id"], "qa": manifest["qa"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
