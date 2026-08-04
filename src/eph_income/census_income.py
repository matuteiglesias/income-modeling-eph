"""Bounded, auditable Census income inference and release utilities.

This module is deliberately separate from the frozen EPH estimator.  Census income
is produced only by an explicitly declared sequence of classifier stages followed
by a regressor, or by packaging a fully matched historical output.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict

TRANSFORMS = {"linear_ars", "log10_ars", "log10_ars_plus_1"}
CANONICAL_COLUMNS = [
    "sample_person_id", "period", "prediction_value", "prediction_transform",
    "monetary_reference", "classification", "model_release_id",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id_hash(values: Iterable[object]) -> str:
    normalized = "\n".join(sorted(str(value) for value in values)) + "\n"
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass(frozen=True)
class StageSpec:
    """One classifier wave whose predictions become downstream features."""

    name: str
    target: str
    output_feature: str
    features: tuple[str, ...]


class StagedClassifierRegressor:
    """Fit classifier waves and a final regressor without duplicating pipelines.

    Out-of-fold classifier predictions are used when fitting later stages and the
    regressor.  Each classifier is then refit on all training rows for inference.
    This prevents in-sample target labels from leaking into downstream training.
    """

    def __init__(self, stages: list[tuple[StageSpec, Any]], regressor: Any,
                 regression_features: list[str], cv: int = 3, random_state: int = 42):
        self.stages = stages
        self.regressor = regressor
        self.regression_features = regression_features
        self.cv = cv
        self.random_state = random_state

    @staticmethod
    def _require(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{context} missing required columns: {missing}")

    def fit(self, frame: pd.DataFrame, regression_target: str):
        work = frame.copy()
        self.fitted_stages_ = []
        splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for spec, estimator in self.stages:
            self._require(work, [*spec.features, spec.target], f"stage {spec.name}")
            model = clone(estimator)
            work[spec.output_feature] = cross_val_predict(
                model, work[list(spec.features)], work[spec.target], cv=splitter, method="predict"
            )
            model.fit(work[list(spec.features)], work[spec.target])
            self.fitted_stages_.append((spec, model))
        self._require(work, [*self.regression_features, regression_target], "regression stage")
        self.regressor_ = clone(self.regressor).fit(
            work[self.regression_features], work[regression_target]
        )
        self.regression_target_ = regression_target
        return self

    def transform_stages(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "fitted_stages_"):
            raise ValueError("Staged model has not been fitted")
        work = frame.copy()
        for spec, model in self.fitted_stages_:
            self._require(work, spec.features, f"stage {spec.name}")
            work[spec.output_feature] = model.predict(work[list(spec.features)])
        return work

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        work = self.transform_stages(frame)
        self._require(work, self.regression_features, "regression stage")
        return np.asarray(self.regressor_.predict(work[self.regression_features]), dtype=float)


def validate_predictions(sample: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    """Enforce exact Census namespace, period, transform, and finite-value coverage."""
    required_sample = {"sample_person_id", "period"}
    if missing := required_sample - set(sample.columns):
        raise ValueError(f"Census sample missing columns: {sorted(missing)}")
    if missing := set(CANONICAL_COLUMNS) - set(predictions.columns):
        raise ValueError(f"Predictions missing canonical columns: {sorted(missing)}")
    for name, frame in (("sample", sample), ("predictions", predictions)):
        if frame.duplicated(["sample_person_id", "period"]).any():
            raise ValueError(f"Duplicate person-period keys in {name}")
    sample_keys = set(map(tuple, sample[["sample_person_id", "period"]].astype(str).to_numpy()))
    pred_keys = set(map(tuple, predictions[["sample_person_id", "period"]].astype(str).to_numpy()))
    if sample_keys != pred_keys:
        raise ValueError(
            f"Person-period coverage mismatch: missing={len(sample_keys-pred_keys)}, "
            f"extra={len(pred_keys-sample_keys)}"
        )
    transforms = set(predictions["prediction_transform"])
    if len(transforms) != 1 or not transforms <= TRANSFORMS:
        raise ValueError(f"Unknown or contradictory transform: {sorted(map(str, transforms))}")
    if set(predictions["classification"]) != {"projected"}:
        raise ValueError("Census income classification must be projected")
    if predictions["monetary_reference"].nunique(dropna=False) != 1:
        raise ValueError("Contradictory monetary references")
    if predictions["model_release_id"].isna().any():
        raise ValueError("Missing model release identity")
    values = pd.to_numeric(predictions["prediction_value"], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Prediction values contain NaN or infinity")
    return {
        "row_count": len(predictions),
        "id_namespace_hash": stable_id_hash(predictions["sample_person_id"]),
        "periods": sorted(map(str, predictions["period"].unique())),
        "transform": next(iter(transforms)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "negative_count": int((values < 0).sum()),
    }


def load_census_release(release_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    manifest_path = release_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing Census release manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("contract") != "research.census-sample/v1":
        raise ValueError("Input is not a research.census-sample/v1 release")
    relative = manifest.get("data_file", "census_sample.csv")
    data_path = release_dir / relative
    frame = pd.read_csv(data_path, dtype={"sample_person_id": str})
    expected = manifest.get("data_sha256")
    if expected and sha256(data_path) != expected:
        raise ValueError("Census sample checksum mismatch")
    return frame, manifest, data_path


def write_release(sample: pd.DataFrame, predictions: pd.DataFrame, census_manifest: dict[str, Any],
                  census_manifest_path: Path, output_dir: Path, *, source_paths: list[Path],
                  stage_identities: list[dict[str, Any]], monetary_status: str,
                  warnings: list[str]) -> Path:
    qa = validate_predictions(sample, predictions)
    output_dir.mkdir(parents=True, exist_ok=False)
    ordered = predictions[CANONICAL_COLUMNS].sort_values(
        ["period", "sample_person_id"], kind="stable"
    )
    output_path = output_dir / "person_income_predictions.csv"
    ordered.to_csv(output_path, index=False)
    qa_path = output_dir / "prediction_qa.json"
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n")
    joined = sample.merge(
        ordered[["sample_person_id", "period", "prediction_value"]],
        on=["sample_person_id", "period"], validate="one_to_one",
    )
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir()
    native = ordered["prediction_value"].astype(float)
    transform = qa["transform"]
    if transform == "linear_ars":
        linear = native
    elif transform == "log10_ars":
        linear = np.power(10.0, native)
    else:
        linear = np.power(10.0, native) - 1.0
    pd.DataFrame({
        "scale": ["native", "linear"], "minimum": [native.min(), linear.min()],
        "q01": [native.quantile(.01), linear.quantile(.01)],
        "q25": [native.quantile(.25), linear.quantile(.25)],
        "median": [native.median(), linear.median()],
        "q75": [native.quantile(.75), linear.quantile(.75)],
        "q99": [native.quantile(.99), linear.quantile(.99)],
        "maximum": [native.max(), linear.max()],
    }).to_csv(diagnostics_dir / "prediction_distribution.csv", index=False)
    for column in ("region", "department", "sex"):
        if column in joined:
            joined.groupby(column, dropna=False)["prediction_value"].agg(
                ["count", "mean", "min", "max"]
            ).reset_index().to_csv(diagnostics_dir / f"predictions_by_{column}.csv", index=False)
    if "age" in joined:
        age = pd.to_numeric(joined["age"], errors="coerce")
        joined.assign(age_group=pd.cut(age, [-np.inf, 17, 24, 44, 64, np.inf])).groupby(
            "age_group", observed=False, dropna=False
        )["prediction_value"].agg(["count", "mean", "min", "max"]).reset_index().to_csv(
            diagnostics_dir / "predictions_by_age.csv", index=False
        )
    pd.DataFrame({
        "check": ["sample_rows", "prediction_rows", "matched_rows"],
        "count": [len(sample), len(ordered), len(joined)],
    }).to_csv(diagnostics_dir / "id_coverage.csv", index=False)
    for values, name, label in ((native, "native", "Prediction (native scale)"),
                                (linear, "linear", "Prediction (ARS scale)")):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist(values, bins=min(30, max(1, len(values))), edgecolor="white")
        axis.set_xlabel(label)
        axis.set_ylabel("Rows")
        axis.set_title(f"Census income predictions — {name} scale")
        figure.tight_layout()
        figure.savefig(diagnostics_dir / f"prediction_distribution_{name}.png", dpi=120)
        plt.close(figure)
    manifest = {
        "contract": "research.person-income-predictions/v1",
        "release_id": output_dir.name,
        "census_release_id": census_manifest.get("release_id"),
        "census_manifest_sha256": sha256(census_manifest_path),
        "sample_id_namespace": census_manifest.get("sample_id_namespace"),
        "id_namespace_hash": qa["id_namespace_hash"],
        "period": qa["periods"],
        "prediction_transform": qa["transform"],
        "currency": "ARS",
        "monetary_reference": str(ordered["monetary_reference"].iloc[0]),
        "monetary_reference_status": monetary_status,
        "classification": "projected",
        "model_stages": stage_identities,
        "source_artifacts": [
            {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
        "output_file": output_path.name,
        "output_sha256": sha256(output_path),
        "row_count": len(ordered),
        "qa_file": qa_path.name,
        "diagnostics_directory": diagnostics_dir.name,
        "warnings": warnings,
        "limitations": [
            "Census projections are predictive, not causal.",
            "EPH validation does not establish Census accuracy.",
            "Retransformation bias is unresolved unless separately documented.",
            "The frozen EPH flagship estimator was not used for Census inference.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "HANDOFF.md").write_text(
        "# Census person-income prediction handoff\n\n"
        f"Release: `{output_dir}`  \nManifest SHA-256: `{sha256(manifest_path)}`  \n"
        f"Rows: {len(ordered)}  \nTransform: `{qa['transform']}`  \n"
        f"Monetary status: `{monetary_status}`\n\n"
        "The IDs exactly cover the declared Census release. The frozen EPH flagship model "
        "was not applied to Census rows. Read `manifest.json` warnings before poverty use.\n"
    )
    return manifest_path


def save_staged_model(model: StagedClassifierRegressor, path: Path) -> None:
    """Serialize only an explicitly fitted staged model artifact."""
    if not hasattr(model, "regressor_"):
        raise ValueError("Cannot save an unfitted staged model")
    joblib.dump(model, path)
