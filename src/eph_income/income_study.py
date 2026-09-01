"""Build the governed EPH-only income-study cohort from a neutral EPH frame.

The neutral ``research.eph-analysis-frame@1`` deliberately contains native
monetary values and no study cohort.  This module owns the historical study
semantics recorded in ``configs/feature_contract.yaml``:

    INGRESO == 1
    P47T > 0
    PROP not missing
    -> log10(P47T)

For new source-backed vintages, ``INGRESO`` can only be reconstructed after an
*approved* ``research.argentina-monetary-conversion/v1`` release converts the
native quarter amount to the January-2016 analytical reference used by the
historical study.  Candidate price releases fail closed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ANALYSIS_FRAME_CONTRACT = "research.eph-analysis-frame@1"
CONVERSION_CONTRACT = "research.argentina-monetary-conversion/v1"
COHORT_CONTRACT = "research.eph-income-study-cohort@1"
EXPECTED_MONETARY_REFERENCE = (
    "research.argentina-price-consensus/curated-official-panel-v2@2016-01=100"
)
HOUSEHOLD_KEY = ["CODUSU", "NRO_HOGAR"]
PERSON_KEY = ["CODUSU", "NRO_HOGAR", "COMPONENTE"]
SURVEY_DESIGN_FIELDS = ["PONDERA", "PONDIIO", "PONDII", "PONDIH"]
PROP_RECODE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 6, 9: 6}
QUARTER_REFERENCE_MONTH = {1: 2, 2: 5, 3: 8, 4: 11}


class IncomeStudyError(ValueError):
    """Raised when the income-study cohort cannot be reproduced safely."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IncomeStudyError(reason) from exc
    if not isinstance(value, dict):
        raise IncomeStudyError(reason)
    return value


def _manifest_file_hash(manifest: dict[str, Any], filename: str) -> str:
    files = manifest.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            path = item.get("path") or item.get("file") or item.get("name")
            if path == filename and isinstance(item.get("sha256"), str):
                return item["sha256"]
    elif isinstance(files, dict):
        item = files.get(filename)
        if isinstance(item, dict) and isinstance(item.get("sha256"), str):
            return item["sha256"]
    raise IncomeStudyError(f"conversion_manifest_missing_file_hash:{filename}")


def validate_conversion_release(conversion_root: Path) -> dict[str, Any]:
    """Validate the minimum consumer contract for an approved monetary release."""
    root = Path(conversion_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path, "missing_or_invalid_conversion_manifest")
    if manifest.get("schema") != "research-artifact-manifest/v1":
        raise IncomeStudyError("unexpected_conversion_manifest_schema")
    if manifest.get("artifact_type") != CONVERSION_CONTRACT:
        raise IncomeStudyError("unexpected_conversion_artifact_type")
    if manifest.get("status") != "approved":
        raise IncomeStudyError(
            f"conversion_release_not_approved:{manifest.get('status')}"
        )
    if manifest.get("monetary_reference_id") != EXPECTED_MONETARY_REFERENCE:
        raise IncomeStudyError("unexpected_monetary_reference")

    factors_path = root / "monthly_conversion_factors.csv"
    if not factors_path.is_file():
        raise IncomeStudyError("missing_monthly_conversion_factors")
    declared_hash = _manifest_file_hash(manifest, factors_path.name)
    if _sha256(factors_path) != declared_hash:
        raise IncomeStudyError("conversion_factor_hash_mismatch")
    return manifest


def _parse_quarter(period: dict[str, Any]) -> tuple[int, int]:
    try:
        year = int(period["year"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IncomeStudyError("analysis_frame_period_year_invalid") from exc
    raw_quarter = period.get("quarter")
    if isinstance(raw_quarter, str):
        value = raw_quarter.strip().upper()
        if value.startswith("Q"):
            value = value[1:]
        try:
            quarter = int(value)
        except ValueError as exc:
            raise IncomeStudyError("analysis_frame_period_quarter_invalid") from exc
    else:
        try:
            quarter = int(raw_quarter)
        except (TypeError, ValueError) as exc:
            raise IncomeStudyError("analysis_frame_period_quarter_invalid") from exc
    if quarter not in QUARTER_REFERENCE_MONTH:
        raise IncomeStudyError("analysis_frame_period_quarter_invalid")
    return year, quarter


def _quarter_reference_period(period: dict[str, Any]) -> str:
    year, quarter = _parse_quarter(period)
    month = QUARTER_REFERENCE_MONTH[quarter]
    return f"{year:04d}-{month:02d}-01"


def _read_conversion_factor(conversion_root: Path, period: str) -> dict[str, Any]:
    path = Path(conversion_root) / "monthly_conversion_factors.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row.get("period") == period]
    if len(matches) != 1:
        raise IncomeStudyError(f"conversion_period_not_unique_or_missing:{period}")
    row = matches[0]
    if row.get("reference_period") != "2016-01-01":
        raise IncomeStudyError("conversion_row_reference_period_mismatch")
    eligible = str(row.get("approved_mode_eligible", "")).strip().lower()
    if eligible not in {"true", "1", "yes"}:
        raise IncomeStudyError(f"conversion_period_not_approved_mode_eligible:{period}")
    try:
        factor = float(row["factor_period_to_reference"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IncomeStudyError("invalid_period_to_reference_factor") from exc
    if not math.isfinite(factor) or factor <= 0:
        raise IncomeStudyError("invalid_period_to_reference_factor")
    return {
        "period": period,
        "reference_period": row["reference_period"],
        "factor_period_to_reference": factor,
        "coverage_class": row.get("coverage_class"),
        "approved_mode_eligible": True,
    }


def _validate_analysis_frame(root: Path) -> dict[str, Any]:
    manifest = _load_json(root / "manifest.json", "missing_or_invalid_analysis_frame_manifest")
    if manifest.get("contract") != ANALYSIS_FRAME_CONTRACT:
        raise IncomeStudyError("unexpected_analysis_frame_contract")
    if manifest.get("identity", {}).get("household_key") != HOUSEHOLD_KEY:
        raise IncomeStudyError("analysis_frame_household_identity_mismatch")
    if manifest.get("identity", {}).get("person_key") != PERSON_KEY:
        raise IncomeStudyError("analysis_frame_person_identity_mismatch")
    policy = manifest.get("field_policy") or {}
    if policy.get("survey_weights_applied") is not False:
        raise IncomeStudyError("analysis_frame_already_applied_survey_weights")
    if policy.get("target_or_cohort_fields_added") is not False:
        raise IncomeStudyError("analysis_frame_is_not_neutral")
    monetary = manifest.get("monetary_lineage") or {}
    if monetary.get("deflation_or_rebasing_applied") is not False:
        raise IncomeStudyError("analysis_frame_monetary_values_are_not_native")
    return manifest


def build_income_study_cohort(
    analysis_frame_root: Path,
    conversion_root: Path,
    output_root: Path,
) -> Path:
    """Build the historical-v1 EPH income-study cohort from a neutral frame."""
    frame_root = Path(analysis_frame_root).resolve()
    conversion_root = Path(conversion_root).resolve()
    output_root = Path(output_root).resolve()
    frame_manifest = _validate_analysis_frame(frame_root)
    conversion_manifest = validate_conversion_release(conversion_root)

    households_path = frame_root / "households.csv"
    persons_path = frame_root / "persons.csv"
    if not households_path.is_file() or not persons_path.is_file():
        raise IncomeStudyError("analysis_frame_payload_missing")
    households = pd.read_csv(households_path, low_memory=False)
    persons = pd.read_csv(persons_path, low_memory=False)

    required_household = {*HOUSEHOLD_KEY, "II7"}
    required_person = {*PERSON_KEY, "P47T"}
    missing_household = sorted(required_household - set(households.columns))
    missing_person = sorted(required_person - set(persons.columns))
    if missing_household:
        raise IncomeStudyError(
            "analysis_frame_missing_household_fields:" + ",".join(missing_household)
        )
    if missing_person:
        raise IncomeStudyError(
            "analysis_frame_missing_person_fields:" + ",".join(missing_person)
        )
    if households.duplicated(HOUSEHOLD_KEY).any():
        raise IncomeStudyError("analysis_frame_household_key_not_unique")
    if persons.duplicated(PERSON_KEY).any():
        raise IncomeStudyError("analysis_frame_person_key_not_unique")

    household_context = households.loc[:, [*HOUSEHOLD_KEY, "II7"]].copy()
    ii7_numeric = pd.to_numeric(household_context["II7"], errors="coerce")
    household_context["PROP"] = ii7_numeric.map(PROP_RECODE)
    household_context = household_context.drop(columns=["II7"])
    joined = persons.merge(
        household_context,
        on=HOUSEHOLD_KEY,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        raise IncomeStudyError("income_study_person_without_household_context")
    joined = joined.drop(columns=["_merge"])

    parent_period = (frame_manifest.get("parent") or {}).get("period")
    if not isinstance(parent_period, dict):
        raise IncomeStudyError("analysis_frame_parent_period_missing")
    reference_period = _quarter_reference_period(parent_period)
    factor_row = _read_conversion_factor(conversion_root, reference_period)
    factor = factor_row["factor_period_to_reference"]

    native_income = pd.to_numeric(joined["P47T"], errors="coerce")
    finite_native = native_income.notna() & np.isfinite(native_income)
    study_income = pd.Series(np.nan, index=joined.index, dtype="float64")
    study_income.loc[finite_native] = np.rint(native_income.loc[finite_native] * factor)
    joined["P47T_native"] = native_income
    joined["P47T"] = study_income
    joined["INGRESO"] = (joined["P47T"] > 100).astype("int8")

    ingreso_ok = joined["INGRESO"] == 1
    positive_ok = joined["P47T"] > 0
    prop_ok = joined["PROP"].notna()
    eligible = ingreso_ok & positive_ok & prop_ok
    cohort = joined.loc[eligible].copy()
    if cohort.empty:
        raise IncomeStudyError("income_study_cohort_empty")
    cohort["logP47T"] = np.log10(cohort["P47T"])

    output_columns = [*PERSON_KEY]
    for optional in ("ANO4", "TRIMESTRE"):
        if optional in cohort.columns:
            output_columns.append(optional)
    output_columns.extend(
        field for field in SURVEY_DESIGN_FIELDS if field in cohort.columns
    )
    output_columns.extend(["P47T_native", "P47T", "INGRESO", "PROP", "logP47T"])
    cohort = cohort.loc[:, output_columns].sort_values(PERSON_KEY).reset_index(drop=True)

    frame_manifest_sha = _sha256(frame_root / "manifest.json")
    conversion_manifest_sha = _sha256(conversion_root / "manifest.json")
    identity = {
        "contract": COHORT_CONTRACT,
        "analysis_frame_release_id": frame_manifest.get("release_id"),
        "analysis_frame_manifest_sha256": frame_manifest_sha,
        "conversion_release_id": conversion_manifest.get("release_id"),
        "conversion_manifest_sha256": conversion_manifest_sha,
        "cohort_policy": "historical_income_study_v1",
    }
    release_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    release_id = f"eph-income-study-cohort-{release_hash[:16]}"
    destination = output_root / release_id
    if destination.exists():
        raise IncomeStudyError(f"immutable_release_exists:{destination}")

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release_id}.", dir=output_root))
    try:
        cohort_path = staging / "cohort.csv"
        cohort.to_csv(cohort_path, index=False, lineterminator="\n")
        qa = {
            "source_persons": int(len(joined)),
            "finite_native_p47t": int(finite_native.sum()),
            "ingreso_equals_one": int(ingreso_ok.sum()),
            "positive_reference_p47t": int(positive_ok.sum()),
            "prop_not_missing": int(prop_ok.sum()),
            "eligible_persons": int(len(cohort)),
            "household_group_count": int(
                cohort.loc[:, HOUSEHOLD_KEY].drop_duplicates().shape[0]
            ),
            "identity_unique": not cohort.duplicated(PERSON_KEY).any(),
            "survey_design_fields_preserved": [
                field for field in SURVEY_DESIGN_FIELDS if field in cohort.columns
            ],
            "survey_weights_used_for_cohort": False,
            "survey_weights_used_for_target": False,
        }
        (staging / "qa.json").write_text(_canonical_json(qa), encoding="utf-8")
        manifest = {
            "contract": COHORT_CONTRACT,
            "release_id": release_id,
            "status": "source_backed_study_cohort",
            "parents": {
                "analysis_frame": {
                    "artifact_type": ANALYSIS_FRAME_CONTRACT,
                    "release_id": frame_manifest.get("release_id"),
                    "manifest_sha256": frame_manifest_sha,
                },
                "monetary_conversion": {
                    "artifact_type": CONVERSION_CONTRACT,
                    "release_id": conversion_manifest.get("release_id"),
                    "manifest_sha256": conversion_manifest_sha,
                    "status": conversion_manifest.get("status"),
                    "monetary_reference_id": conversion_manifest.get(
                        "monetary_reference_id"
                    ),
                },
            },
            "identity": {
                "person_key": PERSON_KEY,
                "household_group_key": HOUSEHOLD_KEY,
            },
            "cohort": {
                "policy_id": "historical_income_study_v1",
                "conditions": ["INGRESO == 1", "P47T > 0", "PROP not missing"],
                "ingreso_definition": "P47T > 100 after approved period-to-2016-01 conversion and legacy rounding",
                "prop_definition": "legacy evidenced II7 -> PROP recode",
                "target": {
                    "name": "logP47T",
                    "source": "P47T",
                    "transform": "log10",
                },
            },
            "monetary_lineage": {
                "source_field": "P47T",
                "source_unit": "native EPH quarter nominal ARS",
                "source_period": parent_period,
                "conversion_period": factor_row["period"],
                "reference_period": factor_row["reference_period"],
                "factor_period_to_reference": factor,
                "rounding": "numpy.rint / nearest integer",
                "reference_semantics": EXPECTED_MONETARY_REFERENCE,
                "coverage_class": factor_row["coverage_class"],
                "approved_mode_eligible": True,
            },
            "weight_policy": {
                "preserved_design_fields": [
                    field for field in SURVEY_DESIGN_FIELDS if field in cohort.columns
                ],
                "fitting": None,
                "calibration": None,
                "evaluation": None,
                "cohort_selection": None,
                "claim_boundary": "sample_conditional",
            },
            "artifacts": {
                "cohort.csv": {
                    "sha256": _sha256(cohort_path),
                    "size_bytes": cohort_path.stat().st_size,
                },
                "qa.json": {
                    "sha256": _sha256(staging / "qa.json"),
                    "size_bytes": (staging / "qa.json").stat().st_size,
                },
            },
            "qa": qa,
            "limitations": [
                "This artifact reconstructs only the historical EPH-only cohort and target surface.",
                "It does not recreate Census-shaped feature aliases, target-derived geography ranks, or the legacy synthetic pooled agglomerate.",
                "Survey-design fields are preserved for diagnostics but unused by the starter study.",
            ],
        }
        (staging / "manifest.json").write_text(
            _canonical_json(manifest), encoding="utf-8"
        )
        staging.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
