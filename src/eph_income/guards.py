"""First-class execution guards for experiment governance.

Contracts remain the pure scientific invariants. This module records how execution-time
checks apply those contracts at named pipeline stages and writes run-local guard
evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from eph_income.contracts import (
    assert_no_forbidden_predictors,
    get_forbidden_predictors,
    validate_target_contract,
)

GuardStatus = Literal["pass", "warn", "fail", "skip"]
GuardAction = Literal["continue", "warn_continue", "fail_run"]

GUARD_SCHEMA_VERSION = "guards_v1"

_MULTICOLLINEARITY_DEFAULTS: dict[str, Any] = {
    "stage": "pre_fit",
    "sample_n": 5000,
    "warn_vif": 5.0,
    "fail_vif": 10.0,
    "max_vif_features": 200,
    "fail_rank_deficiency": True,
    "fail_duplicate_columns": True,
    "fail_constant_columns": False,
    "fail_models": ["linear_regression"],
    "warn_only_models": ["ridge", "lasso", "hist_gradient_boosting", "mlp"],
}

_GUARD_DEFINITIONS: dict[str, dict[str, Any]] = {
    "target_contract": {
        "stage": "contract",
        "default_enabled": True,
        "default_action": "fail_run",
        "notes": "Calls contracts.validate_target_contract to enforce log10(P47T).",
    },
    "forbidden_predictors": {
        "stage": "contract/pre_fit",
        "default_enabled": True,
        "default_action": "fail_run",
        "notes": "Calls contracts.get_forbidden_predictors and assert_no_forbidden_predictors.",
    },
    "runtime_mode": {
        "stage": "runtime_cost",
        "default_enabled": True,
        "default_action": "fail_run",
        "notes": "Represents existing runtime mode and allow_full_run validation.",
    },
    "fixed_effect_design": {
        "stage": "fixed_effects",
        "default_enabled": True,
        "default_action": "fail_run",
        "notes": "Documented seam for existing apply_fixed_effects design validation.",
    },
    "multicollinearity": {
        "stage": "pre_fit",
        "default_enabled": False,
        "default_action": "warn_continue",
        "notes": "Raw-feature guard for constants, duplicates, rank deficiency, and VIF.",
    },
}

_ACTION_ALIASES: dict[str, GuardAction] = {
    "continue": "continue",
    "warn": "warn_continue",
    "warning": "warn_continue",
    "warn_continue": "warn_continue",
    "fail": "fail_run",
    "fail_run": "fail_run",
}


@dataclass
class GuardDecision:
    """Decision emitted by one guard at one execution point."""

    name: str
    stage: str
    status: GuardStatus
    action: GuardAction
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return None
    return value


def _guards_block(experiment_config: Mapping[str, Any]) -> Mapping[str, Any]:
    guards = experiment_config.get("guards", {})
    if guards is None:
        return {}
    if not isinstance(guards, Mapping):
        raise TypeError("guards must be a mapping when provided.")
    return guards


def _normalize_action(value: Any, *, guard_name: str) -> GuardAction:
    if value is None:
        return _GUARD_DEFINITIONS[guard_name]["default_action"]
    normalized = str(value).strip().lower()
    if normalized not in _ACTION_ALIASES:
        allowed = ", ".join(sorted(_ACTION_ALIASES))
        raise ValueError(
            f"Guard {guard_name!r} declares unsupported action {value!r}; allowed: {allowed}."
        )
    return _ACTION_ALIASES[normalized]


def _guard_options(name: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    if name != "multicollinearity":
        return {}

    options = dict(_MULTICOLLINEARITY_DEFAULTS)
    for key in options:
        if key in raw:
            options[key] = raw[key]
    options["sample_n"] = None if options["sample_n"] is None else int(options["sample_n"])
    options["warn_vif"] = float(options["warn_vif"])
    options["fail_vif"] = float(options["fail_vif"])
    options["max_vif_features"] = int(options["max_vif_features"])
    options["fail_rank_deficiency"] = bool(options["fail_rank_deficiency"])
    options["fail_duplicate_columns"] = bool(options["fail_duplicate_columns"])
    options["fail_constant_columns"] = bool(options["fail_constant_columns"])
    options["fail_models"] = [str(model) for model in options["fail_models"]]
    options["warn_only_models"] = [str(model) for model in options["warn_only_models"]]
    if options["max_vif_features"] <= 0:
        raise ValueError("guards.multicollinearity.max_vif_features must be positive.")
    if options["warn_vif"] <= 0 or options["fail_vif"] <= 0:
        raise ValueError("guards.multicollinearity VIF thresholds must be positive.")
    return options


def guard_config(experiment_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized guard configuration with defaults applied.

    Unknown guard names fail config validation. A misspelled guard should not silently
    produce a misleading run.
    """

    guards = _guards_block(experiment_config)
    unknown = sorted(key for key in guards if key != "enabled" and key not in _GUARD_DEFINITIONS)
    if unknown:
        raise ValueError("Unknown guard names in config: " + ", ".join(unknown))

    global_enabled = bool(guards.get("enabled", True))
    normalized: dict[str, Any] = {"enabled": global_enabled, "guards": {}}
    for name, definition in _GUARD_DEFINITIONS.items():
        raw = guards.get(name, {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError(f"guards.{name} must be a mapping when provided.")
        default_enabled = bool(definition["default_enabled"])
        enabled = global_enabled and bool(raw.get("enabled", default_enabled))
        options = _guard_options(name, raw)
        normalized["guards"][name] = {
            "enabled": enabled,
            "stage": str(raw.get("stage", definition["stage"])),
            "action": _normalize_action(raw.get("action"), guard_name=name),
            "notes": str(definition["notes"]),
            **options,
        }
    return normalized


def guards_enabled(experiment_config: Mapping[str, Any]) -> bool:
    """Return whether the top-level guards switch is enabled."""

    return bool(guard_config(experiment_config)["enabled"])


def build_guard_plan(experiment_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build a serializable guard execution plan for a run."""

    normalized = guard_config(experiment_config)
    return {
        "schema_version": GUARD_SCHEMA_VERSION,
        "enabled": normalized["enabled"],
        "guards": normalized["guards"],
        "notes": [
            "Contracts stay in contracts.py; guards apply them at execution stages.",
            "Multicollinearity is a raw-feature pre-fit guard when enabled.",
            "Unknown guard names fail config validation to avoid silent misspellings.",
        ],
    }


def write_guard_plan(run_dir: Path, run_id: str, plan: Mapping[str, Any]) -> Path:
    """Write run-local guard plan evidence."""

    path = run_dir / "guards" / "guard_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, **dict(plan)}
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_guard_decisions(
    run_dir: Path, run_id: str, decisions: Sequence[GuardDecision]
) -> Path:
    """Write run-local guard decisions evidence."""

    path = run_dir / "guards" / "guard_decisions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "schema_version": GUARD_SCHEMA_VERSION,
        "decisions": [asdict(decision) for decision in decisions],
    }
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def apply_guard_decision(decision: GuardDecision) -> None:
    """Apply a guard action, raising when a failed guard must stop the run."""

    if decision.status == "fail" and decision.action == "fail_run":
        reason = "; ".join(decision.reasons) or "guard failed"
        raise ValueError(f"Guard {decision.name} failed at stage {decision.stage}: {reason}")


def _guard_entry(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    guards = plan.get("guards", {})
    if not isinstance(guards, Mapping):
        return {}
    entry = guards.get(name, {})
    return entry if isinstance(entry, Mapping) else {}


def _decision_for_disabled(plan: Mapping[str, Any], name: str) -> GuardDecision | None:
    entry = _guard_entry(plan, name)
    if entry and not bool(entry.get("enabled", False)):
        return GuardDecision(
            name=name,
            stage=str(entry.get("stage", _GUARD_DEFINITIONS[name]["stage"])),
            status="skip",
            action="continue",
            reasons=["Guard disabled by configuration."],
        )
    return None


def target_contract_guard(
    feature_contract: Mapping[str, Any], plan: Mapping[str, Any]
) -> GuardDecision:
    """Apply the governed target-contract guard."""

    disabled = _decision_for_disabled(plan, "target_contract")
    if disabled is not None:
        return disabled
    entry = _guard_entry(plan, "target_contract")
    action = str(entry.get("action", "fail_run"))
    try:
        target = validate_target_contract(feature_contract)
    except (TypeError, ValueError) as exc:
        return GuardDecision(
            name="target_contract",
            stage="contract",
            status="fail",
            action=action,  # type: ignore[arg-type]
            reasons=[str(exc)],
        )
    return GuardDecision(
        name="target_contract",
        stage="contract",
        status="pass",
        action="continue",
        reasons=["Target contract is valid."],
        details={"target": target},
    )


def forbidden_predictors_guard(
    feature_contract: Mapping[str, Any], feature_columns: Sequence[str], plan: Mapping[str, Any]
) -> GuardDecision:
    """Apply the governed forbidden-predictors guard."""

    disabled = _decision_for_disabled(plan, "forbidden_predictors")
    if disabled is not None:
        return disabled
    entry = _guard_entry(plan, "forbidden_predictors")
    action = str(entry.get("action", "fail_run"))
    try:
        forbidden = get_forbidden_predictors(feature_contract)
        assert_no_forbidden_predictors(feature_columns, forbidden)
    except (TypeError, ValueError) as exc:
        return GuardDecision(
            name="forbidden_predictors",
            stage="contract/pre_fit",
            status="fail",
            action=action,  # type: ignore[arg-type]
            reasons=[str(exc)],
            details={"feature_count": len(feature_columns)},
        )
    return GuardDecision(
        name="forbidden_predictors",
        stage="contract/pre_fit",
        status="pass",
        action="continue",
        reasons=["No forbidden predictors are present in feature columns."],
        details={"feature_count": len(feature_columns)},
    )


def runtime_mode_guard(mode: str, plan: Mapping[str, Any]) -> GuardDecision:
    """Record the existing runtime-mode validation as a guard decision."""

    disabled = _decision_for_disabled(plan, "runtime_mode")
    if disabled is not None:
        return disabled
    return GuardDecision(
        name="runtime_mode",
        stage="runtime_cost",
        status="pass",
        action="continue",
        reasons=["Runtime mode passed existing validation."],
        details={"mode": mode},
    )


def fixed_effect_design_guard(
    fixed_effects_used: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]
) -> GuardDecision:
    """Record existing fixed-effect design validation as a guard decision."""

    disabled = _decision_for_disabled(plan, "fixed_effect_design")
    if disabled is not None:
        return disabled
    return GuardDecision(
        name="fixed_effect_design",
        stage="fixed_effects",
        status="pass",
        action="continue",
        reasons=["Fixed-effect design passed existing validation."],
        details={"fixed_effect_count": len(fixed_effects_used)},
    )


def multicollinearity_guard_enabled(plan: Mapping[str, Any]) -> bool:
    """Return whether the multicollinearity guard is enabled in a guard plan."""

    return bool(_guard_entry(plan, "multicollinearity").get("enabled", False))


def sample_guard_frame(
    frame: pd.DataFrame, sample_n: int | None, random_seed: int
) -> pd.DataFrame:
    """Return a deterministic sample for guard audits."""

    if sample_n is None or sample_n >= len(frame):
        return frame.copy()
    return frame.sample(n=sample_n, random_state=random_seed).sort_index().copy()


def detect_constant_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns with a single observed value, treating all-NA as constant."""

    return [str(column) for column in frame.columns if frame[column].nunique(dropna=False) <= 1]


def detect_duplicate_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Return exact duplicate raw columns as duplicate -> first original."""

    duplicate_of: dict[str, str] = {}
    columns = [str(column) for column in frame.columns]
    for idx, column in enumerate(columns):
        if column in duplicate_of:
            continue
        left = frame[column]
        for prior in columns[:idx]:
            if left.equals(frame[prior]):
                duplicate_of[column] = prior
                break
    return duplicate_of


def _numeric_audit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [column for column in frame.columns if is_numeric_dtype(frame[column])]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    for column in numeric.columns:
        if numeric[column].isna().any():
            mean_value = numeric[column].mean()
            numeric[column] = numeric[column].fillna(0.0 if pd.isna(mean_value) else mean_value)
    return numeric.astype(float)


def _matrix_rank(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    return int(np.linalg.matrix_rank(values))


def numeric_design_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize numeric raw-feature rank after constants are removed."""

    numeric = _numeric_audit_frame(frame)
    constant_columns = set(detect_constant_columns(numeric))
    audited = numeric[[column for column in numeric.columns if column not in constant_columns]]
    values = audited.to_numpy(dtype=float)
    if values.size:
        values = values - values.mean(axis=0, keepdims=True)
    matrix_rank = _matrix_rank(values)
    numeric_matrix_columns = int(audited.shape[1])
    return {
        "numeric_feature_count": int(numeric.shape[1]),
        "numeric_matrix_columns": numeric_matrix_columns,
        "matrix_rank": matrix_rank,
        "rank_deficient": bool(numeric_matrix_columns > 0 and matrix_rank < numeric_matrix_columns),
        "audited_numeric_columns": [str(column) for column in audited.columns],
    }


def _r2_for_feature(values: np.ndarray, index: int) -> float:
    y = values[:, index]
    x_other = np.delete(values, index, axis=1)
    if x_other.shape[1] == 0:
        return 0.0
    x_other = np.column_stack([np.ones(len(x_other)), x_other])
    coefficients, *_ = np.linalg.lstsq(x_other, y, rcond=None)
    y_hat = x_other @ coefficients
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= np.finfo(float).eps:
        return 1.0
    residual = float(np.sum((y - y_hat) ** 2))
    return max(0.0, min(1.0, 1.0 - residual / total))


def compute_vif_table(
    numeric_frame: pd.DataFrame,
    *,
    warn_vif: float,
    fail_vif: float,
    max_vif_features: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute raw numeric-feature VIF values and a compact summary."""

    constant_columns = set(detect_constant_columns(numeric_frame))
    audited = numeric_frame[[column for column in numeric_frame.columns if column not in constant_columns]]
    if audited.shape[1] > max_vif_features:
        return pd.DataFrame(columns=["feature", "vif", "vif_status"]), {
            "vif_computed": False,
            "vif_skipped_reason": (
                f"numeric feature count {audited.shape[1]} exceeds max_vif_features={max_vif_features}"
            ),
            "max_vif": None,
            "n_vif_above_warn": 0,
            "n_vif_above_fail": 0,
        }

    values = audited.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(audited.columns):
        r2 = _r2_for_feature(values, index)
        vif = float("inf") if r2 >= 0.999999 else float(1.0 / max(1.0 - r2, np.finfo(float).eps))
        if vif > fail_vif:
            status = "fail"
        elif vif > warn_vif:
            status = "warn"
        else:
            status = "pass"
        rows.append({"feature": str(feature), "vif": vif, "vif_status": status})

    vif_table = pd.DataFrame(rows)
    max_vif = None if vif_table.empty else float(vif_table["vif"].max())
    return vif_table, {
        "vif_computed": True,
        "vif_skipped_reason": None,
        "max_vif": max_vif,
        "n_vif_above_warn": int((vif_table["vif"] > warn_vif).sum()) if not vif_table.empty else 0,
        "n_vif_above_fail": int((vif_table["vif"] > fail_vif).sum()) if not vif_table.empty else 0,
    }


def build_multicollinearity_audit(
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    run_id: str,
    guard_config: Mapping[str, Any],
    run_dir: Path,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    """Build raw-feature multicollinearity audit artifacts once per run."""

    sample_n = guard_config.get("sample_n")
    sample_n_int = None if sample_n is None else int(sample_n)
    feature_frame = train_frame[list(feature_columns)]
    sampled = sample_guard_frame(feature_frame, sample_n_int, random_seed)
    print(
        "guard-start "
        f"run_id={run_id} guard=multicollinearity "
        f"feature_count={len(feature_columns)} sample_n={sample_n_int}"
    )

    constant_columns = detect_constant_columns(sampled)
    duplicate_columns = detect_duplicate_columns(sampled)
    numeric = _numeric_audit_frame(sampled)
    numeric_summary = numeric_design_summary(sampled)
    vif_table, vif_summary = compute_vif_table(
        numeric[[column for column in numeric.columns if column not in set(constant_columns)]],
        warn_vif=float(guard_config["warn_vif"]),
        fail_vif=float(guard_config["fail_vif"]),
        max_vif_features=int(guard_config["max_vif_features"]),
    )
    vif_by_feature = {
        str(row["feature"]): row for row in vif_table.to_dict(orient="records")
    }
    numeric_columns = set(str(column) for column in numeric.columns)
    constant_set = set(constant_columns)

    rows = []
    for feature in feature_columns:
        feature_name = str(feature)
        vif_row = vif_by_feature.get(feature_name)
        audited_as_numeric = feature_name in numeric_columns
        notes: list[str] = []
        if not audited_as_numeric:
            notes.append("non_numeric_raw_feature")
        if feature_name in constant_set:
            notes.append("constant")
        if feature_name in duplicate_columns:
            notes.append("duplicate")
        if audited_as_numeric and vif_row is None and vif_summary["vif_computed"] is False:
            notes.append("vif_skipped")
        rows.append(
            {
                "run_id": run_id,
                "feature": feature_name,
                "dtype": str(sampled[feature].dtype),
                "audited_as_numeric": audited_as_numeric,
                "is_constant": feature_name in constant_set,
                "duplicate_of": duplicate_columns.get(feature_name),
                "vif": None if vif_row is None else vif_row["vif"],
                "vif_status": "not_numeric_or_skipped" if vif_row is None else vif_row["vif_status"],
                "notes": ";".join(notes),
            }
        )

    audit_table = pd.DataFrame(rows)
    summary = {
        "run_id": run_id,
        "guard": "multicollinearity",
        "stage": "pre_fit",
        "sample_n_requested": sample_n_int,
        "sample_n_used": int(len(sampled)),
        "feature_count": int(len(feature_columns)),
        "numeric_feature_count": int(numeric_summary["numeric_feature_count"]),
        "categorical_feature_count": int(len(feature_columns) - numeric_summary["numeric_feature_count"]),
        **vif_summary,
        "constant_columns": constant_columns,
        "duplicate_columns": duplicate_columns,
        "rank_deficient": bool(numeric_summary["rank_deficient"]),
        "matrix_rank": int(numeric_summary["matrix_rank"]),
        "numeric_matrix_columns": int(numeric_summary["numeric_matrix_columns"]),
        "warn_vif": float(guard_config["warn_vif"]),
        "fail_vif": float(guard_config["fail_vif"]),
        "fail_rank_deficiency": bool(guard_config["fail_rank_deficiency"]),
        "fail_duplicate_columns": bool(guard_config["fail_duplicate_columns"]),
        "fail_constant_columns": bool(guard_config["fail_constant_columns"]),
        "todo": "Audit post-preprocessing transformed design matrices in a future guard.",
    }

    guards_dir = run_dir / "guards"
    guards_dir.mkdir(parents=True, exist_ok=True)
    features_path = guards_dir / "multicollinearity_raw_features.csv"
    summary_path = guards_dir / "multicollinearity_raw_summary.json"
    audit_table.to_csv(features_path, index=False)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    return audit_table, summary, {
        "multicollinearity_raw_features": features_path,
        "multicollinearity_raw_summary": summary_path,
    }


def _multicollinearity_reasons(summary: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if bool(config["fail_duplicate_columns"]) and summary.get("duplicate_columns"):
        reasons.append("duplicate_columns")
    if bool(config["fail_constant_columns"]) and summary.get("constant_columns"):
        reasons.append("constant_columns")
    if bool(config["fail_rank_deficiency"]) and bool(summary.get("rank_deficient", False)):
        reasons.append("rank_deficiency")
    max_vif = summary.get("max_vif")
    if max_vif is not None and float(max_vif) > float(config["fail_vif"]):
        reasons.append("max_vif_above_fail")
    elif max_vif is not None and float(max_vif) > float(config["warn_vif"]):
        reasons.append("max_vif_above_warn")
    return reasons


def decide_multicollinearity_action(
    *,
    audit_summary: Mapping[str, Any],
    model_key: str,
    guard_config: Mapping[str, Any],
) -> GuardDecision:
    """Return the model-specific multicollinearity guard decision."""

    reasons = _multicollinearity_reasons(audit_summary, guard_config)
    details = {
        "model_key": model_key,
        "max_vif": audit_summary.get("max_vif"),
        "rank_deficient": audit_summary.get("rank_deficient"),
        "n_duplicate_columns": len(audit_summary.get("duplicate_columns", {})),
        "n_constant_columns": len(audit_summary.get("constant_columns", [])),
        "vif_computed": audit_summary.get("vif_computed"),
        "vif_skipped_reason": audit_summary.get("vif_skipped_reason"),
    }
    if not reasons:
        return GuardDecision(
            name="multicollinearity",
            stage="pre_fit",
            status="pass",
            action="continue",
            reasons=["No blocking raw-feature multicollinearity issues detected."],
            details=details,
        )

    fail_models = set(str(model) for model in guard_config.get("fail_models", []))
    status: GuardStatus = "fail" if model_key in fail_models else "warn"
    action: GuardAction = "fail_run" if status == "fail" else "warn_continue"
    message = "guard-fail" if status == "fail" else "guard-warning"
    print(
        f"{message} run_id={audit_summary.get('run_id')} guard=multicollinearity "
        f"model={model_key} max_vif={audit_summary.get('max_vif')} reasons={','.join(reasons)}"
    )
    return GuardDecision(
        name="multicollinearity",
        stage="pre_fit",
        status=status,
        action=action,
        reasons=reasons,
        details=details,
    )


def run_multicollinearity_guard(
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    run_id: str,
    model_key: str,
    guard_config: Mapping[str, Any],
    run_dir: Path,
    random_seed: int,
) -> GuardDecision:
    """Build a raw-feature audit and return one model-specific guard decision.

    Experiment runs should prefer ``build_multicollinearity_audit`` plus
    ``decide_multicollinearity_action`` to avoid recomputing VIF for each model.
    """

    _, summary, _ = build_multicollinearity_audit(
        train_frame=train_frame,
        feature_columns=feature_columns,
        run_id=run_id,
        guard_config=guard_config,
        run_dir=run_dir,
        random_seed=random_seed,
    )
    return decide_multicollinearity_action(
        audit_summary=summary, model_key=model_key, guard_config=guard_config
    )
