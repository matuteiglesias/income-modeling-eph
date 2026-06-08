"""Diagnostics configuration normalization and planning.

This module is intentionally declarative. It normalizes experiment YAML into a
stable diagnostics contract and resolves which diagnostics are expected for a
run. It does not fit estimators, call prediction APIs, compute model metrics, or
write plots.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

DIAGNOSTICS_SCHEMA_VERSION = "diagnostics_v1"
DIAGNOSTICS_PROFILES = {
    "none",
    "minimal",
    "standard",
    "sweep",
    "smoke_minimal",
    "thesis_core",
    "hgb_capacity_sweep",
    "regularization_interpretation",
    "geo_leakage_probe",
    "linear_fe_interpretation",
}
COEFFICIENT_MODEL_KEYS = {"linear_regression", "ridge", "lasso"}
REGULARIZATION_MODEL_KEYS = {"ridge", "lasso"}
HGB_MODEL_KEY = "hist_gradient_boosting"

STANDARD_DEFAULTS: dict[str, bool] = {
    "residual_summary": True,
    "error_by_income_decile": True,
    "prediction_distribution_summary": True,
    "model_pairwise_error_comparison": True,
    "metric_gaps": True,
    "observed_vs_predicted": True,
    "prediction_distribution": True,
    "residual_distribution": True,
    "mae_by_income_decile": True,
    "mean_residual_by_income_decile": True,
}

MINIMAL_STANDARD_DEFAULTS: dict[str, bool] = {
    key: key in {"residual_summary", "error_by_income_decile", "prediction_distribution_summary"}
    for key in STANDARD_DEFAULTS
}

PROFILE_STANDARD_DEFAULTS: dict[str, dict[str, bool]] = {
    "smoke_minimal": {
        **{key: False for key in STANDARD_DEFAULTS},
        "residual_summary": True,
        "prediction_distribution_summary": True,
    },
    "thesis_core": dict(STANDARD_DEFAULTS),
    "hgb_capacity_sweep": {
        **{key: False for key in STANDARD_DEFAULTS},
        "residual_summary": True,
        "prediction_distribution_summary": True,
    },
    "regularization_interpretation": {
        **{key: False for key in STANDARD_DEFAULTS},
        "residual_summary": True,
        "prediction_distribution_summary": True,
    },
    "geo_leakage_probe": {
        **{key: False for key in STANDARD_DEFAULTS},
        "residual_summary": True,
        "error_by_income_decile": True,
        "prediction_distribution_summary": True,
    },
    "linear_fe_interpretation": {
        **{key: False for key in STANDARD_DEFAULTS},
        "residual_summary": True,
        "prediction_distribution_summary": True,
    },
}

DEFAULT_PLOTS: dict[str, bool] = {
    "enabled": True,
    "observed_vs_predicted": True,
    "prediction_distribution": True,
    "residual_distribution": True,
    "mae_by_income_decile": True,
    "mean_residual_by_income_decile": True,
    "distribution_compression_by_model": False,
    "hgb_cv_curve": False,
    "hgb_train_vs_cv": False,
    "hgb_overfit_gap": False,
    "hgb_fit_time": False,
    "ridge_cv_r2_vs_alpha": False,
    "lasso_cv_r2_vs_alpha": False,
    "ridge_coef_l2_norm_vs_alpha": False,
    "lasso_coef_l1_norm_vs_alpha": False,
    "lasso_nonzero_coefficients_vs_alpha": False,
    "fixed_effect_coefficients": False,
}

DEFAULT_DISTRIBUTION: dict[str, bool] = {
    "compression_summary": False,
    "decile_tail_summary": False,
}

DEFAULT_COMPARISON: dict[str, Any] = {
    "family": None,
    "variant": None,
    "anchor_variant": None,
}

DEFAULT_FIXED_EFFECTS: dict[str, bool] = {
    "enabled": "auto",
    "coefficient_table": True,
    "reference_levels": True,
}

DEFAULT_NORMALIZED_CONFIG: dict[str, Any] = {
    "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
    "enabled": True,
    "profile": "standard",
    "standard": STANDARD_DEFAULTS,
    "plots": DEFAULT_PLOTS,
    "distribution": DEFAULT_DISTRIBUTION,
    "coefficients": {
        "enabled": "auto",
        "best_coefficients": True,
        "fixed_effect_coefficients": True,
    },
    "regularization": {
        "enabled": "auto",
        "alpha_curves": True,
        "coefficient_paths": True,
        "coefficient_norms": False,
        "sparsity_summary": False,
    },
    "hgb": {
        "enabled": "auto",
        "basic_cv_summaries": True,
        "top_configs": True,
        "overfit_gap_by_config": True,
    },
    "sweeps": {
        "enabled": "auto",
        "sweep_type": None,
        "primary_param": None,
        "group_param": None,
    },
    "comparison": DEFAULT_COMPARISON,
    "fixed_effects": DEFAULT_FIXED_EFFECTS,
    "notes": [],
    "compatibility": {
        "used_legacy_regularization_sweep_key": False,
        "used_legacy_sweep_type_key": False,
    },
}


def _copy_default() -> dict[str, Any]:
    return deepcopy(DEFAULT_NORMALIZED_CONFIG)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _merge_section(
    normalized: dict[str, Any], raw: Mapping[str, Any], section: str
) -> None:
    raw_section = raw.get(section)
    if not isinstance(raw_section, Mapping):
        return
    merged = dict(normalized[section])
    merged.update(dict(raw_section))
    normalized[section] = merged


def _normalize_enabled(value: Any) -> bool | str:
    if isinstance(value, str) and value == "auto":
        return "auto"
    return bool(value)


def normalize_diagnostics_config(experiment_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return diagnostics config in the canonical ``diagnostics_v1`` shape."""

    normalized = _copy_default()
    raw = experiment_config.get("diagnostics", {}) if isinstance(experiment_config, Mapping) else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("diagnostics must be a mapping when provided.")

    if "enabled" in raw:
        normalized["enabled"] = bool(raw["enabled"])
    if "profile" in raw:
        profile = str(raw["profile"])
        if profile not in DIAGNOSTICS_PROFILES:
            raise ValueError(
                f"Unsupported diagnostics profile {profile!r}. "
                f"Expected one of: {sorted(DIAGNOSTICS_PROFILES)}"
            )
        normalized["profile"] = profile
    elif raw.get("regularization_sweep") is True or "sweep_type" in raw:
        normalized["profile"] = "sweep"

    if normalized["profile"] == "none":
        normalized["enabled"] = False
    if normalized["profile"] == "minimal":
        normalized["standard"] = dict(MINIMAL_STANDARD_DEFAULTS)
    if normalized["profile"] in PROFILE_STANDARD_DEFAULTS:
        normalized["standard"] = dict(PROFILE_STANDARD_DEFAULTS[normalized["profile"]])

    for section in [
        "standard",
        "plots",
        "distribution",
        "coefficients",
        "regularization",
        "hgb",
        "sweeps",
        "comparison",
        "fixed_effects",
    ]:
        _merge_section(normalized, raw, section)
    if isinstance(raw.get("notes"), list):
        normalized["notes"] = list(raw["notes"])

    if raw.get("regularization_sweep") is True:
        normalized["regularization"]["enabled"] = True
        normalized["compatibility"]["used_legacy_regularization_sweep_key"] = True
    if "hist_gradient_boosting" in raw:
        normalized["hgb"]["enabled"] = bool(raw["hist_gradient_boosting"])
    if "sweep_type" in raw:
        normalized["sweeps"]["enabled"] = True
        normalized["sweeps"]["sweep_type"] = raw.get("sweep_type")
        normalized["sweeps"]["primary_param"] = raw.get("primary_param")
        normalized["sweeps"]["group_param"] = raw.get("group_param")
        normalized["compatibility"]["used_legacy_sweep_type_key"] = True

    for section in ["coefficients", "regularization", "hgb", "sweeps", "fixed_effects"]:
        normalized[section]["enabled"] = _normalize_enabled(normalized[section].get("enabled"))

    return normalized


def diagnostics_enabled(experiment_config: Mapping[str, Any]) -> bool:
    """Return whether optional diagnostics planning is enabled."""

    return bool(normalize_diagnostics_config(experiment_config)["enabled"])


def _model_grid(experiment_config: Mapping[str, Any], model_key: str) -> Mapping[str, Any]:
    models = _mapping(experiment_config.get("models", {}))
    model_config = _mapping(models.get(model_key, {}))
    return _mapping(model_config.get("grid", {}))


def _has_alpha_grid(experiment_config: Mapping[str, Any], enabled_models: set[str]) -> bool:
    for model_key in REGULARIZATION_MODEL_KEYS & enabled_models:
        grid = _model_grid(experiment_config, model_key)
        if "reg__alpha" in grid:
            return True
    return False


def _resolve_auto(value: bool | str, condition: bool) -> bool:
    if value == "auto":
        return condition
    return bool(value)


def _path_map(cv_result_paths: Mapping[str, str | Path] | None) -> dict[str, str]:
    if not cv_result_paths:
        return {}
    return {str(key): str(value) for key, value in cv_result_paths.items()}


def build_diagnostics_plan(
    *,
    experiment_config: Mapping[str, Any],
    enabled_models: Sequence[str],
    fixed_effects_used: Sequence[Mapping[str, Any]] | None = None,
    cv_result_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build a resolved diagnostics plan for an experiment run."""

    normalized = normalize_diagnostics_config(experiment_config)
    enabled_model_set = {str(model) for model in enabled_models}
    fixed_effect_specs = [dict(spec) for spec in (fixed_effects_used or [])]
    notes: list[str] = []

    coefficient_model_enabled = bool(COEFFICIENT_MODEL_KEYS & enabled_model_set)
    coefficients_enabled = _resolve_auto(
        normalized["coefficients"]["enabled"], coefficient_model_enabled
    )
    if not coefficients_enabled and normalized["coefficients"]["enabled"] == "auto":
        notes.append(
            "Coefficient diagnostics disabled because no coefficient-bearing model is enabled."
        )

    fixed_effect_coefficients_enabled = bool(
        coefficients_enabled
        and normalized["coefficients"].get("fixed_effect_coefficients", True)
        and fixed_effect_specs
    )
    if (
        coefficients_enabled
        and normalized["coefficients"].get("fixed_effect_coefficients", True)
        and not fixed_effect_specs
    ):
        notes.append("Fixed-effect coefficient diagnostics disabled because no fixed effects are used.")

    hgb_enabled = _resolve_auto(normalized["hgb"]["enabled"], HGB_MODEL_KEY in enabled_model_set)
    if not hgb_enabled and normalized["hgb"]["enabled"] == "auto":
        notes.append("HGB diagnostics disabled because HistGradientBoosting is not enabled.")

    alpha_grid_present = _has_alpha_grid(experiment_config, enabled_model_set)
    regularization_enabled = _resolve_auto(
        normalized["regularization"]["enabled"], alpha_grid_present
    )
    if not regularization_enabled and normalized["regularization"]["enabled"] == "auto":
        notes.append(
            "Regularization diagnostics disabled because no Ridge/Lasso alpha grid was detected."
        )

    sweep_type = normalized["sweeps"].get("sweep_type")
    primary_param = normalized["sweeps"].get("primary_param")
    sweep_configured = bool(sweep_type and primary_param)
    sweeps_enabled = _resolve_auto(normalized["sweeps"]["enabled"], sweep_configured)
    if not sweeps_enabled and normalized["sweeps"]["enabled"] == "auto":
        notes.append("HGB sweep diagnostics disabled because no sweep_type is configured.")

    fixed_effects_enabled = _resolve_auto(
        normalized["fixed_effects"]["enabled"], bool(fixed_effect_specs)
    )
    standard_enabled = bool(normalized["enabled"] and normalized["profile"] != "none")
    plots_enabled = bool(normalized["plots"].get("enabled", True) and standard_enabled)
    standard_plan = {
        name: bool(enabled and standard_enabled)
        for name, enabled in normalized["standard"].items()
    }
    plots_plan = {
        name: bool(enabled and plots_enabled) if name != "enabled" else plots_enabled
        for name, enabled in normalized["plots"].items()
    }
    distribution_plan = {
        name: bool(enabled and standard_enabled)
        for name, enabled in normalized["distribution"].items()
    }

    resolved = {
        "enabled": bool(normalized["enabled"]),
        "profile": normalized["profile"],
        "standard": standard_plan,
        "plots": plots_plan,
        "distribution": distribution_plan,
        "coefficients": {
            "enabled": bool(coefficients_enabled and normalized["enabled"]),
            "best_coefficients": bool(
                coefficients_enabled
                and normalized["enabled"]
                and normalized["coefficients"].get("best_coefficients", True)
            ),
            "fixed_effect_coefficients": bool(
                fixed_effect_coefficients_enabled and normalized["enabled"]
            ),
            "standardized_coefficients": bool(
                coefficients_enabled
                and normalized["enabled"]
                and normalized["coefficients"].get("standardized_coefficients", False)
            ),
            "coefficient_stability": bool(
                coefficients_enabled
                and normalized["enabled"]
                and normalized["coefficients"].get("coefficient_stability", False)
            ),
        },
        "regularization": {
            "enabled": bool(regularization_enabled and normalized["enabled"]),
            "alpha_curves": bool(
                regularization_enabled
                and normalized["enabled"]
                and normalized["regularization"].get("alpha_curves", True)
            ),
            "coefficient_paths": bool(
                regularization_enabled
                and normalized["enabled"]
                and normalized["regularization"].get("coefficient_paths", True)
            ),
            "coefficient_norms": bool(
                regularization_enabled
                and normalized["enabled"]
                and normalized["regularization"].get("coefficient_norms", False)
            ),
            "sparsity_summary": bool(
                regularization_enabled
                and normalized["enabled"]
                and normalized["regularization"].get("sparsity_summary", False)
            ),
        },
        "hgb": {
            "enabled": bool(hgb_enabled and normalized["enabled"]),
            "basic_cv_summaries": bool(
                hgb_enabled
                and normalized["enabled"]
                and normalized["hgb"].get("basic_cv_summaries", True)
            ),
            "top_configs": bool(
                hgb_enabled and normalized["enabled"] and normalized["hgb"].get("top_configs", True)
            ),
            "overfit_gap_by_config": bool(
                hgb_enabled
                and normalized["enabled"]
                and normalized["hgb"].get("overfit_gap_by_config", True)
            ),
        },
        "sweeps": {
            "enabled": bool(sweeps_enabled and normalized["enabled"]),
            "sweep_type": sweep_type,
            "primary_param": primary_param,
            "group_param": normalized["sweeps"].get("group_param"),
        },
        "fixed_effects": {
            "enabled": bool(fixed_effects_enabled and normalized["enabled"]),
            "coefficient_table": bool(
                fixed_effects_enabled
                and normalized["enabled"]
                and normalized["fixed_effects"].get("coefficient_table", True)
            ),
            "reference_levels": bool(
                fixed_effects_enabled
                and normalized["enabled"]
                and normalized["fixed_effects"].get("reference_levels", True)
            ),
        },
        "comparison": dict(normalized["comparison"]),
    }

    return {
        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "enabled": bool(normalized["enabled"]),
        "profile": normalized["profile"],
        "normalized_config": normalized,
        "resolved": resolved,
        "enabled_models": sorted(enabled_model_set),
        "fixed_effects_used": fixed_effect_specs,
        "cv_result_paths": _path_map(cv_result_paths),
        "notes": [*normalized.get("notes", []), *notes],
    }
