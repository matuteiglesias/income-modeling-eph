"""Scikit-learn preprocessing/model pipelines for baseline experiments.

The important design choice in this module is that the matrix design should be
driven by the feature contract whenever that metadata is available. EPH columns
often use numeric codes for categorical concepts, so pandas dtypes are only a
fallback, not the scientific source of truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _numeric_writeable_array(X):
    return np.array(X, dtype=float, copy=True)


def _categorical_writeable_array(X):
    return np.array(X, dtype=object, copy=True)


numeric_writeable_copy = FunctionTransformer(
    _numeric_writeable_array,
    validate=False,
    feature_names_out="one-to-one",
)

categorical_writeable_copy = FunctionTransformer(
    _categorical_writeable_array,
    validate=False,
    feature_names_out="one-to-one",
)


MODEL_DISPLAY_NAMES = {
    "linear_regression": "LinearRegression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "hist_gradient_boosting": "HistGradientBoostingRegressor",
    "mlp": "MLPRegressor",
}

LINEAR_MODEL_KEYS = {"linear_regression", "ridge", "lasso"}


def infer_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical columns from dtype only.

    This function is retained as a fallback for legacy runs and smoke tests. For
    thesis-facing OLS experiments, prefer passing ``feature_type_spec`` to
    ``make_preprocessor`` / ``make_model_pipeline`` so coded EPH categories such
    as P09, P10, CAT_OCUP, etc. are not accidentally treated as continuous.
    """

    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def _as_string_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if not isinstance(values, Sequence):
        return []
    return [str(value) for value in values]


def _unique_existing(columns: Sequence[str], features: pd.DataFrame) -> list[str]:
    available = set(features.columns)
    return [column for column in dict.fromkeys(str(c) for c in columns) if column in available]


def _columns_from_metadata(feature_type_spec: Mapping[str, Any], feature_type: str) -> list[str]:
    metadata = feature_type_spec.get("feature_metadata")
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)):
        return []
    columns: list[str] = []
    for row in metadata:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("feature_type")) == feature_type:
            column = row.get("column")
            if column is not None:
                columns.append(str(column))
    return columns


def resolve_feature_type_columns(
    features: pd.DataFrame,
    feature_type_spec: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Resolve continuous, binary and categorical columns for preprocessing.

    ``feature_type_spec`` may come from ``experiment_frame.build_feature_type_spec``.
    Accepted aliases are intentionally broad to keep the migration painless:

    - continuous numeric: ``continuous_numeric_columns`` or ``numeric_columns``
    - binary: ``binary_columns``
    - categorical: ``categorical_columns``
    - fixed effects: ``fixed_effect_columns``

    Any columns not covered by the explicit spec fall back to dtype inference.
    """

    if not feature_type_spec:
        numeric_columns, categorical_columns = infer_feature_types(features)
        return {
            "continuous_numeric": numeric_columns,
            "binary": [],
            "categorical": categorical_columns,
        }

    continuous = []
    for key in ["continuous_numeric_columns", "numeric_columns", "continuous_numeric"]:
        continuous.extend(_as_string_list(feature_type_spec.get(key)))
    continuous.extend(_columns_from_metadata(feature_type_spec, "continuous_numeric"))
    continuous.extend(_columns_from_metadata(feature_type_spec, "numeric"))

    binary = []
    for key in ["binary_columns", "binary"]:
        binary.extend(_as_string_list(feature_type_spec.get(key)))
    binary.extend(_columns_from_metadata(feature_type_spec, "binary"))

    categorical = []
    for key in ["categorical_columns", "categorical"]:
        categorical.extend(_as_string_list(feature_type_spec.get(key)))
    categorical.extend(_columns_from_metadata(feature_type_spec, "categorical"))

    fixed_effects = []
    for key in ["fixed_effect_columns", "fixed_effects", "fixed_effect"]:
        fixed_effects.extend(_as_string_list(feature_type_spec.get(key)))
    fixed_effects.extend(_columns_from_metadata(feature_type_spec, "fixed_effect"))

    continuous = _unique_existing(continuous, features)
    binary = _unique_existing(binary, features)
    categorical = _unique_existing([*categorical, *fixed_effects], features)

    explicitly_assigned = set(continuous) | set(binary) | set(categorical)
    remaining = [column for column in features.columns if column not in explicitly_assigned]
    fallback_numeric = features[remaining].select_dtypes(include=["number"]).columns.tolist()
    fallback_categorical = [column for column in remaining if column not in fallback_numeric]

    return {
        "continuous_numeric": [*continuous, *fallback_numeric],
        "binary": binary,
        "categorical": [*categorical, *fallback_categorical],
    }


def _numeric_pipeline(*, scale_numeric: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("copy", numeric_writeable_copy),
        ("imputer", SimpleImputer(strategy="median", copy=True)),
    ]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _binary_pipeline() -> Pipeline:
    """Create a binary-feature pipeline.

    Binary indicators are kept on the 0/1 scale for linear models, because their
    coefficients are directly interpretable as the difference between 1 and 0.
    """

    return Pipeline(
        [
            ("copy", numeric_writeable_copy),
            ("imputer", SimpleImputer(strategy="most_frequent", copy=True)),
        ]
    )


def _categorical_pipeline(*, drop_first: bool) -> Pipeline:
    """Create a categorical pipeline with optional first-level dropping."""

    return Pipeline(
        [
            ("copy", categorical_writeable_copy),
            ("imputer", SimpleImputer(strategy="most_frequent", copy=True)),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first" if drop_first else None,
                ),
            ),
        ]
    )


def make_preprocessor(
    features: pd.DataFrame,
    *,
    scale_numeric: bool,
    drop_first_categorical: bool = False,
    feature_type_spec: Mapping[str, Any] | None = None,
) -> ColumnTransformer:
    """Create preprocessing inside an sklearn pipeline.

    Explicit feature types are preferred over dtype inference. This prevents
    coded EPH categorical variables from becoming standardized numeric columns.
    """

    resolved = resolve_feature_type_columns(features, feature_type_spec)

    transformers: list[tuple[str, Any, list[str]]] = []
    if resolved["continuous_numeric"]:
        transformers.append(
            (
                "num",
                _numeric_pipeline(scale_numeric=scale_numeric),
                resolved["continuous_numeric"],
            )
        )
    if resolved["binary"]:
        transformers.append(("bin", _binary_pipeline(), resolved["binary"]))
    if resolved["categorical"]:
        transformers.append(
            (
                "cat",
                _categorical_pipeline(drop_first=drop_first_categorical),
                resolved["categorical"],
            )
        )

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    # Lightweight metadata for downstream coefficient export; sklearn ignores it.
    preprocessor.feature_type_columns_ = resolved  # type: ignore[attr-defined]
    return preprocessor


def make_model_pipeline(
    model_key: str,
    features: pd.DataFrame,
    random_state: int = 42,
    *,
    drop_first_categorical_columns: set[str] | None = None,
    feature_type_spec: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Build the configured sklearn pipeline for a model family.

    ``drop_first_categorical_columns`` is kept for backwards compatibility with
    earlier callers. The current linear-model policy drops the first level for
    all categorical variables to avoid dummy-variable collinearity with the
    intercept. The exact omitted levels should be exported from the fitted
    OneHotEncoder in coefficient diagnostics.
    """

    _ = drop_first_categorical_columns
    drop_first_categorical = model_key in LINEAR_MODEL_KEYS

    if model_key == "linear_regression":
        return Pipeline(
            [
                (
                    "preproc",
                    make_preprocessor(
                        features,
                        scale_numeric=True,
                        drop_first_categorical=drop_first_categorical,
                        feature_type_spec=feature_type_spec,
                    ),
                ),
                ("reg", LinearRegression()),
            ]
        )

    if model_key == "ridge":
        return Pipeline(
            [
                (
                    "preproc",
                    make_preprocessor(
                        features,
                        scale_numeric=True,
                        drop_first_categorical=drop_first_categorical,
                        feature_type_spec=feature_type_spec,
                    ),
                ),
                ("reg", Ridge()),
            ]
        )

    if model_key == "lasso":
        return Pipeline(
            [
                (
                    "preproc",
                    make_preprocessor(
                        features,
                        scale_numeric=True,
                        drop_first_categorical=drop_first_categorical,
                        feature_type_spec=feature_type_spec,
                    ),
                ),
                ("reg", Lasso()),
            ]
        )

    if model_key == "hist_gradient_boosting":
        return Pipeline(
            [
                (
                    "preproc",
                    make_preprocessor(
                        features,
                        scale_numeric=False,
                        drop_first_categorical=False,
                        feature_type_spec=feature_type_spec,
                    ),
                ),
                ("reg", HistGradientBoostingRegressor(random_state=random_state)),
            ]
        )

    if model_key == "mlp":
        return Pipeline(
            [
                (
                    "preproc",
                    make_preprocessor(
                        features,
                        scale_numeric=True,
                        drop_first_categorical=False,
                        feature_type_spec=feature_type_spec,
                    ),
                ),
                ("reg", MLPRegressor(random_state=random_state)),
            ]
        )

    raise ValueError(f"Unsupported model key: {model_key}")


def enabled_model_configs(experiment_config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return enabled model configs, honoring optional runtime enabled_models."""

    models = experiment_config.get("models", {})
    if not isinstance(models, Mapping):
        raise ValueError("Experiment config must define a 'models' mapping.")
    runtime = experiment_config.get("runtime", {})
    runtime_enabled = runtime.get("enabled_models") if isinstance(runtime, Mapping) else None
    allowed = set(runtime_enabled) if runtime_enabled else None
    return {
        str(key): dict(value)
        for key, value in models.items()
        if isinstance(value, Mapping)
        and value.get("enabled", False)
        and (allowed is None or str(key) in allowed)
    }
