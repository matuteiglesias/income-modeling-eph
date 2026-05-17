"""Scikit-learn preprocessing/model pipelines for baseline experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


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


def infer_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical columns from a feature matrix."""

    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def _categorical_pipeline(*, drop_first: bool) -> Pipeline:
    """Create a categorical pipeline with optional first-level dropping."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
    drop_first_categorical_columns: set[str] | None = None,
) -> ColumnTransformer:
    """Create preprocessing inside an sklearn pipeline."""

    numeric_columns, categorical_columns = infer_feature_types(features)
    numeric_steps: list[tuple[str, Any]] = [
        ("copy", numeric_writeable_copy),
        ("imputer", SimpleImputer(strategy="median", copy=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline(
        [
            ("copy", categorical_writeable_copy),
            ("imputer", SimpleImputer(strategy="most_frequent", copy=True)),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_columns),
            ("cat", categorical_pipe, categorical_columns),
        ],
        remainder="drop",
    )


def make_model_pipeline(
    model_key: str,
    features: pd.DataFrame,
    random_state: int = 42,
    *,
    drop_first_categorical_columns: set[str] | None = None,
) -> Pipeline:
    """Build the configured sklearn pipeline for a model family."""

    if model_key == "linear_regression":
        return Pipeline(
            [
                (
                    "preproc",
                    make_preprocessor(
                        features,
                        scale_numeric=True,
                        drop_first_categorical_columns=drop_first_categorical_columns,
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
                        drop_first_categorical_columns=drop_first_categorical_columns,
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
                        drop_first_categorical_columns=drop_first_categorical_columns,
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
                        drop_first_categorical_columns=drop_first_categorical_columns,
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
                        drop_first_categorical_columns=drop_first_categorical_columns,
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
