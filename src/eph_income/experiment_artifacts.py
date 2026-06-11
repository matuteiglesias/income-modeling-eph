"""Run-local artifacts for EPH income experiments."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from eph_income.contracts import assert_no_forbidden_predictors, get_forbidden_predictors
from eph_income.dataset import ROW_ID_COLUMN, get_metadata_path, resolve_project_path
from eph_income.splits import get_split_path

PREDICTION_COLUMNS = [
    "run_id",
    "model",
    "split",
    ROW_ID_COLUMN,
    "y_true",
    "y_pred",
    "residual",
    "abs_error",
    "squared_error",
]


def _resolve_output_path(path: str | Path) -> Path:
    return resolve_project_path(path)


def _artifacts_mapping(experiment_config: Mapping[str, Any]) -> Mapping[str, Any]:
    artifacts = experiment_config.get("artifacts", {})
    return artifacts if isinstance(artifacts, Mapping) else {}


def _write_training_frame_sample(
    *,
    run_dir: Path,
    run_id: str,
    train: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    experiment_config: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
) -> dict[str, Path]:
    """Write a deterministic training-frame sample for scientific inspection."""

    artifacts_config = _artifacts_mapping(experiment_config)
    configured_sample_n = artifacts_config.get("training_frame_sample_n", 10)
    if configured_sample_n is None:
        return {}

    sample_n_requested = int(configured_sample_n)
    if sample_n_requested <= 0:
        return {}

    forbidden = get_forbidden_predictors(feature_contract)
    assert_no_forbidden_predictors(feature_columns, forbidden)

    columns_written = [ROW_ID_COLUMN, "split", target, *feature_columns]
    forbidden_sample_columns = sorted((set(columns_written) & forbidden) - {target})
    if forbidden_sample_columns:
        joined = ", ".join(forbidden_sample_columns)
        raise ValueError(f"Forbidden predictors present in training-frame sample columns: {joined}")

    missing_columns = [column for column in columns_written if column not in train.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"Training-frame sample columns are missing from train split: {joined}")

    random_seed = int(experiment_config.get("experiment", {}).get("random_seed", 42))
    sample_n_written = min(sample_n_requested, len(train))
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sample_path = artifacts_dir / "training_frame_sample.csv"
    metadata_path = artifacts_dir / "training_frame_sample_metadata.json"

    sample = train[columns_written].sample(n=sample_n_written, random_state=random_seed)
    sample = sample.sort_values(ROW_ID_COLUMN).reset_index(drop=True)
    sample.to_csv(sample_path, index=False)

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_n_requested": sample_n_requested,
        "sample_n_written": int(sample_n_written),
        "source_split": "train",
        "target": target,
        "feature_count": int(len(feature_columns)),
        "columns_written": columns_written,
        "random_state": random_seed,
        "note": "Sample drawn from the train split after runtime sampling and before model fitting.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "training_frame_sample": sample_path,
        "training_frame_sample_metadata": metadata_path,
    }


def _evaluate(model: Any, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "r2": float(r2_score(target, predictions)),
        "mae": float(mean_absolute_error(target, predictions)),
        "mse": float(mean_squared_error(target, predictions)),
    }


def _safe_run_component(value: str) -> str:
    """Return a filesystem-safe run ID component."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def _make_run_id(experiment_config: Mapping[str, Any]) -> str:
    experiment_id = str(experiment_config.get("experiment", {}).get("id", "experiment"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_safe_run_component(experiment_id)}_{timestamp}"


def _prepare_run_directory(experiment_config: Mapping[str, Any], run_id: str) -> Path:
    outputs = experiment_config.get("outputs", {})
    base_dir = outputs.get("runs_dir", "reports/runs")
    run_dir = _resolve_output_path(base_dir) / run_id
    if run_dir.exists():
        suffix = datetime.now(timezone.utc).strftime("%f")
        run_dir = run_dir.with_name(f"{run_dir.name}_{suffix}")
    for child in [
        "metrics",
        "predictions",
        "cv_results",
        "diagnostics",
        "artifacts",
        "plots",
        "guards",
    ]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_yaml_snapshot(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset_metadata(experiment_config: Mapping[str, Any]) -> dict[str, Any]:
    processed_path = _resolve_output_path(experiment_config["data"]["processed_dataset"])
    metadata_path = get_metadata_path(processed_path)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _make_dataset_card(
    experiment_config: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    joined: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    fixed_effects_used: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    processed_path = _resolve_output_path(experiment_config["data"]["processed_dataset"])
    split_path = get_split_path(experiment_config)
    metadata = _load_dataset_metadata(experiment_config)
    split_counts = (
        joined["split"].value_counts().reindex(["train", "test", "validation"], fill_value=0)
    )
    data_config = experiment_config.get("data", {})
    source = (
        data_config.get("input_files")
        or metadata.get("input_files")
        or metadata.get("upstream_input_files")
    )
    return {
        "dataset_name": "modeling_dataset",
        "source": source,
        "processed_dataset": str(processed_path),
        "split_assignments": str(split_path),
        "target": target,
        "target_definition": feature_contract.get("target", {}),
        "row_count": int(len(joined)),
        "feature_count": int(len(feature_columns)),
        "fixed_effects_used": fixed_effects_used or [],
        "digest": _sha256_file(processed_path),
        "split_digest": _sha256_file(split_path),
        "splits": {split: int(count) for split, count in split_counts.items()},
        "metadata": metadata,
    }


def _prediction_frame(
    *,
    run_id: str,
    model_name: str,
    split_name: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    estimator: Any,
    context_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return row-level predictions, optionally preserving diagnostic context columns.

    ``PREDICTION_COLUMNS`` remains the minimal stable contract. Extra context
    columns such as ANO4, TRIMESTRE, AGLOMERADO, Region, or demographic
    indicators are allowed and make temporal/geographic diagnostics less
    dependent on fragile notebook-side joins.
    """

    y_true = frame[target].to_numpy()
    y_pred = estimator.predict(frame[feature_columns])
    predictions = pd.DataFrame(
        {
            "run_id": run_id,
            "model": model_name,
            "split": split_name,
            ROW_ID_COLUMN: frame[ROW_ID_COLUMN].to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    predictions["residual"] = predictions["y_true"] - predictions["y_pred"]
    predictions["abs_error"] = predictions["residual"].abs()
    predictions["squared_error"] = predictions["residual"] ** 2
    try:
        predictions["income_decile"] = pd.qcut(
            predictions["y_true"], q=10, labels=False, duplicates="drop"
        )
        predictions["income_decile"] = predictions["income_decile"].astype("Int64") + 1
    except ValueError:
        predictions["income_decile"] = pd.Series([pd.NA] * len(predictions), dtype="Int64")

    protected_columns = set(predictions.columns)
    for column in context_columns or []:
        column = str(column)
        if column in frame.columns and column not in protected_columns:
            predictions[column] = frame[column].to_numpy()

    return predictions


def _metrics_long(comparison: pd.DataFrame, run_id: str) -> pd.DataFrame:
    metric_columns = [
        column
        for column in comparison.columns
        if column not in {"model", "best_params"}
        and pd.api.types.is_numeric_dtype(comparison[column])
    ]
    long = comparison.melt(
        id_vars=["model"], value_vars=metric_columns, var_name="metric", value_name="value"
    )
    long.insert(0, "run_id", run_id)
    return long


def _write_predictions(
    predictions_by_split: dict[str, list[pd.DataFrame]], run_dir: Path
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for split_name in ["test", "validation"]:
        predictions = pd.concat(predictions_by_split[split_name], ignore_index=True)
        missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
        if missing:
            raise ValueError(
                "Prediction artifact is missing required columns: " + ", ".join(missing)
            )
        path = run_dir / "predictions" / f"{split_name}_predictions.parquet"
        predictions.to_parquet(path, index=False)
        written[split_name] = path
    return written


def _read_saved_predictions(run_dir: Path) -> pd.DataFrame:
    frames = []
    for split_name in ["test", "validation"]:
        path = run_dir / "predictions" / f"{split_name}_predictions.parquet"
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def _write_diagnostics_from_predictions(run_dir: Path) -> dict[str, Path]:
    predictions = _read_saved_predictions(run_dir)
    residual_summary = (
        predictions.groupby(["run_id", "model", "split"], dropna=False)
        .agg(
            row_count=(ROW_ID_COLUMN, "count"),
            residual_mean=("residual", "mean"),
            residual_std=("residual", "std"),
            residual_min=("residual", "min"),
            residual_max=("residual", "max"),
            abs_error_mean=("abs_error", "mean"),
            squared_error_mean=("squared_error", "mean"),
        )
        .reset_index()
    )
    error_by_decile = (
        predictions.groupby(["run_id", "model", "split", "income_decile"], dropna=False)
        .agg(
            row_count=(ROW_ID_COLUMN, "count"),
            y_true_mean=("y_true", "mean"),
            y_pred_mean=("y_pred", "mean"),
            residual_mean=("residual", "mean"),
            abs_error_mean=("abs_error", "mean"),
            squared_error_mean=("squared_error", "mean"),
        )
        .reset_index()
    )
    distribution_summary = (
        predictions.groupby(["run_id", "model", "split"], dropna=False)
        .agg(
            row_count=(ROW_ID_COLUMN, "count"),
            y_true_mean=("y_true", "mean"),
            y_true_std=("y_true", "std"),
            y_true_min=("y_true", "min"),
            y_true_p25=("y_true", lambda s: s.quantile(0.25)),
            y_true_median=("y_true", "median"),
            y_true_p75=("y_true", lambda s: s.quantile(0.75)),
            y_true_max=("y_true", "max"),
            y_pred_mean=("y_pred", "mean"),
            y_pred_std=("y_pred", "std"),
            y_pred_min=("y_pred", "min"),
            y_pred_p25=("y_pred", lambda s: s.quantile(0.25)),
            y_pred_median=("y_pred", "median"),
            y_pred_p75=("y_pred", lambda s: s.quantile(0.75)),
            y_pred_max=("y_pred", "max"),
        )
        .reset_index()
    )
    paths = {
        "residual_summary": run_dir / "diagnostics" / "residual_summary.csv",
        "error_by_income_decile": run_dir / "diagnostics" / "error_by_income_decile.csv",
        "prediction_distribution_summary": run_dir
        / "diagnostics"
        / "prediction_distribution_summary.csv",
    }
    residual_summary.to_csv(paths["residual_summary"], index=False)
    error_by_decile.to_csv(paths["error_by_income_decile"], index=False)
    distribution_summary.to_csv(paths["prediction_distribution_summary"], index=False)
    return paths




def _coefficient_table(estimator: Any) -> pd.DataFrame | None:
    """Return fitted linear-model coefficients from a pipeline when available."""

    if not hasattr(estimator, "named_steps") or "reg" not in estimator.named_steps:
        return None
    regressor = estimator.named_steps["reg"]
    if not hasattr(regressor, "coef_"):
        return None
    coefficients = np.ravel(regressor.coef_).astype(float)
    preprocessor = estimator.named_steps.get("preproc")
    try:
        feature_names = (
            list(preprocessor.get_feature_names_out()) if preprocessor is not None else []
        )
    except Exception:  # noqa: BLE001 - coefficient diagnostics should not fail a completed fit.
        feature_names = []
    if len(feature_names) != len(coefficients):
        feature_names = [f"feature_{index}" for index in range(len(coefficients))]
    table = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    table["abs_coefficient"] = table["coefficient"].abs()
    table = table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    table.insert(0, "abs_rank", range(1, len(table) + 1))
    return table


LINEAR_MODEL_DISPLAY_NAMES = {"LinearRegression", "Ridge", "Lasso"}


def _coefficient_pct_change(coefficient: float) -> float:
    """Convert a log10-income coefficient into a percentage income difference."""

    return float(100.0 * (10.0 ** float(coefficient) - 1.0))


def _metadata_lookup(feature_metadata: Sequence[Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Return feature metadata keyed by raw feature column."""

    lookup: dict[str, dict[str, Any]] = {}
    for row in feature_metadata or []:
        if not isinstance(row, Mapping):
            continue
        column = row.get("column")
        if column is not None:
            lookup[str(column)] = dict(row)
    return lookup


def _as_transformer_columns(columns: Any) -> list[str]:
    """Normalize sklearn ColumnTransformer column selectors into strings."""

    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    if isinstance(columns, slice):
        return []
    if isinstance(columns, Sequence):
        return [str(column) for column in columns]
    return []


def _named_step(transformer: Any, step_name: str) -> Any | None:
    """Return a named pipeline step if available."""

    named_steps = getattr(transformer, "named_steps", None)
    if isinstance(named_steps, Mapping):
        return named_steps.get(step_name)
    return None


def _drop_index_for_feature(drop_idx: Any, index: int) -> int | None:
    """Return the dropped category index for one encoded feature."""

    if drop_idx is None:
        return None
    try:
        value = drop_idx[index]
    except Exception:  # noqa: BLE001 - diagnostics should not fail a fitted model.
        return None
    if value is None or pd.isna(value):
        return None
    return int(value)


def _feature_metadata_for_column(
    raw_feature: str,
    metadata_by_column: Mapping[str, Mapping[str, Any]],
    *,
    default_type: str,
    transformer_name: str,
) -> dict[str, Any]:
    """Return a normalized metadata row for one raw feature."""

    metadata = dict(metadata_by_column.get(raw_feature, {}))
    feature_type = str(metadata.get("feature_type") or default_type)
    is_fixed_effect = bool(metadata.get("is_fixed_effect", False)) or feature_type == "fixed_effect"
    if transformer_name == "cat" and is_fixed_effect:
        feature_type = "fixed_effect"
    return {
        "raw_feature": raw_feature,
        "feature_family": metadata.get(
            "feature_family", "fixed_effect" if is_fixed_effect else "unregistered"
        ),
        "feature_type": feature_type,
        "is_fixed_effect": is_fixed_effect,
        "fe_name": metadata.get("fe_name"),
        "source_columns": metadata.get("source_columns"),
        "declared_families": metadata.get("declared_families"),
        "declared_types": metadata.get("declared_types"),
    }


def _interpretation_unit(
    *,
    transformer_name: str,
    feature_type: str,
    level: str | None,
    reference_level: str | None,
) -> str:
    """Return a thesis-facing coefficient interpretation unit."""

    if feature_type in {"continuous_numeric", "numeric"} or transformer_name == "num":
        return "+1 SD of raw feature"
    if feature_type == "binary" or transformer_name == "bin":
        return "1 vs 0"
    if feature_type == "fixed_effect":
        return "fixed-effect level vs omitted reference"
    if level is not None or reference_level is not None:
        return "category level vs omitted reference"
    return "transformed feature unit"


def _design_matrix_feature_rows(
    estimator: Any,
    feature_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build one row per transformed design-matrix column when possible."""

    if not hasattr(estimator, "named_steps"):
        return []
    preprocessor = estimator.named_steps.get("preproc")
    if preprocessor is None or not hasattr(preprocessor, "transformers_"):
        return []

    metadata_by_column = _metadata_lookup(feature_metadata)
    rows: list[dict[str, Any]] = []

    for transformer_name, fitted_transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder" or fitted_transformer == "drop":
            continue
        raw_columns = _as_transformer_columns(columns)
        if not raw_columns:
            continue

        if transformer_name == "cat":
            encoder = _named_step(fitted_transformer, "ohe")
            if encoder is None or not hasattr(encoder, "categories_"):
                continue
            try:
                encoded_feature_names = list(encoder.get_feature_names_out(raw_columns))
            except Exception:  # noqa: BLE001
                encoded_feature_names = []
            name_cursor = 0
            for column_index, raw_feature in enumerate(raw_columns):
                categories = [str(value) for value in encoder.categories_[column_index].tolist()]
                dropped_index = _drop_index_for_feature(getattr(encoder, "drop_idx_", None), column_index)
                reference_level = (
                    categories[dropped_index]
                    if dropped_index is not None and dropped_index < len(categories)
                    else None
                )
                base_metadata = _feature_metadata_for_column(
                    raw_feature,
                    metadata_by_column,
                    default_type="categorical",
                    transformer_name=transformer_name,
                )
                for category_index, level in enumerate(categories):
                    if dropped_index is not None and category_index == dropped_index:
                        continue
                    if name_cursor < len(encoded_feature_names):
                        inner_name = encoded_feature_names[name_cursor]
                    else:
                        inner_name = f"{raw_feature}_{level}"
                    name_cursor += 1
                    rows.append(
                        {
                            "transformed_feature": f"{transformer_name}__{inner_name}",
                            "transformer": transformer_name,
                            **base_metadata,
                            "level": level,
                            "reference_level": reference_level,
                            "is_reference": False,
                            "interpretation_unit": _interpretation_unit(
                                transformer_name=transformer_name,
                                feature_type=str(base_metadata["feature_type"]),
                                level=level,
                                reference_level=reference_level,
                            ),
                        }
                    )
            continue

        try:
            inner_names = list(fitted_transformer.get_feature_names_out(raw_columns))
        except Exception:  # noqa: BLE001
            inner_names = raw_columns
        for raw_feature, inner_name in zip(raw_columns, inner_names, strict=False):
            default_type = "binary" if transformer_name == "bin" else "continuous_numeric"
            base_metadata = _feature_metadata_for_column(
                raw_feature,
                metadata_by_column,
                default_type=default_type,
                transformer_name=transformer_name,
            )
            rows.append(
                {
                    "transformed_feature": f"{transformer_name}__{inner_name}",
                    "transformer": transformer_name,
                    **base_metadata,
                    "level": None,
                    "reference_level": None,
                    "is_reference": False,
                    "interpretation_unit": _interpretation_unit(
                        transformer_name=transformer_name,
                        feature_type=str(base_metadata["feature_type"]),
                        level=None,
                        reference_level=None,
                    ),
                }
            )

    return rows


def _fallback_design_row(
    transformed_feature: str,
    feature_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Best-effort metadata for a coefficient not matched to design rows."""

    transformer, _, remainder = transformed_feature.partition("__")
    metadata_by_column = _metadata_lookup(feature_metadata)
    raw_feature = remainder
    level = None

    if transformer == "cat":
        candidates = sorted(metadata_by_column, key=len, reverse=True)
        for candidate in candidates:
            prefix = f"{candidate}_"
            if remainder.startswith(prefix):
                raw_feature = candidate
                level = remainder.removeprefix(prefix)
                break

    default_type = {
        "num": "continuous_numeric",
        "bin": "binary",
        "cat": "categorical",
    }.get(transformer, "unknown")
    base_metadata = _feature_metadata_for_column(
        raw_feature,
        metadata_by_column,
        default_type=default_type,
        transformer_name=transformer,
    )
    return {
        "transformed_feature": transformed_feature,
        "transformer": transformer or None,
        **base_metadata,
        "level": level,
        "reference_level": base_metadata.get("reference_level"),
        "is_reference": False,
        "interpretation_unit": _interpretation_unit(
            transformer_name=transformer,
            feature_type=str(base_metadata["feature_type"]),
            level=level,
            reference_level=base_metadata.get("reference_level"),
        ),
    }


def _transformed_coefficient_rows(
    *,
    run_id: str,
    experiment_id: str | None,
    model_name: str,
    estimator: Any,
    feature_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return enriched coefficient rows for fitted linear models."""

    if model_name not in LINEAR_MODEL_DISPLAY_NAMES:
        return []
    coefficient_table = _coefficient_table(estimator)
    if coefficient_table is None:
        return []

    design_rows = _design_matrix_feature_rows(estimator, feature_metadata)
    design_by_feature = {str(row["transformed_feature"]): row for row in design_rows}
    rows: list[dict[str, Any]] = []

    for rank, record in enumerate(
        coefficient_table.sort_values("abs_coefficient", ascending=False).to_dict(orient="records"),
        start=1,
    ):
        transformed_feature = str(record["feature"])
        design = dict(
            design_by_feature.get(
                transformed_feature,
                _fallback_design_row(transformed_feature, feature_metadata),
            )
        )
        coefficient = float(record["coefficient"])
        rows.append(
            {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "model": model_name,
                "abs_rank": int(rank),
                **design,
                "coef_log10": coefficient,
                "abs_coef_log10": float(abs(coefficient)),
                "income_factor": float(10.0 ** coefficient),
                "pct_change": _coefficient_pct_change(coefficient),
                "sign": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
            }
        )
    return rows



def _coefficient_export_cell(value: Any) -> Any:
    """Return a CSV-/pandas-safe scalar for coefficient diagnostic tables."""

    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _coefficient_export_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a coefficient export frame with hashable scalar cells.

    Some metadata fields, especially source_columns and declared_* lists, are
    naturally list-valued. Pandas drop_duplicates/groupby requires hashable
    values, so we serialize list/dict metadata once before writing derived
    tables.
    """

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (list, tuple, set, dict))).any():
            frame[column] = frame[column].map(_coefficient_export_cell)
    return frame

def _write_transformed_coefficients(
    *,
    run_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    """Write thesis-facing coefficient diagnostics from enriched rows."""

    if not rows:
        return {}

    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    frame = _coefficient_export_frame(rows)
    frame = frame.sort_values(["model", "abs_coef_log10"], ascending=[True, False]).reset_index(drop=True)

    preferred_columns = [
        "run_id",
        "experiment_id",
        "model",
        "abs_rank",
        "transformed_feature",
        "raw_feature",
        "feature_family",
        "feature_type",
        "transformer",
        "level",
        "reference_level",
        "is_reference",
        "is_fixed_effect",
        "fe_name",
        "source_columns",
        "coef_log10",
        "abs_coef_log10",
        "income_factor",
        "pct_change",
        "sign",
        "interpretation_unit",
        "declared_families",
        "declared_types",
    ]
    ordered = [column for column in preferred_columns if column in frame.columns]
    ordered.extend([column for column in frame.columns if column not in ordered])
    frame = frame[ordered]

    paths: dict[str, Path] = {}
    coefficients_path = diagnostics_dir / "coefficients_transformed_all.csv"
    frame.to_csv(coefficients_path, index=False)
    paths["coefficients_transformed_all"] = coefficients_path

    design_columns = [
        column
        for column in [
            "model",
            "transformed_feature",
            "raw_feature",
            "feature_family",
            "feature_type",
            "transformer",
            "level",
            "reference_level",
            "is_fixed_effect",
            "fe_name",
            "source_columns",
            "interpretation_unit",
        ]
        if column in frame.columns
    ]
    design_path = diagnostics_dir / "design_matrix_features.csv"
    frame[design_columns].drop_duplicates().to_csv(design_path, index=False)
    paths["design_matrix_features"] = design_path

    ref_frame = frame[
        frame.get("reference_level", pd.Series(index=frame.index, dtype=object)).notna()
    ].copy()
    if not ref_frame.empty:
        reference_columns = [
            column
            for column in [
                "run_id",
                "experiment_id",
                "model",
                "raw_feature",
                "feature_family",
                "feature_type",
                "transformer",
                "reference_level",
                "is_fixed_effect",
                "fe_name",
                "source_columns",
            ]
            if column in ref_frame.columns
        ]
        references = ref_frame[reference_columns].drop_duplicates()
        onehot_references = references[~references.get("is_fixed_effect", False).astype(bool)]
        fixed_effect_references = references[references.get("is_fixed_effect", False).astype(bool)]
        if not onehot_references.empty:
            path = diagnostics_dir / "onehot_reference_levels.csv"
            onehot_references.to_csv(path, index=False)
            paths["onehot_reference_levels"] = path
        if not fixed_effect_references.empty:
            path = diagnostics_dir / "fixed_effect_reference_levels.csv"
            fixed_effect_references.to_csv(path, index=False)
            paths["fixed_effect_reference_levels"] = path

    if {"model", "feature_family", "is_fixed_effect", "abs_coef_log10"}.issubset(frame.columns):
        family_summary = (
            frame.groupby(["model", "feature_family", "is_fixed_effect"], dropna=False)
            .agg(
                n_coefficients=("coef_log10", "count"),
                mean_abs_coef_log10=("abs_coef_log10", "mean"),
                median_abs_coef_log10=("abs_coef_log10", "median"),
                p90_abs_coef_log10=("abs_coef_log10", lambda s: float(s.quantile(0.90))),
                max_abs_coef_log10=("abs_coef_log10", "max"),
                sum_abs_coef_log10=("abs_coef_log10", "sum"),
            )
            .reset_index()
            .sort_values(["model", "is_fixed_effect", "sum_abs_coef_log10"], ascending=[True, True, False])
        )
        path = diagnostics_dir / "coefficient_family_summary.csv"
        family_summary.to_csv(path, index=False)
        paths["coefficient_family_summary"] = path

        top_by_family = (
            frame.sort_values("abs_coef_log10", ascending=False)
            .groupby(["model", "feature_family", "is_fixed_effect"], dropna=False)
            .head(10)
            .reset_index(drop=True)
        )
        path = diagnostics_dir / "top_coefficients_by_family.csv"
        top_by_family.to_csv(path, index=False)
        paths["top_coefficients_by_family"] = path

    return paths




def _coefficient_norms(estimator: Any) -> dict[str, float | int] | None:
    table = _coefficient_table(estimator)
    if table is None:
        return None
    coefficients = table["coefficient"].to_numpy(dtype=float)
    return {
        "coefficient_l1_norm": float(np.sum(np.abs(coefficients))),
        "coefficient_l2_norm": float(np.sqrt(np.sum(coefficients**2))),
        "nonzero_coefficients": int(np.count_nonzero(np.abs(coefficients) > 1e-10)),
    }



def _write_best_coefficient_summary(
    *, run_dir: Path, model_name: str, run_id: str, estimator: Any
) -> Path | None:
    table = _coefficient_table(estimator)
    if table is None:
        return None
    table.insert(0, "run_id", run_id)
    table.insert(1, "model", model_name)
    path = run_dir / "diagnostics" / f"{model_name}_best_coefficients.csv"
    table.to_csv(path, index=False)
    return path


def _fixed_effect_coefficient_rows(
    *,
    run_id: str,
    model_name: str,
    estimator: Any,
    fixed_effects_used: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract coefficient rows corresponding to generated fixed-effect columns."""

    if model_name not in {"LinearRegression", "Ridge", "Lasso"} or not fixed_effects_used:
        return []
    coefficient_table = _coefficient_table(estimator)
    if coefficient_table is None:
        return []

    rows: list[dict[str, Any]] = []
    for fe_spec in fixed_effects_used:
        fe_name = str(fe_spec["fe_name"])
        generated_column = str(fe_spec["generated_column"])
        prefixes = [
            f"cat__{generated_column}_",
            f"fe__{generated_column}_",
            f"cat_drop_first__{generated_column}_",
        ]
        fe_rows = []
        for record in coefficient_table[["feature", "coefficient", "abs_coefficient"]].to_dict(
            orient="records"
        ):
            feature = str(record["feature"])
            level = None
            for prefix in prefixes:
                if feature.startswith(prefix):
                    level = feature.removeprefix(prefix)
                    break
            if level is None:
                continue
            coefficient = float(record["coefficient"])
            fe_rows.append(
                {
                    "run_id": run_id,
                    "model": model_name,
                    "fe_name": fe_name,
                    "level": level,
                    "feature": feature,
                    "coefficient": coefficient,
                    "abs_coefficient": float(record["abs_coefficient"]),
                    "income_factor": float(10.0 ** coefficient),
                    "pct_change": _coefficient_pct_change(coefficient),
                    "sign": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
                }
            )
        fe_rows = sorted(fe_rows, key=lambda row: row["abs_coefficient"], reverse=True)
        for rank, row in enumerate(fe_rows, start=1):
            row["rank"] = rank
            rows.append(row)
    return rows


def _write_fixed_effect_coefficients(
    *, run_dir: Path, rows: list[dict[str, Any]]
) -> Path | None:
    """Write the combined fixed-effect coefficient export when linear rows exist."""

    if not rows:
        return None
    columns = [
        "run_id",
        "model",
        "fe_name",
        "level",
        "feature",
        "coefficient",
        "abs_coefficient",
        "income_factor",
        "pct_change",
        "sign",
        "rank",
    ]
    path = run_dir / "diagnostics" / "fixed_effect_coefficients.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path
