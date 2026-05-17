from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eph_income.config import load_feature_contract
from eph_income.dataset import build_modeling_dataset

TRAINING_DIR = Path("data/training")


def _write_training_sample(source: Path, destination: Path, *, nrows: int) -> pd.DataFrame:
    sample = pd.read_csv(source, nrows=nrows)
    sample.to_csv(destination, index=False)
    return sample


def _add_missing_prop_case(sample: pd.DataFrame) -> pd.DataFrame:
    valid_rows = sample[(sample["INGRESO"] == 1) & (sample["P47T"] > 0) & sample["PROP"].notna()]
    assert not valid_rows.empty, "Training sample must include at least one valid target row."
    missing_prop_row = valid_rows.iloc[[0]].copy()
    missing_prop_row["PROP"] = missing_prop_row["PROP"].astype("float64")
    missing_prop_row.loc[:, "PROP"] = np.nan
    return pd.concat([sample, missing_prop_row], ignore_index=True)


def test_build_modeling_dataset_filters_training_samples_and_writes_metadata(tmp_path) -> None:
    file_2022 = tmp_path / "EPHARG_train_22_sample.csv"
    file_2024 = tmp_path / "EPHARG_train_24_sample.csv"
    output_dataset = tmp_path / "processed" / "modeling_dataset.parquet"
    output_metadata = tmp_path / "processed" / "dataset_metadata.json"

    sample_2022 = _write_training_sample(
        TRAINING_DIR / "EPHARG_train_22.csv", file_2022, nrows=50
    )
    sample_2022 = _add_missing_prop_case(sample_2022)
    sample_2022.to_csv(file_2022, index=False)
    sample_2024 = _write_training_sample(
        TRAINING_DIR / "EPHARG_train_24.csv", file_2024, nrows=20
    )

    experiment_config = {
        "data": {
            "input_files": {"2022": str(file_2022), "2024": str(file_2024)},
            "processed_dataset": str(output_dataset),
        }
    }
    feature_contract = load_feature_contract()

    dataset, metadata = build_modeling_dataset(
        experiment_config,
        feature_contract,
        output_dataset_path=output_dataset,
        output_metadata_path=output_metadata,
    )

    combined = pd.concat(
        [
            sample_2022.assign(source_year="2022"),
            sample_2024.drop(
                columns=["V2_01_M", "V2_02_M", "V2_03_M", "V5_01_M", "V5_02_M", "V5_03_M"],
                errors="ignore",
            ).assign(source_year="2024"),
        ],
        ignore_index=True,
        sort=False,
    )
    expected_filtered = combined[
        (combined["INGRESO"] == 1) & (combined["P47T"] > 0) & combined["PROP"].notna()
    ].reset_index(drop=True)

    assert output_dataset.exists()
    assert output_metadata.exists()
    assert metadata["row_count_before_filters"] == len(combined)
    assert metadata["row_count_after_filters"] == len(expected_filtered)
    assert len(dataset) == len(expected_filtered)

    # The training samples include real rows excluded by INGRESO != 1 and P47T <= 0,
    # plus a copied training row with PROP set missing to exercise that inclusion criterion.
    assert (combined["INGRESO"] != 1).any()
    assert (combined["P47T"] <= 0).any()
    assert combined["PROP"].isna().any()
    assert dataset["PROP"].notna().all()

    assert np.allclose(dataset["logP47T"].to_numpy(), np.log10(expected_filtered["P47T"]))
    assert "row_id" in dataset.columns
    assert dataset["row_id"].is_unique
    assert dataset["row_id"].tolist() == list(range(len(dataset)))

    # Known 2024/2025 harmonization columns are dropped, and forbidden predictors do not enter X.
    assert "V2_01_M" not in dataset.columns
    assert "V2_02_M" not in dataset.columns
    assert "V5_03_M" not in dataset.columns
    assert "P47T" not in dataset.columns
    assert "CODUSU" not in dataset.columns
    assert "Q" not in dataset.columns
    assert "P21" not in dataset.columns

    # Baseline temporal variables remain available for downstream modeling.
    assert "ANO4" in dataset.columns
    assert "TRIMESTRE" in dataset.columns

    written_metadata = json.loads(output_metadata.read_text(encoding="utf-8"))
    assert written_metadata["row_count_before_filters"] == len(combined)
    assert written_metadata["row_count_after_filters"] == len(expected_filtered)
    assert written_metadata["target_definition"] == {
        "name": "logP47T",
        "source": "P47T",
        "transform": "log10",
    }
    assert written_metadata["inclusion_criteria_applied"] == {
        "INGRESO": 1,
        "require_positive": ["P47T"],
        "require_not_missing": ["PROP"],
    }
    assert written_metadata["forbidden_predictor_check_passed"] is True
