from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eph_income.config import load_feature_contract
from eph_income.dataset import build_modeling_dataset

INPUT_DIR = Path("data/annual_preprocessed_inputs")


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
    file_2022 = tmp_path / "EPHARG_annual_input_22_sample.csv"
    file_2024 = tmp_path / "EPHARG_annual_input_24_sample.csv"
    output_dataset = tmp_path / "processed" / "modeling_dataset.parquet"
    output_metadata = tmp_path / "processed" / "dataset_metadata.json"

    sample_2022 = _write_training_sample(
        INPUT_DIR / "EPHARG_annual_input_22.csv", file_2022, nrows=50
    )
    sample_2022 = _add_missing_prop_case(sample_2022)
    sample_2022.to_csv(file_2022, index=False)
    sample_2024 = _write_training_sample(
        INPUT_DIR / "EPHARG_annual_input_24.csv", file_2024, nrows=20
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
    assert "source_year" not in dataset.columns
    assert "INGRESO" not in dataset.columns
    leakage_columns = {
        "P21",
        "T_VI",
        "V12_M",
        "V2_M",
        "V3_M",
        "V5_M",
        "TOT_P12",
        "PP08D1",
        "logP21",
        "logT_VI",
        "logV12_M",
        "logV2_M",
        "logV3_M",
        "logV5_M",
        "logTOT_P12",
        "logPP08D1",
    }
    assert leakage_columns.isdisjoint(dataset.columns)

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
    assert written_metadata["input_artifact_type"] == "annual_preprocessed_eph_inputs"
    assert written_metadata["upstream_input_files"] == written_metadata["input_files"]


def test_second_stage_feature_engineering_constructs_required_features() -> None:
    from eph_income.features import apply_second_stage_features

    raw = pd.DataFrame(
        {
            "CODUSU": ["h1", "h1", "h1", "h2"],
            "P03": [2, 10, 70, 25],
            "P09": [2, 4, 0, 5],
            "H10": [1, 1, 1, 2],
            "H11": [1, 1, 1, 2],
            "H12": [1, 1, 1, 3],
            "PP07G1": [1, 1, 1, 1],
            "PP07G2": [1, 1, 1, 1],
            "PP07G3": [1, 1, 1, 1],
            "PP07G4": [1, 1, 1, 1],
            "PP07H": [1, 1, 1, 1],
            "PP07I": [1, 1, 1, 1],
            "PP07J": [1, 1, 1, 1],
            "PP07K": [1, 1, 1, 1],
            "P07": [1, 1, 1, 1],
            "P08": [1, 1, 1, 1],
            "PP07G_59": [1, 5, 5, 1],
            "P21": [100, 0, 10, 20],
            "T_VI": [0, 0, 0, 0],
            "V12_M": [0, 0, 0, 0],
            "V2_M": [0, 0, 0, 0],
            "V3_M": [0, 0, 0, 0],
            "V5_M": [0, 0, 0, 0],
            "TOT_P12": [0, 0, 0, 0],
            "PP08D1": [0, 0, 0, 0],
            "ANO4": [2022, 2022, 2022, 2023],
            "TRIMESTRE": [1, 1, 1, 2],
        }
    )

    engineered, metadata = apply_second_stage_features(raw)

    assert engineered.loc[0, "sanitacion_nivel"] == 2
    assert engineered.loc[3, "sanitacion_nivel"] == 0
    assert "H10" not in engineered.columns
    assert "H11" not in engineered.columns
    assert "H12" not in engineered.columns
    assert "PP07G1" not in engineered.columns
    assert "P07" not in engineered.columns
    assert engineered.loc[0, "PP07G_59"] == 0

    h1 = engineered[engineered["CODUSU"] == "h1"].iloc[0]
    assert h1["Personas_0-3"] == 1
    assert h1["Personas_9-12"] == 1
    assert h1["Personas_65+"] == 1
    assert h1["Max_Nivel_Educativo"] == 4
    assert "logP21" in engineered.columns
    assert "sanitacion_nivel" in metadata.engineered_columns
    assert "ANO4" in metadata.categorical_columns
    assert "TRIMESTRE" in metadata.categorical_columns
