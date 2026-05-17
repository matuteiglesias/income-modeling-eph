"""Second-stage modeling feature engineering for EPH income experiments.

The input files are annual preprocessed upstream artifacts. This module only
implements modeling-level transformations migrated from the legacy student
feature engineering script; it does not download raw EPH files, deflate monetary
columns, or recompute upstream ranks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LEGACY_RECODE_MAPS: dict[str, dict[object, object]] = {
    "V01": {0: 9},
    "H05": {0: 9},
    "H06": {0: 9},
    "H07": {0: 1},
    "H08": {0: 1},
    "H09": {0: 1},
    "H10": {0: 1},
    "H11": {0: 9},
    "H12": {0: 9},
    "H13": {0: 9},
    "H14": {0: 9},
    "P05": {0: 1},
    "P07": {0: 1},
    "P08": {0: 9},
    "P10": {0: 9},
    "PP07G_59": {1: 0},
    "PROP": {0: 9},
    "CONDACT": {0: 9},
}

DETAILED_LABOR_BENEFIT_COLUMNS = (
    "PP07G1",
    "PP07G2",
    "PP07G3",
    "PP07G4",
    "PP07H",
    "PP07I",
    "PP07J",
    "PP07K",
)
EDUCATION_SUMMARY_DROP_COLUMNS = ("P07", "P08")
SANITATION_SOURCE_COLUMNS = ("H10", "H11", "H12")
INCOME_COMPONENT_COLUMNS = ("P21", "T_VI", "V12_M", "V2_M", "V3_M", "V5_M", "TOT_P12", "PP08D1")
AGE_COMPOSITION_COLUMNS = (
    "Personas_0-3",
    "Personas_4-8",
    "Personas_9-12",
    "Personas_13-17",
    "Personas_18-24",
    "Personas_25-64",
    "Personas_65+",
)
DECLARED_CATEGORICAL_COLUMNS = (
    "ANO4",
    "TRIMESTRE",
    "AGLOMERADO",
    "V01",
    "H05",
    "H06",
    "H07",
    "H08",
    "H09",
    "PROP",
    "H14",
    "H13",
    "CAT_INAC",
    "CAT_OCUP",
    "CH07",
    "P10",
    "P05",
    "Region",
    "P09",
    "CONDACT",
    "PP07G_59",
    "P02",
)


@dataclass
class FeatureEngineeringMetadata:
    """Metadata about deterministic second-stage feature engineering."""

    engineered_columns: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)


def _columns_present(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def recode_legacy_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply legacy categorical recodes where columns are present."""

    result = frame.copy()
    for column, replacements in LEGACY_RECODE_MAPS.items():
        if column in result.columns:
            result[column] = result[column].replace(replacements)
    if "P09" in result.columns:
        result["P09"] = result["P09"].astype("string").replace({"0": "Z_NO_APLICA"})
    return result


def construct_sanitacion_nivel(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Construct the ordinal sanitation index from H10/H12 and drop H10/H11/H12."""

    result = frame.copy()
    engineered: list[str] = []
    if {"H10", "H12"}.issubset(result.columns):
        tiene_banio = result["H10"] == 1
        banio_mejorado = tiene_banio & result["H12"].isin([1, 2])
        result["sanitacion_nivel"] = np.select(
            condlist=[~tiene_banio, banio_mejorado, tiene_banio],
            choicelist=[0, 2, 1],
            default=9,
        )
        engineered.append("sanitacion_nivel")
    return result, engineered


def add_log_income_components(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create legacy log income-component transforms before leakage exclusion."""

    result = frame.copy()
    engineered: list[str] = []
    for column in INCOME_COMPONENT_COLUMNS:
        if column in result.columns:
            new_column = f"log{column}"
            result[new_column] = np.nan
            positive = result[column] > 0
            result.loc[positive, new_column] = np.log10(result.loc[positive, column])
            engineered.append(new_column)
    return result, engineered


def _age_group(age: object) -> str:
    if pd.isna(age):
        return "65+"
    age_value = float(age)
    if age_value <= 3:
        return "0-3"
    if age_value <= 8:
        return "4-8"
    if age_value <= 12:
        return "9-12"
    if age_value <= 17:
        return "13-17"
    if age_value <= 24:
        return "18-24"
    if age_value <= 64:
        return "25-64"
    return "65+"


def add_household_age_composition(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add household age-composition counts by CODUSU from P03."""

    result = frame.copy()
    if not {"CODUSU", "P03"}.issubset(result.columns):
        return result, []

    age_groups = result["P03"].apply(_age_group)
    counts = result.groupby(["CODUSU", age_groups], observed=False).size().unstack(fill_value=0)
    engineered: list[str] = []
    for column in AGE_COMPOSITION_COLUMNS:
        group = column.removeprefix("Personas_")
        if group in counts.columns:
            result[column] = result["CODUSU"].map(counts[group]).fillna(0).astype("int64")
        else:
            result[column] = 0
        engineered.append(column)
    return result, engineered


def add_max_household_education(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add maximum educational level within household from P09 grouped by CODUSU."""

    result = frame.copy()
    if not {"CODUSU", "P09"}.issubset(result.columns):
        return result, []

    p09_num = pd.to_numeric(result["P09"].replace("Z_NO_APLICA", np.nan), errors="coerce")
    result["Max_Nivel_Educativo"] = p09_num.groupby(result["CODUSU"]).transform("max").fillna(0)
    return result, ["Max_Nivel_Educativo"]


def cast_declared_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast declared categorical variables to pandas category where present."""

    result = frame.copy()
    for column in DECLARED_CATEGORICAL_COLUMNS:
        if column in result.columns:
            result[column] = result[column].astype("category")
    return result


def apply_second_stage_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, FeatureEngineeringMetadata]:
    """Apply all deterministic second-stage modeling features."""

    result = recode_legacy_categoricals(frame)
    engineered_columns: list[str] = []
    dropped_columns: list[str] = []

    result, engineered = construct_sanitacion_nivel(result)
    engineered_columns.extend(engineered)

    to_drop = _columns_present(result, DETAILED_LABOR_BENEFIT_COLUMNS)
    result = result.drop(columns=to_drop)
    dropped_columns.extend(to_drop)

    to_drop = _columns_present(result, EDUCATION_SUMMARY_DROP_COLUMNS)
    result = result.drop(columns=to_drop)
    dropped_columns.extend(to_drop)

    to_drop = _columns_present(result, SANITATION_SOURCE_COLUMNS)
    result = result.drop(columns=to_drop)
    dropped_columns.extend(to_drop)

    result, engineered = add_log_income_components(result)
    engineered_columns.extend(engineered)

    result, engineered = add_household_age_composition(result)
    engineered_columns.extend(engineered)

    result, engineered = add_max_household_education(result)
    engineered_columns.extend(engineered)

    result = cast_declared_categoricals(result)
    categorical_columns = result.select_dtypes(include=["category", "object", "string"]).columns.tolist()
    numeric_columns = result.select_dtypes(include=["number", "bool"]).columns.tolist()

    metadata = FeatureEngineeringMetadata(
        engineered_columns=sorted(set(engineered_columns)),
        dropped_columns=dropped_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    return result, metadata
