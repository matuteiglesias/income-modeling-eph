from __future__ import annotations

import pytest

from eph_income.config import load_experiment_config, load_feature_contract
from eph_income.contracts import (
    assert_no_forbidden_predictors,
    get_forbidden_predictors,
    validate_target_contract,
    validate_temporal_features,
)


def test_feature_contract_yaml_loads() -> None:
    contract = load_feature_contract()

    assert contract["target"]["name"] == "logP47T"
    assert "forbidden_predictors" in contract


def test_experiment_config_yaml_loads() -> None:
    config = load_experiment_config()

    assert config["experiment"]["id"] == "baseline_income_prediction_v1"
    assert config["data"]["processed_dataset"] == "data/processed/modeling_dataset.parquet"


def test_forbidden_predictor_groups_flatten_correctly() -> None:
    contract = load_feature_contract()
    forbidden = get_forbidden_predictors(contract)

    assert {"P47T", "logP47T", "P21", "CODUSU"}.issubset(forbidden)
    assert {"Q", "source_year", "INGRESO", "T_VI", "logTOT_P12", "INGRESO_SBS"}.issubset(forbidden)
    assert contract["forbidden_predictors"]["identifiers"] == ["CODUSU"]
    assert contract["forbidden_predictors"]["temporal_derived"] == ["Q", "source_year"]
    assert contract["forbidden_predictors"]["sample_indicators"] == ["INGRESO"]


def test_assert_no_forbidden_predictors_rejects_p47t() -> None:
    forbidden = get_forbidden_predictors(load_feature_contract())

    with pytest.raises(ValueError, match="P47T"):
        assert_no_forbidden_predictors(["P47T", "CH04"], forbidden)


def test_assert_no_forbidden_predictors_rejects_logp47t() -> None:
    forbidden = get_forbidden_predictors(load_feature_contract())

    with pytest.raises(ValueError, match="logP47T"):
        assert_no_forbidden_predictors(["logP47T", "CH04"], forbidden)


def test_assert_no_forbidden_predictors_rejects_income_component_p21() -> None:
    forbidden = get_forbidden_predictors(load_feature_contract())

    with pytest.raises(ValueError, match="P21"):
        assert_no_forbidden_predictors(["P21", "CH04"], forbidden)


def test_assert_no_forbidden_predictors_rejects_identifier_codusu() -> None:
    forbidden = get_forbidden_predictors(load_feature_contract())

    with pytest.raises(ValueError, match="CODUSU"):
        assert_no_forbidden_predictors(["CODUSU", "CH04"], forbidden)


def test_assert_no_forbidden_predictors_lists_all_offending_columns() -> None:
    forbidden = get_forbidden_predictors(load_feature_contract())

    with pytest.raises(ValueError) as excinfo:
        assert_no_forbidden_predictors(["P47T", "P21", "CODUSU", "CH04"], forbidden)

    message = str(excinfo.value)
    assert "P47T" in message
    assert "P21" in message
    assert "CODUSU" in message


def test_assert_no_forbidden_predictors_allows_safe_columns() -> None:
    forbidden = get_forbidden_predictors(load_feature_contract())

    assert_no_forbidden_predictors(["CH04", "CH06", "ANO4", "TRIMESTRE"], forbidden)


def test_target_contract_loads_correctly() -> None:
    target = validate_target_contract(load_feature_contract())

    assert target == {"name": "logP47T", "source": "P47T", "transform": "log10"}


def test_temporal_inclusion_decision_loads_correctly() -> None:
    temporal = validate_temporal_features(load_feature_contract())

    assert set(temporal["include"]) >= {"ANO4", "TRIMESTRE"}
    assert temporal["decision"] == "include_in_baseline"
