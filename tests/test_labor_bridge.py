import numpy as np
import pandas as pd
import pytest

from eph_income.labor_bridge import (
    LABOR_PROBABILITY_COLUMNS,
    LaborBridgeError,
    LaborFeatureContract,
    TwoStageLaborBridge,
    attach_probability_features,
)


def fixture(n=120):
    rng = np.random.default_rng(42)
    household = np.repeat(np.arange(n // 3), 3)
    age = rng.integers(8, 80, n)
    sex = rng.integers(1, 3, n)
    education = rng.integers(1, 8, n)
    active_score = -1.2 + 0.045 * age - 0.0005 * age**2 + 0.15 * (education >= 5)
    p_active = 1 / (1 + np.exp(-active_score))
    active = rng.binomial(1, p_active)
    p_unemp = 1 / (1 + np.exp(-(-1.8 + 0.3 * (education <= 2) + 0.2 * (age < 25))))
    unemployed = rng.binomial(1, p_unemp) * active
    estado = np.where(active == 0, 3, np.where(unemployed == 1, 2, 1))
    estado[age < 10] = 4
    return pd.DataFrame(
        {
            "CODUSU": [f"H{x}" for x in household],
            "NRO_HOGAR": 1,
            "P02": sex.astype(str),
            "P03": age,
            "P09": education.astype(str),
            "IX_TOT": 3,
            "ESTADO": estado,
            "PONDERA": rng.integers(100, 300, n),
        }
    )


CONTRACT = LaborFeatureContract(
    categorical=("P02", "P09"),
    numeric=("P03", "IX_TOT"),
)


@pytest.mark.parametrize("kind", ["logistic", "hist_gradient_boosting"])
def test_two_stage_bridge_crossfits_and_scores(kind):
    frame = fixture()
    bridge = TwoStageLaborBridge(CONTRACT, classifier_kind=kind, cv=4)
    oof = bridge.fit_oof(frame, weight_col="PONDERA")
    assert set(LABOR_PROBABILITY_COLUMNS) <= set(oof)
    assert set(oof.labor_probability_role) == {"OOF"}
    assert np.allclose(
        oof[["labor_p_employed", "labor_p_unemployed", "labor_p_inactive"]].sum(axis=1),
        1.0,
    )
    scored = bridge.predict_probabilities(frame.iloc[:10])
    assert len(scored) == 10
    assert set(scored.labor_probability_role) == {"FULL_FIT_INFERENCE"}


def test_estato_zero_is_not_used_as_supervised_target():
    frame = fixture()
    frame.loc[frame.index[:4], "ESTADO"] = 0
    bridge = TwoStageLaborBridge(CONTRACT, classifier_kind="logistic", cv=4)
    oof = bridge.fit_oof(frame)
    assert not set(frame.index[:4]) & set(oof.index)
    assert len(oof) == len(frame) - 4


def test_attach_probability_features_requires_exact_coverage():
    frame = pd.DataFrame(
        {
            "person_id": ["a", "b"],
            "x": [1, 2],
        }
    )
    probabilities = pd.DataFrame(
        {
            "person_id": ["a", "b"],
            "labor_p_active": [0.6, 0.7],
            "labor_p_employed": [0.54, 0.63],
            "labor_p_unemployed": [0.06, 0.07],
            "labor_p_inactive": [0.4, 0.3],
        }
    )
    joined = attach_probability_features(frame, probabilities, key_columns=["person_id"])
    assert list(joined.person_id) == ["a", "b"]
    assert set(LABOR_PROBABILITY_COLUMNS) <= set(joined)

    with pytest.raises(LaborBridgeError, match="coverage mismatch"):
        attach_probability_features(
            frame, probabilities.iloc[:1], key_columns=["person_id"]
        )


def test_missing_shared_feature_fails_closed():
    frame = fixture().drop(columns=["P09"])
    bridge = TwoStageLaborBridge(CONTRACT)
    with pytest.raises(LaborBridgeError, match="missing labor-bridge features"):
        bridge.fit_oof(frame)
