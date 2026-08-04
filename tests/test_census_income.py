import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from eph_income.census_income import StageSpec, StagedClassifierRegressor, validate_predictions


def _sample():
    return pd.DataFrame({"sample_person_id": ["h1:p1", "h1:p2", "h2:p1"], "period": [2010] * 3})


def _predictions():
    sample = _sample()
    return sample.assign(
        prediction_value=[-0.1, 3.2, 3.8], prediction_transform="log10_ars",
        monetary_reference="unresolved", classification="projected", model_release_id="fixture",
    )


def test_prediction_contract_preserves_ids_and_allows_finite_negative_logs():
    qa = validate_predictions(_sample(), _predictions())
    assert qa["row_count"] == 3
    assert qa["negative_count"] == 1


@pytest.mark.parametrize("mutation,match", [
    (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "Duplicate"),
    (lambda frame: frame.iloc[:-1], "coverage mismatch"),
    (lambda frame: pd.concat([frame, frame.iloc[[0]].assign(sample_person_id="extra")]), "coverage mismatch"),
    (lambda frame: frame.assign(prediction_transform="natural_log"), "transform"),
    (lambda frame: frame.assign(prediction_value=[1, np.nan, 2]), "NaN or infinity"),
])
def test_prediction_contract_rejects_identity_and_value_failures(mutation, match):
    with pytest.raises(ValueError, match=match):
        validate_predictions(_sample(), mutation(_predictions()))


def test_stages_become_features_and_missing_schema_stops():
    frame = pd.DataFrame({
        "x": range(9), "class_a": [0, 1, 0] * 3, "class_b": [1, 0, 1] * 3,
        "income": np.linspace(1, 2, 9),
    })
    model = StagedClassifierRegressor([
        (StageSpec("a", "class_a", "pred_a", ("x",)), DummyClassifier(strategy="most_frequent")),
        (StageSpec("b", "class_b", "pred_b", ("x", "pred_a")), DummyClassifier(strategy="most_frequent")),
    ], DummyRegressor(strategy="mean"), ["x", "pred_a", "pred_b"]).fit(frame, "income")
    transformed = model.transform_stages(frame[["x"]])
    assert {"pred_a", "pred_b"} <= set(transformed)
    assert len(model.predict(frame[["x"]])) == len(frame)
    with pytest.raises(ValueError, match="missing required columns"):
        model.predict(pd.DataFrame({"wrong": [1]}))


def test_release_script_fixture_is_deterministic(tmp_path):
    # Importing the executable module through runpy would obscure its public fixture helper.
    import importlib.util

    path = __import__("pathlib").Path("scripts/13_census_income_release.py")
    spec = importlib.util.spec_from_file_location("census_release_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = module.run_fixture(tmp_path / "one")
    second = module.run_fixture(tmp_path / "two")
    first_manifest = json.loads((first / "manifest.json").read_text())
    second_manifest = json.loads((second / "manifest.json").read_text())
    assert first_manifest["output_sha256"] == second_manifest["output_sha256"]
    assert first_manifest["id_namespace_hash"] == second_manifest["id_namespace_hash"]
    assert first_manifest["monetary_reference_status"] == "unresolved"
