from __future__ import annotations

import io
import time

import pytest

from eph_income.observability import (
    grid_configuration_count,
    heartbeat,
    log_model_fit_start,
)


def test_heartbeat_starts_and_stops_cleanly() -> None:
    stream = io.StringIO()

    with heartbeat("unit-test-model", interval_seconds=0.01, stream=stream):
        time.sleep(0.035)

    first_output = stream.getvalue()
    assert "unit-test-model" in first_output
    assert "still running" in first_output

    time.sleep(0.03)
    assert stream.getvalue() == first_output


def test_heartbeat_stops_when_wrapped_code_raises() -> None:
    stream = io.StringIO()

    with pytest.raises(RuntimeError, match="boom"):
        with heartbeat("exception-model", interval_seconds=0.01, stream=stream):
            time.sleep(0.025)
            raise RuntimeError("boom")

    first_output = stream.getvalue()
    time.sleep(0.03)
    assert stream.getvalue() == first_output


def test_phase_start_log_reports_expected_fit_counts() -> None:
    stream = io.StringIO()
    log_model_fit_start(
        run_id="run-1",
        model_name="Ridge",
        grid_configurations=3,
        cv_folds=5,
        train_rows=70,
        feature_count=12,
        stream=stream,
    )

    output = stream.getvalue()
    assert "run_id=run-1" in output
    assert "model=Ridge" in output
    assert "grid_configurations=3" in output
    assert "cv_folds=5" in output
    assert "total_expected_fits=15" in output
    assert "train_rows=70" in output
    assert "feature_count=12" in output


def test_grid_configuration_count_matches_sklearn_parameter_grid() -> None:
    assert grid_configuration_count({"a": [1, 2], "b": [True, False, None]}) == 6
