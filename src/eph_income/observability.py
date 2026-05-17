"""Lightweight terminal observability for long-running experiment phases."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from sklearn.model_selection import ParameterGrid


def _utc_timestamp() -> str:
    """Return a compact UTC timestamp for terminal logs."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds concisely for human-readable logs."""

    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{remainder:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes:02d}m{remainder:04.1f}s"


@contextmanager
def heartbeat(
    label: str,
    interval_seconds: float = 10,
    *,
    stream: TextIO | None = None,
):
    """Emit a periodic heartbeat while a blocking phase is running.

    The heartbeat thread is always stopped when the context exits, including
    when the wrapped operation raises an exception.
    """

    if interval_seconds <= 0:
        yield
        return

    output = stream or sys.stdout
    stop_event = threading.Event()
    start = time.perf_counter()

    def _run() -> None:
        while not stop_event.wait(interval_seconds):
            elapsed = _format_elapsed(time.perf_counter() - start)
            print(
                f"[{_utc_timestamp()}] {label} elapsed={elapsed} still running",
                file=output,
                flush=True,
            )

    thread = threading.Thread(target=_run, name=f"heartbeat:{label}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=max(1.0, min(float(interval_seconds), 5.0)))


def grid_configuration_count(grid: Mapping[str, Iterable[Any]] | list[Mapping[str, Iterable[Any]]]) -> int:
    """Return the number of concrete GridSearchCV configurations in a parameter grid."""

    return len(ParameterGrid(grid))


def log_model_fit_start(
    *,
    run_id: str,
    model_name: str,
    grid_configurations: int,
    cv_folds: int,
    train_rows: int,
    feature_count: int,
    stream: TextIO | None = None,
) -> None:
    """Print a concise phase-level log line before model fitting starts."""

    output = stream or sys.stdout
    total_expected_fits = int(grid_configurations) * int(cv_folds)
    print(
        " ".join(
            [
                f"[{_utc_timestamp()}] fit-start",
                f"run_id={run_id}",
                f"model={model_name}",
                f"grid_configurations={grid_configurations}",
                f"cv_folds={cv_folds}",
                f"total_expected_fits={total_expected_fits}",
                f"train_rows={train_rows}",
                f"feature_count={feature_count}",
            ]
        ),
        file=output,
        flush=True,
    )


def log_model_fit_end(
    *,
    run_id: str,
    model_name: str,
    elapsed_seconds: float,
    best_params: Mapping[str, Any] | None = None,
    best_cv_score: float | None = None,
    artifact_paths: Iterable[str | Path] = (),
    stream: TextIO | None = None,
) -> None:
    """Print a concise phase-level log line after model fitting artifacts are written."""

    output = stream or sys.stdout
    score_text = "NA" if best_cv_score is None else f"{best_cv_score:.6g}"
    params_text = "{}" if best_params is None else str(dict(best_params))
    artifacts_text = ",".join(str(path) for path in artifact_paths) or "none"
    print(
        " ".join(
            [
                f"[{_utc_timestamp()}] fit-end",
                f"run_id={run_id}",
                f"model={model_name}",
                f"elapsed={_format_elapsed(elapsed_seconds)}",
                f"best_cv_score={score_text}",
                f"best_params={params_text}",
                f"artifacts={artifacts_text}",
            ]
        ),
        file=output,
        flush=True,
    )
