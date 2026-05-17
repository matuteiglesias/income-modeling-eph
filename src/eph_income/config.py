"""Configuration loading helpers for EPH income experiments.

This module intentionally contains only lightweight YAML/path helpers. It must not
load data or trigger experiment execution at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_CONTRACT_PATH = PROJECT_ROOT / "configs" / "feature_contract.yaml"
DEFAULT_EXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_baseline.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path`` using PyYAML's safe loader.

    Parameters
    ----------
    path:
        YAML file path, either absolute or relative to the current working directory.

    Returns
    -------
    dict[str, Any]
        Parsed YAML mapping. Empty YAML files return an empty dictionary.

    Raises
    ------
    TypeError
        If the YAML document is not a mapping.
    """

    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {path}, got {type(data).__name__}.")
    return data


def load_feature_contract(path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH) -> dict[str, Any]:
    """Load the feature contract YAML."""

    return load_yaml(path)


def load_experiment_config(path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH) -> dict[str, Any]:
    """Load the baseline experiment YAML."""

    return load_yaml(path)
