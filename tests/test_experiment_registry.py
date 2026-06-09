# from __future__ import annotations

# from pathlib import Path
# from typing import Any

# import yaml

# from eph_income.config import load_experiment_config
# from eph_income.diagnostics_registry import DIAGNOSTICS_PROFILES, normalize_diagnostics_config

# EXPERIMENT_CONFIGS = sorted(
#     path
#     for path in Path("configs").glob("experiment*.yaml")
#     if path.name != "experiment_registry.yaml"
# )
# REGISTRY_PATH = Path("configs/experiment_registry.yaml")


# def _registry_entries() -> list[dict[str, Any]]:
#     data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
#     return data["experiments"]


# def _config(path: Path) -> dict[str, Any]:
#     return load_experiment_config(str(path))


# def test_every_active_experiment_config_declares_known_diagnostics_profile() -> None:
#     assert EXPERIMENT_CONFIGS
#     for path in EXPERIMENT_CONFIGS:
#         config = _config(path)
#         diagnostics = config.get("diagnostics")
#         assert isinstance(diagnostics, dict), path
#         assert diagnostics.get("enabled") is True, path
#         assert diagnostics.get("profile") in DIAGNOSTICS_PROFILES, path
#         normalized = normalize_diagnostics_config(config)
#         assert normalized["profile"] == diagnostics["profile"]


# def test_experiment_registry_covers_configs_and_has_no_duplicate_ids() -> None:
#     entries = _registry_entries()
#     registry_files = {entry["file"] for entry in entries}
#     config_files = {str(path) for path in EXPERIMENT_CONFIGS}
#     assert registry_files == config_files

#     ids = [entry["experiment_id"] for entry in entries]
#     assert len(ids) == len(set(ids))

#     config_ids = {_config(path)["experiment"]["id"] for path in EXPERIMENT_CONFIGS}
#     assert set(ids) == config_ids


# def test_registry_profiles_match_config_profiles() -> None:
#     entries = _registry_entries()
#     for entry in entries:
#         config = load_experiment_config(entry["file"])
#         assert entry["diagnostics_profile"] == config["diagnostics"]["profile"]
#         assert entry["family"] in {
#             "smoke",
#             "thesis_core",
#             "hgb_tuning",
#             "regularization",
#             "geo_probe",
#             "fixed_effects",
#         }
#         assert entry["role"] in {"smoke", "benchmark", "sweep", "robustness", "interpretability", "optional"}
#         assert entry["priority"] in {"required", "useful", "optional", "archived"}


# def test_profile_assignments_match_experiment_families() -> None:
#     expected_profiles = {
#         "configs/experiment_debug.yaml": "smoke_minimal",
#         "configs/experiment_hgb_debug.yaml": "smoke_minimal",
#         "configs/experiment_baseline.yaml": "thesis_core",
#         "configs/experiment_hgb_quick_benchmark.yaml": "thesis_core",
#         "configs/experiment_hgb_quick_clean_geo_v1.yaml": "thesis_core",
#         "configs/experiment_regularization_sweep.yaml": "regularization_interpretation",
#         "configs/experiment_hgb_pilot.yaml": "hgb_capacity_sweep",
#         "configs/experiment_hgb_sweep.yaml": "hgb_capacity_sweep",
#         "configs/experiment_hgb_leaf_capacity_sweep.yaml": "hgb_capacity_sweep",
#         "configs/experiment_hgb_min_leaf_sweep.yaml": "hgb_capacity_sweep",
#         "configs/experiment_hgb_lr_iter_sweep.yaml": "hgb_capacity_sweep",
#         "configs/experiment_hgb_l2_sweep.yaml": "hgb_capacity_sweep",
#         "configs/experiment_hgb_quick_no_geo_v1.yaml": "geo_leakage_probe",
#         "configs/experiment_hgb_quick_with_geo_ranks_v1.yaml": "geo_leakage_probe",
#         "configs/experiment_hgb_quick_shuffled_geo_ranks_v1.yaml": "geo_leakage_probe",
#         "configs/experiment_linear_region_fe_v1.yaml": "linear_fe_interpretation",
#         "configs/experiment_linear_aglo_fe_v1.yaml": "linear_fe_interpretation",
#         "configs/experiment_linear_year_fe_v1.yaml": "linear_fe_interpretation",
#         "configs/experiment_linear_quarter_fe_v1.yaml": "linear_fe_interpretation",
#         "configs/experiment_linear_aglo_year_fe_v1.yaml": "linear_fe_interpretation",
#     }
#     assert {str(path) for path in EXPERIMENT_CONFIGS} == set(expected_profiles)
#     for file, profile in expected_profiles.items():
#         assert load_experiment_config(file)["diagnostics"]["profile"] == profile


# def test_hgb_capacity_sweeps_have_governed_sweep_metadata() -> None:
#     expected = {
#         "configs/experiment_hgb_leaf_capacity_sweep.yaml": ("hgb_single_param", "max_leaf_nodes", None),
#         "configs/experiment_hgb_min_leaf_sweep.yaml": ("hgb_single_param", "min_samples_leaf", None),
#         "configs/experiment_hgb_l2_sweep.yaml": ("hgb_single_param", "l2_regularization", None),
#         "configs/experiment_hgb_lr_iter_sweep.yaml": ("hgb_lr_iter", "max_iter", "learning_rate"),
#         "configs/experiment_hgb_sweep.yaml": ("hgb_grid", None, None),
#         "configs/experiment_hgb_pilot.yaml": ("hgb_pilot", None, None),
#     }
#     for file, (sweep_type, primary, group) in expected.items():
#         diagnostics = load_experiment_config(file)["diagnostics"]
#         assert diagnostics["profile"] == "hgb_capacity_sweep"
#         assert diagnostics["sweeps"] == {
#             "enabled": True,
#             "sweep_type": sweep_type,
#             "primary_param": primary,
#             "group_param": group,
#         }
#         if sweep_type == "hgb_single_param":
#             assert diagnostics["sweeps"]["primary_param"] is not None


# def test_geo_probe_configs_have_comparison_metadata() -> None:
#     expected_variants = {
#         "configs/experiment_hgb_quick_clean_geo_v1.yaml": "clean_geo",
#         "configs/experiment_hgb_quick_no_geo_v1.yaml": "no_geo",
#         "configs/experiment_hgb_quick_with_geo_ranks_v1.yaml": "with_geo_ranks",
#         "configs/experiment_hgb_quick_shuffled_geo_ranks_v1.yaml": "shuffled_geo_ranks",
#     }
#     for file, variant in expected_variants.items():
#         diagnostics = load_experiment_config(file)["diagnostics"]
#         comparison = diagnostics["comparison"]
#         assert comparison["family"] == "geo_leakage_probe"
#         assert comparison["variant"] == variant
#         assert comparison["anchor_variant"] == "clean_geo"


# def test_old_style_diagnostics_keys_normalize_to_equivalent_new_style() -> None:
#     legacy = {
#         "diagnostics": {
#             "sweep_type": "hgb_single_param",
#             "primary_param": "max_leaf_nodes",
#         }
#     }
#     new = {
#         "diagnostics": {
#             "enabled": True,
#             "profile": "hgb_capacity_sweep",
#             "sweeps": {
#                 "enabled": True,
#                 "sweep_type": "hgb_single_param",
#                 "primary_param": "max_leaf_nodes",
#                 "group_param": None,
#             },
#         }
#     }

#     legacy_normalized = normalize_diagnostics_config(legacy)
#     new_normalized = normalize_diagnostics_config(new)

#     assert legacy_normalized["sweeps"] == new_normalized["sweeps"]
