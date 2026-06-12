.PHONY: help install validate lint test build-dataset run-experiment run-debug run-baseline run-regularization-sweep run-hgb-debug run-hgb-sweep run-hgb-lr-iter-sweep run-hgb-leaf-capacity-sweep run-hgb-min-leaf-sweep run-hgb-l2-sweep run-hgb-quick-benchmark run-hgb-quick-with-geo-ranks run-hgb-quick-clean-geo run-hgb-quick-no-geo run-hgb-quick-shuffled-geo-ranks run-ols-demo run-ols-demo-educ run-ols-demo-educ-labor run-ols-core run-ols-core-no-pyramid run-ols-core-with-pyramid run-ols-core-year-fe run-ols-core-quarter-fe run-ols-core-year-plus-quarter-fe run-ols-core-region-fe run-ols-core-aglo-fe run-ols-core-aglo-plus-time-fe run-ols-core-geo-ranks run-ols-core-shuffled-geo-ranks run-ols-core-aglo-fe-no-pyramid report build-diagnostics all


PYTHON ?= python3
EXPERIMENT_CONFIG ?= configs/experiment_baseline.yaml
FEATURE_CONTRACT ?= configs/feature_contract.yaml
SPLIT ?= test

help:
	@echo "Available targets:"
	@echo "  make install                     Install package with dev dependencies"
	@echo "  make validate                    Validate upstream inputs without writing dataset outputs"
	@echo "  make lint                        Run ruff over source, tests, and scripts"
	@echo "  make test                        Run pytest"
	@echo "  make build-dataset               Build processed modeling dataset and split assignments"
	@echo "  make run-experiment              Run any experiment config: EXPERIMENT_CONFIG=... [ALLOW_FULL_RUN=1]"
	@echo "  make run-debug                   Run the small linear/Ridge debug experiment"
	@echo "  make run-baseline                Run the guarded full baseline experiment"
	@echo "  make run-regularization-sweep    Run Ridge/Lasso regularization sweep diagnostics"
	@echo "  make run-hgb-debug               Run Ridge reference + small HGB debug experiment"
	@echo "  make run-hgb-sweep               Run guarded HGB-only medium sweep"
	@echo "  make run-hgb-lr-iter-sweep       Run targeted HGB learning-rate/max-iter sweep"
	@echo "  make run-hgb-leaf-capacity-sweep Run targeted HGB leaf-capacity sweep"
	@echo "  make run-hgb-min-leaf-sweep      Run targeted HGB minimum-leaf-size sweep"
	@echo "  make run-hgb-l2-sweep            Run targeted HGB L2-regularization sweep"
	@echo "  make run-hgb-quick-benchmark     Run fast stable HGB benchmark and training-frame sample"
	@echo "  make run-hgb-quick-with-geo-ranks Run HGB geography smoke with target-derived geo ranks"
	@echo "  make run-hgb-quick-clean-geo      Run HGB geography smoke dropping geo ranks"
	@echo "  make run-hgb-quick-no-geo         Run HGB geography smoke dropping all geography columns"
	@echo "  make run-hgb-quick-shuffled-geo-ranks Run HGB geography smoke with shuffled geo ranks"
	@echo "  make run-ols-demo                         Run OLS demographic feature block"
	@echo "  make run-ols-demo-educ                    Run OLS demographic + education blocks"
	@echo "  make run-ols-demo-educ-labor              Run OLS demographic + education + labor blocks"
	@echo "  make run-ols-core                         Run clean OLS core benchmark (no household pyramid)"
	@echo "  make run-ols-core-no-pyramid              Legacy alias/config for clean OLS core without household pyramid"
	@echo "  make run-ols-core-with-pyramid            Run OLS core + household pyramid sensitivity"
	@echo "  make run-ols-core-year-fe                 Run OLS core with additive ANO4 FE"
	@echo "  make run-ols-core-quarter-fe              Run OLS core with additive TRIMESTRE FE"
	@echo "  make run-ols-core-year-plus-quarter-fe    Run OLS core with additive ANO4 + TRIMESTRE FE"
	@echo "  make run-ols-core-region-fe               Run OLS core with additive Region FE"
	@echo "  make run-ols-core-aglo-fe                 Run OLS core with additive AGLOMERADO FE"
	@echo "  make run-ols-core-aglo-plus-time-fe       Run OLS core with additive AGLOMERADO + ANO4 + TRIMESTRE FE"
	@echo "  make run-ols-core-geo-ranks               Run OLS core with target-derived geo ranks"
	@echo "  make run-ols-core-shuffled-geo-ranks      Run OLS core with shuffled geo ranks placebo"
	@echo "  make run-ols-core-aglo-fe-no-pyramid      Legacy alias/config for OLS aglo FE without household pyramid"
	@echo "  make thesis-ols-feature-blocks   Run OLS feature-family accumulation suite"
	@echo "  make thesis-ols-temporal-fe      Run OLS temporal fixed-effect suite"
	@echo "  make thesis-ols-geographic-fe    Run OLS geographic fixed-effect suite"
	@echo "  make thesis-ols-sensitivity      Run OLS sensitivity/placebo suite"
	@echo "  make thesis-ols                  Run full OLS experiment suite"
	@echo "  make report                      Build thesis-ready report artifacts from saved outputs"
	@echo "  make build-diagnostics           Build diagnostics for RUN_DIR=reports/runs/<run_id> [SPLIT=test]"
	@echo "  make all                         Build dataset, run baseline, and build report"

install:
	pip install -e ".[dev]"

validate:
	$(PYTHON) scripts/01_build_dataset.py --check-only

lint:
	ruff check src tests scripts

test:
	pytest -q

build-dataset:
	$(PYTHON) scripts/01_build_dataset.py

run-experiment:
	$(PYTHON) scripts/02_run_baseline_experiment.py \
		--config "$(EXPERIMENT_CONFIG)" \
		--feature-contract "$(FEATURE_CONTRACT)" \
		$(if $(ALLOW_FULL_RUN),--allow-full-run,)

run-debug:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_debug.yaml

run-baseline:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_baseline.yaml ALLOW_FULL_RUN=1

run-regularization-sweep:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_regularization_sweep.yaml ALLOW_FULL_RUN=1

run-hgb-debug:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_hgb_debug.yaml

run-hgb-sweep:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_hgb_sweep.yaml ALLOW_FULL_RUN=1



run-hgb-lr-iter-sweep:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_hgb_lr_iter_sweep.yaml ALLOW_FULL_RUN=1

run-hgb-leaf-capacity-sweep:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_hgb_leaf_capacity_sweep.yaml ALLOW_FULL_RUN=1

run-hgb-min-leaf-sweep:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_hgb_min_leaf_sweep.yaml ALLOW_FULL_RUN=1

run-hgb-l2-sweep:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_hgb_l2_sweep.yaml ALLOW_FULL_RUN=1



run-hgb-quick-benchmark:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_hgb_quick_benchmark.yaml ALLOW_FULL_RUN=1



run-hgb-quick-with-geo-ranks:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_geo_probe_v2_with_geo_ranks.yaml ALLOW_FULL_RUN=1

run-hgb-quick-clean-geo:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_geo_probe_v2_clean_geo.yaml ALLOW_FULL_RUN=1

run-hgb-quick-no-geo:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_geo_probe_v2_no_geo.yaml ALLOW_FULL_RUN=1

run-hgb-quick-shuffled-geo-ranks:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_geo_probe_v2_shuffled_geo_ranks.yaml ALLOW_FULL_RUN=1



run-ols-demo:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_demo.yaml ALLOW_FULL_RUN=1

run-ols-demo-educ:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_demo_educ.yaml ALLOW_FULL_RUN=1

run-ols-demo-educ-labor:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_demo_educ_labor.yaml ALLOW_FULL_RUN=1

run-ols-core:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_core.yaml ALLOW_FULL_RUN=1

run-ols-core-no-pyramid:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_core_no_pyramid.yaml ALLOW_FULL_RUN=1

run-ols-core-with-pyramid:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_core_with_pyramid.yaml ALLOW_FULL_RUN=1

run-ols-core-year-fe:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_core_year_fe.yaml ALLOW_FULL_RUN=1

run-ols-core-quarter-fe:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_core_quarter_fe.yaml ALLOW_FULL_RUN=1

run-ols-core-year-plus-quarter-fe:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_core_year_plus_quarter_fe.yaml ALLOW_FULL_RUN=1

run-ols-core-region-fe:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_core_region_fe.yaml ALLOW_FULL_RUN=1

run-ols-core-aglo-fe:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_core_aglo_fe.yaml ALLOW_FULL_RUN=1

run-ols-core-aglo-plus-time-fe:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_core_aglo_plus_time_fe.yaml ALLOW_FULL_RUN=1

run-ols-core-geo-ranks:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_core_geo_ranks.yaml ALLOW_FULL_RUN=1

run-ols-core-shuffled-geo-ranks:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_core_shuffled_geo_ranks.yaml ALLOW_FULL_RUN=1

run-ols-core-aglo-fe-no-pyramid:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_core_aglo_fe_no_pyramid.yaml ALLOW_FULL_RUN=1

run-ols-core-aglo-fe-no-pyramid-no-sanitation:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_ols_core_aglo_fe_no_pyramid_no_sanitation.yaml ALLOW_FULL_RUN=1


report:
	$(PYTHON) scripts/03_build_report_artifacts.py

build-diagnostics:
	@test -n "$(RUN_DIR)" || (echo "Usage: make build-diagnostics RUN_DIR=reports/runs/<run_id> [SPLIT=test]"; exit 2)
	$(PYTHON) scripts/04_build_diagnostics.py --run-dir "$(RUN_DIR)" --split "$(SPLIT)"

all: build-dataset run-baseline report

plan-experiment:
	$(PYTHON) scripts/10_plan_experiment.py --config "$(EXPERIMENT_CONFIG)"

thesis-plan:
	$(PYTHON) scripts/10_plan_experiment.py --config configs/experiment_baseline.yaml
	$(PYTHON) scripts/10_plan_experiment.py --config configs/experiment_hgb_quick_clean_geo_v1.yaml
	$(PYTHON) scripts/10_plan_experiment.py --config configs/experiment_hgb_quick_benchmark.yaml


.PHONY: thesis-smoke thesis-core thesis-regularization thesis-hgb-sweeps thesis-geo-probe thesis-ols-feature-blocks thesis-ols-temporal-fe thesis-ols-geographic-fe thesis-ols-sensitivity thesis-ols thesis-diagnostics thesis-evidence thesis-check thesis-all

THESIS_SPLITS ?= test validation
THESIS_PROFILE ?= thesis_core

thesis-smoke: build-dataset
	$(MAKE) run-debug
	$(MAKE) run-hgb-debug
	$(PYTHON) scripts/07_build_thesis_diagnostics.py --family smoke --splits "$(THESIS_SPLITS)"

thesis-core: build-dataset
	# $(MAKE) run-baseline
	$(MAKE) run-hgb-quick-clean-geo
	$(MAKE) run-hgb-quick-benchmark

thesis-regularization: build-dataset
	$(MAKE) run-regularization-sweep

thesis-hgb-sweeps: build-dataset
	$(MAKE) run-hgb-leaf-capacity-sweep
	$(MAKE) run-hgb-min-leaf-sweep
	$(MAKE) run-hgb-lr-iter-sweep
	$(MAKE) run-hgb-l2-sweep

thesis-geo-probe: build-dataset
	$(MAKE) run-hgb-quick-clean-geo
	$(MAKE) run-hgb-quick-no-geo
	$(MAKE) run-hgb-quick-with-geo-ranks
	$(MAKE) run-hgb-quick-shuffled-geo-ranks

thesis-ols-feature-blocks: build-dataset
	$(MAKE) run-ols-demo
	$(MAKE) run-ols-demo-educ
	$(MAKE) run-ols-demo-educ-labor
	$(MAKE) run-ols-core
	$(MAKE) run-ols-core-with-pyramid

thesis-ols-temporal-fe: build-dataset
	$(MAKE) run-ols-core-year-fe
	$(MAKE) run-ols-core-quarter-fe
	$(MAKE) run-ols-core-year-plus-quarter-fe

thesis-ols-geographic-fe: build-dataset
	$(MAKE) run-ols-core-region-fe
	$(MAKE) run-ols-core-aglo-fe
	$(MAKE) run-ols-core-aglo-plus-time-fe

thesis-ols-sensitivity: build-dataset
	$(MAKE) run-ols-core-geo-ranks
	$(MAKE) run-ols-core-shuffled-geo-ranks
	$(MAKE) run-ols-core-aglo-fe-no-pyramid

thesis-ols: thesis-ols-feature-blocks thesis-ols-temporal-fe thesis-ols-geographic-fe thesis-ols-sensitivity

thesis-core-fast: build-dataset
	$(MAKE) run-hgb-quick-clean-geo
	$(MAKE) run-hgb-quick-benchmark

thesis-core-canonical: build-dataset
	$(MAKE) run-baseline
	$(MAKE) run-hgb-quick-clean-geo
	$(MAKE) run-hgb-quick-benchmark

thesis-all: lint test build-dataset thesis-core-fast thesis-diagnostics thesis-evidence thesis-check

thesis-freeze: lint test build-dataset thesis-core-canonical thesis-diagnostics thesis-evidence thesis-check

thesis-diagnostics:
	$(PYTHON) scripts/07_build_thesis_diagnostics.py --priority "required" --splits "test validation"
	
thesis-evidence:
	$(PYTHON) scripts/08_collect_thesis_artifacts.py --profile "$(THESIS_PROFILE)"

thesis-check:
	$(PYTHON) scripts/09_check_thesis_evidence.py --profile "$(THESIS_PROFILE)"

.PHONY: thesis-support thesis-full

thesis-support: thesis-regularization thesis-hgb-sweeps thesis-geo-probe thesis-ols

