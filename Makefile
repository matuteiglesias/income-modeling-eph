.PHONY: help install validate lint test build-dataset run-experiment run-debug run-baseline run-regularization-sweep run-hgb-debug run-hgb-sweep run-hgb-lr-iter-sweep run-hgb-leaf-capacity-sweep run-hgb-min-leaf-sweep run-hgb-l2-sweep run-hgb-quick-benchmark run-hgb-quick-with-geo-ranks run-hgb-quick-clean-geo run-hgb-quick-no-geo run-hgb-quick-shuffled-geo-ranks run-linear-region-fe run-linear-aglo-fe run-linear-year-fe run-linear-quarter-fe run-linear-aglo-year-fe report build-diagnostics all


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
	@echo "  make run-linear-region-fe       Run Linear/Ridge Region fixed-effect smoke"
	@echo "  make run-linear-aglo-fe         Run Linear/Ridge AGLOMERADO fixed-effect smoke"
	@echo "  make run-linear-year-fe         Run Linear/Ridge ANO4 fixed-effect smoke"
	@echo "  make run-linear-quarter-fe      Run Linear/Ridge TRIMESTRE fixed-effect smoke"
	@echo "  make run-linear-aglo-year-fe    Run Linear/Ridge AGLOMERADO x ANO4 fixed-effect smoke"
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
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_hgb_quick_with_geo_ranks_v1.yaml ALLOW_FULL_RUN=1

run-hgb-quick-clean-geo:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_hgb_quick_clean_geo_v1.yaml ALLOW_FULL_RUN=1

run-hgb-quick-no-geo:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_hgb_quick_no_geo_v1.yaml ALLOW_FULL_RUN=1

run-hgb-quick-shuffled-geo-ranks:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_hgb_quick_shuffled_geo_ranks_v1.yaml ALLOW_FULL_RUN=1

run-linear-region-fe:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_linear_region_fe_v1.yaml ALLOW_FULL_RUN=1

run-linear-aglo-fe:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_linear_aglo_fe_v1.yaml ALLOW_FULL_RUN=1

run-linear-year-fe:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_linear_year_fe_v1.yaml ALLOW_FULL_RUN=1

run-linear-quarter-fe:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_linear_quarter_fe_v1.yaml ALLOW_FULL_RUN=1

run-linear-aglo-year-fe:
	$(MAKE) run-experiment EXPERIMENT_CONFIG=configs/experiment_linear_aglo_year_fe_v1.yaml ALLOW_FULL_RUN=1

report:
	$(PYTHON) scripts/03_build_report_artifacts.py

build-diagnostics:
	@test -n "$(RUN_DIR)" || (echo "Usage: make build-diagnostics RUN_DIR=reports/runs/<run_id> [SPLIT=test]"; exit 2)
	$(PYTHON) scripts/04_build_diagnostics.py --run-dir "$(RUN_DIR)" --split "$(SPLIT)"

all: build-dataset run-baseline report
