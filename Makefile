.PHONY: install validate test build-dataset run-debug run-baseline report build-diagnostics all

install:
	pip install -e ".[dev]"

validate:
	python scripts/01_build_dataset.py --check-only

test:
	pytest -q

build-dataset:
	python scripts/01_build_dataset.py

run-debug:
	python scripts/02_run_baseline_experiment.py --config configs/experiment_debug.yaml

run-baseline:
	python scripts/02_run_baseline_experiment.py --allow-full-run

report:
	python scripts/03_build_report_artifacts.py

build-diagnostics:
	@test -n "$(RUN_DIR)" || (echo "Usage: make build-diagnostics RUN_DIR=reports/runs/<run_id> [SPLIT=test]"; exit 2)
	python scripts/04_build_diagnostics.py --run-dir "$(RUN_DIR)" --split "$(or $(SPLIT),test)"

all: build-dataset run-baseline report