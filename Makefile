.PHONY: install validate test build-dataset run-debug run-baseline report all

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

all: build-dataset run-baseline report