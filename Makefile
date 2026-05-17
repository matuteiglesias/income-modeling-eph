.PHONY: install validate test build-dataset run-baseline report all

install:
	pip install -e ".[dev]"

validate:
	python scripts/01_build_dataset.py --check-only

test:
	pytest -q

build-dataset:
	python scripts/01_build_dataset.py

run-baseline:
	python scripts/02_run_baseline_experiment.py

report:
	python scripts/03_build_report_artifacts.py

all: build-dataset run-baseline report