#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?Usage: scripts/dev/inspect_run.sh reports/runs/<run_id>}"

echo "Tree"
echo "===="
tree --du -h -D -L 4 "$RUN_DIR"

echo
echo "CSV / JSON / YAML artifacts"
echo "=========================="
find "$RUN_DIR" -type f \( -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.md' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %9s  %p\n' \
  | sort

echo
echo "PNG artifacts"
echo "============="
find "$RUN_DIR" -type f -name '*.png' \
  -printf '%TY-%Tm-%Td %TH:%TM %9s  %p\n' \
  | sort
