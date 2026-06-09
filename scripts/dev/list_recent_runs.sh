#!/usr/bin/env bash
set -euo pipefail

N="${1:-12}"

echo "Recent run directories"
echo "======================"
find reports/runs -mindepth 1 -maxdepth 1 -type d \
  -printf '%TY-%Tm-%Td %TH:%TM  %p\n' \
  | sort -r \
  | head -n "$N"
