#!/usr/bin/env bash
set -euo pipefail

N="${1:-40}"

find reports -type f -name '*.png' \
  -printf '%TY-%Tm-%Td %TH:%TM %9s  %p\n' \
  | sort -r \
  | head -n "$N"
