#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${DARC_V04_AUDIT_CONFIG:-configs/darc_v04_audit.yaml}"
BUDGETS="${DARC_MATCHED_BUDGETS:-100000,150000,200000,250000}"

python -m pytest -q tests/test_darc_v04_matched_budget.py
PYTHONUNBUFFERED=1 python scripts/evaluate_darc_v04_matched_budget.py \
  --config "$CONFIG" \
  --budgets "$BUDGETS"

echo "Report: $PROJECT_ROOT/results/darc_v04_audit_31102026/matched_budget/report/matched_budget_report.md"

