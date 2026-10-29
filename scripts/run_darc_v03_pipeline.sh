#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[1/3] DARC v0.3 contract tests"
python -m pytest -q tests/test_darc_v03_contracts.py tests/test_darc_v02_contracts.py tests/test_darc_contracts.py

echo "[2/3] Train + validation-only evaluation"
python scripts/run_darc_v03.py --config configs/darc_v03.yaml

echo "[3/3] Verify artifacts"
test -s results/darc_v03_30102026/report/darc_v03_validation_report.md
test -s results/darc_v03_30102026/tables/validation_policy_results.csv
test -s results/darc_v03_30102026/figures/darc_v03_accuracy_cost.png

echo "DARC v0.3 pipeline completed."
echo "Report: $PROJECT_ROOT/results/darc_v03_30102026/report/darc_v03_validation_report.md"
