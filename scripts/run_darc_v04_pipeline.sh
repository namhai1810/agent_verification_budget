#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[1/3] DARC v0.4 contract tests"
python -m pytest -q tests/test_darc_v04_contracts.py tests/test_darc_v03_contracts.py

echo "[2/3] Train, validation selection, and locked frozen-test benchmark"
python scripts/run_darc_v04.py --config configs/darc_v04.yaml

echo "[3/3] Verify report artifacts"
test -s results/darc_v04_31102026/report/darc_v04_report.md
test -s results/darc_v04_31102026/tables/policy_results.csv
test -s results/darc_v04_31102026/figures/darc_v04_main_results.png

echo "DARC v0.4 pipeline completed."
echo "Report: $PROJECT_ROOT/results/darc_v04_31102026/report/darc_v04_report.md"

