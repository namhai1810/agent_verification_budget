#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[1/3] Verify locked DARC v0.3 selection artifacts"
test -s results/darc_v03_30102026/selection_manifest.json
test -s results/darc_v03_30102026/models/darc_v03_selected.pt
test -s results/darc_v03_30102026/selection_objects/darc_v03_selected_calibrator.joblib
test -s results/darc_v03_30102026/selection_objects/darc_v03_selected_continuation_prior.joblib

echo "[2/3] Run locked exploratory frozen-test replay (no training or LLM inference)"
python scripts/evaluate_darc_v03_test.py --config configs/darc_v03.yaml

echo "[3/3] Verify frozen-test report artifacts"
test -s results/darc_v03_30102026/test_evaluation/report/darc_v03_validation_frozen_test_report.md
test -s results/darc_v03_30102026/test_evaluation/tables/paired_policy_comparison.csv
test -s results/darc_v03_30102026/test_evaluation/figures/darc_v03_validation_test_summary.png

echo "Locked DARC v0.3 frozen-test evaluation completed."
echo "Report: $PROJECT_ROOT/results/darc_v03_30102026/test_evaluation/report/darc_v03_validation_frozen_test_report.md"
