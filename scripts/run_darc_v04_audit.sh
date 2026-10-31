#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${DARC_V04_AUDIT_CONFIG:-configs/darc_v04_audit.yaml}"
OUTPUT="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["output"]["root"])' "$CONFIG")"

echo "[1/3] Lightweight DARC v0.4 audit contracts"
python -m pytest -q tests/test_darc_v04_audit.py tests/test_darc_v04_contracts.py

echo "[2/3] Multi-seed validation-only training and fair replay"
echo "Frozen test access: forbidden"
PYTHONUNBUFFERED=1 python scripts/run_darc_v04_audit.py --config "$CONFIG"

echo "[3/3] Verify reproducible artifacts"
test -s "$OUTPUT/report/darc_v04_audit_report.md"
test -s "$OUTPUT/tables/aggregate_results.csv"
test -s "$OUTPUT/tables/paired_comparisons.csv"
test -s "$OUTPUT/figures/accuracy_cost_frontier.png"
test -s "$OUTPUT/manifest.json"

echo "DARC v0.4 audit completed."
echo "Report: $PROJECT_ROOT/$OUTPUT/report/darc_v04_audit_report.md"
echo "Tables: $PROJECT_ROOT/$OUTPUT/tables/"
echo "Figures: $PROJECT_ROOT/$OUTPUT/figures/"
echo "Models: $PROJECT_ROOT/$OUTPUT/models/"

