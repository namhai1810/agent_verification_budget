#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"
export PYTHONUNBUFFERED=1
CONFIG="configs/joint_value_cost.yaml"
TOTAL_STEPS=8
PIPELINE_STARTED="$(date +%s)"

format_seconds() {
  local total="$1"
  printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

run_step() {
  local number="$1"
  local label="$2"
  shift 2
  local started
  started="$(date +%s)"
  printf '\n[%d/%d] %s\n' "$number" "$TOTAL_STEPS" "$label"
  "$@"
  local elapsed
  elapsed="$(($(date +%s) - started))"
  printf '[%d/%d] completed in %s\n'     "$number" "$TOTAL_STEPS" "$(format_seconds "$elapsed")"
}

run_step 1 "Validate and prepare paired data"   python scripts/joint_value_cost.py --config "$CONFIG" prepare
run_step 2 "Train HumanEval-only Joint + Separate models"   python scripts/joint_value_cost.py --config "$CONFIG" train --mode humaneval --resume
run_step 3 "Train LiveCodeBench-only Joint + Separate models"   python scripts/joint_value_cost.py --config "$CONFIG" train --mode livecodebench --resume
run_step 4 "Train Combined Joint + Separate models"   python scripts/joint_value_cost.py --config "$CONFIG" train --mode combined --resume
run_step 5 "Run in-domain and cross-domain evaluation"   python scripts/joint_value_cost.py --config "$CONFIG" evaluate
run_step 6 "Run token-budgeted offline replay"   python scripts/joint_value_cost.py --config "$CONFIG" budget-eval
run_step 7 "Generate PNG/PDF visualizations"   python scripts/joint_value_cost.py --config "$CONFIG" visualize
run_step 8 "Generate final Markdown report"   python scripts/joint_value_cost.py --config "$CONFIG" report

PIPELINE_ELAPSED="$(($(date +%s) - PIPELINE_STARTED))"
printf '\nAll steps completed in %s\n' "$(format_seconds "$PIPELINE_ELAPSED")"
printf 'Report: %s/results/joint_value_cost/report/joint_value_cost_report.md\n' "$REPO_ROOT"
