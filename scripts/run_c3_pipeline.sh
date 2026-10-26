#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"
export PYTHONUNBUFFERED=1
CONFIG="configs/c3_sequential_controller.yaml"
TOTAL_STEPS=7
PIPELINE_STARTED="$(date +%s)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/results/c3_sequential_controller/logs"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/pipeline_${RUN_STAMP}.log") 2>&1

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
  printf '[%d/%d] completed in %s\n' "$number" "$TOTAL_STEPS" "$(format_seconds "$elapsed")"
}

run_step 1 "Validate frozen data and pretrained Joint models" python scripts/c3_sequential_controller.py --config "$CONFIG" prepare
run_step 2 "Train HumanEval-only C3 continuation and ablation" python scripts/c3_sequential_controller.py --config "$CONFIG" train --mode humaneval --resume
run_step 3 "Train LiveCodeBench-only C3 continuation and ablation" python scripts/c3_sequential_controller.py --config "$CONFIG" train --mode livecodebench --resume
run_step 4 "Train Combined C3 continuation and ablation" python scripts/c3_sequential_controller.py --config "$CONFIG" train --mode combined --resume
run_step 5 "Evaluate C3 and existing policies at common budgets" python scripts/c3_sequential_controller.py --config "$CONFIG" evaluate
run_step 6 "Generate C3 PNG/PDF figures" python scripts/c3_sequential_controller.py --config "$CONFIG" visualize
run_step 7 "Generate C3 Markdown report" python scripts/c3_sequential_controller.py --config "$CONFIG" report

PIPELINE_ELAPSED="$(($(date +%s) - PIPELINE_STARTED))"
printf '\nC3 pipeline completed in %s\n' "$(format_seconds "$PIPELINE_ELAPSED")"
printf 'Reports: %s/results/c3_sequential_controller/report/\n' "$REPO_ROOT"
