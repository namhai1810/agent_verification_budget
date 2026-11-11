#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="configs/darc_v05_offline_rl.yaml"
TRACE="results/darc_v05_offline_rl/diagnostics/tables/task_outcomes.csv"
OUTPUT="results/darc_v05_offline_rl/synergy_audit"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --decision-trace)
      TRACE="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

for required in "$CONFIG" "$TRACE"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT/logs"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG="$OUTPUT/logs/synergy_audit_${STAMP}.log"

echo "DARC v0.5 multi-verification synergy audit"
echo "Scope:      381 train + 98 validation tasks"
echo "Mode:       read-only offline analysis"
echo "Output:     $OUTPUT"
echo "No training, frozen-test access, or Student/Teacher inference."

python scripts/run_darc_v05_synergy_audit.py \
  --config "$CONFIG" \
  --decision-trace "$TRACE" \
  --output "$OUTPUT" \
  2>&1 | tee "$LOG"
