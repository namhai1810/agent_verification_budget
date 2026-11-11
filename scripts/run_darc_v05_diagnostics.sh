#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/darc_v05_offline_rl.yaml"
ARTIFACTS="results/darc_v05_offline_rl_budgetfix_31102026"
OUTPUT="results/darc_v05_offline_rl/diagnostics"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --artifacts-root)
      ARTIFACTS="$2"
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

for required in \
  "$CONFIG" \
  "$ARTIFACTS/selection_manifest.json" \
  "$ARTIFACTS/tables/policy_grid.csv"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required artifact: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT/logs"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG="$OUTPUT/logs/diagnostics_${STAMP}.log"

echo "DARC v0.5 seed/policy diagnostics"
echo "Mode:       validation-only, read-only replay"
echo "Config:     $CONFIG"
echo "Artifacts:  $ARTIFACTS"
echo "Output:     $OUTPUT"
echo "Log:        $LOG"
echo "No training, frozen-test access, or Student/Teacher inference."

python scripts/run_darc_v05_diagnostics.py \
  --config "$CONFIG" \
  --artifacts-root "$ARTIFACTS" \
  --output "$OUTPUT" \
  2>&1 | tee "$LOG"

