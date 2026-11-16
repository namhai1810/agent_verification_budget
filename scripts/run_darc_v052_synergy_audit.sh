#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

FROZEN_ROOT="data/cbv_bench_v01_frozen_28102026"
EXTENSION_ROOT="results/cbv_train_l2_extension_12112026"
SELECTION_MANIFEST="$EXTENSION_ROOT/selection/bigcodebench_train_l2_extension.json"
OUTPUT="results/darc_v052_synergy_audit"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --frozen-root)
      FROZEN_ROOT="$2"
      shift 2
      ;;
    --extension-root)
      EXTENSION_ROOT="$2"
      shift 2
      ;;
    --selection-manifest)
      SELECTION_MANIFEST="$2"
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

REQUIRED=(
  "$FROZEN_ROOT/train/tasks.jsonl"
  "$FROZEN_ROOT/train/trajectories.jsonl"
  "$EXTENSION_ROOT/collection/manifests.jsonl"
  "$EXTENSION_ROOT/collection/trajectories.jsonl"
  "$SELECTION_MANIFEST"
)
for path in "${REQUIRED[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required train-only input: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT/logs"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG="$OUTPUT/logs/synergy_audit_${STAMP}.log"

echo "DARC v0.5.2 expanded Multi-Verification Synergy Audit"
echo "Scope:      531 train tasks = 381 original + 150 extension"
echo "Mode:       read-only offline analysis"
echo "Output:     $OUTPUT"
echo "Guarantee:  no validation/test outcomes, training, or LLM inference"

python scripts/run_darc_v052_synergy_audit.py \
  --frozen-root "$FROZEN_ROOT" \
  --extension-root "$EXTENSION_ROOT" \
  --selection-manifest "$SELECTION_MANIFEST" \
  --output "$OUTPUT" \
  2>&1 | tee "$LOG"

