#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/darc_v05_531_scaling.yaml"
MODE="full"
RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --smoke)
      MODE="smoke"
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
OUTPUT_ROOT="$(python - "$CONFIG" <<'PY'
from pathlib import Path
import sys, yaml
print(yaml.safe_load(Path(sys.argv[1]).read_text())["output"]["root"])
PY
)"
mkdir -p "$OUTPUT_ROOT/logs"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG="$OUTPUT_ROOT/logs/${MODE}_${STAMP}.log"
ln -sfn "$(basename "$LOG")" "$OUTPUT_ROOT/logs/latest.log"

ARGS=(--config "$CONFIG")
[[ "$MODE" == "smoke" ]] && ARGS+=(--smoke)
[[ "$RESUME" -eq 1 ]] && ARGS+=(--resume)

echo "DARC v0.5 Myopic-i0 · 531-task scaling"
echo "Mode:   $MODE"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_ROOT"
echo "Log:    $LOG"
echo "Frozen test: forbidden"
python scripts/run_darc_v05_531_scaling.py "${ARGS[@]}" 2>&1 | tee "$LOG"
