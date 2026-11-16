#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/darc_v053_safe_policy.yaml"
MODE=""
RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--smoke|--full) MODE="$1"; shift ;;
    --resume) RESUME=1; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Choose exactly one of --dry-run, --smoke, or --full" >&2
  exit 2
fi

OUTPUT_ROOT="$(python - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding='utf-8') as handle:
    print(yaml.safe_load(handle)['output']['root'])
PY
)"
mkdir -p "$OUTPUT_ROOT/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$OUTPUT_ROOT/logs/${MODE#--}_${STAMP}.log"
ln -sfn "$(basename "$LOG")" "$OUTPUT_ROOT/logs/latest.log"

COMMAND=(python scripts/run_darc_v053_safe_policy.py --config "$CONFIG" "$MODE")
if [[ "$RESUME" -eq 1 ]]; then
  COMMAND+=(--resume)
fi

echo "DARC v0.5.3 mode: $MODE"
echo "Config: $CONFIG"
echo "Log: $LOG"
"${COMMAND[@]}" 2>&1 | tee "$LOG"
