#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/darc_v05_offline_rl.yaml"
SMOKE=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      EXTRA_ARGS+=("--smoke")
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 2
fi

OUTPUT_ROOT="$(python - "$CONFIG" <<'PY'
from pathlib import Path
import sys, yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
print(config["output"]["root"])
PY
)"
mkdir -p "$OUTPUT_ROOT/logs"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG="$OUTPUT_ROOT/logs/run_${STAMP}.log"

if [[ "$SMOKE" -eq 0 ]]; then
  python - <<'PY'
try:
    import tensorboard  # noqa: F401
except ImportError as error:
    raise SystemExit(
        "TensorBoard is required. Reinstall the updated project environment before full training."
    ) from error
PY
fi

echo "DARC v0.5 Conservative Budget-Aware Offline RL"
echo "Config: $CONFIG"
echo "Mode:   $([[ "$SMOKE" -eq 1 ]] && echo synthetic-smoke || echo full-validation)"
echo "Log:    $LOG"
echo "TensorBoard logdir: $OUTPUT_ROOT/tensorboard/concise"
echo "Budget semantics: global ledger enforces hard reserve; per-task allowance controls shadow price only"
echo "Pre-training guard: positive-budget Verify target coverage must be non-zero at every allowance"
python scripts/run_darc_v05_offline_rl.py --config "$CONFIG" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"

