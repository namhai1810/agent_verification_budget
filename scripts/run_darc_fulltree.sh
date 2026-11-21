#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="configs/darc_qwen38_fulltree.yaml"
MODE=""
RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--smoke|--crossval|--final-fit) MODE="$1"; shift ;;
    --resume) RESUME=1; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${MODE}" ]]; then
  echo "Choose one of --dry-run, --smoke, --crossval, or --final-fit" >&2
  exit 2
fi

OUTPUT_ROOT="$(/home/llm/hainn/miniconda3/envs/agent_verify/bin/python - "${CONFIG}" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    print(yaml.safe_load(handle)["output"]["root"])
PY
)"
mkdir -p "${OUTPUT_ROOT}/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${OUTPUT_ROOT}/logs/${MODE#--}_${STAMP}.log"
ln -sfn "$(basename "${LOG}")" "${OUTPUT_ROOT}/logs/latest.log"

COMMAND=(
  /home/llm/hainn/miniconda3/envs/agent_verify/bin/python
  scripts/run_darc_fulltree.py
  --config "${CONFIG}"
  "${MODE}"
)
if [[ "${RESUME}" -eq 1 ]]; then
  COMMAND+=(--resume)
fi

echo "DARC Full-Tree mode: ${MODE}"
echo "Data: 210 Qwen3.8 Full-8 training tasks; validation/frozen/prospective forbidden"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_ROOT}"
echo "Log: ${LOG}"
"${COMMAND[@]}" 2>&1 | tee "${LOG}"
