#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/home/llm/hainn/hainn/agent_verification_budget"
FROZEN_ROOT="${FROZEN_ROOT:-${PROJECT_ROOT}/data/cbv_bench_v01_frozen_28102026}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/cbv_v01_experiments_29102026}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
LOG_DIR="${OUTPUT_ROOT}/logs"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${LOG_DIR}/overnight_${RUN_STAMP}.log"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

if [[ -f "/home/llm/hainn/miniconda3/etc/profile.d/conda.sh" ]]; then
  # Make the script genuinely one-command even from a fresh tmux shell.
  source "/home/llm/hainn/miniconda3/etc/profile.d/conda.sh"
fi
conda activate agent_verify

if [[ ! -f "${FROZEN_ROOT}/manifest.json" ]]; then
  echo "ERROR: frozen manifest not found: ${FROZEN_ROOT}/manifest.json" >&2
  exit 2
fi
available_kb="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR==2 {print $4}')"
if (( available_kb < 2097152 )); then
  echo "ERROR: less than 2 GiB free on the output filesystem." >&2
  exit 2
fi
python -c "import joblib, matplotlib, numpy, sklearn, tqdm; import agent_verify"

started="$(date +%s)"
echo "[$(date -u +%FT%TZ)] START CBV-Bench v0.1 frozen experiments" | tee -a "${LOG_PATH}"
echo "Frozen data: ${FROZEN_ROOT}" | tee -a "${LOG_PATH}"
echo "Output:      ${OUTPUT_ROOT}" | tee -a "${LOG_PATH}"
echo "Bootstrap:   ${BOOTSTRAP}" | tee -a "${LOG_PATH}"
echo "This run uses cached counterfactual data only; no Student/Teacher inference." | tee -a "${LOG_PATH}"

python -u scripts/run_cbv_v01_experiments.py \
  --frozen-root "${FROZEN_ROOT}" \
  --output "${OUTPUT_ROOT}" \
  --bootstrap "${BOOTSTRAP}" \
  2>&1 | tee -a "${LOG_PATH}"

elapsed="$(( $(date +%s) - started ))"
printf '[%s] DONE in %02d:%02d:%02d\n' \
  "$(date -u +%FT%TZ)" "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" \
  | tee -a "${LOG_PATH}"
echo "Report:    ${OUTPUT_ROOT}/report/cbv_v01_all_experiments.md" | tee -a "${LOG_PATH}"
echo "Summary:   ${OUTPUT_ROOT}/summary.json" | tee -a "${LOG_PATH}"
echo "Tables:    ${OUTPUT_ROOT}/tables/" | tee -a "${LOG_PATH}"
echo "Figures:   ${OUTPUT_ROOT}/figures/" | tee -a "${LOG_PATH}"
echo "Master log:${LOG_PATH}" | tee -a "${LOG_PATH}"
