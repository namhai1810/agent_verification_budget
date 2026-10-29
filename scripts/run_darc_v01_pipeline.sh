#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/home/llm/hainn/hainn/agent_verification_budget"
CONFIG_PATH="${DARC_CONFIG:-${PROJECT_ROOT}/configs/darc_v01.yaml}"
OUTPUT_ROOT="${DARC_OUTPUT_ROOT:-${PROJECT_ROOT}/results/darc_v01_29102026}"
LOG_DIR="${OUTPUT_ROOT}/logs"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${LOG_DIR}/darc_${RUN_STAMP}.log"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

if [[ -f "/home/llm/hainn/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "/home/llm/hainn/miniconda3/etc/profile.d/conda.sh"
fi
conda activate agent_verify

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: DARC config not found: ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ ! -f "${PROJECT_ROOT}/data/cbv_bench_v01_frozen_28102026/manifest.json" ]]; then
  echo "ERROR: frozen CBV v0.1 manifest is missing." >&2
  exit 2
fi
if [[ ! -f "${PROJECT_ROOT}/results/cbv_v01_experiments_29102026/tables/policy_results.csv" ]]; then
  echo "ERROR: existing matched-budget baseline table is missing." >&2
  exit 2
fi
available_kb="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR==2 {print $4}')"
if (( available_kb < 3145728 )); then
  echo "ERROR: less than 3 GiB free on the DARC output filesystem." >&2
  exit 2
fi

if ! python -c "import matplotlib, numpy, sklearn, torch, tqdm, transformers, yaml; import agent_verify"; then
  echo "ERROR: DARC dependencies are missing. Run: pip install -e '.[darc]'" >&2
  exit 2
fi

started="$(date +%s)"
echo "[$(date -u +%FT%TZ)] START DARC v0.1 offline pipeline" | tee -a "${LOG_PATH}"
echo "Config: ${CONFIG_PATH}" | tee -a "${LOG_PATH}"
echo "Output: ${OUTPUT_ROOT}" | tee -a "${LOG_PATH}"
echo "Frozen encoder embeddings are cached; Student/Teacher inference is disabled." | tee -a "${LOG_PATH}"
echo "Frozen test is loaded only after train/validation selection is persisted." | tee -a "${LOG_PATH}"

python -u scripts/run_darc_v01.py \
  --config "${CONFIG_PATH}" \
  --project-root "${PROJECT_ROOT}" \
  2>&1 | tee -a "${LOG_PATH}"

elapsed="$(( $(date +%s) - started ))"
printf '[%s] DONE in %02d:%02d:%02d\n' \
  "$(date -u +%FT%TZ)" "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" \
  | tee -a "${LOG_PATH}"
echo "Report:  ${OUTPUT_ROOT}/report/darc_v01_report.md" | tee -a "${LOG_PATH}"
echo "Summary: ${OUTPUT_ROOT}/summary.json" | tee -a "${LOG_PATH}"
echo "Tables:  ${OUTPUT_ROOT}/tables/" | tee -a "${LOG_PATH}"
echo "Figures: ${OUTPUT_ROOT}/figures/" | tee -a "${LOG_PATH}"
echo "Log:     ${LOG_PATH}" | tee -a "${LOG_PATH}"
