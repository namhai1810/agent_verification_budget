#!/usr/bin/env bash
# Recover the original 100-task run, then collect the remaining 75 v6 tasks.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGINAL_RUN_ID="h1_livecodebench_v6_qwen25_gpu2-20646527df-9e5e2e077f"
EXTENSION_RUN_ID="h1_livecodebench_v6_hard_extension_qwen25_gpu2-d6cd8c8193-28e3450171"
BASE_RESULTS="${PROJECT_DIR}/results/livecodebench"
EXTENSION_RESULTS="${PROJECT_DIR}/results/livecodebench_hard_extension"
MASTER_LOG_DIR="${PROJECT_DIR}/results/livecodebench_overnight_logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${MASTER_LOG_DIR}/overnight_${TIMESTAMP}.log"

mkdir -p "${MASTER_LOG_DIR}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

cd "${PROJECT_DIR}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "agent_verify" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: activate conda environment 'agent_verify' before running this script." >&2
    exit 2
  fi
  eval "$(conda shell.bash hook)"
  conda activate agent_verify
fi

echo "============================================================"
echo "LiveCodeBench overnight collection"
echo "Started: $(date --iso-8601=seconds)"
echo "Master log: ${MASTER_LOG}"
echo "============================================================"

echo
echo "[Preflight] Locking 175 tasks and auditing the original prefix..."
python -m agent_verify.livecodebench_cli prepare \
  --results-dir "${EXTENSION_RESULTS}" \
  --task-count 175 \
  --seed 42
python scripts/verify_livecodebench_extension.py \
  --base "${BASE_RESULTS}/selected_tasks.json" \
  --extension "${EXTENSION_RESULTS}/selected_tasks.json"

echo
echo "[1/2] Recovering only missing records in the original 100-task run..."
set +e
bash scripts/run_livecodebench_pilot.sh \
  --task-count 100 \
  --seed 42 \
  --repeated-pairs 1 \
  --resume \
  --resume-run-id "${ORIGINAL_RUN_ID}"
recovery_status=$?
set -e

# Always leave a dashboard of the state reached, including on interrupted runs.
python scripts/build_livecodebench_report_visuals.py \
  --results-dir "${BASE_RESULTS}" \
  --output-dir "${BASE_RESULTS}/report" \
  --run-id "${ORIGINAL_RUN_ID}" \
  --output-name "livecodebench_100_dashboard.png" || true

if [[ ${recovery_status} -ne 0 ]]; then
  echo "ERROR: recovery ended with status ${recovery_status}; extension was not started." >&2
  echo "Inspect: ${BASE_RESULTS}/logs/lcb_offset0_100x1.log" >&2
  exit "${recovery_status}"
fi

echo
echo "[2/2] Running the remaining 75 Hard tasks (offset 100 of frozen 175)..."
set +e
bash scripts/run_livecodebench_pilot.sh \
  --config "${PROJECT_DIR}/configs/livecodebench_hard_extension_gpu2.yaml" \
  --results-dir "${EXTENSION_RESULTS}" \
  --selection-count 175 \
  --task-offset 100 \
  --task-count 75 \
  --seed 42 \
  --repeated-pairs 1 \
  --resume \
  --resume-run-id "${EXTENSION_RUN_ID}"
extension_status=$?
set -e

# The experiment writes h1_analysis.md, summary.md and result CSVs. This adds
# a publication-ready completion/accuracy/intervention dashboard and CSV.
python scripts/build_livecodebench_report_visuals.py \
  --results-dir "${EXTENSION_RESULTS}" \
  --output-dir "${EXTENSION_RESULTS}/report" \
  --output-name "livecodebench_hard_extension_dashboard.png" || true

if [[ ${extension_status} -ne 0 ]]; then
  echo "ERROR: extension ended with status ${extension_status}." >&2
  echo "Inspect: ${EXTENSION_RESULTS}/logs/lcb_offset100_75x1.log" >&2
  exit "${extension_status}"
fi

echo
echo "============================================================"
echo "All overnight jobs completed: $(date --iso-8601=seconds)"
echo "Original-100 report: ${BASE_RESULTS}/h1_analysis.md"
echo "Original-100 dashboard: ${BASE_RESULTS}/report/livecodebench_100_dashboard.png"
echo "Hard-75 report: ${EXTENSION_RESULTS}/h1_analysis.md"
echo "Hard-75 dashboard: ${EXTENSION_RESULTS}/report/livecodebench_hard_extension_dashboard.png"
echo "Recovery console: ${BASE_RESULTS}/logs/lcb_offset0_100x1.log"
echo "Hard-75 console: ${EXTENSION_RESULTS}/logs/lcb_offset100_75x1.log"
echo "Master log: ${MASTER_LOG}"
echo "============================================================"
