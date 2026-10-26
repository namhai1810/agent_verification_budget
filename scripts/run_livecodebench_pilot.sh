#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_DIR}/configs/livecodebench_vllm_gpu2.yaml"
RESULTS_DIR="${PROJECT_DIR}/results/livecodebench"
SELECTION_COUNT=100
TASK_COUNT=4
TASK_OFFSET=0
SEED=42
REPEATED_PAIRS=1
PREPARE_ONLY=false
RESUME=true
STUDENT_ENDPOINT=""
TEACHER_ENDPOINT=""
RESUME_RUN_ID=""
DEFAULT_STUDENT_ENDPOINT="http://127.0.0.1:8011/v1"
DEFAULT_TEACHER_ENDPOINT="http://127.0.0.1:8012/v1"
STUDENT_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
TEACHER_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --selection-count) SELECTION_COUNT="$2"; shift 2 ;;
    --task-count) TASK_COUNT="$2"; shift 2 ;;
    --task-offset) TASK_OFFSET="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --repeated-pairs) REPEATED_PAIRS="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    --no-resume) RESUME=false; shift ;;
    --resume-run-id) RESUME_RUN_ID="$2"; shift 2 ;;
    --student-endpoint) STUDENT_ENDPOINT="$2"; shift 2 ;;
    --teacher-endpoint) TEACHER_ENDPOINT="$2"; shift 2 ;;
    --prepare-only) PREPARE_ONLY=true; shift ;;
    --help|-h)
      cat <<'EOF'
Usage: bash scripts/run_livecodebench_pilot.sh [options]
  --task-count N           Number of locked tasks to run (default: 4)
  --task-offset N          Start within the locked list (default: 0)
  --results-dir DIR        Result/selection directory (default: results/livecodebench)
  --selection-count N      Size of the frozen selection (default: 100)
  --seed N                 Rollout seed (default: 42)
  --repeated-pairs N       Repetitions per checkpoint (default: 1)
  --resume|--no-resume     Reuse completed append-only records
  --resume-run-id ID       Recover only missing records from an audited earlier run
  --student-endpoint URL   Override Planner/Coder/Reviewer endpoint
  --teacher-endpoint URL   Override Teacher endpoint
  --prepare-only           Lock/preview the selection; do not call models
EOF
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "${PROJECT_DIR}"
if [[ "${RESULTS_DIR}" != /* ]]; then
  RESULTS_DIR="${PROJECT_DIR}/${RESULTS_DIR}"
fi
python -m agent_verify.livecodebench_cli prepare \
  --results-dir "${RESULTS_DIR}" \
  --task-count "${SELECTION_COUNT}" \
  --seed "${SEED}"
if [[ "${PREPARE_ONLY}" == true ]]; then
  exit 0
fi

check_model() {
  local url="$1"
  local model="$2"
  local label="$3"
  local response
  if ! response="$(curl -fsS --max-time 5 "${url}/models")"; then
    echo "${label} endpoint is not ready: ${url}" >&2
    exit 1
  fi
  if [[ "${response}" != *"${model}"* ]]; then
    echo "${label} endpoint does not advertise expected model: ${model}" >&2
    echo "Response: ${response}" >&2
    exit 1
  fi
  echo "${label} ready: ${model}"
}

ACTIVE_STUDENT_ENDPOINT="${STUDENT_ENDPOINT:-${DEFAULT_STUDENT_ENDPOINT}}"
ACTIVE_TEACHER_ENDPOINT="${TEACHER_ENDPOINT:-${DEFAULT_TEACHER_ENDPOINT}}"
check_model "${ACTIVE_STUDENT_ENDPOINT}" "${STUDENT_MODEL}" "Student"
check_model "${ACTIVE_TEACHER_ENDPOINT}" "${TEACHER_MODEL}" "Teacher"

# vLLM started without --api-key accepts this local placeholder. The explicit
# authorization gate is still required for every openai_compatible provider.
# All configured local model prices and max_spend_usd are zero.
export VLLM_API_KEY="${VLLM_API_KEY:-local-vllm}"
export AGENT_VERIFY_ALLOW_SPEND=YES

mkdir -p "${RESULTS_DIR}/logs"
LOG_PATH="${RESULTS_DIR}/logs/lcb_offset${TASK_OFFSET}_${TASK_COUNT}x${REPEATED_PAIRS}.log"
command=(python -m agent_verify.livecodebench_cli run
  --config "${CONFIG}"
  --task-offset "${TASK_OFFSET}"
  --task-count "${TASK_COUNT}"
  --seed "${SEED}"
  --repeated-pairs "${REPEATED_PAIRS}")
if [[ "${RESUME}" == true ]]; then command+=(--resume); else command+=(--no-resume); fi
if [[ -n "${RESUME_RUN_ID}" ]]; then command+=(--resume-run-id "${RESUME_RUN_ID}"); fi
if [[ -n "${STUDENT_ENDPOINT}" ]]; then command+=(--student-endpoint "${STUDENT_ENDPOINT}"); fi
if [[ -n "${TEACHER_ENDPOINT}" ]]; then command+=(--teacher-endpoint "${TEACHER_ENDPOINT}"); fi

printf '\n===== started/resumed %s =====\n' "$(date --iso-8601=seconds)" >>"${LOG_PATH}"
"${command[@]}" >>"${LOG_PATH}" 2>&1 &
experiment_pid=$!
trap 'kill -INT "${experiment_pid}" 2>/dev/null || true; wait "${experiment_pid}" || true; exit 130' INT TERM

set +e
monitor_command=(python scripts/monitor_progress.py
  --results-dir "${RESULTS_DIR}"
  --task-offset "${TASK_OFFSET}"
  --task-count "${TASK_COUNT}"
  --repeated-pairs "${REPEATED_PAIRS}"
  --pid "${experiment_pid}")
if [[ -n "${RESUME_RUN_ID}" ]]; then monitor_command+=(--run-id "${RESUME_RUN_ID}"); fi
"${monitor_command[@]}"
monitor_status=$?
wait "${experiment_pid}"
experiment_status=$?
set -e

if [[ ${experiment_status} -ne 0 ]]; then
  echo "Experiment failed (${experiment_status}). Inspect ${LOG_PATH}" >&2
  exit "${experiment_status}"
fi
if [[ ${monitor_status} -ne 0 ]]; then
  echo "Experiment ended with incomplete planned records. Inspect ${LOG_PATH}" >&2
  exit "${monitor_status}"
fi
echo "Completed. Report: ${RESULTS_DIR}/h1_analysis.md"
echo "Console log: ${LOG_PATH}"
