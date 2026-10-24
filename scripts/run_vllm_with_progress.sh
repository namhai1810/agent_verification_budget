#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_OFFSET=0
TASK_COUNT=24
REPEATED_PAIRS=1
TASK_OFFSET_SEEN=false
TASK_COUNT_SEEN=false
REPEATED_PAIRS_SEEN=false

arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  case "${arguments[index]}" in
    --task-offset)
      TASK_OFFSET="${arguments[index + 1]}"
      TASK_OFFSET_SEEN=true
      ((index += 1))
      ;;
    --task-count)
      TASK_COUNT="${arguments[index + 1]}"
      TASK_COUNT_SEEN=true
      ((index += 1))
      ;;
    --repeated-pairs)
      REPEATED_PAIRS="${arguments[index + 1]}"
      REPEATED_PAIRS_SEEN=true
      ((index += 1))
      ;;
  esac
done

if [[ "${TASK_OFFSET_SEEN}" == false ]]; then
  arguments+=(--task-offset "${TASK_OFFSET}")
fi
if [[ "${TASK_COUNT_SEEN}" == false ]]; then
  arguments+=(--task-count "${TASK_COUNT}")
fi
if [[ "${REPEATED_PAIRS_SEEN}" == false ]]; then
  arguments+=(--repeated-pairs "${REPEATED_PAIRS}")
fi

mkdir -p "${PROJECT_DIR}/results/logs"
if [[ "${TASK_OFFSET}" == 0 ]]; then
  LOG_PATH="${PROJECT_DIR}/results/logs/phase1_${TASK_COUNT}x${REPEATED_PAIRS}_console.log"
else
  LOG_PATH="${PROJECT_DIR}/results/logs/phase1_offset${TASK_OFFSET}_${TASK_COUNT}x${REPEATED_PAIRS}_console.log"
fi

cd "${PROJECT_DIR}"
printf '\n===== resumed/started %s =====\n' "$(date --iso-8601=seconds)" >>"${LOG_PATH}"
bash scripts/run_vllm_pilot.sh "${arguments[@]}" >>"${LOG_PATH}" 2>&1 &
experiment_pid=$!

interrupt_experiment() {
  echo
  if kill -0 "${experiment_pid}" 2>/dev/null; then
    kill -INT "${experiment_pid}" 2>/dev/null || true
    wait "${experiment_pid}" 2>/dev/null || true
  fi
  echo "Interrupted. Re-run the same command to resume completed records."
  exit 130
}
trap interrupt_experiment INT TERM

set +e
python3 scripts/monitor_progress.py \
  --results-dir results \
  --task-offset "${TASK_OFFSET}" \
  --task-count "${TASK_COUNT}" \
  --repeated-pairs "${REPEATED_PAIRS}" \
  --pid "${experiment_pid}"
monitor_status=$?
wait "${experiment_pid}"
experiment_status=$?
set -e

if [[ ${experiment_status} -ne 0 ]]; then
  echo "Experiment exited with status ${experiment_status}."
  echo "Inspect: ${LOG_PATH}"
  exit "${experiment_status}"
fi

if [[ ${monitor_status} -ne 0 ]]; then
  echo "Experiment process ended, but planned records are incomplete."
  echo "Inspect: ${LOG_PATH} and results/events.jsonl"
  exit "${monitor_status}"
fi

bash scripts/visualize_results.sh >>"${LOG_PATH}" 2>&1
echo "Experiment and visualization completed."
echo "Console log: ${LOG_PATH}"
echo "H1 report: ${PROJECT_DIR}/results/summary.md"
echo "Run-specific dashboard/report: ${PROJECT_DIR}/results/extracted/<run_id>/"
