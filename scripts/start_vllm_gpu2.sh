#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${PROJECT_DIR}/results/vllm_gpu2"
GPU_ID="${GPU_ID:-2}"
CHILD_PORT="${CHILD_PORT:-8011}"
TEACHER_PORT="${TEACHER_PORT:-8012}"
CHILD_MODEL="${CHILD_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct-AWQ}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}"
VLLM_BIN="${VLLM_BIN:-vllm}"
CHILD_MEMORY_FRACTION="${CHILD_MEMORY_FRACTION:-0.23}"
TEACHER_MEMORY_FRACTION="${TEACHER_MEMORY_FRACTION:-0.62}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

mkdir -p "${STATE_DIR}"

usage() {
  echo "Usage: bash scripts/start_vllm_gpu2.sh {start|status|stop|logs}"
}

require_vllm() {
  if [[ "${VLLM_BIN}" == */* ]]; then
    [[ -x "${VLLM_BIN}" ]] || {
      echo "vLLM executable not found: ${VLLM_BIN}" >&2
      exit 1
    }
  elif ! command -v "${VLLM_BIN}" >/dev/null 2>&1; then
    echo "vllm is not installed in the active server environment." >&2
    echo "Activate your vLLM environment or set VLLM_BIN=/absolute/path/to/vllm." >&2
    exit 1
  fi
}

port_is_open() {
  (echo >/dev/tcp/127.0.0.1/"$1") >/dev/null 2>&1
}

start_server() {
  local name="$1"
  local model="$2"
  local port="$3"
  local memory_fraction="$4"
  local pid_file="${STATE_DIR}/${name}.pid"
  local log_file="${STATE_DIR}/${name}.log"

  if port_is_open "${port}"; then
    echo "Port ${port} is already in use; refusing to start ${name}." >&2
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup "${VLLM_BIN}" serve "${model}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --gpu-memory-utilization "${memory_fraction}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enforce-eager \
    >"${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${pid_file}"
  echo "Started ${name}: pid=${pid}, model=${model}, port=${port}, log=${log_file}"
}

show_server() {
  local name="$1"
  local port="$2"
  local pid_file="${STATE_DIR}/${name}.pid"
  local pid=""
  [[ -f "${pid_file}" ]] && pid="$(<"${pid_file}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "${name}: process running (pid=${pid}); endpoint=http://127.0.0.1:${port}/v1"
  else
    echo "${name}: no tracked process"
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "${name}: API ready"
  else
    echo "${name}: API not ready"
  fi
}

stop_server() {
  local name="$1"
  local pid_file="${STATE_DIR}/${name}.pid"
  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: no pid file"
    return
  fi
  local pid
  pid="$(<"${pid_file}")"
  if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
    echo "${name}: invalid pid file; remove it manually: ${pid_file}" >&2
    return 1
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "${name}: stale pid ${pid}; removing pid file"
    rm -f "${pid_file}"
    return
  fi
  local command_line
  command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  if [[ "${command_line}" != *"vllm"* || "${command_line}" != *"serve"* ]]; then
    echo "${name}: pid ${pid} is not a vLLM server; refusing to stop it." >&2
    return 1
  fi
  kill "${pid}"
  rm -f "${pid_file}"
  echo "${name}: sent SIGTERM to pid ${pid}"
}

action="${1:-start}"
case "${action}" in
  start)
    require_vllm
    if port_is_open "${CHILD_PORT}" || port_is_open "${TEACHER_PORT}"; then
      echo "One of ports ${CHILD_PORT}/${TEACHER_PORT} is already in use." >&2
      echo "Run: bash scripts/start_vllm_gpu2.sh status" >&2
      exit 1
    fi
    start_server child "${CHILD_MODEL}" "${CHILD_PORT}" "${CHILD_MEMORY_FRACTION}"
    start_server teacher "${TEACHER_MODEL}" "${TEACHER_PORT}" "${TEACHER_MEMORY_FRACTION}"
    echo "Both loaders were launched on physical GPU ${GPU_ID}. Model loading can take several minutes."
    echo "Monitor with: bash scripts/start_vllm_gpu2.sh logs"
    ;;
  status)
    show_server child "${CHILD_PORT}"
    show_server teacher "${TEACHER_PORT}"
    ;;
  stop)
    stop_server child
    stop_server teacher
    ;;
  logs)
    touch "${STATE_DIR}/child.log" "${STATE_DIR}/teacher.log"
    tail -n 40 -F "${STATE_DIR}/child.log" "${STATE_DIR}/teacher.log"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
