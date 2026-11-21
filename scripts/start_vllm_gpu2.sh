#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${STATE_DIR:-${PROJECT_DIR}/results/vllm_gpu2}"
GPU_ID="${GPU_ID:-2}"
CHILD_PORT="${CHILD_PORT:-8011}"
TEACHER_PORT="${TEACHER_PORT:-8012}"
CHILD_MODEL="${CHILD_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct-AWQ}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}"
TEACHER_SERVED_MODEL_NAME="${TEACHER_SERVED_MODEL_NAME:-${TEACHER_MODEL}}"
TEACHER_LANGUAGE_MODEL_ONLY="${TEACHER_LANGUAGE_MODEL_ONLY:-0}"
TEACHER_DEFAULT_CHAT_TEMPLATE_KWARGS="${TEACHER_DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
TEACHER_REASONING_PARSER="${TEACHER_REASONING_PARSER:-}"
VLLM_BIN="${VLLM_BIN:-vllm}"
CHILD_MEMORY_FRACTION="${CHILD_MEMORY_FRACTION:-0.23}"
TEACHER_MEMORY_FRACTION="${TEACHER_MEMORY_FRACTION:-0.62}"
COMBINED_MEMORY_FRACTION="${COMBINED_MEMORY_FRACTION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-600}"

mkdir -p "${STATE_DIR}"

usage() {
  echo "Usage: bash scripts/start_vllm_gpu2.sh {start|start-child|start-teacher|status|stop|stop-child|stop-teacher|logs}"
}

wait_for_api() {
  local name="$1"
  local port="$2"
  local pid_file="${STATE_DIR}/${name}.pid"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  echo "Waiting for ${name} API on port ${port} (timeout ${STARTUP_TIMEOUT_SECONDS}s)..."
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "${name}: API ready"
      return 0
    fi
    if [[ -f "${pid_file}" ]]; then
      local pid
      pid="$(<"${pid_file}")"
      if [[ "${pid}" =~ ^[0-9]+$ ]] && ! kill -0 "${pid}" 2>/dev/null; then
        echo "ERROR: ${name} exited during startup. Inspect ${STATE_DIR}/${name}.log" >&2
        return 1
      fi
    fi
    sleep 5
  done
  echo "ERROR: ${name} API did not become ready within ${STARTUP_TIMEOUT_SECONDS}s." >&2
  return 1
}

start_child() {
  local memory_fraction="$CHILD_MEMORY_FRACTION"
  if port_is_open "$TEACHER_PORT"; then
    # vLLM >=0.32 accounts memory already occupied by another process on the
    # same GPU. The second process therefore needs the cumulative target.
    memory_fraction="$COMBINED_MEMORY_FRACTION"
  fi
  start_server child "$CHILD_MODEL" "$CHILD_PORT" "$memory_fraction"
  wait_for_api child "$CHILD_PORT"
}

start_teacher() {
  local memory_fraction="$TEACHER_MEMORY_FRACTION"
  if port_is_open "$CHILD_PORT"; then
    memory_fraction="$COMBINED_MEMORY_FRACTION"
  fi
  start_server teacher "$TEACHER_MODEL" "$TEACHER_PORT" "$memory_fraction"
  wait_for_api teacher "$TEACHER_PORT"
}

with_start_lock() {
  local name="$1"
  shift
  local lock_file="${STATE_DIR}/${name}.start.lock"
  local lock_fd
  exec {lock_fd}>"${lock_file}"
  if ! flock -n "${lock_fd}"; then
    echo "${name}: another startup is already in progress; refusing a duplicate launch." >&2
    return 1
  fi
  "$@"
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
  local extra_args=()

  if [[ -f "${pid_file}" ]]; then
    local tracked_pid
    tracked_pid="$(<"${pid_file}")"
    if [[ "${tracked_pid}" =~ ^[0-9]+$ ]] && kill -0 "${tracked_pid}" 2>/dev/null; then
      echo "${name}: tracked process ${tracked_pid} is already running; refusing a duplicate launch." >&2
      exit 1
    fi
  fi

  if [[ "${name}" == "teacher" ]]; then
    extra_args+=(--served-model-name "${TEACHER_SERVED_MODEL_NAME}")
    if [[ "${TEACHER_LANGUAGE_MODEL_ONLY}" == "1" ]]; then
      extra_args+=(--language-model-only)
    fi
    if [[ -n "${TEACHER_DEFAULT_CHAT_TEMPLATE_KWARGS}" ]]; then
      extra_args+=(--default-chat-template-kwargs "${TEACHER_DEFAULT_CHAT_TEMPLATE_KWARGS}")
    fi
    if [[ -n "${TEACHER_REASONING_PARSER}" ]]; then
      extra_args+=(--reasoning-parser "${TEACHER_REASONING_PARSER}")
    fi
  fi

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
    "${extra_args[@]}" \
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
  local expected_port expected_model
  if [[ "${name}" == "teacher" ]]; then
    expected_port="${TEACHER_PORT}"
    expected_model="${TEACHER_MODEL}"
  else
    expected_port="${CHILD_PORT}"
    expected_model="${CHILD_MODEL}"
  fi
  if [[ "${command_line}" != *"--port ${expected_port}"* || \
        "${command_line}" != *"serve ${expected_model}"* ]]; then
    echo "${name}: pid ${pid} does not match expected model/port; refusing to stop it." >&2
    echo "Expected model=${expected_model}, port=${expected_port}" >&2
    return 1
  fi
  kill "${pid}"
  echo "${name}: sent SIGTERM to pid ${pid}"
  local deadline=$((SECONDS + 60))
  while kill -0 "${pid}" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "${name}: pid ${pid} did not stop within 60s; refusing to send SIGKILL." >&2
    return 1
  fi
  rm -f "${pid_file}"
  echo "${name}: stopped"
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
    with_start_lock child start_child
    with_start_lock teacher start_teacher
    echo "Both loaders are ready on physical GPU ${GPU_ID}."
    echo "Monitor with: bash scripts/start_vllm_gpu2.sh logs"
    ;;
  start-child)
    require_vllm
    if port_is_open "${CHILD_PORT}"; then
      echo "Port ${CHILD_PORT} is already in use; refusing to start child." >&2
      exit 1
    fi
    with_start_lock child start_child
    ;;
  start-teacher)
    require_vllm
    if port_is_open "${TEACHER_PORT}"; then
      echo "Port ${TEACHER_PORT} is already in use; refusing to start teacher." >&2
      exit 1
    fi
    with_start_lock teacher start_teacher
    ;;
  status)
    show_server child "${CHILD_PORT}"
    show_server teacher "${TEACHER_PORT}"
    ;;
  stop)
    stop_server child
    stop_server teacher
    ;;
  stop-child)
    stop_server child
    ;;
  stop-teacher)
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
