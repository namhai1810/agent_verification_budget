#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${PROJECT_DIR}/configs/vllm_gpu2.yaml"
CHILD_URL="http://127.0.0.1:8011/v1"
TEACHER_URL="http://127.0.0.1:8012/v1"
CHILD_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
TEACHER_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH" >&2
  exit 1
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

check_model "${CHILD_URL}" "${CHILD_MODEL}" "Child"
check_model "${TEACHER_URL}" "${TEACHER_MODEL}" "Teacher"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate agent_verify
cd "${PROJECT_DIR}"

# vLLM started without --api-key ignores this placeholder bearer token.
export VLLM_API_KEY="${VLLM_API_KEY:-local-vllm}"
# Required by the explicit model-call safety gate. Configured local prices and
# max_spend_usd are both zero, so the run cannot accrue configured API spend.
export AGENT_VERIFY_ALLOW_SPEND=YES

python -m agent_verify.cli validate-config --config "${CONFIG_PATH}"
python -m agent_verify.cli check-sandbox --config "${CONFIG_PATH}"
python -m agent_verify.cli run --config "${CONFIG_PATH}" "$@"
