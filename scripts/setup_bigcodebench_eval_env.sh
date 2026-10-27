#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_ENV="${BASE_ENV:-agent_verify}"
BCB_ENV="${BCB_ENV:-agent_verify_bcb}"
SELECTION="${SELECTION:-results/cbv_bench_v2_bigcodebench_26102026/selections/scale_l1_1000.json}"
LOCK_DIR="${LOCK_DIR:-results/environment_manifests}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available." >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk 'NF >= 1 {print $1}' | grep -Fxq "$BCB_ENV"; then
  echo "Creating isolated environment $BCB_ENV from $BASE_ENV..."
  conda create --yes --name "$BCB_ENV" --clone "$BASE_ENV"
else
  echo "Reusing existing environment: $BCB_ENV"
fi

echo "Installing BigCodeBench evaluator dependencies..."
BCB_PREFIX="$(conda env list | awk -v environment="$BCB_ENV" '$1 == environment {print $NF; exit}')"
BCB_PYTHON="$BCB_PREFIX/bin/python"
if [[ ! -x "$BCB_PYTHON" ]]; then
  echo "ERROR: Python executable not found for $BCB_ENV: $BCB_PYTHON" >&2
  exit 2
fi
"$BCB_PYTHON" -m pip install --upgrade pip
"$BCB_PYTHON" -m pip install --no-cache-dir -r requirements-bigcodebench-eval.txt
"$BCB_PYTHON" -m pip install --no-deps --editable "$PROJECT_ROOT"

echo "Checking the frozen 1,000-task selection..."
"$BCB_PYTHON" scripts/check_bigcodebench_dependencies.py \
  "$SELECTION" --import-check
"$BCB_PYTHON" -m pip check

mkdir -p "$LOCK_DIR"
"$BCB_PYTHON" -m pip freeze >"$LOCK_DIR/${BCB_ENV}_pip_freeze.txt"

echo
echo "BigCodeBench evaluator environment is ready."
echo "Environment: $BCB_ENV"
echo "Resolved package manifest: $LOCK_DIR/${BCB_ENV}_pip_freeze.txt"
