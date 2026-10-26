#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${PROJECT_DIR}/.vendor/LiveCodeBench"
REPOSITORY="https://github.com/LiveCodeBench/LiveCodeBench.git"
COMMIT="28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"

mkdir -p "${PROJECT_DIR}/.vendor"
if [[ ! -d "${TARGET}/.git" ]]; then
  git clone --filter=blob:none "${REPOSITORY}" "${TARGET}"
fi
git -C "${TARGET}" fetch --depth 1 origin "${COMMIT}"
git -C "${TARGET}" checkout --detach "${COMMIT}"

python - <<'PY'
import numpy
print(f"Official evaluator dependency ready: numpy={numpy.__version__}")
PY
echo "LiveCodeBench evaluator pinned at: $(git -C "${TARGET}" rev-parse HEAD)"
