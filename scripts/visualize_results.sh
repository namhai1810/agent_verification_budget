#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate agent_verify
cd "${PROJECT_DIR}"

python scripts/extract_and_visualize.py --results-dir results "$@"
