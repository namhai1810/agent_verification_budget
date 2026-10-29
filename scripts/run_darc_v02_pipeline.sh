#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "DARC v0.2: grouped train CV -> final refit -> frozen-validation report"
echo "No Student/Teacher inference; frozen test is not loaded."
python scripts/run_darc_v02.py --config configs/darc_v02.yaml --project-root "$PROJECT_ROOT"
