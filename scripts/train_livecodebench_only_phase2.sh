#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "agent_verify" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate agent_verify
fi

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/train_livecodebench_predictor.py \
  --mode livecodebench \
  --livecodebench-results results/livecodebench \
  --livecodebench-extra-results results/livecodebench_hard_extension \
  --output-dir results/livecodebench_only_phase2_optimized \
  --test-fraction 0.20 \
  --inner-folds 4 \
  --bootstrap-reps 20000 \
  --seed 42

python scripts/build_livecodebench_only_comparison.py \
  --combined-dir results/combined_phase2_optimized \
  --lcb-only-dir results/livecodebench_only_phase2_optimized \
  --bootstrap-reps 20000 \
  --seed 42
