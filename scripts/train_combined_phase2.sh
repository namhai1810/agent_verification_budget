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
  --mode combined \
  --livecodebench-results results/livecodebench \
  --livecodebench-extra-results results/livecodebench_hard_extension \
  --humaneval-results results \
  --humaneval-dataset data/HumanEval.jsonl.gz \
  --humaneval-run-id h1_phase1_qwen25_gpu2-09092ebd5e-d03859f54f \
  --output-dir results/combined_phase2_optimized \
  --test-fraction 0.20 \
  --inner-folds 4 \
  --bootstrap-reps 20000 \
  --seed 42

python scripts/build_combined_phase2_visuals.py \
  --results-dir results/combined_phase2_optimized \
  --bootstrap-reps 20000 \
  --seed 42
