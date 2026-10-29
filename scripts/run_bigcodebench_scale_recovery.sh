#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DATE_SUFFIX="${RUN_DATE_SUFFIX:-28102026}"
CBV_CONDA_ENV="${CBV_CONDA_ENV:-agent_verify_bcb}"
COMPATIBILITY="5d1a0d2f2b0a767df2fda1d34b634236ae97916db93ddd20b2753288ef3589ad"
BASE_RESULTS="results/cbv_bench_v2_bigcodebench_26102026"
SCALE_DIR="results/cbv_bench_scale_${RUN_DATE_SUFFIX}"
L1_RESULTS="results/cbv_bench_v2_bigcodebench_${RUN_DATE_SUFFIX}"
L2_RESULTS="results/cbv_bench_v2_bigcodebench_natural_l2_extension_${RUN_DATE_SUFFIX}"
ORIGINAL_L1="$SCALE_DIR/selections/level1_extension.json"
COMPLETED_L1="$SCALE_DIR/selections/level1_completed_scope.json"
ATTRITION_REPORT="$SCALE_DIR/selections/level1_attrition_report.json"
L2_FULL="$SCALE_DIR/selections/level2_natural_full_250_recovered.json"
L2_EXTENSION="$SCALE_DIR/selections/level2_natural_extension_225_recovered.json"
LOG_DIR="results/cbv_bench_overnight_${RUN_DATE_SUFFIX}/logs/recovery_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/recovery.log") 2>&1

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CBV_CONDA_ENV" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CBV_CONDA_ENV"
fi
export PYTHONUNBUFFERED=1
export VLLM_API_KEY="${VLLM_API_KEY:-local}"
export AGENT_VERIFY_ALLOW_SPEND="${AGENT_VERIFY_ALLOW_SPEND:-YES}"

curl --fail --silent --show-error http://127.0.0.1:8011/v1/models >/dev/null
curl --fail --silent --show-error http://127.0.0.1:8012/v1/models >/dev/null

python scripts/build_cbv_completed_scope.py \
  --source-selection "$ORIGINAL_L1" \
  --trajectories "$L1_RESULTS/trajectories.jsonl" \
  --errors "$L1_RESULTS/errors.jsonl" \
  --compatibility "$COMPATIBILITY" --level 1 \
  --output-selection "$COMPLETED_L1" --report "$ATTRITION_REPORT"

python scripts/build_bigcodebench_scale_extension.py \
  --universe "$SCALE_DIR/selections/bigcodebench_all_1140.json" \
  --completed-level1 "$BASE_RESULTS/selections/recovery_l1_200_v2.json" \
  --frozen-natural-level2 "$BASE_RESULTS/selections/scale_l2_250.json" \
  --completed-level2 "$BASE_RESULTS/selections/recovery_l2_25_v2.json" \
  --errors "$BASE_RESULTS/errors.jsonl" \
  --errors "$L1_RESULTS/errors.jsonl" \
  --level2-errors "$L2_RESULTS/errors.jsonl" \
  --compatibility "$COMPATIBILITY" \
  --output-level1-extension "$SCALE_DIR/selections/level1_extension_after_attrition.json" \
  --output-level2-full "$L2_FULL" \
  --output-level2-extension "$L2_EXTENSION" \
  --report "$SCALE_DIR/selections/extension_after_attrition_report.json"

python scripts/repair_cbv_seed_drift.py \
  --results-dir "$L2_RESULTS" \
  --selection-manifest "$L2_EXTENSION" \
  --compatibility "$COMPATIBILITY" \
  --rollout-seed 20261026 --repeated-rollouts 1 --apply

python scripts/register_cbv_recovery_scope.py \
  --config configs/cbv_bench_bigcodebench_v2.yaml --dataset bigcodebench \
  --selection-manifest "$COMPLETED_L1" --level 1 \
  --results-dir "$L1_RESULTS" --compatibility "$COMPATIBILITY"

python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml validate-data \
  --dataset bigcodebench --level 1 --repeated-rollouts 1 \
  --results-dir "$L1_RESULTS" --selection-manifest "$COMPLETED_L1" --deep-privacy
python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml report \
  --dataset bigcodebench --level 1 --repeated-rollouts 1 \
  --results-dir "$L1_RESULTS" --selection-manifest "$COMPLETED_L1"
python scripts/export_cbv_dataset_safe.py --config configs/cbv_bench_bigcodebench_v2.yaml \
  --dataset bigcodebench --level 1 --repeated-rollouts 1 \
  --results-dir "$L1_RESULTS" --selection-manifest "$COMPLETED_L1" --force

python scripts/check_bigcodebench_dependencies.py "$L2_EXTENSION" --import-check
python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml estimate \
  --dataset bigcodebench --level 2 --repeated-rollouts 1 \
  --results-dir "$L2_RESULTS" --selection-manifest "$L2_EXTENSION"
python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml collect \
  --dataset bigcodebench --level 2 --repeated-rollouts 1 \
  --results-dir "$L2_RESULTS" --selection-manifest "$L2_EXTENSION"
python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml validate-data \
  --dataset bigcodebench --level 2 --repeated-rollouts 1 \
  --results-dir "$L2_RESULTS" --selection-manifest "$L2_EXTENSION" --deep-privacy
python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml report \
  --dataset bigcodebench --level 2 --repeated-rollouts 1 \
  --results-dir "$L2_RESULTS" --selection-manifest "$L2_EXTENSION"
python scripts/export_cbv_dataset_safe.py --config configs/cbv_bench_bigcodebench_v2.yaml \
  --dataset bigcodebench --level 2 --repeated-rollouts 1 \
  --results-dir "$L2_RESULTS" --selection-manifest "$L2_EXTENSION" --force

echo
echo "BigCodeBench scale recovery completed."
echo "L1 attrition audit: $ATTRITION_REPORT"
echo "L1 completed scope: $COMPLETED_L1"
echo "Natural L2 full pool: $L2_FULL"
echo "Recovery log: $LOG_DIR/recovery.log"
