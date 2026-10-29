#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DATE_SUFFIX="${RUN_DATE_SUFFIX:-$(date +%d%m%Y)}"
CBV_CONDA_ENV="${CBV_CONDA_ENV:-agent_verify_bcb}"
BASE_RESULTS="results/cbv_bench_v2_bigcodebench_26102026"
SCALE_DIR="results/cbv_bench_scale_${RUN_DATE_SUFFIX}"
UNIVERSE="$SCALE_DIR/selections/bigcodebench_all_1140.json"
L1_EXTENSION="$SCALE_DIR/selections/level1_extension.json"
L2_NATURAL_FULL="$SCALE_DIR/selections/level2_natural_full_250.json"
L2_EXTENSION="$SCALE_DIR/selections/level2_natural_extension_225.json"
SELECTION_REPORT="$SCALE_DIR/selections/extension_report.json"
COMPATIBILITY="5d1a0d2f2b0a767df2fda1d34b634236ae97916db93ddd20b2753288ef3589ad"

if [[ ! "$RUN_DATE_SUFFIX" =~ ^[0-9]{8}$ ]]; then
  echo "ERROR: RUN_DATE_SUFFIX must use DDMMYYYY." >&2
  exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CBV_CONDA_ENV" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CBV_CONDA_ENV"
fi

mkdir -p "$SCALE_DIR/selections"
if [[ ! -f "$UNIVERSE" ]]; then
  python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml \
    prepare-selection --dataset bigcodebench --mode all --task-count 1140 \
    --selection-manifest "$UNIVERSE"
fi

python scripts/build_bigcodebench_scale_extension.py \
  --universe "$UNIVERSE" \
  --completed-level1 "$BASE_RESULTS/selections/recovery_l1_200_v2.json" \
  --frozen-natural-level2 "$BASE_RESULTS/selections/scale_l2_250.json" \
  --completed-level2 "$BASE_RESULTS/selections/recovery_l2_25_v2.json" \
  --errors "$BASE_RESULTS/errors.jsonl" \
  --compatibility "$COMPATIBILITY" \
  --output-level1-extension "$L1_EXTENSION" \
  --output-level2-full "$L2_NATURAL_FULL" \
  --output-level2-extension "$L2_EXTENSION" \
  --report "$SELECTION_REPORT"

L1_COUNT="$(jq '.task_records | length' "$L1_EXTENSION")"
L2_COUNT="$(jq '.task_records | length' "$L2_EXTENSION")"

echo
echo "Phase A plan:"
echo "  Level 1 extension: $L1_COUNT tasks"
echo "  Natural Level 2 extension: $L2_COUNT tasks"
echo "  Selection audit: $SELECTION_REPORT"
echo

export RUN_DATE_SUFFIX
export CBV_CONDA_ENV
export RUN_HE=0
export RUN_LCB=0
export RUN_BCB=1
export RUN_TESTS="${RUN_TESTS:-0}"
export BCB_L1_COUNT="$L1_COUNT"
export BCB_L2_COUNT="$L2_COUNT"
export BCB_L1_MANIFEST_OVERRIDE="$L1_EXTENSION"
export BCB_L2_MANIFEST_OVERRIDE="$L2_EXTENSION"
export BCB_L2_RESULTS_OVERRIDE="results/cbv_bench_v2_bigcodebench_natural_l2_extension_${RUN_DATE_SUFFIX}"

exec bash scripts/run_cbv_bench_all_overnight.sh
