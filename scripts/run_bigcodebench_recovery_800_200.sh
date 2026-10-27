#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Keep the original suffix because the compatible 768 Level-1 trajectories live
# in this frozen result namespace. Level 2 uses a clean directory to avoid
# duplicate task/seed identities while retaining a stratified 17/4/4 split.
export RUN_DATE_SUFFIX="${RUN_DATE_SUFFIX:-26102026}"
export CBV_CONDA_ENV="${CBV_CONDA_ENV:-agent_verify_bcb}"
export RUN_HE=0
export RUN_LCB=0
export RUN_BCB=1
export RUN_TESTS="${RUN_TESTS:-1}"
export BCB_L1_COUNT=200
export BCB_L2_COUNT=25
export BCB_L1_MANIFEST_OVERRIDE="results/cbv_bench_v2_bigcodebench_26102026/selections/recovery_l1_200_v2.json"
export BCB_L2_MANIFEST_OVERRIDE="results/cbv_bench_v2_bigcodebench_26102026/selections/recovery_l2_25_v2.json"
export BCB_L2_RESULTS_OVERRIDE="results/cbv_bench_v2_bigcodebench_l2_recovery_26102026"

for required in "$BCB_L1_MANIFEST_OVERRIDE" "$BCB_L2_MANIFEST_OVERRIDE"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: required recovery manifest is missing: $required" >&2
    exit 2
  fi
done

exec bash scripts/run_cbv_bench_all_overnight.sh
