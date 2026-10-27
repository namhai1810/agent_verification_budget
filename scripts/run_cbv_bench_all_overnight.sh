#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DATE_SUFFIX="${RUN_DATE_SUFFIX:-$(date +%d%m%Y)}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
CBV_CONDA_ENV="${CBV_CONDA_ENV:-agent_verify}"
if [[ ! "$RUN_DATE_SUFFIX" =~ ^[0-9]{8}$ ]]; then
  echo "ERROR: RUN_DATE_SUFFIX must use DDMMYYYY, for example 26102026." >&2
  exit 2
fi
if [[ ! "$MIN_FREE_GB" =~ ^[0-9]+$ ]] || (( MIN_FREE_GB < 1 )); then
  echo "ERROR: MIN_FREE_GB must be a positive integer." >&2
  exit 2
fi

read -r _ _ _ available_kb _ < <(df -Pk "$PROJECT_ROOT" | tail -n 1)
required_kb=$((MIN_FREE_GB * 1024 * 1024))
if (( available_kb < required_kb )); then
  available_mb=$((available_kb / 1024))
  echo "ERROR: only ${available_mb} MiB is free on the project filesystem; at least ${MIN_FREE_GB} GiB is required." >&2
  echo "Free disk space, then resume with the same RUN_DATE_SUFFIX so compatible cached records are reused." >&2
  exit 28
fi

HE_COUNT="${HE_COUNT:-163}"
LCB_COUNT="${LCB_COUNT:-175}"
BCB_L1_COUNT="${BCB_L1_COUNT:-1000}"
BCB_L2_COUNT="${BCB_L2_COUNT:-250}"
BCB_L1_MANIFEST_OVERRIDE="${BCB_L1_MANIFEST_OVERRIDE:-}"
BCB_L2_MANIFEST_OVERRIDE="${BCB_L2_MANIFEST_OVERRIDE:-}"
BCB_L2_RESULTS_OVERRIDE="${BCB_L2_RESULTS_OVERRIDE:-}"
RUN_HE="${RUN_HE:-1}"
RUN_LCB="${RUN_LCB:-1}"
RUN_BCB="${RUN_BCB:-1}"
RUN_TESTS="${RUN_TESTS:-1}"

HE_RESULTS="results/cbv_bench_v2_${RUN_DATE_SUFFIX}"
LCB_RESULTS="results/cbv_bench_v2_livecodebench_${RUN_DATE_SUFFIX}"
BCB_RESULTS="results/cbv_bench_v2_bigcodebench_${RUN_DATE_SUFFIX}"
BCB_L2_RESULTS="${BCB_L2_RESULTS_OVERRIDE:-$BCB_RESULTS}"
OVERNIGHT_RESULTS="results/cbv_bench_overnight_${RUN_DATE_SUFFIX}"

if (( BCB_L2_COUNT > BCB_L1_COUNT )); then
  echo "ERROR: BCB_L2_COUNT must be <= BCB_L1_COUNT so Level 2 is a nested subset." >&2
  exit 2
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$PROJECT_ROOT/$OVERNIGHT_RESULTS/logs/$RUN_STAMP"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/overnight.log"
STATUS_FILE="$LOG_DIR/status.tsv"
exec > >(tee -a "$MASTER_LOG") 2>&1

current_stage="preflight"
started_epoch="$(date +%s)"

mark_status() {
  local state="$1"
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$current_stage" "$state" \
    >> "$STATUS_FILE"
}

on_error() {
  local exit_code=$?
  mark_status "FAILED(exit=$exit_code,line=${BASH_LINENO[0]:-unknown})"
  echo
  echo "CBV overnight pipeline FAILED in stage: $current_stage"
  echo "Resume with the same command after fixing the error. Compatible cache entries are retained."
  echo "Master log: $MASTER_LOG"
  exit "$exit_code"
}

on_signal() {
  mark_status "INTERRUPTED"
  echo "CBV overnight pipeline interrupted in stage: $current_stage"
  echo "Master log: $MASTER_LOG"
  exit 130
}

trap on_error ERR
trap on_signal INT TERM

run_logged() {
  local stage_name="$1"
  shift
  current_stage="$stage_name"
  mark_status "STARTED"
  echo
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START $stage_name"
  "$@" 2>&1 | tee -a "$LOG_DIR/${stage_name}.log"
  mark_status "COMPLETED"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE  $stage_name"
}

ensure_selection() {
  local config="$1"
  local dataset="$2"
  local mode="$3"
  local count="$4"
  local manifest="$5"
  if [[ -f "$manifest" ]]; then
    echo "Reusing frozen selection: $manifest"
    return 0
  fi
  run_logged "selection_${dataset}_${count}" \
    python scripts/cbv_bench.py --config "$config" prepare-selection \
      --dataset "$dataset" --mode "$mode" --task-count "$count" \
      --selection-manifest "$manifest"
}

run_collection_stage() {
  local stage="$1"
  local config="$2"
  local dataset="$3"
  local level="$4"
  local manifest="$5"
  local results_dir="$6"
  local safe_parquet_export="${7:-0}"

  run_logged "${stage}_estimate" \
    python scripts/cbv_bench.py --config "$config" estimate \
      --dataset "$dataset" --level "$level" --repeated-rollouts 1 \
      --results-dir "$results_dir" \
      --selection-manifest "$manifest"

  run_logged "${stage}_collect" \
    python scripts/cbv_bench.py --config "$config" collect \
      --dataset "$dataset" --level "$level" --repeated-rollouts 1 \
      --results-dir "$results_dir" \
      --selection-manifest "$manifest"

  if [[ "$safe_parquet_export" == "1" ]]; then
    run_logged "${stage}_validate" \
      python scripts/cbv_bench.py --config "$config" validate-data \
        --dataset "$dataset" --level "$level" --repeated-rollouts 1 \
        --results-dir "$results_dir" \
        --selection-manifest "$manifest" --deep-privacy

    run_logged "${stage}_report" \
      python scripts/cbv_bench.py --config "$config" report \
        --dataset "$dataset" --level "$level" --repeated-rollouts 1 \
        --results-dir "$results_dir" \
        --selection-manifest "$manifest"

    run_logged "${stage}_safe_export" \
      python scripts/export_cbv_dataset_safe.py --config "$config" \
        --dataset "$dataset" --level "$level" --repeated-rollouts 1 \
        --results-dir "$results_dir" \
        --selection-manifest "$manifest" --force
  else
    run_logged "${stage}_finalize" \
      python scripts/cbv_bench.py --config "$config" finalize \
        --dataset "$dataset" --level "$level" --repeated-rollouts 1 \
        --results-dir "$results_dir" \
        --selection-manifest "$manifest" --deep-privacy --force-export
  fi
}

echo "CBV-Bench overnight pipeline"
echo "Project: $PROJECT_ROOT"
echo "Date suffix: _$RUN_DATE_SUFFIX"
echo "Conda env: $CBV_CONDA_ENV"
echo "Logs:    $LOG_DIR"
available_mb=$((available_kb / 1024))
echo "Disk:    ${available_mb} MiB free (minimum ${MIN_FREE_GB} GiB)"
echo "Plan: HumanEval L2=$HE_COUNT, LiveCodeBench L2=$LCB_COUNT, " \
     "BigCodeBench L1=$BCB_L1_COUNT, BigCodeBench L2=$BCB_L2_COUNT"
if [[ "$BCB_L2_RESULTS" != "$BCB_RESULTS" ]]; then
  echo "BigCodeBench Level-2 results: $BCB_L2_RESULTS"
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CBV_CONDA_ENV" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is unavailable and $CBV_CONDA_ENV is not active." >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CBV_CONDA_ENV"
fi

export PYTHONUNBUFFERED=1
export VLLM_API_KEY="${VLLM_API_KEY:-local}"
export AGENT_VERIFY_ALLOW_SPEND="${AGENT_VERIFY_ALLOW_SPEND:-YES}"

run_logged "preflight_student_api" \
  curl --fail --silent --show-error http://127.0.0.1:8011/v1/models
run_logged "preflight_teacher_api" \
  curl --fail --silent --show-error http://127.0.0.1:8012/v1/models

if [[ "$RUN_TESTS" == "1" ]]; then
  run_logged "preflight_unit_tests" pytest -q tests/test_cbv_bench_v2.py
fi
run_logged "preflight_humaneval_config" \
  python scripts/cbv_bench.py --config configs/cbv_bench_v2.yaml validate-config
run_logged "preflight_livecodebench_config" \
  python scripts/cbv_bench.py --config configs/cbv_bench_livecodebench_v2.yaml validate-config
run_logged "preflight_bigcodebench_config" \
  python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml validate-config

if [[ "$RUN_HE" == "1" ]]; then
  HE_MANIFEST="$HE_RESULTS/selections/humaneval_existing_l2_${HE_COUNT}.json"
  ensure_selection configs/cbv_bench_v2.yaml humaneval existing "$HE_COUNT" "$HE_MANIFEST"
  run_collection_stage humaneval_l2 configs/cbv_bench_v2.yaml humaneval 2 \
    "$HE_MANIFEST" "$HE_RESULTS"
fi

if [[ "$RUN_LCB" == "1" ]]; then
  LCB_MANIFEST="$LCB_RESULTS/selections/existing_l2_${LCB_COUNT}.json"
  ensure_selection configs/cbv_bench_livecodebench_v2.yaml livecodebench existing \
    "$LCB_COUNT" "$LCB_MANIFEST"
  run_collection_stage livecodebench_l2 configs/cbv_bench_livecodebench_v2.yaml \
    livecodebench 2 "$LCB_MANIFEST" "$LCB_RESULTS"
fi

if [[ "$RUN_BCB" == "1" ]]; then
  BCB_L1_MANIFEST="${BCB_L1_MANIFEST_OVERRIDE:-$BCB_RESULTS/selections/scale_l1_${BCB_L1_COUNT}.json}"
  BCB_L2_MANIFEST="${BCB_L2_MANIFEST_OVERRIDE:-$BCB_RESULTS/selections/scale_l2_${BCB_L2_COUNT}.json}"

  if [[ -n "$BCB_L1_MANIFEST_OVERRIDE" ]]; then
    [[ -f "$BCB_L1_MANIFEST" ]] || {
      echo "ERROR: BCB_L1_MANIFEST_OVERRIDE does not exist: $BCB_L1_MANIFEST" >&2
      exit 2
    }
    echo "Using reviewed BigCodeBench Level-1 manifest: $BCB_L1_MANIFEST"
  else
    ensure_selection configs/cbv_bench_bigcodebench_v2.yaml bigcodebench all \
      "$BCB_L1_COUNT" "$BCB_L1_MANIFEST"
  fi
  if [[ -n "$BCB_L2_MANIFEST_OVERRIDE" ]]; then
    [[ -f "$BCB_L2_MANIFEST" ]] || {
      echo "ERROR: BCB_L2_MANIFEST_OVERRIDE does not exist: $BCB_L2_MANIFEST" >&2
      exit 2
    }
    echo "Using reviewed BigCodeBench Level-2 manifest: $BCB_L2_MANIFEST"
  else
    ensure_selection configs/cbv_bench_bigcodebench_v2.yaml bigcodebench all \
      "$BCB_L2_COUNT" "$BCB_L2_MANIFEST"
  fi

  run_logged "preflight_bigcodebench_dependencies" \
    python scripts/check_bigcodebench_dependencies.py \
      "$BCB_L1_MANIFEST" --import-check

  run_collection_stage bigcodebench_l1 configs/cbv_bench_bigcodebench_v2.yaml \
    bigcodebench 1 "$BCB_L1_MANIFEST" "$BCB_RESULTS" 1
  run_collection_stage bigcodebench_l2 configs/cbv_bench_bigcodebench_v2.yaml \
    bigcodebench 2 "$BCB_L2_MANIFEST" "$BCB_L2_RESULTS" 1
fi

current_stage="complete"
finished_epoch="$(date +%s)"
elapsed_seconds=$((finished_epoch - started_epoch))
mark_status "COMPLETED(elapsed_seconds=$elapsed_seconds)"

echo
echo "CBV-Bench overnight pipeline COMPLETED."
echo "Elapsed seconds: $elapsed_seconds"
echo "Master log: $MASTER_LOG"
echo "Status file: $STATUS_FILE"
echo "HumanEval outputs:      $HE_RESULTS/"
echo "LiveCodeBench outputs:  $LCB_RESULTS/"
echo "BigCodeBench outputs:   $BCB_RESULTS/"
if [[ "$BCB_L2_RESULTS" != "$BCB_RESULTS" ]]; then
  echo "BigCodeBench L2 outputs: $BCB_L2_RESULTS/"
fi
