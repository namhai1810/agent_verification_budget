#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/cbv_bench_bigcodebench_train_l2_extension.yaml"
OUTPUT_DIR="results/cbv_train_l2_extension_12112026"
TASK_COUNT=150
SELECTION_SEED=20261111
MIN_FREE_GB=5
DRY_RUN=0
CBV_CONDA_ENV="${CBV_CONDA_ENV:-agent_verify_bcb}"

usage() {
  echo "Usage: $0 [--dry-run] [--resume] [--output-dir PATH] [--task-count N] [--seed N] [--min-free-gb N]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --resume) shift ;; # Resume is always enabled; retained for an explicit, readable command.
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --task-count) TASK_COUNT="$2"; shift 2 ;;
    --seed) SELECTION_SEED="$2"; shift 2 ;;
    --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$TASK_COUNT" =~ ^[0-9]+$ ]] || (( TASK_COUNT < 1 || TASK_COUNT > 150 )); then
  echo "ERROR: --task-count must be an integer in [1, 150]." >&2
  exit 2
fi
if [[ ! "$SELECTION_SEED" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --seed must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "$MIN_FREE_GB" =~ ^[0-9]+$ ]] || (( MIN_FREE_GB < 1 )); then
  echo "ERROR: --min-free-gb must be a positive integer." >&2
  exit 2
fi

case "$(realpath -m "$OUTPUT_DIR")" in
  "$PROJECT_ROOT/results/cbv_bench_v2_bigcodebench_26102026"|\
  "$PROJECT_ROOT/results/cbv_bench_v2_bigcodebench_28102026"|\
  "$PROJECT_ROOT/results/cbv_bench_v2_bigcodebench_l2_recovery_26102026"|\
  "$PROJECT_ROOT/results/cbv_bench_v2_bigcodebench_natural_l2_extension_28102026")
    echo "ERROR: output directory must not overwrite an existing CBV source." >&2
    exit 2
    ;;
esac

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CBV_CONDA_ENV" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: activate conda environment '$CBV_CONDA_ENV' first." >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CBV_CONDA_ENV"
fi

export PYTHONUNBUFFERED=1
export VLLM_API_KEY="${VLLM_API_KEY:-local}"
export AGENT_VERIFY_ALLOW_SPEND="${AGENT_VERIFY_ALLOW_SPEND:-YES}"

mkdir -p "$OUTPUT_DIR/logs"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER_LOG="$OUTPUT_DIR/logs/run_${RUN_STAMP}.log"
STATUS_FILE="$OUTPUT_DIR/logs/status.tsv"
ln -sfn "$(basename "$MASTER_LOG")" "$OUTPUT_DIR/logs/latest.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

current_stage="startup"
mark_status() {
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$current_stage" "$1" >> "$STATUS_FILE"
}
on_error() {
  local code=$?
  mark_status "FAILED(exit=$code,line=${BASH_LINENO[0]:-unknown})"
  echo "FAILED stage: $current_stage"
  echo "Resume safely with the identical command and --resume; immutable completed cache entries are retained."
  echo "Log: $MASTER_LOG"
  exit "$code"
}
on_signal() {
  mark_status "INTERRUPTED"
  echo "Interrupted stage: $current_stage"
  echo "Resume safely with the identical command and --resume."
  exit 130
}
trap on_error ERR
trap on_signal INT TERM

run_stage() {
  current_stage="$1"
  shift
  mark_status STARTED
  echo
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START $current_stage"
  "$@" 2>&1 | tee -a "$OUTPUT_DIR/logs/${current_stage}.log"
  mark_status COMPLETED
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE  $current_stage"
}

read -r _ _ _ available_kb _ < <(df -Pk "$PROJECT_ROOT" | tail -n 1)
required_kb=$((MIN_FREE_GB * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "ERROR: insufficient disk: $((available_kb / 1024)) MiB free; ${MIN_FREE_GB} GiB required." >&2
  exit 28
fi

SELECTION_MANIFEST="$OUTPUT_DIR/selection/bigcodebench_train_l2_extension.json"
COLLECTION_DIR="$OUTPUT_DIR/collection"

echo "CBV-Bench train-only Level-2 extension"
echo "Mode:       $([[ $DRY_RUN -eq 1 ]] && echo DRY-RUN || echo FULL-COLLECTION)"
echo "Tasks:      $TASK_COUNT"
echo "Seed:       $SELECTION_SEED"
echo "Conda env:  $CBV_CONDA_ENV"
echo "Output:     $OUTPUT_DIR"
echo "Collection: $COLLECTION_DIR"
echo "Master log: $MASTER_LOG"
echo "No validation/frozen-test outcome file is read by selection."

run_stage validate_config \
  python scripts/cbv_bench.py --config "$CONFIG" validate-config

run_stage prepare_selection \
  python scripts/prepare_cbv_train_l2_extension.py prepare \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --max-tasks "$TASK_COUNT" \
    --seed "$SELECTION_SEED"

run_stage audit_level1_reuse \
  python scripts/prepare_cbv_train_l2_extension.py audit-reuse \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR"

run_stage preflight_dependencies \
  python scripts/check_bigcodebench_dependencies.py "$SELECTION_MANIFEST" --import-check

run_stage lightweight_tests \
  pytest -q tests/test_cbv_train_l2_extension.py

run_stage cold_upper_bound_estimate \
  python scripts/cbv_bench.py --config "$CONFIG" estimate \
    --dataset bigcodebench --level 2 --repeated-rollouts 1 \
    --selection-manifest "$SELECTION_MANIFEST" \
    --results-dir "$COLLECTION_DIR"

if (( DRY_RUN == 1 )); then
  current_stage="dry_run_complete"
  mark_status COMPLETED
  echo
  echo "Dry-run completed. No Student/Teacher request was made."
  echo "Plan:      $OUTPUT_DIR/extension_plan.md"
  echo "Selection: $SELECTION_MANIFEST"
  echo "Estimate:  $OUTPUT_DIR/estimates/incremental_level2_estimate.json"
  exit 0
fi

run_stage preflight_student_api \
  curl --fail --silent --show-error http://127.0.0.1:8011/v1/models
run_stage preflight_teacher_api \
  curl --fail --silent --show-error http://127.0.0.1:8012/v1/models

run_stage collect_level2 \
  python scripts/cbv_bench.py --config "$CONFIG" collect \
    --dataset bigcodebench --level 2 --repeated-rollouts 1 \
    --selection-manifest "$SELECTION_MANIFEST" \
    --results-dir "$COLLECTION_DIR"

run_stage standard_integrity_validation \
  python scripts/cbv_bench.py --config "$CONFIG" validate-data \
    --dataset bigcodebench --level 2 --repeated-rollouts 1 \
    --selection-manifest "$SELECTION_MANIFEST" \
    --results-dir "$COLLECTION_DIR" --deep-privacy

run_stage extension_scope_validation \
  python scripts/prepare_cbv_train_l2_extension.py validate \
    --output-dir "$OUTPUT_DIR"

run_stage generate_report \
  python scripts/cbv_bench.py --config "$CONFIG" report \
    --dataset bigcodebench --level 2 --repeated-rollouts 1 \
    --selection-manifest "$SELECTION_MANIFEST" \
    --results-dir "$COLLECTION_DIR"

run_stage safe_export \
  python scripts/export_cbv_dataset_safe.py --config "$CONFIG" \
    --dataset bigcodebench --level 2 --repeated-rollouts 1 \
    --selection-manifest "$SELECTION_MANIFEST" \
    --results-dir "$COLLECTION_DIR" --force

current_stage="complete"
mark_status COMPLETED
echo
echo "CBV train Level-2 extension completed."
echo "Plan:       $OUTPUT_DIR/extension_plan.md"
echo "Integrity:  $OUTPUT_DIR/integrity/extension_integrity.md"
echo "Collection: $COLLECTION_DIR"
echo "Log:        $MASTER_LOG"
