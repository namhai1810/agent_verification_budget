#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DATE_SUFFIX="${RUN_DATE_SUFFIX:-28102026}"
BASE_DATE_SUFFIX="${BASE_DATE_SUFFIX:-26102026}"
CBV_CONDA_ENV="${CBV_CONDA_ENV:-agent_verify_bcb}"
OUTPUT_DIR="results/cbv_bench_all_${RUN_DATE_SUFFIX}"
LOG_DIR="results/cbv_bench_overnight_${RUN_DATE_SUFFIX}/logs/finalize_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/finalize.log") 2>&1

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CBV_CONDA_ENV" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CBV_CONDA_ENV"
fi

export PYTHONUNBUFFERED=1
export VLLM_API_KEY="${VLLM_API_KEY:-local}"
export AGENT_VERIFY_ALLOW_SPEND="${AGENT_VERIFY_ALLOW_SPEND:-YES}"

echo "[1/3] Repair seed drift and complete BigCodeBench Natural L2"
RUN_DATE_SUFFIX="$RUN_DATE_SUFFIX" \
CBV_CONDA_ENV="$CBV_CONDA_ENV" \
  bash scripts/run_bigcodebench_scale_recovery.sh

echo
echo "[2/3] Build cross-dataset summary and visualizations"
python scripts/report_cbv_bench_all.py \
  --base-date-suffix "$BASE_DATE_SUFFIX" \
  --scale-date-suffix "$RUN_DATE_SUFFIX" \
  --output-dir "$OUTPUT_DIR"

echo
echo "[3/3] Verify final artifacts"
required=(
  "$OUTPUT_DIR/CBV_Bench_all_report.md"
  "$OUTPUT_DIR/summary.json"
  "$OUTPUT_DIR/collection_summary.csv"
  "$OUTPUT_DIR/sequence_accuracy_paired.csv"
  "$OUTPUT_DIR/checkpoint_effects.csv"
  "$OUTPUT_DIR/dataset_level_coverage.csv"
  "$OUTPUT_DIR/sequence_inference.csv"
  "$OUTPUT_DIR/checkpoint_inference.csv"
  "$OUTPUT_DIR/baseline_correctness_analysis.csv"
  "$OUTPUT_DIR/difficulty_analysis.csv"
  "$OUTPUT_DIR/interaction_analysis.csv"
  "$OUTPUT_DIR/strict_pareto_frontier.csv"
  "$OUTPUT_DIR/figures/cbv_bench_dashboard.png"
  "$OUTPUT_DIR/figures/dataset_level_coverage.png"
  "$OUTPUT_DIR/figures/accuracy_by_sequence.png"
  "$OUTPUT_DIR/figures/accuracy_cost_frontier.png"
  "$OUTPUT_DIR/figures/statistical_validation.png"
  "$OUTPUT_DIR/figures/strict_pareto_frontier.png"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || {
    echo "ERROR: missing or empty final artifact: $path" >&2
    exit 2
  }
done

echo
echo "CBV-Bench finalization completed."
echo "Report:    $PROJECT_ROOT/$OUTPUT_DIR/CBV_Bench_all_report.md"
echo "Summary:   $PROJECT_ROOT/$OUTPUT_DIR/summary.json"
echo "Dashboard: $PROJECT_ROOT/$OUTPUT_DIR/figures/cbv_bench_dashboard.png"
echo "Log:       $PROJECT_ROOT/$LOG_DIR/finalize.log"
