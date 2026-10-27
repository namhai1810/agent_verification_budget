#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/cbv_bench_v2.yaml}

python scripts/cbv_bench.py --config "$CONFIG" validate-config
python scripts/cbv_bench.py --config "$CONFIG" estimate
python scripts/cbv_bench.py --config "$CONFIG" collect
python scripts/cbv_bench.py --config "$CONFIG" validate-data --deep-privacy
python scripts/cbv_bench.py --config "$CONFIG" report
python scripts/cbv_bench.py --config "$CONFIG" export
