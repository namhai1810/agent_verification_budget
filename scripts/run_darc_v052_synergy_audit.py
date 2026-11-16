#!/usr/bin/env python3
"""Run the expanded 531-task train-only CBV Level-2 synergy audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_verify.analysis.darc_v052_synergy_audit import run_expanded_synergy_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=Path("data/cbv_bench_v01_frozen_28102026"),
    )
    parser.add_argument(
        "--extension-root",
        type=Path,
        default=Path("results/cbv_train_l2_extension_12112026"),
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path(
            "results/cbv_train_l2_extension_12112026/selection/bigcodebench_train_l2_extension.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/darc_v052_synergy_audit"),
    )
    args = parser.parse_args()
    report = run_expanded_synergy_audit(
        frozen_root=args.frozen_root,
        extension_root=args.extension_root,
        selection_manifest=args.selection_manifest,
        output=args.output,
    )
    print("DARC v0.5.2 expanded synergy audit completed.")
    print(f"Report: {report.resolve()}")


if __name__ == "__main__":
    main()
