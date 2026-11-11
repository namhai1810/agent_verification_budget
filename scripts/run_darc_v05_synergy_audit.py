#!/usr/bin/env python3
"""Run the CBV Level-2 multi-verification synergy audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_verify.analysis.darc_v05_synergy_audit import run_synergy_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/darc_v05_offline_rl.yaml"),
    )
    parser.add_argument(
        "--decision-trace",
        type=Path,
        default=Path("results/darc_v05_offline_rl/diagnostics/tables/task_outcomes.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/darc_v05_offline_rl/synergy_audit"),
    )
    args = parser.parse_args()
    report = run_synergy_audit(
        args.config,
        decision_trace_path=args.decision_trace,
        output=args.output,
    )
    print("DARC v0.5 multi-verification synergy audit completed.")
    print(f"Report: {report.resolve()}")


if __name__ == "__main__":
    main()
