#!/usr/bin/env python3
"""Run read-only DARC v0.5 seed/policy diagnostics on validation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_verify.analysis.darc_v05_diagnostics import run_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/darc_v05_offline_rl.yaml"),
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("results/darc_v05_offline_rl_budgetfix_31102026"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/darc_v05_offline_rl/diagnostics"),
    )
    args = parser.parse_args()
    report = run_diagnostics(
        args.config,
        artifact_root=args.artifacts_root,
        output=args.output,
    )
    print("DARC v0.5 diagnostics completed.")
    print(f"Report: {report.resolve()}")


if __name__ == "__main__":
    main()
