#!/usr/bin/env python3
"""CLI entry point for DARC v0.5 offline RL."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_verify.analysis.darc_v05_experiment import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/darc_v05_offline_rl.yaml"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny synthetic CPU contract test; never load CBV data.",
    )
    args = parser.parse_args()
    report = run(args.config, smoke=args.smoke)
    print(f"DARC v0.5 {'smoke' if args.smoke else 'experiment'} completed.")
    print(f"Report: {report.resolve()}")


if __name__ == "__main__":
    main()

