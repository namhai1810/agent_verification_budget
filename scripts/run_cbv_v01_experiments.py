#!/usr/bin/env python3
"""CLI for the frozen CBV-Bench v0.1 overnight experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_verify.analysis.cbv_v01_experiment import run_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", type=Path, default=Path("data/cbv_bench_v01_frozen_28102026"))
    parser.add_argument("--output", type=Path, default=Path("results/cbv_v01_experiments_29102026"))
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    report = run_all(
        args.frozen_root.resolve(), args.output.resolve(), seed=args.seed,
        bootstrap=args.bootstrap, resume=not args.no_resume,
    )
    print(f"Completed. Report: {report}")


if __name__ == "__main__":
    main()

