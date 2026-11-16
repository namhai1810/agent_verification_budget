#!/usr/bin/env python3
"""Run DARC v0.5 Myopic-i0 with the expanded 531-task training cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_verify.analysis.darc_v05_531_scaling import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/darc_v05_531_scaling.yaml"),
    )
    parser.add_argument("--smoke", action="store_true", help="Run data/integrity checks only.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse atomically completed per-seed checkpoints and reconstruct evaluation outputs.",
    )
    args = parser.parse_args()
    report = run(args.config, smoke=args.smoke, resume=args.resume)
    print(f"Completed: {report.resolve()}")


if __name__ == "__main__":
    main()
