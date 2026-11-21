#!/usr/bin/env python3
"""Run DARC Full-Tree preflight, smoke, cross-validation, or final fit."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_verify.analysis.darc_fulltree_experiment import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/darc_qwen38_fulltree.yaml")
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--crossval", action="store_true")
    modes.add_argument("--final-fit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    mode = (
        "dry-run"
        if args.dry_run
        else "smoke"
        if args.smoke
        else "crossval"
        if args.crossval
        else "final-fit"
    )
    output = run(args.config, mode=mode, resume=args.resume)
    print(f"Completed: {output.resolve()}")


if __name__ == "__main__":
    main()
