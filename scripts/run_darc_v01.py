#!/usr/bin/env python3
"""CLI for DARC v0.1 offline training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_verify.analysis.darc_experiment import run_darc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/darc_v01.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    report = run_darc(
        args.config.resolve(),
        project_root=args.project_root.resolve(),
        resume=not args.no_resume,
    )
    print(f"DARC v0.1 completed. Report: {report}")


if __name__ == "__main__":
    main()
