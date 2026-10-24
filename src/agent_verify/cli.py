from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_verify.analysis.analyze_h1 import analyze_h1
from agent_verify.config import load_config
from agent_verify.evaluation.humaneval_runner import HumanEvalRunner, SandboxUnavailable
from agent_verify.experiments.phase1 import Phase1Experiment
from agent_verify.logging import JsonlStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-verify",
        description="Phase-1 downstream-aware verification experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate and fingerprint a config")
    validate.add_argument("--config", default="configs/default.yaml")

    sandbox = subparsers.add_parser("check-sandbox", help="Verify Bubblewrap isolation")
    sandbox.add_argument("--config", default="configs/default.yaml")

    run = subparsers.add_parser("run", help="Run baseline, paired interventions, and H1 analysis")
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--task-offset", type=int)
    run.add_argument("--task-count", type=int)
    run.add_argument("--repeated-pairs", type=int)
    run.add_argument("--results-dir")

    analyze = subparsers.add_parser("analyze", help="Regenerate H1 outputs from append-only logs")
    analyze.add_argument("--results-dir", default="results")
    analyze.add_argument("--run-id")
    return parser


def _overridden_config(args):
    config = load_config(args.config)
    updates = {}
    if getattr(args, "task_offset", None) is not None:
        updates["task_offset"] = args.task_offset
    if getattr(args, "task_count", None) is not None:
        updates["task_count"] = args.task_count
    if getattr(args, "repeated_pairs", None) is not None:
        updates["repeated_pairs"] = args.repeated_pairs
    if getattr(args, "results_dir", None) is not None:
        updates["results_dir"] = Path(args.results_dir)
    if updates:
        experiment = config.experiment.model_copy(update=updates)
        config = type(config).model_validate(
            {**config.model_dump(mode="python"), "experiment": experiment.model_dump(mode="python")}
        )
    return config


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_config(args.config)
            print(f"Config valid. fingerprint={config.fingerprint}")
            return 0
        if args.command == "check-sandbox":
            config = load_config(args.config)
            log = JsonlStore(Path("/tmp") / "agent-verify-sandbox-probe.jsonl")
            HumanEvalRunner(config.evaluation, log).check_sandbox()
            print("Landlock + seccomp sandbox is available; filesystem/network/process escape is denied.")
            return 0
        if args.command == "run":
            config = _overridden_config(args)
            experiment = Phase1Experiment(config)
            run_id = experiment.run()
            print(f"Completed run: {run_id}")
            print(f"Report: {experiment.results_dir / 'summary.md'}")
            return 0
        if args.command == "analyze":
            results_dir = Path(args.results_dir).resolve()
            run_id = args.run_id
            if not run_id:
                manifests = JsonlStore(results_dir / "manifests.jsonl").read_all()
                if not manifests:
                    raise ValueError("No manifest exists; pass --run-id or run an experiment first")
                run_id = manifests[-1]["run_id"]
            analyze_h1(results_dir, run_id)
            print(f"Regenerated report for {run_id}: {results_dir / 'summary.md'}")
            return 0
    except SandboxUnavailable as exc:
        print(f"SAFE-SANDBOX BLOCKER: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
