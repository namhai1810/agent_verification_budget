"""Command-line interface for Phase-3 LiveCodeBench collection and analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_verify.analysis.livecodebench import analyze_livecodebench
from agent_verify.config import AppConfig, load_config
from agent_verify.experiments.livecodebench import LiveCodeBenchExperiment
from agent_verify.livecodebench_dataset import prepare_locked_selection
from agent_verify.logging import JsonlStore


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Download v6 metadata/tests and lock 100 task IDs")
    prepare.add_argument("--results-dir", type=Path, default=Path("results/livecodebench"))
    prepare.add_argument("--task-count", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--force", action="store_true")

    run = commands.add_parser("run", help="Run the frozen counterfactual experiment")
    run.add_argument("--config", default="configs/livecodebench_vllm_gpu2.yaml")
    run.add_argument("--task-count", type=int)
    run.add_argument("--task-offset", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--repeated-pairs", type=int)
    run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument(
        "--resume-run-id",
        help="Explicitly continue an audited earlier run after a compatible code-only recovery fix",
    )
    run.add_argument("--student-endpoint")
    run.add_argument("--teacher-endpoint")

    analyze = commands.add_parser("analyze", help="Regenerate LiveCodeBench H1 reports")
    analyze.add_argument("--results-dir", type=Path, default=Path("results/livecodebench"))
    analyze.add_argument("--run-id")
    analyze.add_argument("--humaneval-results", type=Path, default=Path("results"))
    return root


def override_config(args: argparse.Namespace) -> AppConfig:
    config = load_config(args.config)
    updates = {}
    for key in ("task_count", "task_offset", "seed", "repeated_pairs", "resume"):
        value = getattr(args, key, None)
        if value is not None:
            updates[key] = value
    experiment = config.experiment.model_copy(update=updates)
    models = config.models
    if args.student_endpoint:
        role_updates = {
            role: getattr(models, role).model_copy(update={"base_url": args.student_endpoint})
            for role in ("planner", "coder", "reviewer")
        }
        models = models.model_copy(update=role_updates)
    if args.teacher_endpoint:
        models = models.model_copy(
            update={"teacher": models.teacher.model_copy(update={"base_url": args.teacher_endpoint})}
        )
    return AppConfig.model_validate(
        {
            **config.model_dump(mode="python"),
            "experiment": experiment.model_dump(mode="python"),
            "models": models.model_dump(mode="python"),
        }
    )


def _latest_run_id(results_dir: Path) -> str:
    manifests = JsonlStore(results_dir / "manifests.jsonl").read_all()
    if not manifests:
        raise ValueError("No LiveCodeBench manifest found; pass --run-id after running an experiment")
    return str(manifests[-1]["run_id"])


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare_locked_selection(
                args.results_dir.resolve(),
                task_count=args.task_count,
                seed=args.seed,
                force=args.force,
            )
            print(
                json.dumps(
                    {
                        "task_count": manifest["task_count"],
                        "eligible_count": manifest["eligible_count"],
                        "selected_difficulty": manifest["selected_difficulty"],
                        "selected_platform": manifest["selected_platform"],
                        "selection_sha256": manifest["selection_sha256"],
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "run":
            config = override_config(args)
            experiment = LiveCodeBenchExperiment(config, resume_run_id=args.resume_run_id)
            run_id = experiment.run()
            print(f"Completed LiveCodeBench run: {run_id}")
            print(f"Report: {experiment.results_dir / 'h1_analysis.md'}")
            return 0
        if args.command == "analyze":
            results_dir = args.results_dir.resolve()
            run_id = args.run_id or _latest_run_id(results_dir)
            analyze_livecodebench(results_dir, run_id, humaneval_results=args.humaneval_results.resolve())
            print(f"Regenerated: {results_dir / 'h1_analysis.md'}")
            return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
