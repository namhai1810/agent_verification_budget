"""Command-line interface for offline Joint Value-Cost research experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_verify.analysis.joint_value_cost_budget import run_budget_evaluation
from agent_verify.analysis.joint_value_cost_data import (
    load_config,
    load_frozen_split,
    load_joint_examples,
)
from agent_verify.analysis.joint_value_cost_experiment import (
    TRAINING_DATASETS,
    evaluate_all,
    prepare_data,
    train_mode,
)
from agent_verify.analysis.joint_value_cost_reporting import (
    generate_report,
    generate_visualizations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/joint_value_cost.yaml"))
    parser.add_argument("--output-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Validate logs, token accounting, and frozen split")
    train = subparsers.add_parser("train", help="Select and train Joint and Separate models")
    train.add_argument("--mode", choices=sorted(TRAINING_DATASETS), required=True)
    train.add_argument("--resume", action="store_true")
    subparsers.add_parser("evaluate", help="Evaluate all train modes in- and cross-domain")
    subparsers.add_parser("budget-eval", help="Run token-budgeted paired-outcome replay")
    subparsers.add_parser("visualize", help="Generate PNG and PDF research figures")
    subparsers.add_parser("report", help="Generate a new Markdown research report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = args.output_root or Path(config["experiment"]["output_dir"])
    if args.command == "prepare":
        prepare_data(config, output_dir=output_root / "data")
    elif args.command == "train":
        train_mode(
            config,
            mode=args.mode,
            output_root=output_root,
            resume=args.resume,
        )
    elif args.command == "evaluate":
        evaluate_all(config, output_root=output_root)
    elif args.command == "budget-eval":
        examples = load_joint_examples(config)
        split = load_frozen_split(config, examples)
        run_budget_evaluation(
            config,
            examples=examples,
            split=split,
            output_root=output_root,
        )
    elif args.command == "visualize":
        generate_visualizations(output_root)
    elif args.command == "report":
        report = generate_report(config, output_root)
        print(report.resolve())
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
