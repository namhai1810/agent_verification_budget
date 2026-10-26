"""CLI for the C3 learned sequential budget controller."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_verify.analysis.c3_experiment import (
    evaluate_controllers,
    prepare_c3,
    train_controller_mode,
)
from agent_verify.analysis.c3_reporting import (
    generate_c3_report,
    generate_c3_visualizations,
)
from agent_verify.analysis.joint_value_cost_data import load_config
from agent_verify.analysis.joint_value_cost_experiment import TRAINING_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/c3_sequential_controller.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    train = subparsers.add_parser("train")
    train.add_argument("--mode", choices=sorted(TRAINING_DATASETS), required=True)
    train.add_argument("--resume", action="store_true")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("visualize")
    subparsers.add_parser("report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = args.output_root or Path(config["experiment"]["output_dir"])
    if args.command == "prepare":
        prepare_c3(config, output_root=output_root)
    elif args.command == "train":
        train_controller_mode(
            config,
            mode=args.mode,
            output_root=output_root,
            resume=args.resume,
        )
    elif args.command == "evaluate":
        evaluate_controllers(config, output_root=output_root)
    elif args.command == "visualize":
        generate_c3_visualizations(output_root)
    elif args.command == "report":
        print(generate_c3_report(config, output_root).resolve())
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
