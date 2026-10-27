"""Command-line interface for planning, collection, validation, export, and reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_verify.cbv_bench.collector import CBVCollector
from agent_verify.cbv_bench.config import CBVConfig, load_cbv_config
from agent_verify.cbv_bench.datasets import (
    discover_collected_task_ids,
    prepare_selection_manifest,
)
from agent_verify.cbv_bench.estimator import estimate_collection
from agent_verify.cbv_bench.export import export_cbv_dataset
from agent_verify.cbv_bench.model_metadata import build_model_pair_snapshot
from agent_verify.cbv_bench.reporting import generate_report
from agent_verify.cbv_bench.validation import validate_cbv_data


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="cbv-bench", description="CBV-Bench V2 causal dataset generator")
    root.add_argument("--config", default="configs/cbv_bench_v2.yaml")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-config")
    prepare = commands.add_parser("prepare-selection")
    prepare.add_argument("--dataset", choices=("humaneval", "livecodebench", "bigcodebench"), required=True)
    prepare.add_argument("--mode", choices=("existing", "new", "all"), default="all")
    prepare.add_argument("--task-count", type=int)
    prepare.add_argument("--task-offset", type=int)
    prepare.add_argument("--selection-manifest")
    prepare.add_argument("--force", action="store_true")
    runtime_commands = {}
    for name in ("estimate", "collect", "validate-data", "report", "export", "finalize"):
        command = commands.add_parser(name)
        runtime_commands[name] = command
        command.add_argument("--dataset", choices=("humaneval", "livecodebench", "bigcodebench"))
        command.add_argument("--selection-manifest")
        command.add_argument("--level", type=int, choices=(1, 2))
        command.add_argument("--repeated-rollouts", type=int)
        command.add_argument("--results-dir")
        command.add_argument("--max-teacher-calls", type=int)
        command.add_argument("--max-teacher-tokens", type=int)
    runtime_commands["validate-data"].add_argument("--deep-privacy", action="store_true")
    runtime_commands["export"].add_argument("--force", action="store_true")
    runtime_commands["finalize"].add_argument("--deep-privacy", action="store_true")
    runtime_commands["finalize"].add_argument("--force-export", action="store_true")
    return root


def _override(config: CBVConfig, args: argparse.Namespace) -> CBVConfig:
    collection_updates = {}
    for arg, field in (("level", "level"), ("repeated_rollouts", "repeated_rollouts"),
                       ("max_teacher_calls", "max_teacher_calls"),
                       ("max_teacher_tokens", "max_teacher_tokens")):
        value = getattr(args, arg, None)
        if value is not None:
            collection_updates[field] = value
    if getattr(args, "results_dir", None):
        collection_updates["results_dir"] = Path(args.results_dir)
    collection = config.collection.model_copy(update=collection_updates)
    selected_dataset = getattr(args, "dataset", None)
    manifest_override = getattr(args, "selection_manifest", None)
    datasets = []
    enabled_names = [item.name for item in config.datasets if item.enabled]
    manifest_target = selected_dataset or (enabled_names[0] if len(enabled_names) == 1 else None)
    for item in config.datasets:
        updates = {}
        if selected_dataset:
            updates["enabled"] = item.name == selected_dataset
        if manifest_override and item.name == manifest_target:
            updates["selection_manifest"] = Path(manifest_override)
        datasets.append(item.model_copy(update=updates) if updates else item)
    return CBVConfig.model_validate({**config.model_dump(mode="python"),
                                     "collection": collection.model_dump(mode="python"),
                                     "datasets": [item.model_dump(mode="python") for item in datasets]})


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_cbv_config(args.config)
        if args.command == "validate-config":
            pair = build_model_pair_snapshot(config)
            print(json.dumps({
                "valid": True,
                "config_fingerprint": config.fingerprint,
                "model_pair_id": pair.model_pair_id,
                "model_pair_fingerprint": pair.model_pair_fingerprint,
                "student_model": pair.student_model.model_id,
                "teacher_model": pair.teacher_model.model_id,
                "prompt_version": pair.prompt_version,
                "inference_config": pair.inference_config,
            }, indent=2))
            return 0
        if args.command == "prepare-selection":
            spec = next(item for item in config.datasets if item.name == args.dataset)
            updates = {}
            if args.task_count is not None:
                updates["task_count"] = args.task_count
            if args.task_offset is not None:
                updates["task_offset"] = args.task_offset
            if args.selection_manifest:
                updates["selection_manifest"] = Path(args.selection_manifest)
            spec = spec.model_copy(update=updates)
            known_paths = [
                path for item in config.datasets for path in item.legacy_result_dirs
            ]
            known = discover_collected_task_ids(known_paths, Path.cwd())
            payload = prepare_selection_manifest(config, spec, Path.cwd(), mode=args.mode,
                                                 known_task_ids=known, force=args.force)
            print(json.dumps({"selection_manifest": str(spec.selection_manifest),
                              "tasks": len(payload["task_records"]),
                              "selection_sha256": payload["selection_sha256"]}, indent=2))
            return 0
        config = _override(config, args)
        if args.command == "estimate":
            print(json.dumps(estimate_collection(config), indent=2))
            return 0
        if args.command == "collect":
            collector = CBVCollector(config)
            run_id = collector.collect()
            print(f"Collection complete: {run_id}")
            return 0
        if args.command == "validate-data":
            report = validate_cbv_data(config, deep_privacy=args.deep_privacy)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 2
        if args.command == "report":
            print(generate_report(config))
            return 0
        if args.command == "export":
            print(export_cbv_dataset(config, force=args.force))
            return 0
        if args.command == "finalize":
            validation = validate_cbv_data(config, deep_privacy=args.deep_privacy)
            if not validation["valid"]:
                print(json.dumps(validation, indent=2), file=sys.stderr)
                return 2
            report = generate_report(config)
            export = export_cbv_dataset(config, force=args.force_export)
            print(f"Report: {report}\nExport: {export}")
            return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
