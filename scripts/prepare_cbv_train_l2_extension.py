#!/usr/bin/env python3
"""Prepare or validate the outcome-blind BigCodeBench train Level-2 extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_verify.cbv_bench.config import load_cbv_config
from agent_verify.cbv_bench.datasets import project_path
from agent_verify.cbv_bench.legacy import audit_native_level1_coverage
from agent_verify.cbv_bench.train_l2_extension import (
    ExtensionPlanningError,
    prepare_extension,
    validate_extension_output,
)


DEFAULT_SPLIT = Path("data/cbv_bench_v01_frozen_28102026/task_split_manifest.csv")
DEFAULT_UNIVERSE = Path(
    "results/cbv_bench_scale_28102026/selections/bigcodebench_all_1140.json"
)
DEFAULT_LEVEL2 = Path(
    "results/cbv_bench_scale_28102026/selections/level2_natural_full_250_recovered.json"
)
DEFAULT_EXPLORATORY = Path(
    "results/cbv_bench_v2_bigcodebench/selections/new_pilot15.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--max-tasks", type=int, default=150)
    prepare.add_argument("--seed", type=int, default=20261111)
    prepare.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    prepare.add_argument("--universe-manifest", type=Path, default=DEFAULT_UNIVERSE)
    prepare.add_argument(
        "--existing-level2-manifest", type=Path, action="append", default=[]
    )
    prepare.add_argument(
        "--exploratory-manifest", type=Path, action="append", default=[]
    )
    validate = subcommands.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    validate.add_argument(
        "--exploratory-manifest", type=Path, action="append", default=[]
    )
    reuse = subcommands.add_parser("audit-reuse")
    reuse.add_argument("--config", type=Path, required=True)
    reuse.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _paths_or_default(values: list[Path], default: Path) -> list[Path]:
    return values or ([default] if default.exists() else [])


def main() -> int:
    args = parse_args()
    try:
        if args.command == "prepare":
            result = prepare_extension(
                config=load_cbv_config(args.config),
                output_root=args.output_dir,
                split_manifest=args.split_manifest,
                universe_manifest=args.universe_manifest,
                existing_level2_manifests=_paths_or_default(
                    args.existing_level2_manifest, DEFAULT_LEVEL2
                ),
                exploratory_manifests=_paths_or_default(
                    args.exploratory_manifest, DEFAULT_EXPLORATORY
                ),
                max_tasks=args.max_tasks,
                seed=args.seed,
            )
            paths = result["paths"]
            print(json.dumps({
                "selected_tasks": result["audit"]["selected_tasks"],
                "eligible_before_cap": result["audit"]["eligible_before_cap"],
                "selection_manifest": str(paths.selection_manifest),
                "selection_audit": str(paths.audit_json),
                "estimate": result["estimate"],
                "report": str(paths.report_md),
            }, indent=2))
            return 0
        if args.command == "audit-reuse":
            config = load_cbv_config(args.config)
            manifest_path = args.output_dir / "selection" / "bigcodebench_train_l2_extension.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            task_ids = [str(row["task_id"]) for row in payload.get("task_records", [])]
            spec = next(item for item in config.datasets if item.enabled and item.name == "bigcodebench")
            report = audit_native_level1_coverage(
                task_ids,
                [project_path(path, Path.cwd()) for path in spec.legacy_result_dirs],
            )
            output = args.output_dir / "selection" / "level1_reuse_audit.json"
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({**report, "report": str(output)}, indent=2))
            return 0 if not report["missing_level1_cache_tasks"] else 2
        report = validate_extension_output(
            output_root=args.output_dir,
            split_manifest=args.split_manifest,
            exploratory_manifests=_paths_or_default(
                args.exploratory_manifest, DEFAULT_EXPLORATORY
            ),
        )
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 2
    except (ExtensionPlanningError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
