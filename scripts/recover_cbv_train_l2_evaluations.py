#!/usr/bin/env python3
"""Recover only missing evaluator records in the train Level-2 extension cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_verify.cbv_bench.cache import cache_key
from agent_verify.cbv_bench.cli import _override, parser as cbv_parser
from agent_verify.cbv_bench.collector import CBVCollector
from agent_verify.cbv_bench.config import load_cbv_config
from agent_verify.cbv_bench.datasets import load_selection
from agent_verify.cbv_bench.labels import build_effect_records
from agent_verify.cbv_bench.sequences import required_sequences
from agent_verify.evaluation.bigcodebench_runner import BigCodeBenchRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--memory-mb", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path.cwd().resolve()
    cli_args = cbv_parser().parse_args([
        "--config", args.config, "collect",
        "--dataset", "bigcodebench",
        "--selection-manifest", args.selection_manifest,
        "--level", "2",
        "--repeated-rollouts", "1",
        "--results-dir", args.results_dir,
    ])
    config = _override(load_cbv_config(args.config), cli_args)
    if any(item.enabled and item.name != "bigcodebench" for item in config.datasets):
        raise RuntimeError("Recovery scope must contain BigCodeBench only")
    collector = CBVCollector(config, project_root)
    manifests_path = Path(args.results_dir) / "manifests.jsonl"
    manifests = [
        json.loads(line) for line in manifests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not manifests or manifests[-1].get("compatibility_fingerprint") != collector.compatibility:
        raise RuntimeError("Current code/config fingerprint does not match the cached collection")

    selections = [
        load_selection(config, spec, project_root)
        for spec in config.datasets if spec.enabled
    ]
    if any(record.split != "train" for selection in selections for record in selection.task_records.values()):
        raise RuntimeError("Recovery refuses any non-train task")
    recovery_config = config.evaluation.model_copy(update={
        "timeout_seconds": args.timeout_seconds,
        "memory_mb": args.memory_mb,
    })
    evaluator = BigCodeBenchRunner(recovery_config, collector.evaluations)
    evaluator.worker_path = (project_root / "scripts/cbv_bigcodebench_recovery_worker.py").resolve()
    sequences = required_sequences(2)
    teacher_calls_before = collector.collection_budget.used_calls
    teacher_tokens_before = collector.collection_budget.used_tokens
    attempted: dict[str, list[str]] = {}

    for selection in selections:
        for task_index, task in enumerate(selection.public_tasks):
            rollout_seed = config.collection.rollout_seed + task_index * 100_003
            missing = []
            for sequence in sequences:
                key = cache_key("trajectory", collector.compatibility, {
                    "dataset": selection.spec.name,
                    "task_id": task.task_id,
                    "rollout_index": 0,
                    "rollout_seed": rollout_seed,
                    "sequence": sequence,
                })
                if collector.trajectory_cache.get(key) is None:
                    missing.append(sequence)
            if not missing:
                continue
            attempted[task.task_id] = missing
            collector._collect_task_rollout(
                selection, task, task_index, 0, evaluator
            )

    if collector.collection_budget.used_calls != teacher_calls_before:
        raise RuntimeError("Recovery unexpectedly made a new Teacher call")
    if collector.collection_budget.used_tokens != teacher_tokens_before:
        raise RuntimeError("Recovery unexpectedly changed Teacher token usage")
    build_effect_records(Path(args.results_dir), collector.compatibility, 2)
    expected = sum(len(item.public_tasks) * len(sequences) for item in selections)
    completed = collector._completed_requested_count(selections)
    collector._manifest("complete" if completed == expected else "incomplete", selections)
    report = {
        "schema_version": "cbv-train-l2-evaluator-recovery-v1",
        "attempted_missing_sequences": attempted,
        "expected_trajectories": expected,
        "completed_trajectories": completed,
        "teacher_calls_before": teacher_calls_before,
        "teacher_calls_after": collector.collection_budget.used_calls,
        "teacher_tokens_before": teacher_tokens_before,
        "teacher_tokens_after": collector.collection_budget.used_tokens,
        "timeout_seconds": args.timeout_seconds,
        "memory_mb": args.memory_mb,
        "complete": completed == expected,
    }
    output = Path(args.results_dir).parent / "integrity" / "evaluator_recovery.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report": str(output)}, indent=2))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
