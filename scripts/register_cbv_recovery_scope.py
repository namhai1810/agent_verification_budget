#!/usr/bin/env python3
"""Register a completed cached subset so standard validation/report/export can audit it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_verify.cbv_bench.cli import _override, parser as cbv_parser
from agent_verify.cbv_bench.config import load_cbv_config
from agent_verify.cbv_bench.datasets import load_selection
from agent_verify.cbv_bench.model_metadata import build_model_pair_snapshot
from agent_verify.cbv_bench.schemas import RunManifest
from agent_verify.cbv_bench.sequences import required_sequences
from agent_verify.logging import JsonlStore, content_id


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--level", type=int, choices=(1, 2), required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--compatibility", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cli_args = [
        "--dataset", args.dataset,
        "--selection-manifest", args.selection_manifest,
        "--level", str(args.level),
        "--repeated-rollouts", "1",
        "--results-dir", args.results_dir,
    ]
    parsed = cbv_parser().parse_args(["--config", args.config, "validate-data", *cli_args])
    config = _override(load_cbv_config(args.config), parsed)
    root = Path.cwd()
    results = Path(args.results_dir)
    selections = [
        load_selection(config, spec, root) for spec in config.datasets if spec.enabled
    ]
    allowed = {
        (selection.spec.name, task.task_id)
        for selection in selections for task in selection.public_tasks
    }
    trajectories = [
        row for row in _rows(results / "trajectories.jsonl")
        if row.get("compatibility_fingerprint") == args.compatibility
        and (row.get("dataset"), row.get("task_id")) in allowed
        and row.get("status") == "completed"
        and int(row.get("rollout_index", -1)) == 0
        and row.get("action_sequence") in required_sequences(args.level)
    ]
    expected = len(allowed) * len(required_sequences(args.level))
    if len(trajectories) != expected:
        raise RuntimeError(
            f"Recovery scope is not complete: expected {expected}, found {len(trajectories)}"
        )

    manifests = [
        row for row in _rows(results / "manifests.jsonl")
        if row.get("compatibility_fingerprint") == args.compatibility
    ]
    if not manifests:
        raise RuntimeError("No source run manifest matches the compatibility fingerprint")
    source = manifests[-1]
    pair = build_model_pair_snapshot(config)
    selection_hash = content_id(
        "selection",
        {
            item.spec.name: [
                record.model_dump(mode="json") for record in item.task_records.values()
            ]
            for item in selections
        },
    )
    existing = next(
        (
            row for row in reversed(manifests)
            if row.get("config_fingerprint") == config.fingerprint
            and row.get("status") == "complete"
        ),
        None,
    )
    if existing is not None:
        print(f"Recovery scope already registered: {config.fingerprint}")
        return

    manifest = RunManifest(
        run_id=str(source["run_id"]),
        status="complete",
        requested_level=args.level,
        required_sequences=list(required_sequences(args.level)),
        config_fingerprint=config.fingerprint,
        compatibility_fingerprint=args.compatibility,
        model_pair_id=pair.model_pair_id,
        model_pair_fingerprint=pair.model_pair_fingerprint,
        model_pair=pair,
        software_sha256=str(source["software_sha256"]),
        config=config.model_dump(mode="json"),
        selection_manifest_sha256=selection_hash,
        counts={
            "completed_trajectories": len(trajectories),
            "teacher_calls_used": sum(int(row.get("teacher_calls", 0)) for row in trajectories),
            "teacher_tokens_used": sum(
                int(row.get("teacher_total_tokens", 0)) for row in trajectories
            ),
        },
        limitations=list(source.get("limitations", [])) + [
            "Post-collection recovery scope excludes fully audited incomplete task groups; "
            "see the adjacent attrition report."
        ],
    )
    JsonlStore(results / "manifests.jsonl").append(manifest)
    print(f"Registered complete recovery scope: {config.fingerprint}")


if __name__ == "__main__":
    main()
