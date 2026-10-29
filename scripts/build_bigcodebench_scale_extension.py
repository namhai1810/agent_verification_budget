#!/usr/bin/env python3
"""Build outcome-blind BigCodeBench scale-up manifests.

The generated Level-1 manifest contains every currently eligible task that is
not already present in the validated 200-task pilot.  The Natural Level-2 pool
keeps the previously frozen 250-task selection, replaces only tasks with logged
infrastructure errors, and emits only the tasks not already completed in the
25-task pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_verify.cbv_bench.cache import sha256_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _records_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = manifest["task_records"]
    indexed = {str(record["task_id"]): record for record in records}
    if len(indexed) != len(records):
        raise ValueError("Selection manifest contains duplicate task IDs")
    return indexed


def _normalize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for rank, record in enumerate(records):
        copied = dict(record)
        copied["selection_rank"] = rank
        normalized.append(copied)
    return normalized


def _write_manifest(
    source: dict[str, Any],
    records: list[dict[str, Any]],
    output: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in source.items()
        if key not in {"selection_sha256", "task_records", "recovery", "extension"}
    }
    payload.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "scale_extension",
            "selection_protocol": (
                "frozen metadata-only selection with infrastructure-only exclusions; "
                "evaluation correctness labels were not used"
            ),
            "task_records": _normalize(records),
            "extension": metadata,
        }
    )
    payload["selection_sha256"] = sha256_json(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _hashed_order(record: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(
        f"cbv-natural-l2-replacement:{seed}:{record['task_id']}".encode()
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--completed-level1", type=Path, required=True)
    parser.add_argument("--frozen-natural-level2", type=Path, required=True)
    parser.add_argument("--completed-level2", type=Path, required=True)
    parser.add_argument(
        "--errors",
        type=Path,
        action="append",
        required=True,
        help=(
            "Level-1/general error JSONL; repeat to combine earlier and current "
            "infrastructure attrition."
        ),
    )
    parser.add_argument(
        "--level2-errors",
        type=Path,
        action="append",
        default=[],
        help=(
            "Level-2-only error JSONL. These tasks remain valid for Level 1 but are "
            "excluded/replaced in the Natural Level-2 pool."
        ),
    )
    parser.add_argument("--compatibility", required=True)
    parser.add_argument("--natural-level2-count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output-level1-extension", type=Path, required=True)
    parser.add_argument("--output-level2-full", type=Path, required=True)
    parser.add_argument("--output-level2-extension", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    universe = _read_json(args.universe)
    completed_l1 = _read_json(args.completed_level1)
    frozen_l2 = _read_json(args.frozen_natural_level2)
    completed_l2 = _read_json(args.completed_level2)

    universe_by_id = _records_by_id(universe)
    completed_l1_ids = set(_records_by_id(completed_l1))
    completed_l2_ids = set(_records_by_id(completed_l2))
    frozen_l2_by_id = _records_by_id(frozen_l2)

    unknown_l1 = completed_l1_ids - set(universe_by_id)
    unknown_l2 = completed_l2_ids - set(universe_by_id)
    if unknown_l1 or unknown_l2:
        raise ValueError("Completed manifests are not subsets of the frozen universe")

    latest_errors: dict[str, dict[str, Any]] = {}
    for error_path in args.errors:
        for row in _read_jsonl(error_path):
            if row.get("compatibility_fingerprint") == args.compatibility:
                latest_errors[str(row.get("task_id"))] = row
    level1_error_ids = set(latest_errors)

    latest_level2_errors: dict[str, dict[str, Any]] = {}
    for error_path in args.level2_errors:
        for row in _read_jsonl(error_path):
            if row.get("compatibility_fingerprint") == args.compatibility:
                latest_level2_errors[str(row.get("task_id"))] = row
    level2_error_ids = level1_error_ids | set(latest_level2_errors)

    if completed_l1_ids & level1_error_ids or completed_l2_ids & level2_error_ids:
        raise ValueError("A validated completed task is also marked as an active error")

    eligible_level1 = [
        record
        for task_id, record in universe_by_id.items()
        if task_id not in level1_error_ids
    ]
    eligible_level2 = [
        record
        for task_id, record in universe_by_id.items()
        if task_id not in level2_error_ids
    ]
    level1_extension = [
        record
        for record in eligible_level1
        if str(record["task_id"]) not in completed_l1_ids
    ]

    # First reproduce the Natural-L2 pool that existed before any Level-2-only
    # attrition.  Its order determines task_index and therefore rollout_seed, so
    # re-sorting after a replacement would silently invalidate otherwise reusable
    # cache entries.
    base_natural_records = [
        record
        for task_id, record in frozen_l2_by_id.items()
        if task_id not in level1_error_ids
    ]
    original_split_counts = Counter(
        str(record["split"]) for record in frozen_l2_by_id.values()
    )
    retained_split_counts = Counter(
        str(record["split"]) for record in base_natural_records
    )
    base_natural_ids = {
        str(record["task_id"]) for record in base_natural_records
    }
    candidates_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in original_split_counts:
        candidates = [
            record
            for record in eligible_level1
            if str(record["split"]) == split
            and str(record["task_id"]) not in frozen_l2_by_id
            and str(record["task_id"]) not in base_natural_ids
        ]
        candidates.sort(key=lambda record: _hashed_order(record, args.seed))
        candidates_by_split[split] = candidates

    base_replacements: list[dict[str, Any]] = []
    for split, target in original_split_counts.items():
        deficit = target - retained_split_counts[split]
        if deficit < 0 or len(candidates_by_split[split]) < deficit:
            raise ValueError(f"Cannot restore Natural-L2 split quota for {split}")
        chosen = candidates_by_split[split][:deficit]
        base_replacements.extend(chosen)
        base_natural_records.extend(chosen)
        base_natural_ids.update(str(record["task_id"]) for record in chosen)

    base_natural_records.sort(
        key=lambda record: hashlib.sha256(
            f"cbv-natural-l2-order:{args.seed}:{record['task_id']}".encode()
        ).hexdigest()
    )

    # Replace Level-2-only failures in-place.  Keeping the slot fixed preserves
    # task_index/rollout_seed for every unaffected task and prevents a recovery
    # run from regenerating cached trajectories.
    natural_records = list(base_natural_records)
    natural_ids = {str(record["task_id"]) for record in natural_records}
    level2_slot_replacements: list[dict[str, str]] = []
    level2_only_error_ids = set(latest_level2_errors)
    for index, failed_record in enumerate(list(natural_records)):
        failed_id = str(failed_record["task_id"])
        if failed_id not in level2_only_error_ids:
            continue
        split = str(failed_record["split"])
        candidates = [
            record
            for record in eligible_level2
            if str(record["split"]) == split
            and str(record["task_id"]) not in frozen_l2_by_id
            and str(record["task_id"]) not in natural_ids
        ]
        candidates.sort(key=lambda record: _hashed_order(record, args.seed))
        if not candidates:
            raise ValueError(
                f"Cannot replace Level-2-only failure {failed_id} in split {split}"
            )
        replacement = candidates[0]
        replacement_id = str(replacement["task_id"])
        natural_records[index] = replacement
        natural_ids.remove(failed_id)
        natural_ids.add(replacement_id)
        level2_slot_replacements.append(
            {
                "failed_task_id": failed_id,
                "replacement_task_id": replacement_id,
                "split": split,
                "slot": str(index),
            }
        )

    replacements = [
        *base_replacements,
        *[
            universe_by_id[row["replacement_task_id"]]
            for row in level2_slot_replacements
        ],
    ]

    if len(natural_records) != args.natural_level2_count:
        raise ValueError(
            f"Natural-L2 selection has {len(natural_records)} tasks; "
            f"expected {args.natural_level2_count}"
        )
    if not completed_l2_ids.issubset(natural_ids):
        raise ValueError("Completed Level-2 pilot is not retained in Natural-L2")

    level2_extension = [
        record
        for record in natural_records
        if str(record["task_id"]) not in completed_l2_ids
    ]

    common = {
        "schema_version": "cbv-scale-extension-v1",
        "compatibility_fingerprint": args.compatibility,
        "universe_manifest": str(args.universe),
        "universe_selection_sha256": universe["selection_sha256"],
        "selection_seed": args.seed,
    }
    level1_attrition = {
        "infrastructure_error_ids": sorted(level1_error_ids),
        "infrastructure_error_count": len(level1_error_ids),
        "infrastructure_error_logs": [str(path) for path in args.errors],
    }
    level2_attrition = {
        "infrastructure_error_ids": sorted(level2_error_ids),
        "infrastructure_error_count": len(level2_error_ids),
        "infrastructure_error_logs": [
            *[str(path) for path in args.errors],
            *[str(path) for path in args.level2_errors],
        ],
        "level2_only_error_ids": sorted(set(latest_level2_errors)),
        "level2_only_error_count": len(latest_level2_errors),
        "level2_only_error_logs": [str(path) for path in args.level2_errors],
    }
    l1_payload = _write_manifest(
        universe,
        level1_extension,
        args.output_level1_extension,
        {
            **common,
            **level1_attrition,
            "purpose": "collect all remaining infrastructure-eligible Level-1 tasks",
            "completed_level1_manifest": str(args.completed_level1),
            "completed_level1_tasks": len(completed_l1_ids),
            "extension_tasks": len(level1_extension),
            "final_eligible_level1_tasks": len(completed_l1_ids) + len(level1_extension),
        },
    )
    full_l2_payload = _write_manifest(
        frozen_l2,
        natural_records,
        args.output_level2_full,
        {
            **common,
            **level2_attrition,
            "purpose": "frozen outcome-blind Natural Level-2 evaluation pool",
            "original_frozen_manifest": str(args.frozen_natural_level2),
            "replacement_task_ids": sorted(str(row["task_id"]) for row in replacements),
            "replacement_count": len(replacements),
            "slot_preserving_level2_replacements": level2_slot_replacements,
            "split_counts": dict(Counter(str(row["split"]) for row in natural_records)),
        },
    )
    l2_payload = _write_manifest(
        frozen_l2,
        level2_extension,
        args.output_level2_extension,
        {
            **common,
            **level2_attrition,
            "purpose": "collect the untouched remainder of Natural Level-2",
            "full_natural_level2_manifest": str(args.output_level2_full),
            "full_natural_level2_selection_sha256": full_l2_payload["selection_sha256"],
            "completed_level2_manifest": str(args.completed_level2),
            "completed_level2_tasks": len(completed_l2_ids),
            "extension_tasks": len(level2_extension),
        },
    )

    report = {
        "schema_version": "cbv-scale-extension-report-v1",
        "universe_tasks": len(universe_by_id),
        "known_infrastructure_errors": len(level1_error_ids),
        "known_level1_infrastructure_errors": len(level1_error_ids),
        "known_level2_infrastructure_errors": len(level2_error_ids),
        "known_level2_only_infrastructure_errors": len(latest_level2_errors),
        "completed_level1_tasks": len(completed_l1_ids),
        "level1_extension_tasks": len(level1_extension),
        "final_eligible_level1_tasks": len(completed_l1_ids) + len(level1_extension),
        "completed_level2_tasks": len(completed_l2_ids),
        "natural_level2_tasks": len(natural_records),
        "level2_extension_tasks": len(level2_extension),
        "natural_level2_replacements": len(replacements),
        "level1_extension_trajectories": len(level1_extension) * 4,
        "level2_extension_trajectories": len(level2_extension) * 8,
        "outputs": {
            "level1_extension": str(args.output_level1_extension),
            "level1_selection_sha256": l1_payload["selection_sha256"],
            "level2_full": str(args.output_level2_full),
            "level2_extension": str(args.output_level2_extension),
            "level2_selection_sha256": l2_payload["selection_sha256"],
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
