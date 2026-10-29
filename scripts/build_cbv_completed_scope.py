#!/usr/bin/env python3
"""Create an auditable selection containing only fully completed CBV task groups."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_verify.cbv_bench.cache import sha256_json


SEQUENCES = {
    1: frozenset({"SSS", "VSS", "SVS", "SSV"}),
    2: frozenset({"SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV"}),
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--compatibility", required=True)
    parser.add_argument("--level", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-selection", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _json(args.source_selection)
    records = list(source["task_records"])
    source_ids = {str(record["task_id"]) for record in records}
    required = SEQUENCES[args.level]

    observed: dict[str, set[str]] = defaultdict(set)
    for row in _jsonl(args.trajectories):
        task_id = str(row.get("task_id", ""))
        if (
            row.get("compatibility_fingerprint") == args.compatibility
            and task_id in source_ids
            and row.get("status") == "completed"
            and int(row.get("rollout_index", -1)) == 0
            and row.get("action_sequence") in required
        ):
            observed[task_id].add(str(row["action_sequence"]))

    complete_ids = {
        task_id for task_id in source_ids if required.issubset(observed.get(task_id, set()))
    }
    retained = [record for record in records if str(record["task_id"]) in complete_ids]
    incomplete_ids = sorted(source_ids - complete_ids)

    latest_errors: dict[str, dict[str, Any]] = {}
    for row in _jsonl(args.errors):
        task_id = str(row.get("task_id", ""))
        if row.get("compatibility_fingerprint") == args.compatibility and task_id in source_ids:
            latest_errors[task_id] = row

    payload = {
        key: value
        for key, value in source.items()
        if key not in {"selection_sha256", "task_records", "recovery", "extension"}
    }
    payload.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "postcollection_complete_scope",
            "selection_protocol": (
                "retains task groups with every required trajectory completed under the frozen "
                "compatibility fingerprint; correctness labels were not used for attrition"
            ),
            "task_records": retained,
            "recovery": {
                "schema_version": "cbv-postcollection-scope-v1",
                "source_selection": str(args.source_selection),
                "source_selection_sha256": source["selection_sha256"],
                "compatibility_fingerprint": args.compatibility,
                "level": args.level,
                "required_sequences": sorted(required),
                "source_tasks": len(records),
                "complete_tasks": len(retained),
                "incomplete_tasks": len(incomplete_ids),
                "incomplete_task_ids": incomplete_ids,
                "scientific_caveat": (
                    "This is infrastructure/model-output attrition, not a correctness-based filter; "
                    "the excluded IDs and error types must be reported with benchmark results."
                ),
            },
        }
    )
    payload["selection_sha256"] = sha256_json(payload)
    args.output_selection.parent.mkdir(parents=True, exist_ok=True)
    args.output_selection.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    error_types = Counter(
        str(latest_errors.get(task_id, {}).get("error_type", "missing_error_record"))
        for task_id in incomplete_ids
    )
    report = {
        "schema_version": "cbv-postcollection-scope-report-v1",
        "compatibility_fingerprint": args.compatibility,
        "level": args.level,
        "source_tasks": len(records),
        "complete_tasks": len(retained),
        "incomplete_tasks": len(incomplete_ids),
        "completion_rate": len(retained) / len(records) if records else 0.0,
        "complete_trajectories": len(retained) * len(required),
        "incomplete_task_ids": incomplete_ids,
        "error_types": dict(error_types),
        "errors": {
            task_id: {
                "error_type": latest_errors.get(task_id, {}).get("error_type", ""),
                "message": latest_errors.get(task_id, {}).get("message", ""),
            }
            for task_id in incomplete_ids
        },
        "output_selection": str(args.output_selection),
        "selection_sha256": payload["selection_sha256"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
