#!/usr/bin/env python3
"""Build auditable, iterative BigCodeBench recovery selections.

The Level-1 selection keeps all current task positions. Tasks with incomplete
infrastructure/model-output collection are replaced by the next frozen candidate
that has not already logged an error under the active compatibility fingerprint.
This preserves rollout seeds (and exact-cache keys) for every retained task and can
be applied repeatedly. The Level-2 selection is an outcome-blind, split-stratified
subset of the recovered Level-1 task set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_verify.cbv_bench.cache import sha256_json


LEVEL1_SEQUENCES = frozenset({"SSS", "VSS", "SVS", "SSV"})


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_manifest(
    source: dict[str, Any],
    records: list[dict[str, Any]],
    output: Path,
    recovery: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in source.items()
        if key not in {"selection_sha256", "task_records"}
    }
    payload.update(
        {
            "created_at_utc": _utc_now(),
            "mode": "infrastructure_recovery",
            "selection_protocol": (
                "metadata-only frozen selection with infrastructure-completeness "
                "recovery; no evaluation correctness outcome used"
            ),
            "task_records": records,
            "recovery": recovery,
        }
    )
    payload["selection_sha256"] = sha256_json(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def _stratified_subset(
    records: list[dict[str, Any]], count: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if count > len(records):
        raise ValueError(f"Level-2 count {count} exceeds recovered Level-1 size {len(records)}")

    # Largest-remainder allocation for the project's frozen 70/15/15 protocol.
    fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}
    exact = {name: count * fraction for name, fraction in fractions.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    remainder = count - sum(quotas.values())
    order = sorted(
        fractions,
        key=lambda name: (exact[name] - quotas[name], name == "validation", name),
        reverse=True,
    )
    for name in order[:remainder]:
        quotas[name] += 1

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["split"])].append(record)
    for split, quota in quotas.items():
        if len(groups[split]) < quota:
            raise ValueError(
                f"Only {len(groups[split])} {split} tasks are available; need {quota}"
            )
        groups[split].sort(
            key=lambda row: hashlib.sha256(
                f"cbv-recovery-l2:{seed}:{row['task_id']}".encode()
            ).hexdigest()
        )

    chosen = [record for split in ("train", "validation", "test") for record in groups[split][: quotas[split]]]
    chosen.sort(
        key=lambda row: hashlib.sha256(
            f"cbv-recovery-l2-order:{seed}:{row['task_id']}".encode()
        ).hexdigest()
    )
    normalized = []
    for rank, record in enumerate(chosen):
        copied = dict(record)
        copied["selection_rank"] = rank
        normalized.append(copied)
    return normalized, quotas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-level1", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--compatibility", required=True)
    parser.add_argument("--output-level1", type=Path, required=True)
    parser.add_argument("--output-level2", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--level2-count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261027)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _read_json(args.source_level1)
    pool = _read_json(args.candidate_pool)
    original = list(source["task_records"])
    candidates = list(pool["task_records"])
    if len(candidates) <= len(original):
        raise ValueError("Candidate pool must contain tasks beyond the Level-1 selection")
    pool_by_id = {str(record["task_id"]): record for record in candidates}
    for record in original:
        candidate = pool_by_id.get(str(record["task_id"]))
        if candidate is None or candidate["public_task_sha256"] != record["public_task_sha256"]:
            raise ValueError(
                "Candidate pool is not a content-compatible frozen superset of Level-1"
            )

    completed: dict[str, set[str]] = defaultdict(set)
    for row in _read_jsonl(args.trajectories):
        if (
            row.get("compatibility_fingerprint") == args.compatibility
            and row.get("dataset") == "bigcodebench"
            and row.get("status") == "completed"
            and int(row.get("rollout_index", -1)) == 0
            and row.get("action_sequence") in LEVEL1_SEQUENCES
        ):
            completed[str(row["task_id"])].add(str(row["action_sequence"]))

    incomplete_positions = [
        index
        for index, record in enumerate(original)
        if not LEVEL1_SEQUENCES.issubset(completed.get(str(record["task_id"]), set()))
    ]
    if not incomplete_positions:
        raise ValueError("Original selection is already complete; recovery is unnecessary")

    latest_errors: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(args.errors):
        if row.get("compatibility_fingerprint") == args.compatibility:
            latest_errors[str(row.get("task_id"))] = row

    original_ids = {str(record["task_id"]) for record in original}
    known_error_ids = set(latest_errors)
    replacements = [
        record
        for record in candidates
        if str(record["task_id"]) not in original_ids
        and str(record["task_id"]) not in known_error_ids
    ][: len(incomplete_positions)]
    if len(replacements) != len(incomplete_positions):
        raise ValueError("Candidate pool does not contain enough unused, error-free replacements")

    recovered = [dict(record) for record in original]
    replacement_rows: list[dict[str, Any]] = []
    for position, replacement in zip(incomplete_positions, replacements):
        removed = original[position]
        inserted = dict(replacement)
        inserted["selection_rank"] = position
        recovered[position] = inserted
        error = latest_errors.get(str(removed["task_id"]), {})
        replacement_rows.append(
            {
                "position": position,
                "excluded_task_id": removed["task_id"],
                "replacement_task_id": inserted["task_id"],
                "excluded_completed_sequences": sorted(
                    completed.get(str(removed["task_id"]), set())
                ),
                "latest_error_type": error.get("error_type", "unknown"),
                "latest_error_message": error.get("message", "")[:500],
            }
        )

    common_recovery = {
        "schema_version": "cbv-infrastructure-recovery-v1",
        "source_selection": str(args.source_level1),
        "source_selection_sha256": source["selection_sha256"],
        "parent_recovery_schema_version": source.get("recovery", {}).get(
            "schema_version", ""
        ),
        "candidate_pool": str(args.candidate_pool),
        "candidate_pool_selection_sha256": pool["selection_sha256"],
        "compatibility_fingerprint": args.compatibility,
        "eligibility_rule": (
            "all four Level-1 action sequences completed under the exact compatibility "
            "fingerprint; pass/fail correctness was not inspected"
        ),
        "preserves_original_positions_and_rollout_seeds": True,
        "replacement_rows": replacement_rows,
    }
    level1 = _write_manifest(
        source,
        recovered,
        args.output_level1,
        {**common_recovery, "purpose": "complete 800 Level-1 trajectories"},
    )

    level2_records, quotas = _stratified_subset(recovered, args.level2_count, args.seed)
    level2 = _write_manifest(
        source,
        level2_records,
        args.output_level2,
        {
            **common_recovery,
            "purpose": "outcome-blind split-stratified Level-2 pilot",
            "level2_seed": args.seed,
            "level2_split_quotas": quotas,
            "level2_note": (
                "Task positions are re-ranked for the Level-2 selection; Level-1 cache reuse "
                "is not assumed for this stratified subset."
            ),
        },
    )

    report_payload = {
        "schema_version": "cbv-recovery-report-v1",
        "created_at_utc": _utc_now(),
        "compatibility_fingerprint": args.compatibility,
        "original_tasks": len(original),
        "fully_complete_original_tasks": len(original) - len(incomplete_positions),
        "incomplete_original_tasks": len(incomplete_positions),
        "retained_completed_level1_trajectories": (
            len(original) - len(incomplete_positions)
        )
        * len(LEVEL1_SEQUENCES),
        "new_level1_trajectories_required": len(incomplete_positions)
        * len(LEVEL1_SEQUENCES),
        "recovered_level1_tasks": len(recovered),
        "recovered_level1_target_trajectories": len(recovered)
        * len(LEVEL1_SEQUENCES),
        "level2_tasks": len(level2_records),
        "level2_target_trajectories": len(level2_records) * 8,
        "level2_split_counts": dict(Counter(row["split"] for row in level2_records)),
        "replacement_rows": replacement_rows,
        "level1_manifest": str(args.output_level1),
        "level1_selection_sha256": level1["selection_sha256"],
        "level2_manifest": str(args.output_level2),
        "level2_selection_sha256": level2["selection_sha256"],
        "scientific_caveat": (
            "The listed tasks were removed only because collection could not produce complete "
            "records. This is post-selection infrastructure attrition and must be reported; "
            "no pass/fail label was used to choose replacements."
        ),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "recovery_selection_report.json").write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# BigCodeBench recovery selection",
        "",
        f"- Compatibility: `{args.compatibility}`",
        f"- Complete original tasks: **{report_payload['fully_complete_original_tasks']}/{len(original)}**",
        f"- Replacements: **{len(replacement_rows)}**",
        f"- Recovered Level 1: **{len(recovered)} tasks / {len(recovered) * 4} trajectories**",
        f"- Level 2: **{len(level2_records)} tasks / {len(level2_records) * 8} trajectories**",
        f"- Level-2 split: **{report_payload['level2_split_counts']}**",
        "",
        "## Replacement audit",
        "",
        "| Position | Excluded | Replacement | Completed branches | Error |",
        "|---:|---|---|---|---|",
    ]
    for row in replacement_rows:
        branches = ", ".join(row["excluded_completed_sequences"]) or "none"
        lines.append(
            f"| {row['position']} | `{row['excluded_task_id']}` | "
            f"`{row['replacement_task_id']}` | {branches} | "
            f"{row['latest_error_type']} |"
        )
    lines.extend(
        [
            "",
            "## Scientific caveat",
            "",
            report_payload["scientific_caveat"],
            "",
        ]
    )
    (args.report_dir / "recovery_selection_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(report_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
