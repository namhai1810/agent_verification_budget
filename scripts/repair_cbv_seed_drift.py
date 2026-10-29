#!/usr/bin/env python3
"""Quarantine CBV recovery rows whose task-position-derived rollout seed drifted.

The collector intentionally keys cache entries by rollout seed.  Replacing a task and
re-sorting a selection can therefore create a second set of otherwise identical task
branches.  This utility keeps only rows matching the active manifest's exact
``task_index -> rollout_seed`` mapping.  Removed rows are retained with original line
numbers in an auditable quarantine directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


Json = dict[str, Any]


def read_json(path: Path) -> Json:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[tuple[int, str, Json]]:
    if not path.exists():
        return []
    output = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if raw.strip():
            output.append((number, raw, json.loads(raw)))
    return output


def digest(lines: list[str]) -> str:
    payload = "".join(line + "\n" for line in lines).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            for line in lines:
                handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def split_rows(
    rows: list[tuple[int, str, Json]],
    predicate: Callable[[Json], bool],
    unique_key: Callable[[Json], str] | None = None,
) -> tuple[list[str], list[Json]]:
    kept: list[str] = []
    removed: list[Json] = []
    seen: set[str] = set()
    for number, raw, row in rows:
        key = unique_key(row) if unique_key is not None else ""
        keep = predicate(row) and (unique_key is None or key not in seen)
        if keep:
            kept.append(raw)
            if unique_key is not None:
                seen.add(key)
        else:
            removed.append({"original_line_number": number, "record": row})
    return kept, removed


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--compatibility", required=True)
    parser.add_argument("--rollout-seed", type=int, default=20261026)
    parser.add_argument("--repeated-rollouts", type=int, default=1)
    parser.add_argument("--apply", action="store_true", help="Rewrite active JSONL files.")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    records = read_json(args.selection_manifest)["task_records"]
    expected_seed = {
        (str(record["task_id"]), rollout):
        args.rollout_seed + index * 100_003 + rollout
        for index, record in enumerate(records)
        for rollout in range(args.repeated_rollouts)
    }
    selected = {task_id for task_id, _ in expected_seed}

    def canonical_seed(row: Json) -> bool:
        if row.get("compatibility_fingerprint") != args.compatibility:
            return True
        task_id = str(row.get("task_id", ""))
        rollout = int(row.get("rollout_index", -1))
        if task_id not in selected or (task_id, rollout) not in expected_seed:
            return False
        return int(row.get("rollout_seed", -1)) == expected_seed[(task_id, rollout)]

    targets: dict[str, tuple[Callable[[Json], bool], Callable[[Json], str]]] = {
        "trajectories.jsonl": (canonical_seed, lambda row: str(row.get("cache_key"))),
        "cache/transitions.jsonl": (canonical_seed, lambda row: str(row.get("cache_key"))),
        "cache/roots.jsonl": (canonical_seed, lambda row: str(row.get("cache_key"))),
    }
    staged: dict[str, tuple[list[str], list[Json], list[str]]] = {}
    for relative, (predicate, unique_key) in targets.items():
        source = args.results_dir / relative
        rows = read_jsonl(source)
        kept, removed = split_rows(rows, predicate, unique_key)
        staged[relative] = (kept, removed, [raw for _, raw, _ in rows])

    kept_trajectories = [json.loads(line) for line in staged["trajectories.jsonl"][0]]
    kept_transitions = [json.loads(line) for line in staged["cache/transitions.jsonl"][0]]
    kept_roots = [json.loads(line) for line in staged["cache/roots.jsonl"][0]]
    invocation_ids = {
        str(value)
        for row in [*kept_roots, *kept_transitions]
        for value in (
            row.get("invocation_id"),
            row.get("teacher_invocation_id"),
            row.get("downstream_invocation_id"),
        )
        if value
    }
    state_ids = {
        str(value)
        for row in kept_trajectories
        for value in row.get("checkpoint_state_ids", [])
        if value
    }
    state_ids.update(
        str(value)
        for row in kept_transitions
        for value in (row.get("input_state_id"), row.get("output_state_id"))
        if value
    )
    evaluation_ids = {
        str(row.get("evaluation", {}).get("evaluation_id"))
        for row in kept_trajectories
        if row.get("evaluation", {}).get("evaluation_id")
    }

    reference_targets: dict[
        str, tuple[Callable[[Json], bool], Callable[[Json], str]]
    ] = {
        "llm_invocations.jsonl": (
            lambda row: str(row.get("invocation_id")) in invocation_ids,
            lambda row: str(row.get("invocation_id")),
        ),
        "checkpoint_states.jsonl": (
            lambda row: str(row.get("state_id")) in state_ids,
            lambda row: str(row.get("state_id")),
        ),
        "evaluations.jsonl": (
            lambda row: str(row.get("evaluation_id")) in evaluation_ids,
            lambda row: str(row.get("evaluation_id")),
        ),
    }
    for relative, (predicate, unique_key) in reference_targets.items():
        source = args.results_dir / relative
        rows = read_jsonl(source)
        kept, removed = split_rows(rows, predicate, unique_key)
        staged[relative] = (kept, removed, [raw for _, raw, _ in rows])

    summary = {
        "schema_version": "cbv-seed-drift-repair-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "results_dir": str(args.results_dir),
        "selection_manifest": str(args.selection_manifest),
        "selection_sha256": read_json(args.selection_manifest).get("selection_sha256"),
        "compatibility_fingerprint": args.compatibility,
        "selected_tasks": len(selected),
        "expected_trajectories": len(selected) * args.repeated_rollouts * 8,
        "files": {},
    }
    for relative, (kept, removed, before) in staged.items():
        summary["files"][relative] = {
            "before": len(before),
            "kept": len(kept),
            "quarantined": len(removed),
            "before_sha256": digest(before),
            "after_sha256": digest(kept),
        }

    if args.apply:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        quarantine = args.results_dir / "recovery_quarantine" / f"seed_drift_{stamp}"
        quarantine.mkdir(parents=True, exist_ok=False)
        for relative, (kept, removed, _before) in staged.items():
            if removed:
                removed_path = quarantine / f"{relative.replace('/', '__')}.removed.jsonl"
                atomic_write(
                    removed_path,
                    [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in removed],
                )
            atomic_write(args.results_dir / relative, kept)
        summary["quarantine_dir"] = str(quarantine)
        (quarantine / "audit.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
