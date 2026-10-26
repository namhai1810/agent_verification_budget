#!/usr/bin/env python3
"""Audit that an extended LiveCodeBench lock preserves the original task prefix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_tasks(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"{path} does not contain a task list")
    return tasks


def audit_extension(
    base_path: Path,
    extension_path: Path,
    *,
    base_count: int = 100,
    total_count: int = 175,
) -> dict:
    base = load_tasks(base_path)
    extension = load_tasks(extension_path)
    if len(base) != base_count:
        raise ValueError(f"Expected {base_count} base tasks, found {len(base)}")
    if len(extension) != total_count:
        raise ValueError(f"Expected {total_count} extended tasks, found {len(extension)}")

    base_ids = [str(row["task_id"]) for row in base]
    extension_ids = [str(row["task_id"]) for row in extension]
    if len(set(extension_ids)) != total_count:
        raise ValueError("Extended selection contains duplicate task IDs")
    if extension_ids[:base_count] != base_ids:
        raise ValueError("Extended selection does not preserve the original 100-task prefix")

    remaining = extension[base_count:]
    difficulties = Counter(str(row["difficulty"]).lower() for row in remaining)
    if difficulties != {"hard": total_count - base_count}:
        raise ValueError(f"Expected only Hard extension tasks, found {dict(difficulties)}")
    return {
        "base_tasks": len(base),
        "extended_tasks": len(extension),
        "new_tasks": len(remaining),
        "new_difficulty": dict(difficulties),
        "prefix_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--base-count", type=int, default=100)
    parser.add_argument("--total-count", type=int, default=175)
    args = parser.parse_args()
    result = audit_extension(
        args.base,
        args.extension,
        base_count=args.base_count,
        total_count=args.total_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
