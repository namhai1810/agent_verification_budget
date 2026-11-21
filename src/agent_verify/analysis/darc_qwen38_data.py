"""Adapters and split contracts for DARC training on Qwen3.8 Full-8 data.

The Qwen3.8 collection stores a shared-prefix tree per task rather than the
flat CBV-Bench trajectory rows consumed by the DARC replay code.  This module
performs a lossless, read-only conversion while preserving branch-specific
observable artifacts and actual Teacher token totals.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent_verify.analysis.cbv_v01_experiment import (
    ALL_SEQUENCES,
    FrozenTask,
    TaskTrajectories,
    load_frozen_partition,
)

PREFIXES = ("S", "V", "SS", "SV", "VS", "VV")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _task_catalog(
    frozen_root: Path, splits: tuple[str, ...] = ("train", "validation", "test")
) -> dict[str, FrozenTask]:
    tasks: dict[str, FrozenTask] = {}
    for split in splits:
        partition, _rows = load_frozen_partition(frozen_root, split)
        overlap = set(tasks) & set(partition)
        if overlap:
            raise ValueError(f"Duplicate frozen task keys: {sorted(overlap)[:3]}")
        tasks.update(partition)
    return tasks


def _trajectory_row(
    checkpoint: dict[str, Any], sequence: str, *, root_name: str
) -> dict[str, Any]:
    edges = checkpoint["edges"]
    trajectory = checkpoint["trajectories"][sequence]
    snapshots = []
    for depth in (1, 2, 3):
        prefix = sequence[:depth]
        edge = edges[prefix]
        if not bool(edge.get("complete")):
            raise ValueError(f"Incomplete edge {checkpoint['task_key']}:{prefix}")
        artifact = edge.get("input_artifact")
        if not isinstance(artifact, dict):
            raise ValueError(f"Missing input artifact {checkpoint['task_key']}:{prefix}")
        snapshots.append(artifact)
    terminal = edges[sequence].get("effective_artifact")
    if not isinstance(terminal, dict) or not isinstance(terminal.get("code"), str):
        raise ValueError(f"Missing terminal code {checkpoint['task_key']}:{sequence}")
    passed = trajectory.get("passed")
    if not isinstance(passed, bool):
        raise ValueError(f"Missing outcome {checkpoint['task_key']}:{sequence}")
    return {
        "schema_version": "cbv-qwen38-darc-adapter-v1",
        "record_type": "trajectory",
        "trajectory_id": hashlib.sha256(
            f"{root_name}:{checkpoint['task_key']}:{sequence}".encode()
        ).hexdigest(),
        "run_id": str(checkpoint.get("run_id", root_name)),
        "task_id": str(checkpoint["task_id"]),
        "dataset": str(checkpoint["dataset"]).lower(),
        "level": 2,
        "status": "completed",
        "action_sequence": sequence,
        "actions": list(sequence),
        "artifact_snapshots": snapshots,
        "final_code": terminal["code"],
        "passed": passed,
        "teacher_calls": int(trajectory.get("teacher_calls", sequence.count("V"))),
        "teacher_input_tokens": int(trajectory.get("teacher_input_tokens", 0)),
        "teacher_output_tokens": int(trajectory.get("teacher_output_tokens", 0)),
        "teacher_reasoning_tokens": int(trajectory.get("teacher_reasoning_tokens", 0)),
        "teacher_total_tokens": int(trajectory.get("teacher_total_tokens", 0)),
        "total_latency_seconds": float(trajectory.get("teacher_latency_seconds", 0.0)),
        "teacher_model": "Qwen/Qwen3.8-27B",
        "source": "qwen38_full8_shared_prefix_adapter",
        "source_root": root_name,
    }


def load_qwen38_full8_groups(
    roots: Iterable[Path],
    *,
    frozen_root: Path,
    expected_tasks: int | None = None,
    catalog_splits: tuple[str, ...] = ("train", "validation", "test"),
) -> tuple[dict[str, TaskTrajectories], dict[str, dict[str, Any]]]:
    """Load completed, mutually disjoint Qwen3.8 shared-prefix trees."""

    catalog = _task_catalog(frozen_root, catalog_splits)
    groups: dict[str, TaskTrajectories] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for root in roots:
        root = Path(root)
        checkpoint_dir = root / "task_checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Missing task checkpoint directory: {checkpoint_dir}")
        for path in sorted(checkpoint_dir.glob("*.json")):
            payload = _read_json(path)
            if payload.get("status") != "completed":
                continue
            task_key = str(payload.get("task_key", ""))
            if task_key in groups:
                raise ValueError(f"Qwen3.8 task appears in multiple roots: {task_key}")
            if task_key not in catalog:
                raise ValueError(f"Qwen3.8 task is absent from frozen catalog: {task_key}")
            if set(payload.get("trajectories", {})) != set(ALL_SEQUENCES):
                raise ValueError(f"Task does not have exactly eight trajectories: {task_key}")
            needed_edges = {sequence[:depth] for sequence in ALL_SEQUENCES for depth in (1, 2, 3)}
            if not needed_edges <= set(payload.get("edges", {})):
                missing = sorted(needed_edges - set(payload.get("edges", {})))
                raise ValueError(f"Task {task_key} is missing shared-prefix edges: {missing}")
            rows = {
                sequence: _trajectory_row(payload, sequence, root_name=root.name)
                for sequence in ALL_SEQUENCES
            }
            groups[task_key] = TaskTrajectories(catalog[task_key], rows, 2)
            metadata[task_key] = {
                "task_key": task_key,
                "dataset": payload["dataset"],
                "task_id": payload["task_id"],
                "selection_cohort": str(payload.get("selection_cohort", "unknown")),
                "source_root": str(root),
                "checkpoint_path": str(path),
                "old_oracle_passed": bool(payload.get("old_oracle_passed")),
                "new_oracle_passed": bool(payload.get("new_oracle_passed")),
            }
    if expected_tasks is not None and len(groups) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} Qwen3.8 tasks, found {len(groups)}")
    return groups, metadata


def load_selection_keys(root: Path) -> set[str]:
    path = Path(root) / "selection" / "selected_tasks.json"
    payload = _read_json(path)
    rows = payload.get("tasks", payload.get("selected_tasks", []))
    if not isinstance(rows, list):
        raise ValueError(f"Invalid task selection: {path}")
    keys = {
        str(row.get("task_key") or f"{str(row['dataset']).lower()}::{row['task_id']}")
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError(f"Duplicate task in selection: {path}")
    return keys


def assign_stratified_task_folds(
    metadata: dict[str, dict[str, Any]], *, folds: int, seed: int
) -> dict[str, int]:
    """Deterministic dataset/cohort-stratified task-level fold assignment."""

    if folds < 2:
        raise ValueError("At least two folds are required")
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, row in metadata.items():
        strata[(str(row["dataset"]).lower(), str(row["selection_cohort"]))].append(key)
    assignment: dict[str, int] = {}
    # Rotate each stratum so small strata do not always fill the same fold.
    for stratum_index, stratum in enumerate(sorted(strata)):
        ordered = sorted(
            strata[stratum],
            key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(),
        )
        offset = stratum_index % folds
        for index, key in enumerate(ordered):
            assignment[key] = (index + offset) % folds
    counts = Counter(assignment.values())
    if set(counts) != set(range(folds)):
        raise ValueError(f"Empty cross-validation fold: {dict(counts)}")
    return assignment


def split_audit_rows(
    groups: dict[str, TaskTrajectories],
    metadata: dict[str, dict[str, Any]],
    folds: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(groups):
        group = groups[key]
        rows.append(
            {
                **metadata[key],
                "fold": folds[key],
                "sss_passed": int(bool(group.rows["SSS"]["passed"])),
                "qwen38_oracle_passed": int(
                    any(bool(group.rows[sequence]["passed"]) for sequence in ALL_SEQUENCES)
                ),
                "vvv_teacher_tokens": int(group.rows["VVV"]["teacher_total_tokens"]),
            }
        )
    return rows

