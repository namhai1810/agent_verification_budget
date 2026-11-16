"""Audited import of deterministic Phase-1 Level-1 branches into the CBV cache."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent_verify.cbv_bench.cache import ExactCache, cache_key, sha256_json
from agent_verify.cbv_bench.schemas import TrajectoryRecord, TransitionRecord
from agent_verify.cbv_bench.config import CBVConfig
from agent_verify.cbv_bench.model_metadata import build_model_pair_snapshot
from agent_verify.logging import JsonlStore, canonical_json, content_id
from agent_verify.schemas import CheckpointState, EvaluationResult, InvocationRecord


class LegacyImportError(RuntimeError):
    pass


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=8)
def _native_level1_index(directory: str) -> dict[str, Any]:
    """Index a native CBV result directory once for deterministic L1 reuse.

    Historical BigCodeBench Level-1 data is already in CBV format, rather than the
    older baseline/intervention-pair format handled below.  Keeping this read-only
    index avoids reparsing large JSONL files once per selected task.
    """
    root = Path(directory)
    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read(root / "trajectories.jsonl"):
        if (
            row.get("dataset") == "bigcodebench"
            and row.get("status") == "completed"
            and row.get("action_sequence") in {"SSS", "VSS", "SVS", "SSV"}
            and int(row.get("rollout_index", -1)) == 0
        ):
            trajectories[str(row.get("task_id"))].append(row)
    transitions: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _read(root / "cache" / "transitions.jsonl"):
        task_id = str(row.get("task_id", ""))
        compatibility = str(row.get("compatibility_fingerprint", ""))
        if task_id in trajectories and row.get("status") == "completed":
            transitions[(task_id, compatibility)][str(row.get("transition_id"))] = row
    states: dict[str, dict[str, Any]] = {}
    wanted_tasks = set(trajectories)
    for row in _read(root / "checkpoint_states.jsonl"):
        if str(row.get("task_id", "")) in wanted_tasks:
            states[str(row.get("state_id"))] = row
    roots: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _read(root / "cache" / "roots.jsonl"):
        if str(row.get("task_id", "")) in wanted_tasks and row.get("status") == "completed":
            roots[(str(row.get("task_id")), str(row.get("compatibility_fingerprint", "")))].append(row)
    return {
        "trajectories": dict(trajectories),
        "transitions": dict(transitions),
        "states": states,
        "roots": dict(roots),
    }


def audit_native_level1_coverage(
    task_ids: list[str], source_dirs: list[Path]
) -> dict[str, Any]:
    """Locate complete native Level-1 shared-prefix trees without using outcomes."""
    required = {"SSS", "VSS", "SVS", "SSV"}
    remaining = set(task_ids)
    source_by_task: dict[str, str] = {}
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        index = _native_level1_index(str(source_dir.resolve()))
        for task_id in sorted(remaining):
            by_compatibility: dict[str, set[str]] = defaultdict(set)
            for row in index["trajectories"].get(task_id, []):
                by_compatibility[str(row.get("compatibility_fingerprint", ""))].add(
                    str(row.get("action_sequence", ""))
                )
            if any(required.issubset(sequences) for sequences in by_compatibility.values()):
                source_by_task[task_id] = str(source_dir)
        remaining.difference_update(source_by_task)
        if not remaining:
            break
    return {
        "requested_tasks": len(task_ids),
        "reusable_level1_tasks": len(source_by_task),
        "missing_level1_cache_tasks": sorted(remaining),
        "source_counts": dict(Counter(source_by_task.values())),
        "source_by_task": source_by_task,
        "outcomes_used_for_selection": False,
    }


def _reconstruct_partitioned_level1(
    *,
    legacy_dir: Path,
    dataset: str,
    dataset_version: str,
    task,
    task_record,
    logical_rollout_seed: int,
    run_id: str,
    compatibility: str,
    config: CBVConfig,
    root_cache: ExactCache,
    transition_cache: ExactCache,
    trajectory_cache: ExactCache,
    state_log: JsonlStore,
    source_compatibility: str,
    source_trajectories: dict[str, dict[str, Any]],
) -> bool:
    """Rebuild L1 cache records from the frozen *train* trajectory partition.

    The frozen partition intentionally contains no mutable cache internals.  Four
    single-intervention trajectories nevertheless contain every artifact required to
    restore the shared L1 prefix tree.  Synthetic invocation IDs are provenance only;
    no model result or token count is invented.
    """
    required = {"SSS", "VSS", "SVS", "SSV"}
    if not required.issubset(source_trajectories):
        return False
    snapshots = {
        sequence: source_trajectories[sequence].get("artifact_snapshots", [])
        for sequence in required
    }
    if any(len(values) != 3 for values in snapshots.values()):
        return False
    model_pair = build_model_pair_snapshot(config)
    root_identity = {
        "dataset": dataset,
        "task_sha256": sha256_json(task.model_dump(mode="json")),
        "rollout_index": 0,
        "rollout_seed": logical_rollout_seed,
        "role": "planner",
    }
    root_key = cache_key("root", compatibility, root_identity)
    if root_cache.get(root_key) is not None:
        return True

    def state(
        *, branch: str, checkpoint: int, preceding: list[dict[str, Any]],
        artifact: dict[str, Any], parent_transition_id: str | None,
    ) -> CheckpointState:
        identity = {
            "compatibility": compatibility,
            "task_id": task.task_id,
            "branch": branch,
            "rollout_seed": logical_rollout_seed,
            "checkpoint": checkpoint,
            "parent": parent_transition_id,
            "preceding": preceding,
            "artifact": artifact,
        }
        role = {1: "planner", 2: "coder", 3: "reviewer"}[checkpoint]
        return CheckpointState(
            state_id=content_id("state", identity),
            run_id=run_id,
            task_id=task.task_id,
            checkpoint_index=checkpoint,
            role=role,
            branch=f"cbv:{branch}",
            rollout_seed=logical_rollout_seed,
            parent_state_id=parent_transition_id,
            preceding_artifacts=preceding,
            current_artifact=artifact,
            messages=[],
            llm_config=getattr(config.models, role),
            invocation_id=content_id("imported_invocation", identity),
        )

    root_state = state(
        branch="root", checkpoint=1, preceding=[],
        artifact=snapshots["SSS"][0], parent_transition_id=None,
    )
    existing_state_ids = {
        str(row.get("state_id")) for row in state_log.read_all()
        if row.get("record_type") == "checkpoint_state"
    }
    if root_state.state_id not in existing_state_ids:
        state_log.append(root_state)
        existing_state_ids.add(root_state.state_id)
    root_cache.put({
        "record_type": "cbv_root",
        "cache_key": root_key,
        "compatibility_fingerprint": compatibility,
        "status": "completed",
        "dataset": dataset,
        "model_pair_id": model_pair.model_pair_id,
        "model_pair_fingerprint": model_pair.model_pair_fingerprint,
        "task_id": task.task_id,
        "rollout_index": 0,
        "rollout_seed": logical_rollout_seed,
        "state_id": root_state.state_id,
        "invocation_id": root_state.invocation_id,
        "model": config.models.planner.model_dump(mode="json"),
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "latency_seconds": 0.0,
        "source": "legacy_level1",
        "native_source_dir": str(legacy_dir),
        "native_source_compatibility": source_compatibility,
        "accounting_note": "aggregate student accounting is preserved on imported trajectories",
    })

    # effective artifact and downstream output for every unique L1 prefix.
    edge_artifacts = {
        "S": (snapshots["SSS"][0], snapshots["SSS"][1]),
        "V": (snapshots["VSS"][0], snapshots["VSS"][1]),
        "SS": (snapshots["SSS"][1], snapshots["SSS"][2]),
        "SV": (snapshots["SVS"][1], snapshots["SVS"][2]),
        "VS": (snapshots["VSS"][1], snapshots["VSS"][2]),
        "SSS": (snapshots["SSS"][2], snapshots["SSS"][2]),
        "SSV": (snapshots["SSV"][2], snapshots["SSV"][2]),
        "SVS": (snapshots["SVS"][2], snapshots["SVS"][2]),
        "VSS": (snapshots["VSS"][2], snapshots["VSS"][2]),
    }
    teacher_source = {"V": "VSS", "SV": "SVS", "SSV": "SSV"}
    node_states: dict[str, CheckpointState] = {"": root_state}
    current_transitions: dict[str, TransitionRecord] = {}
    for prefix in sorted(edge_artifacts, key=lambda value: (len(value), value)):
        checkpoint = len(prefix)
        action = prefix[-1]
        parent = prefix[:-1]
        input_state = node_states[parent]
        effective, output_artifact = edge_artifacts[prefix]
        identity = {
            "dataset": dataset,
            "task_id": task.task_id,
            "rollout_index": 0,
            "rollout_seed": logical_rollout_seed,
            "prefix": prefix,
            "action": action,
            "input_state_id": input_state.state_id,
        }
        key = cache_key("transition", compatibility, identity)
        transition_id = content_id("cbv_transition", {"cache_key": key, "prefix": prefix})
        if checkpoint == 1:
            next_state = state(
                branch=prefix,
                checkpoint=2,
                preceding=[effective],
                artifact=output_artifact,
                parent_transition_id=transition_id,
            )
        elif checkpoint == 2:
            next_state = state(
                branch=prefix,
                checkpoint=3,
                preceding=[input_state.preceding_artifacts[0], effective],
                artifact=output_artifact,
                parent_transition_id=transition_id,
            )
        else:
            next_state = None
        if next_state is not None:
            node_states[prefix] = next_state
            if next_state.state_id not in existing_state_ids:
                state_log.append(next_state)
                existing_state_ids.add(next_state.state_id)
            output_state_id = next_state.state_id
        else:
            output_state_id = content_id(
                "terminal", {"transition_id": transition_id, "artifact": output_artifact}
            )
        source_sequence = teacher_source.get(prefix)
        source_trajectory = source_trajectories[source_sequence] if source_sequence else None
        teacher_input_tokens = int(
            source_trajectory.get("teacher_input_tokens", 0) if source_trajectory else 0
        )
        teacher_output_tokens = int(
            source_trajectory.get("teacher_output_tokens", 0) if source_trajectory else 0
        )
        teacher_invocation_id = (
            content_id("imported_teacher_invocation", identity) if action == "V" else None
        )
        transition = TransitionRecord(
            transition_id=transition_id,
            cache_key=key,
            compatibility_fingerprint=compatibility,
            run_id=run_id,
            model_pair_id=model_pair.model_pair_id,
            model_pair_fingerprint=model_pair.model_pair_fingerprint,
            dataset=dataset,
            task_id=task.task_id,
            rollout_index=0,
            rollout_seed=logical_rollout_seed,
            checkpoint=checkpoint,
            prefix=prefix,
            action=action,
            parent_prefix=parent,
            parent_state_id=input_state.parent_state_id,
            input_state_id=input_state.state_id,
            output_state_id=output_state_id,
            input_artifact=input_state.current_artifact,
            effective_artifact=effective,
            output_artifact=output_artifact,
            preceding_artifacts=input_state.preceding_artifacts,
            teacher_model=config.models.teacher if action == "V" else None,
            teacher_invocation_id=teacher_invocation_id,
            teacher_input_tokens=teacher_input_tokens,
            teacher_output_tokens=teacher_output_tokens,
            teacher_total_tokens=teacher_input_tokens + teacher_output_tokens,
            teacher_cost_usd=(
                float(source_trajectory.get("teacher_cost_usd", 0.0))
                if source_trajectory else 0.0
            ),
            source="legacy_level1",
            provenance={
                "partitioned_train_import": True,
                "native_source_dir": str(legacy_dir),
                "native_source_compatibility": source_compatibility,
                "source_sequence": source_sequence or "",
                "outcome_used_for_selection": False,
            },
        )
        transition_cache.put(transition.model_dump(mode="json"))
        current_transitions[prefix] = transition

    for sequence in sorted(required):
        source = dict(source_trajectories[sequence])
        identity = {
            "dataset": dataset,
            "task_id": task.task_id,
            "rollout_index": 0,
            "rollout_seed": logical_rollout_seed,
            "sequence": sequence,
        }
        key = cache_key("trajectory", compatibility, identity)
        edges = [current_transitions[sequence[:depth]] for depth in (1, 2, 3)]
        source.update({
            "trajectory_id": content_id("trajectory", {"cache_key": key}),
            "branch_id": content_id("branch", identity),
            "parent_branch_id": content_id(
                "branch_prefix", {**identity, "prefix": sequence[:2]}
            ),
            "cache_key": key,
            "compatibility_fingerprint": compatibility,
            "run_id": run_id,
            "model_pair_id": model_pair.model_pair_id,
            "model_pair_fingerprint": model_pair.model_pair_fingerprint,
            "model_pair": model_pair.model_dump(mode="json"),
            "dataset": dataset,
            "dataset_version": dataset_version,
            "difficulty": task_record.difficulty,
            "split": "train",
            "level": 1,
            "student_models": {
                role: getattr(config.models, role).model_dump(mode="json")
                for role in ("planner", "coder", "reviewer")
            },
            "teacher_model": config.models.teacher.model_dump(mode="json"),
            "prompt_versions": {
                role: getattr(config.models, role).prompt_version
                for role in ("planner", "coder", "reviewer", "teacher")
            },
            "rollout_seed": logical_rollout_seed,
            "transition_ids": [edge.transition_id for edge in edges],
            "checkpoint_state_ids": [
                root_state.state_id,
                node_states[sequence[:1]].state_id,
                node_states[sequence[:2]].state_id,
            ],
            "source": "legacy_level1",
            "provenance": {
                **dict(source.get("provenance") or {}),
                "partitioned_train_import": True,
                "native_source_dir": str(legacy_dir),
                "native_source_trajectory_id": source.get("trajectory_id"),
                "native_source_compatibility": source_compatibility,
                "native_rollout_seed": source.get("rollout_seed"),
                "outcome_used_for_selection": False,
            },
        })
        source = {
            key: value for key, value in source.items()
            if key in TrajectoryRecord.model_fields
        }
        trajectory_cache.put(TrajectoryRecord.model_validate(source).model_dump(mode="json"))
    return True


def _import_native_cbv_level1(
    *,
    legacy_dir: Path,
    dataset: str,
    dataset_version: str,
    task,
    task_record,
    logical_rollout_seed: int,
    run_id: str,
    compatibility: str,
    config: CBVConfig,
    root_cache: ExactCache,
    transition_cache: ExactCache,
    trajectory_cache: ExactCache,
    state_log: JsonlStore,
) -> bool:
    """Translate complete native L1 branches into the active immutable cache.

    The source artifacts/outcomes are reused only for a train task already certified
    L1-complete by the frozen split manifest.  Cache identities and rollout seeds are
    regenerated for the active run; source IDs remain in provenance.
    """
    index = _native_level1_index(str(legacy_dir.resolve()))
    source_rows = index["trajectories"].get(task.task_id, [])
    if not source_rows:
        return False
    by_compatibility: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in source_rows:
        by_compatibility[str(row.get("compatibility_fingerprint", ""))][
            str(row.get("action_sequence", ""))
        ] = row
    required = {"SSS", "VSS", "SVS", "SSV"}
    complete_groups = [
        (source_compatibility, rows)
        for source_compatibility, rows in by_compatibility.items()
        if required.issubset(rows)
    ]
    if not complete_groups:
        return False
    source_compatibility, source_trajectories = complete_groups[-1]

    for sequence, row in source_trajectories.items():
        if sequence not in required:
            continue
        if row.get("dataset_version") != dataset_version:
            raise LegacyImportError(f"Native Level-1 dataset version mismatch for {task.task_id}")
        for role in ("planner", "coder", "reviewer"):
            if canonical_json(row.get("student_models", {}).get(role)) != canonical_json(
                getattr(config.models, role).model_dump(mode="json")
            ):
                raise LegacyImportError(
                    f"Native Level-1 {role} model/generation config mismatch for {task.task_id}"
                )
        if canonical_json(row.get("teacher_model")) != canonical_json(
            config.models.teacher.model_dump(mode="json")
        ):
            raise LegacyImportError(
                f"Native Level-1 teacher model/generation config mismatch for {task.task_id}"
            )

    source_transitions = index["transitions"].get(
        (task.task_id, source_compatibility), {}
    )
    if not source_transitions:
        return _reconstruct_partitioned_level1(
            legacy_dir=legacy_dir,
            dataset=dataset,
            dataset_version=dataset_version,
            task=task,
            task_record=task_record,
            logical_rollout_seed=logical_rollout_seed,
            run_id=run_id,
            compatibility=compatibility,
            config=config,
            root_cache=root_cache,
            transition_cache=transition_cache,
            trajectory_cache=trajectory_cache,
            state_log=state_log,
            source_compatibility=source_compatibility,
            source_trajectories=source_trajectories,
        )

    transition_by_prefix: dict[str, dict[str, Any]] = {}
    for trajectory in source_trajectories.values():
        for transition_id in trajectory.get("transition_ids", []):
            row = source_transitions.get(str(transition_id))
            if row is not None:
                transition_by_prefix[str(row.get("prefix", ""))] = row
    required_prefixes = {"S", "V", "SS", "SV", "VS", "SSS", "SSV", "SVS", "VSS"}
    if not required_prefixes.issubset(transition_by_prefix):
        return False
    source_root_state_id = str(source_trajectories["SSS"]["checkpoint_state_ids"][0])
    roots = [
        row for row in index["roots"].get((task.task_id, source_compatibility), [])
        if str(row.get("state_id")) == source_root_state_id
    ]
    if not roots:
        return False
    source_root = roots[-1]

    root_identity = {
        "dataset": dataset,
        "task_sha256": sha256_json(task.model_dump(mode="json")),
        "rollout_index": 0,
        "rollout_seed": logical_rollout_seed,
        "role": "planner",
    }
    current_root_key = cache_key("root", compatibility, root_identity)
    if root_cache.get(current_root_key) is not None:
        return True

    state_ids = {source_root_state_id}
    for row in transition_by_prefix.values():
        state_ids.add(str(row.get("input_state_id", "")))
        if int(row.get("checkpoint", 0)) < 3:
            state_ids.add(str(row.get("output_state_id", "")))
    source_states = index["states"]
    missing_states = state_ids.difference(source_states)
    if missing_states:
        return False
    existing_state_ids = {
        str(row.get("state_id")) for row in state_log.read_all()
        if row.get("record_type") == "checkpoint_state"
    }
    for state_id in sorted(state_ids):
        if state_id not in existing_state_ids:
            state_log.append(source_states[state_id])

    model_pair = build_model_pair_snapshot(config)
    root_cache.put({
        "record_type": "cbv_root",
        "cache_key": current_root_key,
        "compatibility_fingerprint": compatibility,
        "status": "completed",
        "dataset": dataset,
        "model_pair_id": model_pair.model_pair_id,
        "model_pair_fingerprint": model_pair.model_pair_fingerprint,
        "task_id": task.task_id,
        "rollout_index": 0,
        "rollout_seed": logical_rollout_seed,
        "state_id": source_root_state_id,
        "invocation_id": source_root.get("invocation_id"),
        "model": config.models.planner.model_dump(mode="json"),
        "input_tokens": int(source_root.get("input_tokens", 0)),
        "output_tokens": int(source_root.get("output_tokens", 0)),
        "cost_usd": float(source_root.get("cost_usd", 0.0)),
        "latency_seconds": float(source_root.get("latency_seconds", 0.0)),
        "source": "legacy_level1",
        "native_source_dir": str(legacy_dir),
        "native_source_compatibility": source_compatibility,
        "native_rollout_seed": source_root.get("rollout_seed"),
    })

    current_transitions: dict[str, TransitionRecord] = {}
    for prefix in sorted(required_prefixes, key=lambda value: (len(value), value)):
        source = dict(transition_by_prefix[prefix])
        identity = {
            "dataset": dataset,
            "task_id": task.task_id,
            "rollout_index": 0,
            "rollout_seed": logical_rollout_seed,
            "prefix": prefix,
            "action": source["action"],
            "input_state_id": source["input_state_id"],
        }
        key = cache_key("transition", compatibility, identity)
        transition_id = content_id("cbv_transition", {"cache_key": key, "prefix": prefix})
        provenance = dict(source.get("provenance") or {})
        provenance.update({
            "native_level1_import": True,
            "native_source_dir": str(legacy_dir),
            "native_source_transition_id": source.get("transition_id"),
            "native_source_compatibility": source_compatibility,
            "native_rollout_seed": source.get("rollout_seed"),
        })
        source.update({
            "transition_id": transition_id,
            "cache_key": key,
            "compatibility_fingerprint": compatibility,
            "model_pair_id": model_pair.model_pair_id,
            "model_pair_fingerprint": model_pair.model_pair_fingerprint,
            "run_id": run_id,
            "rollout_seed": logical_rollout_seed,
            "source": "legacy_level1",
            "provenance": provenance,
        })
        transition = TransitionRecord.model_validate(source)
        transition_cache.put(transition.model_dump(mode="json"))
        current_transitions[prefix] = transition

    for sequence in sorted(required):
        source = source_trajectories[sequence]
        identity = {
            "dataset": dataset,
            "task_id": task.task_id,
            "rollout_index": 0,
            "rollout_seed": logical_rollout_seed,
            "sequence": sequence,
        }
        key = cache_key("trajectory", compatibility, identity)
        edges = [current_transitions[sequence[:depth]] for depth in (1, 2, 3)]
        provenance = dict(source.get("provenance") or {})
        provenance.update({
            "native_level1_import": True,
            "native_source_dir": str(legacy_dir),
            "native_source_trajectory_id": source.get("trajectory_id"),
            "native_source_compatibility": source_compatibility,
            "native_rollout_seed": source.get("rollout_seed"),
        })
        copied = dict(source)
        copied.update({
            "trajectory_id": content_id("trajectory", {"cache_key": key}),
            "branch_id": content_id("branch", identity),
            "parent_branch_id": content_id(
                "branch_prefix", {**identity, "prefix": sequence[:2]}
            ),
            "cache_key": key,
            "compatibility_fingerprint": compatibility,
            "run_id": run_id,
            "model_pair_id": model_pair.model_pair_id,
            "model_pair_fingerprint": model_pair.model_pair_fingerprint,
            "model_pair": model_pair.model_dump(mode="json"),
            "dataset": dataset,
            "dataset_version": dataset_version,
            "difficulty": task_record.difficulty,
            "split": task_record.split,
            "level": 1,
            "student_models": {
                role: getattr(config.models, role).model_dump(mode="json")
                for role in ("planner", "coder", "reviewer")
            },
            "teacher_model": config.models.teacher.model_dump(mode="json"),
            "prompt_versions": {
                role: getattr(config.models, role).prompt_version
                for role in ("planner", "coder", "reviewer", "teacher")
            },
            "rollout_seed": logical_rollout_seed,
            "transition_ids": [edge.transition_id for edge in edges],
            "source": "legacy_level1",
            "provenance": provenance,
        })
        copied = {
            key: value for key, value in copied.items()
            if key in TrajectoryRecord.model_fields
        }
        trajectory_cache.put(TrajectoryRecord.model_validate(copied).model_dump(mode="json"))
    return True


def import_legacy_level1(
    *,
    legacy_dir: Path,
    dataset: str,
    dataset_version: str,
    task,
    task_record,
    logical_rollout_seed: int,
    run_id: str,
    compatibility: str,
    config: CBVConfig,
    root_cache: ExactCache,
    transition_cache: ExactCache,
    trajectory_cache: ExactCache,
    state_log: JsonlStore,
) -> bool:
    """Import SSS/VSS/SVS/SSV without inference when strict deterministic checks pass.

    Historical Phase-1 used checkpoint-specific pair seeds, so import is allowed only when all
    generation temperatures are zero and repeated_rollouts is one. Actual historical seeds remain
    in provenance; stochastic CBV studies must generate a native tree instead.
    """
    if config.collection.repeated_rollouts != 1:
        return False
    if any(getattr(config.models, role).temperature != 0.0
           for role in ("planner", "coder", "reviewer", "teacher")):
        return False
    if (legacy_dir / "trajectories.jsonl").exists():
        imported_native = _import_native_cbv_level1(
            legacy_dir=legacy_dir,
            dataset=dataset,
            dataset_version=dataset_version,
            task=task,
            task_record=task_record,
            logical_rollout_seed=logical_rollout_seed,
            run_id=run_id,
            compatibility=compatibility,
            config=config,
            root_cache=root_cache,
            transition_cache=transition_cache,
            trajectory_cache=trajectory_cache,
            state_log=state_log,
        )
        if imported_native:
            return True
    model_pair = build_model_pair_snapshot(config)
    manifests = _read(legacy_dir / "manifests.jsonl")
    baselines = [row for row in _read(legacy_dir / "baseline.jsonl")
                 if row.get("task_id") == task.task_id and row.get("status") == "completed"]
    pairs = [row for row in _read(legacy_dir / "intervention_pairs.jsonl")
             if row.get("task_id") == task.task_id and row.get("status") == "completed"
             and int(row.get("repetition", 0)) == 0]
    if not baselines or {int(row["checkpoint"]) for row in pairs} != {1, 2, 3}:
        return False
    baseline = baselines[-1]
    legacy_run = baseline["run_id"]
    manifest = next((row for row in reversed(manifests) if row.get("run_id") == legacy_run), None)
    if manifest is None:
        raise LegacyImportError(f"Legacy manifest missing for run {legacy_run}")
    legacy_models = manifest.get("config", {}).get("models", {})
    for role in ("planner", "coder", "reviewer", "teacher"):
        if canonical_json(legacy_models.get(role)) != canonical_json(
            getattr(config.models, role).model_dump(mode="json")
        ):
            raise LegacyImportError(f"Legacy {role} model/generation config is not compatible")

    states = {row["state_id"]: CheckpointState.model_validate(row)
              for row in _read(legacy_dir / "checkpoint_states.jsonl")}
    for row in baseline.get("checkpoint_states", []):
        states[row["state_id"]] = CheckpointState.model_validate(row)
    invocations = {row["invocation_id"]: InvocationRecord.model_validate(row)
                   for row in _read(legacy_dir / "llm_invocations.jsonl")}
    evaluations = {row["evaluation_id"]: EvaluationResult.model_validate(row)
                   for row in _read(legacy_dir / "evaluations.jsonl")}
    baseline_states = [CheckpointState.model_validate(row) for row in baseline["checkpoint_states"]]

    root_identity = {
        "dataset": dataset, "task_sha256": sha256_json(task.model_dump(mode="json")),
        "rollout_index": 0, "rollout_seed": logical_rollout_seed, "role": "planner",
    }
    root_key = cache_key("root", compatibility, root_identity)
    planner_invocation = invocations[baseline_states[0].invocation_id]
    if root_cache.get(root_key) is not None:
        return True
    for state in states.values():
        state_log.append(state)
    root_cache.put({
        "record_type": "cbv_root", "cache_key": root_key,
        "compatibility_fingerprint": compatibility, "status": "completed", "dataset": dataset,
        "model_pair_id": model_pair.model_pair_id,
        "model_pair_fingerprint": model_pair.model_pair_fingerprint,
        "task_id": task.task_id, "rollout_index": 0, "rollout_seed": logical_rollout_seed,
        "state_id": baseline_states[0].state_id,
        "invocation_id": planner_invocation.invocation_id,
        "model": planner_invocation.llm_config.model_dump(mode="json"),
        "input_tokens": planner_invocation.input_tokens, "output_tokens": planner_invocation.output_tokens,
        "cost_usd": planner_invocation.cost_usd, "latency_seconds": planner_invocation.latency_seconds,
        "source": "legacy_level1", "legacy_run_id": legacy_run,
        "legacy_rollout_seed": baseline["rollout_seed"],
    })

    pair_by_cp = {int(row["checkpoint"]): row for row in pairs}
    p1, p2, p3 = pair_by_cp[1], pair_by_cp[2], pair_by_cp[3]
    v1_ids = p1["verify"]["trajectory_state_ids"]
    v2_ids = p2["verify"]["trajectory_state_ids"]
    state_map = {
        "": baseline_states[0], "S": baseline_states[1], "SS": baseline_states[2],
        "V": states[v1_ids[1]], "VS": states[v1_ids[2]], "SV": states[v2_ids[1]],
    }
    pair_for_prefix = {"V": p1, "SV": p2, "SSV": p3}
    output_state_for_prefix = {"S": state_map["S"], "SS": state_map["SS"],
                               "V": state_map["V"], "VS": state_map["VS"], "SV": state_map["SV"]}
    terminal_artifact = {
        "SSS": baseline_states[2].current_artifact,
        "VSS": state_map["VS"].current_artifact,
        "SVS": state_map["SV"].current_artifact,
        "SSV": p3["corrected_artifact"],
    }

    transition_rows: dict[str, TransitionRecord] = {}
    for prefix in ("S", "V", "SS", "SV", "VS", "SSS", "SSV", "SVS", "VSS"):
        checkpoint, action, parent = len(prefix), prefix[-1], prefix[:-1]
        input_state = state_map[parent]
        identity = {"dataset": dataset, "task_id": task.task_id, "rollout_index": 0,
                    "rollout_seed": logical_rollout_seed, "prefix": prefix, "action": action,
                    "input_state_id": input_state.state_id}
        key = cache_key("transition", compatibility, identity)
        pair = pair_for_prefix.get(prefix)
        teacher_invocation = invocations[pair["teacher_invocation_id"]] if pair else None
        output_state = output_state_for_prefix.get(prefix)
        downstream = invocations[output_state.invocation_id] if output_state is not None else None
        output_artifact = (output_state.current_artifact if output_state is not None
                           else terminal_artifact[prefix])
        transition_id = content_id("cbv_transition", {"cache_key": key, "prefix": prefix})
        row = TransitionRecord(
            transition_id=transition_id, cache_key=key, compatibility_fingerprint=compatibility,
            model_pair_id=model_pair.model_pair_id,
            model_pair_fingerprint=model_pair.model_pair_fingerprint,
            run_id=run_id, dataset=dataset, task_id=task.task_id, rollout_index=0,
            rollout_seed=logical_rollout_seed, checkpoint=checkpoint, prefix=prefix,
            action=action, parent_prefix=parent, parent_state_id=input_state.parent_state_id,
            input_state_id=input_state.state_id,
            output_state_id=output_state.state_id if output_state else content_id(
                "terminal", {"transition_id": transition_id, "artifact": output_artifact}),
            input_artifact=input_state.current_artifact,
            effective_artifact=(
                pair["corrected_artifact"] if pair else input_state.current_artifact
            ),
            output_artifact=output_artifact,
            preceding_artifacts=input_state.preceding_artifacts,
            teacher_model=teacher_invocation.llm_config if teacher_invocation else None,
            teacher_verdict=pair.get("teacher_verdict") if pair else None,
            teacher_changed=pair.get("teacher_changed") if pair else None,
            teacher_explanation=pair.get("teacher_explanation") if pair else None,
            teacher_invocation_id=teacher_invocation.invocation_id if teacher_invocation else None,
            teacher_messages=teacher_invocation.messages if teacher_invocation else [],
            teacher_raw_output=teacher_invocation.raw_output if teacher_invocation else None,
            teacher_input_tokens=teacher_invocation.input_tokens if teacher_invocation else 0,
            teacher_output_tokens=teacher_invocation.output_tokens if teacher_invocation else 0,
            teacher_total_tokens=(teacher_invocation.input_tokens + teacher_invocation.output_tokens)
            if teacher_invocation else 0,
            teacher_cost_usd=teacher_invocation.cost_usd if teacher_invocation else 0.0,
            teacher_latency_seconds=teacher_invocation.latency_seconds if teacher_invocation else 0.0,
            downstream_invocation_id=downstream.invocation_id if downstream else None,
            downstream_model=downstream.llm_config if downstream else None,
            downstream_input_tokens=downstream.input_tokens if downstream else 0,
            downstream_output_tokens=downstream.output_tokens if downstream else 0,
            downstream_cost_usd=downstream.cost_usd if downstream else 0.0,
            downstream_latency_seconds=downstream.latency_seconds if downstream else 0.0,
            source="legacy_level1",
            provenance={"legacy_run_id": legacy_run, "legacy_rollout_seed":
                        pair.get("rollout_seed") if pair else baseline["rollout_seed"]},
        )
        transition_cache.put(row.model_dump(mode="json"))
        transition_rows[prefix] = row

    evaluation_by_sequence = {
        "SSS": evaluations[baseline["evaluation_id"]],
        "VSS": evaluations[p1["verify"]["evaluation_id"]],
        "SVS": evaluations[p2["verify"]["evaluation_id"]],
        "SSV": evaluations[p3["verify"]["evaluation_id"]],
    }
    final_code = {"SSS": baseline["final_implementation"], "VSS": p1["verify"]["final_implementation"],
                  "SVS": p2["verify"]["final_implementation"], "SSV": p3["verify"]["final_implementation"]}
    for sequence in ("SSS", "VSS", "SVS", "SSV"):
        identity = {"dataset": dataset, "task_id": task.task_id, "rollout_index": 0,
                    "rollout_seed": logical_rollout_seed, "sequence": sequence}
        key = cache_key("trajectory", compatibility, identity)
        edges = [transition_rows[sequence[:depth]] for depth in (1, 2, 3)]
        teacher_edges = [edge for edge in edges if edge.action == "V"]
        evaluation = evaluation_by_sequence[sequence]
        trajectory_cache.put(TrajectoryRecord(
            trajectory_id=content_id("trajectory", {"cache_key": key}),
            branch_id=content_id("branch", identity),
            parent_branch_id=content_id("branch_prefix", {**identity, "prefix": sequence[:2]}),
            cache_key=key, compatibility_fingerprint=compatibility, run_id=run_id,
            model_pair_id=model_pair.model_pair_id,
            model_pair_fingerprint=model_pair.model_pair_fingerprint,
            model_pair=model_pair,
            dataset=dataset, dataset_version=dataset_version, task_id=task.task_id,
            difficulty=task_record.difficulty, split=task_record.split, level=1,
            workflow_version="fixed-planner-coder-reviewer-v1",
            student_models={
                "planner": config.models.planner,
                "coder": config.models.coder,
                "reviewer": config.models.reviewer,
            },
            teacher_model=config.models.teacher,
            prompt_versions={
                role: getattr(config.models, role).prompt_version
                for role in ("planner", "coder", "reviewer", "teacher")
            },
            action_sequence=sequence, actions=list(sequence), rollout_index=0,
            rollout_seed=logical_rollout_seed, transition_ids=[edge.transition_id for edge in edges],
            checkpoint_state_ids=[baseline_states[0].state_id, edges[0].output_state_id,
                                  edges[1].output_state_id],
            artifact_snapshots=[edge.effective_artifact for edge in edges], final_code=final_code[sequence],
            evaluation=evaluation, passed=evaluation.passed, evaluator_status=evaluation.status,
            teacher_calls=len(teacher_edges),
            teacher_input_tokens=sum(edge.teacher_input_tokens for edge in teacher_edges),
            teacher_output_tokens=sum(edge.teacher_output_tokens for edge in teacher_edges),
            teacher_total_tokens=sum(edge.teacher_total_tokens for edge in teacher_edges),
            teacher_cost_usd=sum(edge.teacher_cost_usd for edge in teacher_edges),
            student_input_tokens=planner_invocation.input_tokens +
            sum(edge.downstream_input_tokens for edge in edges),
            student_output_tokens=planner_invocation.output_tokens +
            sum(edge.downstream_output_tokens for edge in edges),
            student_cost_usd=planner_invocation.cost_usd + sum(edge.downstream_cost_usd for edge in edges),
            total_latency_seconds=planner_invocation.latency_seconds +
            sum(edge.teacher_latency_seconds + edge.downstream_latency_seconds for edge in edges),
            source="legacy_level1", provenance={"legacy_run_id": legacy_run,
                                                "deterministic_import_only": True},
        ).model_dump(mode="json"))
    return True
