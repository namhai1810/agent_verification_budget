"""Structural, causal, privacy, accounting, and completeness validation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agent_verify.cbv_bench.cache import cache_key
from agent_verify.cbv_bench.config import CBVConfig
from agent_verify.cbv_bench.datasets import load_selection, project_path
from agent_verify.cbv_bench.model_metadata import build_model_pair_snapshot
from agent_verify.cbv_bench.sequences import required_sequences
from agent_verify.cbv_bench.tree import PrivacyGuard
from agent_verify.logging import canonical_json
from agent_verify.schemas import CheckpointState, Message


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _contains_private_key(value: Any) -> bool:
    if isinstance(value, dict):
        if {"test", "canonical_solution", "private_test_cases"}.intersection(value):
            return True
        return any(_contains_private_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_private_key(item) for item in value)
    return False


def validate_cbv_data(config: CBVConfig, project_root: Path | None = None,
                      *, deep_privacy: bool = False) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[3]
    results = project_path(config.collection.results_dir, root)
    trajectories = _rows(results / "trajectories.jsonl")
    transitions = _rows(results / "cache" / "transitions.jsonl")
    states = _rows(results / "checkpoint_states.jsonl")
    errors_log = _rows(results / "errors.jsonl")
    errors: list[str] = []
    warnings: list[str] = []
    expected_model_pair = build_model_pair_snapshot(config)

    compatibility_values = {row.get("compatibility_fingerprint") for row in trajectories}
    selections = [load_selection(config, spec, root) for spec in config.datasets if spec.enabled]
    allowed_task_keys = {
        (selection.spec.name, task.task_id)
        for selection in selections for task in selection.public_tasks
    }
    enabled_names = {selection.spec.name for selection in selections}
    manifests = _rows(results / "manifests.jsonl")
    matching_manifests = []
    for manifest in manifests:
        manifest_config = manifest.get("config", {})
        manifest_names = {
            row.get("name") for row in manifest_config.get("datasets", []) if row.get("enabled")
        }
        if (
            manifest.get("requested_level") == config.collection.level
            and manifest_names == enabled_names
            and manifest.get("config_fingerprint") == config.fingerprint
            and manifest.get("model_pair_fingerprint")
            == expected_model_pair.model_pair_fingerprint
        ):
            matching_manifests.append(manifest)
    if matching_manifests:
        compatibility = matching_manifests[-1].get("compatibility_fingerprint")
        trajectories = [
            row for row in trajectories
            if row.get("compatibility_fingerprint") == compatibility
            and (row.get("dataset"), row.get("task_id")) in allowed_task_keys
            and int(row.get("rollout_index", -1)) < config.collection.repeated_rollouts
        ]
        transitions = [
            row for row in transitions
            if row.get("compatibility_fingerprint") == compatibility
            and (row.get("dataset"), row.get("task_id")) in allowed_task_keys
            and int(row.get("rollout_index", -1)) < config.collection.repeated_rollouts
        ]
        errors_log = [
            row for row in errors_log
            if row.get("compatibility_fingerprint") == compatibility
            and (row.get("dataset"), row.get("task_id")) in allowed_task_keys
        ]
    else:
        compatibility = next(iter(compatibility_values), "")
        errors.append("No CBV run manifest matches the active dataset scope and level")

    state_by_id = {row.get("state_id"): row for row in states}
    transition_by_id: dict[str, dict] = {}
    for row in transitions:
        transition_id = row.get("transition_id")
        if (
            transition_id in transition_by_id
            and canonical_json(row) != canonical_json(transition_by_id[transition_id])
        ):
            errors.append(f"Conflicting duplicate transition_id: {transition_id}")
        transition_by_id[transition_id] = row
        if row.get("model_pair_id") != expected_model_pair.model_pair_id:
            errors.append(f"Transition model_pair_id mismatch in {transition_id}")
        if row.get("model_pair_fingerprint") != expected_model_pair.model_pair_fingerprint:
            errors.append(f"Transition model-pair fingerprint mismatch in {transition_id}")
        prefix = str(row.get("prefix", ""))
        checkpoint = int(row.get("checkpoint", 0))
        expected_cache_key = cache_key("transition", str(compatibility), {
            "dataset": row.get("dataset"), "task_id": row.get("task_id"),
            "rollout_index": int(row.get("rollout_index", -1)),
            "rollout_seed": int(row.get("rollout_seed", -1)),
            "prefix": prefix, "action": row.get("action"),
            "input_state_id": row.get("input_state_id"),
        })
        if row.get("cache_key") != expected_cache_key:
            errors.append(f"Transition cache key is not reproducible for {transition_id}")
        action = row.get("action")
        if checkpoint != len(prefix) or action != prefix[-1:]:
            errors.append(f"Temporal prefix/action mismatch in {transition_id}")
        teacher_id = row.get("teacher_invocation_id")
        if (action == "V") != bool(teacher_id):
            errors.append(f"Teacher invocation/action mismatch in {transition_id}")
        total = int(row.get("teacher_input_tokens", 0)) + int(row.get("teacher_output_tokens", 0))
        if total != int(row.get("teacher_total_tokens", 0)):
            errors.append(f"Teacher token accounting mismatch in {transition_id}")
        if checkpoint < 3:
            output_state = state_by_id.get(row.get("output_state_id"))
            if output_state is None:
                errors.append(f"Missing child state for {transition_id}")
            elif output_state.get("parent_state_id") != transition_id:
                if row.get("source") == "legacy_level1":
                    warnings.append(
                        f"Legacy state {output_state.get('state_id')} preserves its original parent ID; "
                        f"transition provenance was audited during import"
                    )
                else:
                    errors.append(f"Child state does not point to causal transition {transition_id}")
            if output_state is not None and output_state.get("task_id") != row.get("task_id"):
                errors.append(f"Cross-task state edge in {transition_id}")
            if output_state is not None:
                preceding = output_state.get("preceding_artifacts", [])
                effective = row.get("effective_artifact")
                position = 0 if checkpoint == 1 else 1
                if (
                    len(preceding) <= position
                    or canonical_json(preceding[position]) != canonical_json(effective)
                ):
                    errors.append(f"Child state did not consume effective artifact for {transition_id}")
                if canonical_json(output_state.get("current_artifact")) != canonical_json(
                    row.get("output_artifact")
                ):
                    errors.append(f"Transition output artifact/state mismatch for {transition_id}")

    trajectory_keys: set[tuple] = set()
    grouped: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in trajectories:
        key = (row.get("dataset"), row.get("task_id"), int(row.get("rollout_index", -1)),
               row.get("action_sequence"))
        if key in trajectory_keys:
            errors.append(f"Duplicate completed branch: {key}")
        trajectory_keys.add(key)
        if row.get("model_pair_id") != expected_model_pair.model_pair_id:
            errors.append(f"Trajectory model_pair_id mismatch in {row.get('trajectory_id')}")
        if row.get("model_pair_fingerprint") != expected_model_pair.model_pair_fingerprint:
            errors.append(
                f"Trajectory model-pair fingerprint mismatch in {row.get('trajectory_id')}"
            )
        if canonical_json(row.get("model_pair")) != canonical_json(
            expected_model_pair.model_dump(mode="json")
        ):
            errors.append(f"Incomplete/drifted model metadata in {row.get('trajectory_id')}")
        grouped[key[:3]].add(str(key[3]))
        sequence = str(row.get("action_sequence", ""))
        transition_ids = row.get("transition_ids", [])
        if len(transition_ids) != 3:
            errors.append(f"Trajectory {row.get('trajectory_id')} has {len(transition_ids)} transitions")
            continue
        edge_rows = [transition_by_id.get(value) for value in transition_ids]
        if any(edge is None for edge in edge_rows):
            errors.append(f"Trajectory {row.get('trajectory_id')} references missing transitions")
            continue
        expected_prefixes = [sequence[:depth] for depth in (1, 2, 3)]
        if [edge.get("prefix") for edge in edge_rows if edge] != expected_prefixes:
            errors.append(f"Trajectory {row.get('trajectory_id')} splices incompatible branch states")
        for previous, current in zip(edge_rows, edge_rows[1:]):
            if previous and current and previous.get("output_state_id") != current.get("input_state_id"):
                errors.append(f"Trajectory {row.get('trajectory_id')} violates state restoration causality")
        expected_trajectory_key = cache_key("trajectory", str(compatibility), {
            "dataset": row.get("dataset"), "task_id": row.get("task_id"),
            "rollout_index": int(row.get("rollout_index", -1)),
            "rollout_seed": int(row.get("rollout_seed", -1)),
            "sequence": sequence,
        })
        if row.get("cache_key") != expected_trajectory_key:
            errors.append(f"Trajectory cache key is not reproducible for {row.get('trajectory_id')}")
        terminal = edge_rows[-1] if edge_rows else None
        if terminal and row.get("final_code", "").strip() != str(
            terminal.get("effective_artifact", {}).get("code", "")
        ).strip():
            errors.append(f"Trajectory final code does not match CP3 effective artifact")
        evaluation = row.get("evaluation", {})
        if bool(row.get("passed")) != bool(evaluation.get("passed")):
            errors.append(f"Trajectory/evaluation correctness mismatch for {row.get('trajectory_id')}")
        if int(row.get("teacher_calls", -1)) != sequence.count("V"):
            errors.append(f"Trajectory {row.get('trajectory_id')} Teacher-call count mismatch")
        if int(row.get("teacher_total_tokens", -1)) != sum(
            int(edge.get("teacher_total_tokens", 0)) for edge in edge_rows if edge
        ):
            errors.append(f"Trajectory {row.get('trajectory_id')} Teacher-token sum mismatch")

    required = set(required_sequences(config.collection.level))
    expected_groups = {
        (selection.spec.name, task.task_id, rollout)
        for selection in selections for task in selection.public_tasks
        for rollout in range(config.collection.repeated_rollouts)
    }
    missing: dict[str, list[str]] = {}
    for key in sorted(expected_groups):
        absent = sorted(required.difference(grouped.get(key, set())))
        if absent:
            missing["/".join(map(str, key))] = absent
    unexpected = set(grouped).difference(expected_groups)
    if unexpected:
        warnings.append(f"Found {len(unexpected)} task/rollout groups outside the active selection")

    split_by_task: dict[tuple[str, str], set[str]] = defaultdict(set)
    for selection in selections:
        for task_id, record in selection.task_records.items():
            split_by_task[(selection.spec.name, task_id)].add(record.split)
            raw = record.model_dump(mode="json")
            if _contains_private_key(raw):
                errors.append(f"Private field leaked into selection manifest for {task_id}")
    overlap = [key for key, values in split_by_task.items() if len(values) != 1]
    if overlap:
        errors.append(f"Task IDs assigned to multiple splits: {overlap[:5]}")

    if deep_privacy:
        invocations = _rows(results / "llm_invocations.jsonl")
        messages_by_task: dict[str, list[Message]] = defaultdict(list)
        for row in invocations:
            messages_by_task[str(row.get("context", {}).get("task_id", ""))].extend(
                Message.model_validate(value) for value in row.get("messages", [])
            )
        for selection in selections:
            for task_id, private in selection.private_cases.items():
                try:
                    PrivacyGuard.assert_messages_public(messages_by_task.get(task_id, []), private)
                except RuntimeError as exc:
                    errors.append(f"{task_id}: {exc}")

    completed = len(expected_groups) - len(missing)
    report = {
        "schema_version": "cbv-validation-v2",
        "model_pair_id": expected_model_pair.model_pair_id,
        "model_pair_fingerprint": expected_model_pair.model_pair_fingerprint,
        "student_model": expected_model_pair.student_model.model_dump(mode="json"),
        "teacher_model": expected_model_pair.teacher_model.model_dump(mode="json"),
        "prompt_version": expected_model_pair.prompt_version,
        "inference_config": expected_model_pair.inference_config,
        "valid": not errors and not missing,
        "compatibility_fingerprint": compatibility,
        "scope_task_keys": [f"{dataset}::{task_id}" for dataset, task_id in sorted(allowed_task_keys)],
        "requested_level": config.collection.level,
        "expected_task_rollouts": len(expected_groups),
        "complete_task_rollouts": completed,
        "completion_rate": completed / len(expected_groups) if expected_groups else 0.0,
        "trajectory_count": len(trajectories),
        "transition_count": len(transitions),
        "logged_collection_errors": len(errors_log),
        "missing_branches": missing,
        "errors": errors,
        "warnings": warnings,
        "teacher_calls": sum(1 for row in transitions if row.get("action") == "V"),
        "teacher_tokens": sum(int(row.get("teacher_total_tokens", 0)) for row in transitions),
        "error_types": dict(Counter(row.get("error_type", "unknown") for row in errors_log)),
    }
    output = results / "validation" / compatibility / config.fingerprint[:16]
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    markdown = [
        "# CBV-Bench V2 validation", "", f"- Valid: **{report['valid']}**",
        f"- Complete task/rollouts: **{completed}/{len(expected_groups)}**",
        f"- Trajectories: **{len(trajectories)}**", f"- Transitions: **{len(transitions)}**",
        f"- Logged collection errors: **{len(errors_log)}**", "",
        "## Structural errors", "",
    ]
    markdown.extend(f"- {value}" for value in errors) if errors else markdown.append("- None")
    markdown.extend(["", "## Missing branches", ""])
    markdown.extend(f"- `{key}`: {', '.join(value)}" for key, value in missing.items()) \
        if missing else markdown.append("- None")
    (output / "validation.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report
