"""Audited import of deterministic Phase-1 Level-1 branches into the CBV cache."""

from __future__ import annotations

import json
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
