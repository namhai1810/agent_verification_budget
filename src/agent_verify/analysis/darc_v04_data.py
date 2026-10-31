"""Leakage-safe Level-2 states and counterfactual targets for DARC v0.4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from agent_verify.analysis.cbv_v01_experiment import (
    ALL_SEQUENCES,
    ROLES,
    TaskTrajectories,
    _artifact,
    _domain_metadata,
    _passed,
)
from agent_verify.analysis.darc_data import (
    DARCExample,
    DOWNSTREAM_DEPENDENCY_TEMPLATES,
    ROLE_DESCRIPTIONS,
    downstream_dependency_texts,
)
from agent_verify.analysis.joint_value_cost_data import value_label
from agent_verify.analysis.phase2_data import build_observable_features


PREFIXES = ("", "S", "V", "SS", "SV", "VS", "VV")


@dataclass(frozen=True)
class MultiHistoryState:
    task_key: str
    dataset: str
    task_id: str
    prefix: str
    checkpoint: int
    encoder_example: DARCExample

    @property
    def state_key(self) -> str:
        return f"{self.task_key}::{self.prefix or 'START'}"


@dataclass(frozen=True)
class TargetTable:
    state_indices: np.ndarray
    allowance: np.ndarray
    allowance_fraction: np.ndarray
    q_skip: np.ndarray
    q_verify: np.ndarray
    verify_mask: np.ndarray
    effect_label: np.ndarray
    verify_cost: np.ndarray


def prefix_cost(group: TaskTrajectories, prefix: str) -> int:
    sequence = prefix + "S" * (3 - len(prefix))
    return int(group.rows[sequence].get("teacher_total_tokens", 0) or 0)


def verify_cost(group: TaskTrajectories, prefix: str) -> int:
    before = prefix_cost(group, prefix)
    sequence = prefix + "V" + "S" * (2 - len(prefix))
    return max(
        0,
        int(group.rows[sequence].get("teacher_total_tokens", 0) or 0) - before,
    )


def _direct_sequence(prefix: str, action: str) -> str:
    return prefix + action + "S" * (2 - len(prefix))


def _effect_label(group: TaskTrajectories, prefix: str) -> int:
    skip = _passed(group.rows[_direct_sequence(prefix, "S")])
    verify = _passed(group.rows[_direct_sequence(prefix, "V")])
    return value_label(skip, verify)


def _target(
    group: TaskTrajectories,
    prefix: str,
    action: str,
    allowance: float,
    *,
    counterfactual: bool,
) -> float:
    direct = _direct_sequence(prefix, action)
    if not counterfactual or len(prefix) == 2:
        return float(_passed(group.rows[direct]))
    before = prefix_cost(group, prefix)
    candidates = [
        sequence
        for sequence in ALL_SEQUENCES
        if sequence.startswith(prefix + action)
        and int(group.rows[sequence].get("teacher_total_tokens", 0) or 0) - before
        <= allowance + 1e-9
    ]
    if not candidates:
        return float(_passed(group.rows[direct]))
    selected = max(
        candidates,
        key=lambda sequence: (
            _passed(group.rows[sequence]),
            -(
                int(group.rows[sequence].get("teacher_total_tokens", 0) or 0)
                - before
            ),
        ),
    )
    return float(_passed(group.rows[selected]))


def build_states(
    groups: dict[str, TaskTrajectories],
) -> tuple[list[MultiHistoryState], dict[tuple[str, str], int]]:
    states: list[MultiHistoryState] = []
    lookup: dict[tuple[str, str], int] = {}
    metadata_schema: tuple[str, ...] | None = None
    for task_key in sorted(groups):
        group = groups[task_key]
        for prefix in PREFIXES:
            checkpoint = len(prefix) + 1
            observable_sequence = prefix + "S" * (3 - len(prefix))
            row = group.rows[observable_sequence]
            current = _artifact(row, checkpoint)
            preceding = [_artifact(row, cp) for cp in range(1, checkpoint)]
            _text, metadata, names = build_observable_features(
                prompt=group.task.prompt,
                entry_point=group.task.entry_point,
                checkpoint=checkpoint,
                role=ROLES[checkpoint],
                current_artifact=current,
                preceding_artifacts=preceding,
                local_risk=0.0,
            )
            domain, domain_names = _domain_metadata(group.task)
            metadata = (*metadata, *domain)
            names = (*names, *domain_names)
            if metadata_schema is None:
                metadata_schema = names
            elif metadata_schema != names:
                raise ValueError("DARC v0.4 metadata schema changed")
            task_text = "\n".join(
                (
                    f"Dataset: {group.task.dataset}",
                    f"Difficulty: {group.task.difficulty}",
                    f"Entry point: {group.task.entry_point}",
                    f"Task:\n{group.task.prompt}",
                )
            )
            history = (
                "No previous decisions or artifacts."
                if not prefix
                else "\n".join(
                    (
                        f"Observed action history: {prefix}",
                        "Resulting observed artifacts:",
                        json.dumps(preceding, sort_keys=True, ensure_ascii=False),
                    )
                )
            )
            skip_y = _passed(group.rows[_direct_sequence(prefix, "S")])
            verify_y = _passed(group.rows[_direct_sequence(prefix, "V")])
            example = DARCExample(
                group_key=f"{task_key}::history={prefix or 'START'}",
                task_id=group.task.task_id,
                dataset=group.task.dataset,
                difficulty=group.task.difficulty,
                checkpoint=checkpoint,
                task_text=task_text,
                artifact_text=(
                    f"Current {ROLES[checkpoint]} artifact after history "
                    f"{prefix or 'START'}:\n"
                    + json.dumps(current, sort_keys=True, ensure_ascii=False)
                ),
                history_text=history,
                current_role_text=ROLE_DESCRIPTIONS[checkpoint],
                downstream_role_texts=downstream_dependency_texts(
                    checkpoint=checkpoint,
                    task_prompt=group.task.prompt,
                    entry_point=group.task.entry_point,
                    current_artifact=current,
                ),
                metadata=metadata,
                metadata_names=names,
                y_skip=skip_y,
                y_verify=verify_y,
                delta=verify_y - skip_y,
                value_label=value_label(skip_y, verify_y),
                teacher_total_tokens=verify_cost(group, prefix),
            )
            lookup[(task_key, prefix)] = len(states)
            states.append(
                MultiHistoryState(
                    task_key=task_key,
                    dataset=group.task.dataset,
                    task_id=group.task.task_id,
                    prefix=prefix,
                    checkpoint=checkpoint,
                    encoder_example=example,
                )
            )
    return states, lookup


def build_targets(
    states: list[MultiHistoryState],
    groups: dict[str, TaskTrajectories],
    *,
    allowances: list[float],
    reference_budget: float,
    counterfactual: bool,
) -> TargetTable:
    state_indices: list[int] = []
    allowance_values: list[float] = []
    fractions: list[float] = []
    q_skip: list[float] = []
    q_verify: list[float] = []
    verify_masks: list[bool] = []
    effects: list[int] = []
    costs: list[float] = []
    for allowance in allowances:
        for index, state in enumerate(states):
            group = groups[state.task_key]
            cost = verify_cost(group, state.prefix)
            state_indices.append(index)
            allowance_values.append(float(allowance))
            fractions.append(float(allowance) / max(reference_budget, 1.0))
            q_skip.append(
                _target(
                    group,
                    state.prefix,
                    "S",
                    allowance,
                    counterfactual=counterfactual,
                )
            )
            q_verify.append(
                _target(
                    group,
                    state.prefix,
                    "V",
                    allowance,
                    counterfactual=counterfactual,
                )
            )
            verify_masks.append(cost <= allowance + 1e-9)
            effects.append(_effect_label(group, state.prefix))
            costs.append(float(cost))
    return TargetTable(
        state_indices=np.asarray(state_indices, dtype=np.int64),
        allowance=np.asarray(allowance_values, dtype=np.float32),
        allowance_fraction=np.asarray(fractions, dtype=np.float32),
        q_skip=np.asarray(q_skip, dtype=np.float32),
        q_verify=np.asarray(q_verify, dtype=np.float32),
        verify_mask=np.asarray(verify_masks, dtype=bool),
        effect_label=np.asarray(effects, dtype=np.int64),
        verify_cost=np.asarray(costs, dtype=np.float32),
    )


def audit_level2(
    partitions: dict[str, dict[str, TaskTrajectories]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, groups in partitions.items():
        datasets = sorted({group.task.dataset for group in groups.values()})
        for dataset in (*datasets, "combined"):
            selected = [
                group
                for group in groups.values()
                if dataset == "combined" or group.task.dataset == dataset
            ]
            sequence_complete = sum(
                set(group.rows) == set(ALL_SEQUENCES) for group in selected
            )
            token_valid = sum(
                all(
                    int(group.rows[sequence].get("teacher_total_tokens", 0) or 0)
                    >= 0
                    for sequence in ALL_SEQUENCES
                )
                for group in selected
            )
            rows.append(
                {
                    "split": split,
                    "dataset": dataset,
                    "tasks": len(selected),
                    "trajectories": len(selected) * 8,
                    "decision_states": len(selected) * 7,
                    "sequence_complete": sequence_complete,
                    "token_valid": token_valid,
                }
            )
    return rows

