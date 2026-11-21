"""Full-tree tensors for leakage-safe Qwen3.8 policy optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agent_verify.analysis.cbv_v01_experiment import _passed
from agent_verify.analysis.darc_v04_data import MultiHistoryState, prefix_cost, verify_cost

PREFIXES = ("", "S", "V", "SS", "SV", "VS", "VV")
SEQUENCES = ("SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV")
PREFIX_INDEX = {prefix: index for index, prefix in enumerate(PREFIXES)}


@dataclass(frozen=True)
class FullTreeDataset:
    """Task-level trees whose state indices point into one EncodedDARC matrix."""

    task_keys: tuple[str, ...]
    datasets: tuple[str, ...]
    state_indices: np.ndarray  # [tasks, 7]
    outcomes: np.ndarray  # [tasks, 8]
    trajectory_tokens: np.ndarray  # [tasks, 8]
    edge_tokens: np.ndarray  # [tasks, 7], cost of Verify at each prefix
    prefix_spent: np.ndarray  # [tasks, 7], cost already observed before the action

    def subset(self, rows: np.ndarray | list[int]) -> FullTreeDataset:
        selected = np.asarray(rows, dtype=np.int64)
        return FullTreeDataset(
            task_keys=tuple(self.task_keys[int(index)] for index in selected),
            datasets=tuple(self.datasets[int(index)] for index in selected),
            state_indices=self.state_indices[selected],
            outcomes=self.outcomes[selected],
            trajectory_tokens=self.trajectory_tokens[selected],
            edge_tokens=self.edge_tokens[selected],
            prefix_spent=self.prefix_spent[selected],
        )

    def remaining_allowance(self, initial_allowance: float) -> np.ndarray:
        """Observable allowance after completed calls on each prefix."""

        return np.maximum(float(initial_allowance) - self.prefix_spent, 0.0).astype(
            np.float32
        )


def _path_edge_cost(edge_tokens: np.ndarray, sequence: str) -> int:
    prefix = ""
    total = 0
    for action in sequence:
        if action == "V":
            total += int(edge_tokens[PREFIX_INDEX[prefix]])
        prefix += action
    return total


def build_fulltree_dataset(
    groups: dict[str, Any],
    states: list[MultiHistoryState],
    lookup: dict[tuple[str, str], int],
) -> FullTreeDataset:
    """Build and validate exact shared-prefix trees.

    The equality between path-summed edge cost and recorded terminal cumulative
    cost is the central no-double-counting contract.
    """

    keys = tuple(sorted(groups))
    state_rows: list[list[int]] = []
    outcome_rows: list[list[int]] = []
    token_rows: list[list[int]] = []
    edge_rows: list[list[int]] = []
    spent_rows: list[list[int]] = []
    datasets: list[str] = []
    for key in keys:
        group = groups[key]
        state_rows.append([lookup[(key, prefix)] for prefix in PREFIXES])
        outcome_rows.append([int(_passed(group.rows[sequence])) for sequence in SEQUENCES])
        recorded = [
            int(group.rows[sequence].get("teacher_total_tokens", 0) or 0)
            for sequence in SEQUENCES
        ]
        edges = [int(verify_cost(group, prefix)) for prefix in PREFIXES]
        spent = [int(prefix_cost(group, prefix)) for prefix in PREFIXES]
        reconstructed = [_path_edge_cost(np.asarray(edges), sequence) for sequence in SEQUENCES]
        if reconstructed != recorded:
            raise ValueError(
                f"Shared-prefix token accounting mismatch for {key}: "
                f"recorded={recorded}, reconstructed={reconstructed}"
            )
        if any(value < 0 for value in (*recorded, *edges, *spent)):
            raise ValueError(f"Negative Teacher cost in {key}")
        token_rows.append(recorded)
        edge_rows.append(edges)
        spent_rows.append(spent)
        datasets.append(str(group.task.dataset).lower())

    if len(states) != 7 * len(keys):
        raise ValueError(f"Expected seven observable states/task, got {len(states)}/{len(keys)}")
    state_indices = np.asarray(state_rows, dtype=np.int64)
    expected = np.arange(len(states), dtype=np.int64)
    if not np.array_equal(np.sort(state_indices.reshape(-1)), expected):
        raise ValueError("Full-tree state lookup is not a one-to-one cover")
    return FullTreeDataset(
        task_keys=keys,
        datasets=tuple(datasets),
        state_indices=state_indices,
        outcomes=np.asarray(outcome_rows, dtype=np.float32),
        trajectory_tokens=np.asarray(token_rows, dtype=np.float32),
        edge_tokens=np.asarray(edge_rows, dtype=np.float32),
        prefix_spent=np.asarray(spent_rows, dtype=np.float32),
    )


def assert_disjoint_folds(task_keys: tuple[str, ...], folds: dict[str, int], count: int) -> None:
    if set(task_keys) != set(folds):
        raise AssertionError("Fold mapping must cover every and only Full-8 task")
    observed = set(folds.values())
    if observed != set(range(count)):
        raise AssertionError(f"Expected folds 0..{count - 1}, got {sorted(observed)}")
    for fold in range(count):
        train = {key for key in task_keys if folds[key] != fold}
        heldout = {key for key in task_keys if folds[key] == fold}
        if not train or not heldout or train & heldout:
            raise AssertionError(f"Invalid task split at fold {fold}")
