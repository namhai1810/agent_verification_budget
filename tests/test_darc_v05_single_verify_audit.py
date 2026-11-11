from __future__ import annotations

from types import SimpleNamespace

from agent_verify.analysis.darc_v04_data import PREFIXES
from agent_verify.analysis.darc_v05_single_verify_audit import (
    SINGLE_LABEL,
    _replay_policy,
    _task_comparisons,
)

SEQUENCES = ("SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV")


def _group(*, successful: set[str] | None = None) -> SimpleNamespace:
    successful = successful or set()
    rows = {
        sequence: {
            "passed": int(sequence in successful),
            "teacher_total_tokens": 100 * sequence.count("V"),
        }
        for sequence in SEQUENCES
    }
    task = SimpleNamespace(dataset="synthetic", task_id="0")
    return SimpleNamespace(task=task, rows=rows)


def _lookup() -> dict[tuple[str, str], int]:
    return {("task", prefix): index for index, prefix in enumerate(PREFIXES)}


def _always_verify(_task_key: str, _order: int):
    def provider(_observation):
        return "V", {"predicted_cost": 100.0, "q_skip": 0.0, "q_verify": 1.0}

    return provider


def test_single_verify_constraint_replays_branch_consistent_vss() -> None:
    unrestricted, unrestricted_tasks, _ = _replay_policy(
        {"task": _group(successful={"VSS", "VVV"})},
        _lookup(),
        _always_verify,
        label="unrestricted",
        token_budget=1000,
        budget_fraction=1.0,
        hard_call_token_cap=100,
        seed=1,
        max_verifies_per_task=None,
    )
    replay, tasks, decisions = _replay_policy(
        {"task": _group(successful={"VSS", "VVV"})},
        _lookup(),
        _always_verify,
        label=SINGLE_LABEL,
        token_budget=1000,
        budget_fraction=1.0,
        hard_call_token_cap=100,
        seed=1,
        max_verifies_per_task=1,
    )
    assert unrestricted.calls == 3
    assert unrestricted_tasks[0]["sequence"] == "VVV"
    assert replay.calls == 1
    assert tasks[0]["sequence"] == "VSS"
    assert [row["prefix_before"] for row in decisions] == ["START", "V", "VS"]
    assert [row["proposed_action"] for row in decisions] == ["V", "V", "V"]
    assert [row["action"] for row in decisions] == ["V", "S", "S"]
    assert [row["forced_by_single_verify"] for row in decisions] == [0, 1, 1]


def test_strict_global_budget_forces_skip_without_call_reserve() -> None:
    def feasible_factory(_task_key: str, _order: int):
        def provider(observation):
            action = "V" if observation.global_remaining_budget >= 100 else "S"
            return action, {"predicted_cost": 100.0}

        return provider

    replay, tasks, _decisions = _replay_policy(
        {"task": _group(successful={"SSS"})},
        _lookup(),
        feasible_factory,
        label=SINGLE_LABEL,
        token_budget=99,
        budget_fraction=0.1,
        hard_call_token_cap=100,
        seed=1,
        max_verifies_per_task=1,
    )
    assert replay.actual_tokens == 0
    assert tasks[0]["sequence"] == "SSS"


def test_loss_analysis_detects_another_successful_single_verify() -> None:
    groups = {"task": _group(successful={"SSV", "VVV"})}
    common = {
        "seed": 1,
        "budget_fraction": 1.0,
        "token_budget": 1000,
        "task_key": "task",
        "dataset": "synthetic",
        "task_order": 0,
        "baseline_correctness": 0,
        "helpful_calls": 0,
        "harmful_calls": 0,
        "no_change_calls": 0,
    }
    task_rows = [
        {
            **common,
            "policy": "DARC v0.5 Myopic i0 (unrestricted)",
            "sequence": "VVV",
            "final_correctness": 1,
            "teacher_tokens": 300,
            "teacher_calls": 3,
        },
        {
            **common,
            "policy": SINGLE_LABEL,
            "sequence": "VSS",
            "final_correctness": 0,
            "teacher_tokens": 100,
            "teacher_calls": 1,
        },
    ]
    comparisons, losses = _task_comparisons(task_rows, groups)
    assert len(comparisons) == 1
    assert len(losses) == 1
    assert losses[0]["another_single_verify_can_solve"] == 1
    assert losses[0]["successful_single_verify_sequences"] == "SSV"
