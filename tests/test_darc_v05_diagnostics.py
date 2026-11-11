from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from agent_verify.analysis.darc_v04_data import PREFIXES, MultiHistoryState
from agent_verify.analysis.darc_v05_diagnostics import (
    _multi_verify_rows,
    _seed_task_comparison,
    detailed_replay,
)
from agent_verify.analysis.darc_v05_model import V05Prediction
from agent_verify.analysis.darc_v05_training import ConservativePolicy

SEQUENCES = ("SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV")


def _group() -> SimpleNamespace:
    rows = {
        sequence: {
            "passed": int(sequence == "VVS"),
            "teacher_total_tokens": 100 * sequence.count("V"),
        }
        for sequence in SEQUENCES
    }
    return SimpleNamespace(task=SimpleNamespace(dataset="synthetic", task_id="0"), rows=rows)


def _states() -> tuple[list[MultiHistoryState], dict[tuple[str, str], int]]:
    states = [
        MultiHistoryState(
            "task",
            "synthetic",
            "0",
            prefix,
            len(prefix) + 1,
            None,  # type: ignore[arg-type]
        )
        for prefix in PREFIXES
    ]
    return states, {("task", state.prefix): index for index, state in enumerate(states)}


def _prediction(states: list[MultiHistoryState]) -> V05Prediction:
    advantage = np.asarray(
        [0.6 if state.prefix in {"", "V"} else -0.6 for state in states],
        dtype=float,
    )
    q_skip = np.full(len(states), 0.2)
    q_verify = np.clip(q_skip + advantage, 0.0, 1.0)
    q_success = np.column_stack((q_skip, q_verify))
    q_cost = np.column_stack((np.zeros(len(states)), np.full(len(states), 100.0)))
    return V05Prediction(
        q_success=q_success,
        q_cost=q_cost,
        advantage=q_verify - q_skip,
        advantage_uncertainty=np.zeros(len(states)),
        additional_cost=np.full(len(states), 100.0),
        additional_cost_uncertainty=np.zeros(len(states)),
        effect_probabilities=np.tile(np.asarray([[0.0, 0.5, 0.5]]), (len(states), 1)),
        immediate_cost=np.full(len(states), 80.0),
        immediate_cost_uncertainty=np.zeros(len(states)),
    )


def test_detailed_replay_logs_vvs_q_values_and_strict_budget() -> None:
    states, lookup = _states()
    policy = ConservativePolicy(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    summary, tasks, decisions = detailed_replay(
        {"task": _group()},
        lookup,
        [_prediction(states)],
        [360.0],
        policy,
        reference_budget=360.0,
        token_budget=360,
        hard_call_token_cap=120,
        model_seed=1,
        order_seed=1,
        variant="synthetic",
        iteration=0,
        allowance_mode="dynamic",
        analysis_family="test",
    )
    assert tasks[0]["sequence"] == "VVS"
    assert tasks[0]["final_correctness"] == 1
    assert tasks[0]["teacher_tokens"] == 200
    assert summary["max_budget_overshoot"] == 0
    assert [row["action"] for row in decisions] == ["V", "V", "S"]
    assert all("q_skip" in row and "q_verify" in row for row in decisions)
    assert sum(int(row["helpful_selected"]) for row in decisions) == 2


def test_seed_comparison_separates_correctness_from_sequence_instability() -> None:
    common = {
        "analysis_family": "cross_seed",
        "allowance_mode": "dynamic",
        "task_key": "task",
        "dataset": "synthetic",
        "baseline_correctness": 0,
        "final_correctness": 1,
    }
    rows = [
        {
            **common,
            "model_seed": 1,
            "order_seed": 1,
            "sequence": "VVS",
            "teacher_tokens": 200,
            "teacher_calls": 2,
        },
        {
            **common,
            "model_seed": 2,
            "order_seed": 2,
            "sequence": "VVV",
            "teacher_tokens": 300,
            "teacher_calls": 3,
        },
    ]
    comparison = _seed_task_comparison(rows, [1, 2])
    assert comparison[0]["all_seeds_same_correctness"] == 1
    assert comparison[0]["all_seeds_same_sequence"] == 0
    assert comparison[0]["token_range"] == 100


def test_multi_verify_audit_detects_strict_vvs_synergy_and_selection() -> None:
    task_rows = []
    for iteration in (0, 1):
        task_rows.append(
            {
                "analysis_family": "policy_improvement",
                "allowance_mode": "dynamic",
                "model_seed": 1,
                "order_seed": 1,
                "task_key": "task",
                "variant": "conservative_dynamic",
                "iteration": iteration,
                "sequence": "VVS",
            }
        )
    rows, summary = _multi_verify_rows({"task": _group()}, task_rows, ["conservative_dynamic"])
    vvs = next(row for row in rows if row["sequence"] == "VVS")
    vvs_summary = next(row for row in summary if row["sequence"] == "VVS")
    assert vvs["strict_positive_synergy"] == 1
    assert vvs["incremental_vs_best_single"] == 1
    assert vvs_summary["selected_conservative_dynamic_i0"] == 1
    assert vvs_summary["selected_conservative_dynamic_i1"] == 1
