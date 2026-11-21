from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from agent_verify.analysis.darc_fulltree_data import (
    PREFIX_INDEX,
    SEQUENCES,
    FullTreeDataset,
    assert_disjoint_folds,
)
from agent_verify.analysis.darc_fulltree_experiment import _aggregate
from agent_verify.analysis.darc_fulltree_objective import (
    edge_expected_tokens,
    fulltree_expectation,
    trajectory_probabilities,
)


def _terminal_costs(edge: object) -> object:
    torch = pytest.importorskip("torch")
    rows = []
    for task in range(edge.shape[0]):
        values = []
        for sequence in SEQUENCES:
            prefix = ""
            cost = torch.zeros((), dtype=edge.dtype)
            for action in sequence:
                if action == "V":
                    cost = cost + edge[task, PREFIX_INDEX[prefix]]
                prefix += action
            values.append(cost)
        rows.append(torch.stack(values))
    return torch.stack(rows)


def test_eight_trajectory_probabilities_sum_to_one() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(5)
    probability = torch.sigmoid(torch.randn(11, 7))
    paths = trajectory_probabilities(probability)
    assert paths.shape == (11, 8)
    assert torch.allclose(paths.sum(dim=1), torch.ones(11), atol=1e-6)


def test_shared_prefix_cost_is_not_double_counted() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    probability = torch.sigmoid(torch.randn(4, 7))
    edge = torch.randint(1, 200, (4, 7), dtype=torch.float32)
    terminal = _terminal_costs(edge)
    outcomes = torch.zeros(4, 8)
    result = fulltree_expectation(
        probability,
        outcomes,
        terminal,
        target_tokens=100,
        reference_tokens=500,
        lagrange_multiplier=0.2,
    )
    assert torch.allclose(
        edge_expected_tokens(probability, edge), result.expected_tokens, atol=1e-5
    )


def test_gradient_flows_through_all_three_decision_depths() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(13)
    logits = torch.randn(3, 7, requires_grad=True)
    probability = torch.sigmoid(logits)
    outcomes = torch.tensor(
        [
            [0, 1, 0, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    edge = torch.arange(1, 22, dtype=torch.float32).reshape(3, 7)
    result = fulltree_expectation(
        probability,
        outcomes,
        _terminal_costs(edge),
        target_tokens=25,
        reference_tokens=100,
        lagrange_multiplier=0.03,
    )
    result.loss.backward()
    assert logits.grad is not None
    # CP1=root, CP2=S/V, CP3=four length-two prefixes.
    assert float(logits.grad[:, :1].abs().sum()) > 0
    assert float(logits.grad[:, 1:3].abs().sum()) > 0
    assert float(logits.grad[:, 3:].abs().sum()) > 0


def test_remaining_allowance_uses_only_completed_prefix_cost() -> None:
    tree = FullTreeDataset(
        task_keys=("task",),
        datasets=("humaneval",),
        state_indices=np.arange(7, dtype=np.int64).reshape(1, 7),
        outcomes=np.zeros((1, 8), dtype=np.float32),
        trajectory_tokens=np.zeros((1, 8), dtype=np.float32),
        edge_tokens=np.ones((1, 7), dtype=np.float32),
        prefix_spent=np.asarray([[0, 0, 10, 0, 20, 10, 30]], dtype=np.float32),
    )
    remaining = tree.remaining_allowance(25)
    assert remaining.tolist() == [[25.0, 25.0, 15.0, 25.0, 5.0, 15.0, 0.0]]


def test_task_level_folds_are_disjoint_and_complete() -> None:
    keys = tuple(f"task-{index}" for index in range(10))
    folds = {key: index % 5 for index, key in enumerate(keys)}
    assert_disjoint_folds(keys, folds, 5)
    with pytest.raises(AssertionError, match="cover"):
        assert_disjoint_folds(keys, {key: 0 for key in keys[:-1]}, 5)


def test_fulltree_config_locks_protocol_and_modes() -> None:
    config = yaml.safe_load(Path("configs/darc_qwen38_fulltree.yaml").read_text())
    assert config["data"]["expected_tasks"] == 210
    assert config["reproducibility"]["folds"] == 5
    assert len(config["reproducibility"]["seeds"]) == 3
    assert config["crossval"]["variants"] == ["cp3_anchor", "no_anchor"]
    assert config["evaluation"]["frozen_test_access"] == "forbidden"
    assert config["final_fit"]["label"] == "trained_on_all_210_not_independently_evaluated"
    runner = Path("scripts/run_darc_fulltree.sh").read_text()
    for mode in ("--dry-run", "--smoke", "--crossval", "--final-fit", "--resume"):
        assert mode in runner


def test_shadow_allowance_is_not_a_training_feasibility_mask() -> None:
    training = Path("src/agent_verify/analysis/darc_fulltree_training.py").read_text()
    experiment = Path("src/agent_verify/analysis/darc_fulltree_experiment.py").read_text()
    assert "allowance_tensor > 0" not in training
    assert "allowance_tensor > 0" not in experiment
    assert "remaining_global >= hard_reserve" in experiment


def test_summary_has_paired_ci_token_saving_and_pareto_flag() -> None:
    rows = []
    outcomes = {
        "Never Verify (SSS)": [0, 1, 0],
        "VVV": [1, 1, 0],
        "Diagnostic oracle": [1, 1, 1],
        "DARC Full-Tree · CP3 anchor": [1, 1, 0],
    }
    for policy, correctness in outcomes.items():
        for index, correct in enumerate(correctness):
            rows.append(
                {
                    "policy": policy,
                    "budget_fraction": 1.0,
                    "seed": 1,
                    "task_key": f"task-{index}",
                    "correct": correct,
                    "teacher_tokens": 0 if policy == "Never Verify (SSS)" else 10,
                    "teacher_calls": 0 if policy == "Never Verify (SSS)" else 1,
                    "cp1_calls": 0,
                    "cp2_calls": 0,
                    "cp3_calls": 0 if policy == "Never Verify (SSS)" else 1,
                    "helpful_calls": 0,
                    "harmful_calls": 0,
                    "no_change_calls": 0,
                }
            )
    summary = _aggregate(rows, repetitions=100, seed=7)
    learned = next(row for row in summary if "CP3 anchor" in str(row["policy"]))
    assert learned["paired_gain_vs_never_pp"] > 0
    assert learned["token_saving_vs_vvv"] == 0
    assert isinstance(learned["pareto_nondominated"], bool)
