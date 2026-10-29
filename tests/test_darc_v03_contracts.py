from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from agent_verify.analysis.darc_data import DARCExample
from agent_verify.analysis.darc_model import (
    DARCPrediction,
    _effective_binary_weights,
    _hierarchical_value_loss,
    _outcome_probabilities,
)
from agent_verify.analysis.darc_v02 import ContinuationPrior
from agent_verify.analysis.darc_v03 import (
    RiskCalibratedPolicy,
    replay_risk_calibrated_policy,
)
from agent_verify.analysis import darc_v03_experiment, darc_v03_test_evaluation


def _examples(tasks: int = 4) -> list[DARCExample]:
    rows: list[DARCExample] = []
    for task in range(tasks):
        for checkpoint in (1, 2, 3):
            delta = int(task % 2 == 0 and checkpoint == 2)
            rows.append(
                DARCExample(
                    group_key=f"demo::task-{task}",
                    task_id=f"task-{task}",
                    dataset="humaneval",
                    difficulty="unknown",
                    checkpoint=checkpoint,
                    task_text="public task",
                    artifact_text=f"artifact cp{checkpoint}",
                    history_text="observed history",
                    current_role_text=f"role cp{checkpoint}",
                    downstream_role_texts=("observable dependency",),
                    metadata=(float(checkpoint),),
                    metadata_names=("checkpoint",),
                    y_skip=0,
                    y_verify=delta,
                    delta=delta,
                    value_label=2 if delta else 1,
                    teacher_total_tokens=5,
                )
            )
    return rows


def _prediction(examples: list[DARCExample]) -> DARCPrediction:
    probabilities = np.tile(np.asarray([[0.10, 0.20, 0.70]]), (len(examples), 1))
    return DARCPrediction(
        probabilities=probabilities,
        advantage=probabilities[:, 2] - probabilities[:, 0],
        advantage_uncertainty=np.zeros(len(examples)),
        expected_total_tokens=np.full(len(examples), 5.0),
        upper_total_tokens=np.full(len(examples), 7.0),
        latency_seconds=0.0,
    )


def test_hierarchical_probabilities_factorize_exactly() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 0.0], [2.0, -1.0]], dtype=torch.float32)
    probabilities = _outcome_probabilities(
        logits, head_mode="hierarchical", temperature=1.0
    )
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    assert torch.all(probabilities >= 0)
    # Zero logits imply P(Change)=0.5 and an even Helpful/Harmful split.
    assert torch.allclose(
        probabilities[0], torch.tensor([0.25, 0.50, 0.25]), atol=1e-6
    )


def test_no_change_rows_do_not_supervise_direction() -> None:
    torch = pytest.importorskip("torch")
    labels = torch.ones(3, dtype=torch.long)
    weights = torch.tensor(_effective_binary_weights(np.zeros(3, dtype=int), 0.999))
    direction = torch.ones(2)
    first, first_change, first_direction = _hierarchical_value_loss(
        torch.tensor([[0.1, -20.0], [0.2, 0.0], [-0.1, 20.0]]),
        labels,
        mode="balanced_ce",
        change_weights=weights,
        direction_weights=direction,
        gamma=2.0,
        change_weight=1.0,
        direction_weight=1.0,
    )
    second, second_change, second_direction = _hierarchical_value_loss(
        torch.tensor([[0.1, 20.0], [0.2, -10.0], [-0.1, -20.0]]),
        labels,
        mode="balanced_ce",
        change_weights=weights,
        direction_weights=direction,
        gamma=2.0,
        change_weight=1.0,
        direction_weight=1.0,
    )
    assert torch.allclose(first, second)
    assert torch.allclose(first_change, second_change)
    assert first_direction.item() == second_direction.item() == 0.0


def test_cost_density_can_wait_for_a_cheaper_checkpoint() -> None:
    examples = _examples(1)
    prediction = _prediction(examples)
    prediction.expected_total_tokens[:] = np.asarray([500.0, 5.0, 5.0])
    prediction.upper_total_tokens[:] = prediction.expected_total_tokens
    prior = ContinuationPrior.fit(examples, shrinkage_tasks=0.0)
    cost_aware, _trace, _audit = replay_risk_calibrated_policy(
        examples,
        prediction,
        prior,
        RiskCalibratedPolicy(
            harmful_weight=1.0,
            uncertainty_beta=0.0,
            cost_exponent=1.0,
            cost_lambda=0.0,
            continuation_weight=0.0,
            checkpoint_margins=(0.05, 0.0, 0.0),
        ),
        policy_name="cost-aware",
        token_budget=100,
        budget_fraction=1.0,
        hard_call_token_cap=7,
        reference_cost=5.0,
        seed=3,
    )
    assert cost_aware.checkpoints.tolist() == [2]
    no_cost, _trace, _audit = replay_risk_calibrated_policy(
        examples,
        prediction,
        prior,
        RiskCalibratedPolicy(
            harmful_weight=1.0,
            uncertainty_beta=0.0,
            cost_exponent=0.0,
            cost_lambda=0.0,
            continuation_weight=0.0,
            checkpoint_margins=(0.05, 0.0, 0.0),
        ),
        policy_name="no-cost",
        token_budget=100,
        budget_fraction=1.0,
        hard_call_token_cap=7,
        reference_cost=5.0,
        seed=3,
    )
    assert no_cost.checkpoints.tolist() == [1]


def test_v03_strict_budget_never_overshoots() -> None:
    examples = [replace(row, teacher_total_tokens=7) for row in _examples(3)]
    replay, _trace, audit = replay_risk_calibrated_policy(
        examples,
        _prediction(examples),
        ContinuationPrior.fit(examples),
        RiskCalibratedPolicy(continuation_weight=0.0),
        policy_name="strict",
        token_budget=10,
        budget_fraction=1.0,
        hard_call_token_cap=7,
        reference_cost=7.0,
        seed=5,
    )
    assert replay.calls == 1
    assert replay.actual_tokens == 7
    assert audit["actual_budget_overshoot_tokens"] == 0


def test_v03_config_and_source_forbid_frozen_test_development() -> None:
    config = yaml.safe_load(Path("configs/darc_v03.yaml").read_text())
    assert config["model"]["ensemble_members"] == 3
    assert config["evaluation"]["max_teacher_calls_per_task"] == 1
    assert config["evaluation"]["frozen_test_access"] == "forbidden"
    candidates = {row["name"]: row for row in config["ablation"]}
    assert candidates["loss_only_flat_focal"]["head_mode"] == "flat"
    assert candidates["hierarchical_natural"]["head_mode"] == "hierarchical"
    assert candidates["hierarchical_balanced"]["loss_mode"] == "balanced_ce"
    assert candidates["hierarchical_focal"]["loss_mode"] == "focal"
    source = inspect.getsource(darc_v03_experiment.run)
    assert '_load_split(frozen, "test")' not in source


def test_validation_gate_requires_accuracy_cost_and_harmful_safety() -> None:
    candidates = {"efficient": {}, "expensive": {}, "harmful": {}}
    policy_rows = [
        {
            "policy": "DARC v0.2 Full",
            "budget_fraction": 1.0,
            "accuracy": 0.40,
            "actual_teacher_tokens": 100,
            "teacher_calls": 10,
        },
        {
            "policy": "DARC v0.3 · efficient",
            "budget_fraction": 1.0,
            "accuracy": 0.41,
            "actual_teacher_tokens": 90,
            "teacher_calls": 9,
        },
        {
            "policy": "DARC v0.3 · expensive",
            "budget_fraction": 1.0,
            "accuracy": 0.42,
            "actual_teacher_tokens": 110,
            "teacher_calls": 11,
        },
        {
            "policy": "DARC v0.3 · harmful",
            "budget_fraction": 1.0,
            "accuracy": 0.43,
            "actual_teacher_tokens": 80,
            "teacher_calls": 8,
        },
    ]
    failures = [
        {
            "policy": row["policy"],
            "budget_fraction": 1.0,
            "harmful_selected": 1 if row["policy"].endswith("harmful") else 0,
        }
        for row in policy_rows
    ]
    selected, accepted = darc_v03_experiment._validation_gate(
        candidates, policy_rows, failures
    )
    assert selected == "efficient"
    assert accepted == ["efficient"]


def test_v03_test_evaluator_is_locked_and_never_trains_or_tunes() -> None:
    policy = darc_v03_test_evaluation._locked_policy(
        {
            "harmful_weight": 2.0,
            "uncertainty_beta": 0.5,
            "cost_exponent": 1.0,
            "cost_lambda": 0.03,
            "minimum_change_probability": 0.03,
            "continuation_weight": 0.5,
            "checkpoint_margins": [0.04, 0.02, 0.0],
            "cost_mode": "upper",
        }
    )
    assert policy.cost_mode == "upper"
    assert policy.checkpoint_margins == (0.04, 0.02, 0.0)
    source = inspect.getsource(darc_v03_test_evaluation.run)
    assert "_train_final" not in source
    assert "_tune_policy" not in source
    assert 'selection["development_selected_candidate"]' in source
    assert 'selection["selected_policy"]' in source
    assert 'split="test"' in source
