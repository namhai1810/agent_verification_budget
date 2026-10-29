from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from agent_verify.analysis.darc_data import DARCExample
from agent_verify.analysis.darc_model import DARCPrediction
from agent_verify.analysis.darc_v02 import (
    CheckpointCalibrator,
    ContinuationPrior,
    RiskCostPolicy,
    grouped_folds,
    replay_risk_cost_policy,
)


def _examples(tasks: int = 8) -> list[DARCExample]:
    rows: list[DARCExample] = []
    for task in range(tasks):
        baseline = int(task % 5 == 0)
        for checkpoint in (1, 2, 3):
            delta = int(checkpoint == 2 and task % 2 == 0 and baseline == 0)
            if checkpoint == 3 and baseline == 1:
                delta = -1
            y_skip = baseline
            y_verify = y_skip + delta
            rows.append(
                DARCExample(
                    group_key=f"demo::task-{task}",
                    task_id=f"task-{task}",
                    dataset="humaneval" if task % 2 else "livecodebench",
                    difficulty="unknown",
                    checkpoint=checkpoint,
                    task_text="public task",
                    artifact_text=f"artifact cp{checkpoint}",
                    history_text="observable history",
                    current_role_text=f"role cp{checkpoint}",
                    downstream_role_texts=("observable downstream dependency",),
                    metadata=(float(checkpoint),),
                    metadata_names=("checkpoint",),
                    y_skip=y_skip,
                    y_verify=y_verify,
                    delta=delta,
                    value_label=2 if delta > 0 else (0 if delta < 0 else 1),
                    teacher_total_tokens=5,
                )
            )
    return rows


def _prediction(examples: list[DARCExample]) -> DARCPrediction:
    probabilities = np.asarray(
        [
            [0.10, 0.25, 0.65]
            if row.checkpoint == 1
            else ([0.05, 0.25, 0.70] if row.checkpoint == 2 else [0.20, 0.60, 0.20])
            for row in examples
        ],
        dtype=float,
    )
    return DARCPrediction(
        probabilities=probabilities,
        advantage=probabilities[:, 2] - probabilities[:, 0],
        advantage_uncertainty=np.full(len(examples), 0.05),
        expected_total_tokens=np.full(len(examples), 4.0),
        upper_total_tokens=np.full(len(examples), 6.0),
        latency_seconds=0.0,
    )


def test_grouped_folds_never_split_a_task() -> None:
    examples = _examples(18)
    seen: set[str] = set()
    for train, held_out in grouped_folds(examples, folds=3, seed=17):
        train_groups = {examples[int(index)].group_key for index in train}
        held_groups = {examples[int(index)].group_key for index in held_out}
        assert train_groups.isdisjoint(held_groups)
        seen.update(held_groups)
    assert seen == {row.group_key for row in examples}


def test_checkpoint_calibration_is_probability_safe() -> None:
    examples = _examples(20)
    raw = _prediction(examples)
    calibrator = CheckpointCalibrator.fit(examples, raw, seed=19, shrinkage_events=5)
    calibrated = calibrator.transform(examples, raw)
    assert np.all(calibrated.probabilities >= 0.0)
    assert np.allclose(calibrated.probabilities.sum(axis=1), 1.0)
    assert np.allclose(
        calibrated.advantage,
        calibrated.probabilities[:, 2] - calibrated.probabilities[:, 0],
    )


def test_continuation_prior_defers_cp1_without_future_inference_features() -> None:
    examples = _examples(10)
    prediction = _prediction(examples)
    cp1 = np.asarray([row.checkpoint == 1 for row in examples])
    prediction.probabilities[cp1] = np.asarray([0.10, 0.50, 0.40])
    prediction.advantage[cp1] = 0.30
    prior = ContinuationPrior.fit(examples, shrinkage_tasks=0.0)
    policy = RiskCostPolicy(
        harmful_weight=1.0,
        uncertainty_beta=0.0,
        cost_lambda=0.0,
        continuation_weight=1.0,
        checkpoint_margins=(0.0, 0.0, 0.0),
    )
    replay, trace, _audit = replay_risk_cost_policy(
        examples,
        prediction,
        prior,
        policy,
        policy_name="test",
        token_budget=100,
        budget_fraction=1.0,
        hard_call_token_cap=5,
        reference_cost=5.0,
        seed=7,
    )
    assert replay.actual_tokens <= replay.token_budget
    no_wait, _no_wait_trace, _no_wait_audit = replay_risk_cost_policy(
        examples,
        prediction,
        prior,
        replace(policy, continuation_weight=0.0),
        policy_name="no-wait",
        token_budget=100,
        budget_fraction=1.0,
        hard_call_token_cap=5,
        reference_cost=5.0,
        seed=7,
    )
    assert np.sum(replay.checkpoints == 1) < np.sum(no_wait.checkpoints == 1)
    cp1_trace = [row for row in trace if row["checkpoint"] == 1]
    assert cp1_trace
    assert all("actual_delta" in row for row in cp1_trace)  # audit-only, never used by utility
    assert all(
        0.0 <= row["wait_utility"] <= prior.value(row["dataset"], 1)
        for row in cp1_trace
    )


def test_risk_and_cost_utility_can_reject_high_raw_advantage() -> None:
    examples = _examples(4)
    prediction = _prediction(examples)
    prediction.probabilities[:, 0] = 0.40
    prediction.probabilities[:, 2] = 0.50
    prediction.probabilities[:, 1] = 0.10
    prior = ContinuationPrior.fit(examples)
    policy = RiskCostPolicy(
        harmful_weight=3.0,
        uncertainty_beta=1.0,
        cost_lambda=0.1,
        continuation_weight=0.0,
        checkpoint_margins=(0.0, 0.0, 0.0),
    )
    replay, _trace, _audit = replay_risk_cost_policy(
        examples,
        prediction,
        prior,
        policy,
        policy_name="test",
        token_budget=100,
        budget_fraction=1.0,
        hard_call_token_cap=5,
        reference_cost=5.0,
        seed=3,
    )
    assert replay.calls == 0


def test_hard_reservation_guarantees_no_overshoot() -> None:
    examples = [replace(row, teacher_total_tokens=7) for row in _examples(3)]
    replay, _trace, audit = replay_risk_cost_policy(
        examples,
        _prediction(examples),
        ContinuationPrior.fit(examples),
        RiskCostPolicy(continuation_weight=0.0),
        policy_name="test",
        token_budget=10,
        budget_fraction=1.0,
        hard_call_token_cap=7,
        reference_cost=7.0,
        seed=5,
    )
    assert replay.calls == 1
    assert replay.actual_tokens == 7
    assert audit["actual_budget_overshoot_tokens"] == 0


def test_config_keeps_ranking_out_of_main_selection() -> None:
    config = yaml.safe_load(Path("configs/darc_v02.yaml").read_text(encoding="utf-8"))
    candidates = {row["name"]: row for row in config["loss_ablation"]}
    assert candidates["focal"]["main_eligible"] is True
    assert candidates["balanced"]["main_eligible"] is True
    assert candidates["balanced_ranking_ablation"]["paired_ranking"] is True
    assert candidates["balanced_ranking_ablation"]["main_eligible"] is False
