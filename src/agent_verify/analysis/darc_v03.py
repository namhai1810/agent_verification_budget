"""DARC v0.3 risk-calibrated single-verification controller.

The controller consumes only decision-time predictions.  Actual counterfactual
outcomes and costs are attached to traces after the decision for offline audit and
are never used by the policy utility.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from agent_verify.analysis.cbv_v01_experiment import ReplayResult, _stable_order
from agent_verify.analysis.darc_data import DARCExample
from agent_verify.analysis.darc_model import DARCPrediction
from agent_verify.analysis.darc_v02 import ContinuationPrior


@dataclass(frozen=True)
class RiskCalibratedPolicy:
    """Conservative value-density policy for one intervention per task."""

    harmful_weight: float = 2.0
    uncertainty_beta: float = 0.5
    cost_exponent: float = 0.5
    cost_lambda: float = 0.02
    minimum_change_probability: float = 0.0
    continuation_weight: float = 0.5
    checkpoint_margins: tuple[float, float, float] = (0.04, 0.01, 0.0)
    cost_mode: str = "expected"

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["checkpoint_margins"] = list(self.checkpoint_margins)
        return row


def _group_indices(examples: list[DARCExample]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(examples):
        grouped.setdefault(row.group_key, []).append(index)
    for indices in grouped.values():
        indices.sort(key=lambda index: examples[index].checkpoint)
    return grouped


def replay_risk_calibrated_policy(
    examples: list[DARCExample],
    prediction: DARCPrediction,
    continuation: ContinuationPrior,
    policy: RiskCalibratedPolicy,
    *,
    policy_name: str,
    token_budget: int,
    budget_fraction: float,
    hard_call_token_cap: int,
    reference_cost: float,
    seed: int,
    scope: str = "combined",
) -> tuple[ReplayResult, list[dict[str, Any]], dict[str, Any]]:
    """Sequential replay with risk-adjusted value per predicted token.

    The hard provider cap reserves worst-case tokens before each call.  Actual cost
    is used only to update spent budget after an admitted call, matching the v0.2
    strict replay contract.
    """

    if policy.cost_mode not in {"expected", "upper"}:
        raise ValueError(f"Unknown cost mode: {policy.cost_mode}")
    if hard_call_token_cap <= 0 or reference_cost <= 0:
        raise ValueError("Token caps and reference cost must be positive")
    costs = (
        prediction.expected_total_tokens
        if policy.cost_mode == "expected"
        else prediction.upper_total_tokens
    )
    grouped = _group_indices(examples)
    task_keys = _stable_order(grouped, seed)
    actual_spent = 0
    predicted_spent = 0.0
    outcomes: list[float] = []
    baselines: list[float] = []
    checkpoints: list[int] = []
    traces: list[dict[str, Any]] = []
    for order, key in enumerate(task_keys):
        tasks_remaining = len(task_keys) - order
        remaining = max(token_budget - actual_spent, 0)
        remaining_fraction = remaining / max(token_budget, 1)
        allowance = remaining / max(tasks_remaining, 1)
        chosen: int | None = None
        for index in grouped[key]:
            row = examples[index]
            probability_helpful = float(prediction.probabilities[index, 2])
            probability_harmful = float(prediction.probabilities[index, 0])
            probability_change = probability_helpful + probability_harmful
            predicted_cost = max(float(costs[index]), 1.0)
            cost_ratio = predicted_cost / max(reference_cost, 1.0)
            cost_pressure = predicted_cost / max(allowance, reference_cost, 1.0)
            expected_gain = probability_helpful - probability_harmful
            risk_adjusted_gain = (
                probability_helpful
                - policy.harmful_weight * probability_harmful
                - policy.uncertainty_beta
                * float(prediction.advantage_uncertainty[index])
            )
            value_density = risk_adjusted_gain / max(
                cost_ratio ** policy.cost_exponent, 1e-6
            )
            verify_utility = (
                value_density
                - policy.cost_lambda * cost_pressure
                - policy.checkpoint_margins[row.checkpoint - 1]
            )
            wait_utility = (
                policy.continuation_weight
                * continuation.value(row.dataset, row.checkpoint)
                * remaining_fraction
            )
            feasible = actual_spent + hard_call_token_cap <= token_budget
            change_allowed = probability_change >= policy.minimum_change_probability
            verify = feasible and change_allowed and verify_utility > wait_utility
            traces.append(
                {
                    "task_key": key,
                    "dataset": row.dataset,
                    "checkpoint": row.checkpoint,
                    "budget_fraction": budget_fraction,
                    "tasks_remaining": tasks_remaining,
                    "remaining_budget": remaining,
                    "allowance_per_task": allowance,
                    "prob_helpful": probability_helpful,
                    "prob_harmful": probability_harmful,
                    "prob_change": probability_change,
                    "expected_gain": expected_gain,
                    "uncertainty": float(
                        prediction.advantage_uncertainty[index]
                    ),
                    "predicted_cost": predicted_cost,
                    "cost_ratio": cost_ratio,
                    "cost_pressure": cost_pressure,
                    "risk_adjusted_gain": risk_adjusted_gain,
                    "value_density": value_density,
                    "continuation_prior": continuation.value(
                        row.dataset, row.checkpoint
                    ),
                    "verify_utility": verify_utility,
                    "wait_utility": wait_utility,
                    "change_allowed": change_allowed,
                    "feasible": feasible,
                    "verify": verify,
                    # Audit-only fields below are never referenced above.
                    "actual_delta": row.delta,
                    "actual_cost": row.teacher_total_tokens,
                }
            )
            if verify:
                chosen = index
                break
        baseline = float(examples[grouped[key][0]].y_skip)
        baselines.append(baseline)
        if chosen is None:
            outcomes.append(baseline)
            checkpoints.append(0)
            continue
        actual_cost = int(examples[chosen].teacher_total_tokens)
        if actual_cost > hard_call_token_cap:
            raise RuntimeError(
                f"Actual cost {actual_cost} exceeds enforced cap {hard_call_token_cap}"
            )
        actual_spent += actual_cost
        predicted_spent += max(float(costs[chosen]), 0.0)
        outcomes.append(float(examples[chosen].y_verify))
        checkpoints.append(examples[chosen].checkpoint)
    if actual_spent > token_budget:
        raise AssertionError("DARC v0.3 hard-budget replay overshot actual tokens")
    selected = np.asarray(checkpoints, dtype=int)
    replay = ReplayResult(
        policy=policy_name,
        protocol="A-single-strict",
        train_mode="combined",
        scope=scope,
        budget_fraction=budget_fraction,
        token_budget=int(token_budget),
        task_keys=tuple(task_keys),
        outcomes=np.asarray(outcomes, dtype=float),
        baseline=np.asarray(baselines, dtype=float),
        checkpoints=selected,
        calls=int(np.sum(selected > 0)),
        actual_tokens=int(actual_spent),
        predicted_tokens=float(predicted_spent),
    )
    return replay, traces, {
        "hard_call_token_cap": int(hard_call_token_cap),
        "actual_budget_overshoot_tokens": max(actual_spent - token_budget, 0),
        "predicted_tokens": float(predicted_spent),
        "actual_tokens": int(actual_spent),
        "reservation_rule": "provider_total_token_cap",
    }


def policy_grid(config: dict[str, Any]) -> Iterable[RiskCalibratedPolicy]:
    for harmful_weight in config["harmful_weight"]:
        for uncertainty_beta in config["uncertainty_beta"]:
            for cost_exponent in config["cost_exponent"]:
                for cost_lambda in config["cost_lambda"]:
                    for minimum_change in config["minimum_change_probability"]:
                        for continuation_weight in config["continuation_weight"]:
                            for margins in config["checkpoint_margins"]:
                                for cost_mode in config.get("cost_mode", ["expected"]):
                                    yield RiskCalibratedPolicy(
                                        harmful_weight=float(harmful_weight),
                                        uncertainty_beta=float(uncertainty_beta),
                                        cost_exponent=float(cost_exponent),
                                        cost_lambda=float(cost_lambda),
                                        minimum_change_probability=float(
                                            minimum_change
                                        ),
                                        continuation_weight=float(
                                            continuation_weight
                                        ),
                                        checkpoint_margins=tuple(
                                            float(value) for value in margins
                                        ),
                                        cost_mode=str(cost_mode),
                                    )
