"""Leakage-safe calibration and risk/cost-aware control for DARC v0.2.

The module deliberately contains no model fitting on the frozen validation or
test partitions.  It consumes out-of-fold predictions produced on training
tasks, learns small calibrators/priors, and implements deployable replay with a
provider-enforced per-call token reservation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from agent_verify.analysis.cbv_v01_experiment import ReplayResult, _stable_order
from agent_verify.analysis.darc_data import DARCExample
from agent_verify.analysis.darc_model import DARCPrediction


def _logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


@dataclass
class BinaryPlatt:
    """One-dimensional Platt scaling with a constant-rate fallback."""

    coefficient: float
    intercept: float
    constant: float | None = None

    @classmethod
    def fit(cls, probability: np.ndarray, target: np.ndarray, *, seed: int) -> BinaryPlatt:
        target = np.asarray(target, dtype=int)
        if len(np.unique(target)) < 2:
            # Jeffreys smoothing avoids exact zero/one probabilities.
            rate = (float(target.sum()) + 0.5) / (len(target) + 1.0)
            return cls(0.0, 0.0, rate)
        model = LogisticRegression(C=1.0, solver="lbfgs", random_state=seed)
        model.fit(_logit(np.asarray(probability))[:, None], target)
        return cls(float(model.coef_[0, 0]), float(model.intercept_[0]), None)

    def predict(self, probability: np.ndarray) -> np.ndarray:
        probability = np.asarray(probability, dtype=float)
        if self.constant is not None:
            return np.full_like(probability, self.constant, dtype=float)
        return _sigmoid(self.coefficient * _logit(probability) + self.intercept)


@dataclass
class CheckpointCalibrator:
    """Global + checkpoint Platt calibration for Helpful and Harmful events."""

    global_helpful: BinaryPlatt
    global_harmful: BinaryPlatt
    checkpoint_helpful: dict[int, BinaryPlatt]
    checkpoint_harmful: dict[int, BinaryPlatt]
    checkpoint_helpful_weight: dict[int, float]
    checkpoint_harmful_weight: dict[int, float]
    shrinkage_events: float

    @classmethod
    def fit(
        cls,
        examples: list[DARCExample],
        prediction: DARCPrediction,
        *,
        seed: int,
        shrinkage_events: float = 12.0,
    ) -> CheckpointCalibrator:
        labels = np.asarray([row.value_label for row in examples], dtype=int)
        checkpoints = np.asarray([row.checkpoint for row in examples], dtype=int)
        helpful = (labels == 2).astype(int)
        harmful = (labels == 0).astype(int)
        global_helpful = BinaryPlatt.fit(prediction.probabilities[:, 2], helpful, seed=seed)
        global_harmful = BinaryPlatt.fit(prediction.probabilities[:, 0], harmful, seed=seed + 1)
        cp_helpful: dict[int, BinaryPlatt] = {}
        cp_harmful: dict[int, BinaryPlatt] = {}
        help_weight: dict[int, float] = {}
        harm_weight: dict[int, float] = {}
        for checkpoint in (1, 2, 3):
            selected = checkpoints == checkpoint
            cp_helpful[checkpoint] = BinaryPlatt.fit(
                prediction.probabilities[selected, 2], helpful[selected], seed=seed + checkpoint * 7
            )
            cp_harmful[checkpoint] = BinaryPlatt.fit(
                prediction.probabilities[selected, 0], harmful[selected], seed=seed + checkpoint * 11
            )
            helpful_events = int(helpful[selected].sum())
            harmful_events = int(harmful[selected].sum())
            help_weight[checkpoint] = helpful_events / (helpful_events + shrinkage_events)
            harm_weight[checkpoint] = harmful_events / (harmful_events + shrinkage_events)
        return cls(
            global_helpful,
            global_harmful,
            cp_helpful,
            cp_harmful,
            help_weight,
            harm_weight,
            float(shrinkage_events),
        )

    def transform(
        self, examples: list[DARCExample], prediction: DARCPrediction
    ) -> DARCPrediction:
        checkpoints = np.asarray([row.checkpoint for row in examples], dtype=int)
        raw_helpful = prediction.probabilities[:, 2]
        raw_harmful = prediction.probabilities[:, 0]
        global_helpful = self.global_helpful.predict(raw_helpful)
        global_harmful = self.global_harmful.predict(raw_harmful)
        helpful = np.zeros(len(examples), dtype=float)
        harmful = np.zeros(len(examples), dtype=float)
        for checkpoint in (1, 2, 3):
            selected = checkpoints == checkpoint
            local_helpful = self.checkpoint_helpful[checkpoint].predict(raw_helpful[selected])
            local_harmful = self.checkpoint_harmful[checkpoint].predict(raw_harmful[selected])
            helpful[selected] = (
                self.checkpoint_helpful_weight[checkpoint] * local_helpful
                + (1.0 - self.checkpoint_helpful_weight[checkpoint]) * global_helpful[selected]
            )
            harmful[selected] = (
                self.checkpoint_harmful_weight[checkpoint] * local_harmful
                + (1.0 - self.checkpoint_harmful_weight[checkpoint]) * global_harmful[selected]
            )
        total = helpful + harmful
        overflow = total > 0.98
        helpful[overflow] *= 0.98 / total[overflow]
        harmful[overflow] *= 0.98 / total[overflow]
        no_change = np.maximum(1.0 - helpful - harmful, 0.0)
        probabilities = np.column_stack((harmful, no_change, helpful))
        return DARCPrediction(
            probabilities=probabilities,
            advantage=helpful - harmful,
            advantage_uncertainty=prediction.advantage_uncertainty.copy(),
            expected_total_tokens=prediction.expected_total_tokens.copy(),
            upper_total_tokens=prediction.upper_total_tokens.copy(),
            latency_seconds=prediction.latency_seconds,
        )


@dataclass
class ContinuationPrior:
    """Shrunk empirical value of waiting, without fitted-Q bootstrapping."""

    global_by_checkpoint: dict[int, float]
    dataset_by_checkpoint: dict[str, dict[int, float]]
    dataset_weights: dict[str, dict[int, float]]
    shrinkage_tasks: float

    @classmethod
    def fit(
        cls, examples: list[DARCExample], *, shrinkage_tasks: float = 20.0
    ) -> ContinuationPrior:
        grouped: dict[str, list[DARCExample]] = {}
        for row in examples:
            grouped.setdefault(row.group_key, []).append(row)
        targets: list[tuple[str, int, float]] = []
        for rows in grouped.values():
            rows.sort(key=lambda row: row.checkpoint)
            for row in rows:
                future = [candidate.delta for candidate in rows if candidate.checkpoint > row.checkpoint]
                target = max([0, *future]) if future else 0
                targets.append((row.dataset, row.checkpoint, float(target)))
        global_values: dict[int, float] = {}
        dataset_values: dict[str, dict[int, float]] = {}
        weights: dict[str, dict[int, float]] = {}
        for checkpoint in (1, 2, 3):
            values = [value for _dataset, cp, value in targets if cp == checkpoint]
            global_values[checkpoint] = float(np.mean(values)) if values else 0.0
        for dataset in sorted({dataset for dataset, _cp, _value in targets}):
            dataset_values[dataset] = {}
            weights[dataset] = {}
            for checkpoint in (1, 2, 3):
                values = [
                    value for name, cp, value in targets
                    if name == dataset and cp == checkpoint
                ]
                local = float(np.mean(values)) if values else global_values[checkpoint]
                weight = len(values) / (len(values) + shrinkage_tasks)
                dataset_values[dataset][checkpoint] = (
                    weight * local + (1.0 - weight) * global_values[checkpoint]
                )
                weights[dataset][checkpoint] = float(weight)
        return cls(global_values, dataset_values, weights, float(shrinkage_tasks))

    def value(self, dataset: str, checkpoint: int) -> float:
        return float(
            self.dataset_by_checkpoint.get(dataset, {}).get(
                checkpoint, self.global_by_checkpoint.get(checkpoint, 0.0)
            )
        )


@dataclass(frozen=True)
class RiskCostPolicy:
    harmful_weight: float = 2.0
    uncertainty_beta: float = 0.5
    cost_lambda: float = 0.02
    continuation_weight: float = 0.5
    checkpoint_margins: tuple[float, float, float] = (0.02, 0.0, 0.0)
    cost_mode: str = "expected"

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["checkpoint_margins"] = list(self.checkpoint_margins)
        return row


def grouped_folds(
    examples: list[DARCExample], *, folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic task-level stratification by dataset and event profile."""

    if folds < 2:
        raise ValueError("folds must be at least two")
    grouped: dict[str, list[DARCExample]] = {}
    for row in examples:
        grouped.setdefault(row.group_key, []).append(row)
    buckets: dict[tuple[str, bool, bool], list[str]] = {}
    for key, rows in grouped.items():
        profile = (rows[0].dataset, any(row.delta > 0 for row in rows), any(row.delta < 0 for row in rows))
        buckets.setdefault(profile, []).append(key)
    fold_groups: list[set[str]] = [set() for _ in range(folds)]
    for profile, keys in sorted(buckets.items()):
        ordered = sorted(
            keys,
            key=lambda key: hashlib.sha256(f"{seed}:{profile}:{key}".encode()).hexdigest(),
        )
        for offset, key in enumerate(ordered):
            fold_groups[offset % folds].add(key)
    all_indices = np.arange(len(examples))
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for held_out in fold_groups:
        valid = np.asarray([index for index, row in enumerate(examples) if row.group_key in held_out])
        train = np.asarray([index for index in all_indices if index not in set(valid.tolist())])
        output.append((train, valid))
    return output


def stratified_task_partition(
    examples: list[DARCExample], *, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split task groups while retaining rare Helpful/Harmful profiles."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    folds = grouped_folds(examples, folds=2, seed=seed)
    left, right = folds[0][1], folds[1][1]
    if fraction == 0.5:
        return left, right
    # The v0.2 protocol intentionally uses equal calibration/policy halves.
    raise ValueError("Only a 0.5 task partition is supported to preserve the frozen protocol")


def _group_indices(examples: list[DARCExample]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(examples):
        grouped.setdefault(row.group_key, []).append(index)
    for indices in grouped.values():
        indices.sort(key=lambda index: examples[index].checkpoint)
    return grouped


def replay_risk_cost_policy(
    examples: list[DARCExample],
    prediction: DARCPrediction,
    continuation: ContinuationPrior,
    policy: RiskCostPolicy,
    *,
    policy_name: str,
    token_budget: int,
    budget_fraction: float,
    hard_call_token_cap: int,
    reference_cost: float,
    seed: int,
    scope: str = "combined",
) -> tuple[ReplayResult, list[dict[str, Any]], dict[str, Any]]:
    """Sequential one-call replay with a deployable, non-overshooting cap."""

    if policy.cost_mode not in {"expected", "upper"}:
        raise ValueError(f"Unknown cost mode: {policy.cost_mode}")
    if hard_call_token_cap <= 0:
        raise ValueError("hard_call_token_cap must be positive")
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
        allowance = remaining / max(tasks_remaining, 1)
        remaining_fraction = remaining / max(token_budget, 1)
        chosen: int | None = None
        for index in grouped[key]:
            row = examples[index]
            predicted_cost = max(float(costs[index]), 1.0)
            cost_pressure = predicted_cost / max(allowance, reference_cost, 1.0)
            verify_utility = (
                float(prediction.probabilities[index, 2])
                - policy.harmful_weight * float(prediction.probabilities[index, 0])
                - policy.uncertainty_beta * float(prediction.advantage_uncertainty[index])
                - policy.cost_lambda * cost_pressure
                - policy.checkpoint_margins[row.checkpoint - 1]
            )
            wait_utility = (
                policy.continuation_weight
                * continuation.value(row.dataset, row.checkpoint)
                * remaining_fraction
            )
            feasible = actual_spent + hard_call_token_cap <= token_budget
            verify = feasible and verify_utility > wait_utility
            traces.append(
                {
                    "task_key": key,
                    "dataset": row.dataset,
                    "checkpoint": row.checkpoint,
                    "budget_fraction": budget_fraction,
                    "tasks_remaining": tasks_remaining,
                    "remaining_budget": remaining,
                    "allowance_per_task": allowance,
                    "prob_helpful": float(prediction.probabilities[index, 2]),
                    "prob_harmful": float(prediction.probabilities[index, 0]),
                    "prob_no_change": float(prediction.probabilities[index, 1]),
                    "uncertainty": float(prediction.advantage_uncertainty[index]),
                    "predicted_cost": predicted_cost,
                    "cost_pressure": cost_pressure,
                    "continuation_prior": continuation.value(row.dataset, row.checkpoint),
                    "verify_utility": verify_utility,
                    "wait_utility": wait_utility,
                    "feasible": feasible,
                    "verify": verify,
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
        raise AssertionError("DARC v0.2 hard-budget replay overshot actual tokens")
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
    audit = {
        "hard_call_token_cap": int(hard_call_token_cap),
        "actual_budget_overshoot_tokens": max(actual_spent - token_budget, 0),
        "predicted_tokens": predicted_spent,
        "actual_tokens": actual_spent,
        "reservation_rule": "provider_total_token_cap",
    }
    return replay, traces, audit


def failure_metrics(
    examples: list[DARCExample], replay: ReplayResult
) -> dict[str, int | float]:
    grouped = _group_indices(examples)
    lookup = {key: grouped[key] for key in replay.task_keys}
    helpful = harmful = wasted = missed = wrong = 0
    cp = {1: 0, 2: 0, 3: 0}
    for offset, key in enumerate(replay.task_keys):
        selected_cp = int(replay.checkpoints[offset])
        rows = [examples[index] for index in lookup[key]]
        positive = [row for row in rows if row.delta > 0]
        if selected_cp == 0:
            missed += int(bool(positive))
            continue
        cp[selected_cp] += 1
        selected = next(row for row in rows if row.checkpoint == selected_cp)
        helpful += int(selected.delta > 0)
        harmful += int(selected.delta < 0)
        wasted += int(selected.delta == 0)
        if positive and selected.delta < max(row.delta for row in positive):
            wrong += 1
    return {
        "teacher_calls": replay.calls,
        "helpful_selected": helpful,
        "harmful_selected": harmful,
        "wasted_no_change": wasted,
        "missed_helpful": missed,
        "incorrect_checkpoint": wrong,
        "cp1_calls": cp[1],
        "cp2_calls": cp[2],
        "cp3_calls": cp[3],
    }


def policy_grid(config: dict[str, Any]) -> Iterable[RiskCostPolicy]:
    for harmful_weight in config["harmful_weight"]:
        for uncertainty_beta in config["uncertainty_beta"]:
            for cost_lambda in config["cost_lambda"]:
                for continuation_weight in config["continuation_weight"]:
                    for margins in config["checkpoint_margins"]:
                        for cost_mode in config.get("cost_mode", ["expected"]):
                            yield RiskCostPolicy(
                                float(harmful_weight),
                                float(uncertainty_beta),
                                float(cost_lambda),
                                float(continuation_weight),
                                tuple(float(value) for value in margins),
                                str(cost_mode),
                            )
