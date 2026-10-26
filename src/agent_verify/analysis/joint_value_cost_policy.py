"""Sequential, single-call, token-budgeted offline replay policies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from agent_verify.analysis.joint_value_cost_data import JointExample
from agent_verify.analysis.joint_value_cost_models import ValueCostPrediction


@dataclass(frozen=True)
class PolicyReplay:
    policy: str
    admission_mode: str
    token_budget: int
    task_ids: tuple[str, ...]
    outcomes: np.ndarray
    selected_indices: np.ndarray
    selected_checkpoints: np.ndarray
    actual_input_tokens: int
    actual_output_tokens: int
    actual_total_tokens: int
    predicted_total_tokens: float


def grouped_indices(examples: list[JointExample]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(example.group_key, []).append(index)
    for indices in grouped.values():
        indices.sort(key=lambda index: examples[index].checkpoint)
    return grouped


def _stable_task_order(task_ids: list[str], seed: int) -> list[str]:
    return sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(),
    )


def learned_candidates(
    examples: list[JointExample],
    prediction: ValueCostPrediction,
    *,
    gain_per_1k_threshold: float,
) -> dict[str, int]:
    """Choose the first eligible checkpoint without consulting later checkpoints."""

    if len(examples) != len(prediction.expected_gain):
        raise ValueError("Prediction/example length mismatch")
    candidates: dict[str, int] = {}
    for task_id, indices in grouped_indices(examples).items():
        for index in indices:
            expected_gain = float(prediction.expected_gain[index])
            expected_cost = max(float(prediction.expected_total_tokens[index]), 1.0)
            gain_per_1k = 1_000.0 * expected_gain / expected_cost
            if expected_gain > 0.0 and gain_per_1k >= gain_per_1k_threshold:
                candidates[task_id] = index
                break
    return candidates


def baseline_candidates(
    examples: list[JointExample],
    policy: str,
    *,
    seed: int,
) -> dict[str, int]:
    grouped = grouped_indices(examples)
    candidates: dict[str, int] = {}
    for task_id, indices in grouped.items():
        if policy == "Never Verify":
            continue
        if policy.startswith("Always CP"):
            checkpoint = int(policy[-1])
            candidates[task_id] = next(
                index for index in indices if examples[index].checkpoint == checkpoint
            )
        elif policy == "Random Verification":
            digest = hashlib.sha256(f"{seed}:{task_id}:random-cp".encode()).digest()
            candidates[task_id] = indices[int.from_bytes(digest[:4], "big") % len(indices)]
        elif policy == "Highest Local Risk":
            # Diagnostic offline-lookahead baseline: it reads all three local-risk scores.
            candidates[task_id] = max(
                indices,
                key=lambda index: (examples[index].local_risk, -examples[index].checkpoint),
            )
        elif policy == "Offline Oracle":
            helpful = [index for index in indices if examples[index].delta > 0]
            if helpful:
                candidates[task_id] = helpful[0]
        else:
            raise ValueError(f"Unknown baseline policy: {policy}")
    return candidates


def replay_candidates(
    examples: list[JointExample],
    candidates: dict[str, int],
    *,
    policy: str,
    token_budget: int,
    seed: int,
    admission_mode: str,
    predicted_costs: np.ndarray | None = None,
) -> PolicyReplay:
    """Replay selected interventions under expected-cost or retrospective hard admission.

    expected_cost uses deployable predicted cost for admission and may overshoot the
    actual budget. retrospective_hard_cap uses observed cost to reject an intervention
    that would cross the cap; it is an accounting upper bound, not deployable foresight.
    """

    if admission_mode not in {"expected_cost", "retrospective_hard_cap"}:
        raise ValueError(f"Unknown admission mode: {admission_mode}")
    grouped = grouped_indices(examples)
    task_ids = _stable_task_order(list(grouped), seed)
    actual_remaining = int(token_budget)
    predicted_remaining = float(token_budget)
    outcomes: list[float] = []
    selected: list[int] = []
    checkpoints: list[int] = []
    actual_input = 0
    actual_output = 0
    predicted_total = 0.0
    for task_id in task_ids:
        indices = grouped[task_id]
        baseline_values = {examples[index].y_skip for index in indices}
        if len(baseline_values) != 1:
            raise ValueError(f"Y_skip differs within task {task_id}")
        candidate = candidates.get(task_id)
        admitted = False
        if candidate is not None:
            actual_cost = examples[candidate].teacher_total_tokens
            predicted_cost = (
                float(predicted_costs[candidate]) if predicted_costs is not None else float(actual_cost)
            )
            if admission_mode == "expected_cost":
                admitted = predicted_cost <= predicted_remaining
            else:
                admitted = actual_cost <= actual_remaining
            if admitted:
                predicted_remaining -= predicted_cost
                actual_remaining -= actual_cost
                predicted_total += predicted_cost
                actual_input += examples[candidate].teacher_input_tokens
                actual_output += examples[candidate].teacher_output_tokens
        if candidate is not None and admitted:
            outcomes.append(float(examples[candidate].y_verify))
            selected.append(candidate)
            checkpoints.append(examples[candidate].checkpoint)
        else:
            outcomes.append(float(examples[indices[0]].y_skip))
            selected.append(-1)
            checkpoints.append(0)
    return PolicyReplay(
        policy=policy,
        admission_mode=admission_mode,
        token_budget=int(token_budget),
        task_ids=tuple(task_ids),
        outcomes=np.asarray(outcomes, dtype=float),
        selected_indices=np.asarray(selected, dtype=int),
        selected_checkpoints=np.asarray(checkpoints, dtype=int),
        actual_input_tokens=actual_input,
        actual_output_tokens=actual_output,
        actual_total_tokens=actual_input + actual_output,
        predicted_total_tokens=predicted_total,
    )


def token_budgets(
    training_examples: list[JointExample],
    evaluation_task_count: int,
    quantiles: list[float],
) -> list[int]:
    costs = np.asarray([row.teacher_total_tokens for row in training_examples], dtype=float)
    budgets = {0}
    for quantile in quantiles:
        per_task = float(np.quantile(costs, quantile))
        budgets.add(int(round(per_task * evaluation_task_count)))
    return sorted(budgets)


def tune_gain_per_cost_threshold(
    examples: list[JointExample],
    prediction: ValueCostPrediction,
    *,
    thresholds: list[float],
    budget_quantiles: list[float],
    seed: int,
) -> float:
    """Tune a deployable threshold using only grouped out-of-fold train predictions."""

    budgets = token_budgets(examples, len(grouped_indices(examples)), budget_quantiles)
    best_key: tuple[float, float, float] | None = None
    selected_threshold = thresholds[0]
    for threshold in thresholds:
        candidates = learned_candidates(
            examples,
            prediction,
            gain_per_1k_threshold=float(threshold),
        )
        accuracies = []
        token_use = []
        for budget in budgets:
            replay = replay_candidates(
                examples,
                candidates,
                policy="validation",
                token_budget=budget,
                seed=seed,
                admission_mode="expected_cost",
                predicted_costs=prediction.expected_total_tokens,
            )
            accuracies.append(float(replay.outcomes.mean()))
            token_use.append(float(replay.actual_total_tokens))
        key = (float(np.mean(accuracies)), -float(np.mean(token_use)), float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            selected_threshold = float(threshold)
    return selected_threshold


def replay_summary(
    examples: list[JointExample],
    replay: PolicyReplay,
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, float | int | str]:
    grouped = grouped_indices(examples)
    never = np.asarray(
        [examples[grouped[task_id][0]].y_skip for task_id in replay.task_ids], dtype=float
    )
    gain = replay.outcomes - never
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(replay.task_ids), size=(bootstrap_repetitions, len(replay.task_ids)))
    accuracy_bootstrap = replay.outcomes[samples].mean(axis=1)
    gain_bootstrap = gain[samples].mean(axis=1) * 100.0
    accuracy_low, accuracy_high = np.percentile(accuracy_bootstrap, [2.5, 97.5])
    gain_low, gain_high = np.percentile(gain_bootstrap, [2.5, 97.5])
    chosen = [int(index) for index in replay.selected_indices if index >= 0]
    return {
        "policy": replay.policy,
        "admission_mode": replay.admission_mode,
        "token_budget": replay.token_budget,
        "accuracy": float(replay.outcomes.mean()),
        "accuracy_ci95_low": float(accuracy_low),
        "accuracy_ci95_high": float(accuracy_high),
        "gain_vs_never_pp": float(gain.mean() * 100.0),
        "gain_ci95_low_pp": float(gain_low),
        "gain_ci95_high_pp": float(gain_high),
        "teacher_calls": len(chosen),
        "teacher_input_tokens": replay.actual_input_tokens,
        "teacher_output_tokens": replay.actual_output_tokens,
        "teacher_total_tokens": replay.actual_total_tokens,
        "predicted_total_tokens": replay.predicted_total_tokens,
        "budget_overshoot_tokens": max(0, replay.actual_total_tokens - replay.token_budget),
        "helpful_selected": sum(examples[index].delta > 0 for index in chosen),
        "harmful_selected": sum(examples[index].delta < 0 for index in chosen),
        "no_change_selected": sum(examples[index].delta == 0 for index in chosen),
        "cp1_calls": sum(examples[index].checkpoint == 1 for index in chosen),
        "cp2_calls": sum(examples[index].checkpoint == 2 for index in chosen),
        "cp3_calls": sum(examples[index].checkpoint == 3 for index in chosen),
    }


def legacy_candidates_from_rows(
    examples: list[JointExample],
    rows: list[dict[str, str]],
    *,
    model_name: str,
) -> dict[str, int]:
    """Map frozen legacy controller selections onto the current example indices."""

    lookup = {
        (row.task_id, row.checkpoint): index for index, row in enumerate(examples)
    }
    candidates: dict[str, int] = {}
    for row in rows:
        if row.get("model") != model_name or int(row.get("selected_by_controller", 0)) != 1:
            continue
        key = (str(row["task_id"]), int(row["checkpoint"]))
        if key in lookup:
            index = lookup[key]
            candidates[examples[index].group_key] = index
    return candidates
