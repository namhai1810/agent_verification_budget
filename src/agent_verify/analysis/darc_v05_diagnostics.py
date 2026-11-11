"""Read-only diagnostics for DARC v0.5 seed instability and policy improvement.

The analysis reuses saved validation checkpoints and CBV Level-2 branches.  It
never trains a model, loads the frozen test partition, or invokes Student/Teacher
inference.  Recorded outcomes and costs are used only after a replay action and
for offline target/error auditing.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm

from agent_verify.analysis.cbv_v01_experiment import _passed, _stable_order
from agent_verify.analysis.darc_data import encode_or_load
from agent_verify.analysis.darc_v04_data import MultiHistoryState, build_states
from agent_verify.analysis.darc_v04_experiment import (
    _assert_disjoint,
    _load_groups,
    _precompute,
    _write_csv,
)
from agent_verify.analysis.darc_v04_model import DARCv04Ensemble
from agent_verify.analysis.darc_v05_env import CBVLevel2OfflineEnv, OfflineObservation
from agent_verify.analysis.darc_v05_experiment import (
    _selected_v04_policy,
    _v04_continuation_factory,
    _v05_continuation_factory,
)
from agent_verify.analysis.darc_v05_model import DARCv05Ensemble, V05Prediction
from agent_verify.analysis.darc_v05_training import (
    ConservativePolicy,
    V05TargetTable,
    generate_policy_targets,
    nearest_prediction,
    validate_target_coverage,
)

SEQUENCES = ("SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV")
MULTI_SEQUENCES = ("VVS", "VSV", "SVV", "VVV")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def _selected_policy(
    artifact_root: Path,
    *,
    seed: int,
    variant: str,
    iteration: int,
) -> ConservativePolicy:
    label = f"DARC v0.5 · {variant} · i{iteration}"
    rows = [
        row
        for row in _read_csv(artifact_root / "tables" / "policy_grid.csv")
        if int(row["seed"]) == seed
        and row["variant"] == label
        and int(row["iteration"]) == iteration
        and _truthy(row["selected"])
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one selected policy for {variant}/seed={seed}/i{iteration}; found {len(rows)}"
        )
    fields = ConservativePolicy.__dataclass_fields__
    values: dict[str, Any] = {}
    boolean_fields = {"conservative", "dynamic_token_price", "uncertainty_penalty"}
    for field_name in fields:
        raw = rows[0][field_name]
        values[field_name] = _truthy(raw) if field_name in boolean_fields else float(raw)
    return ConservativePolicy(**values)


def _prediction_components(
    observation: OfflineObservation,
    prediction: V05Prediction,
    policy: ConservativePolicy,
    *,
    reference_budget: float,
    hard_call_token_cap: int,
) -> dict[str, float | int | bool]:
    index = observation.state_index
    q_skip = float(prediction.q_success[index, 0])
    q_verify = float(prediction.q_success[index, 1])
    advantage = q_verify - q_skip
    advantage_uncertainty = float(prediction.advantage_uncertainty[index])
    additional_cost = float(prediction.additional_cost[index])
    additional_cost_uncertainty = float(prediction.additional_cost_uncertainty[index])
    immediate_cost = float(prediction.immediate_cost[index])
    immediate_cost_uncertainty = float(prediction.immediate_cost_uncertainty[index])
    harmful_probability = float(prediction.effect_probabilities[index, 0])
    helpful_probability = float(prediction.effect_probabilities[index, 2])
    uncertainty_penalty = (
        policy.advantage_uncertainty_beta * advantage_uncertainty if policy.uncertainty_penalty else 0.0
    )
    harmful_penalty = policy.harmful_weight * harmful_probability if policy.conservative else 0.0
    upper_cost = additional_cost
    if policy.uncertainty_penalty:
        upper_cost += policy.cost_uncertainty_beta * additional_cost_uncertainty
    budget_pressure = 1.0
    if policy.dynamic_token_price:
        budget_pressure = (
            reference_budget / max(observation.per_task_allowance, 1.0)
        ) ** policy.price_elasticity
    dynamic_token_price = policy.base_token_price * budget_pressure
    cost_penalty = dynamic_token_price * upper_cost / max(reference_budget, 1.0)
    score = advantage - uncertainty_penalty - harmful_penalty - cost_penalty
    predicted_call_upper = immediate_cost + immediate_cost_uncertainty
    feasible = (
        observation.global_remaining_budget >= hard_call_token_cap
        and predicted_call_upper <= observation.global_remaining_budget
    )
    return {
        "q_skip": q_skip,
        "q_verify": q_verify,
        "advantage": advantage,
        "advantage_uncertainty": advantage_uncertainty,
        "additional_cost": additional_cost,
        "additional_cost_uncertainty": additional_cost_uncertainty,
        "immediate_cost": immediate_cost,
        "immediate_cost_uncertainty": immediate_cost_uncertainty,
        "harmful_probability": harmful_probability,
        "helpful_probability": helpful_probability,
        "uncertainty_penalty": uncertainty_penalty,
        "harmful_penalty": harmful_penalty,
        "budget_pressure": budget_pressure,
        "dynamic_token_price": dynamic_token_price,
        "cost_penalty": cost_penalty,
        "decision_score": score,
        "decision_margin": policy.decision_margin,
        "score_minus_margin": score - policy.decision_margin,
        "predicted_call_upper": predicted_call_upper,
        "feasible": feasible,
    }


def _conditional_verify_gain(
    group: Any,
    sequence: str,
    checkpoint: int,
) -> int:
    verify_sequence = sequence[: checkpoint - 1] + "V" + sequence[checkpoint:]
    skip_sequence = sequence[: checkpoint - 1] + "S" + sequence[checkpoint:]
    return int(_passed(group.rows[verify_sequence]) - _passed(group.rows[skip_sequence]))


def detailed_replay(
    groups: dict[str, Any],
    state_lookup: dict[tuple[str, str], int],
    predictions: list[V05Prediction],
    allowances: list[float],
    policy: ConservativePolicy,
    *,
    reference_budget: float,
    token_budget: int,
    hard_call_token_cap: int,
    model_seed: int,
    order_seed: int,
    variant: str,
    iteration: int,
    allowance_mode: str,
    analysis_family: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay one saved policy while logging every observable decision."""

    if allowance_mode not in {"dynamic", "fixed"}:
        raise ValueError(f"Unknown allowance mode {allowance_mode!r}")
    keys = _stable_order(groups, order_seed)
    environment = CBVLevel2OfflineEnv(groups, state_lookup, hard_call_token_cap=hard_call_token_cap)
    actual_spent = 0
    task_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for order, task_key in enumerate(keys):
        global_remaining = max(token_budget - actual_spent, 0)
        task_allowance = (
            global_remaining / max(len(keys) - order, 1) if allowance_mode == "dynamic" else reference_budget
        )
        observation = environment.reset(
            task_key,
            global_remaining,
            per_task_allowance=task_allowance,
        )
        local_rows: list[dict[str, Any]] = []
        task_tokens = 0
        while True:
            global_remaining = max(token_budget - actual_spent, 0)
            task_allowance = (
                global_remaining / max(len(keys) - order, 1)
                if allowance_mode == "dynamic"
                else reference_budget
            )
            environment.set_per_task_allowance(task_allowance)
            observation = environment.observation()
            prediction = nearest_prediction(predictions, allowances, observation.per_task_allowance)
            components = _prediction_components(
                observation,
                prediction,
                policy,
                reference_budget=reference_budget,
                hard_call_token_cap=hard_call_token_cap,
            )
            action = (
                "V"
                if bool(components["feasible"])
                and float(components["decision_score"]) > policy.decision_margin
                else "S"
            )
            prefix = observation.prefix
            direct_skip = prefix + "S" + "S" * (2 - len(prefix))
            direct_verify = prefix + "V" + "S" * (2 - len(prefix))
            direct_delta = int(
                _passed(groups[task_key].rows[direct_verify]) - _passed(groups[task_key].rows[direct_skip])
            )
            transition = environment.step(action)
            actual_spent += transition.incremental_teacher_tokens
            task_tokens += transition.incremental_teacher_tokens
            if actual_spent > token_budget:
                raise AssertionError("Diagnostic replay exceeded strict global budget")
            local_rows.append(
                {
                    "analysis_family": analysis_family,
                    "variant": variant,
                    "iteration": iteration,
                    "model_seed": model_seed,
                    "order_seed": order_seed,
                    "allowance_mode": allowance_mode,
                    "task_key": task_key,
                    "dataset": groups[task_key].task.dataset,
                    "task_order": order,
                    "checkpoint": observation.checkpoint,
                    "prefix_before": prefix or "START",
                    "action": action,
                    "global_remaining_before": observation.global_remaining_budget,
                    "per_task_allowance": observation.per_task_allowance,
                    "allowance_bucket": allowances[
                        int(np.argmin(np.abs(np.asarray(allowances) - observation.per_task_allowance)))
                    ],
                    **components,
                    "actual_incremental_tokens": transition.incremental_teacher_tokens,
                    "direct_verify_gain_all_future_skip_audit": direct_delta,
                }
            )
            if transition.done:
                break
            assert transition.next_observation is not None
            observation = transition.next_observation
        sequence = environment.prefix
        correctness = int(_passed(groups[task_key].rows[sequence]))
        baseline = int(_passed(groups[task_key].rows["SSS"]))
        calls = sequence.count("V")
        helpful = harmful = no_change = 0
        for row in local_rows:
            conditional_gain = _conditional_verify_gain(groups[task_key], sequence, int(row["checkpoint"]))
            row.update(
                {
                    "final_sequence": sequence,
                    "final_correctness": correctness,
                    "baseline_correctness": baseline,
                    "task_teacher_tokens": task_tokens,
                    "task_teacher_calls": calls,
                    "conditional_verify_gain_same_suffix_audit": conditional_gain,
                    "helpful_selected": int(row["action"] == "V" and conditional_gain > 0),
                    "harmful_selected": int(row["action"] == "V" and conditional_gain < 0),
                    "no_change_selected": int(row["action"] == "V" and conditional_gain == 0),
                }
            )
            helpful += int(row["helpful_selected"])
            harmful += int(row["harmful_selected"])
            no_change += int(row["no_change_selected"])
        decision_rows.extend(local_rows)
        task_rows.append(
            {
                "analysis_family": analysis_family,
                "variant": variant,
                "iteration": iteration,
                "model_seed": model_seed,
                "order_seed": order_seed,
                "allowance_mode": allowance_mode,
                "task_key": task_key,
                "dataset": groups[task_key].task.dataset,
                "task_order": order,
                "sequence": sequence,
                "final_correctness": correctness,
                "baseline_correctness": baseline,
                "teacher_tokens": task_tokens,
                "teacher_calls": calls,
                "helpful_calls": helpful,
                "harmful_calls": harmful,
                "no_change_calls": no_change,
            }
        )
    summary = {
        "analysis_family": analysis_family,
        "variant": variant,
        "iteration": iteration,
        "model_seed": model_seed,
        "order_seed": order_seed,
        "allowance_mode": allowance_mode,
        "tasks": len(task_rows),
        "accuracy": float(np.mean([row["final_correctness"] for row in task_rows])),
        "teacher_tokens": actual_spent,
        "teacher_calls": sum(int(row["teacher_calls"]) for row in task_rows),
        "helpful_calls": sum(int(row["helpful_calls"]) for row in task_rows),
        "harmful_calls": sum(int(row["harmful_calls"]) for row in task_rows),
        "no_change_calls": sum(int(row["no_change_calls"]) for row in task_rows),
        "token_budget": token_budget,
        "budget_utilization": actual_spent / max(token_budget, 1),
        "max_budget_overshoot": max(actual_spent - token_budget, 0),
    }
    for sequence in SEQUENCES:
        summary[f"sequence_{sequence}"] = sum(int(row["sequence"] == sequence) for row in task_rows)
    return summary, task_rows, decision_rows


def _seed_task_comparison(task_rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in task_rows
        if row["analysis_family"] == "cross_seed"
        and row["allowance_mode"] == "dynamic"
        and int(row["model_seed"]) == int(row["order_seed"])
    ]
    by_task: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        by_task[str(row["task_key"])][int(row["model_seed"])] = row
    output: list[dict[str, Any]] = []
    for task_key, seed_rows in sorted(by_task.items()):
        if set(seed_rows) != set(seeds):
            raise AssertionError(f"Missing natural replay seed for {task_key}")
        row: dict[str, Any] = {
            "task_key": task_key,
            "dataset": seed_rows[seeds[0]]["dataset"],
            "baseline_correctness": seed_rows[seeds[0]]["baseline_correctness"],
        }
        correctness: list[int] = []
        sequences: list[str] = []
        tokens: list[int] = []
        for seed in seeds:
            item = seed_rows[seed]
            row[f"correct_seed_{seed}"] = item["final_correctness"]
            row[f"sequence_seed_{seed}"] = item["sequence"]
            row[f"tokens_seed_{seed}"] = item["teacher_tokens"]
            row[f"calls_seed_{seed}"] = item["teacher_calls"]
            correctness.append(int(item["final_correctness"]))
            sequences.append(str(item["sequence"]))
            tokens.append(int(item["teacher_tokens"]))
        row["all_seeds_same_correctness"] = int(len(set(correctness)) == 1)
        row["all_seeds_same_sequence"] = int(len(set(sequences)) == 1)
        row["token_range"] = max(tokens) - min(tokens)
        output.append(row)
    return output


def _iteration_comparison(
    task_rows: list[dict[str, Any]], variants: list[str], seeds: list[int]
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in task_rows
        if row["analysis_family"] == "policy_improvement"
        and row["allowance_mode"] == "dynamic"
        and int(row["model_seed"]) == int(row["order_seed"])
    ]
    keyed = {
        (
            str(row["variant"]),
            int(row["model_seed"]),
            int(row["iteration"]),
            str(row["task_key"]),
        ): row
        for row in selected
    }
    output: list[dict[str, Any]] = []
    task_keys = sorted({str(row["task_key"]) for row in selected})
    for variant in variants:
        for seed in seeds:
            for task_key in task_keys:
                before = keyed[(variant, seed, 0, task_key)]
                after = keyed[(variant, seed, 1, task_key)]
                output.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "task_key": task_key,
                        "dataset": before["dataset"],
                        "sequence_i0": before["sequence"],
                        "sequence_i1": after["sequence"],
                        "action_sequence_changed": int(before["sequence"] != after["sequence"]),
                        "correct_i0": before["final_correctness"],
                        "correct_i1": after["final_correctness"],
                        "correctness_delta": int(after["final_correctness"])
                        - int(before["final_correctness"]),
                        "tokens_i0": before["teacher_tokens"],
                        "tokens_i1": after["teacher_tokens"],
                        "token_delta": int(after["teacher_tokens"]) - int(before["teacher_tokens"]),
                        "calls_i0": before["teacher_calls"],
                        "calls_i1": after["teacher_calls"],
                        "call_delta": int(after["teacher_calls"]) - int(before["teacher_calls"]),
                        "helpful_i0": before["helpful_calls"],
                        "helpful_i1": after["helpful_calls"],
                        "harmful_i0": before["harmful_calls"],
                        "harmful_i1": after["harmful_calls"],
                        "no_change_i0": before["no_change_calls"],
                        "no_change_i1": after["no_change_calls"],
                    }
                )
    return output


def _prediction_rows(
    states: list[MultiHistoryState],
    predictions: list[V05Prediction],
    allowances: list[float],
    *,
    variant: str,
    seed: int,
    iteration: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for allowance, prediction in zip(allowances, predictions, strict=True):
        for index, state in enumerate(states):
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "iteration": iteration,
                    "allowance": allowance,
                    "state_index": index,
                    "task_key": state.task_key,
                    "dataset": state.dataset,
                    "checkpoint": state.checkpoint,
                    "prefix": state.prefix or "START",
                    "q_skip": float(prediction.q_success[index, 0]),
                    "q_verify": float(prediction.q_success[index, 1]),
                    "advantage": float(prediction.advantage[index]),
                    "advantage_uncertainty": float(prediction.advantage_uncertainty[index]),
                    "q_cost_skip": float(prediction.q_cost[index, 0]),
                    "q_cost_verify": float(prediction.q_cost[index, 1]),
                    "additional_cost": float(prediction.additional_cost[index]),
                    "additional_cost_uncertainty": float(prediction.additional_cost_uncertainty[index]),
                    "immediate_cost": float(prediction.immediate_cost[index]),
                    "immediate_cost_uncertainty": float(prediction.immediate_cost_uncertainty[index]),
                    "prob_harmful": float(prediction.effect_probabilities[index, 0]),
                    "prob_no_change": float(prediction.effect_probabilities[index, 1]),
                    "prob_helpful": float(prediction.effect_probabilities[index, 2]),
                }
            )
    return rows


def _prediction_change_rows(
    before: list[V05Prediction],
    after: list[V05Prediction],
    states: list[MultiHistoryState],
    allowances: list[float],
    *,
    variant: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for allowance, first, second in zip(allowances, before, after, strict=True):
        for index, state in enumerate(states):
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "allowance": allowance,
                    "task_key": state.task_key,
                    "dataset": state.dataset,
                    "checkpoint": state.checkpoint,
                    "prefix": state.prefix or "START",
                    "q_skip_i0": float(first.q_success[index, 0]),
                    "q_skip_i1": float(second.q_success[index, 0]),
                    "q_skip_delta": float(second.q_success[index, 0] - first.q_success[index, 0]),
                    "q_verify_i0": float(first.q_success[index, 1]),
                    "q_verify_i1": float(second.q_success[index, 1]),
                    "q_verify_delta": float(second.q_success[index, 1] - first.q_success[index, 1]),
                    "advantage_i0": float(first.advantage[index]),
                    "advantage_i1": float(second.advantage[index]),
                    "advantage_delta": float(second.advantage[index] - first.advantage[index]),
                    "uncertainty_i0": float(first.advantage_uncertainty[index]),
                    "uncertainty_i1": float(second.advantage_uncertainty[index]),
                    "uncertainty_delta": float(
                        second.advantage_uncertainty[index] - first.advantage_uncertainty[index]
                    ),
                    "predicted_cost_i0": float(first.additional_cost[index]),
                    "predicted_cost_i1": float(second.additional_cost[index]),
                    "predicted_cost_delta": float(
                        second.additional_cost[index] - first.additional_cost[index]
                    ),
                }
            )
    return rows


def _target_error_rows(
    targets: V05TargetTable,
    predictions: list[V05Prediction],
    states: list[MultiHistoryState],
    allowances: list[float],
    *,
    variant: str,
    seed: int,
    iteration: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, state_index in enumerate(targets.state_indices):
        allowance = float(targets.allowance[row_index])
        prediction = nearest_prediction(predictions, allowances, allowance)
        state = states[int(state_index)]
        for action_index, action in enumerate(("S", "V")):
            valid = bool(targets.action_mask[row_index, action_index])
            predicted_success = float(prediction.q_success[int(state_index), action_index])
            target_success = float(targets.q_success[row_index, action_index])
            predicted_cost = float(prediction.q_cost[int(state_index), action_index])
            target_cost = float(targets.q_cost[row_index, action_index])
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "iteration": iteration,
                    "allowance": allowance,
                    "task_key": state.task_key,
                    "dataset": state.dataset,
                    "checkpoint": state.checkpoint,
                    "prefix": state.prefix or "START",
                    "action": action,
                    "valid_target": int(valid),
                    "target_success": target_success,
                    "predicted_success": predicted_success,
                    "success_error": predicted_success - target_success,
                    "success_squared_error": (predicted_success - target_success) ** 2,
                    "target_cumulative_cost": target_cost,
                    "predicted_cumulative_cost": predicted_cost,
                    "cost_absolute_error": abs(predicted_cost - target_cost),
                }
            )
    return rows


def _target_summaries(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["valid_target"]):
            groups[
                (
                    row["variant"],
                    row["seed"],
                    row["iteration"],
                    row["allowance"],
                    row["checkpoint"],
                    row["action"],
                )
            ].append(row)
    summaries: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        target = np.asarray([float(row["target_success"]) for row in values])
        predicted = np.asarray([float(row["predicted_success"]) for row in values])
        cost_error = np.asarray([float(row["cost_absolute_error"]) for row in values])
        summaries.append(
            {
                "variant": key[0],
                "seed": key[1],
                "iteration": key[2],
                "allowance": key[3],
                "checkpoint": key[4],
                "action": key[5],
                "count": len(values),
                "target_mean": float(target.mean()),
                "target_std": float(target.std()),
                "target_positive_rate": float((target > 0.5).mean()),
                "predicted_mean": float(predicted.mean()),
                "prediction_bias": float((predicted - target).mean()),
                "brier": float(np.mean((predicted - target) ** 2)),
                "success_mae": float(np.mean(np.abs(predicted - target))),
                "cumulative_cost_mae": float(cost_error.mean()),
            }
        )
    calibration_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not int(row["valid_target"]):
            continue
        probability = min(max(float(row["predicted_success"]), 0.0), 1.0)
        bin_index = min(int(probability * 10), 9)
        calibration_groups[
            (
                row["variant"],
                row["seed"],
                row["iteration"],
                row["action"],
                bin_index,
            )
        ].append(row)
    calibration: list[dict[str, Any]] = []
    for key, values in sorted(calibration_groups.items()):
        calibration.append(
            {
                "variant": key[0],
                "seed": key[1],
                "iteration": key[2],
                "action": key[3],
                "probability_bin": key[4],
                "bin_low": key[4] / 10,
                "bin_high": (key[4] + 1) / 10,
                "count": len(values),
                "predicted_mean": float(np.mean([float(row["predicted_success"]) for row in values])),
                "target_mean": float(np.mean([float(row["target_success"]) for row in values])),
            }
        )
    return summaries, calibration


def _multi_verify_rows(
    groups: dict[str, Any],
    task_rows: list[dict[str, Any]],
    variants: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    natural = [
        row
        for row in task_rows
        if row["analysis_family"] == "policy_improvement"
        and row["allowance_mode"] == "dynamic"
        and int(row["model_seed"]) == int(row["order_seed"])
    ]
    selections = Counter(
        (
            str(row["variant"]),
            int(row["iteration"]),
            str(row["task_key"]),
            str(row["sequence"]),
        )
        for row in natural
    )
    rows: list[dict[str, Any]] = []
    for task_key, group in sorted(groups.items()):
        baseline = int(_passed(group.rows["SSS"]))
        for sequence in MULTI_SEQUENCES:
            outcome = int(_passed(group.rows[sequence]))
            relevant_singles = [
                "".join("V" if index == position else "S" for index in range(3))
                for position, action in enumerate(sequence)
                if action == "V"
            ]
            single_outcomes = [int(_passed(group.rows[item])) for item in relevant_singles]
            best_single = max(single_outcomes)
            row: dict[str, Any] = {
                "task_key": task_key,
                "dataset": group.task.dataset,
                "sequence": sequence,
                "baseline_correctness": baseline,
                "sequence_correctness": outcome,
                "best_constituent_single_correctness": best_single,
                "gain_vs_baseline": outcome - baseline,
                "incremental_vs_best_single": outcome - best_single,
                "strict_positive_synergy": int(outcome == 1 and baseline == 0 and best_single == 0),
                "destructive_interaction": int(outcome == 0 and best_single == 1),
                "teacher_tokens": int(group.rows[sequence].get("teacher_total_tokens", 0) or 0),
            }
            for variant in variants:
                for iteration in (0, 1):
                    row[f"selected_{variant}_i{iteration}"] = selections[
                        (variant, iteration, task_key, sequence)
                    ]
            rows.append(row)
    summary: list[dict[str, Any]] = []
    for sequence in MULTI_SEQUENCES:
        selected = [row for row in rows if row["sequence"] == sequence]
        item: dict[str, Any] = {
            "sequence": sequence,
            "tasks": len(selected),
            "strict_positive_synergy_tasks": sum(int(row["strict_positive_synergy"]) for row in selected),
            "destructive_interaction_tasks": sum(int(row["destructive_interaction"]) for row in selected),
            "positive_gain_tasks": sum(int(row["gain_vs_baseline"] > 0) for row in selected),
        }
        for variant in variants:
            for iteration in (0, 1):
                item[f"selected_{variant}_i{iteration}"] = sum(
                    int(row[f"selected_{variant}_i{iteration}"]) for row in selected
                )
                item[f"selected_strict_synergy_{variant}_i{iteration}"] = sum(
                    int(row[f"selected_{variant}_i{iteration}"]) * int(row["strict_positive_synergy"])
                    for row in selected
                )
        summary.append(item)
    return rows, summary


def _plots(
    output: Path,
    run_rows: list[dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    multi_summary: list[dict[str, Any]],
    seeds: list[int],
) -> None:
    figure_root = output / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    natural = [
        row
        for row in run_rows
        if row["analysis_family"] == "cross_seed"
        and row["allowance_mode"] == "dynamic"
        and int(row["model_seed"]) == int(row["order_seed"])
    ]
    natural.sort(key=lambda row: int(row["model_seed"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    labels = [str(row["model_seed"])[-5:] for row in natural]
    axes[0].bar(labels, [int(row["teacher_tokens"]) / 1000 for row in natural], color="#4477AA")
    axes[0].set(
        title="Selected Myopic i0 token use by seed",
        ylabel="Teacher tokens (K)",
        xlabel="Seed suffix",
    )
    bottoms = np.zeros(len(natural))
    colors = plt.cm.Set3(np.linspace(0, 1, len(SEQUENCES)))
    for color, sequence in zip(colors, SEQUENCES, strict=True):
        values = np.asarray([int(row[f"sequence_{sequence}"]) for row in natural])
        axes[1].bar(labels, values, bottom=bottoms, label=sequence, color=color)
        bottoms += values
    axes[1].set(title="Full action-sequence distribution", ylabel="Validation tasks", xlabel="Seed suffix")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    for axis in axes:
        axis.grid(axis="y", color="#E5E7EB")
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_root / "cross_seed_tokens_sequences.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in iteration_rows:
        grouped[(str(row["variant"]), int(row["seed"]))].append(row)
    for (variant, seed), rows in sorted(grouped.items()):
        acc0 = np.mean([int(row["correct_i0"]) for row in rows])
        acc1 = np.mean([int(row["correct_i1"]) for row in rows])
        token0 = sum(int(row["tokens_i0"]) for row in rows) / 1000
        token1 = sum(int(row["tokens_i1"]) for row in rows) / 1000
        label = f"{variant} · {str(seed)[-5:]}"
        axes[0].plot([0, 1], [100 * acc0, 100 * acc1], marker="o", label=label)
        axes[1].plot([0, 1], [token0, token1], marker="o", label=label)
    axes[0].set(title="Accuracy before/after policy iteration", ylabel="Accuracy (%)")
    axes[1].set(title="Tokens before/after policy iteration", ylabel="Teacher tokens (K)")
    for axis in axes:
        axis.set_xticks([0, 1], ["i0", "i1"])
        axis.grid(color="#E5E7EB")
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_root / "iteration0_vs_iteration1.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for iteration in (0, 1):
        selected = [
            row
            for row in calibration_rows
            if row["variant"] == "conservative_dynamic"
            and int(row["iteration"]) == iteration
            and row["action"] == "V"
        ]
        by_bin: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_bin[int(row["probability_bin"])].append(row)
        x = [
            np.average(
                [float(row["predicted_mean"]) for row in values],
                weights=[int(row["count"]) for row in values],
            )
            for _key, values in sorted(by_bin.items())
        ]
        y = [
            np.average(
                [float(row["target_mean"]) for row in values],
                weights=[int(row["count"]) for row in values],
            )
            for _key, values in sorted(by_bin.items())
        ]
        axes[0].plot(x, y, marker="o", label=f"iteration {iteration}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777")
    axes[0].set(
        title="Verify Q calibration",
        xlabel="Predicted success",
        ylabel="Target success",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    sequences = [row["sequence"] for row in multi_summary]
    x_values = np.arange(len(sequences))
    axes[1].bar(
        x_values - 0.18,
        [int(row["strict_positive_synergy_tasks"]) for row in multi_summary],
        width=0.36,
        label="Available strict synergy",
        color="#2A9D8F",
    )
    axes[1].bar(
        x_values + 0.18,
        [int(row["selected_conservative_dynamic_i1"]) for row in multi_summary],
        width=0.36,
        label="Selected by continuation i1",
        color="#E76F51",
    )
    axes[1].set_xticks(x_values, sequences)
    axes[1].set(title="Multi-Verify synergy availability vs selection", ylabel="Task/selection count")
    for axis in axes:
        axis.grid(color="#E5E7EB")
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "q_calibration_multi_verify.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _report(
    output: Path,
    run_rows: list[dict[str, Any]],
    seed_comparison: list[dict[str, Any]],
    iteration_comparison: list[dict[str, Any]],
    target_summary: list[dict[str, Any]],
    multi_summary: list[dict[str, Any]],
    *,
    seeds: list[int],
) -> Path:
    natural = [
        row
        for row in run_rows
        if row["analysis_family"] == "cross_seed"
        and row["allowance_mode"] == "dynamic"
        and int(row["model_seed"]) == int(row["order_seed"])
    ]
    natural.sort(key=lambda row: int(row["model_seed"]))
    same_correct = sum(int(row["all_seeds_same_correctness"]) for row in seed_comparison)
    same_sequence = sum(int(row["all_seeds_same_sequence"]) for row in seed_comparison)
    low_token = min(natural, key=lambda row: int(row["teacher_tokens"]))
    high_token = max(natural, key=lambda row: int(row["teacher_tokens"]))
    low_seed = int(low_token["model_seed"])
    high_seed = int(high_token["model_seed"])
    low_high_same = sum(
        int(row[f"correct_seed_{low_seed}"] == row[f"correct_seed_{high_seed}"]) for row in seed_comparison
    )
    crossed_dynamic = [
        row
        for row in run_rows
        if row["analysis_family"] == "cross_seed" and row["allowance_mode"] == "dynamic"
    ]
    crossed_fixed = [
        row for row in run_rows if row["analysis_family"] == "cross_seed" and row["allowance_mode"] == "fixed"
    ]
    fixed_by_key = {(int(row["model_seed"]), int(row["order_seed"])): row for row in crossed_fixed}
    allocation_token_deltas = [
        int(row["teacher_tokens"])
        - int(fixed_by_key[(int(row["model_seed"]), int(row["order_seed"]))]["teacher_tokens"])
        for row in crossed_dynamic
    ]
    allocation_accuracy_deltas = [
        float(row["accuracy"])
        - float(fixed_by_key[(int(row["model_seed"]), int(row["order_seed"]))]["accuracy"])
        for row in crossed_dynamic
    ]
    common_order = seeds[0]
    common_order_rows = [row for row in crossed_dynamic if int(row["order_seed"]) == common_order]
    common_order_token_spread = max(int(row["teacher_tokens"]) for row in common_order_rows) - min(
        int(row["teacher_tokens"]) for row in common_order_rows
    )
    lines = [
        "# DARC v0.5 — Seed instability and policy-improvement diagnostics",
        "",
        "> Validation-only, read-only analysis. No training, frozen-test access, or LLM inference.",
        "",
        "## Cross-seed evidence",
        "",
        "| Model/order seed | Accuracy | Tokens | Calls | Helpful / Harmful / No-change |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in natural:
        lines.append(
            f"| {row['model_seed']} | {100 * float(row['accuracy']):.2f}% | "
            f"{int(row['teacher_tokens']):,} | {int(row['teacher_calls'])} | "
            f"{int(row['helpful_calls'])} / {int(row['harmful_calls'])} / "
            f"{int(row['no_change_calls'])} |"
        )
    lines.extend(
        [
            "",
            f"All seeds have identical final correctness on **{same_correct}/{len(seed_comparison)}** tasks, "
            f"but identical action sequences on only **{same_sequence}/{len(seed_comparison)}** tasks.",
            "",
            (
                f"The lowest-token seed **{low_seed}** uses "
                f"**{int(high_token['teacher_tokens']) - int(low_token['teacher_tokens']):,} fewer "
                f"tokens** and **{int(high_token['teacher_calls']) - int(low_token['teacher_calls'])} "
                f"fewer calls** than seed {high_seed}, while their correctness agrees on "
                f"**{low_high_same}/{len(seed_comparison)} tasks**. The high-token run contains "
                f"{int(high_token['no_change_calls']) - int(low_token['no_change_calls'])} more "
                "conditional No-change calls; this is direct evidence for redundant-call "
                "efficiency variance, not an accuracy advantage."
            ),
            "",
            "The model-seed × order-seed grid and dynamic-vs-fixed allowance replay are in "
            "`tables/replay_run_summary.csv`. They separate model prediction, task ordering, "
            "and budget-allocation effects without changing outcomes or using future information.",
            "",
            "| Held model seed | Token range across order seeds | Accuracy range across order seeds |",
            "|---:|---:|---:|",
        ]
    )
    for seed in seeds:
        selected = [row for row in crossed_dynamic if int(row["model_seed"]) == seed]
        lines.append(
            f"| {seed} | "
            f"{min(int(row['teacher_tokens']) for row in selected):,}–"
            f"{max(int(row['teacher_tokens']) for row in selected):,} | "
            f"{100 * min(float(row['accuracy']) for row in selected):.2f}%–"
            f"{100 * max(float(row['accuracy']) for row in selected):.2f}% |"
        )
    lines.extend(
        [
            "",
            (
                f"With task order fixed to seed {common_order}, changing only the model seed "
                f"produces a **{common_order_token_spread:,}-token spread**. Across all nine "
                f"model/order pairs, dynamic allowance changes tokens by "
                f"**{np.mean(allocation_token_deltas):+,.0f} on average** relative to fixed "
                f"allowance and changes accuracy by "
                f"**{100 * np.mean(allocation_accuracy_deltas):+.2f} pp on average**. These "
                "controlled replays quantify the three effects; they do not assert a causal "
                "mechanism beyond the observed policy/replay."
            ),
            "",
            "![Cross-seed behavior](../figures/cross_seed_tokens_sequences.png)",
            "",
            "## Iteration 0 versus iteration 1",
            "",
            "| Variant / seed | i0 accuracy → i1 | i0 tokens → i1 | Changed sequences | Wins / losses |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant in sorted({str(row["variant"]) for row in iteration_comparison}):
        for seed in seeds:
            selected = [
                row for row in iteration_comparison if row["variant"] == variant and int(row["seed"]) == seed
            ]
            acc0 = np.mean([int(row["correct_i0"]) for row in selected])
            acc1 = np.mean([int(row["correct_i1"]) for row in selected])
            token0 = sum(int(row["tokens_i0"]) for row in selected)
            token1 = sum(int(row["tokens_i1"]) for row in selected)
            changed = sum(int(row["action_sequence_changed"]) for row in selected)
            wins = sum(int(row["correctness_delta"] > 0) for row in selected)
            losses = sum(int(row["correctness_delta"] < 0) for row in selected)
            lines.append(
                f"| {variant} / {seed} | {100 * acc0:.2f}% → {100 * acc1:.2f}% | "
                f"{token0:,} → {token1:,} | {changed} | {wins} / {losses} |"
            )
    lines.append("")
    for variant in sorted({str(row["variant"]) for row in iteration_comparison}):
        selected = [row for row in iteration_comparison if row["variant"] == variant]
        changed = sum(int(row["action_sequence_changed"]) for row in selected)
        wins = sum(int(row["correctness_delta"] > 0) for row in selected)
        losses = sum(int(row["correctness_delta"] < 0) for row in selected)
        token_delta = sum(int(row["token_delta"]) for row in selected)
        lines.append(
            f"- **{variant}:** {changed} task-seed sequences changed; {wins} wins, "
            f"{losses} losses, aggregate token delta {token_delta:+,}."
        )
    lines.extend(
        [
            "",
            "![Policy iteration changes](../figures/iteration0_vs_iteration1.png)",
            "",
            "## Q-target error and calibration",
            "",
            "| Variant | Iteration | Action | Mean Brier | Mean cost MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant in sorted({str(row["variant"]) for row in target_summary}):
        for iteration in (0, 1):
            for action in ("S", "V"):
                selected = [
                    row
                    for row in target_summary
                    if row["variant"] == variant
                    and int(row["iteration"]) == iteration
                    and row["action"] == action
                ]
                lines.append(
                    f"| {variant} | {iteration} | {action} | "
                    f"{np.mean([float(row['brier']) for row in selected]):.4f} | "
                    f"{np.mean([float(row['cumulative_cost_mae']) for row in selected]):,.1f} |"
                )
    lines.extend(
        [
            "",
            "![Q calibration and multi-Verify](../figures/q_calibration_multi_verify.png)",
            "",
            "## Multi-Verify interactions",
            "",
            (
                "| Sequence | Strict synergy available | Destructive interactions | "
                "Selected by continuation i0 / i1 | Selected strict synergy i0 / i1 |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in multi_summary:
        lines.append(
            f"| {row['sequence']} | {row['strict_positive_synergy_tasks']} | "
            f"{row['destructive_interaction_tasks']} | "
            f"{row['selected_conservative_dynamic_i0']} / "
            f"{row['selected_conservative_dynamic_i1']} | "
            f"{row['selected_strict_synergy_conservative_dynamic_i0']} / "
            f"{row['selected_strict_synergy_conservative_dynamic_i1']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence-supported diagnosis",
            "",
            "1. **Seed instability is primarily an action-efficiency issue when final task "
            "correctness agrees.** The per-task table identifies which extra calls are "
            "conditional No-change and how much token variation they create.",
            "2. **Task ordering, model predictions, and allocation are reported separately.** "
            "Only effects visible in the crossed replay should be treated as evidence; causal "
            "claims beyond this replay remain hypotheses.",
            "3. **Policy improvement is useful only if changed sequences improve correctness "
            "or reduce tokens without increasing harmful calls.** The iteration table reports "
            "wins/losses rather than inferring improvement from Q-loss alone.",
            "4. **Multi-Verify learning is evaluated against observed strict-synergy branches.** "
            "Availability does not imply predictability; missed synergy may reflect sparse "
            "targets, prediction error, conservative margins, or all three.",
            "",
            "## Hypotheses requiring a future controlled intervention",
            "",
            "- Continuation targets may amplify Q/calibration error across checkpoints.",
            "- Rare strict-synergy trajectories may be insufficient for stable ranking.",
            "- Different seeds may learn equally correct but differently calibrated no-change "
            "regions, producing large token variance.",
            "",
            "These are hypotheses until tested without tuning on frozen test. This diagnostic "
            "run does not redesign or retrain DARC.",
            "",
            "## Artifacts",
            "",
            "- `tables/decision_trace_detailed.csv`",
            "- `tables/task_outcomes.csv` and `seed_task_comparison.csv`",
            "- `tables/replay_run_summary.csv`",
            "- `tables/iteration_task_comparison.csv` and `iteration_prediction_changes.csv`",
            "- `tables/q_target_errors.csv`, `bellman_target_summary.csv`, `q_calibration.csv`",
            "- `tables/multi_verify_synergy.csv` and `multi_verify_summary.csv`",
            "- `diagnostics_manifest.json`",
        ]
    )
    path = output / "report" / "darc_v05_seed_policy_diagnostics.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_diagnostics(
    config_path: Path,
    *,
    artifact_root: Path,
    output: Path,
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["evaluation"]["partition"] != "validation":
        raise ValueError("Diagnostics must remain validation-only")
    if config["evaluation"]["frozen_test_access"] != "forbidden":
        raise ValueError("Frozen-test access must remain forbidden")
    manifest = json.loads((artifact_root / "selection_manifest.json").read_text(encoding="utf-8"))
    if bool(manifest.get("frozen_test_loaded")):
        raise ValueError("Refusing artifacts whose manifest loaded frozen test")
    for child in ("tables", "figures", "report"):
        (output / child).mkdir(parents=True, exist_ok=True)
    frozen_root = Path(config["data"]["frozen_root"])
    train_groups = _load_groups(frozen_root, "train")
    validation_groups = _load_groups(frozen_root, "validation")
    _assert_disjoint(train_groups, validation_groups)
    del train_groups
    states, state_lookup = build_states(validation_groups)
    encoded, cache_manifest = encode_or_load(
        [state.encoder_example for state in states],
        split="darc_v05_validation",
        config=config["semantic_encoder"],
        cache_dir=Path(config["semantic_encoder"]["cache_dir"]),
    )
    seeds = [int(value) for value in config["reproducibility"]["seeds"]]
    fractions = [float(value) for value in config["offline_rl"]["budget_fractions"]]
    reference_budget = float(manifest["reference_budget_per_task"])
    allowances = [fraction * reference_budget for fraction in fractions]
    hard_cap = int(config["offline_rl"]["hard_call_token_cap"])
    token_budget = int(round(reference_budget * len(validation_groups)))
    device = str(config["semantic_encoder"]["device"])
    variants = ["myopic_targets", "conservative_dynamic"]

    prediction_cache: dict[tuple[str, int, int], list[V05Prediction]] = {}
    policy_cache: dict[tuple[str, int, int], ConservativePolicy] = {}
    for variant in variants:
        for seed in seeds:
            for iteration in (0, 1):
                model = DARCv05Ensemble.load(
                    artifact_root / "checkpoints" / variant / f"seed_{seed}" / f"iteration_{iteration}.pt"
                )
                prediction_cache[(variant, seed, iteration)] = [
                    model.predict(
                        encoded,
                        states,
                        allowance=allowance,
                        device_name=device,
                    )
                    for allowance in allowances
                ]
                policy_cache[(variant, seed, iteration)] = _selected_policy(
                    artifact_root,
                    seed=seed,
                    variant=variant,
                    iteration=iteration,
                )

    run_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    replay_cache: dict[
        tuple[Any, ...],
        tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]],
    ] = {}

    def replay_once(
        *,
        variant: str,
        iteration: int,
        model_seed: int,
        order_seed: int,
        allowance_mode: str,
        family: str,
    ) -> None:
        key = (variant, iteration, model_seed, order_seed, allowance_mode, family)
        if key not in replay_cache:
            replay_cache[key] = detailed_replay(
                validation_groups,
                state_lookup,
                prediction_cache[(variant, model_seed, iteration)],
                allowances,
                policy_cache[(variant, model_seed, iteration)],
                reference_budget=reference_budget,
                token_budget=token_budget,
                hard_call_token_cap=hard_cap,
                model_seed=model_seed,
                order_seed=order_seed,
                variant=variant,
                iteration=iteration,
                allowance_mode=allowance_mode,
                analysis_family=family,
            )
        summary, tasks, decisions = replay_cache[key]
        run_rows.append(summary)
        task_rows.extend(tasks)
        decision_rows.extend(decisions)

    crossed = [
        (model_seed, order_seed, allowance_mode)
        for model_seed in seeds
        for order_seed in seeds
        for allowance_mode in ("dynamic", "fixed")
    ]
    for model_seed, order_seed, allowance_mode in tqdm(crossed, desc="Cross-seed/order replay", unit="run"):
        replay_once(
            variant="myopic_targets",
            iteration=0,
            model_seed=model_seed,
            order_seed=order_seed,
            allowance_mode=allowance_mode,
            family="cross_seed",
        )
    improvement_jobs = [
        (variant, seed, iteration) for variant in variants for seed in seeds for iteration in (0, 1)
    ]
    for variant, seed, iteration in tqdm(improvement_jobs, desc="Policy-improvement replay", unit="run"):
        replay_once(
            variant=variant,
            iteration=iteration,
            model_seed=seed,
            order_seed=seed,
            allowance_mode="dynamic",
            family="policy_improvement",
        )

    seed_comparison = _seed_task_comparison(task_rows, seeds)
    iteration_comparison = _iteration_comparison(task_rows, variants, seeds)
    state_prediction_rows: list[dict[str, Any]] = []
    prediction_change_rows: list[dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            for iteration in (0, 1):
                state_prediction_rows.extend(
                    _prediction_rows(
                        states,
                        prediction_cache[(variant, seed, iteration)],
                        allowances,
                        variant=variant,
                        seed=seed,
                        iteration=iteration,
                    )
                )
            prediction_change_rows.extend(
                _prediction_change_rows(
                    prediction_cache[(variant, seed, 0)],
                    prediction_cache[(variant, seed, 1)],
                    states,
                    allowances,
                    variant=variant,
                    seed=seed,
                )
            )

    target_error_rows: list[dict[str, Any]] = []
    validation_fold = {key: 0 for key in validation_groups}
    v04_root = Path(config["data"]["darc_v04_audit"])

    def always_skip_factory(_state: MultiHistoryState, _allowance: int) -> Any:
        return lambda _observation: "S"

    for variant in variants:
        for seed in tqdm(seeds, desc=f"Q-target audit {variant}", unit="seed"):
            for iteration in (0, 1):
                if variant == "myopic_targets":
                    continuation_factory: Any = always_skip_factory
                    target_mode = "myopic"
                elif iteration == 0:
                    initial = DARCv04Ensemble.load(v04_root / "models" / f"seed_{seed}" / "myopic_targets.pt")
                    initial_predictions = _precompute(initial, encoded, states, allowances, device=device)
                    initial_policy = _selected_v04_policy(v04_root, seed, "myopic_targets")
                    continuation_factory = _v04_continuation_factory(
                        initial_predictions,
                        allowances,
                        initial_policy,
                        reference_budget=reference_budget,
                        hard_cap=hard_cap,
                    )
                    target_mode = "policy_aware"
                else:
                    continuation_factory = _v05_continuation_factory(
                        {0: prediction_cache[(variant, seed, 0)]},
                        allowances,
                        validation_fold,
                        policy_cache[(variant, seed, 0)],
                        reference_budget=reference_budget,
                        hard_cap=hard_cap,
                    )
                    target_mode = "policy_aware"
                targets = generate_policy_targets(
                    states,
                    validation_groups,
                    state_lookup,
                    allowances=allowances,
                    reference_budget=reference_budget,
                    hard_call_token_cap=hard_cap,
                    continuation_factory=continuation_factory,
                    target_mode=target_mode,
                )
                validate_target_coverage(
                    targets,
                    name=f"diagnostic/{variant}/seed_{seed}/iteration_{iteration}",
                )
                target_error_rows.extend(
                    _target_error_rows(
                        targets,
                        prediction_cache[(variant, seed, iteration)],
                        states,
                        allowances,
                        variant=variant,
                        seed=seed,
                        iteration=iteration,
                    )
                )
    target_summary, calibration = _target_summaries(target_error_rows)
    multi_rows, multi_summary = _multi_verify_rows(validation_groups, task_rows, variants)

    table_root = output / "tables"
    _write_csv(table_root / "replay_run_summary.csv", run_rows)
    _write_csv(table_root / "task_outcomes.csv", task_rows)
    _write_csv(table_root / "decision_trace_detailed.csv", decision_rows)
    _write_csv(table_root / "seed_task_comparison.csv", seed_comparison)
    _write_csv(table_root / "iteration_task_comparison.csv", iteration_comparison)
    _write_csv(table_root / "state_predictions.csv", state_prediction_rows)
    _write_csv(table_root / "iteration_prediction_changes.csv", prediction_change_rows)
    _write_csv(table_root / "q_target_errors.csv", target_error_rows)
    _write_csv(table_root / "bellman_target_summary.csv", target_summary)
    _write_csv(table_root / "q_calibration.csv", calibration)
    _write_csv(table_root / "multi_verify_synergy.csv", multi_rows)
    _write_csv(table_root / "multi_verify_summary.csv", multi_summary)
    _plots(output, run_rows, iteration_comparison, calibration, multi_summary, seeds)
    report = _report(
        output,
        run_rows,
        seed_comparison,
        iteration_comparison,
        target_summary,
        multi_summary,
        seeds=seeds,
    )
    diagnostic_manifest = {
        "schema_version": "darc-v0.5-diagnostics-v1",
        "source_artifacts": str(artifact_root),
        "source_selection_manifest": manifest,
        "partition_loaded": "validation",
        "frozen_test_loaded": False,
        "training_run": False,
        "llm_inference": False,
        "seeds": seeds,
        "variants": variants,
        "iterations": [0, 1],
        "task_count": len(validation_groups),
        "state_count": len(states),
        "reference_budget_per_task": reference_budget,
        "hard_call_token_cap": hard_cap,
        "embedding_cache": cache_manifest,
        "output_tables": sorted(path.name for path in table_root.glob("*.csv")),
        "report": str(report),
    }
    (output / "diagnostics_manifest.json").write_text(
        json.dumps(diagnostic_manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report
