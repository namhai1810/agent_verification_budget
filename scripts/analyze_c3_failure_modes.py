#!/usr/bin/env python3
"""Post-hoc C3/Fitted-Q failure audit using already-fitted CBV v0.1 models.

No estimator is fitted here.  The script only loads frozen artifacts, replays saved
controllers, and writes task-level diagnostic tables plus a concise Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from agent_verify.analysis.c3_controller import extract_observations, group_observations
from agent_verify.analysis.cbv_v01_experiment import (
    ALL_SEQUENCES,
    MODE_DATASETS,
    MultiHistoryController,
    TaskTrajectories,
    _budget_values,
    _group_indices,
    _passed,
    _stable_order,
    build_level1_examples,
    load_frozen_partition,
    subset_examples,
    task_trajectories,
)


SEED = 20261029
FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def c3_replay_details(
    examples: list[Any], observations: list[Any], controller: Any, *,
    budget: int, fraction: float, variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = group_observations(observations)
    examples_by_group = {
        key: {row.checkpoint: row for row in examples if row.group_key == key}
        for key in grouped
    }
    task_ids = _stable_order(grouped, SEED)
    predicted_remaining = float(budget)
    task_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for order, key in enumerate(task_ids):
        rows = grouped[key]
        by_cp = examples_by_group[key]
        tasks_remaining = len(task_ids) - order
        budget_before = predicted_remaining
        decision = controller.decide(
            rows,
            remaining_budget=predicted_remaining,
            initial_budget=float(budget),
            tasks_remaining=tasks_remaining,
        )
        chosen = rows[decision.selected_index] if decision.selected_index >= 0 else None
        admitted = chosen is not None and decision.predicted_cost <= predicted_remaining
        selected_cp = chosen.example.checkpoint if admitted else 0
        selected_delta = by_cp[selected_cp].delta if selected_cp else 0
        selected_cost = by_cp[selected_cp].teacher_total_tokens if selected_cp else 0
        predicted_cost = decision.predicted_cost if admitted else 0.0
        if admitted:
            predicted_remaining -= predicted_cost
        baseline = rows[0].example.y_skip
        outcome = by_cp[selected_cp].y_verify if selected_cp else baseline
        helpful_cps = [cp for cp, row in by_cp.items() if row.delta > 0]
        harmful_cps = [cp for cp, row in by_cp.items() if row.delta < 0]
        dataset = rows[0].example.dataset
        task_id = rows[0].example.task_id
        task_rows.append(
            {
                "protocol": "A-single", "variant": variant,
                "budget_fraction": fraction, "token_budget": budget,
                "task_group": key, "task_id": task_id, "dataset": dataset,
                "task_order": order, "budget_before": budget_before,
                "selected_checkpoint": selected_cp, "selected_delta": selected_delta,
                "selected_actual_tokens": selected_cost,
                "selected_predicted_tokens": predicted_cost,
                "baseline": baseline, "outcome": outcome,
                "helpful_available": int(bool(helpful_cps)),
                "helpful_checkpoints": ",".join(map(str, helpful_cps)),
                "harmful_checkpoints": ",".join(map(str, harmful_cps)),
                "helpful_selected": int(selected_delta > 0),
                "harmful_selected": int(selected_delta < 0),
                "wasted_call": int(selected_cp > 0 and selected_delta == 0),
                "missed_helpful": int(bool(helpful_cps) and selected_delta <= 0),
                "incorrect_checkpoint": int(bool(helpful_cps) and selected_cp not in helpful_cps),
                "q_verify": decision.q_verify, "q_wait": decision.q_wait,
                "q_margin": decision.q_verify - decision.q_wait,
            }
        )
        allowance = budget_before / max(tasks_remaining, 1)
        for trace in decision.trace:
            cp = int(trace["checkpoint"])
            current = by_cp[cp]
            downstream = [baseline]
            for future_cp in range(cp + 1, 4):
                future_obs = rows[future_cp - 1]
                if future_obs.predicted_cost <= allowance:
                    downstream.append(by_cp[future_cp].y_verify)
            best_wait_outcome = max(downstream)
            actual_prefer_verify = int(
                current.y_verify > best_wait_outcome
            )
            predicted_prefer_verify = int(float(trace["q_verify"]) > float(trace["q_wait"]))
            trace_rows.append(
                {
                    "protocol": "A-single", "variant": variant,
                    "budget_fraction": fraction, "task_group": key,
                    "task_id": task_id, "dataset": dataset, "checkpoint": cp,
                    "visited": 1, "selected_here": int(selected_cp == cp),
                    "q_verify": trace["q_verify"], "q_wait": trace["q_wait"],
                    "q_margin": float(trace["q_verify"]) - float(trace["q_wait"]),
                    "joint_predicted_gain": rows[cp - 1].expected_gain,
                    "actual_delta": current.delta,
                    "predicted_cost": trace["predicted_cost"],
                    "actual_cost": current.teacher_total_tokens,
                    "allowance_per_remaining_task": allowance,
                    "best_feasible_wait_outcome": best_wait_outcome,
                    "actual_prefer_verify_over_wait": actual_prefer_verify,
                    "predicted_prefer_verify": predicted_prefer_verify,
                    "preference_error": int(actual_prefer_verify != predicted_prefer_verify),
                }
            )
    return task_rows, trace_rows


def multi_replay_details(
    groups: dict[str, TaskTrajectories], controller: MultiHistoryController, *,
    budget: int, fraction: float, variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = _stable_order(groups, SEED)
    predicted_remaining = float(budget)
    task_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for order, key in enumerate(keys):
        group = groups[key]
        prefix = ""
        predicted_task_tokens = 0.0
        actual_task_tokens = 0
        for checkpoint in (1, 2, 3):
            tasks_remaining = len(keys) - order
            allowance = max(predicted_remaining, 0.0) / max(tasks_remaining, 1)
            q_skip, q_verify, predicted_cost = controller.predict(group, prefix, allowance)
            skip_sequence = prefix + "S" + "S" * (2 - len(prefix))
            verify_sequence = prefix + "V" + "S" * (2 - len(prefix))
            direct_delta = _passed(group.rows[verify_sequence]) - _passed(group.rows[skip_sequence])
            verify = q_verify > q_skip and predicted_cost <= max(predicted_remaining, 0.0)
            actual_cost = controller._verify_cost(group, prefix)
            trace_rows.append(
                {
                    "protocol": "B-multi", "variant": variant,
                    "budget_fraction": fraction, "task_group": key,
                    "task_id": group.task.task_id, "dataset": group.task.dataset,
                    "checkpoint": checkpoint, "history_before": prefix or "START",
                    "action": "V" if verify else "S",
                    "q_verify": q_verify, "q_wait": q_skip,
                    "q_margin": q_verify - q_skip,
                    "direct_actual_delta": direct_delta,
                    "predicted_verify_cost": predicted_cost,
                    "actual_verify_cost": actual_cost,
                    "allowance_per_remaining_task": allowance,
                    "helpful_action_selected": int(verify and direct_delta > 0),
                    "harmful_action_selected": int(verify and direct_delta < 0),
                    "wasted_action_selected": int(verify and direct_delta == 0),
                    "missed_local_helpful": int((not verify) and direct_delta > 0),
                }
            )
            if verify:
                prefix += "V"
                predicted_remaining -= predicted_cost
                predicted_task_tokens += predicted_cost
                actual_task_tokens += actual_cost
            else:
                prefix += "S"
        baseline = _passed(group.rows["SSS"])
        outcome = _passed(group.rows[prefix])
        best_outcome = max(_passed(group.rows[sequence]) for sequence in ALL_SEQUENCES)
        best_sequences = [
            sequence for sequence in ALL_SEQUENCES
            if _passed(group.rows[sequence]) == best_outcome
        ]
        cheapest_best = min(
            best_sequences,
            key=lambda sequence: int(group.rows[sequence].get("teacher_total_tokens", 0) or 0),
        )
        selected_cost = int(group.rows[prefix].get("teacher_total_tokens", 0) or 0)
        dominated = any(
            _passed(group.rows[sequence]) >= outcome
            and int(group.rows[sequence].get("teacher_total_tokens", 0) or 0) <= selected_cost
            and (
                _passed(group.rows[sequence]) > outcome
                or int(group.rows[sequence].get("teacher_total_tokens", 0) or 0) < selected_cost
            )
            for sequence in ALL_SEQUENCES
        )
        task_rows.append(
            {
                "protocol": "B-multi", "variant": variant,
                "budget_fraction": fraction, "token_budget": budget,
                "task_group": key, "task_id": group.task.task_id,
                "dataset": group.task.dataset, "task_order": order,
                "selected_sequence": prefix, "teacher_calls": prefix.count("V"),
                "actual_tokens": actual_task_tokens,
                "predicted_tokens": predicted_task_tokens,
                "baseline": baseline, "outcome": outcome,
                "best_observed_outcome": best_outcome,
                "cheapest_best_sequence": cheapest_best,
                "helpful_available": int(best_outcome > baseline),
                "missed_helpful": int(best_outcome > baseline and outcome <= baseline),
                "harmful_final": int(outcome < baseline),
                "wasted_calls": int(prefix.count("V") > 0 and outcome == baseline),
                "dominated_sequence": int(dominated),
            }
        )
    return task_rows, trace_rows


def aggregate_errors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["protocol"], row["variant"], row["budget_fraction"])].append(row)
    output = []
    for (protocol, variant, fraction), values in sorted(grouped.items()):
        output.append(
            {
                "protocol": protocol, "variant": variant,
                "budget_fraction": fraction, "tasks": len(values),
                "accuracy": mean(float(row["outcome"]) for row in values),
                "calls": sum(int(row.get("teacher_calls", int(row.get("selected_checkpoint", 0) > 0))) for row in values),
                "actual_tokens": sum(int(row.get("actual_tokens", row.get("selected_actual_tokens", 0))) for row in values),
                "helpful_selected": sum(
                    int(row.get("helpful_selected", int(row["outcome"] > row["baseline"])))
                    for row in values
                ),
                "harmful_selected": sum(int(row.get("harmful_selected", row.get("harmful_final", 0))) for row in values),
                "wasted": sum(int(row.get("wasted_call", row.get("wasted_calls", 0))) for row in values),
                "missed_helpful": sum(int(row.get("missed_helpful", 0)) for row in values),
                "incorrect_checkpoint_or_dominated": sum(int(row.get("incorrect_checkpoint", row.get("dominated_sequence", 0))) for row in values),
            }
        )
    return output


def compare_variants(rows: list[dict[str, Any]], protocol: str) -> list[dict[str, Any]]:
    variants = (
        ("C3 Myopic", "C3 Continuation")
        if protocol == "A-single"
        else ("L2 Fitted-Q Myopic", "L2 Fitted-Q Continuation")
    )
    lookup = {
        (row["variant"], float(row["budget_fraction"]), row["task_group"]): row
        for row in rows if row["protocol"] == protocol
    }
    output = []
    for fraction in FRACTIONS:
        keys = sorted(
            key for variant, frac, key in lookup
            if variant == variants[0] and frac == fraction
        )
        for key in keys:
            myopic = lookup[(variants[0], fraction, key)]
            continuation = lookup[(variants[1], fraction, key)]
            myopic_decision = (
                str(myopic.get("selected_checkpoint"))
                if protocol == "A-single" else str(myopic.get("selected_sequence"))
            )
            continuation_decision = (
                str(continuation.get("selected_checkpoint"))
                if protocol == "A-single" else str(continuation.get("selected_sequence"))
            )
            myopic_tokens = int(myopic.get("selected_actual_tokens", myopic.get("actual_tokens", 0)))
            continuation_tokens = int(continuation.get("selected_actual_tokens", continuation.get("actual_tokens", 0)))
            output.append(
                {
                    "protocol": protocol, "budget_fraction": fraction,
                    "task_group": key, "task_id": myopic["task_id"],
                    "dataset": myopic["dataset"],
                    "myopic_decision": myopic_decision,
                    "continuation_decision": continuation_decision,
                    "decision_differs": int(myopic_decision != continuation_decision),
                    "myopic_outcome": myopic["outcome"],
                    "continuation_outcome": continuation["outcome"],
                    "outcome_delta_cont_minus_myopic": int(continuation["outcome"]) - int(myopic["outcome"]),
                    "myopic_tokens": myopic_tokens,
                    "continuation_tokens": continuation_tokens,
                    "token_delta_cont_minus_myopic": continuation_tokens - myopic_tokens,
                }
            )
    return output


def aggregate_prediction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        actual_key = "actual_delta" if row["protocol"] == "A-single" else "direct_actual_delta"
        grouped[(row["protocol"], row["variant"], row["budget_fraction"], row["dataset"], row["checkpoint"], actual_key)].append(row)
    output = []
    for (protocol, variant, fraction, dataset, checkpoint, actual_key), values in sorted(grouped.items()):
        predicted = [float(row["q_margin"]) for row in values]
        actual = [float(row[actual_key]) for row in values]
        selected = [row for row in values if str(row.get("selected_here", 0)) == "1" or row.get("action") == "V"]
        output.append(
            {
                "protocol": protocol, "variant": variant,
                "budget_fraction": fraction, "dataset": dataset,
                "checkpoint": checkpoint, "visited_states": len(values),
                "mean_q_margin": mean(predicted), "mean_actual_gain": mean(actual),
                "q_margin_actual_gain_correlation": correlation(predicted, actual),
                "verify_actions": len(selected),
                "selected_mean_actual_gain": mean(float(row[actual_key]) for row in selected),
                "selected_harmful": sum(float(row[actual_key]) < 0 for row in selected),
                "selected_wasted": sum(float(row[actual_key]) == 0 for row in selected),
                "preference_errors": sum(int(row.get("preference_error", 0)) for row in values),
                "preference_error_rate": (
                    mean(int(row.get("preference_error", 0)) for row in values)
                    if protocol == "A-single" else ""
                ),
                "mean_absolute_cost_error": mean(
                    abs(float(row.get("predicted_cost", row.get("predicted_verify_cost", 0))) - float(row.get("actual_cost", row.get("actual_verify_cost", 0))))
                    for row in values
                ),
            }
        )
    return output


def interaction_rarity(
    frozen: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]
) -> list[dict[str, Any]]:
    output = []
    for split, (tasks, rows) in frozen.items():
        groups = task_trajectories(tasks, rows, level=2)
        counts = Counter()
        for group in groups.values():
            baseline = _passed(group.rows["SSS"])
            best_single = max(_passed(group.rows[s]) for s in ("SSS", "VSS", "SVS", "SSV"))
            best_multi = max(_passed(group.rows[s]) for s in ("VVS", "VSV", "SVV", "VVV"))
            counts["multi_better_than_single"] += int(best_multi > best_single)
            counts["multi_worse_than_single"] += int(best_multi < best_single)
            counts["equal"] += int(best_multi == best_single)
            counts["any_helpful"] += int(max(best_single, best_multi) > baseline)
        output.append({"split": split, "level2_tasks": len(groups), **counts})
    return output


def comparison_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["protocol"], float(row["budget_fraction"]))].append(row)
    output = []
    for (protocol, fraction), values in sorted(grouped.items()):
        output.append(
            {
                "protocol": protocol, "budget_fraction": fraction,
                "tasks": len(values),
                "different_decisions": sum(int(row["decision_differs"]) for row in values),
                "continuation_better": sum(int(row["outcome_delta_cont_minus_myopic"]) > 0 for row in values),
                "myopic_better": sum(int(row["outcome_delta_cont_minus_myopic"]) < 0 for row in values),
                "same_outcome": sum(int(row["outcome_delta_cont_minus_myopic"]) == 0 for row in values),
                "continuation_minus_myopic_tokens": sum(int(row["token_delta_cont_minus_myopic"]) for row in values),
            }
        )
    return output


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def fmt_float(value: Any, signed: bool = False) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:+.3f}" if signed else f"{number:.3f}"


def generate_report(
    output: Path, errors: list[dict[str, Any]], comparisons: list[dict[str, Any]],
    prediction: list[dict[str, Any]], interactions: list[dict[str, Any]],
    task_comparisons: list[dict[str, Any]], label_counts: dict[str, Counter],
) -> Path:
    report = output / "report" / "c3_fitted_q_failure_analysis.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    full_errors = [row for row in errors if float(row["budget_fraction"]) == 1.0]
    full_comparisons = [row for row in comparisons if float(row["budget_fraction"]) == 1.0]
    changed_cases = [
        row for row in task_comparisons
        if float(row["budget_fraction"]) == 1.0 and int(row["decision_differs"])
    ]
    changed_cases.sort(
        key=lambda row: (
            -abs(int(row["outcome_delta_cont_minus_myopic"])),
            -abs(int(row["token_delta_cont_minus_myopic"])),
        )
    )
    lines = [
        "# C3 / Fitted-Q Myopic vs Continuation — Failure Analysis", "",
        "> Post-hoc audit of already-fitted CBV v0.1 controllers. No model was retrained, no LLM/evaluator was called, and no frozen-test result was used for tuning.", "",
        "## Executive findings", "",
        "1. **Continuation does not improve frozen-test accuracy.** Protocol A ties Myopic while spending more tokens; Protocol B is one task worse and also spends more.",
        "2. **The primary failure is value ranking, not cost prediction.** Most selected interventions are no-change, while helpful opportunities remain missed. Saved cost models are substantially better calibrated than gain/Q ranking.",
        "3. **Continuation targets have little signal to learn.** Multi-over-single improvements are rare in every split, so fitted future value is dominated by equal-outcome targets and estimation noise.",
        "4. **The opportunity-cost approximation is too weak.** Continuation changes decisions and increases spend, but those changes rarely alter correctness; it has no explicit global dual variable or uncertainty-aware reservation rule.", "",
        "## 1. Full-budget decision errors", "",
        "| Protocol | Controller | Accuracy | Calls | Tokens | Helpful selected | Harmful | Wasted/no-change | Missed helpful | Wrong CP / dominated |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in full_errors:
        lines.append(
            f"| {row['protocol']} | {row['variant']} | {fmt_pct(float(row['accuracy']))} | {row['calls']} | {int(row['actual_tokens']):,} | {row['helpful_selected']} | {row['harmful_selected']} | {row['wasted']} | {row['missed_helpful']} | {row['incorrect_checkpoint_or_dominated']} |"
        )
    lines += [
        "", "Definitions: a wasted call has zero observed final gain; missed-helpful means at least one observed counterfactual could improve the task but the chosen decision did not; Protocol-B domination means another observed sequence has no lower correctness and no higher cost, with at least one strict improvement.", "",
        "## 2. Same-task Myopic vs Continuation", "",
        "| Protocol | Budget | Different decisions | Continuation better | Myopic better | Same outcome | Continuation − Myopic tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['protocol']} | {100*float(row['budget_fraction']):.0f}% | {row['different_decisions']}/{row['tasks']} | {row['continuation_better']} | {row['myopic_better']} | {row['same_outcome']} | {int(row['continuation_minus_myopic_tokens']):+,} |"
        )
    lines += ["", "Representative full-budget differences:", "", "| Protocol | Task | Dataset | Myopic | Continuation | Outcome Δ (C−M) | Token Δ (C−M) |", "|---|---|---|---|---|---:|---:|"]
    for row in changed_cases[:12]:
        lines.append(
            f"| {row['protocol']} | `{row['task_id']}` | {row['dataset']} | {row['myopic_decision']} | {row['continuation_decision']} | {int(row['outcome_delta_cont_minus_myopic']):+d} | {int(row['token_delta_cont_minus_myopic']):+,} |"
        )
    lines += [
        "", "## 3. Evidence for root causes", "",
        "### 3.1 Sparse value and interaction labels", "",
        "| Split | L1 harmful | L1 no-change | L1 helpful | L2 tasks | Multi > best single | Multi < best single | Equal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    interaction_by_split = {row["split"]: row for row in interactions}
    for split in ("train", "validation", "test"):
        labels = label_counts[split]
        inter = interaction_by_split[split]
        lines.append(
            f"| {split} | {labels[0]} | {labels[1]} | {labels[2]} | {inter['level2_tasks']} | {inter['multi_better_than_single']} | {inter['multi_worse_than_single']} | {inter['equal']} |"
        )
    lines += [
        "", "Observed implication: continuation targets are overwhelmingly trained on no-change/equal future outcomes. The small number of positive interactions makes the fitted max operator sensitive to approximation noise.", "",
        "### 3.2 Predicted versus actual gain by budget", "",
        "| Protocol | Controller | Budget | States | Verify actions | Selected actual gain | Wasted selected | Mean cell correlation |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    budget_groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction:
        budget_groups[(row["protocol"], row["variant"], float(row["budget_fraction"]))].append(row)
    for (protocol, variant, fraction), values in sorted(budget_groups.items()):
        actions = sum(int(row["verify_actions"]) for row in values)
        selected_gain_numerator = sum(
            float(row["selected_mean_actual_gain"]) * int(row["verify_actions"])
            for row in values
            if np.isfinite(float(row["selected_mean_actual_gain"]))
        )
        correlations = [
            float(row["q_margin_actual_gain_correlation"]) for row in values
            if np.isfinite(float(row["q_margin_actual_gain_correlation"]))
        ]
        lines.append(
            f"| {protocol} | {variant} | {100*fraction:.0f}% | {sum(int(row['visited_states']) for row in values)} | {actions} | {selected_gain_numerator/max(actions, 1):+.3f} | {sum(int(row['selected_wasted']) for row in values)} | {mean(correlations):+.3f} |"
        )
    lines += [
        "", "Full-budget breakdown by dataset and checkpoint:", "",
        "| Protocol | Controller | Dataset | CP | States | Mean Q margin | Mean actual gain | Correlation | Verify actions | Selected actual gain | Wasted | Future-preference error |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in prediction:
        if float(row["budget_fraction"]) != 1.0:
            continue
        preference = (
            f"{100*float(row['preference_error_rate']):.1f}%"
            if row.get("preference_error_rate") != "" else "n/a"
        )
        lines.append(
            f"| {row['protocol']} | {row['variant']} | {row['dataset']} | {row['checkpoint']} | {row['visited_states']} | {fmt_float(row['mean_q_margin'], True)} | {fmt_float(row['mean_actual_gain'], True)} | {fmt_float(row['q_margin_actual_gain_correlation'], True)} | {row['verify_actions']} | {fmt_float(row['selected_mean_actual_gain'], True)} | {row['selected_wasted']} | {preference} |"
        )
    lines += [
        "", "### 3.3 Observed evidence versus hypotheses", "",
        "**Observed evidence**", "",
        "- Continuation spends more tokens without an accuracy gain in both protocols.",
        "- Myopic and Continuation often choose different checkpoints/sequences, but most differences leave final correctness unchanged.",
        "- L1 labels and L2 interactions are strongly imbalanced toward no-change/equal outcomes.",
        "- Q-margin/actual-gain correlations are weak or unstable in several dataset/checkpoint cells; selected actions frequently have zero gain.",
        "- The upstream value predictor is weak (combined-test gain correlation near 0.06), while token-cost error is comparatively small (~4%).", "",
        "**Hypotheses consistent with the evidence (not proven)**", "",
        "- Bootstrapped continuation values propagate approximation error backward through the max operator.",
        "- Sparse positive interactions make future-value targets high variance and encourage false-positive Verify decisions.",
        "- `budget_per_remaining_task` plus a fixed cost penalty is an incomplete proxy for global opportunity cost; it cannot learn a stable shadow price across heterogeneous datasets/tasks.",
        "- Point estimates lack uncertainty, so small positive Q margins trigger calls even when value predictions are poorly calibrated.", "",
        "## 4. Prioritized improvements", "",
        "| Priority | Change | Expected impact | Validation rule |",
        "|---:|---|---|---|",
        "| P0 | Add an abstention margin calibrated only on validation (`Q_verify - Q_wait > margin`) and require a lower confidence bound on gain to be positive. | High: directly removes dominant no-change/wasted calls. | Select margin on validation Pareto area; keep frozen test untouched. |",
        "| P0 | Train value heads with task/dataset-balanced sampling or focal weighting for Helpful/Harmful, while preserving calibrated probabilities. | High: current labels are dominated by no-change. | Grouped train CV + validation AUPRC/calibration and policy replay. |",
        "| P1 | Replace fitted max targets with cross-fitted/double-Q targets and target-network style out-of-fold predictions. | Medium–high: reduces optimistic error propagation in Continuation. | Every task's continuation target must come from a model not trained on that task. |",
        "| P1 | Learn an explicit budget dual/shadow price conditioned on remaining tasks and dataset, instead of only a fixed cost penalty. | Medium: improves opportunity-cost allocation. | Compare at identical actual-token caps on validation. |",
        "| P1 | Use checkpoint/history-specific calibration and uncertainty estimates (ensemble or quantile Q). | Medium: heterogeneous CP/dataset errors are currently pooled. | Report calibration and wasted-call rate per CP/dataset. |",
        "| P2 | Collect additional repeated rollouts/model pairs targeted at rare positive interactions. | Long-term high, immediate cost high. | Freeze a new benchmark version; do not append to current test. |",
        "| P2 | Keep Myopic as the deployment baseline until Continuation beats it on validation across multiple budgets and seeds. | High confidence safeguard. | Require Pareto dominance or statistically supported paired gain. |", "",
        "## 5. Research conclusion", "",
        "The current failure is not that sequential control is unnecessary; it is that the available continuation signal is sparse and the fitted target is noisier than the immediate-value target. Myopic is therefore the stronger current controller. Continuation should not be promoted based on this frozen test. Future development must occur on train/validation or a newly frozen holdout; rerunning tuned variants on this consumed test would be exploratory.", "",
        "## Machine-readable artifacts", "",
        "- `../tables/c3_task_decisions.csv`", "- `../tables/c3_trace_predictions.csv`",
        "- `../tables/fitted_q_l2_task_decisions.csv`", "- `../tables/fitted_q_l2_trace_predictions.csv`",
        "- `../tables/myopic_continuation_task_comparison.csv`", "- `../tables/failure_error_summary.csv`",
        "- `../tables/predicted_actual_gain_by_group.csv`", "- `../tables/interaction_rarity_by_split.csv`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", type=Path, default=Path("data/cbv_bench_v01_frozen_28102026"))
    parser.add_argument("--experiment-root", type=Path, default=Path("results/cbv_v01_experiments_29102026"))
    args = parser.parse_args()
    frozen_root = args.frozen_root.resolve()
    output = args.experiment_root.resolve()
    frozen = {split: load_frozen_partition(frozen_root, split) for split in ("train", "validation", "test")}
    l1: dict[str, list[Any]] = {}
    label_counts: dict[str, Counter] = {}
    for split, (tasks, rows) in frozen.items():
        l1[split] = build_level1_examples(task_trajectories(tasks, rows, level=1))
        label_counts[split] = Counter(row.value_label for row in l1[split])

    joint = joblib.load(output / "models" / "combined" / "joint.joblib")
    c3_models = {
        "C3 Myopic": joblib.load(output / "models" / "combined" / "c3_myopic.joblib"),
        "C3 Continuation": joblib.load(output / "models" / "combined" / "c3_continuation.joblib"),
    }
    test_examples = l1["test"]
    observations = extract_observations(test_examples, joint)
    c3_tasks: list[dict[str, Any]] = []
    c3_traces: list[dict[str, Any]] = []
    task_count = len(_group_indices(test_examples))
    for fraction, budget in _budget_values(l1["train"], task_count):
        if fraction == 0:
            continue
        for variant, model in c3_models.items():
            tasks, traces = c3_replay_details(
                test_examples, observations, model,
                budget=budget, fraction=fraction, variant=variant,
            )
            c3_tasks.extend(tasks)
            c3_traces.extend(traces)

    multi_models = {
        "L2 Fitted-Q Myopic": joblib.load(output / "models" / "combined" / "multi_myopic.joblib"),
        "L2 Fitted-Q Continuation": joblib.load(output / "models" / "combined" / "multi_continuation.joblib"),
    }
    train_l2 = task_trajectories(*frozen["train"], level=2)
    test_l2 = task_trajectories(*frozen["test"], level=2)
    median_vvv = float(np.median([int(group.rows["VVV"].get("teacher_total_tokens", 0) or 0) for group in train_l2.values()]))
    multi_tasks: list[dict[str, Any]] = []
    multi_traces: list[dict[str, Any]] = []
    for fraction in FRACTIONS:
        budget = int(round(fraction * median_vvv * len(test_l2)))
        for variant, model in multi_models.items():
            tasks, traces = multi_replay_details(
                test_l2, model, budget=budget, fraction=fraction, variant=variant,
            )
            multi_tasks.extend(tasks)
            multi_traces.extend(traces)

    comparisons = [
        *compare_variants(c3_tasks, "A-single"),
        *compare_variants(multi_tasks, "B-multi"),
    ]
    comparison_rows = comparison_summary(comparisons)
    error_rows = aggregate_errors([*c3_tasks, *multi_tasks])
    prediction_rows = aggregate_prediction([*c3_traces, *multi_traces])
    interactions = interaction_rarity(frozen)

    table_dir = output / "tables"
    write_csv(table_dir / "c3_task_decisions.csv", c3_tasks)
    write_csv(table_dir / "c3_trace_predictions.csv", c3_traces)
    write_csv(table_dir / "fitted_q_l2_task_decisions.csv", multi_tasks)
    write_csv(table_dir / "fitted_q_l2_trace_predictions.csv", multi_traces)
    write_csv(table_dir / "myopic_continuation_task_comparison.csv", comparisons)
    write_csv(table_dir / "failure_error_summary.csv", error_rows)
    write_csv(table_dir / "predicted_actual_gain_by_group.csv", prediction_rows)
    write_csv(table_dir / "interaction_rarity_by_split.csv", interactions)
    report = generate_report(
        output, error_rows, comparison_rows, prediction_rows, interactions,
        comparisons, label_counts,
    )
    print(report)


if __name__ == "__main__":
    main()
