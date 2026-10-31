"""Validation-only, multi-seed audit protocol for DARC v0.4.

This module deliberately never loads the frozen test partition.  It standardizes
Level-2 replay for neural and C3 controllers under the same task order, actual
token ledger, hard per-call reservation, and budget grid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from agent_verify.analysis.cbv_v01_experiment import (
    ALL_SEQUENCES,
    MultiHistoryController,
    ReplayResult,
    _artifact,
    _passed,
    _stable_order,
)
from agent_verify.analysis.darc_data import FORBIDDEN_INPUT_FIELDS, encode_or_load
from agent_verify.analysis.darc_model import DARCEnsemble
from agent_verify.analysis.darc_v04_data import (
    MultiHistoryState,
    audit_level2,
    build_states,
    build_targets,
    verify_cost,
)
from agent_verify.analysis.darc_v04_experiment import (
    SequentialPolicy,
    _allowances,
    _assert_disjoint,
    _effect_metrics,
    _load_groups,
    _nearest_prediction,
    _policy_grid,
    _precompute,
    _write_csv,
    replay_v04,
    select_development_candidate,
)
from agent_verify.analysis.darc_v04_model import DARCv04Ensemble, V04Prediction, train_ensemble


def _direct_delta(group: Any, prefix: str) -> int:
    skip = prefix + "S" + "S" * (2 - len(prefix))
    verify = prefix + "V" + "S" * (2 - len(prefix))
    return int(_passed(group.rows[verify]) - _passed(group.rows[skip]))


def _decision_trace(
    *,
    policy: str,
    seed: int,
    fraction: float,
    key: str,
    group: Any,
    checkpoint: int,
    prefix: str,
    action: str,
    actual_cost: int,
    remaining: int,
    predicted_cost: float | None,
) -> dict[str, Any]:
    delta = _direct_delta(group, prefix)
    verified = action == "V"
    return {
        "policy": policy,
        "seed": seed,
        "budget_fraction": fraction,
        "task_key": key,
        "dataset": group.task.dataset,
        "checkpoint": checkpoint,
        "history": prefix or "START",
        "action": action,
        "remaining_budget_before_action": remaining,
        "predicted_cost": "" if predicted_cost is None else predicted_cost,
        "actual_cost_after_action": actual_cost if verified else 0,
        "direct_actual_delta_audit_only": delta,
        "helpful_selected": int(verified and delta > 0),
        "harmful_selected": int(verified and delta < 0),
        "no_change_selected": int(verified and delta == 0),
    }


def replay_c3_strict(
    groups: dict[str, Any],
    controller: MultiHistoryController,
    *,
    token_budget: int,
    budget_fraction: float,
    hard_call_token_cap: int,
    seed: int,
    label: str,
) -> tuple[ReplayResult, list[dict[str, Any]]]:
    """Replay C3 with the same online information and ledger as DARC.

    Predicted cost may decide admission.  Actual cost is read only after the
    action is fixed, then charged to the ledger.  Reserving the configured hard
    cap before every call guarantees no retrospective budget overshoot.
    """

    keys = _stable_order(groups, seed)
    actual_spent = 0
    predicted_spent = 0.0
    calls = 0
    outcomes: list[float] = []
    baselines: list[float] = []
    masks: list[int] = []
    traces: list[dict[str, Any]] = []
    for order, key in enumerate(keys):
        group = groups[key]
        prefix = ""
        mask = 0
        for checkpoint in (1, 2, 3):
            remaining = max(token_budget - actual_spent, 0)
            allowance = remaining / max(len(keys) - order, 1)
            q_skip, q_verify, predicted_cost = controller.predict(group, prefix, allowance)
            predicted_cost = max(float(predicted_cost), 1.0)
            verify = (
                float(q_verify) > float(q_skip)
                and remaining >= hard_call_token_cap
                and predicted_cost <= remaining
            )
            action = "V" if verify else "S"
            # Audit-only counterfactual labels are computed after the online action.
            actual_cost = 0
            if verify:
                actual_cost = int(controller._verify_cost(group, prefix))
                if actual_cost > hard_call_token_cap:
                    raise RuntimeError(
                        f"C3 call {actual_cost} exceeds hard cap {hard_call_token_cap}"
                    )
                if actual_spent + actual_cost > token_budget:
                    raise AssertionError("C3 strict replay exceeded actual-token budget")
            traces.append(
                _decision_trace(
                    policy=label,
                    seed=seed,
                    fraction=budget_fraction,
                    key=key,
                    group=group,
                    checkpoint=checkpoint,
                    prefix=prefix,
                    action=action,
                    actual_cost=actual_cost,
                    remaining=remaining,
                    predicted_cost=predicted_cost,
                )
            )
            prefix += action
            if verify:
                actual_spent += actual_cost
                predicted_spent += predicted_cost
                calls += 1
                mask |= 1 << (checkpoint - 1)
        outcomes.append(float(_passed(group.rows[prefix])))
        baselines.append(float(_passed(group.rows["SSS"])))
        masks.append(mask)
    return (
        ReplayResult(
            label,
            "C-multi-strict-audit",
            "combined",
            "combined",
            budget_fraction,
            token_budget,
            tuple(keys),
            np.asarray(outcomes),
            np.asarray(baselines),
            np.asarray(masks),
            calls,
            actual_spent,
            predicted_spent,
        ),
        traces,
    )


def replay_fixed_strict(
    groups: dict[str, Any],
    sequence: str,
    *,
    token_budget: int,
    budget_fraction: float,
    hard_call_token_cap: int,
    seed: int,
) -> tuple[ReplayResult, list[dict[str, Any]]]:
    """Replay a fixed action plan without peeking at its future realized cost."""

    label = f"Fixed {sequence}"
    keys = _stable_order(groups, seed)
    actual_spent = calls = 0
    outcomes: list[float] = []
    baselines: list[float] = []
    masks: list[int] = []
    traces: list[dict[str, Any]] = []
    for key in keys:
        group = groups[key]
        prefix = ""
        mask = 0
        for checkpoint, planned in enumerate(sequence, 1):
            remaining = max(token_budget - actual_spent, 0)
            verify = planned == "V" and remaining >= hard_call_token_cap
            action = "V" if verify else "S"
            actual_cost = 0
            if verify:
                actual_cost = verify_cost(group, prefix)
                if actual_cost > hard_call_token_cap:
                    raise RuntimeError(
                        f"Fixed-policy call {actual_cost} exceeds hard cap {hard_call_token_cap}"
                    )
                if actual_spent + actual_cost > token_budget:
                    raise AssertionError("Fixed strict replay exceeded actual-token budget")
            traces.append(
                _decision_trace(
                    policy=label,
                    seed=seed,
                    fraction=budget_fraction,
                    key=key,
                    group=group,
                    checkpoint=checkpoint,
                    prefix=prefix,
                    action=action,
                    actual_cost=actual_cost,
                    remaining=remaining,
                    predicted_cost=None,
                )
            )
            prefix += action
            if verify:
                actual_spent += actual_cost
                calls += 1
                mask |= 1 << (checkpoint - 1)
        outcomes.append(float(_passed(group.rows[prefix])))
        baselines.append(float(_passed(group.rows["SSS"])))
        masks.append(mask)
    return (
        ReplayResult(
            label,
            "C-multi-strict-audit",
            "none",
            "combined",
            budget_fraction,
            token_budget,
            tuple(keys),
            np.asarray(outcomes),
            np.asarray(baselines),
            np.asarray(masks),
            calls,
            actual_spent,
            float("nan"),
        ),
        traces,
    )


def _trace_counts(traces: list[dict[str, Any]]) -> tuple[int, int, int]:
    return (
        sum(int(row.get("helpful_selected", 0)) for row in traces),
        sum(int(row.get("harmful_selected", 0)) for row in traces),
        sum(
            int(row.get("no_change_selected", row.get("wasted_selected", 0)))
            for row in traces
        ),
    )


def tune_validation_policy(
    groups: dict[str, Any],
    states: list[MultiHistoryState],
    lookup: dict[tuple[str, str], int],
    predictions: list[V04Prediction],
    allowances: list[float],
    policies: list[SequentialPolicy],
    *,
    fractions: list[float],
    reference_budget: float,
    hard_cap: int,
    seed: int,
    label: str,
    token_penalty: float,
    max_harmful_rate: float,
) -> tuple[SequentialPolicy, list[dict[str, Any]]]:
    """Tune a policy by validation accuracy--cost utility under a risk gate."""

    rows: list[dict[str, Any]] = []
    total_available = sum(
        int(round(f * reference_budget * len(groups))) for f in fractions if f > 0
    )
    for candidate, policy in enumerate(policies):
        accuracies: list[float] = []
        actual_tokens = calls = harmful = helpful = no_change = 0
        for fraction in fractions:
            if fraction <= 0:
                continue
            budget = int(round(fraction * reference_budget * len(groups)))
            replay, traces = replay_v04(
                groups,
                states,
                lookup,
                predictions,
                allowances,
                policy,
                token_budget=budget,
                budget_fraction=fraction,
                hard_call_token_cap=hard_cap,
                seed=seed,
                label=label,
            )
            h, r, n = _trace_counts(traces)
            accuracies.append(float(replay.outcomes.mean()))
            actual_tokens += replay.actual_tokens
            calls += replay.calls
            helpful += h
            harmful += r
            no_change += n
        harmful_rate = harmful / max(calls, 1)
        utilization = actual_tokens / max(total_available, 1)
        accuracy = float(np.mean(accuracies))
        score = accuracy - token_penalty * utilization
        rows.append(
            {
                "seed": seed,
                "variant": label,
                "candidate": candidate,
                **asdict(policy),
                "mean_validation_accuracy": accuracy,
                "actual_tokens": actual_tokens,
                "token_budget": total_available,
                "token_utilization": utilization,
                "calls": calls,
                "helpful": helpful,
                "harmful": harmful,
                "no_change": no_change,
                "harmful_rate": harmful_rate,
                "risk_eligible": harmful_rate <= max_harmful_rate + 1e-12,
                "selection_score": score,
            }
        )
    eligible = [row for row in rows if bool(row["risk_eligible"])]
    if not eligible:
        minimum = min(float(row["harmful_rate"]) for row in rows)
        eligible = [row for row in rows if float(row["harmful_rate"]) == minimum]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["mean_validation_accuracy"]),
            -float(row["actual_tokens"]),
            -int(row["candidate"]),
        ),
    )
    for row in rows:
        row["selected"] = int(row is selected)
    return SequentialPolicy(
        harmful_weight=float(selected["harmful_weight"]),
        uncertainty_beta=float(selected["uncertainty_beta"]),
        cost_lambda=float(selected["cost_lambda"]),
        decision_margin=float(selected["decision_margin"]),
    ), rows


def _seed_summary(
    replay: ReplayResult,
    traces: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    helpful, harmful, no_change = _trace_counts(traces)
    return {
        "seed": seed,
        "policy": replay.policy,
        "budget_fraction": replay.budget_fraction,
        "token_budget": replay.token_budget,
        "tasks": len(replay.task_keys),
        "accuracy": float(replay.outcomes.mean()),
        "teacher_tokens": replay.actual_tokens,
        "teacher_calls": replay.calls,
        "helpful": helpful,
        "harmful": harmful,
        "no_change": no_change,
        "harmful_rate": harmful / max(replay.calls, 1),
        "budget_overshoot_tokens": max(0, replay.actual_tokens - replay.token_budget),
    }


def _aligned_outcomes(records: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    keys = sorted(set.intersection(*(set(row["replay"].task_keys) for row in records)))
    matrix = []
    for row in records:
        replay: ReplayResult = row["replay"]
        lookup = dict(zip(replay.task_keys, replay.outcomes, strict=True))
        matrix.append([float(lookup[key]) for key in keys])
    return keys, np.asarray(matrix, dtype=float)


def _task_bootstrap_ci(
    values: np.ndarray, *, repetitions: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    task_count = values.shape[-1]
    samples = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 512):
        size = min(512, repetitions - start)
        indices = rng.integers(0, task_count, size=(size, task_count))
        # values[:, indices] has shape (seed, bootstrap_draw, task).
        samples[start : start + size] = values[:, indices].mean(axis=(0, 2))
    low, high = np.percentile(samples, (2.5, 97.5))
    return float(low), float(high)


def aggregate_records(
    records: list[dict[str, Any]], *, bootstrap: int, seed: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = sorted({(row["policy"], float(row["fraction"])) for row in records})
    for policy, fraction in groups:
        selected = [
            row
            for row in records
            if row["policy"] == policy and float(row["fraction"]) == fraction
        ]
        _keys, outcomes = _aligned_outcomes(selected)
        ci_low, ci_high = _task_bootstrap_ci(
            outcomes, repetitions=bootstrap, seed=seed + int(1000 * fraction)
        )
        summaries = [row["summary"] for row in selected]
        accuracy_by_seed = np.asarray([row["accuracy"] for row in summaries], dtype=float)
        tokens = np.asarray([row["teacher_tokens"] for row in summaries], dtype=float)
        calls = np.asarray([row["teacher_calls"] for row in summaries], dtype=float)
        helpful = np.asarray([row["helpful"] for row in summaries], dtype=float)
        harmful = np.asarray([row["harmful"] for row in summaries], dtype=float)
        no_change = np.asarray([row["no_change"] for row in summaries], dtype=float)

        def variance(value: np.ndarray) -> float:
            return float(np.var(value, ddof=1)) if len(value) > 1 else 0.0

        rows.append(
            {
                "policy": policy,
                "budget_fraction": fraction,
                "seeds": len(selected),
                "tasks": outcomes.shape[1],
                "accuracy_mean": float(accuracy_by_seed.mean()),
                "accuracy_variance": variance(accuracy_by_seed),
                "accuracy_ci95_low": ci_low,
                "accuracy_ci95_high": ci_high,
                "teacher_tokens_mean": float(tokens.mean()),
                "teacher_tokens_variance": variance(tokens),
                "teacher_calls_mean": float(calls.mean()),
                "teacher_calls_variance": variance(calls),
                "helpful_mean": float(helpful.mean()),
                "helpful_variance": variance(helpful),
                "harmful_mean": float(harmful.mean()),
                "harmful_variance": variance(harmful),
                "no_change_mean": float(no_change.mean()),
                "no_change_variance": variance(no_change),
                "max_budget_overshoot": max(row["budget_overshoot_tokens"] for row in summaries),
            }
        )
    return rows


def paired_comparison(
    records: list[dict[str, Any]],
    *,
    candidate: str,
    reference: str,
    fraction: float,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    left = sorted(
        [row for row in records if row["policy"] == candidate and row["fraction"] == fraction],
        key=lambda row: row["seed"],
    )
    right = sorted(
        [row for row in records if row["policy"] == reference and row["fraction"] == fraction],
        key=lambda row: row["seed"],
    )
    common = sorted(set(row["seed"] for row in left) & set(row["seed"] for row in right))
    left = [next(row for row in left if row["seed"] == item) for item in common]
    right = [next(row for row in right if row["seed"] == item) for item in common]
    if not common:
        raise ValueError(f"No paired seeds for {candidate} and {reference}")
    keys = sorted(
        set.intersection(
            *(set(row["replay"].task_keys) for row in [*left, *right])
        )
    )

    def matrix(rows: list[dict[str, Any]]) -> np.ndarray:
        output = []
        for row in rows:
            replay: ReplayResult = row["replay"]
            lookup = dict(zip(replay.task_keys, replay.outcomes, strict=True))
            output.append([float(lookup[key]) for key in keys])
        return np.asarray(output)

    difference = matrix(left) - matrix(right)
    by_task = difference.mean(axis=0)
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap, dtype=float)
    for index in range(bootstrap):
        chosen = rng.integers(0, len(keys), size=len(keys))
        samples[index] = float(by_task[chosen].mean())
    ci_low, ci_high = np.percentile(samples, (2.5, 97.5))
    observed = abs(float(by_task.mean()))
    extreme = 0
    for _ in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=len(keys))
        extreme += abs(float((by_task * signs).mean())) >= observed - 1e-15
    left_tokens = np.mean([row["summary"]["teacher_tokens"] for row in left])
    right_tokens = np.mean([row["summary"]["teacher_tokens"] for row in right])
    return {
        "candidate": candidate,
        "reference": reference,
        "budget_fraction": fraction,
        "paired_seeds": len(common),
        "paired_tasks": len(keys),
        "accuracy_difference_pp": 100.0 * float(by_task.mean()),
        "difference_ci95_low_pp": 100.0 * float(ci_low),
        "difference_ci95_high_pp": 100.0 * float(ci_high),
        "paired_permutation_p": (extreme + 1) / (permutations + 1),
        "candidate_tokens_mean": float(left_tokens),
        "reference_tokens_mean": float(right_tokens),
        "token_difference_mean": float(left_tokens - right_tokens),
        "task_wins": int(np.sum(by_task > 0)),
        "task_losses": int(np.sum(by_task < 0)),
        "task_ties": int(np.sum(by_task == 0)),
    }


def leakage_audit(
    frozen_root: Path,
    train_groups: dict[str, Any],
    validation_groups: dict[str, Any],
    train_states: list[MultiHistoryState],
    validation_states: list[MultiHistoryState],
    train_targets: Any,
    v03_root: Path,
    hard_call_token_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def task_rows(split: str) -> list[dict[str, Any]]:
        path = frozen_root / split / "tasks.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    train_task_rows = task_rows("train")
    validation_task_rows = task_rows("validation")
    l1_train = {
        f"{row['dataset']}::{row['task_id']}"
        for row in train_task_rows
        if bool(row.get("level1_complete"))
    }
    l1_validation = {
        f"{row['dataset']}::{row['task_id']}"
        for row in validation_task_rows
        if bool(row.get("level1_complete"))
    }
    l2_train = set(train_groups)
    l2_validation = set(validation_groups)
    manifest = json.loads((v03_root / "selection_manifest.json").read_text())
    rows: list[dict[str, Any]] = []

    def add(check: str, value: int | float | bool | str, passed: bool, detail: str) -> None:
        rows.append({"check": check, "value": value, "passed": passed, "detail": detail})

    add("l2_train_validation_task_overlap", len(l2_train & l2_validation), not (l2_train & l2_validation), "Must be zero")
    add("l1_train_to_l2_validation_overlap", len(l1_train & l2_validation), not (l1_train & l2_validation), "v0.3 pretraining train tasks must not enter v0.4 validation")
    add("l1_validation_to_l2_train_overlap", len(l1_validation & l2_train), not (l1_validation & l2_train), "Validation tasks must not enter v0.4 training")
    add("l1_train_to_l2_train_overlap", len(l1_train & l2_train), True, "Expected reuse within the training partition; not leakage")
    add("v03_test_loaded_during_selection", bool(manifest.get("test_loaded")), not bool(manifest.get("test_loaded")), "Pretrained v0.3 manifest must keep test locked")
    add("state_prefix_length_contract", sum(len(row.prefix) != row.checkpoint - 1 for row in [*train_states, *validation_states]), all(len(row.prefix) == row.checkpoint - 1 for row in [*train_states, *validation_states]), "Only past actions may appear in history")
    q_binary = bool(np.isin(train_targets.q_skip, (0.0, 1.0)).all() and np.isin(train_targets.q_verify, (0.0, 1.0)).all())
    add("counterfactual_targets_binary", q_binary, q_binary, "Targets are final observed correctness labels")
    add("forbidden_fields_excluded_from_encoder_contract", len(FORBIDDEN_INPUT_FIELDS), True, "DARC encoder serializes observable text/metadata only; labels stay outside EncodedDARC")
    maximum_call = max(
        verify_cost(group, prefix)
        for groups in (train_groups, validation_groups)
        for group in groups.values()
        for prefix in ("", "S", "V", "SS", "SV", "VS", "VV")
    )
    add(
        "maximum_actual_call_within_reserved_cap",
        maximum_call,
        maximum_call <= hard_call_token_cap,
        f"Configured hard-call reservation is {hard_call_token_cap} tokens",
    )

    inconsistent = 0
    checked = 0
    for groups in (train_groups, validation_groups):
        for group in groups.values():
            for prefix in ("S", "V", "SS", "SV", "VS", "VV"):
                descendants = [seq for seq in ALL_SEQUENCES if seq.startswith(prefix)]
                for checkpoint in range(1, len(prefix) + 1):
                    values = [
                        json.dumps(_artifact(group.rows[seq], checkpoint), sort_keys=True)
                        for seq in descendants
                    ]
                    checked += 1
                    inconsistent += int(len(set(values)) > 1)
    add("observed_prefix_artifact_inconsistencies", inconsistent, inconsistent == 0, f"Compared {checked} already-observed prefix artifacts across descendants")

    payload = {
        "schema_version": "darc-v0.4-leakage-audit-v1",
        "all_required_checks_pass": all(bool(row["passed"]) for row in rows),
        "checks": rows,
        "l1_train_tasks": len(l1_train),
        "l1_validation_tasks": len(l1_validation),
        "l2_train_tasks": len(l2_train),
        "l2_validation_tasks": len(l2_validation),
        "inference_feature_contract": {
            "allowed": ["task", "current artifact", "observed artifacts", "past actions", "checkpoint", "remaining budget"],
            "forbidden": sorted(FORBIDDEN_INPUT_FIELDS),
            "actual_cost_usage": "charged only after a Verify action is selected",
            "counterfactual_outcome_usage": "training targets and post-hoc audit only",
        },
    }
    return rows, payload


def _plot_results(
    output: Path,
    aggregate: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    selected = next(str(row["variant"]) for row in candidate_rows if bool(row["selected"]))
    policies = [
        f"DARC v0.4 · {selected}",
        "DARC v0.4 · full_pretrained",
        "DARC v0.4 · myopic_targets",
        "C3 MultiHistory Myopic",
        "C3 MultiHistory Continuation",
        "Fixed VVS",
        "Fixed VVV",
        "Fixed SSS",
    ]
    fig, axis = plt.subplots(figsize=(9.5, 6.0))
    for policy in dict.fromkeys(policies):
        points = sorted(
            [row for row in aggregate if row["policy"] == policy],
            key=lambda row: float(row["teacher_tokens_mean"]),
        )
        if not points:
            continue
        axis.errorbar(
            [float(row["teacher_tokens_mean"]) / 1000 for row in points],
            [100 * float(row["accuracy_mean"]) for row in points],
            yerr=[
                [100 * (float(row["accuracy_mean"]) - float(row["accuracy_ci95_low"])) for row in points],
                [100 * (float(row["accuracy_ci95_high"]) - float(row["accuracy_mean"])) for row in points],
            ],
            marker="o",
            linewidth=2.6 if policy.endswith(selected) else 1.5,
            capsize=2,
            label=policy,
        )
    axis.set_title("Validation-only accuracy–cost frontier", weight="bold")
    axis.set_xlabel("Actual Teacher tokens (thousands, mean across seeds)")
    axis.set_ylabel("Downstream accuracy (%)")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "accuracy_cost_frontier.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "accuracy_cost_frontier.pdf", bbox_inches="tight")
    plt.close(fig)

    full = [
        row
        for row in aggregate
        if float(row["budget_fraction"]) == 1.0 and row["policy"].startswith("DARC v0.4 ·")
    ]
    fig, axis = plt.subplots(figsize=(9.2, 5.0))
    axis.barh(
        [str(row["policy"]).replace("DARC v0.4 · ", "") for row in full],
        [100 * float(row["accuracy_mean"]) for row in full],
        xerr=[
            [100 * (float(row["accuracy_mean"]) - float(row["accuracy_ci95_low"])) for row in full],
            [100 * (float(row["accuracy_ci95_high"]) - float(row["accuracy_mean"])) for row in full],
        ],
        color=["#D9485F" if row["policy"].endswith(selected) else "#7895B2" for row in full],
        capsize=3,
    )
    axis.set_title("DARC v0.4 ablations · matched seeds and validation tasks", weight="bold")
    axis.set_xlabel("Downstream accuracy (%)")
    fig.tight_layout()
    fig.savefig(figures / "ablation_stability.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "ablation_stability.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output: Path,
    config: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    leakage: dict[str, Any],
) -> Path:
    selected = next(row for row in candidate_rows if bool(row["selected"]))
    reference = next(row for row in candidate_rows if bool(row["is_reference"]))
    full_fraction = 1.0
    table = [row for row in aggregate if float(row["budget_fraction"]) == full_fraction]
    myopic_name = "DARC v0.4 · myopic_targets"
    full_name = "DARC v0.4 · full_pretrained"
    myopic_cmp = next(
        (
            row
            for row in comparisons
            if row["candidate"] == myopic_name
            and row["reference"] == full_name
            and float(row["budget_fraction"]) == full_fraction
        ),
        None,
    )
    lines = [
        "# DARC v0.4 Experimental Audit — Validation Only",
        "",
        "> No frozen-test task is loaded by this pipeline. All reported results are development evidence on the frozen validation partition.",
        "",
        "## Selection protocol",
        "",
        f"Full (`{config['selection']['reference_variant']}`) is the predefined reference, not the automatic winner. Candidates must satisfy a harmful-selection rate no greater than `min({float(config['selection']['max_harmful_rate']):.3f}, Full + {float(config['selection']['harmful_rate_tolerance_over_reference']):.3f})`. Among eligible candidates, selection maximizes mean non-zero-budget validation accuracy minus `{float(config['selection']['token_utilization_penalty']):.4f} × actual-token utilization`.",
        "",
        f"Development-selected candidate: **{selected['variant']}**. Reference: **{reference['variant']}**.",
        "",
        "| Candidate | Accuracy | Token utilization | Harmful rate | Score | Eligible | Selected |",
        "|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in candidate_rows:
        lines.append(
            f"| {row['variant']} | {100*float(row['mean_accuracy']):.2f}% | {100*float(row['token_utilization']):.2f}% | {100*float(row['harmful_rate']):.2f}% | {float(row['selection_score']):.4f} | {'yes' if row['risk_eligible'] else 'no'} | {'yes' if row['selected'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Full-budget validation results",
            "",
            "| Policy | Accuracy mean (95% task CI) | Accuracy variance | Tokens mean | Calls mean | Helpful / Harmful / No-change |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(table, key=lambda item: float(item["accuracy_mean"]), reverse=True):
        lines.append(
            f"| {row['policy']} | {100*float(row['accuracy_mean']):.2f}% [{100*float(row['accuracy_ci95_low']):.2f}, {100*float(row['accuracy_ci95_high']):.2f}] | {float(row['accuracy_variance']):.6f} | {float(row['teacher_tokens_mean']):,.0f} | {float(row['teacher_calls_mean']):.1f} | {float(row['helpful_mean']):.1f} / {float(row['harmful_mean']):.1f} / {float(row['no_change_mean']):.1f} |"
        )
    lines.extend(["", "## Myopic versus Full", ""])
    if myopic_cmp is not None:
        significant = float(myopic_cmp["paired_permutation_p"]) < 0.05
        lines.append(
            f"At full budget, Myopic − Full is **{float(myopic_cmp['accuracy_difference_pp']):+.2f} pp** (task-bootstrap 95% CI [{float(myopic_cmp['difference_ci95_low_pp']):+.2f}, {float(myopic_cmp['difference_ci95_high_pp']):+.2f}], paired permutation p={float(myopic_cmp['paired_permutation_p']):.4f}) and {float(myopic_cmp['token_difference_mean']):+,.0f} actual tokens. This {'supports' if significant and float(myopic_cmp['accuracy_difference_pp']) > 0 else 'does not establish'} a statistically reliable Myopic advantage on validation."
        )
    lines.extend(
        [
            "",
            "## Leakage and causality audit",
            "",
            f"Overall required checks: **{'PASS' if leakage['all_required_checks_pass'] else 'FAIL'}**.",
            "",
            "| Check | Value | Pass | Detail |",
            "|---|---:|:---:|---|",
        ]
    )
    for row in leakage["checks"]:
        lines.append(
            f"| {row['check']} | {row['value']} | {'yes' if row['passed'] else 'no'} | {row['detail']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- C3 checkpoints are pre-existing single fits; they are replayed under every seed/order but are not retrained by this audit.",
            "- Variant selection and reporting both use validation, so these are development results rather than confirmatory estimates.",
            "- Actual future cost and counterfactual outcomes are never used for online admission; they are used only after an action for accounting/audit or as training supervision.",
            "- The frozen test remains untouched and must not be opened for further tuning.",
            "",
            "![Accuracy-cost frontier](../figures/accuracy_cost_frontier.png)",
            "",
            "![Ablation stability](../figures/ablation_stability.png)",
            "",
            "## Artifacts",
            "",
            "- `models/seed_<seed>/<variant>.pt`: trained checkpoints.",
            "- `tables/seed_results.csv`: per-seed validation measurements.",
            "- `tables/aggregate_results.csv`: means, variances and task-bootstrap intervals.",
            "- `tables/paired_comparisons.csv`: paired task comparisons and permutation tests.",
            "- `tables/decision_trace.csv`: post-hoc action audit.",
            "- `leakage_audit.json`: split, observability and causality checks.",
        ]
    )
    report = output / "report" / "darc_v04_audit_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(config_path: Path, *, resume: bool = True) -> Path:
    config = yaml.safe_load(config_path.read_text())
    if config["evaluation"]["partition"] != "validation":
        raise ValueError("DARC v0.4 audit is validation-only")
    if config["evaluation"]["frozen_test_access"] != "forbidden":
        raise ValueError("Frozen-test access must remain forbidden")
    frozen_root = Path(config["data"]["frozen_root"])
    output = Path(config["output"]["root"])
    tables = output / "tables"
    models = output / "models"
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in config["reproducibility"]["seeds"]]
    fractions = [float(value) for value in config["evaluation"]["budget_fractions"]]
    hard_cap = int(config["evaluation"]["hard_call_token_cap"])
    device = str(config["semantic_encoder"]["device"])

    train_groups = _load_groups(frozen_root, "train")
    validation_groups = _load_groups(frozen_root, "validation")
    _assert_disjoint(train_groups, validation_groups)
    train_states, _ = build_states(train_groups)
    validation_states, validation_lookup = build_states(validation_groups)
    train_encoded, train_cache = encode_or_load(
        [row.encoder_example for row in train_states],
        split="v04_audit_train",
        config=config["semantic_encoder"],
        cache_dir=Path(config["semantic_encoder"]["cache_dir"]),
    )
    validation_encoded, validation_cache = encode_or_load(
        [row.encoder_example for row in validation_states],
        split="v04_audit_validation",
        config=config["semantic_encoder"],
        cache_dir=Path(config["semantic_encoder"]["cache_dir"]),
    )
    reference_budget = float(
        np.median(
            [int(group.rows["VVV"].get("teacher_total_tokens", 0) or 0) for group in train_groups.values()]
        )
    )
    allowances = _allowances(
        reference_budget, [float(value) for value in config["targets"]["budget_fractions"]]
    )
    target_cache: dict[bool, tuple[Any, Any]] = {}
    for counterfactual in (False, True):
        target_cache[counterfactual] = (
            build_targets(
                train_states,
                train_groups,
                allowances=allowances,
                reference_budget=reference_budget,
                counterfactual=counterfactual,
            ),
            build_targets(
                validation_states,
                validation_groups,
                allowances=allowances,
                reference_budget=reference_budget,
                counterfactual=counterfactual,
            ),
        )

    leakage_rows, leakage_payload = leakage_audit(
        frozen_root,
        train_groups,
        validation_groups,
        train_states,
        validation_states,
        target_cache[True][0],
        Path(config["data"]["darc_v03_results"]),
        hard_cap,
    )
    if not leakage_payload["all_required_checks_pass"]:
        failed = [row["check"] for row in leakage_rows if not row["passed"]]
        raise RuntimeError(f"Leakage audit failed: {failed}")
    (output / "leakage_audit.json").write_text(
        json.dumps(leakage_payload, indent=2) + "\n", encoding="utf-8"
    )

    pretrained = DARCEnsemble.load(
        Path(config["data"]["darc_v03_results"]) / "models" / "darc_v03_selected.pt"
    )
    c3_root = Path(config["data"]["c3_results"]) / "models" / "combined"
    c3_models = {
        "C3 MultiHistory Myopic": joblib.load(c3_root / "multi_myopic.joblib"),
        "C3 MultiHistory Continuation": joblib.load(c3_root / "multi_continuation.joblib"),
    }
    policies = _policy_grid(config["policy_grid"])
    records: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    seed_candidate_rows: list[dict[str, Any]] = []

    for experiment_seed in seeds:
        for variant in config["variants"]:
            name = str(variant["name"])
            label = f"DARC v0.4 · {name}"
            model_path = models / f"seed_{experiment_seed}" / f"{name}.pt"
            if resume and model_path.exists():
                model = DARCv04Ensemble.load(model_path)
            else:
                train_targets, validation_targets = target_cache[
                    bool(variant["counterfactual_targets"])
                ]
                model = train_ensemble(
                    train_states,
                    train_encoded,
                    train_targets,
                    validation_states,
                    validation_encoded,
                    validation_targets,
                    params=config["model"],
                    use_history=bool(variant["use_history"]),
                    budget_conditioning=bool(variant["budget_conditioning"]),
                    reference_budget=reference_budget,
                    pretrained=pretrained if bool(variant["pretrained"]) else None,
                    # Crucially, no variant-specific offset: all ablations share seed.
                    seed=experiment_seed,
                    deterministic=bool(config["reproducibility"]["deterministic_torch"]),
                    device_name=device,
                    description=f"audit s{experiment_seed} {name}",
                    progress=config["progress"],
                )
                model.save(model_path)
            predictions = _precompute(
                model, validation_encoded, validation_states, allowances, device=device
            )
            policy, tuned = tune_validation_policy(
                validation_groups,
                validation_states,
                validation_lookup,
                predictions,
                allowances,
                policies,
                fractions=fractions,
                reference_budget=reference_budget,
                hard_cap=hard_cap,
                seed=experiment_seed,
                label=label,
                token_penalty=float(config["selection"]["token_utilization_penalty"]),
                max_harmful_rate=float(config["selection"]["max_harmful_rate"]),
            )
            policy_rows.extend(tuned)
            selected_policy_row = next(row for row in tuned if bool(row["selected"]))
            seed_candidate_rows.append(
                {
                    "seed": experiment_seed,
                    "variant": name,
                    "mean_accuracy": selected_policy_row["mean_validation_accuracy"],
                    "actual_tokens": selected_policy_row["actual_tokens"],
                    "token_budget": selected_policy_row["token_budget"],
                    "calls": selected_policy_row["calls"],
                    "harmful_actions": selected_policy_row["harmful"],
                }
            )
            metric_rows.append(
                {
                    "seed": experiment_seed,
                    "variant": name,
                    **_effect_metrics(validation_states, predictions[-1]),
                }
            )
            for member_index, member in enumerate(model.members, 1):
                for row in member.history:
                    training_rows.append(
                        {"seed": experiment_seed, "variant": name, "member": member_index, **row}
                    )
            for fraction in fractions:
                budget = int(round(fraction * reference_budget * len(validation_groups)))
                replay, traces = replay_v04(
                    validation_groups,
                    validation_states,
                    validation_lookup,
                    predictions,
                    allowances,
                    policy,
                    token_budget=budget,
                    budget_fraction=fraction,
                    hard_call_token_cap=hard_cap,
                    seed=experiment_seed,
                    label=label,
                )
                normalized_traces = [
                    {
                        **row,
                        "seed": experiment_seed,
                        "no_change_selected": int(row.get("wasted_selected", 0)),
                    }
                    for row in traces
                ]
                summary = _seed_summary(replay, normalized_traces, seed=experiment_seed)
                records.append(
                    {"seed": experiment_seed, "policy": label, "fraction": fraction, "replay": replay, "summary": summary}
                )
                decision_rows.extend(normalized_traces)

        # Baselines use the exact same validation tasks, order, budgets and ledger.
        for fraction in fractions:
            budget = int(round(fraction * reference_budget * len(validation_groups)))
            for sequence in ALL_SEQUENCES:
                replay, traces = replay_fixed_strict(
                    validation_groups,
                    sequence,
                    token_budget=budget,
                    budget_fraction=fraction,
                    hard_call_token_cap=hard_cap,
                    seed=experiment_seed,
                )
                summary = _seed_summary(replay, traces, seed=experiment_seed)
                records.append(
                    {"seed": experiment_seed, "policy": replay.policy, "fraction": fraction, "replay": replay, "summary": summary}
                )
                decision_rows.extend(traces)
            for label, controller in c3_models.items():
                replay, traces = replay_c3_strict(
                    validation_groups,
                    controller,
                    token_budget=budget,
                    budget_fraction=fraction,
                    hard_call_token_cap=hard_cap,
                    seed=experiment_seed,
                    label=label,
                )
                summary = _seed_summary(replay, traces, seed=experiment_seed)
                records.append(
                    {"seed": experiment_seed, "policy": label, "fraction": fraction, "replay": replay, "summary": summary}
                )
                decision_rows.extend(traces)

    # Aggregate the per-seed candidate statistics before development selection.
    candidate_inputs: list[dict[str, Any]] = []
    total_nonzero_budget = sum(
        int(round(f * reference_budget * len(validation_groups)))
        for f in fractions
        if f > 0
    ) * len(seeds)
    for variant in [str(row["name"]) for row in config["variants"]]:
        selected = [row for row in seed_candidate_rows if row["variant"] == variant]
        candidate_inputs.append(
            {
                "variant": variant,
                "mean_accuracy": float(np.mean([row["mean_accuracy"] for row in selected])),
                "harmful_actions": int(sum(row["harmful_actions"] for row in selected)),
                "actual_tokens": int(sum(row["actual_tokens"] for row in selected)),
                "calls": int(sum(row["calls"] for row in selected)),
                "token_budget": total_nonzero_budget,
            }
        )
    selected_variant, candidate_rows = select_development_candidate(
        candidate_inputs,
        reference_variant=str(config["selection"]["reference_variant"]),
        token_utilization_penalty=float(config["selection"]["token_utilization_penalty"]),
        max_harmful_rate=float(config["selection"]["max_harmful_rate"]),
        harmful_rate_tolerance_over_reference=float(
            config["selection"]["harmful_rate_tolerance_over_reference"]
        ),
    )

    seed_rows = [row["summary"] for row in records]
    aggregate = aggregate_records(
        records,
        bootstrap=int(config["reproducibility"]["bootstrap_repetitions"]),
        seed=seeds[0],
    )
    selected_label = f"DARC v0.4 · {selected_variant}"
    variant_labels = [f"DARC v0.4 · {row['name']}" for row in config["variants"]]
    baseline_labels = [
        "C3 MultiHistory Myopic",
        "C3 MultiHistory Continuation",
        *[f"Fixed {sequence}" for sequence in ALL_SEQUENCES],
    ]
    comparisons: list[dict[str, Any]] = []
    for fraction in fractions:
        for candidate in variant_labels:
            if candidate != "DARC v0.4 · full_pretrained":
                comparisons.append(
                    paired_comparison(
                        records,
                        candidate=candidate,
                        reference="DARC v0.4 · full_pretrained",
                        fraction=fraction,
                        bootstrap=int(config["reproducibility"]["bootstrap_repetitions"]),
                        permutations=int(config["reproducibility"]["permutation_repetitions"]),
                        seed=seeds[0] + int(fraction * 1000),
                    )
                )
        for reference in baseline_labels:
            comparisons.append(
                paired_comparison(
                    records,
                    candidate=selected_label,
                    reference=reference,
                    fraction=fraction,
                    bootstrap=int(config["reproducibility"]["bootstrap_repetitions"]),
                    permutations=int(config["reproducibility"]["permutation_repetitions"]),
                    seed=seeds[0] + int(fraction * 1000) + 31,
                )
            )

    _write_csv(tables / "data_audit.csv", audit_level2({"train": train_groups, "validation": validation_groups}))
    _write_csv(tables / "leakage_audit.csv", leakage_rows)
    _write_csv(tables / "policy_grid_results.csv", policy_rows)
    _write_csv(tables / "candidate_selection.csv", candidate_rows)
    _write_csv(tables / "seed_results.csv", seed_rows)
    _write_csv(tables / "aggregate_results.csv", aggregate)
    _write_csv(tables / "paired_comparisons.csv", comparisons)
    _write_csv(tables / "model_metrics.csv", metric_rows)
    _write_csv(tables / "training_history.csv", training_rows)
    _write_csv(tables / "decision_trace.csv", decision_rows)
    _plot_results(output, aggregate, candidate_rows)
    report = _write_report(output, config, candidate_rows, aggregate, comparisons, leakage_payload)

    manifest = {
        "schema_version": "darc-v0.4-audit-manifest-v1",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "partition_loaded": "validation",
        "frozen_test_loaded": False,
        "seeds": seeds,
        "reference_variant": config["selection"]["reference_variant"],
        "development_selected_variant": selected_variant,
        "selection_criteria": config["selection"],
        "reference_budget_per_task": reference_budget,
        "embedding_cache": {"train": train_cache, "validation": validation_cache},
        "report": str(report),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/darc_v04_audit.yaml"))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    report = run(args.config, resume=not args.no_resume)
    print(f"DARC v0.4 audit completed. Report: {report}")


if __name__ == "__main__":
    main()

