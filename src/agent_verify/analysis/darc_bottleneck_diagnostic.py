"""Offline bottleneck diagnostic for DARC v0.5 Myopic iteration 0.

Only frozen train and validation Level-2 data are loaded.  Training labels are
used for grouped OOF learnability experiments; validation outcomes are report-
only.  The frozen test partition is never opened.  Local counterfactual effects
and final sequential-policy outcomes are stored in separate tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from agent_verify.analysis.cbv_v01_experiment import _passed
from agent_verify.analysis.darc_data import EncodedDARC, encode_or_load
from agent_verify.analysis.darc_v04_data import MultiHistoryState, build_states, verify_cost
from agent_verify.analysis.darc_v04_experiment import _assert_disjoint, _load_groups, _write_csv
from agent_verify.analysis.darc_v05_experiment import _precompute_v05
from agent_verify.analysis.darc_v05_model import DARCv05Ensemble, V05Prediction
from agent_verify.analysis.darc_v05_synergy_audit import (
    MULTI_SEQUENCES,
    SEQUENCES,
    SINGLE_SEQUENCES,
    strict_synergy,
)
from agent_verify.analysis.darc_v05_training import assign_task_folds
from agent_verify.analysis.darc_v053_safe_policy import oof_predictions


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tokens(row: dict[str, Any]) -> int:
    return int(row.get("teacher_total_tokens", 0) or 0)


def _features(encoded: EncodedDARC) -> np.ndarray:
    downstream = np.zeros((len(encoded.state_keys), encoded.downstream.shape[-1]), dtype=np.float32)
    for index in range(len(encoded.state_keys)):
        mask = encoded.downstream_mask[index]
        if mask.any():
            downstream[index] = encoded.downstream[index][mask].mean(axis=0)
    checkpoint = np.eye(3, dtype=np.float32)[encoded.checkpoint - 1]
    return np.concatenate(
        (
            encoded.task,
            encoded.artifact,
            encoded.history,
            encoded.current_role,
            downstream,
            encoded.metadata,
            checkpoint,
        ),
        axis=1,
    )


def _labels(states: list[MultiHistoryState]) -> np.ndarray:
    return np.asarray([int(state.encoder_example.value_label == 2) for state in states])


def _constant_or_probability(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        if probabilities.shape[1] == 1:
            label = int(model.classes_[0])
            return np.full(len(x), float(label))
        positive = int(np.where(np.asarray(model.classes_) == 1)[0][0])
        return probabilities[:, positive]
    return np.asarray(model.predict(x), dtype=float)


def _fit_logistic(x: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> tuple[Any, Any]:
    scaler = StandardScaler().fit(x)
    if len(np.unique(y)) < 2:
        return scaler, int(y[0])
    model = LogisticRegression(
        C=float(config["logistic_c"]),
        max_iter=int(config["logistic_max_iter"]),
        class_weight="balanced",
        solver="liblinear",
        random_state=0,
    ).fit(scaler.transform(x), y)
    return scaler, model


def _logistic_predict(bundle: tuple[Any, Any], x: np.ndarray) -> np.ndarray:
    scaler, model = bundle
    if isinstance(model, int):
        return np.full(len(x), float(model))
    return _constant_or_probability(model, scaler.transform(x))


def _fit_lightgbm(x: np.ndarray, y: np.ndarray, config: dict[str, Any], seed: int) -> Any:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise RuntimeError(
            "LightGBM is required for this diagnostic. Install the project dependencies first."
        ) from error
    if len(np.unique(y)) < 2:
        return int(y[0])
    params = dict(config["lightgbm"])
    return LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        verbosity=-1,
        random_state=seed,
        **params,
    ).fit(x, y)


def _lightgbm_predict(model: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(model, int):
        return np.full(len(x), float(model))
    probabilities = model.predict_proba(x, validate_features=False)
    positive = int(np.where(np.asarray(model.classes_) == 1)[0][0])
    return probabilities[:, positive]


def grouped_oof_baselines(
    train_encoded: EncodedDARC,
    train_states: list[MultiHistoryState],
    validation_encoded: EncodedDARC,
    *,
    folds: dict[str, int],
    fold_count: int,
    config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_x = _features(train_encoded)
    validation_x = _features(validation_encoded)
    y = _labels(train_states)
    oof = {"Logistic Regression": np.zeros(len(y)), "LightGBM": np.zeros(len(y))}
    jobs = tqdm(range(fold_count), desc="Helpful baselines train-OOF", unit="fold", dynamic_ncols=True)
    for fold in jobs:
        heldout = np.asarray([folds[state.task_key] == fold for state in train_states])
        fit = ~heldout
        logistic = _fit_logistic(train_x[fit], y[fit], config)
        lightgbm = _fit_lightgbm(train_x[fit], y[fit], config, seed + fold)
        oof["Logistic Regression"][heldout] = _logistic_predict(logistic, train_x[heldout])
        oof["LightGBM"][heldout] = _lightgbm_predict(lightgbm, train_x[heldout])
        jobs.set_postfix(fold=fold, heldout=int(heldout.sum()), helpful=int(y[fit].sum()))
    logistic = _fit_logistic(train_x, y, config)
    lightgbm = _fit_lightgbm(train_x, y, config, seed)
    validation = {
        "Logistic Regression": _logistic_predict(logistic, validation_x),
        "LightGBM": _lightgbm_predict(lightgbm, validation_x),
    }
    return oof, validation


def learnability_metrics(
    scores: dict[str, np.ndarray],
    states: list[MultiHistoryState],
    *,
    split: str,
    top_k: list[int],
) -> list[dict[str, Any]]:
    labels = _labels(states)
    rows: list[dict[str, Any]] = []
    for model, values in scores.items():
        for checkpoint in (0, 1, 2, 3):
            mask = (
                np.ones(len(states), dtype=bool)
                if checkpoint == 0
                else np.asarray([state.checkpoint == checkpoint for state in states])
            )
            y = labels[mask]
            score = values[mask]
            positives = int(y.sum())
            row: dict[str, Any] = {
                "split": split,
                "model": model,
                "checkpoint": "all" if checkpoint == 0 else f"CP{checkpoint}",
                "rows": len(y),
                "helpful": positives,
                "prevalence": positives / max(len(y), 1),
                "auprc": float(average_precision_score(y, score)) if positives else math.nan,
                "auroc": (float(roc_auc_score(y, score)) if 0 < positives < len(y) else math.nan),
            }
            order = np.argsort(-score)
            for k in top_k:
                chosen = order[: min(k, len(order))]
                found = int(y[chosen].sum())
                row[f"precision_at_{k}"] = found / max(len(chosen), 1)
                row[f"recall_at_{k}"] = found / max(positives, 1)
            rows.append(row)
    return rows


def trajectory_feasible(group: Any, sequence: str, hard_cap: int) -> bool:
    prefix = ""
    for action in sequence:
        if action == "V" and verify_cost(group, prefix) > hard_cap:
            return False
        prefix += action
    return True


def oracle_task_rows(split: str, groups: dict[str, Any], *, hard_cap: int) -> list[dict[str, Any]]:
    rows = []
    for task_key, group in sorted(groups.items()):
        candidates = [
            sequence
            for sequence in SEQUENCES
            if _passed(group.rows[sequence]) and trajectory_feasible(group, sequence, hard_cap)
        ]
        cheapest = min(
            candidates,
            key=lambda sequence: (_tokens(group.rows[sequence]), sequence.count("V"), sequence),
            default=None,
        )
        baseline = int(_passed(group.rows["SSS"]))
        rows.append(
            {
                "split": split,
                "task_key": task_key,
                "dataset": group.task.dataset,
                "baseline_correct": baseline,
                "any_observed_success": int(bool(candidates)),
                "cheapest_success_sequence": cheapest or "",
                "cheapest_success_tokens": _tokens(group.rows[cheapest]) if cheapest else "",
                "rescue_cost": (
                    _tokens(group.rows[cheapest]) if cheapest and not baseline else 0 if baseline else ""
                ),
                "strict_synergy_available": int(
                    any(strict_synergy(group, sequence) for sequence in MULTI_SEQUENCES)
                ),
            }
        )
    return rows


def oracle_minimum_cost(task_rows: list[dict[str, Any]], target_correct: int) -> tuple[int | None, list[str]]:
    baseline = [row for row in task_rows if int(row["baseline_correct"])]
    required = max(target_correct - len(baseline), 0)
    rescues = sorted(
        [row for row in task_rows if not int(row["baseline_correct"]) and row["rescue_cost"] != ""],
        key=lambda row: (int(row["rescue_cost"]), row["task_key"]),
    )
    if required > len(rescues):
        return None, []
    chosen = rescues[:required]
    return sum(int(row["rescue_cost"]) for row in chosen), [str(row["task_key"]) for row in chosen]


def oracle_comparison_rows(
    validation_tasks: list[dict[str, Any]], source_results: list[dict[str, str]]
) -> list[dict[str, Any]]:
    rows = []
    for source in source_results:
        target = int(round(float(source["accuracy"]) * int(source["tasks"])))
        cost, chosen = oracle_minimum_cost(validation_tasks, target)
        actual = int(source["teacher_tokens"])
        rows.append(
            {
                "seed": int(source["seed"]),
                "budget_fraction": float(source["budget_fraction"]),
                "target_correct": target,
                "target_accuracy": float(source["accuracy"]),
                "policy_actual_tokens": actual,
                "oracle_minimum_tokens": cost if cost is not None else "",
                "avoidable_tokens": actual - cost if cost is not None else "",
                "policy_to_oracle_cost_ratio": actual / cost if cost else math.inf if actual else 1.0,
                "oracle_selected_rescues": len(chosen),
                "oracle_task_keys": ",".join(chosen),
            }
        )
    return rows


def local_effect_rows(
    split: str, states: list[MultiHistoryState], groups: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for state in tqdm(states, desc=f"{split} local effects", unit="state", dynamic_ncols=True):
        example = state.encoder_example
        delta = int(example.delta)
        rows.append(
            {
                "split": split,
                "task_key": state.task_key,
                "dataset": state.dataset,
                "checkpoint": state.checkpoint,
                "history": state.prefix or "START",
                "y_skip": int(example.y_skip),
                "y_verify": int(example.y_verify),
                "local_gain": delta,
                "local_effect": "Helpful" if delta > 0 else "Harmful" if delta < 0 else "No-change",
                "actual_verify_tokens": verify_cost(groups[state.task_key], state.prefix),
            }
        )
    return rows


def checkpoint_summary(
    local_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    splits = sorted({row["split"] for row in local_rows})
    for split in splits:
        for dataset in sorted({row["dataset"] for row in local_rows if row["split"] == split} | {"combined"}):
            for checkpoint in (1, 2, 3):
                selected = [
                    row
                    for row in local_rows
                    if row["split"] == split
                    and row["checkpoint"] == checkpoint
                    and (dataset == "combined" or row["dataset"] == dataset)
                ]
                if not selected:
                    continue
                effects = Counter(row["local_effect"] for row in selected)
                policy = (
                    [
                        row
                        for row in policy_rows
                        if row["checkpoint"] == checkpoint
                        and (dataset == "combined" or row["dataset"] == dataset)
                    ]
                    if split == "validation"
                    else []
                )
                verifies = [row for row in policy if row["action"] == "V"]
                output.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "checkpoint": checkpoint,
                        "local_states": len(selected),
                        "helpful": effects["Helpful"],
                        "harmful": effects["Harmful"],
                        "no_change": effects["No-change"],
                        "helpful_rate": effects["Helpful"] / len(selected),
                        "harmful_rate": effects["Harmful"] / len(selected),
                        "no_change_rate": effects["No-change"] / len(selected),
                        "mean_actual_verify_tokens": float(
                            np.mean([row["actual_verify_tokens"] for row in selected])
                        ),
                        "policy_decisions": len(policy),
                        "policy_verify_calls": len(verifies),
                        "policy_verify_rate": len(verifies) / max(len(policy), 1),
                        "policy_actual_tokens": sum(
                            int(float(row["actual_cost_after_action"])) for row in verifies
                        ),
                    }
                )
    return output


def policy_diagnostic_rows(
    source_rows: list[dict[str, str]], groups: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for source in source_rows:
        delta = int(float(source["direct_actual_delta_audit_only"]))
        action = source["action"]
        row = {
            **source,
            "seed": int(source["seed"]),
            "budget_fraction": float(source["budget_fraction"]),
            "checkpoint": int(source["checkpoint"]),
            "local_gain": delta,
            "false_positive_no_change": int(action == "V" and delta == 0),
            "harmful_call": int(action == "V" and delta < 0),
            "helpful_call": int(action == "V" and delta > 0),
            "missed_helpful": int(action == "S" and delta > 0),
            "wasted_tokens": int(float(source["actual_cost_after_action"]))
            if action == "V" and delta == 0
            else 0,
            "harmful_tokens": int(float(source["actual_cost_after_action"]))
            if action == "V" and delta < 0
            else 0,
        }
        decisions.append(row)
    keys = sorted({(row["seed"], row["budget_fraction"], row["task_key"]) for row in decisions})
    for seed, fraction, task_key in keys:
        chosen = [
            row
            for row in decisions
            if (row["seed"], row["budget_fraction"], row["task_key"]) == (seed, fraction, task_key)
        ]
        chosen.sort(key=lambda row: row["checkpoint"])
        sequence = "".join(row["action"] for row in chosen)
        group = groups[task_key]
        baseline = int(_passed(group.rows["SSS"]))
        final = int(_passed(group.rows[sequence]))
        tasks.append(
            {
                "seed": seed,
                "budget_fraction": fraction,
                "task_key": task_key,
                "dataset": group.task.dataset,
                "sequence": sequence,
                "baseline_correct": baseline,
                "policy_correct": final,
                "final_policy_gain": final - baseline,
                "teacher_tokens": _tokens(group.rows[sequence]),
                "local_helpful_calls": sum(row["helpful_call"] for row in chosen),
                "local_harmful_calls": sum(row["harmful_call"] for row in chosen),
                "local_no_change_calls": sum(row["false_positive_no_change"] for row in chosen),
            }
        )
    for seed, fraction in sorted({(row["seed"], row["budget_fraction"]) for row in decisions}):
        chosen = [row for row in decisions if row["seed"] == seed and row["budget_fraction"] == fraction]
        task_chosen = [row for row in tasks if row["seed"] == seed and row["budget_fraction"] == fraction]
        summaries.append(
            {
                "seed": seed,
                "budget_fraction": fraction,
                "decisions": len(chosen),
                "verify_calls": sum(row["action"] == "V" for row in chosen),
                "helpful_calls_local": sum(row["helpful_call"] for row in chosen),
                "harmful_calls_local": sum(row["harmful_call"] for row in chosen),
                "false_positive_no_change": sum(row["false_positive_no_change"] for row in chosen),
                "missed_helpful_local": sum(row["missed_helpful"] for row in chosen),
                "wasted_no_change_tokens": sum(row["wasted_tokens"] for row in chosen),
                "harmful_tokens": sum(row["harmful_tokens"] for row in chosen),
                "final_task_wins": sum(row["final_policy_gain"] > 0 for row in task_chosen),
                "final_task_losses": sum(row["final_policy_gain"] < 0 for row in task_chosen),
                "final_task_ties": sum(row["final_policy_gain"] == 0 for row in task_chosen),
            }
        )
    return decisions, tasks, summaries


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def q_quality_rows(
    predictions: dict[int, V05Prediction],
    states: list[MultiHistoryState],
    *,
    split: str,
    bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gains = np.asarray([state.encoder_example.delta for state in states], dtype=float)
    helpful = (gains > 0).astype(int)
    quality: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    for seed, prediction in predictions.items():
        for checkpoint in (0, 1, 2, 3):
            mask = (
                np.ones(len(states), dtype=bool)
                if checkpoint == 0
                else np.asarray([state.checkpoint == checkpoint for state in states])
            )
            score = prediction.advantage[mask]
            y = helpful[mask]
            actual = gains[mask]
            rank_correlation = (
                float(np.corrcoef(_rankdata(score), _rankdata(actual))[0, 1])
                if np.std(score) > 0 and np.std(actual) > 0
                else math.nan
            )
            positives = int(y.sum())
            quality.append(
                {
                    "split": split,
                    "seed": seed,
                    "checkpoint": "all" if checkpoint == 0 else f"CP{checkpoint}",
                    "states": len(score),
                    "helpful": positives,
                    "q_advantage_auprc": (
                        float(average_precision_score(y, score)) if positives else math.nan
                    ),
                    "q_advantage_auroc": (
                        float(roc_auc_score(y, score)) if 0 < positives < len(y) else math.nan
                    ),
                    "q_advantage_gain_spearman": rank_correlation,
                    "q_advantage_mae": float(np.mean(np.abs(score - actual))),
                    "helpful_probability_brier": float(
                        np.mean((prediction.effect_probabilities[mask, 2] - y) ** 2)
                    ),
                }
            )
            order = np.argsort(score)
            for bin_index, indices in enumerate(np.array_split(order, bins)):
                if not len(indices):
                    continue
                calibration.append(
                    {
                        "split": split,
                        "seed": seed,
                        "checkpoint": "all" if checkpoint == 0 else f"CP{checkpoint}",
                        "bin": bin_index + 1,
                        "states": len(indices),
                        "mean_q_advantage": float(score[indices].mean()),
                        "mean_actual_local_gain": float(actual[indices].mean()),
                        "helpful_rate": float(y[indices].mean()),
                    }
                )
    return quality, calibration


def synergy_rows(split: str, groups: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task_key, group in tqdm(
        sorted(groups.items()), desc=f"{split} synergy", unit="task", dynamic_ncols=True
    ):
        outcome = {sequence: int(_passed(group.rows[sequence])) for sequence in SEQUENCES}
        singles = [sequence for sequence in SINGLE_SEQUENCES if outcome[sequence]]
        multis = [sequence for sequence in MULTI_SEQUENCES if outcome[sequence]]
        strict = [sequence for sequence in MULTI_SEQUENCES if strict_synergy(group, sequence)]
        requires_multi = not outcome["SSS"] and not singles and bool(multis)
        cheapest_multi = min(
            multis,
            key=lambda sequence: (_tokens(group.rows[sequence]), sequence.count("V"), sequence),
            default=None,
        )
        myopic_deltas: list[int] = []
        if cheapest_multi:
            prefix = ""
            for action in cheapest_multi:
                if action == "V":
                    skip = prefix + "S" + "S" * (2 - len(prefix))
                    verify = prefix + "V" + "S" * (2 - len(prefix))
                    myopic_deltas.append(int(_passed(group.rows[verify]) - _passed(group.rows[skip])))
                prefix += action
        rows.append(
            {
                "split": split,
                "task_key": task_key,
                "dataset": group.task.dataset,
                "baseline_correct": outcome["SSS"],
                "single_verify_success": int(bool(singles)),
                "multi_verify_success": int(bool(multis)),
                "requires_multi_verify": int(requires_multi),
                "strict_synergy": int(bool(strict)),
                "strict_synergy_sequences": ",".join(strict),
                "cheapest_multi_success": cheapest_multi or "",
                "cheapest_multi_tokens": _tokens(group.rows[cheapest_multi]) if cheapest_multi else "",
                "myopic_verify_deltas_along_cheapest_multi": ",".join(map(str, myopic_deltas)),
                "all_required_myopic_steps_nonpositive": int(
                    bool(myopic_deltas) and all(value <= 0 for value in myopic_deltas)
                ),
            }
        )
    return rows


def synergy_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for split in sorted({row["split"] for row in rows}):
        for dataset in sorted({row["dataset"] for row in rows if row["split"] == split} | {"combined"}):
            selected = [
                row
                for row in rows
                if row["split"] == split and (dataset == "combined" or row["dataset"] == dataset)
            ]
            output.append(
                {
                    "split": split,
                    "dataset": dataset,
                    "tasks": len(selected),
                    "baseline_solved": sum(row["baseline_correct"] for row in selected),
                    "single_verify_solvable": sum(
                        not row["baseline_correct"] and row["single_verify_success"] for row in selected
                    ),
                    "requires_multi_verify": sum(row["requires_multi_verify"] for row in selected),
                    "strict_synergy": sum(row["strict_synergy"] for row in selected),
                    "multi_required_invisible_to_myopic": sum(
                        row["requires_multi_verify"] and row["all_required_myopic_steps_nonpositive"]
                        for row in selected
                    ),
                    "unresolved": sum(
                        not row["baseline_correct"]
                        and not row["single_verify_success"]
                        and not row["multi_verify_success"]
                        for row in selected
                    ),
                }
            )
    return output


def _plots(
    output: Path,
    oracle: list[dict[str, Any]],
    learnability: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    q_quality: list[dict[str, Any]],
    synergy: list[dict[str, Any]],
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    full_oracle = [row for row in oracle if row["budget_fraction"] == 1.0]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(len(full_oracle))
    ax.bar(x - 0.18, [row["policy_actual_tokens"] for row in full_oracle], 0.36, label="DARC actual")
    ax.bar(x + 0.18, [row["oracle_minimum_tokens"] for row in full_oracle], 0.36, label="Oracle minimum")
    ax.set(
        xticks=x,
        xticklabels=[str(row["seed"])[-4:] for row in full_oracle],
        ylabel="Teacher tokens",
        title="Cost required for the same final accuracy",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "01_oracle_minimum_cost.png", dpi=220)
    plt.close(fig)

    metric_rows = [row for row in learnability if row["split"] == "train_oof" and row["checkpoint"] == "all"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(
        [row["model"] for row in metric_rows],
        [row["auprc"] for row in metric_rows],
        color=["#64748b", "#2563eb", "#16a34a", "#f97316"][: len(metric_rows)],
    )
    ax.axhline(metric_rows[0]["prevalence"], color="black", linestyle="--", label="Prevalence")
    ax.set(ylabel="Helpful AUPRC", title="Train-OOF Helpful learnability")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "02_helpful_learnability.png", dpi=220)
    plt.close(fig)

    cp = [row for row in checkpoints if row["split"] == "train" and row["dataset"] == "combined"]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    labels = [f"CP{row['checkpoint']}" for row in cp]
    bottom = np.zeros(len(cp))
    for key, name, color in (
        ("helpful_rate", "Helpful", "#16a34a"),
        ("harmful_rate", "Harmful", "#dc2626"),
        ("no_change_rate", "No-change", "#94a3b8"),
    ):
        values = np.asarray([row[key] for row in cp])
        ax.bar(labels, values, bottom=bottom, label=name, color=color)
        bottom += values
    ax.set(ylabel="Local state proportion", title="Counterfactual effects by checkpoint")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "03_checkpoint_effects.png", dpi=220)
    plt.close(fig)

    full_failures = [row for row in failures if row["budget_fraction"] == 1.0]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(
        [str(row["seed"])[-4:] for row in full_failures],
        [row["wasted_no_change_tokens"] for row in full_failures],
        label="No-change tokens",
        color="#f59e0b",
    )
    ax.bar(
        [str(row["seed"])[-4:] for row in full_failures],
        [row["harmful_tokens"] for row in full_failures],
        bottom=[row["wasted_no_change_tokens"] for row in full_failures],
        label="Harmful tokens",
        color="#dc2626",
    )
    ax.set(ylabel="Teacher tokens", title="Policy waste at full budget")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "04_policy_waste.png", dpi=220)
    plt.close(fig)

    quality = [row for row in q_quality if row["split"] == "train_oof" and row["checkpoint"] == "all"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(
        [str(row["seed"])[-4:] for row in quality],
        [row["q_advantage_auprc"] for row in quality],
        color="#7c3aed",
    )
    ax.set(ylabel="AUPRC", title="Q-advantage Helpful ranking across seeds")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "05_q_ranking.png", dpi=220)
    plt.close(fig)

    combined = [row for row in synergy if row["dataset"] == "combined"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    names = [row["split"] for row in combined]
    categories = [
        ("baseline_solved", "SSS solved"),
        ("single_verify_solvable", "Single Verify"),
        ("requires_multi_verify", "Requires multi"),
        ("unresolved", "Unresolved"),
    ]
    bottom = np.zeros(len(combined))
    for key, label in categories:
        values = np.asarray([row[key] for row in combined])
        ax.bar(names, values, bottom=bottom, label=label)
        bottom += values
    ax.set(ylabel="Tasks", title="Single- vs multi-Verify necessity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "06_single_multi_necessity.png", dpi=220)
    plt.close(fig)


def _report(
    output: Path,
    oracle: list[dict[str, Any]],
    learnability: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    q_quality: list[dict[str, Any]],
    synergy: list[dict[str, Any]],
) -> Path:
    full_oracle = [row for row in oracle if row["budget_fraction"] == 1.0]
    oof = [row for row in learnability if row["split"] == "train_oof" and row["checkpoint"] == "all"]
    val = [
        row for row in learnability if row["split"] == "validation_diagnostic" and row["checkpoint"] == "all"
    ]
    train_cp = [row for row in checkpoints if row["split"] == "train" and row["dataset"] == "combined"]
    full_fail = [row for row in failures if row["budget_fraction"] == 1.0]
    q_oof = [row for row in q_quality if row["split"] == "train_oof" and row["checkpoint"] == "all"]
    syn = next(row for row in synergy if row["split"] == "train" and row["dataset"] == "combined")
    lines = [
        "# DARC v0.5 Bottleneck Diagnostic",
        "",
        "This report uses frozen Level-2 train data, train-OOF predictions, and the "
        "unchanged validation split for diagnostics only. No frozen-test partition was "
        "accessed and no policy was selected from validation outcomes.",
        "",
        "## 1. Oracle minimum cost at matched accuracy",
        "",
        "| Seed | DARC tokens | Oracle minimum | Avoidable tokens | Cost ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in full_oracle:
        lines.append(
            f"| {row['seed']} | {row['policy_actual_tokens']:,} | "
            f"{row['oracle_minimum_tokens']:,} | {row['avoidable_tokens']:,} | "
            f"{row['policy_to_oracle_cost_ratio']:.1f}× |"
        )
    lines.extend(
        [
            "",
            "![Oracle cost](../figures/01_oracle_minimum_cost.png)",
            "",
            "## 2. Helpful learnability",
            "",
            "| Model | Train-OOF AUPRC | Train-OOF AUROC | Validation AUPRC | Validation AUROC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in oof:
        diagnostic = next(item for item in val if item["model"] == row["model"])
        lines.append(
            f"| {row['model']} | {row['auprc']:.3f} | {row['auroc']:.3f} | "
            f"{diagnostic['auprc']:.3f} | {diagnostic['auroc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "![Helpful learnability](../figures/02_helpful_learnability.png)",
            "",
            "## 3. Local effects by checkpoint",
            "",
            "| CP | Helpful | Harmful | No-change | Mean call tokens |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in train_cp:
        lines.append(
            f"| {row['checkpoint']} | {100 * row['helpful_rate']:.2f}% | "
            f"{100 * row['harmful_rate']:.2f}% | {100 * row['no_change_rate']:.2f}% | "
            f"{row['mean_actual_verify_tokens']:.0f} |"
        )
    lines.extend(
        [
            "",
            "![Checkpoint effects](../figures/03_checkpoint_effects.png)",
            "",
            "## 4. Policy errors at full budget",
            "",
            "| Seed | Calls | Helpful | Harmful | No-change | Missed Helpful | Wasted tokens |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in full_fail:
        lines.append(
            f"| {row['seed']} | {row['verify_calls']} | {row['helpful_calls_local']} | "
            f"{row['harmful_calls_local']} | {row['false_positive_no_change']} | "
            f"{row['missed_helpful_local']} | {row['wasted_no_change_tokens']:,} |"
        )
    lines.extend(
        [
            "",
            "![Policy waste](../figures/04_policy_waste.png)",
            "",
            "## 5. Q-advantage quality",
            "",
            "| Seed | OOF AUPRC | OOF AUROC | Gain rank correlation | Q MAE |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in q_oof:
        lines.append(
            f"| {row['seed']} | {row['q_advantage_auprc']:.3f} | "
            f"{row['q_advantage_auroc']:.3f} | "
            f"{row['q_advantage_gain_spearman']:.3f} | {row['q_advantage_mae']:.3f} |"
        )
    lines.extend(
        [
            "",
            "![Q ranking](../figures/05_q_ranking.png)",
            "",
            "## 6. Single- versus multi-Verify",
            "",
            f"On the {syn['tasks']} training tasks: {syn['baseline_solved']} are solved by "
            f"SSS, {syn['single_verify_solvable']} additional tasks are solvable by a single "
            f"Verify, {syn['requires_multi_verify']} require multiple Verify calls, and "
            f"{syn['strict_synergy']} exhibit strict synergy. For "
            f"{syn['multi_required_invisible_to_myopic']} multi-required tasks, all Verify "
            "steps along the cheapest successful multi path have non-positive myopic targets.",
            "",
            "![Single versus multi](../figures/06_single_multi_necessity.png)",
            "",
            "## Evidence-supported conclusion",
            "",
            "The bottleneck is classified from three independent signals: oracle cost "
            "headroom, Helpful ranking quality, and policy waste. Local counterfactual labels "
            "must not be interpreted as final policy-level effects; both are exported "
            "separately for audit.",
        ]
    )
    report = output / "report" / "darc_bottleneck_diagnostic.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(config_path: Path, *, smoke: bool) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "darc-bottleneck-diagnostic-v1":
        raise ValueError("Unexpected bottleneck diagnostic config schema")
    if config["data"].get("frozen_test_access") != "forbidden":
        raise ValueError("Frozen-test access must remain forbidden")
    output = Path(config["output"]["root"])
    for directory in ("tables", "figures", "report", "logs"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    root = Path(config["data"]["frozen_root"])
    train_groups = _load_groups(root, config["data"]["train_split"])
    validation_groups = _load_groups(root, config["data"]["validation_split"])
    _assert_disjoint(train_groups, validation_groups)
    if smoke:
        train_groups = dict(list(sorted(train_groups.items()))[: int(config["smoke"]["train_tasks"])])
        validation_groups = dict(
            list(sorted(validation_groups.items()))[: int(config["smoke"]["validation_tasks"])]
        )
        config["learnability"]["lightgbm"]["n_estimators"] = int(config["smoke"]["lightgbm_estimators"])
    train_states, train_lookup = build_states(train_groups)
    validation_states, _validation_lookup = build_states(validation_groups)
    encoder_config = dict(config["semantic_encoder"])
    cache = Path(encoder_config.pop("cache_dir"))
    train_encoded, train_cache = encode_or_load(
        [state.encoder_example for state in train_states],
        split="darc_v05_train",
        config=encoder_config,
        cache_dir=cache,
    )
    validation_encoded, validation_cache = encode_or_load(
        [state.encoder_example for state in validation_states],
        split="darc_v05_validation",
        config=encoder_config,
        cache_dir=cache,
    )

    analysis = config["analysis"]
    allowance = float(analysis["model_allowance"])
    seeds = list(map(int, config["reproducibility"]["seeds"]))
    folds_count = int(config["reproducibility"]["crossfit_folds"])
    if smoke:
        folds_count = min(2, len(train_groups))
    task_folds = assign_task_folds(
        list(train_groups), folds_count, int(config["reproducibility"]["crossfit_seed"])
    )
    train_predictions: dict[int, V05Prediction] = {}
    validation_predictions: dict[int, V05Prediction] = {}
    artifact_root = Path(config["baseline"]["artifact_root"])
    for seed in tqdm(seeds, desc="DARC v0.5 seeds", unit="seed", dynamic_ncols=True):
        model = DARCv05Ensemble.load(
            artifact_root / config["baseline"]["checkpoint_template"].format(seed=seed)
        )
        train_predictions[seed] = oof_predictions(
            model,
            train_encoded,
            train_states,
            [allowance],
            task_folds,
            folds=folds_count,
            device=str(analysis["device"]),
            description=f"seed {seed} train OOF",
        )[0]
        validation_predictions[seed] = _precompute_v05(
            model,
            validation_encoded,
            validation_states,
            [allowance],
            device=str(analysis["device"]),
            description=f"seed {seed} validation",
        )[0]

    baseline_oof, baseline_validation = grouped_oof_baselines(
        train_encoded,
        train_states,
        validation_encoded,
        folds=task_folds,
        fold_count=folds_count,
        config=config["learnability"],
        seed=int(config["reproducibility"]["crossfit_seed"]),
    )
    learnability_scores_train = {**baseline_oof}
    learnability_scores_validation = {**baseline_validation}
    for seed in seeds:
        learnability_scores_train[f"DARC Helpful head s{seed}"] = train_predictions[
            seed
        ].effect_probabilities[:, 2]
        learnability_scores_validation[f"DARC Helpful head s{seed}"] = validation_predictions[
            seed
        ].effect_probabilities[:, 2]
    learnability = learnability_metrics(
        learnability_scores_train, train_states, split="train_oof", top_k=list(map(int, analysis["top_k"]))
    )
    learnability += learnability_metrics(
        learnability_scores_validation,
        validation_states,
        split="validation_diagnostic",
        top_k=list(map(int, analysis["top_k"])),
    )

    train_oracle_tasks = oracle_task_rows(
        "train", train_groups, hard_cap=int(analysis["hard_call_token_cap"])
    )
    validation_oracle_tasks = oracle_task_rows(
        "validation", validation_groups, hard_cap=int(analysis["hard_call_token_cap"])
    )
    source_results = [
        row
        for row in _read_csv(artifact_root / config["baseline"]["result_table"])
        if row["policy"] == config["baseline"]["policy_label"]
    ]
    if smoke:
        source_results = []
        for seed in seeds:
            target = sum(int(_passed(group.rows["SSS"])) for group in validation_groups.values())
            source_results.append(
                {
                    "seed": str(seed),
                    "budget_fraction": "1.0",
                    "accuracy": str(target / len(validation_groups)),
                    "tasks": str(len(validation_groups)),
                    "teacher_tokens": "0",
                }
            )
    oracle_compare = oracle_comparison_rows(validation_oracle_tasks, source_results)

    local = local_effect_rows("train", train_states, train_groups)
    local += local_effect_rows("validation", validation_states, validation_groups)
    source_traces = [
        row
        for row in _read_csv(artifact_root / config["baseline"]["trace_table"])
        if row["policy"] == config["baseline"]["policy_label"]
    ]
    if smoke:
        source_traces = [
            row
            for row in source_traces
            if row["task_key"] in validation_groups and float(row["budget_fraction"]) == 1.0
        ]
    decisions, policy_tasks, failure_summary = policy_diagnostic_rows(source_traces, validation_groups)
    cp_summary = checkpoint_summary(local, decisions)

    q_quality_train, q_calibration_train = q_quality_rows(
        train_predictions, train_states, split="train_oof", bins=int(analysis["calibration_bins"])
    )
    q_quality_validation, q_calibration_validation = q_quality_rows(
        validation_predictions,
        validation_states,
        split="validation_diagnostic",
        bins=int(analysis["calibration_bins"]),
    )
    q_quality = q_quality_train + q_quality_validation
    q_calibration = q_calibration_train + q_calibration_validation

    synergy = synergy_rows("train", train_groups) + synergy_rows("validation", validation_groups)
    synergy_summaries = synergy_summary(synergy)
    examples = sorted(
        [
            row
            for row in decisions
            if row["false_positive_no_change"] or row["harmful_call"] or row["missed_helpful"]
        ],
        key=lambda row: (-row["harmful_call"], -row["missed_helpful"], -row["wasted_tokens"]),
    )[:100]

    table = output / "tables"
    _write_csv(table / "oracle_task_costs.csv", train_oracle_tasks + validation_oracle_tasks)
    _write_csv(table / "oracle_matched_accuracy.csv", oracle_compare)
    _write_csv(table / "helpful_learnability.csv", learnability)
    _write_csv(table / "local_counterfactual_effects.csv", local)
    _write_csv(table / "checkpoint_summary.csv", cp_summary)
    _write_csv(table / "policy_decision_errors.csv", decisions)
    _write_csv(table / "policy_task_outcomes.csv", policy_tasks)
    _write_csv(table / "policy_failure_summary.csv", failure_summary)
    _write_csv(table / "q_advantage_quality.csv", q_quality)
    _write_csv(table / "q_advantage_calibration.csv", q_calibration)
    _write_csv(table / "single_multi_task_analysis.csv", synergy)
    _write_csv(table / "single_multi_summary.csv", synergy_summaries)
    _write_csv(table / "task_level_examples.csv", examples or [{"note": "no diagnostic examples"}])
    _plots(output, oracle_compare, learnability, cp_summary, failure_summary, q_quality, synergy_summaries)
    report = _report(
        output, oracle_compare, learnability, cp_summary, failure_summary, q_quality, synergy_summaries
    )
    manifest = {
        "schema_version": "darc-bottleneck-diagnostic-result-v1",
        "mode": "smoke" if smoke else "full",
        "train_tasks": len(train_groups),
        "validation_tasks": len(validation_groups),
        "frozen_test_loaded": False,
        "student_teacher_inference": False,
        "model_training": ["Logistic Regression", "LightGBM"],
        "darc_weights_modified": False,
        "train_embedding_cache": train_cache,
        "validation_embedding_cache": validation_cache,
        "report": str(report),
    }
    (output / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/darc_bottleneck_diagnostic.yaml"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    report = run(args.config, smoke=args.smoke)
    print(f"Completed in {time.monotonic() - started:.1f}s")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
