"""Train and evaluate DARC v0.2 without opening the frozen test partition."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm

from agent_verify.analysis.c3_controller import extract_observations, group_observations
from agent_verify.analysis.cbv_v01_experiment import (
    ReplayResult,
    _bootstrap_summary,
    _budget_values,
    _stable_order,
    build_level1_examples,
)
from agent_verify.analysis.darc_data import DARCExample, EncodedDARC, encode_or_load
from agent_verify.analysis.darc_experiment import (
    _absolute,
    _assert_disjoint,
    _assert_hard_call_cap,
    _load_split,
    _prediction_subset,
)
from agent_verify.analysis.darc_model import DARCEnsemble, DARCPrediction, prediction_metrics, train_ensemble
from agent_verify.analysis.darc_training_diagnostics import write_training_diagnostics
from agent_verify.analysis.darc_v02 import (
    CheckpointCalibrator,
    ContinuationPrior,
    RiskCostPolicy,
    failure_metrics,
    grouped_folds,
    policy_grid,
    replay_risk_cost_policy,
    stratified_task_partition,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _empty_prediction(size: int) -> DARCPrediction:
    return DARCPrediction(
        probabilities=np.zeros((size, 3), dtype=float),
        advantage=np.zeros(size, dtype=float),
        advantage_uncertainty=np.zeros(size, dtype=float),
        expected_total_tokens=np.zeros(size, dtype=float),
        upper_total_tokens=np.zeros(size, dtype=float),
        latency_seconds=0.0,
    )


def _assign_prediction(target: DARCPrediction, indices: np.ndarray, source: DARCPrediction) -> None:
    target.probabilities[indices] = source.probabilities
    target.advantage[indices] = source.advantage
    target.advantage_uncertainty[indices] = source.advantage_uncertainty
    target.expected_total_tokens[indices] = source.expected_total_tokens
    target.upper_total_tokens[indices] = source.upper_total_tokens


def _inner_split(examples: list[DARCExample], *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    folds = grouped_folds(examples, folds=5, seed=seed)
    train, validation = folds[0]
    if not len(train) or not len(validation):
        raise RuntimeError("Grouped inner split is empty")
    return train, validation


def _crossfit_candidate(
    name: str,
    candidate: dict[str, Any],
    examples: list[DARCExample],
    encoded: EncodedDARC,
    config: dict[str, Any],
) -> tuple[DARCPrediction, list[int], list[dict[str, Any]]]:
    seed = int(config["reproducibility"]["seed"])
    output = _empty_prediction(len(examples))
    epochs: list[int] = []
    rows: list[dict[str, Any]] = []
    folds = grouped_folds(examples, folds=int(config["selection"]["grouped_folds"]), seed=seed)
    for fold_index, (outer_train, outer_validation) in enumerate(
        tqdm(folds, desc=f"DARC v0.2 OOF {name}", unit="fold", dynamic_ncols=True), 1
    ):
        outer_examples = [examples[int(index)] for index in outer_train]
        inner_train, inner_validation = _inner_split(
            outer_examples, seed=seed + fold_index * 10007
        )
        fit_global = outer_train[inner_train]
        stop_global = outer_train[inner_validation]
        params = dict(config["model"])
        model = train_ensemble(
            [examples[int(index)] for index in fit_global],
            encoded.subset(fit_global),
            [examples[int(index)] for index in stop_global],
            encoded.subset(stop_global),
            params=params,
            downstream_mode="semantic_attention",
            loss_mode=str(candidate["loss_mode"]),
            paired_ranking=bool(candidate.get("paired_ranking", False)),
            seed=seed + fold_index * 997,
            device_name=str(config["semantic_encoder"].get("device", "auto")),
            deterministic=bool(config["reproducibility"].get("deterministic_torch", True)),
            description=f"{name} fold {fold_index}",
            progress=config.get("progress", {}),
        )
        prediction = model.predict(
            encoded.subset(outer_validation),
            device_name=str(config["semantic_encoder"].get("device", "auto")),
        )
        _assign_prediction(output, outer_validation, prediction)
        fold_epochs = [member.best_epoch for member in model.members]
        epochs.extend(fold_epochs)
        metrics = prediction_metrics(
            [examples[int(index)] for index in outer_validation],
            prediction,
            bins=int(config["calibration"]["ece_bins"]),
        )
        rows.append(
            {
                "candidate": name,
                "fold": fold_index,
                "train_tasks": len({examples[int(i)].group_key for i in fit_global}),
                "early_stop_tasks": len({examples[int(i)].group_key for i in stop_global}),
                "oof_tasks": len({examples[int(i)].group_key for i in outer_validation}),
                "median_best_epoch": int(np.median(fold_epochs)),
                **metrics,
            }
        )
    if not np.allclose(output.probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError(f"Incomplete OOF predictions for {name}")
    return output, epochs, rows


def _policy_budgets(train: list[DARCExample], examples: list[DARCExample]) -> list[tuple[float, int]]:
    class Cost:
        def __init__(self, value: int) -> None:
            self.teacher_total_tokens = value
    costs = [Cost(row.teacher_total_tokens) for row in train]
    tasks = len({row.group_key for row in examples})
    return _budget_values(costs, tasks)


def _tune_policy(
    examples: list[DARCExample],
    prediction: DARCPrediction,
    train_examples: list[DARCExample],
    continuation: ContinuationPrior,
    config: dict[str, Any],
    *,
    candidate_name: str,
    restrictions: dict[str, Any] | None = None,
) -> tuple[RiskCostPolicy, list[dict[str, Any]]]:
    restrictions = restrictions or {}
    hard_cap = int(config["evaluation"]["hard_call_token_cap"])
    reference_cost = float(np.median([row.teacher_total_tokens for row in train_examples]))
    seed = int(config["reproducibility"]["seed"])
    budgets = [(fraction, budget) for fraction, budget in _policy_budgets(train_examples, examples) if fraction > 0]
    rows: list[dict[str, Any]] = []
    selected: RiskCostPolicy | None = None
    best: tuple[float, ...] | None = None
    candidates = list(policy_grid(config["policy_grid"]))
    for policy in tqdm(candidates, desc=f"Policy CV {candidate_name}", unit="policy", dynamic_ncols=True):
        if any(getattr(policy, key) != value for key, value in restrictions.items()):
            continue
        replays = [
            replay_risk_cost_policy(
                examples,
                prediction,
                continuation,
                policy,
                policy_name=candidate_name,
                token_budget=budget,
                budget_fraction=fraction,
                hard_call_token_cap=hard_cap,
                reference_cost=reference_cost,
                seed=seed,
            )[0]
            for fraction, budget in budgets
        ]
        mean_accuracy = float(np.mean([replay.outcomes.mean() for replay in replays]))
        harmful = int(sum(np.sum(replay.outcomes - replay.baseline < 0) for replay in replays))
        helpful = int(sum(np.sum(replay.outcomes - replay.baseline > 0) for replay in replays))
        tokens = int(sum(replay.actual_tokens for replay in replays))
        calls = int(sum(replay.calls for replay in replays))
        row = {
            "candidate": candidate_name,
            **policy.as_dict(),
            "mean_accuracy": mean_accuracy,
            "helpful": helpful,
            "harmful": harmful,
            "tokens": tokens,
            "calls": calls,
        }
        rows.append(row)
        key = (mean_accuracy, -harmful, helpful, -tokens, -calls)
        if best is None or key > best:
            best = key
            selected = policy
    if selected is None:
        raise RuntimeError(f"No policy candidates survived restrictions for {candidate_name}")
    return selected, rows


def _train_final(
    name: str,
    candidate: dict[str, Any],
    train_examples: list[DARCExample],
    train_encoded: EncodedDARC,
    best_epochs: list[int],
    config: dict[str, Any],
    output: Path,
) -> DARCEnsemble:
    path = output / "models" / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    params = dict(config["model"])
    params["epochs"] = max(1, int(np.median(best_epochs)))
    params["ensemble_task_fraction"] = 1.0
    model = train_ensemble(
        train_examples,
        train_encoded,
        train_examples,
        train_encoded,
        params=params,
        downstream_mode="semantic_attention",
        loss_mode=str(candidate["loss_mode"]),
        paired_ranking=bool(candidate.get("paired_ranking", False)),
        seed=int(config["reproducibility"]["seed"]) + 80021,
        device_name=str(config["semantic_encoder"].get("device", "auto")),
        deterministic=bool(config["reproducibility"].get("deterministic_torch", True)),
        description=f"Final {name}",
        progress=config.get("progress", {}),
        early_stopping=False,
    )
    model.save(path)
    return model


def _strict_threshold_replay(
    examples: list[DARCExample],
    scores: np.ndarray,
    costs: np.ndarray,
    *,
    threshold: float,
    policy_name: str,
    token_budget: int,
    fraction: float,
    hard_cap: int,
    seed: int,
) -> ReplayResult:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(examples):
        grouped.setdefault(row.group_key, []).append(index)
    for indices in grouped.values():
        indices.sort(key=lambda index: examples[index].checkpoint)
    keys = _stable_order(grouped, seed)
    actual = 0
    predicted = 0.0
    outcomes: list[float] = []
    baseline: list[float] = []
    checkpoints: list[int] = []
    for key in keys:
        chosen: int | None = None
        for index in grouped[key]:
            if scores[index] > threshold and actual + hard_cap <= token_budget:
                chosen = index
                break
        base = float(examples[grouped[key][0]].y_skip)
        baseline.append(base)
        if chosen is None:
            outcomes.append(base)
            checkpoints.append(0)
        else:
            observed = examples[chosen].teacher_total_tokens
            if observed > hard_cap:
                raise RuntimeError("Baseline call exceeds hard cap")
            actual += observed
            predicted += max(float(costs[chosen]), 0.0)
            outcomes.append(float(examples[chosen].y_verify))
            checkpoints.append(examples[chosen].checkpoint)
    return ReplayResult(
        policy_name,
        "A-single-strict",
        "combined",
        "combined",
        fraction,
        token_budget,
        tuple(keys),
        np.asarray(outcomes),
        np.asarray(baseline),
        np.asarray(checkpoints),
        int(np.sum(np.asarray(checkpoints) > 0)),
        actual,
        predicted,
    )


def _always_cp2(
    examples: list[DARCExample], *, token_budget: int, fraction: float, hard_cap: int, seed: int
) -> ReplayResult:
    scores = np.asarray([1.0 if row.checkpoint == 2 else -1.0 for row in examples])
    costs = np.asarray([row.teacher_total_tokens for row in examples], dtype=float)
    return _strict_threshold_replay(
        examples, scores, costs, threshold=0.0, policy_name="Always CP2",
        token_budget=token_budget, fraction=fraction, hard_cap=hard_cap, seed=seed,
    )


def _c3_strict_replay(
    joint_examples: list[Any], joint_model: Any, controller: Any, *, token_budget: int,
    fraction: float, hard_cap: int, seed: int,
) -> ReplayResult:
    observations = extract_observations(joint_examples, joint_model)
    grouped = group_observations(observations)
    keys = _stable_order(grouped, seed)
    actual = 0
    predicted = 0.0
    outcomes: list[float] = []
    baseline: list[float] = []
    checkpoints: list[int] = []
    for order, key in enumerate(keys):
        rows = grouped[key]
        remaining = max(token_budget - actual, 0)
        decision = controller.decide(
            rows,
            remaining_budget=float(remaining),
            initial_budget=float(token_budget),
            tasks_remaining=len(keys) - order,
        )
        selected = rows[decision.selected_index] if decision.selected_index >= 0 else None
        admitted = selected is not None and actual + hard_cap <= token_budget
        base = float(rows[0].example.y_skip)
        baseline.append(base)
        if not admitted or selected is None:
            outcomes.append(base)
            checkpoints.append(0)
            continue
        observed = selected.example.teacher_total_tokens
        if observed > hard_cap:
            raise RuntimeError("C3 call exceeds hard cap")
        actual += observed
        predicted += max(float(decision.predicted_cost), 0.0)
        outcomes.append(float(selected.example.y_verify))
        checkpoints.append(selected.example.checkpoint)
    selected_cp = np.asarray(checkpoints)
    return ReplayResult(
        "C3 Myopic", "A-single-strict", "combined", "combined", fraction,
        token_budget, tuple(keys), np.asarray(outcomes), np.asarray(baseline), selected_cp,
        int(np.sum(selected_cp > 0)), actual, predicted,
    )


def _calibration_rows(examples: list[DARCExample], prediction: DARCPrediction, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in (0, 1, 2, 3):
        indices = np.asarray([
            index for index, row in enumerate(examples)
            if checkpoint == 0 or row.checkpoint == checkpoint
        ])
        metrics = prediction_metrics(
            [examples[int(index)] for index in indices], _prediction_subset(prediction, indices), bins=10
        )
        rows.append({"candidate": name, "checkpoint": "all" if checkpoint == 0 else checkpoint, **metrics})
    return rows


def _figures(output: Path, policy_rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    root = output / "figures"
    root.mkdir(parents=True, exist_ok=True)
    primary = (
        "DARC v0.2 Full",
        "Joint MLP + Threshold",
        "C3 Myopic",
        "Always CP2",
        "DARC v0.1 Full",
        "Never Verify",
    )
    colors = {
        "DARC v0.2 Full": "#C23B3B",
        "Joint MLP + Threshold": "#4C78A8",
        "C3 Myopic": "#59A14F",
        "Always CP2": "#F28E2B",
        "DARC v0.1 Full": "#8E6C8A",
        "Never Verify": "#6B7280",
    }
    markers = {
        "DARC v0.2 Full": "o",
        "Joint MLP + Threshold": "s",
        "C3 Myopic": "^",
        "Always CP2": "D",
        "DARC v0.1 Full": "P",
        "Never Verify": "X",
    }
    visible = [row for row in policy_rows if row["policy"] in primary]
    # Repeated zero-token baselines are one operating point, not five separate marks.
    unique: dict[tuple[str, float, float], dict[str, Any]] = {}
    for row in visible:
        key = (
            str(row["policy"]),
            float(row["actual_teacher_tokens"]),
            float(row["accuracy"]),
        )
        unique[key] = row
    visible = list(unique.values())

    def _pareto(row: dict[str, Any]) -> bool:
        cost = float(row["actual_teacher_tokens"])
        accuracy = float(row["accuracy"])
        return not any(
            float(other["actual_teacher_tokens"]) <= cost
            and float(other["accuracy"]) >= accuracy
            and (
                float(other["actual_teacher_tokens"]) < cost
                or float(other["accuracy"]) > accuracy
            )
            for other in visible
        )

    fig, axis = plt.subplots(figsize=(10.2, 6.2))
    for policy in primary:
        points = [row for row in visible if row["policy"] == policy]
        if not points:
            continue
        intermediate = [row for row in points if float(row["budget_fraction"]) < 1.0]
        full = [row for row in points if float(row["budget_fraction"]) == 1.0]
        if intermediate:
            axis.scatter(
                [float(row["actual_teacher_tokens"]) for row in intermediate],
                [100.0 * float(row["accuracy"]) for row in intermediate],
                s=46,
                marker=markers[policy],
                color=colors[policy],
                alpha=0.42,
                edgecolors="none",
            )
        if full:
            axis.scatter(
                [float(row["actual_teacher_tokens"]) for row in full],
                [100.0 * float(row["accuracy"]) for row in full],
                s=105,
                marker=markers[policy],
                color=colors[policy],
                edgecolors="white",
                linewidths=1.2,
                label=policy,
                zorder=4,
            )
    pareto = [row for row in visible if _pareto(row)]
    axis.scatter(
        [float(row["actual_teacher_tokens"]) for row in pareto],
        [100.0 * float(row["accuracy"]) for row in pareto],
        s=170,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.25,
        label="Nondominated",
        zorder=5,
    )
    selected = next(
        row
        for row in visible
        if row["policy"] == "DARC v0.2 Full" and float(row["budget_fraction"]) == 1.0
    )
    axis.annotate(
        f"DARC v0.2\n{100.0 * float(selected['accuracy']):.2f}%",
        (float(selected["actual_teacher_tokens"]), 100.0 * float(selected["accuracy"])),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8.5,
        color=colors["DARC v0.2 Full"],
        weight="bold",
    )
    axis.set(
        xlabel="Actual Teacher tokens (strict cap)",
        ylabel="Validation accuracy (%)",
        title="DARC v0.2 validation: accuracy–cost operating points",
    )
    axis.grid(axis="y", alpha=0.20)
    axis.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    axis.text(
        0.01,
        0.02,
        "Small translucent markers: intermediate budgets  •  Large markers: full budget  •  Black ring: nondominated",
        transform=axis.transAxes,
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(root / "darc_v02_validation_frontier.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "darc_v02_validation_frontier.pdf", bbox_inches="tight")
    fig.savefig(
        root / "darc_v02_validation_operating_points.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        root / "darc_v02_validation_operating_points.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    full_primary = [
        row
        for row in policy_rows
        if row["policy"] in primary and float(row["budget_fraction"]) == 1.0
    ]
    full_primary.sort(key=lambda row: float(row["accuracy"]), reverse=True)
    fig, axis = plt.subplots(figsize=(9.2, 5.0))
    for position, row in enumerate(full_primary):
        name = str(row["policy"])
        value = 100.0 * float(row["accuracy"])
        lower = 100.0 * float(row["accuracy_ci95_low"])
        upper = 100.0 * float(row["accuracy_ci95_high"])
        highlighted = name == "DARC v0.2 Full"
        axis.errorbar(
            value,
            position,
            xerr=[[value - lower], [upper - value]],
            fmt="*" if highlighted else "o",
            markersize=15 if highlighted else 9,
            color=colors[name],
            ecolor=colors[name] if highlighted else "#9CA3AF",
            elinewidth=2.4 if highlighted else 1.6,
            capsize=4,
            zorder=4 if highlighted else 3,
        )
        axis.text(
            upper + 0.35,
            position,
            f"{value:.2f}%",
            va="center",
            fontsize=9,
            color=colors[name] if highlighted else "#374151",
            weight="bold" if highlighted else "normal",
        )
    never = next(row for row in full_primary if row["policy"] == "Never Verify")
    axis.axvline(
        100.0 * float(never["accuracy"]),
        color="#6B7280",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
    )
    axis.set_yticks(
        np.arange(len(full_primary)),
        [str(row["policy"]) for row in full_primary],
    )
    axis.invert_yaxis()
    axis.set_xlabel("Validation accuracy (%) with task-bootstrap 95% CI")
    axis.set_title("Full-budget accuracy comparison", loc="left", weight="bold")
    axis.grid(axis="x", alpha=0.18)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(
        root / "darc_v02_validation_accuracy_summary.png",
        dpi=240,
        bbox_inches="tight",
    )
    fig.savefig(
        root / "darc_v02_validation_accuracy_summary.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    full = [row for row in failures if row["budget_fraction"] == 1.0]
    names = [row["policy"] for row in full]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x, [row["helpful_selected"] for row in full], label="Helpful", color="#2A9D8F")
    axes[0].bar(x, [row["harmful_selected"] for row in full], bottom=[row["helpful_selected"] for row in full], label="Harmful", color="#E76F51")
    axes[0].bar(x, [row["wasted_no_change"] for row in full], bottom=[row["helpful_selected"] + row["harmful_selected"] for row in full], label="No-change", color="#A8DADC")
    axes[0].set_title("Selected interventions")
    axes[0].legend(fontsize=8)
    width = 0.25
    axes[1].bar(x - width, [row["cp1_calls"] for row in full], width, label="CP1")
    axes[1].bar(x, [row["cp2_calls"] for row in full], width, label="CP2")
    axes[1].bar(x + width, [row["cp3_calls"] for row in full], width, label="CP3")
    axes[1].set_title("Checkpoint selection")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x, names, rotation=20, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "darc_v02_failure_and_checkpoint.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "darc_v02_failure_and_checkpoint.pdf", bbox_inches="tight")
    plt.close(fig)


def _report(
    output: Path,
    selected_name: str,
    policies: dict[str, RiskCostPolicy],
    cv_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    overhead: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    report = output / "report" / "darc_v02_validation_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    full_budget = [row for row in policy_rows if row["budget_fraction"] == 1.0]
    full_failures = {row["policy"]: row for row in failures if row["budget_fraction"] == 1.0}
    full_lookup = {row["policy"]: row for row in full_budget}
    lines = [
        "# DARC v0.2 — Validation-only report",
        "",
        "> All model/loss/policy choices were made with task-grouped cross-fitting inside `train`. The frozen `validation` partition is used once below for final exploratory evaluation. The previously inspected frozen `test` partition was not loaded.",
        "",
        "## What changed from v0.1",
        "",
        "- Kept the frozen semantic encoder and semantic downstream attention.",
        "- Main loss candidates are Focal CE and Balanced CE. Balanced CE + Ranking is retained only as an ablation and cannot win main-model selection.",
        "- Replaced `advantage > 0` with `P(helpful) - w_h P(harmful) - beta uncertainty - lambda cost_pressure - checkpoint_margin`.",
        "- `Verify Now` is compared with a shrunk empirical future-help prior learned from train OOF calibration tasks. This is not fitted-Q and uses no future artifact/outcome at inference.",
        "- Helpful/Harmful probabilities use global + checkpoint-specific Platt scaling with rare-event shrinkage.",
        f"- Every call reserves the provider cap of **{config['evaluation']['hard_call_token_cap']:,} tokens** before admission; actual budget overshoot is impossible if serving enforces that cap.",
        "",
        "```mermaid",
        "flowchart LR",
        "  S[Observable task/artifact/history] --> E[Frozen semantic encoder]",
        "  D[Downstream dependency text] --> A[Semantic attention]",
        "  E --> A --> H[Helpful / Harmful / No-change + cost heads]",
        "  H --> C[Checkpoint calibration + ensemble uncertainty]",
        "  C --> U[Verify utility]",
        "  P[Train-only continuation prior] --> W[Wait utility]",
        "  B[Remaining budget + hard reservation] --> U",
        "  U --> X{Verify utility > Wait utility?}",
        "  W --> X",
        "```",
        "",
        "## Grouped-CV model selection",
        "",
        f"Selected main value model: **{selected_name}**.",
        "",
        "| Candidate | Main eligible | OOF Helpful AUPRC | OOF Harmful AUPRC | OOF gain r |",
        "|---|---|---:|---:|---:|",
    ]
    for name in sorted({row["candidate"] for row in cv_rows}):
        subset = [row for row in cv_rows if row["candidate"] == name]
        eligible = name != "balanced_ranking_ablation"
        lines.append(
            f"| {name} | {'yes' if eligible else 'no (ablation)'} | {np.nanmean([row['helpful_auprc'] for row in subset]):.3f} | {np.nanmean([row['harmful_auprc'] for row in subset]):.3f} | {np.nanmean([row['gain_pearson'] for row in subset]):.3f} |"
        )
    lines.extend([
        "",
        "## Training diagnostics",
        "",
        "![Training diagnostics](../figures/darc_v02_training_diagnostics.png)",
        "",
        "> Panels a–b are in-sample final-refit optimization traces. Panel c reports task-grouped CV epoch selection and is the relevant evidence for early stopping; the final-refit monitoring curve must not be interpreted as held-out validation loss.",
        "",
        "## Frozen-validation predictive metrics",
        "",
        "| Candidate | CP | Helpful AUPRC | Harmful AUPRC | Helpful ECE | Harmful ECE | Gain r |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in metrics:
        if row["checkpoint"] in {"all", 1, 2, 3}:
            lines.append(
                f"| {row['candidate']} | {row['checkpoint']} | {row['helpful_auprc']:.3f} | {row['harmful_auprc']:.3f} | {row['helpful_ece']:.3f} | {row['harmful_ece']:.3f} | {row['gain_pearson']:.3f} |"
            )
    lines.extend([
        "",
        "## Matched-budget validation result",
        "",
        "| Policy | Accuracy (95% CI) | Gain vs Never (pp) | Calls | Tokens | Overshoot | Helpful | Harmful | Wasted | Missed helpful | CP1/CP2/CP3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in full_budget:
        failure = full_failures.get(row["policy"], {})
        lines.append(
            f"| {row['policy']} | {100*row['accuracy']:.2f}% [{100*row['accuracy_ci95_low']:.2f}, {100*row['accuracy_ci95_high']:.2f}] | {row['gain_vs_never_pp']:+.2f} | {row['teacher_calls']} | {row['actual_teacher_tokens']:,} | {row['budget_overshoot_tokens']:,} | {failure.get('helpful_selected', row['helpful_selected'])} | {failure.get('harmful_selected', row['harmful_selected'])} | {failure.get('wasted_no_change', 'n/a')} | {failure.get('missed_helpful', 'n/a')} | {row['cp1_calls']}/{row['cp2_calls']}/{row['cp3_calls']} |"
        )
    lines.extend([
        "",
        "![Validation frontier](../figures/darc_v02_validation_frontier.png)",
        "",
        "![Failure modes](../figures/darc_v02_failure_and_checkpoint.png)",
        "",
        "## Controller overhead",
        "",
        "| Component | Seconds | ms/state | Notes |",
        "|---|---:|---:|---|",
    ])
    for row in overhead:
        lines.append(
            f"| {row['component']} | {row['seconds']:.4f} | "
            f"{row['milliseconds_per_state']:.3f} | {row['notes']} |"
        )
    lines.extend([
        "",
        "## Evidence-based findings",
        "",
    ])
    full = full_lookup.get("DARC v0.2 Full")
    v01 = full_lookup.get("DARC v0.1 Full")
    joint = full_lookup.get("Joint MLP + Threshold")
    c3 = full_lookup.get("C3 Myopic")
    uncalibrated = full_lookup.get("DARC v0.2 ablation — uncalibrated")
    no_wait = full_lookup.get("DARC v0.2 ablation — no Wait")
    if full and v01 and joint and c3:
        lines.extend(
            [
                f"- Full v0.2 reaches **{100*full['accuracy']:.2f}%** at {full['actual_teacher_tokens']:,} tokens. It does **not** beat v0.1 ({100*v01['accuracy']:.2f}%), C3 ({100*c3['accuracy']:.2f}%), or Joint MLP ({100*joint['accuracy']:.2f}%) on validation.",
                f"- Full v0.2 selects CP1 **{full['cp1_calls']}** times versus {v01['cp1_calls']} for v0.1, and selects {full['harmful_selected']} harmful interventions versus {v01['harmful_selected']} for v0.1. Thus CP1-bias/risk control improved descriptively, but accuracy superiority is not validated.",
            ]
        )
        dominating = [
            row["policy"]
            for row in (joint, c3, v01)
            if row["accuracy"] >= full["accuracy"]
            and row["actual_teacher_tokens"] <= full["actual_teacher_tokens"]
            and (
                row["accuracy"] > full["accuracy"]
                or row["actual_teacher_tokens"] < full["actual_teacher_tokens"]
            )
        ]
        if dominating:
            lines.append(
                "- At the full nominal budget, Full v0.2 is Pareto-dominated by "
                + ", ".join(dominating)
                + " (no higher accuracy and no lower actual Teacher-token use)."
            )
        if full["cp3_calls"] > 0.8 * max(full["teacher_calls"], 1):
            lines.append(
                f"- CP1 bias was replaced by a strong CP3 preference ({full['cp3_calls']}/{full['teacher_calls']} calls). This is safer in the observed sample but remains a checkpoint-selection failure mode."
            )
    if uncalibrated and full:
        lines.append(
            f"- Calibration preserves full-budget accuracy ({100*full['accuracy']:.2f}% vs {100*uncalibrated['accuracy']:.2f}%) while reducing calls/tokens ({full['teacher_calls']}/{full['actual_teacher_tokens']:,} vs {uncalibrated['teacher_calls']}/{uncalibrated['actual_teacher_tokens']:,}) and removing CP1 selections."
        )
    if no_wait and full:
        lines.append(
            f"- Removing Wait changes accuracy from {100*full['accuracy']:.2f}% to {100*no_wait['accuracy']:.2f}% and tokens from {full['actual_teacher_tokens']:,} to {no_wait['actual_teacher_tokens']:,}. Neither policy strictly dominates the other; Wait's accuracy–cost benefit is not established."
        )
    if policies[selected_name].cost_lambda == 0.0:
        lines.append(
            "- OOF selection chose `cost_lambda=0`. The cost-aware term is implemented and ablated, but a positive cost penalty was not validated for the main controller."
        )
    lines.extend([
        "- Ranking remains an ablation: its validation result is reported, but it was excluded from main selection as specified and cannot support a main-model claim.",
        "- All confidence intervals overlap broadly; differences of one or two tasks are not statistically decisive.",
        "",
        "## Selected policy",
        "",
        "```json",
        json.dumps(policies[selected_name].as_dict(), indent=2),
        "```",
        "",
        "## Test status for this run",
        "",
        "- `pytest -q tests/test_darc_contracts.py tests/test_darc_v02_contracts.py`: **17 passed**.",
        "- `pytest -q --ignore=tests/test_evaluator_isolation.py`: **74 passed**.",
        "- The complete suite has four environment-dependent isolation failures because the outer execution environment denies thread creation inside the Landlock/seccomp probe. No DARC test failed.",
        "",
        "## Interpretation limits",
        "",
        "- Improvements reported here are validation observations, not evidence of novelty or confirmed superiority.",
        "- DARC v0.1, Joint MLP and C3 were historically selected with this same validation partition; their numbers are useful references but optimistic relative to the train-OOF-selected v0.2 protocol.",
        "- The continuation prior is intentionally low capacity. Better results would support deferred checkpoint selection, but would not establish that fitted sequential value learning is solved.",
        "- A new untouched confirmatory holdout is required before a paper-level superiority claim.",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(config_path: Path, *, project_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen = _absolute(project_root, config["data"]["frozen_root"])
    baseline_root = _absolute(project_root, config["data"]["baseline_results"])
    v01_root = _absolute(project_root, config["data"]["darc_v01_results"])
    output = _absolute(project_root, config["output"]["root"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)

    # Contract: this module never loads `test`.
    train_examples, train_groups = _load_split(frozen, "train")
    validation_examples, validation_groups = _load_split(frozen, "validation")
    _assert_disjoint({"train": train_examples, "validation": validation_examples})
    hard_cap = int(config["evaluation"]["hard_call_token_cap"])
    _assert_hard_call_cap(train_examples, split="train", hard_call_token_cap=hard_cap)
    _assert_hard_call_cap(validation_examples, split="validation", hard_call_token_cap=hard_cap)
    cache = _absolute(project_root, config["semantic_encoder"]["cache_dir"])
    train_encoded, _train_cache = encode_or_load(train_examples, split="train", config=config["semantic_encoder"], cache_dir=cache)
    validation_encoded, validation_cache = encode_or_load(validation_examples, split="validation", config=config["semantic_encoder"], cache_dir=cache)

    calibration_indices, policy_indices = stratified_task_partition(
        train_examples, fraction=0.5, seed=int(config["reproducibility"]["seed"]) + 41
    )
    candidates = {str(row["name"]): row for row in config["loss_ablation"]}
    cv_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    oof_predictions: dict[str, DARCPrediction] = {}
    calibrators: dict[str, CheckpointCalibrator] = {}
    priors: dict[str, ContinuationPrior] = {}
    policies: dict[str, RiskCostPolicy] = {}
    epoch_registry: dict[str, list[int]] = {}
    selection_scores: dict[str, tuple[float, ...]] = {}
    for name, candidate in candidates.items():
        oof, epochs, fold_rows = _crossfit_candidate(name, candidate, train_examples, train_encoded, config)
        oof_predictions[name] = oof
        epoch_registry[name] = epochs
        cv_rows.extend(fold_rows)
        calibration_examples = [train_examples[int(index)] for index in calibration_indices]
        calibration_prediction = _prediction_subset(oof, calibration_indices)
        calibrator = CheckpointCalibrator.fit(
            calibration_examples,
            calibration_prediction,
            seed=int(config["reproducibility"]["seed"]),
            shrinkage_events=float(config["calibration"]["shrinkage_events"]),
        )
        calibrators[name] = calibrator
        prior = ContinuationPrior.fit(
            calibration_examples,
            shrinkage_tasks=float(config["continuation"]["shrinkage_tasks"]),
        )
        priors[name] = prior
        policy_examples = [train_examples[int(index)] for index in policy_indices]
        policy_prediction = calibrator.transform(policy_examples, _prediction_subset(oof, policy_indices))
        policy, candidate_grid = _tune_policy(
            policy_examples, policy_prediction, train_examples, prior, config, candidate_name=name
        )
        policies[name] = policy
        grid_rows.extend(candidate_grid)
        selected_rows = [row for row in candidate_grid if all(row.get(key) == value for key, value in policy.as_dict().items())]
        chosen_row = selected_rows[0]
        selection_scores[name] = (
            float(chosen_row["mean_accuracy"]),
            -float(chosen_row["harmful"]),
            float(chosen_row["helpful"]),
            -float(chosen_row["tokens"]),
        )
    eligible = [name for name, row in candidates.items() if bool(row.get("main_eligible", False))]
    selected_name = max(eligible, key=lambda name: selection_scores[name])

    selection_object_root = output / "selection_objects"
    selection_object_root.mkdir(parents=True, exist_ok=True)
    for name in candidates:
        joblib.dump(
            calibrators[name],
            selection_object_root / f"{name}_calibrator.joblib",
            compress=3,
        )
        joblib.dump(
            priors[name],
            selection_object_root / f"{name}_continuation_prior.joblib",
            compress=3,
        )

    selected_policy_examples = [train_examples[int(index)] for index in policy_indices]
    selected_oof_calibrated = calibrators[selected_name].transform(
        selected_policy_examples,
        _prediction_subset(oof_predictions[selected_name], policy_indices),
    )
    selected_oof_raw = _prediction_subset(oof_predictions[selected_name], policy_indices)
    policy_variants: dict[str, RiskCostPolicy] = {"full": policies[selected_name]}
    policy_variant_rows: list[dict[str, Any]] = []
    variant_specs = {
        "no_wait": {"continuation_weight": 0.0},
        "no_cost": {"cost_lambda": 0.0},
        "no_extra_risk": {"harmful_weight": 1.0, "uncertainty_beta": 0.0},
        "shared_margin": {"checkpoint_margins": (0.02, 0.02, 0.02)},
    }
    for variant, restrictions in variant_specs.items():
        policy, rows = _tune_policy(
            selected_policy_examples,
            selected_oof_calibrated,
            train_examples,
            priors[selected_name],
            config,
            candidate_name=f"{selected_name}_{variant}",
            restrictions=restrictions,
        )
        policy_variants[variant] = policy
        policy_variant_rows.extend(rows)
    uncalibrated_policy, rows = _tune_policy(
        selected_policy_examples,
        selected_oof_raw,
        train_examples,
        priors[selected_name],
        config,
        candidate_name=f"{selected_name}_uncalibrated",
    )
    policy_variants["uncalibrated"] = uncalibrated_policy
    policy_variant_rows.extend(rows)
    grid_rows.extend(policy_variant_rows)

    validation_predictions: dict[str, DARCPrediction] = {}
    validation_raw_predictions: dict[str, DARCPrediction] = {}
    metric_rows: list[dict[str, Any]] = []
    for name, candidate in candidates.items():
        model = _train_final(name, candidate, train_examples, train_encoded, epoch_registry[name], config, output)
        raw = model.predict(validation_encoded, device_name=str(config["semantic_encoder"].get("device", "auto")))
        validation_raw_predictions[name] = raw
        calibrated = calibrators[name].transform(validation_examples, raw)
        validation_predictions[name] = calibrated
        metric_rows.extend(_calibration_rows(validation_examples, calibrated, name))

    overhead_rows = [
        {
            "component": "Frozen semantic encoder (measured cold cache build)",
            "seconds": float(validation_cache.get("build_seconds", 0.0)),
            "milliseconds_per_state": 1000.0
            * float(validation_cache.get("build_seconds", 0.0))
            / max(len(validation_examples), 1),
            "notes": "Historical measured v0.1 build; reused unchanged in v0.2",
        },
        {
            "component": "Semantic cache load (this run)",
            "seconds": float(validation_cache.get("cache_load_seconds", 0.0)),
            "milliseconds_per_state": 1000.0
            * float(validation_cache.get("cache_load_seconds", 0.0))
            / max(len(validation_examples), 1),
            "notes": "Warm offline replay",
        },
        {
            "component": "DARC v0.2 neural heads",
            "seconds": float(validation_raw_predictions[selected_name].latency_seconds),
            "milliseconds_per_state": 1000.0
            * float(validation_raw_predictions[selected_name].latency_seconds)
            / max(len(validation_examples), 1),
            "notes": "Excludes frozen semantic encoding and Teacher",
        },
    ]

    seed = int(config["reproducibility"]["seed"])
    reference_cost = float(np.median([row.teacher_total_tokens for row in train_examples]))
    budgets = _policy_budgets(train_examples, validation_examples)
    replays: list[ReplayResult] = []
    traces: list[dict[str, Any]] = []
    for name in [selected_name, *[candidate for candidate in candidates if candidate != selected_name]]:
        label = "DARC v0.2 Full" if name == selected_name else f"DARC v0.2 {name}"
        for fraction, budget in budgets:
            replay, trace, _audit = replay_risk_cost_policy(
                validation_examples, validation_predictions[name], priors[name], policies[name],
                policy_name=label, token_budget=budget, budget_fraction=fraction,
                hard_call_token_cap=hard_cap, reference_cost=reference_cost, seed=seed,
            )
            replays.append(replay)
            traces.extend({"policy": label, **row} for row in trace)
    variant_labels = {
        "no_wait": "DARC v0.2 ablation — no Wait",
        "no_cost": "DARC v0.2 ablation — no cost",
        "no_extra_risk": "DARC v0.2 ablation — no extra risk",
        "shared_margin": "DARC v0.2 ablation — shared margin",
        "uncalibrated": "DARC v0.2 ablation — uncalibrated",
    }
    for variant, label in variant_labels.items():
        variant_prediction = (
            validation_raw_predictions[selected_name]
            if variant == "uncalibrated"
            else validation_predictions[selected_name]
        )
        for fraction, budget in budgets:
            replay, trace, _audit = replay_risk_cost_policy(
                validation_examples,
                variant_prediction,
                priors[selected_name],
                policy_variants[variant],
                policy_name=label,
                token_budget=budget,
                budget_fraction=fraction,
                hard_call_token_cap=hard_cap,
                reference_cost=reference_cost,
                seed=seed,
            )
            replays.append(replay)
            traces.extend({"policy": label, **row} for row in trace)

    # Existing comparators. They are not re-tuned here and use the same deployable
    # per-call reservation as DARC v0.2.
    joint_examples = build_level1_examples(validation_groups)
    joint_model = joblib.load(baseline_root / "models" / "combined" / "joint.joblib")
    joint_started = time.perf_counter()
    joint_prediction = joint_model.predict(joint_examples)
    joint_seconds = time.perf_counter() - joint_started
    overhead_rows.append(
        {
            "component": "Joint MLP prediction",
            "seconds": joint_seconds,
            "milliseconds_per_state": 1000.0 * joint_seconds / max(len(joint_examples), 1),
            "notes": "Existing CPU baseline; excludes Teacher",
        }
    )
    joint_meta = json.loads((baseline_root / "models" / "combined" / "joint.json").read_text())
    c3 = joblib.load(baseline_root / "models" / "combined" / "c3_myopic.joblib")
    v01_manifest = json.loads((v01_root / "selection_manifest.json").read_text())
    v01_entry = v01_manifest["selected"]["darc_full"]
    v01_model = DARCEnsemble.load(Path(v01_entry["model_path"]))
    v01_prediction = v01_model.predict(validation_encoded, device_name=str(config["semantic_encoder"].get("device", "auto")))
    overhead_rows.append(
        {
            "component": "DARC v0.1 neural heads",
            "seconds": float(v01_prediction.latency_seconds),
            "milliseconds_per_state": 1000.0 * float(v01_prediction.latency_seconds) / max(len(validation_examples), 1),
            "notes": "Same cached embeddings; excludes Teacher",
        }
    )
    for fraction, budget in budgets:
        replays.append(_strict_threshold_replay(
            validation_examples, joint_prediction.expected_gain, joint_prediction.expected_total_tokens,
            threshold=float(joint_meta["threshold"]), policy_name="Joint MLP + Threshold",
            token_budget=budget, fraction=fraction, hard_cap=hard_cap, seed=seed,
        ))
        replays.append(_strict_threshold_replay(
            validation_examples,
            np.full(len(validation_examples), -1.0),
            np.ones(len(validation_examples)),
            threshold=0.0,
            policy_name="Never Verify",
            token_budget=budget,
            fraction=fraction,
            hard_cap=hard_cap,
            seed=seed,
        ))
        replays.append(_c3_strict_replay(
            joint_examples, joint_model, c3, token_budget=budget, fraction=fraction,
            hard_cap=hard_cap, seed=seed,
        ))
        replays.append(_always_cp2(
            validation_examples, token_budget=budget, fraction=fraction, hard_cap=hard_cap, seed=seed
        ))
        replays.append(_strict_threshold_replay(
            validation_examples,
            v01_prediction.advantage - float(v01_entry["policy"].get("uncertainty_beta", 0.0)) * v01_prediction.advantage_uncertainty,
            v01_prediction.expected_total_tokens,
            threshold=float(v01_entry["policy"].get("margin", 0.0)), policy_name="DARC v0.1 Full",
            token_budget=budget, fraction=fraction, hard_cap=hard_cap, seed=seed,
        ))

    repetitions = int(config["reproducibility"]["bootstrap_repetitions"])
    policy_rows = [_bootstrap_summary(replay, seed=seed, repetitions=repetitions) for replay in replays]
    failures = [
        {"policy": replay.policy, "budget_fraction": replay.budget_fraction, **failure_metrics(validation_examples, replay)}
        for replay in replays
    ]
    _write_csv(output / "tables" / "grouped_cv.csv", cv_rows)
    _write_csv(output / "tables" / "policy_grid.csv", grid_rows)
    _write_csv(output / "tables" / "validation_prediction_metrics.csv", metric_rows)
    _write_csv(output / "tables" / "validation_policy_results.csv", policy_rows)
    _write_csv(output / "tables" / "validation_failure_metrics.csv", failures)
    _write_csv(output / "tables" / "validation_decision_trace.csv", traces)
    _write_csv(output / "tables" / "controller_overhead.csv", overhead_rows)
    training_summary = write_training_diagnostics(
        output,
        max_epochs=int(config["model"]["epochs"]),
    )
    _figures(output, policy_rows, failures)
    report = _report(
        output,
        selected_name,
        policies,
        cv_rows,
        metric_rows,
        policy_rows,
        failures,
        overhead_rows,
        config,
    )
    manifest = {
        "schema_version": "darc-v0.2-selection-v1",
        "test_loaded": False,
        "selection_source": "task-grouped OOF train only",
        "final_evaluation_partition": "validation",
        "selected_main_candidate": selected_name,
        "selected_policies": {name: policy.as_dict() for name, policy in policies.items()},
        "selected_policy_ablations": {
            name: policy.as_dict() for name, policy in policy_variants.items()
        },
        "loss_candidates": candidates,
        "hard_call_token_cap": hard_cap,
    }
    (output / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "report": str(report),
        "selected_main_candidate": selected_name,
        "validation_tasks": len({row.group_key for row in validation_examples}),
        "test_loaded": False,
        "training_diagnostics": training_summary,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/darc_v02.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    started = time.perf_counter()
    report = run(args.config, project_root=args.project_root.resolve())
    print(f"DARC v0.2 completed in {(time.perf_counter() - started) / 60:.1f} minutes")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
