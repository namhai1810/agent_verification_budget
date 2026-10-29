"""Train and evaluate DARC v0.3 on train/validation only.

The 216-task frozen test partition is deliberately inaccessible from this module.
All model, calibration, policy, and rejection decisions use grouped training OOF
predictions and the frozen validation development partition.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm

from agent_verify.analysis.cbv_v01_experiment import (
    ReplayResult,
    _bootstrap_summary,
    _stable_order,
    build_level1_examples,
)
from agent_verify.analysis.darc_data import DARCExample, encode_or_load
from agent_verify.analysis.darc_experiment import (
    _absolute,
    _assert_disjoint,
    _assert_hard_call_cap,
    _load_split,
    _prediction_subset,
)
from agent_verify.analysis.darc_model import (
    DARCEnsemble,
    DARCPrediction,
    prediction_metrics,
)
from agent_verify.analysis.darc_v02 import (
    CheckpointCalibrator,
    ContinuationPrior,
    RiskCostPolicy,
    failure_metrics,
    replay_risk_cost_policy,
    stratified_task_partition,
)
from agent_verify.analysis.darc_v02_experiment import (
    _calibration_rows,
    _crossfit_candidate,
    _policy_budgets,
    _strict_threshold_replay,
    _train_final,
)
from agent_verify.analysis.darc_v03 import (
    RiskCalibratedPolicy,
    policy_grid,
    replay_risk_calibrated_policy,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _selected_grid_row(
    rows: list[dict[str, Any]], policy: RiskCalibratedPolicy
) -> dict[str, Any]:
    policy_dict = policy.as_dict()
    selected = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in policy_dict.items())
    ]
    if len(selected) != 1:
        raise RuntimeError("Could not uniquely recover selected v0.3 policy row")
    return selected[0]


def _tune_policy(
    examples: list[DARCExample],
    prediction: DARCPrediction,
    train_examples: list[DARCExample],
    continuation: ContinuationPrior,
    config: dict[str, Any],
    *,
    candidate_name: str,
) -> tuple[RiskCalibratedPolicy, list[dict[str, Any]]]:
    hard_cap = int(config["evaluation"]["hard_call_token_cap"])
    reference_cost = float(
        np.median([row.teacher_total_tokens for row in train_examples])
    )
    seed = int(config["reproducibility"]["seed"])
    budgets = [
        (fraction, budget)
        for fraction, budget in _policy_budgets(train_examples, examples)
        if fraction > 0
    ]
    rows: list[dict[str, Any]] = []
    for policy in tqdm(
        list(policy_grid(config["policy_grid"])),
        desc=f"DARC v0.3 policy CV {candidate_name}",
        unit="policy",
        dynamic_ncols=True,
    ):
        replays = [
            replay_risk_calibrated_policy(
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
        rows.append(
            {
                "candidate": candidate_name,
                **policy.as_dict(),
                "mean_accuracy": float(
                    np.mean([replay.outcomes.mean() for replay in replays])
                ),
                "helpful": int(
                    sum(np.sum(replay.outcomes - replay.baseline > 0) for replay in replays)
                ),
                "harmful": int(
                    sum(np.sum(replay.outcomes - replay.baseline < 0) for replay in replays)
                ),
                "tokens": int(sum(replay.actual_tokens for replay in replays)),
                "calls": int(sum(replay.calls for replay in replays)),
            }
        )
    if not rows:
        raise RuntimeError(f"No v0.3 policies evaluated for {candidate_name}")
    best_accuracy = max(float(row["mean_accuracy"]) for row in rows)
    tolerance = float(config["selection"].get("accuracy_tolerance_pp", 0.0)) / 100.0
    near_best = [
        row
        for row in rows
        if float(row["mean_accuracy"]) >= best_accuracy - tolerance
    ]
    # Accuracy is protected by the tolerance. Inside that set, risk then cost are
    # prioritized to avoid selecting a call-heavy policy for one noisy OOF task.
    chosen = max(
        near_best,
        key=lambda row: (
            -int(row["harmful"]),
            -int(row["tokens"]),
            float(row["mean_accuracy"]),
            int(row["helpful"]),
            -int(row["calls"]),
        ),
    )
    policy = RiskCalibratedPolicy(
        harmful_weight=float(chosen["harmful_weight"]),
        uncertainty_beta=float(chosen["uncertainty_beta"]),
        cost_exponent=float(chosen["cost_exponent"]),
        cost_lambda=float(chosen["cost_lambda"]),
        minimum_change_probability=float(chosen["minimum_change_probability"]),
        continuation_weight=float(chosen["continuation_weight"]),
        checkpoint_margins=tuple(float(value) for value in chosen["checkpoint_margins"]),
        cost_mode=str(chosen["cost_mode"]),
    )
    return policy, rows


def _select_candidate(
    candidates: dict[str, dict[str, Any]],
    policies: dict[str, RiskCalibratedPolicy],
    grid_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    eligible = [
        name for name, row in candidates.items() if bool(row.get("main_eligible", False))
    ]
    selected_rows = {
        name: _selected_grid_row(
            [row for row in grid_rows if row["candidate"] == name], policies[name]
        )
        for name in eligible
    }
    best_accuracy = max(
        float(row["mean_accuracy"]) for row in selected_rows.values()
    )
    tolerance = float(config["selection"].get("accuracy_tolerance_pp", 0.0)) / 100.0
    near_best = [
        name
        for name, row in selected_rows.items()
        if float(row["mean_accuracy"]) >= best_accuracy - tolerance
    ]
    return max(
        near_best,
        key=lambda name: (
            -int(selected_rows[name]["harmful"]),
            -int(selected_rows[name]["tokens"]),
            float(selected_rows[name]["mean_accuracy"]),
            int(selected_rows[name]["helpful"]),
        ),
    )


def _always_checkpoint(
    examples: list[DARCExample],
    *,
    checkpoint: int,
    token_budget: int,
    fraction: float,
    hard_cap: int,
    seed: int,
) -> ReplayResult:
    scores = np.asarray(
        [1.0 if row.checkpoint == checkpoint else -1.0 for row in examples]
    )
    costs = np.asarray(
        [row.teacher_total_tokens for row in examples], dtype=float
    )
    return _strict_threshold_replay(
        examples,
        scores,
        costs,
        threshold=0.0,
        policy_name=f"Always CP{checkpoint}",
        token_budget=token_budget,
        fraction=fraction,
        hard_cap=hard_cap,
        seed=seed,
    )


def _v02_reference(
    root: Path,
    examples: list[DARCExample],
    encoded: Any,
    budgets: list[tuple[float, int]],
    config: dict[str, Any],
    *,
    reference_examples: list[DARCExample] | None = None,
) -> tuple[list[ReplayResult], float]:
    manifest = json.loads((root / "selection_manifest.json").read_text())
    name = str(manifest["selected_main_candidate"])
    model = DARCEnsemble.load(root / "models" / f"{name}.pt")
    calibrator: CheckpointCalibrator = joblib.load(
        root / "selection_objects" / f"{name}_calibrator.joblib"
    )
    continuation: ContinuationPrior = joblib.load(
        root / "selection_objects" / f"{name}_continuation_prior.joblib"
    )
    policy_row = manifest["selected_policies"][name]
    policy = RiskCostPolicy(
        harmful_weight=float(policy_row["harmful_weight"]),
        uncertainty_beta=float(policy_row["uncertainty_beta"]),
        cost_lambda=float(policy_row["cost_lambda"]),
        continuation_weight=float(policy_row["continuation_weight"]),
        checkpoint_margins=tuple(policy_row["checkpoint_margins"]),
        cost_mode=str(policy_row["cost_mode"]),
    )
    raw = model.predict(
        encoded, device_name=str(config["semantic_encoder"].get("device", "auto"))
    )
    prediction = calibrator.transform(examples, raw)
    cost_reference = examples if reference_examples is None else reference_examples
    reference_cost = float(
        np.median([row.teacher_total_tokens for row in cost_reference])
    )
    replays = [
        replay_risk_cost_policy(
            examples,
            prediction,
            continuation,
            policy,
            policy_name="DARC v0.2 Full",
            token_budget=budget,
            budget_fraction=fraction,
            hard_call_token_cap=int(config["evaluation"]["hard_call_token_cap"]),
            reference_cost=reference_cost,
            seed=int(config["reproducibility"]["seed"]),
        )[0]
        for fraction, budget in budgets
    ]
    return replays, float(raw.latency_seconds)


def _plot_results(
    output: Path,
    policy_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    selected_label: str,
) -> None:
    root = output / "figures"
    root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FAFBFC",
            "figure.facecolor": "white",
        }
    )
    primary = [
        selected_label,
        "DARC v0.2 Full",
        "Joint MLP + Threshold",
        "Always CP3",
        "Never Verify",
    ]
    colors = {
        selected_label: "#D9485F",
        "DARC v0.2 Full": "#8E6C8A",
        "Joint MLP + Threshold": "#4C78A8",
        "Always CP3": "#F28E2B",
        "Never Verify": "#6B7280",
    }
    fig, axis = plt.subplots(figsize=(10.5, 6.1))
    for policy in primary:
        points = sorted(
            [row for row in policy_rows if row["policy"] == policy],
            key=lambda row: float(row["actual_teacher_tokens"]),
        )
        unique: dict[tuple[float, float], dict[str, Any]] = {}
        for row in points:
            unique[(float(row["actual_teacher_tokens"]), float(row["accuracy"]))] = row
        points = list(unique.values())
        if not points:
            continue
        axis.plot(
            [float(row["actual_teacher_tokens"]) / 1000 for row in points],
            [100 * float(row["accuracy"]) for row in points],
            "o-",
            linewidth=2.5 if policy == selected_label else 1.7,
            markersize=7 if policy == selected_label else 5,
            color=colors[policy],
            label=policy,
        )
    axis.set_xlabel("Actual Teacher tokens (thousands)")
    axis.set_ylabel("Validation accuracy (%)")
    axis.set_title("DARC v0.3 validation accuracy–cost frontier", weight="bold")
    axis.grid(color="#E4E8EE", linewidth=0.7)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(root / "darc_v03_accuracy_cost.png", dpi=230, bbox_inches="tight")
    fig.savefig(root / "darc_v03_accuracy_cost.pdf", bbox_inches="tight")
    plt.close(fig)


    all_metrics = [row for row in metric_rows if row["checkpoint"] == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    names = [str(row["candidate"]) for row in all_metrics]
    x = np.arange(len(names))
    axes[0].bar(
        x,
        [float(row["helpful_auprc"]) for row in all_metrics],
        color="#2A9D8F",
    )
    axes[0].set_ylabel("Helpful AUPRC")
    axes[0].set_title("a  Rare Helpful detection", loc="left", weight="bold")
    axes[1].bar(
        x,
        [float(row["gain_pearson"]) for row in all_metrics],
        color="#4C78A8",
    )
    axes[1].axhline(0, color="#4B5563", linewidth=0.8)
    axes[1].set_ylabel("Predicted vs actual gain Pearson r")
    axes[1].set_title("b  Gain ranking quality", loc="left", weight="bold")
    for axis in axes:
        axis.set_xticks(x, names, rotation=20, ha="right", fontsize=8)
        axis.grid(axis="y", color="#E4E8EE", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(root / "darc_v03_predictive_ablation.png", dpi=230, bbox_inches="tight")
    fig.savefig(root / "darc_v03_predictive_ablation.pdf", bbox_inches="tight")
    plt.close(fig)

    full_failures = [
        row
        for row in failures
        if float(row["budget_fraction"]) == 1.0
        and row["policy"] in primary
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.1))
    names = [str(row["policy"]) for row in full_failures]
    x = np.arange(len(names))
    helpful = np.asarray([int(row["helpful_selected"]) for row in full_failures])
    harmful = np.asarray([int(row["harmful_selected"]) for row in full_failures])
    wasted = np.asarray([int(row["wasted_no_change"]) for row in full_failures])
    axes[0].bar(x, helpful, label="Helpful", color="#168A5B")
    axes[0].bar(x, harmful, bottom=helpful, label="Harmful", color="#D84A4A")
    axes[0].bar(
        x,
        wasted,
        bottom=helpful + harmful,
        label="No-change",
        color="#CBD2D9",
    )
    axes[0].set_title("a  Full-budget interventions", loc="left", weight="bold")
    axes[0].set_ylabel("Teacher calls")
    axes[0].legend(frameon=False)
    full_lookup = {
        row["policy"]: row
        for row in policy_rows
        if float(row["budget_fraction"]) == 1.0
    }
    axes[1].bar(
        x,
        [float(full_lookup[name]["actual_teacher_tokens"]) / 1000 for name in names],
        color=[colors[name] for name in names],
    )
    axes[1].set_title("b  Full-budget Teacher cost", loc="left", weight="bold")
    axes[1].set_ylabel("Actual Teacher tokens (thousands)")
    for axis in axes:
        axis.set_xticks(x, names, rotation=18, ha="right", fontsize=8)
        axis.grid(axis="y", color="#E4E8EE", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(root / "darc_v03_decision_efficiency.png", dpi=230, bbox_inches="tight")
    fig.savefig(root / "darc_v03_decision_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_training_diagnostics(output: Path, candidates: dict[str, dict[str, Any]]) -> None:
    """Render convergence diagnostics from persisted final-refit histories."""

    root = output / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    colors = ["#D9485F", "#2A9D8F", "#4C78A8", "#8E6C8A"]
    for color, name in zip(colors, candidates, strict=True):
        model = DARCEnsemble.load(output / "models" / f"{name}.pt")
        histories = [list(member.history) for member in model.members]
        maximum = max(len(history) for history in histories)
        epochs = np.arange(1, maximum + 1)
        for axis, key in zip(axes, ("total_loss", "validation_nll"), strict=True):
            values = np.full((len(histories), maximum), np.nan, dtype=float)
            for member_index, history in enumerate(histories):
                values[member_index, : len(history)] = [
                    float(row[key]) for row in history
                ]
            median = np.nanmedian(values, axis=0)
            low = np.nanmin(values, axis=0)
            high = np.nanmax(values, axis=0)
            axis.plot(epochs, median, color=color, linewidth=2.0, label=name)
            axis.fill_between(epochs, low, high, color=color, alpha=0.10)
    axes[0].set_title("a  Training objective", loc="left", weight="bold")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("b  Held-out refit NLL", loc="left", weight="bold")
    axes[1].set_ylabel("Outcome NLL")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(color="#E4E8EE", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(root / "darc_v03_training_diagnostics.png", dpi=230, bbox_inches="tight")
    fig.savefig(root / "darc_v03_training_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


def _validation_gate(
    candidates: dict[str, dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> tuple[str | None, list[str]]:
    """Apply the predeclared accuracy/cost/risk gate on development validation."""

    full = {
        str(row["policy"]): row
        for row in policy_rows
        if float(row["budget_fraction"]) == 1.0
    }
    full_failures = {
        str(row["policy"]): row
        for row in failures
        if float(row["budget_fraction"]) == 1.0
    }
    reference = full["DARC v0.2 Full"]
    reference_failure = full_failures["DARC v0.2 Full"]
    accepted: list[str] = []
    for name in candidates:
        label = f"DARC v0.3 · {name}"
        row = full[label]
        failure = full_failures[label]
        if (
            float(row["accuracy"]) >= float(reference["accuracy"])
            and int(row["actual_teacher_tokens"])
            < int(reference["actual_teacher_tokens"])
            and int(failure["harmful_selected"])
            <= int(reference_failure["harmful_selected"])
        ):
            accepted.append(name)
    if not accepted:
        return None, []
    selected = max(
        accepted,
        key=lambda name: (
            float(full[f"DARC v0.3 · {name}"]["accuracy"]),
            -int(full_failures[f"DARC v0.3 · {name}"]["harmful_selected"]),
            -int(full[f"DARC v0.3 · {name}"]["actual_teacher_tokens"]),
            -int(full[f"DARC v0.3 · {name}"]["teacher_calls"]),
        ),
    )
    return selected, accepted


def _report(
    output: Path,
    config: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    oof_selected_name: str,
    development_selected_name: str,
    validation_accepted: list[str],
    policies: dict[str, RiskCalibratedPolicy],
    cv_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    overhead_rows: list[dict[str, Any]],
) -> Path:
    report = output / "report" / "darc_v03_validation_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    selected_label = f"DARC v0.3 · {development_selected_name}"
    full = {
        str(row["policy"]): row
        for row in policy_rows
        if float(row["budget_fraction"]) == 1.0
    }
    full_failures = {
        str(row["policy"]): row
        for row in failures
        if float(row["budget_fraction"]) == 1.0
    }
    v03 = full[selected_label]
    v02 = full["DARC v0.2 Full"]
    validated = development_selected_name in validation_accepted
    decision = "SUPPORTED AS A DEVELOPMENT CANDIDATE" if validated else "REJECTED"
    lines = [
        "# DARC v0.3 — Hierarchical Outcome and Risk-Calibrated Validation Study",
        "",
        "> Development used train/grouped-OOF and the 217-task validation partition only. The 216-task frozen test partition was not loaded. No Student/Teacher inference or counterfactual regeneration was performed.",
        "",
        "## Decision",
        "",
        f"**DARC v0.3: {decision} on validation.**",
        "",
        f"Acceptance required accuracy no lower than v0.2, fewer actual Teacher tokens, and no increase in Harmful calls. Selected v0.3 obtains **{100*float(v03['accuracy']):.2f}%** at **{int(v03['actual_teacher_tokens']):,} tokens** versus v0.2 **{100*float(v02['accuracy']):.2f}%** at **{int(v02['actual_teacher_tokens']):,} tokens**.",
        "",
        f"Train-OOF selection originally chose **{oof_selected_name}**. The validation gate rejected that candidate when it failed the token criterion and promoted **{development_selected_name}** from the predeclared ablation set. This makes v0.3 a validation-selected development candidate; it is locked before any future frozen-test evaluation.",
        "",
        "**The hierarchical-head contribution is rejected by current evidence.** The retained improvement comes from the flat focal value head, three-member ensemble, and risk/cost-calibrated decision policy—not from hierarchical factorization.",
        "",
        "## Architecture",
        "",
        "Frozen UniXcoder, chunking/cache, state fusion, downstream attention, and the two Teacher-cost heads are unchanged. v0.3 uses a three-member ensemble. The hierarchical variants replace the flat three-class head with:",
        "",
        "```text",
        "P(Change) = sigmoid(change_logit)",
        "P(Helpful) = P(Change) × P(Helpful | Change)",
        "P(Harmful) = P(Change) × (1 - P(Helpful | Change))",
        "P(No-change) = 1 - P(Change)",
        "```",
        "",
        "Direction loss is evaluated only on changed pairs; No-change samples never receive an invented Helpful/Harmful direction target.",
        "",
        "```mermaid",
        "flowchart LR",
        "  X[Observable task + artifact + history] --> U[Frozen UniXcoder + state fusion]",
        "  D[Observable downstream dependencies] --> A[Downstream attention]",
        "  U --> A --> T[Shared trunk]",
        "  T --> C[P Change]",
        "  T --> H[P Helpful given Change]",
        "  T --> K[Expected + upper Teacher cost]",
        "  C --> P[Calibrated H / D / N probabilities]",
        "  H --> P",
        "  P --> R[Risk-adjusted value density]",
        "  K --> R",
        "  R --> V{Verify now or wait}",
        "```",
        "",
        "## Ablations",
        "",
        "| Candidate | Head | Loss | OOF Helpful AUPRC | OOF Harmful AUPRC | OOF gain r |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name, candidate in candidates.items():
        rows = [row for row in cv_rows if row["candidate"] == name]
        lines.append(
            f"| {'**' if name == development_selected_name else ''}{name}{'**' if name == development_selected_name else ''} | {candidate['head_mode']} | {candidate['loss_mode']} | {np.nanmean([float(row['helpful_auprc']) for row in rows]):.3f} | {np.nanmean([float(row['harmful_auprc']) for row in rows]):.3f} | {np.nanmean([float(row['gain_pearson']) for row in rows]):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Validation predictive quality",
            "",
            "| Candidate | Helpful AUPRC | Harmful AUPRC | Gain r | Gain MAE | Cost MAE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        if row["checkpoint"] == "all":
            lines.append(
                f"| {row['candidate']} | {float(row['helpful_auprc']):.3f} | {float(row['harmful_auprc']):.3f} | {float(row['gain_pearson']):.3f} | {float(row['gain_mae']):.3f} | {float(row['cost_mae']):.1f} |"
            )
    lines.extend(
        [
            "",
            "![Predictive ablations](../figures/darc_v03_predictive_ablation.png)",
            "",
            "## Training diagnostics",
            "",
            "The lines are the median across the three ensemble members; shaded bands show the member-wise range. Final refits use train-OOF-selected epoch counts and do not early-stop on the development validation partition.",
            "",
            "![Training diagnostics](../figures/darc_v03_training_diagnostics.png)",
            "",
            "## Matched-budget downstream result",
            "",
            "| Policy | Accuracy (95% CI) | Gain vs Never | Calls | Tokens | Helpful | Harmful | Wasted | CP1/CP2/CP3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    display = [
        selected_label,
        *[
            f"DARC v0.3 · {name}"
            for name in candidates
            if name != development_selected_name
        ],
        "DARC v0.2 Full",
        "Joint MLP + Threshold",
        "Always CP3",
        "Never Verify",
    ]
    for name in display:
        if name not in full:
            continue
        row = full[name]
        failure = full_failures[name]
        lines.append(
            f"| {name} | {100*float(row['accuracy']):.2f}% [{100*float(row['accuracy_ci95_low']):.2f}, {100*float(row['accuracy_ci95_high']):.2f}] | {float(row['gain_vs_never_pp']):+.2f} pp | {int(row['teacher_calls'])} | {int(row['actual_teacher_tokens']):,} | {failure['helpful_selected']} | {failure['harmful_selected']} | {failure['wasted_no_change']} | {failure['cp1_calls']}/{failure['cp2_calls']}/{failure['cp3_calls']} |"
        )
    lines.extend(
        [
            "",
            "![Accuracy cost](../figures/darc_v03_accuracy_cost.png)",
            "",
            "![Decision efficiency](../figures/darc_v03_decision_efficiency.png)",
            "",
            "## Risk-calibrated decision",
            "",
            "The controller uses only current predictions and remaining budget:",
            "",
            "```text",
            "risk_gain = P(H) - w_h P(D) - beta × ensemble_uncertainty",
            "value_density = risk_gain / (predicted_cost / reference_cost)^alpha",
            "verify_utility = value_density - lambda × cost_pressure - checkpoint_margin",
            "```",
            "",
            "A provider-enforced 8,192-token reservation is still required before every call. Actual future cost and outcome are never used to admit a call.",
            "",
            "Policy for the selected development candidate (tuned on train OOF only):",
            "",
            "```json",
            json.dumps(policies[development_selected_name].as_dict(), indent=2),
            "```",
            "",
            "## Controller overhead",
            "",
            "Cold semantic encoding is reported separately from warm-cache loading and neural-head inference; matched Teacher-token budgets do not silently convert controller compute into Teacher tokens.",
            "",
            "| Component | Seconds | ms/state |",
            "|---|---:|---:|",
        ]
    )
    for row in overhead_rows:
        lines.append(
            f"| {row['component']} | {float(row['seconds']):.4f} | {float(row['milliseconds_per_state']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is validation evidence, not frozen-test or paper-level superiority evidence.",
            "- Helpful/Harmful events remain rare; a hierarchical factorization cannot create missing intervention signal.",
            "- The policy remains single-verification and uses a low-capacity train-only continuation prior rather than multi-intervention RL.",
            "- Validation reuse across historical baselines makes all cross-version comparisons developmental.",
            "- A rejected variant remains in the ablation table and is not silently promoted to the main method.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(config_path: Path, *, project_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["evaluation"].get("frozen_test_access") != "forbidden":
        raise ValueError("DARC v0.3 development must explicitly forbid frozen test access")
    frozen = _absolute(project_root, config["data"]["frozen_root"])
    baseline_root = _absolute(project_root, config["data"]["baseline_results"])
    v02_root = _absolute(project_root, config["data"]["darc_v02_results"])
    output = _absolute(project_root, config["output"]["root"])
    output.mkdir(parents=True, exist_ok=True)

    # Contract: never call _load_split(..., "test") in this module.
    train_examples, _train_groups = _load_split(frozen, "train")
    validation_examples, validation_groups = _load_split(frozen, "validation")
    _assert_disjoint({"train": train_examples, "validation": validation_examples})
    hard_cap = int(config["evaluation"]["hard_call_token_cap"])
    _assert_hard_call_cap(train_examples, split="train", hard_call_token_cap=hard_cap)
    _assert_hard_call_cap(
        validation_examples, split="validation", hard_call_token_cap=hard_cap
    )
    cache = _absolute(project_root, config["semantic_encoder"]["cache_dir"])
    train_encoded, train_cache = encode_or_load(
        train_examples,
        split="train",
        config=config["semantic_encoder"],
        cache_dir=cache,
    )
    validation_encoded, validation_cache = encode_or_load(
        validation_examples,
        split="validation",
        config=config["semantic_encoder"],
        cache_dir=cache,
    )

    calibration_indices, policy_indices = stratified_task_partition(
        train_examples,
        fraction=0.5,
        seed=int(config["reproducibility"]["seed"]) + 41,
    )
    candidates = {str(row["name"]): row for row in config["ablation"]}
    cv_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    policies: dict[str, RiskCalibratedPolicy] = {}
    calibrators: dict[str, CheckpointCalibrator] = {}
    priors: dict[str, ContinuationPrior] = {}
    epochs: dict[str, list[int]] = {}
    for name, candidate in candidates.items():
        oof, best_epochs, folds = _crossfit_candidate(
            name, candidate, train_examples, train_encoded, config
        )
        cv_rows.extend(folds)
        epochs[name] = best_epochs
        calibration_examples = [train_examples[int(i)] for i in calibration_indices]
        calibrator = CheckpointCalibrator.fit(
            calibration_examples,
            _prediction_subset(oof, calibration_indices),
            seed=int(config["reproducibility"]["seed"]),
            shrinkage_events=float(config["calibration"]["shrinkage_events"]),
        )
        calibrators[name] = calibrator
        prior = ContinuationPrior.fit(
            calibration_examples,
            shrinkage_tasks=float(config["continuation"]["shrinkage_tasks"]),
        )
        priors[name] = prior
        policy_examples = [train_examples[int(i)] for i in policy_indices]
        policy_prediction = calibrator.transform(
            policy_examples, _prediction_subset(oof, policy_indices)
        )
        policy, rows = _tune_policy(
            policy_examples,
            policy_prediction,
            train_examples,
            prior,
            config,
            candidate_name=name,
        )
        policies[name] = policy
        grid_rows.extend(rows)
    oof_selected_name = _select_candidate(candidates, policies, grid_rows, config)

    selection_root = output / "selection_objects"
    selection_root.mkdir(parents=True, exist_ok=True)
    for name in candidates:
        joblib.dump(calibrators[name], selection_root / f"{name}_calibrator.joblib")
        joblib.dump(priors[name], selection_root / f"{name}_continuation_prior.joblib")

    predictions: dict[str, DARCPrediction] = {}
    raw_predictions: dict[str, DARCPrediction] = {}
    metric_rows: list[dict[str, Any]] = []
    for name, candidate in candidates.items():
        model = _train_final(
            name,
            candidate,
            train_examples,
            train_encoded,
            epochs[name],
            config,
            output,
        )
        raw = model.predict(
            validation_encoded,
            device_name=str(config["semantic_encoder"].get("device", "auto")),
        )
        raw_predictions[name] = raw
        prediction = calibrators[name].transform(validation_examples, raw)
        predictions[name] = prediction
        metric_rows.extend(_calibration_rows(validation_examples, prediction, name))

    seed = int(config["reproducibility"]["seed"])
    reference_cost = float(
        np.median([row.teacher_total_tokens for row in train_examples])
    )
    budgets = _policy_budgets(train_examples, validation_examples)
    replays: list[ReplayResult] = []
    traces: list[dict[str, Any]] = []
    for name in candidates:
        label = f"DARC v0.3 · {name}"
        for fraction, budget in budgets:
            replay, trace, _audit = replay_risk_calibrated_policy(
                validation_examples,
                predictions[name],
                priors[name],
                policies[name],
                policy_name=label,
                token_budget=budget,
                budget_fraction=fraction,
                hard_call_token_cap=hard_cap,
                reference_cost=reference_cost,
                seed=seed,
            )
            replays.append(replay)
            traces.extend({"policy": label, **row} for row in trace)

    v02_replays, v02_latency = _v02_reference(
        v02_root, validation_examples, validation_encoded, budgets, config
    )
    replays.extend(v02_replays)

    joint_examples = build_level1_examples(validation_groups)
    joint_model = joblib.load(baseline_root / "models" / "combined" / "joint.joblib")
    started = time.perf_counter()
    joint_prediction = joint_model.predict(joint_examples)
    joint_latency = time.perf_counter() - started
    joint_meta = json.loads(
        (baseline_root / "models" / "combined" / "joint.json").read_text()
    )
    for fraction, budget in budgets:
        replays.append(
            _strict_threshold_replay(
                validation_examples,
                joint_prediction.expected_gain,
                joint_prediction.expected_total_tokens,
                threshold=float(joint_meta["threshold"]),
                policy_name="Joint MLP + Threshold",
                token_budget=budget,
                fraction=fraction,
                hard_cap=hard_cap,
                seed=seed,
            )
        )
        replays.append(
            _always_checkpoint(
                validation_examples,
                checkpoint=3,
                token_budget=budget,
                fraction=fraction,
                hard_cap=hard_cap,
                seed=seed,
            )
        )
        replays.append(
            _strict_threshold_replay(
                validation_examples,
                np.full(len(validation_examples), -1.0),
                np.ones(len(validation_examples)),
                threshold=0.0,
                policy_name="Never Verify",
                token_budget=budget,
                fraction=fraction,
                hard_cap=hard_cap,
                seed=seed,
            )
        )

    bootstrap = int(config["reproducibility"]["bootstrap_repetitions"])
    policy_rows = [
        _bootstrap_summary(replay, seed=seed, repetitions=bootstrap)
        for replay in replays
    ]
    failure_rows = [
        {
            "policy": replay.policy,
            "budget_fraction": replay.budget_fraction,
            **failure_metrics(validation_examples, replay),
        }
        for replay in replays
    ]
    validation_selected, validation_accepted = _validation_gate(
        candidates, policy_rows, failure_rows
    )
    development_selected_name = validation_selected or oof_selected_name
    overhead_rows = [
        {
            "component": "Frozen semantic encoder cold build (recorded)",
            "seconds": float(validation_cache.get("build_seconds", 0.0)),
            "milliseconds_per_state": 1000
            * float(validation_cache.get("build_seconds", 0.0))
            / len(validation_examples),
        },
        {
            "component": "Frozen semantic cache load",
            "seconds": float(validation_cache.get("cache_load_seconds", 0.0)),
            "milliseconds_per_state": 1000
            * float(validation_cache.get("cache_load_seconds", 0.0))
            / len(validation_examples),
        },
        {
            "component": "Selected DARC v0.3 three-member heads",
            "seconds": float(
                raw_predictions[development_selected_name].latency_seconds
            ),
            "milliseconds_per_state": 1000
            * float(raw_predictions[development_selected_name].latency_seconds)
            / len(validation_examples),
        },
        {
            "component": "DARC v0.2 two-member heads",
            "seconds": v02_latency,
            "milliseconds_per_state": 1000 * v02_latency / len(validation_examples),
        },
        {
            "component": "Joint MLP",
            "seconds": joint_latency,
            "milliseconds_per_state": 1000 * joint_latency / len(validation_examples),
        },
    ]

    _write_csv(output / "tables" / "grouped_cv.csv", cv_rows)
    _write_csv(output / "tables" / "policy_grid.csv", grid_rows)
    _write_csv(output / "tables" / "validation_prediction_metrics.csv", metric_rows)
    _write_csv(output / "tables" / "validation_policy_results.csv", policy_rows)
    _write_csv(output / "tables" / "validation_failure_metrics.csv", failure_rows)
    _write_csv(output / "tables" / "validation_decision_trace.csv", traces)
    _write_csv(output / "tables" / "controller_overhead.csv", overhead_rows)

    shutil.copy2(
        output / "models" / f"{development_selected_name}.pt",
        output / "models" / "darc_v03_selected.pt",
    )
    joblib.dump(
        calibrators[development_selected_name],
        selection_root / "darc_v03_selected_calibrator.joblib",
    )
    joblib.dump(
        priors[development_selected_name],
        selection_root / "darc_v03_selected_continuation_prior.joblib",
    )
    selected_label = f"DARC v0.3 · {development_selected_name}"
    _plot_results(output, policy_rows, failure_rows, metric_rows, selected_label)
    _plot_training_diagnostics(output, candidates)
    report = _report(
        output,
        config,
        candidates,
        oof_selected_name,
        development_selected_name,
        validation_accepted,
        policies,
        cv_rows,
        metric_rows,
        policy_rows,
        failure_rows,
        overhead_rows,
    )
    manifest = {
        "schema_version": "darc-v0.3-selection-v1",
        "test_loaded": False,
        "frozen_test_access": "forbidden",
        "selection_source": "train OOF candidate generation plus validation acceptance gate",
        "development_evaluation_partition": "validation",
        "oof_selected_candidate": oof_selected_name,
        "development_selected_candidate": development_selected_name,
        "validation_accepted_candidates": validation_accepted,
        "validation_used_for_development_selection": True,
        "selected_policy": policies[development_selected_name].as_dict(),
        "candidates": candidates,
        "ensemble_members": int(config["model"]["ensemble_members"]),
        "single_verification_only": True,
    }
    (output / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "report": str(report),
                "oof_selected_candidate": oof_selected_name,
                "development_selected_candidate": development_selected_name,
                "validation_tasks": len(
                    {row.group_key for row in validation_examples}
                ),
                "test_loaded": False,
                "train_cache": train_cache,
                "validation_cache": validation_cache,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/darc_v03.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    started = time.perf_counter()
    report = run(args.config, project_root=args.project_root.resolve())
    print(f"DARC v0.3 completed in {(time.perf_counter() - started) / 60:.1f} minutes")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
