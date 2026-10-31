"""End-to-end train/validation/locked-test experiment for DARC v0.4."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

from agent_verify.analysis.cbv_v01_experiment import (
    ALL_SEQUENCES,
    MultiHistoryController,
    ReplayResult,
    _bootstrap_summary,
    _passed,
    _sequence_replay,
    _stable_order,
    load_frozen_partition,
    replay_multi_controller,
    task_trajectories,
)
from agent_verify.analysis.darc_data import build_darc_examples, encode_or_load
from agent_verify.analysis.darc_model import DARCEnsemble
from agent_verify.analysis.darc_v02 import CheckpointCalibrator, ContinuationPrior
from agent_verify.analysis.darc_v03 import RiskCalibratedPolicy, replay_risk_calibrated_policy
from agent_verify.analysis.darc_v04_data import (
    MultiHistoryState,
    audit_level2,
    build_states,
    build_targets,
    prefix_cost,
    verify_cost,
)
from agent_verify.analysis.darc_v04_model import (
    DARCv04Ensemble,
    V04Prediction,
    train_ensemble,
)


@dataclass(frozen=True)
class SequentialPolicy:
    harmful_weight: float
    uncertainty_beta: float
    cost_lambda: float
    decision_margin: float


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _assert_disjoint(*partitions: dict[str, Any]) -> None:
    sets = [set(partition) for partition in partitions]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(f"Task leakage across splits: {sorted(overlap)[:3]}")


def _load_groups(root: Path, split: str) -> dict[str, Any]:
    tasks, rows = load_frozen_partition(root, split)
    return task_trajectories(tasks, rows, level=2)


def _allowances(reference_budget: float, fractions: list[float]) -> list[float]:
    return [float(fraction) * reference_budget for fraction in fractions]


def _precompute(
    model: DARCv04Ensemble,
    encoded: Any,
    states: list[MultiHistoryState],
    allowances: list[float],
    *,
    device: str,
) -> list[V04Prediction]:
    return [
        model.predict(encoded, states, allowance=value, device_name=device)
        for value in tqdm(
            allowances,
            desc="DARC v0.4 budget-conditioned prediction",
            unit="budget",
            leave=False,
            dynamic_ncols=True,
        )
    ]


def _nearest_prediction(
    predictions: list[V04Prediction], allowances: list[float], allowance: float
) -> V04Prediction:
    index = int(np.argmin(np.abs(np.asarray(allowances) - allowance)))
    return predictions[index]


def replay_v04(
    groups: dict[str, Any],
    states: list[MultiHistoryState],
    lookup: dict[tuple[str, str], int],
    predictions: list[V04Prediction],
    allowances: list[float],
    policy: SequentialPolicy,
    *,
    token_budget: int,
    budget_fraction: float,
    hard_call_token_cap: int,
    seed: int,
    label: str,
) -> tuple[ReplayResult, list[dict[str, Any]]]:
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
            tasks_remaining = len(keys) - order
            allowance = remaining / max(tasks_remaining, 1)
            prediction = _nearest_prediction(predictions, allowances, allowance)
            index = lookup[(key, prefix)]
            q_skip = float(prediction.q_skip[index])
            q_verify = float(prediction.q_verify[index])
            q_advantage = q_verify - q_skip
            harmful = float(prediction.effect_probabilities[index, 0])
            helpful = float(prediction.effect_probabilities[index, 2])
            uncertainty = float(prediction.uncertainty[index])
            predicted_cost = max(float(prediction.expected_cost[index]), 1.0)
            cost_pressure = predicted_cost / max(allowance, 1.0)
            utility = (
                q_advantage
                - policy.harmful_weight * harmful
                - policy.uncertainty_beta * uncertainty
                - policy.cost_lambda * cost_pressure
            )
            feasible = (
                remaining >= hard_call_token_cap and predicted_cost <= remaining
            )
            verify = feasible and utility > policy.decision_margin
            direct_skip = prefix + "S" + "S" * (2 - len(prefix))
            direct_verify = prefix + "V" + "S" * (2 - len(prefix))
            direct_delta = _passed(group.rows[direct_verify]) - _passed(group.rows[direct_skip])
            actual_cost = verify_cost(group, prefix)
            traces.append(
                {
                    "policy": label,
                    "budget_fraction": budget_fraction,
                    "task_key": key,
                    "dataset": group.task.dataset,
                    "checkpoint": checkpoint,
                    "history": prefix or "START",
                    "q_skip": q_skip,
                    "q_verify": q_verify,
                    "q_advantage": q_advantage,
                    "prob_helpful": helpful,
                    "prob_harmful": harmful,
                    "uncertainty": uncertainty,
                    "predicted_cost": predicted_cost,
                    "actual_cost": actual_cost,
                    "remaining_budget": remaining,
                    "allowance": allowance,
                    "utility": utility,
                    "feasible": feasible,
                    "action": "V" if verify else "S",
                    "direct_actual_delta": direct_delta,
                    "helpful_selected": int(verify and direct_delta > 0),
                    "harmful_selected": int(verify and direct_delta < 0),
                    "wasted_selected": int(verify and direct_delta == 0),
                }
            )
            if verify:
                if actual_cost > hard_call_token_cap:
                    raise RuntimeError(
                        f"Teacher call {actual_cost} exceeds hard cap {hard_call_token_cap}"
                    )
                if actual_spent + actual_cost > token_budget:
                    raise AssertionError("DARC v0.4 hard budget overshoot")
                prefix += "V"
                actual_spent += actual_cost
                predicted_spent += predicted_cost
                calls += 1
                mask |= 1 << (checkpoint - 1)
            else:
                prefix += "S"
        outcomes.append(float(_passed(group.rows[prefix])))
        baselines.append(float(_passed(group.rows["SSS"])))
        masks.append(mask)
    return (
        ReplayResult(
            policy=label,
            protocol="C-multi-strict",
            train_mode="combined",
            scope="combined",
            budget_fraction=budget_fraction,
            token_budget=token_budget,
            task_keys=tuple(keys),
            outcomes=np.asarray(outcomes),
            baseline=np.asarray(baselines),
            checkpoints=np.asarray(masks),
            calls=calls,
            actual_tokens=actual_spent,
            predicted_tokens=predicted_spent,
        ),
        traces,
    )


def _policy_grid(config: dict[str, Any]) -> list[SequentialPolicy]:
    return [
        SequentialPolicy(*map(float, values))
        for values in itertools.product(
            config["harmful_weight"],
            config["uncertainty_beta"],
            config["cost_lambda"],
            config["decision_margin"],
        )
    ]


def _tune_policy(
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
) -> tuple[SequentialPolicy, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    chosen: tuple[tuple[float, float, float, float], SequentialPolicy] | None = None
    for candidate, policy in enumerate(
        tqdm(policies, desc=f"Tune {label}", unit="policy", dynamic_ncols=True)
    ):
        accuracies: list[float] = []
        harmful = 0
        tokens = 0
        calls = 0
        for fraction in fractions:
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
            accuracies.append(float(replay.outcomes.mean()))
            harmful += sum(int(row["harmful_selected"]) for row in traces)
            tokens += replay.actual_tokens
            calls += replay.calls
        score = float(np.mean(accuracies))
        row = {
            "variant": label,
            "candidate": candidate,
            **asdict(policy),
            "mean_validation_accuracy": score,
            "validation_harmful_actions": harmful,
            "validation_tokens": tokens,
            "validation_calls": calls,
        }
        rows.append(row)
        key = (score, -float(harmful), -float(tokens), -float(calls))
        if chosen is None or key > chosen[0]:
            chosen = (key, policy)
    assert chosen is not None
    return chosen[1], rows


def _effect_metrics(
    states: list[MultiHistoryState], prediction: V04Prediction
) -> dict[str, float]:
    labels = np.asarray([state.encoder_example.value_label for state in states])
    helpful = (labels == 2).astype(int)
    harmful = (labels == 0).astype(int)
    return {
        "helpful_auprc": float(average_precision_score(helpful, prediction.effect_probabilities[:, 2])),
        "harmful_auprc": float(average_precision_score(harmful, prediction.effect_probabilities[:, 0])),
        "q_advantage_mean": float(np.mean(prediction.q_advantage)),
        "cost_mae": float(
            np.mean(
                np.abs(
                    prediction.expected_cost
                    - np.asarray([state.encoder_example.teacher_total_tokens for state in states])
                )
            )
        ),
    }


def _v03_baseline(
    groups: dict[str, Any],
    config: dict[str, Any],
    root: Path,
    cache_dir: Path,
    budgets: list[tuple[float, int]],
    reference_cost: float,
) -> list[ReplayResult]:
    examples = build_darc_examples(groups)
    encoded, _ = encode_or_load(
        examples,
        split=f"v04_v03_{next(iter(groups.values())).task.dataset if groups else 'empty'}_{len(groups)}",
        config=config["semantic_encoder"],
        cache_dir=cache_dir,
    )
    manifest = json.loads((root / "selection_manifest.json").read_text())
    name = str(manifest["development_selected_candidate"])
    model = DARCEnsemble.load(root / "models" / f"{name}.pt")
    calibrator: CheckpointCalibrator = joblib.load(
        root / "selection_objects" / f"{name}_calibrator.joblib"
    )
    continuation: ContinuationPrior = joblib.load(
        root / "selection_objects" / f"{name}_continuation_prior.joblib"
    )
    raw = model.predict(encoded, device_name=str(config["semantic_encoder"]["device"]))
    prediction = calibrator.transform(examples, raw)
    row = manifest["selected_policy"]
    policy = RiskCalibratedPolicy(
        harmful_weight=float(row["harmful_weight"]),
        uncertainty_beta=float(row["uncertainty_beta"]),
        cost_exponent=float(row["cost_exponent"]),
        cost_lambda=float(row["cost_lambda"]),
        minimum_change_probability=float(row["minimum_change_probability"]),
        continuation_weight=float(row["continuation_weight"]),
        checkpoint_margins=tuple(float(value) for value in row["checkpoint_margins"]),
        cost_mode=str(row["cost_mode"]),
    )
    return [
        replay_risk_calibrated_policy(
            examples,
            prediction,
            continuation,
            policy,
            policy_name="DARC v0.3 (single)",
            token_budget=budget,
            budget_fraction=fraction,
            hard_call_token_cap=int(config["evaluation"]["hard_call_token_cap"]),
            reference_cost=reference_cost,
            seed=int(config["reproducibility"]["seed"]),
        )[0]
        for fraction, budget in budgets
    ]


def _baselines(
    groups: dict[str, Any],
    config: dict[str, Any],
    *,
    fractions: list[float],
    reference_budget: float,
    split: str,
) -> list[ReplayResult]:
    seed = int(config["reproducibility"]["seed"])
    budgets = [
        (fraction, int(round(fraction * reference_budget * len(groups))))
        for fraction in fractions
    ]
    output: list[ReplayResult] = []
    for fraction, budget in budgets:
        for sequence in ALL_SEQUENCES:
            output.append(
                _sequence_replay(
                    groups,
                    sequence,
                    budget=budget,
                    fraction=fraction,
                    scope="combined",
                    seed=seed,
                )
            )
    c3_root = Path(config["data"]["c3_results"])
    for filename, label in (
        ("multi_myopic.joblib", "C3 MultiHistory Myopic"),
        ("multi_continuation.joblib", "C3 MultiHistory Continuation"),
    ):
        controller: MultiHistoryController = joblib.load(
            c3_root / "models" / "combined" / filename
        )
        for fraction, budget in budgets:
            output.append(
                replay_multi_controller(
                    groups,
                    controller,
                    budget=budget,
                    fraction=fraction,
                    scope="combined",
                    seed=seed,
                    label=label,
                )
            )
    output.extend(
        _v03_baseline(
            groups,
            config,
            Path(config["data"]["darc_v03_results"]),
            Path(config["semantic_encoder"]["cache_dir"]),
            budgets,
            reference_budget / 3.0,
        )
    )
    return output


def _summaries(
    replays: list[ReplayResult], config: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            **_bootstrap_summary(
                replay,
                seed=int(config["reproducibility"]["seed"]),
                repetitions=int(config["reproducibility"]["bootstrap_repetitions"]),
            ),
        }
        for replay in replays
    ]


def _plot(
    output: Path,
    rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    selected_label: str,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FAFBFC",
            "figure.facecolor": "white",
        }
    )
    policies = [
        selected_label,
        "DARC v0.3 (single)",
        "C3 MultiHistory Myopic",
        "C3 MultiHistory Continuation",
        "Fixed VVS",
        "Fixed VVV",
        "Fixed SSS",
    ]
    colors = dict(
        zip(
            policies,
            ["#D9485F", "#8E6C8A", "#4C78A8", "#72A0C1", "#F28E2B", "#E59F00", "#6B7280"],
            strict=True,
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.7))
    test_rows = [row for row in rows if row["split"] == "test"]
    for policy in policies:
        points = sorted(
            [row for row in test_rows if row["policy"] == policy],
            key=lambda row: float(row["actual_teacher_tokens"]),
        )
        if not points:
            continue
        axes[0].plot(
            [float(row["actual_teacher_tokens"]) / 1000 for row in points],
            [100 * float(row["accuracy"]) for row in points],
            "o-",
            color=colors[policy],
            linewidth=2.8 if policy == selected_label else 1.5,
            markersize=7 if policy == selected_label else 5,
            label=policy,
        )
    axes[0].set_title("a  Frozen-test accuracy–cost frontier", loc="left", weight="bold")
    axes[0].set_xlabel("Actual Teacher tokens (thousands)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    full = [
        row
        for row in test_rows
        if float(row["budget_fraction"]) == 1.0 and row["policy"] in policies
    ]
    full.sort(key=lambda row: float(row["accuracy"]))
    axes[1].barh(
        [str(row["policy"]) for row in full],
        [100 * float(row["accuracy"]) for row in full],
        color=[colors[str(row["policy"])] for row in full],
    )
    for index, row in enumerate(full):
        axes[1].text(
            100 * float(row["accuracy"]) + 0.15,
            index,
            f"{100*float(row['accuracy']):.2f}% · {int(row['actual_teacher_tokens'])/1000:.0f}k",
            va="center",
            fontsize=8,
        )
    axes[1].set_title("b  Full-budget accuracy and cost", loc="left", weight="bold")
    axes[1].set_xlabel("Accuracy (%)")
    for axis in axes:
        axis.grid(axis="both", color="#E4E8EE", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(figures / "darc_v04_main_results.png", dpi=230, bbox_inches="tight")
    fig.savefig(figures / "darc_v04_main_results.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.8, 5.5))
    for variant in sorted({str(row["variant"]) for row in training_rows}):
        subset = [row for row in training_rows if row["variant"] == variant and int(row["member"]) == 1]
        axis.plot(
            [int(row["epoch"]) for row in subset],
            [float(row["validation_q_loss"]) for row in subset],
            label=variant,
            linewidth=2,
        )
    axis.set_title("DARC v0.4 training convergence", weight="bold")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation Q loss")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    axis.grid(color="#E4E8EE", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(figures / "darc_v04_training.png", dpi=230, bbox_inches="tight")
    fig.savefig(figures / "darc_v04_training.pdf", bbox_inches="tight")
    plt.close(fig)


def _report(
    output: Path,
    config: dict[str, Any],
    audit_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    results: list[dict[str, Any]],
    selected_variant: str,
    selected_policy: SequentialPolicy,
    trace_rows: list[dict[str, Any]],
) -> Path:
    report = output / "report" / "darc_v04_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    validation = {
        (row["policy"], float(row["budget_fraction"])): row
        for row in results
        if row["split"] == "validation"
    }
    test = {
        (row["policy"], float(row["budget_fraction"])): row
        for row in results
        if row["split"] == "test"
    }
    selected_label = f"DARC v0.4 · {selected_variant}"
    comparators = [
        selected_label,
        "DARC v0.3 (single)",
        "C3 MultiHistory Myopic",
        "C3 MultiHistory Continuation",
        "Fixed SSS",
        "Fixed VSS",
        "Fixed SVS",
        "Fixed SSV",
        "Fixed VVS",
        "Fixed VSV",
        "Fixed SVV",
        "Fixed VVV",
    ]
    selected_test = test[(selected_label, 1.0)]
    baseline_best = max(
        (test[(name, 1.0)] for name in comparators[1:] if (name, 1.0) in test),
        key=lambda row: float(row["accuracy"]),
    )
    matched_cap = int(selected_test["actual_teacher_tokens"])
    baseline_names = set(comparators[1:])
    matched_candidates = [
        row
        for row in results
        if row["split"] == "test"
        and row["policy"] in baseline_names
        and int(row["actual_teacher_tokens"]) <= matched_cap
    ]
    matched_best = max(
        matched_candidates,
        key=lambda row: (
            float(row["accuracy"]),
            -int(row["actual_teacher_tokens"]),
        ),
    )
    selected_at_cap = max(
        (
            row
            for row in results
            if row["split"] == "test"
            and row["policy"] == selected_label
            and int(row["actual_teacher_tokens"]) <= matched_cap
        ),
        key=lambda row: (
            float(row["accuracy"]),
            -int(row["actual_teacher_tokens"]),
        ),
    )
    outperformed = float(selected_at_cap["accuracy"]) > float(matched_best["accuracy"])
    full_traces = [
        row
        for row in trace_rows
        if row["split"] == "test"
        and row["policy"] == selected_label
        and float(row["budget_fraction"]) == 1.0
    ]
    helpful = sum(int(row["helpful_selected"]) for row in full_traces)
    harmful = sum(int(row["harmful_selected"]) for row in full_traces)
    wasted = sum(int(row["wasted_selected"]) for row in full_traces)
    lines = [
        "# DARC v0.4 — Risk-Calibrated Counterfactual Multi-Verification",
        "",
        "> Offline CBV-Bench experiment. Development uses train/validation only; the frozen test is loaded after model and policy selection are persisted. No Student/Teacher inference is run.",
        "",
        "## Verdict",
        "",
        (
            f"**DARC v0.4 {'outperforms' if outperformed else 'does not outperform'} the strongest measured baseline under a matched actual-token cap.** "
            f"Its best point at or below **{matched_cap:,} tokens** is **{100*float(selected_at_cap['accuracy']):.2f}%** at **{int(selected_at_cap['actual_teacher_tokens']):,} tokens**; "
            f"the strongest comparator is **{matched_best['policy']}** at **{100*float(matched_best['accuracy']):.2f}%** and **{int(matched_best['actual_teacher_tokens']):,} tokens**."
        ),
        "",
        f"At each policy's full reported budget, v0.4 obtains **{100*float(selected_test['accuracy']):.2f}% / {int(selected_test['actual_teacher_tokens']):,} tokens**, while the strongest fixed-history point is **{baseline_best['policy']}: {100*float(baseline_best['accuracy']):.2f}% / {int(baseline_best['actual_teacher_tokens']):,} tokens**.",
        "",
        "This is evidence for this fixed model pair/workflow and offline replay, not a general novelty or superiority claim.",
        "",
        "## Level-2 audit",
        "",
        "| Split | Dataset | Tasks | Trajectories | Decision states | Complete | Token-valid |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit_rows:
        lines.append(
            f"| {row['split']} | {row['dataset']} | {row['tasks']} | {row['trajectories']} | {row['decision_states']} | {row['sequence_complete']} | {row['token_valid']} |"
        )
    lines.extend(
        [
            "",
            "No task IDs overlap across train/validation/test. Every included task has all eight branches. Future outcomes and actual future costs are used only to construct training targets; inference features contain observed prefixes/artifacts and a remaining-budget bucket.",
            "",
            "## Architecture and target construction",
            "",
            "```text",
            "task/current artifact/observed artifact history",
            "        → frozen UniXcoder → State Fusion + downstream attention",
            "previous S/V actions → action embeddings ─┐",
            "remaining budget → budget projection ─────┼→ Q(Skip), Q(Verify)",
            "                                           ├→ Helpful/No-change/Harmful",
            "                                           └→ incremental Teacher cost",
            "```",
            "",
            "Q-targets are computed backwards from valid shared-prefix descendants. At a given budget, only observed descendant branches whose additional recorded cost fits the allowance are eligible. This is controlled fitted counterfactual supervision—not ground-truth online lookahead. v0.3 compatible semantic/state/trunk weights initialize the pretrained variants; new action, budget and Q heads are trained from scratch.",
            "",
            "## Validation-selected policy",
            "",
            "```json",
            json.dumps(asdict(selected_policy), indent=2),
            "```",
            "",
            "## Predictive ablations",
            "",
            "| Variant | Split | Helpful AUPRC | Harmful AUPRC | Cost MAE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['variant']} | {row['split']} | {float(row['helpful_auprc']):.3f} | {float(row['harmful_auprc']):.3f} | {float(row['cost_mae']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Controller ablations at full budget",
            "",
            "| Variant | Validation accuracy / tokens | Frozen-test accuracy / tokens |",
            "|---|---:|---:|",
        ]
    )
    for name in (
        "full_pretrained",
        "full_scratch",
        "no_history",
        "myopic_targets",
        "no_budget_conditioning",
    ):
        label = f"DARC v0.4 · {name}"
        val = validation[(label, 1.0)]
        tst = test[(label, 1.0)]
        lines.append(
            f"| {name} | {100*float(val['accuracy']):.2f}% / {int(val['actual_teacher_tokens']):,} | "
            f"{100*float(tst['accuracy']):.2f}% / {int(tst['actual_teacher_tokens']):,} |"
        )
    lines.extend(
        [
            "",
            "## Full-budget downstream comparison",
            "",
            "| Policy | Validation accuracy | Frozen-test accuracy (95% CI) | Calls | Test tokens | Gain vs SSS |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    sss = test[("Fixed SSS", 1.0)]
    for name in comparators:
        if (name, 1.0) not in test:
            continue
        tst = test[(name, 1.0)]
        val = validation.get((name, 1.0))
        validation_cell = f"{100*float(val['accuracy']):.2f}%" if val else "n/a"
        lines.append(
            f"| {name} | {validation_cell} | "
            f"{100*float(tst['accuracy']):.2f}% [{100*float(tst['accuracy_ci95_low']):.2f}, {100*float(tst['accuracy_ci95_high']):.2f}] | "
            f"{int(tst['teacher_calls'])} | {int(tst['actual_teacher_tokens']):,} | "
            f"{100*(float(tst['accuracy'])-float(sss['accuracy'])):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            f"On the frozen test at full budget, selected v0.4 actions are Helpful/Harmful/No-change = **{helpful}/{harmful}/{wasted}**. These are local shared-prefix intervention effects; final task accuracy is reported separately.",
            "",
            "![Main results](../figures/darc_v04_main_results.png)",
            "",
            "![Training convergence](../figures/darc_v04_training.png)",
            "",
            "## Limitations and failure modes",
            "",
            "- Level-2 has only 381 training tasks and rare changed outcomes; Q-targets can still be noisy and optimistic.",
            "- Budget is represented by five discrete training buckets; replay maps the live allowance to its nearest bucket.",
            "- Fixed sequences and the best-descendant target are diagnostics; neither supplies deployable future knowledge.",
            "- The frozen test was inspected in earlier project phases, so this remains exploratory rather than a pristine confirmatory result.",
            "- Controller compute overhead is not charged as Teacher tokens and is reported separately through cached semantic inference artifacts.",
            "",
            "## Artifacts",
            "",
            "- `tables/policy_results.csv`: all validation/test budget points with task-bootstrap intervals.",
            "- `tables/decision_trace.csv`: observable decision inputs plus audit-only outcomes.",
            "- `tables/model_metrics.csv`: value/risk/cost metrics.",
            "- `models/`: all v0.4 ablations and training histories.",
            "- `selection_manifest.json`: selection fixed before test loading.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text())
    frozen_root = Path(config["data"]["frozen_root"])
    output = Path(config["output"]["root"])
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    models_dir = output / "models"
    cache_dir = Path(config["semantic_encoder"]["cache_dir"])
    seed = int(config["reproducibility"]["seed"])
    device = str(config["semantic_encoder"]["device"])
    fractions = [float(value) for value in config["evaluation"]["budget_fractions"]]

    # Development phase: test is deliberately not loaded here.
    train_groups = _load_groups(frozen_root, "train")
    validation_groups = _load_groups(frozen_root, "validation")
    _assert_disjoint(train_groups, validation_groups)
    audit_rows = audit_level2({"train": train_groups, "validation": validation_groups})
    train_states, _train_lookup = build_states(train_groups)
    validation_states, validation_lookup = build_states(validation_groups)
    train_encoded, train_cache = encode_or_load(
        [state.encoder_example for state in train_states],
        split="v04_train",
        config=config["semantic_encoder"],
        cache_dir=cache_dir,
    )
    validation_encoded, validation_cache = encode_or_load(
        [state.encoder_example for state in validation_states],
        split="v04_validation",
        config=config["semantic_encoder"],
        cache_dir=cache_dir,
    )
    reference_budget = float(
        np.median(
            [
                int(group.rows["VVV"].get("teacher_total_tokens", 0) or 0)
                for group in train_groups.values()
            ]
        )
    )
    allowances = _allowances(
        reference_budget, [float(value) for value in config["targets"]["budget_fractions"]]
    )
    pretrained = DARCEnsemble.load(
        Path(config["data"]["darc_v03_results"]) / "models" / "darc_v03_selected.pt"
    )
    variants: dict[str, dict[str, Any]] = {}
    policy_selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    policy_candidates = _policy_grid(config["policy_grid"])
    for offset, variant in enumerate(config["variants"]):
        name = str(variant["name"])
        counterfactual = bool(variant["counterfactual_targets"])
        train_targets = build_targets(
            train_states,
            train_groups,
            allowances=allowances,
            reference_budget=reference_budget,
            counterfactual=counterfactual,
        )
        validation_targets = build_targets(
            validation_states,
            validation_groups,
            allowances=allowances,
            reference_budget=reference_budget,
            counterfactual=counterfactual,
        )
        model_path = models_dir / f"{name}.pt"
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
            seed=seed + offset * 10007,
            deterministic=bool(config["reproducibility"]["deterministic_torch"]),
            device_name=device,
            description=f"DARC v0.4 {name}",
            progress=config["progress"],
        )
        model.save(model_path)
        predictions = _precompute(
            model, validation_encoded, validation_states, allowances, device=device
        )
        policy, selection = _tune_policy(
            validation_groups,
            validation_states,
            validation_lookup,
            predictions,
            allowances,
            policy_candidates,
            fractions=fractions,
            reference_budget=reference_budget,
            hard_cap=int(config["evaluation"]["hard_call_token_cap"]),
            seed=seed,
            label=f"DARC v0.4 · {name}",
        )
        policy_selection_rows.extend(selection)
        variants[name] = {"model": model, "policy": policy, "predictions": predictions}
        full_prediction = predictions[-1]
        metric_rows.append(
            {"variant": name, "split": "validation", **_effect_metrics(validation_states, full_prediction)}
        )
        for member_index, member in enumerate(model.members, 1):
            for row in member.history:
                training_rows.append({"variant": name, "member": member_index, **row})

    selected_variant = "full_pretrained"
    selected_policy: SequentialPolicy = variants[selected_variant]["policy"]
    selection_manifest = {
        "schema_version": "darc-v0.4-selection-v1",
        "test_loaded": False,
        "selection_source": "train plus validation only",
        "selected_variant": selected_variant,
        "selected_policy": asdict(selected_policy),
        "reference_budget_per_task": reference_budget,
        "allowance_buckets": allowances,
        "train_tasks": len(train_groups),
        "validation_tasks": len(validation_groups),
        "test_tasks": None,
    }
    (output / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2) + "\n"
    )

    validation_replays: list[ReplayResult] = []
    validation_traces: list[dict[str, Any]] = []
    for name, entry in variants.items():
        label = f"DARC v0.4 · {name}"
        for fraction in fractions:
            replay, traces = replay_v04(
                validation_groups,
                validation_states,
                validation_lookup,
                entry["predictions"],
                allowances,
                entry["policy"],
                token_budget=int(round(fraction * reference_budget * len(validation_groups))),
                budget_fraction=fraction,
                hard_call_token_cap=int(config["evaluation"]["hard_call_token_cap"]),
                seed=seed,
                label=label,
            )
            validation_replays.append(replay)
            validation_traces.extend({**row, "split": "validation"} for row in traces)
    validation_replays.extend(
        _baselines(
            validation_groups,
            config,
            fractions=fractions,
            reference_budget=reference_budget,
            split="validation",
        )
    )

    # Locked evaluation phase starts only after the manifest above is durable.
    test_groups = _load_groups(frozen_root, "test")
    _assert_disjoint(train_groups, validation_groups, test_groups)
    audit_rows.extend(audit_level2({"test": test_groups}))
    test_states, test_lookup = build_states(test_groups)
    test_encoded, test_cache = encode_or_load(
        [state.encoder_example for state in test_states],
        split="v04_test",
        config=config["semantic_encoder"],
        cache_dir=cache_dir,
    )
    test_replays: list[ReplayResult] = []
    test_traces: list[dict[str, Any]] = []
    for name, entry in variants.items():
        model: DARCv04Ensemble = entry["model"]
        predictions = _precompute(
            model, test_encoded, test_states, allowances, device=device
        )
        metric_rows.append(
            {"variant": name, "split": "test", **_effect_metrics(test_states, predictions[-1])}
        )
        label = f"DARC v0.4 · {name}"
        for fraction in fractions:
            replay, traces = replay_v04(
                test_groups,
                test_states,
                test_lookup,
                predictions,
                allowances,
                entry["policy"],
                token_budget=int(round(fraction * reference_budget * len(test_groups))),
                budget_fraction=fraction,
                hard_call_token_cap=int(config["evaluation"]["hard_call_token_cap"]),
                seed=seed,
                label=label,
            )
            test_replays.append(replay)
            test_traces.extend({**row, "split": "test"} for row in traces)
    test_replays.extend(
        _baselines(
            test_groups,
            config,
            fractions=fractions,
            reference_budget=reference_budget,
            split="test",
        )
    )
    result_rows = [
        *_summaries(validation_replays, config, "validation"),
        *_summaries(test_replays, config, "test"),
    ]
    all_traces = [*validation_traces, *test_traces]
    _write_csv(tables / "data_audit.csv", audit_rows)
    _write_csv(tables / "policy_selection.csv", policy_selection_rows)
    _write_csv(tables / "model_metrics.csv", metric_rows)
    _write_csv(tables / "training_history.csv", training_rows)
    _write_csv(tables / "policy_results.csv", result_rows)
    _write_csv(tables / "decision_trace.csv", all_traces)
    selection_manifest.update(
        {
            "test_loaded": True,
            "test_tasks": len(test_groups),
            "embedding_cache": {
                "train": train_cache,
                "validation": validation_cache,
                "test": test_cache,
            },
        }
    )
    (output / "evaluation_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2) + "\n"
    )
    selected_label = f"DARC v0.4 · {selected_variant}"
    _plot(output, result_rows, training_rows, selected_label)
    report = _report(
        output,
        config,
        audit_rows,
        metric_rows,
        result_rows,
        selected_variant,
        selected_policy,
        all_traces,
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "darc-v0.4-summary-v1",
                "selected_variant": selected_variant,
                "selected_policy": asdict(selected_policy),
                "report": str(report),
                "policy_results": len(result_rows),
            },
            indent=2,
        )
        + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/darc_v04.yaml"))
    args = parser.parse_args()
    report = run(args.config)
    print(f"DARC v0.4 completed. Report: {report}")


if __name__ == "__main__":
    main()

