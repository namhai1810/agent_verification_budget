"""Validation-only replay of DARC v0.5 with an online one-Verify constraint.

The audit reuses saved Myopic iteration-0 checkpoints.  It never edits an
existing trace: both unrestricted and constrained policies are replayed from
the initial state through the branch-consistent Level-2 environment.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm

from agent_verify.analysis.cbv_v01_experiment import (
    MultiHistoryController,
    ReplayResult,
    _passed,
    _stable_order,
)
from agent_verify.analysis.darc_data import encode_or_load
from agent_verify.analysis.darc_v04_audit import (
    _seed_summary,
    aggregate_records,
    paired_comparison,
)
from agent_verify.analysis.darc_v04_data import build_states
from agent_verify.analysis.darc_v04_experiment import (
    _assert_disjoint,
    _load_groups,
    _write_csv,
)
from agent_verify.analysis.darc_v05_diagnostics import (
    _conditional_verify_gain,
    _prediction_components,
    _selected_policy,
)
from agent_verify.analysis.darc_v05_env import CBVLevel2OfflineEnv, OfflineObservation
from agent_verify.analysis.darc_v05_model import DARCv05Ensemble, V05Prediction
from agent_verify.analysis.darc_v05_training import ConservativePolicy, nearest_prediction

ORIGINAL_LABEL = "DARC v0.5 Myopic i0 (unrestricted)"
SINGLE_LABEL = "DARC v0.5 Myopic i0 (single-Verify)"
C3_LABEL = "C3 MultiHistory Myopic"
SINGLE_SEQUENCES = ("VSS", "SVS", "SSV")

DecisionProvider = Callable[[OfflineObservation], tuple[str, dict[str, Any]]]


def _replay_policy(
    groups: dict[str, Any],
    state_lookup: dict[tuple[str, str], int],
    provider_factory: Callable[[str, int], DecisionProvider],
    *,
    label: str,
    token_budget: int,
    budget_fraction: float,
    hard_call_token_cap: int,
    seed: int,
    max_verifies_per_task: int | None,
) -> tuple[ReplayResult, list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay one policy without future labels or costs in online decisions."""

    keys = _stable_order(groups, seed)
    environment = CBVLevel2OfflineEnv(
        groups,
        state_lookup,
        hard_call_token_cap=hard_call_token_cap,
    )
    actual_spent = 0
    predicted_spent = 0.0
    outcomes: list[float] = []
    baselines: list[float] = []
    masks: list[int] = []
    decision_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for order, task_key in enumerate(keys):
        remaining = max(token_budget - actual_spent, 0)
        allowance = remaining / max(len(keys) - order, 1)
        environment.reset(task_key, remaining, per_task_allowance=allowance)
        provider = provider_factory(task_key, order)
        verifies = 0
        local_rows: list[dict[str, Any]] = []
        while True:
            remaining = max(token_budget - actual_spent, 0)
            allowance = remaining / max(len(keys) - order, 1)
            environment.set_per_task_allowance(allowance)
            observation = environment.observation()
            proposed_action, metadata = provider(observation)
            if proposed_action not in {"S", "V"}:
                raise ValueError(f"Invalid proposed action {proposed_action!r}")
            forced = bool(
                proposed_action == "V"
                and max_verifies_per_task is not None
                and verifies >= max_verifies_per_task
            )
            action = "S" if forced else proposed_action
            prefix = observation.prefix
            transition = environment.step(action)
            actual_spent += transition.incremental_teacher_tokens
            if actual_spent > token_budget:
                raise AssertionError(f"{label} exceeded the strict global budget")
            if action == "V":
                verifies += 1
                predicted_spent += float(metadata.get("predicted_cost", 0.0) or 0.0)
            local_rows.append(
                {
                    "policy": label,
                    "seed": seed,
                    "budget_fraction": budget_fraction,
                    "token_budget": token_budget,
                    "task_key": task_key,
                    "dataset": groups[task_key].task.dataset,
                    "task_order": order,
                    "checkpoint": observation.checkpoint,
                    "prefix_before": prefix or "START",
                    "proposed_action": proposed_action,
                    "action": action,
                    "forced_by_single_verify": int(forced),
                    "verifies_before_action": verifies - int(action == "V"),
                    "remaining_budget_before_action": observation.global_remaining_budget,
                    "per_task_allowance": observation.per_task_allowance,
                    "actual_incremental_tokens": transition.incremental_teacher_tokens,
                    **metadata,
                }
            )
            if transition.done:
                break
        sequence = environment.prefix
        outcome = int(_passed(groups[task_key].rows[sequence]))
        baseline = int(_passed(groups[task_key].rows["SSS"]))
        task_tokens = sum(int(row["actual_incremental_tokens"]) for row in local_rows)
        helpful = harmful = no_change = 0
        for row in local_rows:
            gain = _conditional_verify_gain(groups[task_key], sequence, int(row["checkpoint"]))
            row.update(
                {
                    "final_sequence": sequence,
                    "final_correctness": outcome,
                    "baseline_correctness": baseline,
                    "conditional_verify_gain_same_suffix_audit": gain,
                    "helpful_selected": int(row["action"] == "V" and gain > 0),
                    "harmful_selected": int(row["action"] == "V" and gain < 0),
                    "no_change_selected": int(row["action"] == "V" and gain == 0),
                }
            )
            helpful += int(row["helpful_selected"])
            harmful += int(row["harmful_selected"])
            no_change += int(row["no_change_selected"])
        decision_rows.extend(local_rows)
        task_rows.append(
            {
                "policy": label,
                "seed": seed,
                "budget_fraction": budget_fraction,
                "token_budget": token_budget,
                "task_key": task_key,
                "dataset": groups[task_key].task.dataset,
                "task_order": order,
                "sequence": sequence,
                "final_correctness": outcome,
                "baseline_correctness": baseline,
                "teacher_tokens": task_tokens,
                "teacher_calls": sequence.count("V"),
                "helpful_calls": helpful,
                "harmful_calls": harmful,
                "no_change_calls": no_change,
            }
        )
        outcomes.append(float(outcome))
        baselines.append(float(baseline))
        masks.append(sum(1 << index for index, action in enumerate(sequence) if action == "V"))
    return (
        ReplayResult(
            label,
            "darc-v05-single-verify-audit",
            "combined",
            "combined",
            budget_fraction,
            token_budget,
            tuple(keys),
            np.asarray(outcomes),
            np.asarray(baselines),
            np.asarray(masks),
            sum(int(row["teacher_calls"]) for row in task_rows),
            actual_spent,
            predicted_spent,
        ),
        task_rows,
        decision_rows,
    )


def replay_darc(
    groups: dict[str, Any],
    state_lookup: dict[tuple[str, str], int],
    predictions: list[V05Prediction],
    allowances: list[float],
    policy: ConservativePolicy,
    *,
    reference_budget: float,
    token_budget: int,
    budget_fraction: float,
    hard_call_token_cap: int,
    seed: int,
    single_verify: bool,
) -> tuple[ReplayResult, list[dict[str, Any]], list[dict[str, Any]]]:
    def factory(_task_key: str, _order: int) -> DecisionProvider:
        def provider(observation: OfflineObservation) -> tuple[str, dict[str, Any]]:
            prediction = nearest_prediction(
                predictions,
                allowances,
                observation.per_task_allowance,
            )
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
            return action, {**components, "predicted_cost": components["immediate_cost"]}

        return provider

    return _replay_policy(
        groups,
        state_lookup,
        factory,
        label=SINGLE_LABEL if single_verify else ORIGINAL_LABEL,
        token_budget=token_budget,
        budget_fraction=budget_fraction,
        hard_call_token_cap=hard_call_token_cap,
        seed=seed,
        max_verifies_per_task=1 if single_verify else None,
    )


def replay_c3(
    groups: dict[str, Any],
    state_lookup: dict[tuple[str, str], int],
    controller: MultiHistoryController,
    *,
    token_budget: int,
    budget_fraction: float,
    hard_call_token_cap: int,
    seed: int,
) -> tuple[ReplayResult, list[dict[str, Any]], list[dict[str, Any]]]:
    def factory(task_key: str, _order: int) -> DecisionProvider:
        group = groups[task_key]

        def provider(observation: OfflineObservation) -> tuple[str, dict[str, Any]]:
            q_skip, q_verify, predicted_cost = controller.predict(
                group,
                observation.prefix,
                observation.per_task_allowance,
            )
            predicted_cost = max(float(predicted_cost), 1.0)
            feasible = (
                observation.global_remaining_budget >= hard_call_token_cap
                and predicted_cost <= observation.global_remaining_budget
            )
            action = "V" if float(q_verify) > float(q_skip) and feasible else "S"
            return action, {
                "q_skip": float(q_skip),
                "q_verify": float(q_verify),
                "advantage": float(q_verify) - float(q_skip),
                "predicted_cost": predicted_cost,
                "feasible": feasible,
            }

        return provider

    return _replay_policy(
        groups,
        state_lookup,
        factory,
        label=C3_LABEL,
        token_budget=token_budget,
        budget_fraction=budget_fraction,
        hard_call_token_cap=hard_call_token_cap,
        seed=seed,
        max_verifies_per_task=None,
    )


def _task_comparisons(
    task_rows: list[dict[str, Any]],
    groups: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {
        (
            str(row["policy"]),
            int(row["seed"]),
            int(row["token_budget"]),
            str(row["task_key"]),
        ): row
        for row in task_rows
    }
    comparisons: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    budgets = sorted({int(row["token_budget"]) for row in task_rows})
    seeds = sorted({int(row["seed"]) for row in task_rows})
    task_keys = sorted(groups)
    for budget in budgets:
        for seed in seeds:
            for task_key in task_keys:
                original = indexed[(ORIGINAL_LABEL, seed, budget, task_key)]
                single = indexed[(SINGLE_LABEL, seed, budget, task_key)]
                correct_delta = int(single["final_correctness"]) - int(original["final_correctness"])
                successful_singles = [
                    sequence
                    for sequence in SINGLE_SEQUENCES
                    if int(_passed(groups[task_key].rows[sequence])) == 1
                ]
                cheapest_single = (
                    min(
                        successful_singles,
                        key=lambda sequence: int(
                            groups[task_key].rows[sequence].get("teacher_total_tokens", 0) or 0
                        ),
                    )
                    if successful_singles
                    else ""
                )
                row = {
                    "seed": seed,
                    "token_budget": budget,
                    "task_key": task_key,
                    "dataset": original["dataset"],
                    "original_sequence": original["sequence"],
                    "single_verify_sequence": single["sequence"],
                    "action_sequence_changed": int(original["sequence"] != single["sequence"]),
                    "checkpoint_action_changes": sum(
                        left != right
                        for left, right in zip(
                            str(original["sequence"]),
                            str(single["sequence"]),
                            strict=True,
                        )
                    ),
                    "original_correctness": original["final_correctness"],
                    "single_verify_correctness": single["final_correctness"],
                    "correctness_delta": correct_delta,
                    "single_verify_win": int(correct_delta > 0),
                    "single_verify_loss": int(correct_delta < 0),
                    "original_tokens": original["teacher_tokens"],
                    "single_verify_tokens": single["teacher_tokens"],
                    "token_savings": int(original["teacher_tokens"]) - int(single["teacher_tokens"]),
                    "original_calls": original["teacher_calls"],
                    "single_verify_calls": single["teacher_calls"],
                    "successful_single_verify_sequences": ",".join(successful_singles),
                    "another_single_verify_can_solve": int(bool(successful_singles)),
                    "cheapest_successful_single_verify": cheapest_single,
                }
                comparisons.append(row)
                if correct_delta < 0:
                    losses.append(row)
    return comparisons, losses


def _comparison_summary(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[(int(row["seed"]), int(row["token_budget"]))].append(row)
    return [
        {
            "seed": seed,
            "token_budget": budget,
            "tasks": len(rows),
            "action_sequence_changes": sum(int(row["action_sequence_changed"]) for row in rows),
            "checkpoint_action_changes": sum(int(row["checkpoint_action_changes"]) for row in rows),
            "single_verify_wins": sum(int(row["single_verify_win"]) for row in rows),
            "single_verify_losses": sum(int(row["single_verify_loss"]) for row in rows),
            "single_verify_ties": sum(int(row["correctness_delta"] == 0) for row in rows),
            "token_savings": sum(int(row["token_savings"]) for row in rows),
        }
        for (seed, budget), rows in sorted(grouped.items())
    ]


def _pareto_rows(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in aggregate:
        tokens = float(candidate["teacher_tokens_mean"])
        accuracy = float(candidate["accuracy_mean"])
        dominators = [
            row
            for row in aggregate
            if float(row["teacher_tokens_mean"]) <= tokens + 1e-9
            and float(row["accuracy_mean"]) >= accuracy - 1e-12
            and (
                float(row["teacher_tokens_mean"]) < tokens - 1e-9
                or float(row["accuracy_mean"]) > accuracy + 1e-12
            )
        ]
        rows.append(
            {
                **candidate,
                "pareto_nondominated": int(not dominators),
                "dominated_by": ",".join(
                    sorted({f"{row['policy']}@{float(row['budget_fraction']):.2f}" for row in dominators})
                ),
            }
        )
    return rows


def _plots(output: Path, aggregate: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    figure_root = output / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    colors = {
        ORIGINAL_LABEL: "#4c78a8",
        SINGLE_LABEL: "#59a14f",
        C3_LABEL: "#f28e2b",
    }
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    for policy in (ORIGINAL_LABEL, SINGLE_LABEL, C3_LABEL):
        selected = sorted(
            [row for row in aggregate if row["policy"] == policy],
            key=lambda row: float(row["teacher_tokens_mean"]),
        )
        ax.plot(
            [float(row["teacher_tokens_mean"]) / 1000 for row in selected],
            [100 * float(row["accuracy_mean"]) for row in selected],
            marker="o",
            linewidth=2.2,
            label=policy,
            color=colors[policy],
        )
    ax.set(
        title="Single-Verify constrained accuracy–token frontier",
        xlabel="Actual Teacher tokens (K; mean across seeds)",
        ylabel="Validation accuracy (%)",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_root / "accuracy_token_frontier.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    grouped = _comparison_summary(comparisons)
    budgets = sorted({int(row["token_budget"]) for row in grouped})
    savings = [
        float(np.mean([row["token_savings"] for row in grouped if int(row["token_budget"]) == budget])) / 1000
        for budget in budgets
    ]
    losses = [
        float(np.mean([row["single_verify_losses"] for row in grouped if int(row["token_budget"]) == budget]))
        for budget in budgets
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    labels = [f"{budget / 1000:.0f}K" for budget in budgets]
    axes[0].bar(labels, savings, color="#59a14f")
    axes[0].set(title="Token savings vs unrestricted", ylabel="Teacher tokens saved (K)")
    axes[1].bar(labels, losses, color="#e15759")
    axes[1].set(title="Task losses from constraint", ylabel="Mean tasks across seeds")
    fig.tight_layout()
    fig.savefig(figure_root / "constraint_effects.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output: Path,
    aggregate: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    comparison_summary: list[dict[str, Any]],
    loss_rows: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
) -> Path:
    lines = [
        "# DARC v0.5 — Single-Verify-Constrained Policy Evaluation",
        "",
        "> Validation-only, branch-consistent offline replay. No retraining, frozen-test "
        "access, or LLM inference.",
        "",
        "## Aggregate matched-budget results",
        "",
        "| Budget cap | Policy | Accuracy mean | Accuracy variance | Actual tokens mean | "
        "Calls mean | Helpful / Harmful / No-change |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        aggregate,
        key=lambda item: (float(item["budget_fraction"]), str(item["policy"])),
    ):
        lines.append(
            f"| {int(row['token_budget']):,} | {row['policy']} | "
            f"{100 * float(row['accuracy_mean']):.2f}% | "
            f"{float(row['accuracy_variance']):.6f} | "
            f"{float(row['teacher_tokens_mean']):,.0f} | "
            f"{float(row['teacher_calls_mean']):.1f} | "
            f"{float(row['helpful_mean']):.1f} / {float(row['harmful_mean']):.1f} / "
            f"{float(row['no_change_mean']):.1f} |"
        )
    lines.extend(
        [
            "",
            "![Accuracy–token frontier](../figures/accuracy_token_frontier.png)",
            "",
            "## Per-seed matched-budget results",
            "",
            "| Budget cap | Seed | Policy | Accuracy | Actual tokens | Calls | "
            "Helpful / Harmful / No-change |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        seed_rows,
        key=lambda item: (int(item["token_budget"]), int(item["seed"]), str(item["policy"])),
    ):
        lines.append(
            f"| {int(row['token_budget']):,} | {row['seed']} | {row['policy']} | "
            f"{100 * float(row['accuracy']):.2f}% | {int(row['teacher_tokens']):,} | "
            f"{int(row['teacher_calls'])} | {int(row['helpful'])} / {int(row['harmful'])} / "
            f"{int(row['no_change'])} |"
        )
    lines.extend(
        [
            "",
            "## Accuracy–token Pareto set",
            "",
            "Only non-dominated mean points are shown; a point is dominated when another policy "
            "uses no more tokens and reaches no lower accuracy.",
            "",
            "| Budget cap | Policy | Accuracy mean | Actual tokens mean |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in sorted(
        (item for item in pareto if int(item["pareto_nondominated"]) == 1),
        key=lambda item: (float(item["teacher_tokens_mean"]), -float(item["accuracy_mean"])),
    ):
        lines.append(
            f"| {int(row['token_budget']):,} | {row['policy']} | "
            f"{100 * float(row['accuracy_mean']):.2f}% | "
            f"{float(row['teacher_tokens_mean']):,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed constrained-vs-unrestricted effects",
            "",
            "| Budget cap | Seed | Tasks changed | CP actions changed | "
            "Wins / losses / ties | Token savings |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_summary:
        lines.append(
            f"| {int(row['token_budget']):,} | {row['seed']} | "
            f"{row['action_sequence_changes']} | {row['checkpoint_action_changes']} | "
            f"{row['single_verify_wins']} / "
            f"{row['single_verify_losses']} / {row['single_verify_ties']} | "
            f"{int(row['token_savings']):,} |"
        )
    lines.extend(
        [
            "",
            "![Constraint effects](../figures/constraint_effects.png)",
            "",
            "## Correctness losses",
            "",
            "| Budget | Seed | Task | Original → constrained | Tokens saved | Other successful single path |",
            "|---:|---:|---|---|---:|---|",
        ]
    )
    if loss_rows:
        for row in loss_rows:
            lines.append(
                f"| {int(row['token_budget']):,} | {row['seed']} | {row['task_key']} | "
                f"{row['original_sequence']} → {row['single_verify_sequence']} | "
                f"{int(row['token_savings']):,} | "
                f"{row['successful_single_verify_sequences'] or 'none'} |"
            )
    else:
        lines.append("| — | — | No task losses | — | — | — |")
    lines.extend(
        [
            "",
            "## Protocol notes",
            "",
            "- Saved DARC v0.5 Myopic iteration-0 checkpoints and selected policies are reused.",
            "- After the first realized Verify, later Verify proposals are forced to Skip online.",
            "- The environment follows the resulting prefix and retrieves branch-consistent artifacts.",
            "- All methods share task order, absolute token cap, 8,192-token hard-call reserve, "
            "and post-action actual-cost ledger within each seed/budget run.",
            "- Future outcomes and actual future costs are used only after replay for diagnostics.",
            "",
            "## Artifacts",
            "",
            "- `tables/seed_results.csv` and `aggregate_results.csv`",
            "- `tables/task_results.csv` and `single_vs_unrestricted_tasks.csv`",
            "- `tables/single_vs_unrestricted_summary.csv` and `loss_case_analysis.csv`",
            "- `tables/decision_trace.csv`",
            "- `tables/paired_policy_comparisons.csv` and `pareto_frontier.csv`",
            "- `single_verify_audit_manifest.json`",
        ]
    )
    path = output / "report" / "darc_v05_single_verify_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_single_verify_audit(
    config_path: Path,
    *,
    artifact_root: Path,
    output: Path,
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["evaluation"]["partition"] != "validation":
        raise ValueError("Single-Verify audit must remain validation-only")
    if config["evaluation"]["frozen_test_access"] != "forbidden":
        raise ValueError("Frozen-test access must remain forbidden")
    manifest = json.loads((artifact_root / "selection_manifest.json").read_text(encoding="utf-8"))
    if bool(manifest.get("frozen_test_loaded")):
        raise ValueError("Refusing artifacts whose manifest loaded frozen test")
    if manifest.get("selected_candidate") != "myopic_targets@i0":
        raise ValueError("Expected saved Myopic iteration-0 candidate")

    frozen_root = Path(config["data"]["frozen_root"])
    train_groups = _load_groups(frozen_root, "train")
    groups = _load_groups(frozen_root, "validation")
    _assert_disjoint(train_groups, groups)
    del train_groups
    if len(groups) != 98:
        raise ValueError(f"Expected 98 validation tasks, found {len(groups)}")
    states, state_lookup = build_states(groups)
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
    budgets = [int(round(fraction * reference_budget * len(groups))) for fraction in fractions]
    hard_cap = int(config["offline_rl"]["hard_call_token_cap"])
    device = str(config["semantic_encoder"]["device"])
    c3: MultiHistoryController = joblib.load(
        Path(config["data"]["c3_results"]) / "models" / "combined" / "multi_myopic.joblib"
    )

    records: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(seeds) * len(fractions),
        desc="Single-Verify audit",
        unit="replay set",
    )
    for seed in seeds:
        model = DARCv05Ensemble.load(
            artifact_root / "checkpoints" / "myopic_targets" / f"seed_{seed}" / "iteration_0.pt"
        )
        predictions = [
            model.predict(encoded, states, allowance=value, device_name=device) for value in allowances
        ]
        policy = _selected_policy(
            artifact_root,
            seed=seed,
            variant="myopic_targets",
            iteration=0,
        )
        for fraction, budget in zip(fractions, budgets, strict=True):
            replay_set = [
                replay_darc(
                    groups,
                    state_lookup,
                    predictions,
                    allowances,
                    policy,
                    reference_budget=reference_budget,
                    token_budget=budget,
                    budget_fraction=fraction,
                    hard_call_token_cap=hard_cap,
                    seed=seed,
                    single_verify=False,
                ),
                replay_darc(
                    groups,
                    state_lookup,
                    predictions,
                    allowances,
                    policy,
                    reference_budget=reference_budget,
                    token_budget=budget,
                    budget_fraction=fraction,
                    hard_call_token_cap=hard_cap,
                    seed=seed,
                    single_verify=True,
                ),
                replay_c3(
                    groups,
                    state_lookup,
                    c3,
                    token_budget=budget,
                    budget_fraction=fraction,
                    hard_call_token_cap=hard_cap,
                    seed=seed,
                ),
            ]
            for replay, tasks, decisions in replay_set:
                summary = _seed_summary(replay, decisions, seed=seed)
                summary["token_budget"] = budget
                records.append(
                    {
                        "seed": seed,
                        "policy": replay.policy,
                        "fraction": fraction,
                        "replay": replay,
                        "summary": summary,
                    }
                )
                seed_rows.append(summary)
                task_rows.extend(tasks)
                decision_rows.extend(decisions)
            progress.update(1)
    progress.close()

    bootstrap = int(config["reproducibility"]["bootstrap_repetitions"])
    permutation = int(config["reproducibility"]["permutation_repetitions"])
    statistical_seed = int(config["reproducibility"]["crossfit_seed"])
    aggregate = aggregate_records(records, bootstrap=bootstrap, seed=statistical_seed)
    budget_lookup = dict(zip(fractions, budgets, strict=True))
    for row in aggregate:
        row["token_budget"] = budget_lookup[float(row["budget_fraction"])]
    comparisons: list[dict[str, Any]] = []
    for fraction in fractions:
        for candidate, reference in (
            (SINGLE_LABEL, ORIGINAL_LABEL),
            (SINGLE_LABEL, C3_LABEL),
            (ORIGINAL_LABEL, C3_LABEL),
        ):
            row = paired_comparison(
                records,
                candidate=candidate,
                reference=reference,
                fraction=fraction,
                bootstrap=bootstrap,
                permutations=permutation,
                seed=statistical_seed,
            )
            row["token_budget"] = budget_lookup[fraction]
            comparisons.append(row)

    task_comparisons, loss_rows = _task_comparisons(task_rows, groups)
    comparison_summary = _comparison_summary(task_comparisons)
    pareto = _pareto_rows(aggregate)
    _plots(output, aggregate, task_comparisons)

    table_root = output / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    _write_csv(table_root / "seed_results.csv", seed_rows)
    _write_csv(table_root / "aggregate_results.csv", aggregate)
    _write_csv(table_root / "task_results.csv", task_rows)
    _write_csv(table_root / "decision_trace.csv", decision_rows)
    _write_csv(table_root / "single_vs_unrestricted_tasks.csv", task_comparisons)
    _write_csv(table_root / "single_vs_unrestricted_summary.csv", comparison_summary)
    _write_csv(table_root / "loss_case_analysis.csv", loss_rows)
    _write_csv(table_root / "paired_policy_comparisons.csv", comparisons)
    _write_csv(table_root / "pareto_frontier.csv", pareto)
    report = _write_report(output, aggregate, seed_rows, comparison_summary, loss_rows, pareto)
    audit_manifest = {
        "schema_version": "darc-v05-single-verify-audit-v1",
        "config": str(config_path),
        "source_artifacts": str(artifact_root),
        "partition_loaded": "validation",
        "frozen_test_loaded": False,
        "tasks": len(groups),
        "seeds": seeds,
        "budget_fractions": fractions,
        "token_budgets": budgets,
        "hard_call_token_cap": hard_cap,
        "darc_checkpoint": "myopic_targets/iteration_0",
        "constraint": "at most one realized Verify per task; later Verify proposals force Skip",
        "embedding_cache": cache_manifest,
        "report": str(report),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "single_verify_audit_manifest.json").write_text(
        json.dumps(audit_manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report
