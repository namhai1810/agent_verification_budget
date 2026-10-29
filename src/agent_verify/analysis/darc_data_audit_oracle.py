"""Offline DARC data audit and observed-branch oracle upper-bound analysis.

This module is intentionally post-hoc.  The oracle may inspect paired outcomes and
actual Teacher costs, so it is an analysis upper bound and is never a deployable
policy or a source of inference-time features.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from agent_verify.analysis.darc_experiment import _load_split
from agent_verify.analysis.joint_value_cost_data import VALUE_TO_INDEX, value_label


SPLITS = ("train", "validation", "test")
CHECKPOINTS = (1, 2, 3)
LABELS = ("Helpful", "Harmful", "No-change")
LABEL_COLORS = {
    "Helpful": "#168A5B",
    "Harmful": "#D84A4A",
    "No-change": "#CBD2D9",
}
DATASET_NAMES = {
    "humaneval": "HumanEval",
    "livecodebench": "LiveCodeBench",
    "bigcodebench": "BigCodeBench",
}


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _label(delta: int) -> str:
    if delta > 0:
        return "Helpful"
    if delta < 0:
        return "Harmful"
    return "No-change"


def _group(examples: Iterable[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in examples:
        grouped[row.group_key].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row.checkpoint)
    return dict(grouped)


def _audit_integrity(partitions: dict[str, list[Any]]) -> dict[str, Any]:
    state_keys: set[str] = set()
    group_sets: dict[str, set[str]] = {}
    counters = Counter()
    for split, examples in partitions.items():
        groups = _group(examples)
        group_sets[split] = set(groups)
        counters["states"] += len(examples)
        counters["tasks"] += len(groups)
        for row in examples:
            counters["delta_checked"] += 1
            counters["delta_mismatch"] += int(row.delta != row.y_verify - row.y_skip)
            counters["label_checked"] += 1
            counters["label_mismatch"] += int(
                row.value_label != value_label(row.y_skip, row.y_verify)
            )
            counters["non_binary_outcome"] += int(
                row.y_skip not in (0, 1) or row.y_verify not in (0, 1)
            )
            counters["negative_cost"] += int(row.teacher_total_tokens < 0)
            counters["duplicate_state"] += int(row.state_key in state_keys)
            state_keys.add(row.state_key)
        for rows in groups.values():
            counters["incomplete_task"] += int(
                [row.checkpoint for row in rows] != [1, 2, 3]
            )
            counters["baseline_disagreement"] += int(
                len({row.y_skip for row in rows}) != 1
            )
    overlap = {
        f"{left}_{right}": len(group_sets[left] & group_sets[right])
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    }
    failures = (
        counters["delta_mismatch"]
        + counters["label_mismatch"]
        + counters["non_binary_outcome"]
        + counters["negative_cost"]
        + counters["duplicate_state"]
        + counters["incomplete_task"]
        + counters["baseline_disagreement"]
        + sum(overlap.values())
    )
    return {
        **dict(counters),
        "split_overlap": overlap,
        "checks_passed": failures == 0,
    }


def _label_rows(partitions: dict[str, list[Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        examples = partitions[split]
        for checkpoint in CHECKPOINTS:
            subset = [row for row in examples if row.checkpoint == checkpoint]
            counts = Counter(_label(row.delta) for row in subset)
            total = len(subset)
            rows.append(
                {
                    "split": split,
                    "checkpoint": checkpoint,
                    "states": total,
                    "helpful": counts["Helpful"],
                    "harmful": counts["Harmful"],
                    "no_change": counts["No-change"],
                    "helpful_rate": counts["Helpful"] / total,
                    "harmful_rate": counts["Harmful"] / total,
                    "no_change_rate": counts["No-change"] / total,
                    "mean_actual_delta": float(np.mean([row.delta for row in subset])),
                    "mean_teacher_tokens": float(
                        np.mean([row.teacher_total_tokens for row in subset])
                    ),
                }
            )
    return rows


def _gain_rows(partitions: dict[str, list[Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        examples = partitions[split]
        datasets = sorted({row.dataset for row in examples})
        for dataset in (*datasets, "combined"):
            for checkpoint in CHECKPOINTS:
                subset = [
                    row
                    for row in examples
                    if row.checkpoint == checkpoint
                    and (dataset == "combined" or row.dataset == dataset)
                ]
                if not subset:
                    continue
                helpful = sum(row.delta > 0 for row in subset)
                harmful = sum(row.delta < 0 for row in subset)
                rows.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "checkpoint": checkpoint,
                        "states": len(subset),
                        "helpful": helpful,
                        "harmful": harmful,
                        "no_change": len(subset) - helpful - harmful,
                        "mean_actual_delta": float(
                            np.mean([row.delta for row in subset])
                        ),
                        "mean_teacher_tokens": float(
                            np.mean([row.teacher_total_tokens for row in subset])
                        ),
                    }
                )
    return rows


def _task_rows(examples: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_key, states in sorted(_group(examples).items()):
        positive = [state for state in states if state.delta > 0]
        harmful = [state for state in states if state.delta < 0]
        cheapest = min(
            positive,
            key=lambda state: (state.teacher_total_tokens, state.checkpoint),
            default=None,
        )
        rows.append(
            {
                "group_key": group_key,
                "dataset": states[0].dataset,
                "task_id": states[0].task_id,
                "baseline_correct": states[0].y_skip,
                "has_helpful": int(bool(positive)),
                "has_harmful": int(bool(harmful)),
                "helpful_checkpoints": ",".join(
                    str(state.checkpoint) for state in positive
                ),
                "harmful_checkpoints": ",".join(
                    str(state.checkpoint) for state in harmful
                ),
                "oracle_checkpoint": cheapest.checkpoint if cheapest else 0,
                "oracle_actual_tokens": cheapest.teacher_total_tokens if cheapest else 0,
            }
        )
    return rows


def _oracle_at_budget(
    task_rows: list[dict[str, Any]], budget: int
) -> dict[str, Any]:
    baseline_correct = sum(int(row["baseline_correct"]) for row in task_rows)
    candidates = sorted(
        (row for row in task_rows if int(row["has_helpful"])),
        key=lambda row: (
            int(row["oracle_actual_tokens"]),
            int(row["oracle_checkpoint"]),
            str(row["group_key"]),
        ),
    )
    spent = 0
    selected: list[dict[str, Any]] = []
    for row in candidates:
        cost = int(row["oracle_actual_tokens"])
        if spent + cost > budget:
            continue
        spent += cost
        selected.append(row)
    correct = baseline_correct + len(selected)
    return {
        "budget": int(budget),
        "actual_teacher_tokens": spent,
        "teacher_calls": len(selected),
        "correct": correct,
        "tasks": len(task_rows),
        "accuracy": correct / len(task_rows),
        "gain_vs_never_pp": 100.0 * len(selected) / len(task_rows),
        "selected_task_keys": [str(row["group_key"]) for row in selected],
    }


def _oracle_frontier(
    task_rows: list[dict[str, Any]], max_budget: int
) -> list[dict[str, Any]]:
    candidates = sorted(
        (row for row in task_rows if int(row["has_helpful"])),
        key=lambda row: (
            int(row["oracle_actual_tokens"]),
            int(row["oracle_checkpoint"]),
            str(row["group_key"]),
        ),
    )
    budgets = {0, max_budget}
    cumulative = 0
    for row in candidates:
        cumulative += int(row["oracle_actual_tokens"])
        budgets.add(cumulative)
    return [
        {key: value for key, value in _oracle_at_budget(task_rows, budget).items()
         if key != "selected_task_keys"}
        for budget in sorted(budgets)
    ]


def _darc_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in _read_csv(path)
        if row["policy"] == "DARC v0.2 Full"
    ]
    rows.sort(key=lambda row: float(row["budget_fraction"]))
    return [
        {
            "budget_fraction": float(row["budget_fraction"]),
            "nominal_budget": int(row["token_budget"]),
            "actual_teacher_tokens": int(row["actual_teacher_tokens"]),
            "teacher_calls": int(row["teacher_calls"]),
            "accuracy": float(row["accuracy"]),
            "correct": round(float(row["accuracy"]) * int(row["tasks"])),
            "tasks": int(row["tasks"]),
        }
        for row in rows
    ]


def _oracle_comparison(
    task_rows: list[dict[str, Any]], darc_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for row in darc_rows:
        oracle_actual = _oracle_at_budget(
            task_rows, int(row["actual_teacher_tokens"])
        )
        oracle_nominal = _oracle_at_budget(task_rows, int(row["nominal_budget"]))
        comparisons.append(
            {
                **row,
                "oracle_accuracy_same_actual_spend": oracle_actual["accuracy"],
                "oracle_correct_same_actual_spend": oracle_actual["correct"],
                "oracle_calls_same_actual_spend": oracle_actual["teacher_calls"],
                "oracle_tokens_same_actual_spend": oracle_actual["actual_teacher_tokens"],
                "oracle_accuracy_nominal_budget": oracle_nominal["accuracy"],
                "selection_gap_pp": 100.0
                * (oracle_actual["accuracy"] - float(row["accuracy"])),
            }
        )
    return comparisons


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFC",
            "savefig.facecolor": "white",
        }
    )


def _plot_label_distribution(rows: list[dict[str, Any]], output: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), sharey=True)
    for axis, split in zip(axes, SPLITS, strict=True):
        subset = [row for row in rows if row["split"] == split]
        bottoms = np.zeros(3)
        x = np.arange(3)
        for label, field in (
            ("No-change", "no_change_rate"),
            ("Helpful", "helpful_rate"),
            ("Harmful", "harmful_rate"),
        ):
            values = np.asarray([float(row[field]) for row in subset])
            axis.bar(
                x,
                100 * values,
                bottom=100 * bottoms,
                color=LABEL_COLORS[label],
                width=0.66,
                label=label,
                edgecolor="white",
                linewidth=0.7,
            )
            if label != "No-change":
                for index, value in enumerate(values):
                    if value > 0:
                        axis.text(
                            index,
                            100 * (bottoms[index] + value / 2),
                            f"{100*value:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=8.5,
                            color="white",
                            fontweight="bold",
                        )
            bottoms += values
        for index, row in enumerate(subset):
            axis.text(
                index,
                50,
                f"No-change\n{100*float(row['no_change_rate']):.1f}%",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#354052",
            )
        axis.set_title(f"{split.title()}  ·  n={subset[0]['states']} tasks")
        axis.set_xticks(x, ["CP1\nPlanner", "CP2\nCoder", "CP3\nReviewer"])
        axis.set_ylim(0, 100)
        axis.grid(axis="y", color="#E6EAF0", linewidth=0.7)
    axes[0].set_ylabel("Share of paired interventions (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    order = [labels.index(name) for name in LABELS]
    fig.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "CBV v0.1 label audit: intervention value is severely imbalanced",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_gain_audit(
    label_rows: list[dict[str, Any]],
    gain_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    _style()
    fig = plt.figure(figsize=(16.4, 5.2))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.08, 1.12, 1.2), wspace=0.52)
    axis = fig.add_subplot(grid[0, 0])
    test = [row for row in label_rows if row["split"] == "test"]
    y = np.arange(3)
    helpful = 100 * np.asarray([float(row["helpful_rate"]) for row in test])
    harmful = -100 * np.asarray([float(row["harmful_rate"]) for row in test])
    axis.barh(y, helpful, color=LABEL_COLORS["Helpful"], height=0.52, label="Helpful")
    axis.barh(y, harmful, color=LABEL_COLORS["Harmful"], height=0.52, label="Harmful")
    axis.axvline(0, color="#354052", linewidth=0.8)
    for index, row in enumerate(test):
        axis.text(
            0.97,
            index,
            f"net {100*float(row['mean_actual_delta']):+.2f} pp",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
    axis.set_yticks(y, ["CP1", "CP2", "CP3"])
    axis.invert_yaxis()
    axis.set_xlabel("Observed pair rate (%)")
    axis.set_title("a  Actual downstream gain\n(frozen test)", loc="left", fontweight="bold")
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )
    axis.grid(axis="x", color="#E6EAF0", linewidth=0.7)

    heat_axis = fig.add_subplot(grid[0, 1])
    datasets = ("humaneval", "livecodebench", "bigcodebench")
    matrix = np.zeros((len(datasets), 3))
    for i, dataset in enumerate(datasets):
        for j, checkpoint in enumerate(CHECKPOINTS):
            match = next(
                row
                for row in gain_rows
                if row["split"] == "test"
                and row["dataset"] == dataset
                and row["checkpoint"] == checkpoint
            )
            matrix[i, j] = 100 * float(match["mean_actual_delta"])
    limit = max(1.0, float(np.max(np.abs(matrix))))
    image = heat_axis.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            heat_axis.text(
                j,
                i,
                f"{matrix[i,j]:+.1f} pp",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#17202A",
            )
    heat_axis.set_xticks(range(3), ["CP1", "CP2", "CP3"])
    heat_axis.set_yticks(
        range(3), [DATASET_NAMES[dataset] for dataset in datasets]
    )
    heat_axis.set_title("b  Net gain by dataset", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean Δ (percentage points)")

    opportunity_axis = fig.add_subplot(grid[0, 2])
    already = sum(int(row["baseline_correct"]) for row in task_rows)
    rescuable = sum(int(row["has_helpful"]) for row in task_rows)
    unavailable = len(task_rows) - already - rescuable
    values = (already, rescuable, unavailable)
    colors = ("#4C78A8", LABEL_COLORS["Helpful"], "#D8DDE5")
    names = ("Already correct", "Observed rescue", "No observed rescue")
    left = 0
    for value, color, name in zip(values, colors, names, strict=True):
        opportunity_axis.barh([0], [value], left=left, color=color, height=0.55, label=name)
        opportunity_axis.text(
            left + value / 2,
            0,
            f"{value}\n({100*value/len(task_rows):.1f}%)",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white" if color != "#D8DDE5" else "#354052",
        )
        left += value
    opportunity_axis.set_xlim(0, len(task_rows))
    opportunity_axis.set_yticks([])
    opportunity_axis.set_xlabel("Frozen-test tasks")
    opportunity_axis.set_title("c  Task-level opportunity", loc="left", fontweight="bold")
    opportunity_axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=8.5,
    )
    opportunity_axis.text(
        len(task_rows) / 2,
        -0.55,
        "Only 19/216 tasks can be improved by any observed single-VERIFY branch",
        ha="center",
        va="center",
        fontsize=9,
        color="#354052",
    )
    fig.suptitle(
        "Observed counterfactual gain and available improvement headroom",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_oracle(
    frontier: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    failure_csv: Path,
    output: Path,
) -> None:
    _style()
    full = max(comparisons, key=lambda row: float(row["budget_fraction"]))
    failure_rows = _read_csv(failure_csv)
    failure = next(
        row
        for row in failure_rows
        if row["policy"] == "DARC v0.2 Full"
        and float(row["budget_fraction"]) == 1.0
    )
    baseline_correct = sum(int(row["baseline_correct"]) for row in task_rows)
    oracle_correct = baseline_correct + sum(int(row["has_helpful"]) for row in task_rows)
    tasks = len(task_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.2), gridspec_kw={"width_ratios": [1.45, 1]})

    axis = axes[0]
    x_oracle = np.asarray([int(row["actual_teacher_tokens"]) for row in frontier]) / 1000
    y_oracle = 100 * np.asarray([float(row["accuracy"]) for row in frontier])
    x_darc = np.asarray([int(row["actual_teacher_tokens"]) for row in comparisons]) / 1000
    y_darc = 100 * np.asarray([float(row["accuracy"]) for row in comparisons])
    axis.step(
        x_oracle,
        y_oracle,
        where="post",
        color="#7353BA",
        linewidth=3,
        label="Observed-branch Oracle",
    )
    axis.plot(
        x_darc,
        y_darc,
        "o-",
        color="#E45756",
        linewidth=2.2,
        markersize=7,
        label="DARC v0.2",
    )
    baseline_accuracy = 100 * baseline_correct / tasks
    axis.axhline(
        baseline_accuracy,
        color="#56616F",
        linestyle="--",
        linewidth=1.4,
        label="Never Verify",
    )
    saturation = _oracle_at_budget(task_rows, 10**12)
    saturation_cost = int(saturation["actual_teacher_tokens"])
    axis.scatter(
        [saturation_cost / 1000],
        [100 * float(saturation["accuracy"])],
        s=90,
        color="#7353BA",
        edgecolor="white",
        zorder=4,
    )
    axis.annotate(
        f"Oracle saturation\n{100*float(saturation['accuracy']):.2f}% @ {saturation_cost/1000:.1f}k tokens",
        (saturation_cost / 1000, 100 * float(saturation["accuracy"])),
        xytext=(64, -6),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#7353BA"},
        fontsize=9,
        fontweight="bold",
    )
    axis.annotate(
        f"DARC full\n{100*float(full['accuracy']):.2f}% @ {int(full['actual_teacher_tokens'])/1000:.1f}k",
        (int(full["actual_teacher_tokens"]) / 1000, 100 * float(full["accuracy"])),
        xytext=(-105, -38),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#E45756"},
        fontsize=9,
        fontweight="bold",
    )
    axis.set_xlabel("Actual Teacher tokens (thousands)")
    axis.set_ylabel("Final task accuracy (%)")
    axis.set_title("a  Accuracy–cost upper bound", loc="left", fontweight="bold")
    axis.grid(color="#E6EAF0", linewidth=0.7)
    axis.legend(frameon=False, loc="lower right")

    axis = axes[1]
    darc_correct = int(full["correct"])
    values = (baseline_correct, darc_correct, oracle_correct)
    names = ("Never\nVerify", "DARC\nv0.2", "Oracle\nupper bound")
    colors = ("#7A8694", "#E45756", "#7353BA")
    bars = axis.bar(names, 100 * np.asarray(values) / tasks, color=colors, width=0.62)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.45,
            f"{100*value/tasks:.2f}%\n({value}/{tasks})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axis.set_ylim(0, max(52, 100 * oracle_correct / tasks + 10))
    axis.set_ylabel("Final task accuracy (%)")
    axis.set_title("b  Where DARC loses Oracle gain", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E6EAF0", linewidth=0.7)
    captured = int(failure["helpful_selected"])
    wrong_cp = int(failure["incorrect_checkpoint"])
    skipped = int(failure["missed_helpful"])
    axis.text(
        0.5,
        0.98,
        f"19 rescuable tasks:  {captured} captured  ·  {wrong_cp} wrong CP  ·  {skipped} skipped\n"
        f"121/127 DARC calls were No-change",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FFF3E8", "edgecolor": "#F1B36A"},
    )
    fig.suptitle(
        "Oracle diagnosis: improvement exists, but DARC does not identify it efficiently",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _report(
    output: Path,
    integrity: dict[str, Any],
    label_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    failure_csv: Path,
) -> Path:
    test_labels = [row for row in label_rows if row["split"] == "test"]
    full = max(comparisons, key=lambda row: float(row["budget_fraction"]))
    failures = _read_csv(failure_csv)
    failure = next(
        row
        for row in failures
        if row["policy"] == "DARC v0.2 Full"
        and float(row["budget_fraction"]) == 1.0
    )
    tasks = len(task_rows)
    baseline = sum(int(row["baseline_correct"]) for row in task_rows)
    rescuable = sum(int(row["has_helpful"]) for row in task_rows)
    cheapest_total = sum(
        int(row["oracle_actual_tokens"])
        for row in task_rows
        if int(row["has_helpful"])
    )
    oracle_accuracy = (baseline + rescuable) / tasks
    report = output / "report" / "darc_data_audit_oracle.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DARC Data Audit + Observed-Branch Oracle Upper Bound",
        "",
        "> Offline post-hoc analysis only. No Student/Teacher inference and no model retraining. The Oracle reads actual paired outcomes and actual Teacher costs; it is an upper bound, not a deployable policy and not a tuning target.",
        "",
        "## Executive answer",
        "",
        f"The frozen test contains **{rescuable}/{tasks} ({100*rescuable/tasks:.1f}%)** tasks with at least one observed Helpful single-VERIFY branch. Never Verify solves **{baseline}/{tasks} ({100*baseline/tasks:.2f}%)**; the observed-branch Oracle can solve **{baseline+rescuable}/{tasks} ({100*oracle_accuracy:.2f}%)**.",
        "",
        f"DARC v0.2 solves **{int(full['correct'])}/{tasks} ({100*float(full['accuracy']):.2f}%)** using **{int(full['actual_teacher_tokens']):,}** Teacher tokens. The Oracle reaches its maximum using only **{cheapest_total:,}** actual tokens by choosing the cheapest Helpful checkpoint for each rescuable task. Therefore, the data offers real but sparse headroom; DARC's dominant failure is selection, not absence of counterfactual opportunities.",
        "",
        "## 1. Label distribution",
        "",
        "| Split | CP | Helpful | Harmful | No-change | Net mean Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in label_rows:
        lines.append(
            f"| {row['split']} | CP{row['checkpoint']} | {row['helpful']} ({100*float(row['helpful_rate']):.1f}%) | {row['harmful']} ({100*float(row['harmful_rate']):.1f}%) | {row['no_change']} ({100*float(row['no_change_rate']):.1f}%) | {100*float(row['mean_actual_delta']):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "![Label distribution](../figures/01_label_distribution.png)",
            "",
            "No-change dominates every split/checkpoint. On frozen test it accounts for 94.4%–95.4% of states. This is a clean-label but rare-event prediction problem: having many paired states does not imply having many informative Helpful/Harmful labels.",
            "",
            "## 2. Label and actual-gain integrity",
            "",
            f"- Paired states checked: **{integrity['states']:,}** across **{integrity['tasks']:,}** task groups.",
            f"- Delta mismatches: **{integrity.get('delta_mismatch', 0)}**; value-label mismatches: **{integrity.get('label_mismatch', 0)}**.",
            f"- Incomplete 3-checkpoint groups: **{integrity.get('incomplete_task', 0)}**; split overlaps: **{sum(integrity['split_overlap'].values())}**.",
            "- `Delta = Y_verify - Y_skip` is exactly in {-1, 0, +1}: -1 is Harmful, 0 is No-change, and +1 is Helpful.",
            "",
            "| Frozen-test checkpoint | Helpful | Harmful | No-change | Actual mean Δ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in test_labels:
        lines.append(
            f"| CP{row['checkpoint']} | {row['helpful']} | {row['harmful']} | {row['no_change']} | {100*float(row['mean_actual_delta']):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "![Actual downstream gain](../figures/02_actual_gain_audit.png)",
            "",
            "CP3 has the highest unconditional mean gain and no Harmful state in this test sample, but this does not mean CP3 is the cost-optimal checkpoint per task. The Oracle's cheapest Helpful choices are task-specific.",
            "",
            "## 3. Same-budget Oracle upper bound",
            "",
            "The Oracle allows at most one VERIFY per task, may skip all checkpoints, and only chooses among observed CP1/CP2/CP3 branches. Because every Helpful intervention contributes the same +1 correct task, the exact budgeted solution is to retain each task's cheapest Helpful checkpoint and select those tasks in ascending actual-token cost.",
            "",
            "| DARC budget fraction | DARC actual tokens | DARC accuracy | Oracle accuracy at same actual spend | Gap |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {float(row['budget_fraction']):.2f} | {int(row['actual_teacher_tokens']):,} | {100*float(row['accuracy']):.2f}% | {100*float(row['oracle_accuracy_same_actual_spend']):.2f}% | {float(row['selection_gap_pp']):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "![Oracle upper bound](../figures/03_oracle_upper_bound.png)",
            "",
            "## 4. DARC error decomposition",
            "",
            f"Among the **{rescuable}** rescuable test tasks, DARC selected a Helpful checkpoint for **{failure['helpful_selected']}**, selected the wrong checkpoint for **{failure['incorrect_checkpoint']}**, and skipped verification for **{failure['missed_helpful']}**. It made **{failure['teacher_calls']}** calls, of which **{failure['wasted_no_change']}** were No-change.",
            "",
            f"DARC therefore captures **{100*int(failure['helpful_selected'])/rescuable:.1f}%** of the available task-level gain. The remaining Oracle–DARC gap is **{100*(oracle_accuracy-float(full['accuracy'])):.2f} pp**.",
            "",
            "## 5. Conclusion",
            "",
            "**Both statements are true, but selection is the larger immediate problem.** Helpful opportunities are intrinsically sparse (8.8% of test tasks), which makes learning difficult. Nevertheless, the observed data contains enough opportunity to raise accuracy from 32.87% to 41.67%. DARC reaches only 35.65%, spends far more tokens than the Oracle upper bound, and routes most calls to No-change states. The next improvement should target Helpful ranking and per-task checkpoint discrimination, not merely increase the number of calls or move every decision to CP3.",
            "",
            "## Artifacts",
            "",
            "- `tables/label_distribution.csv`",
            "- `tables/gain_by_dataset_checkpoint.csv`",
            "- `tables/test_task_oracle.csv`",
            "- `tables/oracle_frontier.csv`",
            "- `tables/oracle_vs_darc.csv`",
            "- `data_integrity.json`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(
    *,
    frozen_root: Path,
    policy_csv: Path,
    failure_csv: Path,
    output: Path,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    partitions = {split: _load_split(frozen_root, split)[0] for split in SPLITS}
    integrity = _audit_integrity(partitions)
    if not integrity["checks_passed"]:
        raise RuntimeError(f"DARC data integrity audit failed: {integrity}")
    label_rows = _label_rows(partitions)
    gain_rows = _gain_rows(partitions)
    task_rows = _task_rows(partitions["test"])
    darc_rows = _darc_rows(policy_csv)
    if not darc_rows:
        raise RuntimeError("No DARC v0.2 policy rows found")
    comparisons = _oracle_comparison(task_rows, darc_rows)
    max_budget = max(int(row["nominal_budget"]) for row in darc_rows)
    frontier = _oracle_frontier(task_rows, max_budget)

    _write_csv(output / "tables" / "label_distribution.csv", label_rows)
    _write_csv(output / "tables" / "gain_by_dataset_checkpoint.csv", gain_rows)
    _write_csv(output / "tables" / "test_task_oracle.csv", task_rows)
    _write_csv(output / "tables" / "oracle_frontier.csv", frontier)
    _write_csv(output / "tables" / "oracle_vs_darc.csv", comparisons)
    (output / "data_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot_label_distribution(label_rows, figures / "01_label_distribution")
    _plot_gain_audit(
        label_rows,
        gain_rows,
        task_rows,
        figures / "02_actual_gain_audit",
    )
    _plot_oracle(
        frontier,
        comparisons,
        task_rows,
        failure_csv,
        figures / "03_oracle_upper_bound",
    )
    return _report(
        output,
        integrity,
        label_rows,
        task_rows,
        comparisons,
        failure_csv,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit DARC paired labels and compute an observed-branch Oracle upper bound."
    )
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=Path("data/cbv_bench_v01_frozen_28102026"),
    )
    parser.add_argument(
        "--policy-csv",
        type=Path,
        default=Path(
            "results/darc_v02_29102026/test_evaluation/tables/policy_results.csv"
        ),
    )
    parser.add_argument(
        "--failure-csv",
        type=Path,
        default=Path(
            "results/darc_v02_29102026/test_evaluation/tables/failure_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/darc_v02_29102026/data_audit_oracle"),
    )
    args = parser.parse_args()
    report = run(
        frozen_root=args.frozen_root,
        policy_csv=args.policy_csv,
        failure_csv=args.failure_csv,
        output=args.output,
    )
    print(f"Completed DARC data audit + Oracle analysis: {report}")


if __name__ == "__main__":
    main()
