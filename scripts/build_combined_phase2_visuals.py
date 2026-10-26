#!/usr/bin/env python3
"""Visualize combined Phase-2 data, frozen split, and optimization results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAVY = "#0F172A"
SLATE = "#475569"
GRID = "#E2E8F0"
PAPER = "#F8FAFC"
BLUE = "#2563EB"
CYAN = "#0891B2"
GREEN = "#10B981"
RED = "#EF4444"
AMBER = "#F59E0B"
MUTED = "#94A3B8"
PURPLE = "#7C3AED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/combined_phase2_optimized"))
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=SLATE)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=NAVY, pad=18)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=8.5, color=SLATE, va="bottom")


def policy_outcomes(path: Path, model: str = "Ours Boosting") -> tuple[dict[str, float], int]:
    rows = [row for row in read_csv(path) if row["model"] == model]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    outcomes: dict[str, float] = {}
    calls = 0
    for task_id, task_rows in grouped.items():
        selected = [row for row in task_rows if int(row["selected_by_controller"]) == 1]
        if selected:
            if len(selected) != 1:
                raise ValueError(f"Expected at most one selected checkpoint for {task_id}")
            outcomes[task_id] = float(selected[0]["y_verify"])
            calls += 1
        else:
            outcomes[task_id] = float(task_rows[0]["y_skip"])
    return outcomes, calls


def paired_optimization_rows(results_dir: Path, repetitions: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for scope in ("combined", "humaneval", "livecodebench"):
        compact, compact_calls = policy_outcomes(results_dir / "predictions" / f"{scope}_compact.csv")
        optimized, optimized_calls = policy_outcomes(
            results_dir / "predictions" / f"{scope}_optimized_boosting.csv"
        )
        task_ids = sorted(compact)
        if task_ids != sorted(optimized):
            raise ValueError(f"Prediction universes differ for {scope}")
        compact_values = np.asarray([compact[task_id] for task_id in task_ids])
        optimized_values = np.asarray([optimized[task_id] for task_id in task_ids])
        differences = optimized_values - compact_values
        samples = rng.integers(0, len(task_ids), size=(repetitions, len(task_ids)))
        bootstrap = differences[samples].mean(axis=1) * 100
        low, high = np.percentile(bootstrap, [2.5, 97.5])
        rows.append(
            {
                "benchmark": scope,
                "tasks": len(task_ids),
                "compact_accuracy": float(compact_values.mean()),
                "optimized_accuracy": float(optimized_values.mean()),
                "optimized_minus_compact_pp": float(differences.mean() * 100),
                "difference_ci95_low_pp": float(low),
                "difference_ci95_high_pp": float(high),
                "compact_calls": compact_calls,
                "optimized_calls": optimized_calls,
            }
        )
    with (results_dir / "optimization_delta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_data_overview(results_dir: Path) -> Path:
    assignments = read_csv(results_dir / "split_assignments.csv")
    summary = read_csv(results_dir / "dataset_summary.csv")
    manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    strata_order = [
        ("humaneval", "unspecified", "HumanEval"),
        ("livecodebench", "easy", "LCB Easy"),
        ("livecodebench", "medium", "LCB Medium"),
        ("livecodebench", "hard", "LCB Hard"),
    ]
    train = Counter((row["benchmark"], row["difficulty"]) for row in assignments if row["split"] == "train")
    test = Counter((row["benchmark"], row["difficulty"]) for row in assignments if row["split"] == "test")
    all_summary = {row["benchmark"]: row for row in summary if row["split"] == "all"}

    fig = plt.figure(figsize=(16, 10), facecolor=PAPER)
    grid = fig.add_gridspec(2, 2, left=0.07, right=0.96, top=0.82, bottom=0.09, hspace=0.4, wspace=0.28)
    fig.text(
        0.07, 0.95, "Combined Phase 2 — data and leakage audit", fontsize=24, fontweight="bold", color=NAVY
    )
    fig.text(
        0.07,
        0.91,
        f"{manifest['all_tasks']} complete task triples • {manifest['all_checkpoint_rows']} rows • "
        f"train {manifest['train_tasks']} / frozen test {manifest['fixed_test_tasks']}",
        fontsize=11,
        color=SLATE,
    )
    fig.text(
        0.07,
        0.865,
        "Split uses task ID + benchmark/difficulty only; all CP1/CP2/CP3 rows remain together; overlap = 0",
        fontsize=11,
        fontweight="bold",
        color=GREEN,
    )

    composition = fig.add_subplot(grid[0, 0])
    style(composition, "Task composition and frozen split", "Stratified by benchmark/difficulty")
    labels = [label for _, _, label in strata_order]
    train_values = np.asarray([train[(benchmark, difficulty)] for benchmark, difficulty, _ in strata_order])
    test_values = np.asarray([test[(benchmark, difficulty)] for benchmark, difficulty, _ in strata_order])
    x = np.arange(len(labels))
    composition.bar(x, train_values, color=BLUE, label="Train", zorder=3)
    composition.bar(x, test_values, bottom=train_values, color=AMBER, label="Frozen test", zorder=3)
    composition.set_xticks(x, labels)
    composition.set_ylabel("Tasks", color=SLATE)
    composition.legend(frameon=False, ncol=2)
    for index, total in enumerate(train_values + test_values):
        composition.text(index, total + 2, f"{total}", ha="center", fontweight="bold", color=NAVY)

    labels_ax = fig.add_subplot(grid[0, 1])
    style(labels_ax, "Observed verification outcomes", "Labels retained only as training/evaluation targets")
    benchmark_labels = ["HumanEval", "LiveCodeBench"]
    helpful = [int(all_summary[name]["helpful"]) for name in ("humaneval", "livecodebench")]
    harmful = [int(all_summary[name]["harmful"]) for name in ("humaneval", "livecodebench")]
    unchanged = [int(all_summary[name]["no_change"]) for name in ("humaneval", "livecodebench")]
    y = np.arange(2)
    labels_ax.barh(y, unchanged, color=MUTED, label="No change", zorder=3)
    labels_ax.barh(y, helpful, left=unchanged, color=GREEN, label="Helpful", zorder=3)
    labels_ax.barh(y, harmful, left=np.asarray(unchanged) + helpful, color=RED, label="Harmful", zorder=3)
    labels_ax.set_yticks(y, benchmark_labels)
    labels_ax.invert_yaxis()
    labels_ax.set_xlabel("Checkpoint pairs", color=SLATE)
    labels_ax.legend(frameon=False, ncol=3, loc="lower right")
    for index, total in enumerate(np.asarray(unchanged) + helpful + harmful):
        labels_ax.text(
            total + 5,
            index,
            f"{helpful[index]}/{harmful[index]}/{unchanged[index]}",
            va="center",
            color=NAVY,
            fontweight="bold",
        )

    baseline_ax = fig.add_subplot(grid[1, 0])
    style(
        baseline_ax,
        "Baseline accuracy shift",
        "Why pooled accuracy must be reported alongside per-benchmark metrics",
    )
    scopes = ["humaneval", "livecodebench", "all"]
    names = ["HumanEval", "LiveCodeBench", "Combined"]
    values = [100 * float(all_summary[scope]["baseline_accuracy"]) for scope in scopes]
    bars = baseline_ax.bar(names, values, color=[PURPLE, CYAN, BLUE], width=0.62, zorder=3)
    baseline_ax.set_ylim(0, 92)
    baseline_ax.set_ylabel("Accuracy (%)", color=SLATE)
    for bar, value in zip(bars, values, strict=True):
        baseline_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            fontweight="bold",
            color=NAVY,
        )

    audit_ax = fig.add_subplot(grid[1, 1])
    audit_ax.axis("off")
    audit_ax.set_facecolor("white")
    audit_ax.text(0, 1, "Leakage controls", fontsize=13, fontweight="bold", color=NAVY, va="top")
    controls = [
        ("0", "task IDs shared by train and test"),
        ("3", "checkpoint rows per included task"),
        ("250", "train tasks used for tuning"),
        ("63", "frozen tasks evaluated once"),
        ("1", "incomplete task dropped: arc191_d"),
    ]
    for index, (value, text) in enumerate(controls):
        top = 0.83 - index * 0.16
        audit_ax.text(
            0.02, top, value, fontsize=23, fontweight="bold", color=GREEN if index < 2 else BLUE, va="center"
        )
        audit_ax.text(0.17, top, text, fontsize=10.5, color=SLATE, va="center")
        audit_ax.plot([0.02, 0.96], [top - 0.075, top - 0.075], color=GRID, linewidth=1)

    output = results_dir / "combined_data_overview.png"
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def plot_model_results(results_dir: Path, optimization_rows: list[dict[str, Any]]) -> Path:
    rows = read_csv(results_dir / "policy_comparison_by_benchmark.csv")
    methods = [
        "Never Verify",
        "Local Risk Controller",
        "Ours Logistic",
        "Ours Boosting",
        "Optimized Boosting",
        "Always Verify CP2",
        "Always Verify CP3",
        "Offline Oracle",
    ]
    combined = {row["method"]: row for row in rows if row["benchmark"] == "combined"}
    scoped = {(row["benchmark"], row["method"]): row for row in rows}

    fig = plt.figure(figsize=(17, 10), facecolor=PAPER)
    grid = fig.add_gridspec(2, 2, left=0.07, right=0.96, top=0.84, bottom=0.09, hspace=0.42, wspace=0.3)
    fig.text(
        0.07,
        0.955,
        "Combined Phase 2 — frozen-test model comparison",
        fontsize=24,
        fontweight="bold",
        color=NAVY,
    )
    fig.text(
        0.07,
        0.915,
        "Hyperparameters and thresholds selected on 250 train tasks; 63 test tasks evaluated once",
        fontsize=11,
        color=SLATE,
    )
    optimized = combined["Optimized Boosting"]
    compact = combined["Ours Boosting"]
    optimized_accuracy = 100 * float(optimized["accuracy"])
    compact_accuracy = 100 * float(compact["accuracy"])
    fig.text(
        0.07,
        0.875,
        f"Optimized {optimized_accuracy:.2f}% vs compact {compact_accuracy:.2f}% • "
        f"calls {float(optimized['teacher_calls']):.0f} vs {float(compact['teacher_calls']):.0f}",
        fontsize=11,
        fontweight="bold",
        color=NAVY,
    )

    accuracy_ax = fig.add_subplot(grid[0, :])
    style(accuracy_ax, "Combined frozen-test accuracy", "Task-bootstrap 95% confidence intervals")
    names = methods
    accuracy = np.asarray([100 * float(combined[name]["accuracy"]) for name in names])
    low = np.asarray([100 * float(combined[name]["accuracy_ci95_low"]) for name in names])
    high = np.asarray([100 * float(combined[name]["accuracy_ci95_high"]) for name in names])
    y = np.arange(len(names))
    colors = [MUTED, AMBER, CYAN, BLUE, GREEN, PURPLE, "#8B5CF6", RED]
    accuracy_ax.barh(y, accuracy, color=colors, zorder=3)
    accuracy_ax.errorbar(
        accuracy,
        y,
        xerr=np.vstack([accuracy - low, high - accuracy]),
        fmt="none",
        ecolor=NAVY,
        capsize=3,
        zorder=4,
    )
    accuracy_ax.set_yticks(y, names)
    accuracy_ax.invert_yaxis()
    accuracy_ax.set_xlabel("Accuracy (%)", color=SLATE)
    accuracy_ax.set_xlim(20, 55)
    for index, value in enumerate(accuracy):
        accuracy_ax.text(
            value + 0.5, index, f"{value:.2f}%", va="center", fontsize=8.5, fontweight="bold", color=NAVY
        )

    gain_ax = fig.add_subplot(grid[1, 0])
    style(gain_ax, "Gain versus Never Verify", "Distribution shift remains visible after combined training")
    compared = ["Ours Logistic", "Ours Boosting", "Optimized Boosting"]
    x = np.arange(len(compared))
    width = 0.35
    human_gain = [float(scoped[("humaneval", name)]["gain_vs_never_pp"]) for name in compared]
    lcb_gain = [float(scoped[("livecodebench", name)]["gain_vs_never_pp"]) for name in compared]
    gain_ax.bar(x - width / 2, human_gain, width, color=PURPLE, label="HumanEval", zorder=3)
    gain_ax.bar(x + width / 2, lcb_gain, width, color=CYAN, label="LiveCodeBench", zorder=3)
    gain_ax.axhline(0, color=NAVY, linewidth=0.8)
    gain_ax.set_xticks(x, ["Logistic", "Compact Boost", "Optimized Boost"])
    gain_ax.set_ylabel("Accuracy gain (pp)", color=SLATE)
    gain_ax.legend(frameon=False, ncol=2)

    cost_ax = fig.add_subplot(grid[1, 1])
    style(cost_ax, "Optimization gain versus verification cost", "Paired CI uses the same frozen test tasks")
    names = [row["benchmark"] for row in optimization_rows]
    differences = np.asarray([float(row["optimized_minus_compact_pp"]) for row in optimization_rows])
    low = np.asarray([float(row["difference_ci95_low_pp"]) for row in optimization_rows])
    high = np.asarray([float(row["difference_ci95_high_pp"]) for row in optimization_rows])
    calls_added = np.asarray(
        [int(row["optimized_calls"]) - int(row["compact_calls"]) for row in optimization_rows]
    )
    positions = np.arange(len(names))
    bars = cost_ax.bar(positions, differences, color=[GREEN, PURPLE, CYAN], zorder=3)
    cost_ax.errorbar(
        positions,
        differences,
        yerr=np.vstack([differences - low, high - differences]),
        fmt="none",
        ecolor=NAVY,
        capsize=4,
        zorder=4,
    )
    cost_ax.axhline(0, color=NAVY, linewidth=0.8)
    cost_ax.set_xticks(positions, [name.title() for name in names])
    cost_ax.set_ylabel("Optimized − compact (pp)", color=SLATE)
    for bar, added in zip(bars, calls_added, strict=True):
        cost_ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 0) + 0.5,
            f"+{added} calls",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=NAVY,
        )

    output = results_dir / "combined_model_optimization.png"
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    optimization_rows = paired_optimization_rows(args.results_dir, args.bootstrap_reps, args.seed)
    overview = plot_data_overview(args.results_dir)
    comparison = plot_model_results(args.results_dir, optimization_rows)
    print(overview)
    print(comparison)
    print(args.results_dir / "optimization_delta.csv")


if __name__ == "__main__":
    main()
