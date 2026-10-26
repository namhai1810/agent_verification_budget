#!/usr/bin/env python3
# ruff: noqa: E501
"""Build Phase-2 training-flow and model-diagnostic figures for the dated report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE2_DIR = PROJECT_ROOT / "results" / "phase2"
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_report_241026"

NAVY = "#0F172A"
SLATE = "#475569"
MUTED = "#94A3B8"
GRID = "#E2E8F0"
PAPER = "#F8FAFC"
WHITE = "#FFFFFF"
BLUE = "#2563EB"
CYAN = "#0891B2"
GREEN = "#10B981"
AMBER = "#F59E0B"
RED = "#EF4444"
PURPLE = "#7C3AED"


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    title: str,
    body: str,
    color: str,
) -> None:
    x, y = xy
    width, height = size
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=WHITE,
        edgecolor=color,
        linewidth=1.7,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            0.012,
            height,
            boxstyle="round,pad=0.001,rounding_size=0.008",
            facecolor=color,
            edgecolor=color,
            transform=ax.transAxes,
        )
    )
    ax.text(x + 0.025, y + height * 0.72, title, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=NAVY)
    ax.text(x + 0.025, y + height * 0.48, body, transform=ax.transAxes, fontsize=8.6, color=SLATE, va="top", linespacing=1.35)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = MUTED) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=color,
            transform=ax.transAxes,
        )
    )


def plot_training_flow() -> Path:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=PAPER)
    ax.set_axis_off()
    fig.text(0.045, 0.94, "Phase 2 training flow — leakage-safe downstream value prediction", fontsize=23, fontweight="bold", color=NAVY)
    fig.text(0.045, 0.902, "Every reported prediction is out-of-fold; Teacher and HumanEval evaluator are never called", fontsize=10.5, color=SLATE)

    add_box(ax, (0.04, 0.64), (0.18, 0.19), "1 • Phase-1 records", "139 tasks\n417 checkpoint pairs\nY_skip / Y_verify labels", BLUE)
    add_box(ax, (0.29, 0.64), (0.20, 0.19), "2 • Pre-verification features", "Problem + current artifact\nprior artifact metadata\nstatic AST + local risk", CYAN)
    add_box(ax, (0.56, 0.64), (0.18, 0.19), "3 • Outer GroupKFold", "5 held-out task folds\nall CPs of one task stay\nin the same fold", PURPLE)
    add_box(ax, (0.81, 0.64), (0.15, 0.19), "4 • Inner CV", "4 task-grouped folds\nselect hyperparameters\nand Delta threshold", AMBER)
    add_arrow(ax, (0.22, 0.735), (0.29, 0.735))
    add_arrow(ax, (0.49, 0.735), (0.56, 0.735))
    add_arrow(ax, (0.74, 0.735), (0.81, 0.735))

    add_box(ax, (0.10, 0.29), (0.21, 0.20), "5A • Logistic models", "Metadata-only Logistic\nTF-IDF + metadata Logistic\ntwo independent action heads", BLUE)
    add_box(ax, (0.395, 0.29), (0.21, 0.20), "5B • Boosting model", "TF-IDF → Truncated SVD\n+ metadata → HistGradientBoosting\ntwo independent action heads", GREEN)
    add_box(ax, (0.69, 0.29), (0.21, 0.20), "6 • Held-out predictions", "Q_skip and Q_verify\nDelta_pred = Q_verify − Q_skip\n1,251 OOF prediction rows", CYAN)
    add_arrow(ax, (0.885, 0.64), (0.50, 0.51), AMBER)
    add_arrow(ax, (0.81, 0.64), (0.205, 0.51), AMBER)
    add_arrow(ax, (0.31, 0.39), (0.69, 0.39))
    add_arrow(ax, (0.605, 0.39), (0.69, 0.39))

    add_box(ax, (0.20, 0.04), (0.25, 0.15), "7 • Sequential controller", "Observe CP1 → CP2 → CP3\nverify first Delta_pred > threshold\nmaximum one Teacher call", PURPLE)
    add_box(ax, (0.55, 0.04), (0.25, 0.15), "8 • Offline policy evaluation", "Use stored counterfactual outcome\npaired task bootstrap 95% CI\naccuracy + calls + tokens", GREEN)
    add_arrow(ax, (0.795, 0.29), (0.45, 0.19), CYAN)
    add_arrow(ax, (0.45, 0.115), (0.55, 0.115))

    guard = FancyBboxPatch(
        (0.03, 0.875),
        0.94,
        0.045,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        facecolor="#FEF2F2",
        edgecolor=RED,
        linewidth=1.0,
        transform=ax.transAxes,
    )
    ax.add_patch(guard)
    ax.text(
        0.5,
        0.897,
        "LEAKAGE GUARD  •  excluded: task ID, outcomes, Delta, Teacher correction, future agents, private tests",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.4,
        color="#991B1B",
        fontweight="bold",
    )
    output = OUTPUT_DIR / "phase2_training_flow.png"
    fig.savefig(output, dpi=190, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_training_diagnostics() -> Path:
    selections = json.loads((PHASE2_DIR / "fold_selections.json").read_text(encoding="utf-8"))
    predictor_rows = read_csv(PHASE2_DIR / "predictor_metrics.csv")
    policy_rows = read_csv(PHASE2_DIR / "policy_comparison.csv")
    predictor = {row["model"]: row for row in predictor_rows}
    policy = {row["method"]: row for row in policy_rows}
    models = ["Metadata-only Logistic", "Ours Logistic", "Ours Boosting"]
    colors = [MUTED, BLUE, GREEN]
    folds = np.arange(5)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor=PAPER)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.09, hspace=0.43, wspace=0.27)
    fig.suptitle("Phase 2 model training diagnostics", x=0.07, ha="left", fontsize=23, fontweight="bold", color=NAVY)
    fig.text(0.07, 0.91, "Nested GroupKFold selection and out-of-fold behavior on 139 tasks", fontsize=10.5, color=SLATE)

    inner_ax = axes[0, 0]
    for model, color in zip(models, colors, strict=True):
        values = [100 * float(row["inner_accuracy"]) for row in selections[model]]
        inner_ax.plot(folds, values, marker="o", linewidth=2, color=color, label=model)
    inner_ax.set_title("Inner-validation policy accuracy", loc="left", fontweight="bold", color=NAVY)
    inner_ax.set_xlabel("Outer fold")
    inner_ax.set_ylabel("Inner OOF accuracy (%)")
    inner_ax.set_xticks(folds)
    inner_ax.grid(alpha=0.25)
    inner_ax.legend(frameon=False, fontsize=8)

    threshold_ax = axes[0, 1]
    width = 0.24
    for offset, (model, color) in enumerate(zip(models, colors, strict=True)):
        values = [float(row["threshold"]) for row in selections[model]]
        threshold_ax.bar(folds + (offset - 1) * width, values, width, color=color, label=model)
    threshold_ax.axhline(0, color=NAVY, linewidth=0.8)
    threshold_ax.set_title("Selected Delta threshold varies by fold", loc="left", fontweight="bold", color=NAVY)
    threshold_ax.set_xlabel("Outer fold")
    threshold_ax.set_ylabel("Inner-selected threshold")
    threshold_ax.set_xticks(folds)
    threshold_ax.grid(axis="y", alpha=0.25)

    quality_ax = axes[1, 0]
    quality_models = ["Local Risk heuristic", *models]
    x = np.arange(len(quality_models))
    pearson = [float(predictor[name]["delta_pearson"]) for name in quality_models]
    helpful_ap = [float(predictor[name]["helpful_average_precision"]) for name in quality_models]
    quality_ax.bar(x - 0.18, pearson, 0.36, color=BLUE, label="Delta Pearson")
    quality_ax.bar(x + 0.18, helpful_ap, 0.36, color=AMBER, label="Helpful AP")
    quality_ax.axhline(0, color=NAVY, linewidth=0.8)
    quality_ax.set_title("OOF benefit prediction remains weak", loc="left", fontweight="bold", color=NAVY)
    quality_ax.set_xticks(x, ["Local\nRisk", "Metadata\nLogistic", "TF-IDF\nLogistic", "SVD +\nBoosting"])
    quality_ax.set_ylabel("Score")
    quality_ax.grid(axis="y", alpha=0.25)
    quality_ax.legend(frameon=False, fontsize=8)

    policy_ax = axes[1, 1]
    chosen = ["Never Verify", "Local Risk Controller", "Ours Logistic", "Ours Boosting", "Always Verify CP3", "Offline Oracle"]
    accuracy = [100 * float(policy[name]["accuracy"]) for name in chosen]
    calls = [float(policy[name]["average_teacher_calls_per_task"]) for name in chosen]
    scatter_colors = [MUTED, AMBER, BLUE, GREEN, PURPLE, RED]
    for name, acc, call, color in zip(chosen, accuracy, calls, scatter_colors, strict=True):
        policy_ax.scatter(call, acc, s=110, color=color, edgecolor=WHITE, linewidth=0.8, zorder=3)
        policy_ax.annotate(name.replace(" Verify", ""), (call, acc), xytext=(5, 5), textcoords="offset points", fontsize=7.8)
    policy_ax.set_title("Learned routing trades accuracy for calls", loc="left", fontweight="bold", color=NAVY)
    policy_ax.set_xlabel("Teacher calls / task")
    policy_ax.set_ylabel("Final accuracy (%)")
    policy_ax.grid(alpha=0.25)

    for ax in axes.ravel():
        ax.set_facecolor(WHITE)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=SLATE)
    output = OUTPUT_DIR / "phase2_model_training_diagnostics.png"
    fig.savefig(output, dpi=190, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for output in (plot_training_flow(), plot_training_diagnostics()):
        print(output)


if __name__ == "__main__":
    main()
