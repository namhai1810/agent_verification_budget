#!/usr/bin/env python3
"""Create compact, paper-oriented DARC v0.5 architecture/result figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "darc_v05_offline_rl_budgetfix_31102026"
FIGURE_ROOT = RESULT_ROOT / "figures"

NAVY = "#17324D"
BLUE = "#3B6EA8"
LIGHT_BLUE = "#EAF2F8"
ORANGE = "#E67E22"
LIGHT_ORANGE = "#FCEBDD"
GREEN = "#2B8A66"
LIGHT_GREEN = "#E5F4EE"
GRAY = "#6B7785"
LIGHT_GRAY = "#F1F3F5"
RED = "#C94C4C"


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    facecolor: str,
    edgecolor: str,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.6,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=11,
        color=edgecolor,
        fontweight="bold",
    )
    axis.text(
        xy[0] + width / 2,
        xy[1] + height * 0.31,
        body,
        ha="center",
        va="center",
        fontsize=8.5,
        color=NAVY,
        linespacing=1.35,
    )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    dashed: bool = False,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            linestyle="--" if dashed else "-",
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def architecture_figure() -> None:
    fig, axis = plt.subplots(figsize=(15.5, 7.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.02,
        0.965,
        "DARC v0.5: observable-state learning with strict global-budget replay",
        fontsize=18,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    axis.text(
        0.02,
        0.915,
        "Per-task allowance conditions value; the global ledger alone enforces feasibility",
        fontsize=10.5,
        color=GRAY,
        va="top",
    )

    _box(
        axis,
        (0.02, 0.57),
        0.19,
        0.23,
        "Observable semantic state",
        "Task + current artifact\npreceding artifacts/history\ncheckpoint + agent roles",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
    )
    _box(
        axis,
        (0.02, 0.29),
        0.19,
        0.18,
        "Action history",
        "Two prefix tokens\n0=padding, 1=Skip, 2=Verify",
        facecolor=LIGHT_GRAY,
        edgecolor=GRAY,
    )
    _box(
        axis,
        (0.02, 0.07),
        0.19,
        0.14,
        "Per-task allowance",
        "Budget bucket + shadow price\nNOT a hard feasibility gate",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
    )

    _box(
        axis,
        (0.285, 0.56),
        0.20,
        0.25,
        "State encoder",
        "Frozen UniXcoder embeddings\nsegment projections + State Fusion\ndownstream multi-head attention",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
    )
    _box(
        axis,
        (0.285, 0.23),
        0.20,
        0.22,
        "Budget/history fusion",
        "Action embedding + 3 budget features\nsemantic/context gated fusion\nshared nonlinear trunk",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
    )

    _box(
        axis,
        (0.555, 0.47),
        0.19,
        0.34,
        "Four prediction heads",
        (
            "Qsuccess(S), Qsuccess(V)\nQcost(S), Qcost(V)\n"
            "P(Harmful/No-change/Helpful)\nimmediate Teacher-token cost\n"
            "+ ensemble uncertainty"
        ),
        facecolor=LIGHT_GREEN,
        edgecolor=GREEN,
    )

    _box(
        axis,
        (0.80, 0.52),
        0.18,
        0.29,
        "Conservative policy",
        "advantage − harmful risk\n− uncertainty − dynamic token price\nVerify iff score > margin",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
    )
    _box(
        axis,
        (0.80, 0.19),
        0.18,
        0.22,
        "Strict action gate",
        (
            "global remaining ≥ 8,192 reserve\n"
            "predicted call ≤ global remaining\n"
            "actual cost charged after action"
        ),
        facecolor="#FBEAEA",
        edgecolor=RED,
    )
    _box(
        axis,
        (0.53, 0.08),
        0.22,
        0.17,
        "Offline counterfactual targets",
        (
            "Force S or V, then follow one fixed\n"
            "observable continuation policy\n"
            "record final correctness + realized cost"
        ),
        facecolor=LIGHT_GRAY,
        edgecolor=GRAY,
    )

    _arrow(axis, (0.21, 0.68), (0.285, 0.68), color=BLUE)
    _arrow(axis, (0.21, 0.38), (0.285, 0.34), color=GRAY)
    _arrow(axis, (0.21, 0.14), (0.285, 0.30), color=ORANGE)
    _arrow(axis, (0.485, 0.68), (0.555, 0.68), color=GREEN)
    _arrow(axis, (0.485, 0.34), (0.555, 0.55), color=GREEN)
    _arrow(axis, (0.745, 0.64), (0.80, 0.66), color=ORANGE)
    _arrow(axis, (0.89, 0.52), (0.89, 0.41), color=RED)
    _arrow(axis, (0.80, 0.30), (0.75, 0.19), color=RED)
    _arrow(axis, (0.64, 0.25), (0.64, 0.47), color=GRAY, dashed=True)

    axis.text(
        0.69,
        0.025,
        "No future artifact, outcome, or realized future cost is exposed online",
        fontsize=9,
        color=RED,
        ha="center",
        fontweight="bold",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURE_ROOT / f"darc_v05_budget_semantics_flow.{suffix}",
            dpi=260 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def _aggregate_rows() -> list[dict[str, str]]:
    with (RESULT_ROOT / "tables" / "aggregate_results.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def efficiency_figure() -> None:
    rows = _aggregate_rows()
    selected = "DARC v0.5 · myopic_targets · i0"
    line_policies = {
        selected: ("DARC v0.5 Myopic i0", ORANGE, "o", 3.1),
        "DARC v0.4 · myopic_targets": ("DARC v0.4 Myopic", BLUE, "s", 1.8),
        "C3 MultiHistory Myopic": ("C3 Myopic", "#7A6FA8", "^", 1.6),
        "C3 MultiHistory Continuation": ("C3 Continuation", GRAY, "D", 1.6),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})

    for policy, (label, color, marker, width) in line_policies.items():
        points = sorted(
            (row for row in rows if row["policy"] == policy),
            key=lambda row: float(row["teacher_tokens_mean"]),
        )
        axes[0].plot(
            [float(row["teacher_tokens_mean"]) / 1000 for row in points],
            [100 * float(row["accuracy_mean"]) for row in points],
            label=label,
            color=color,
            marker=marker,
            linewidth=width,
            markersize=7 if policy == selected else 5.5,
            zorder=5 if policy == selected else 2,
        )
    axes[0].scatter(
        [186.867],
        [52.0408],
        s=190,
        facecolors="none",
        edgecolors=ORANGE,
        linewidths=2.2,
        zorder=6,
    )
    axes[0].annotate(
        "52.04% @ 187K tokens",
        xy=(186.867, 52.0408),
        xytext=(112, 52.75),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.5},
        color=ORANGE,
        fontsize=10,
        fontweight="bold",
    )
    axes[0].set(
        title="(a) Matched-budget validation frontier",
        xlabel="Actual Teacher tokens (thousands)",
        ylabel="Final task accuracy (%)",
        xlim=(-8, 385),
        ylim=(46.2, 53.1),
    )
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")

    full_labels = [
        (selected, "DARC v0.5\nMyopic i0"),
        ("DARC v0.4 · myopic_targets", "DARC v0.4\nMyopic"),
        ("C3 MultiHistory Continuation", "C3\nContinuation"),
        ("C3 MultiHistory Myopic", "C3\nMyopic"),
        ("Fixed VVV", "Always\nVVV"),
    ]
    values: list[float] = []
    accuracies: list[float] = []
    for policy, _label in full_labels:
        row = next(row for row in rows if row["policy"] == policy and float(row["budget_fraction"]) == 1.0)
        values.append(float(row["teacher_tokens_mean"]) / 1000)
        accuracies.append(100 * float(row["accuracy_mean"]))
    colors = [ORANGE, BLUE, GRAY, "#7A6FA8", "#AAB2BD"]
    bars = axes[1].barh(
        [label for _policy, label in full_labels],
        values,
        color=colors,
        height=0.62,
    )
    axes[1].invert_yaxis()
    for bar, tokens, accuracy in zip(bars, values, accuracies, strict=True):
        axes[1].text(
            tokens + 7,
            bar.get_y() + bar.get_height() / 2,
            f"{tokens:.0f}K  |  {accuracy:.2f}%",
            va="center",
            fontsize=9.5,
            color=NAVY,
            fontweight="bold" if bar is bars[0] else "normal",
        )
    axes[1].set(
        title="(b) Full-budget cost at achieved accuracy",
        xlabel="Actual Teacher tokens (thousands)",
        xlim=(0, 535),
    )
    axes[1].text(
        0.98,
        0.03,
        "DARC matches C3 accuracy\nwith 46–48% fewer tokens",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=ORANGE,
        fontweight="bold",
    )
    for axis in axes:
        axis.grid(axis="both", color="#E6E9ED", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=NAVY)
        axis.title.set_color(NAVY)
        axis.title.set_fontweight("bold")
    fig.suptitle(
        "DARC v0.5 improves verification efficiency, not raw validation accuracy",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
        y=1.01,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURE_ROOT / f"darc_v05_efficiency_summary.{suffix}",
            dpi=260 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    architecture_figure()
    efficiency_figure()
    print(FIGURE_ROOT / "darc_v05_budget_semantics_flow.png")
    print(FIGURE_ROOT / "darc_v05_efficiency_summary.png")


if __name__ == "__main__":
    main()
