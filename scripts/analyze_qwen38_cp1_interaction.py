#!/usr/bin/env python3
"""Build the 210-task Qwen3.8 CP1 downstream-interaction audit figure.

This script only reads completed Full-8 trajectory tables.  It does not call a
model, evaluate code, train a controller, or access a frozen test split.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOTS = [
    "results/darc_qwen38_vvv40_full8",
    "results/darc_qwen38_full20",
    "results/darc_qwen38_full40_gpu0",
    "results/darc_qwen38_full40_gpu2",
    "results/darc_qwen38_full30_prospective_gpu2",
    "results/darc_qwen38_full30_extra_gpu0",
    "results/darc_qwen38_full10_extra_gpu0",
]
SEQUENCES = {"SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV"}


def _as_bool(value: str) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes"})


def load_tasks() -> dict[str, dict[str, object]]:
    tasks: dict[str, dict[str, object]] = {}
    for relative_root in RUN_ROOTS:
        path = ROOT / relative_root / "tables" / "trajectory_comparison.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                task = tasks.setdefault(
                    row["task_key"],
                    {"dataset": row["dataset"], "outcomes": {}},
                )
                outcomes = task["outcomes"]
                assert isinstance(outcomes, dict)
                sequence = row["sequence"]
                outcome = _as_bool(row["new_qwen38_passed"])
                if sequence in outcomes and outcomes[sequence] != outcome:
                    raise RuntimeError(
                        f"Conflicting duplicate for {row['task_key']} / {sequence}"
                    )
                outcomes[sequence] = outcome

    if len(tasks) != 210:
        raise RuntimeError(f"Expected 210 distinct tasks, found {len(tasks)}")
    incomplete = [
        key
        for key, task in tasks.items()
        if set(task["outcomes"]) != SEQUENCES
    ]
    if incomplete:
        raise RuntimeError(f"Incomplete Full-8 tasks: {incomplete[:5]}")
    return tasks


def effect(outcomes: dict[str, int], suffix: str) -> int:
    """Effect of CP1 Verify while holding CP2/CP3 suffix fixed."""
    return outcomes["V" + suffix] - outcomes["S" + suffix]


def bootstrap_ci(values: np.ndarray, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def main() -> None:
    tasks = load_tasks()
    output_root = ROOT / "results" / "qwen38_downstream_interaction_210"
    figure_dir = output_root / "figures"
    table_dir = output_root / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    contexts = [
        ("SS", "CP2 Skip, CP3 Skip"),
        ("SV", "CP2 Skip, CP3 Verify"),
        ("VS", "CP2 Verify, CP3 Skip"),
        ("VV", "CP2 Verify, CP3 Verify"),
    ]
    arrays: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, object]] = []
    for suffix, label in contexts:
        values = np.asarray(
            [effect(task["outcomes"], suffix) for task in tasks.values()],
            dtype=float,
        )
        arrays[suffix] = values
        counts = Counter(values.astype(int))
        low, high = bootstrap_ci(values)
        summary_rows.append(
            {
                "downstream_context": label,
                "suffix": suffix,
                "tasks": len(values),
                "helpful": counts[1],
                "harmful": counts[-1],
                "no_change": counts[0],
                "net_benefit_pp": 100 * values.mean(),
                "bootstrap_ci_low_pp": 100 * low,
                "bootstrap_ci_high_pp": 100 * high,
            }
        )

    exact_rows: list[dict[str, object]] = []
    for task_key, task in sorted(tasks.items()):
        value = int(effect(task["outcomes"], "SV"))
        exact_rows.append(
            {
                "task_key": task_key,
                "dataset": task["dataset"],
                "SSV": task["outcomes"]["SSV"],
                "VSV": task["outcomes"]["VSV"],
                "cp1_effect": value,
                "effect_label": {1: "Helpful", 0: "No-change", -1: "Harmful"}[value],
            }
        )

    with (table_dir / "cp1_context_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (table_dir / "ssv_vsv_task_effects.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exact_rows[0]))
        writer.writeheader()
        writer.writerows(exact_rows)

    context_dependent = 0
    strict_reversal = 0
    for task in tasks.values():
        values = [effect(task["outcomes"], suffix) for suffix, _ in contexts]
        context_dependent += len(set(values)) > 1
        strict_reversal += min(values) < 0 < max(values)

    cp2_skip_transition = Counter(
        (int(arrays["SS"][index]), int(arrays["SV"][index]))
        for index in range(len(tasks))
    )
    transition = np.asarray(
        [
            [cp2_skip_transition[(row, col)] for col in (-1, 0, 1)]
            for row in (-1, 0, 1)
        ]
    )
    interaction = arrays["SV"] - arrays["SS"]
    interaction_low, interaction_high = bootstrap_ci(interaction)

    dataset_counts = Counter(str(task["dataset"]) for task in tasks.values())
    result = {
        "tasks": len(tasks),
        "dataset_counts": dict(dataset_counts),
        "context_dependent_tasks": context_dependent,
        "strict_context_reversal_tasks": strict_reversal,
        "ssv_vsv": {
            "helpful": sum(row["effect_label"] == "Helpful" for row in exact_rows),
            "harmful": sum(row["effect_label"] == "Harmful" for row in exact_rows),
            "no_change": sum(row["effect_label"] == "No-change" for row in exact_rows),
        },
        "cp1_cp3_interaction_cp2_skip_pp": 100 * interaction.mean(),
        "interaction_bootstrap_95_ci_pp": [
            100 * interaction_low,
            100 * interaction_high,
        ],
    }
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#334155",
        }
    )
    fig = plt.figure(figsize=(15.8, 5.8), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.08, 0.95], wspace=0.34)

    # Panel A: paired net effect and task-bootstrap confidence intervals.
    ax = fig.add_subplot(grid[0, 0])
    labels = [label.replace(", ", "\n") for _, label in contexts]
    means = np.asarray([row["net_benefit_pp"] for row in summary_rows])
    lows = np.asarray([row["bootstrap_ci_low_pp"] for row in summary_rows])
    highs = np.asarray([row["bootstrap_ci_high_pp"] for row in summary_rows])
    colors = ["#167D9A" if value >= 0 else "#D1495B" for value in means]
    positions = np.arange(len(means))[::-1]
    ax.axvline(0, color="#64748B", linewidth=1.2, zorder=1)
    ax.barh(positions, means, color=colors, height=0.52, zorder=2)
    ax.errorbar(
        means,
        positions,
        xerr=np.vstack([means - lows, highs - means]),
        fmt="none",
        ecolor="#0F172A",
        elinewidth=1.25,
        capsize=3,
        zorder=3,
    )
    for y, value in zip(positions, means):
        offset = 1.15 if value >= 0 else -1.15
        ax.text(
            value + offset,
            y,
            f"{value:+.2f} pp",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            fontweight="bold",
            color="#0F172A",
        )
    ax.set_yticks(positions, labels)
    ax.set_xlim(-14, 36)
    ax.set_xlabel("Net final-accuracy effect of Verify at CP1 (pp)")
    ax.set_title("a  CP1 utility depends on downstream actions", loc="left")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Panel B: effect transition when only the future CP3 decision changes.
    ax = fig.add_subplot(grid[0, 1])
    image = ax.imshow(np.sqrt(transition), cmap="Blues", vmin=0)
    for row in range(3):
        for col in range(3):
            value = int(transition[row, col])
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white" if image.norm(np.sqrt(value)) > 0.48 else "#0F172A",
            )
    effect_labels = ["Harmful", "No-change", "Helpful"]
    ax.set_xticks(range(3), effect_labels, rotation=18, ha="right")
    ax.set_yticks(range(3), effect_labels)
    ax.set_xlabel("CP1 effect when CP3 = Verify")
    ax.set_ylabel("CP1 effect when CP3 = Skip")
    ax.set_title("b  CP3 changes CP1's observed value", loc="left")
    ax.text(
        0.5,
        -0.31,
        "CP2 fixed to Skip · off-diagonal = context-dependent",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#64748B",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # Panel C: presentation-ready evidence cards.
    ax = fig.add_subplot(grid[0, 2])
    ax.axis("off")
    ax.set_title("c  Evidence for non-myopic verification", loc="left", pad=10)
    cards = [
        ("83 / 210", "tasks have context-dependent\nCP1 utility", "#0F4C81"),
        ("8 / 210", "tasks contain a strict Helpful ↔\nHarmful reversal across contexts", "#7C3AED"),
        ("16 vs 14", "Helpful vs Harmful when CP3\nis already verified (SSV → VSV)", "#B45309"),
        (
            "−21.90 pp",
            "CP1×CP3 interaction\n95% CI [−29.05, −14.76]",
            "#B42318",
        ),
    ]
    y_positions = [0.84, 0.62, 0.40, 0.18]
    for (headline, description, color), y in zip(cards, y_positions):
        ax.add_patch(
            plt.Rectangle(
                (0.0, y - 0.105),
                1.0,
                0.18,
                transform=ax.transAxes,
                facecolor="#F8FAFC",
                edgecolor="#E2E8F0",
                linewidth=0.9,
                clip_on=False,
            )
        )
        ax.text(
            0.055,
            y + 0.012,
            headline,
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            color=color,
            va="center",
        )
        ax.text(
            0.055,
            y - 0.060,
            description,
            transform=ax.transAxes,
            fontsize=9.2,
            color="#475569",
            va="center",
            linespacing=1.2,
        )
    ax.text(
        0.0,
        -0.02,
        "Observed paired trajectories; 210 stratified training tasks.\nNot a population prevalence estimate.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#64748B",
        va="top",
    )

    fig.suptitle(
        "The value of early verification is downstream-dependent",
        x=0.03,
        y=1.04,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.03,
        0.995,
        "Qwen3.8 Teacher · 210 tasks with all 8 branch-consistent trajectories · task-level paired analysis",
        ha="left",
        fontsize=10.5,
        color="#64748B",
    )
    fig.subplots_adjust(top=0.87, bottom=0.18, left=0.08, right=0.985)
    png_path = figure_dir / "cp1_downstream_interaction.png"
    pdf_path = figure_dir / "cp1_downstream_interaction.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
