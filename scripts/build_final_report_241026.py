#!/usr/bin/env python3
# ruff: noqa: E501
"""Build publication-ready summary artifacts for the 2026-10-24 experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "final_report_241026"

RUNS = {
    "dev_v1": "h1_phase1_qwen25_gpu2-b08c6b8c6d-2af13ca7b2",
    "dev_v2": "h1_phase1_qwen25_gpu2-72700d2d7f-f68c87f537",
    "dev_v2_r3": "h1_phase1_qwen25_gpu2-79205f0481-f68c87f537",
    "heldout_v2": "h1_phase1_qwen25_gpu2-09092ebd5e-d03859f54f",
}

NAVY = "#0F172A"
SLATE = "#475569"
MUTED = "#94A3B8"
GRID = "#E2E8F0"
PAPER = "#F8FAFC"
WHITE = "#FFFFFF"
BLUE = "#2563EB"
CYAN = "#0891B2"
GREEN = "#10B981"
GREEN_DARK = "#047857"
RED = "#EF4444"
RED_DARK = "#B91C1C"
AMBER = "#F59E0B"


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    pairs: int
    skip_passes: int
    verify_passes: int
    helpful: int
    harmful: int
    unchanged: int
    teacher_changes: int
    by_checkpoint: dict[int, dict[str, int]]

    @property
    def net(self) -> int:
        return self.helpful - self.harmful

    @property
    def delta_pp(self) -> float:
        return 100 * self.net / self.pairs


def load_metrics(run_id: str) -> RunMetrics:
    path = RESULTS_DIR / "reports" / run_id / "h1_analysis.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_checkpoint: dict[int, dict[str, int]] = {}
    for checkpoint in (1, 2, 3):
        selected = [row for row in rows if int(row["checkpoint"]) == checkpoint]
        by_checkpoint[checkpoint] = {
            "n": len(selected),
            "skip": sum(int(row["y_skip"]) for row in selected),
            "verify": sum(int(row["y_verify"]) for row in selected),
            "helpful": sum(int(row["delta"]) == 1 for row in selected),
            "harmful": sum(int(row["delta"]) == -1 for row in selected),
            "changed": sum(row["teacher_changed"] == "True" for row in selected),
        }

    return RunMetrics(
        run_id=run_id,
        pairs=len(rows),
        skip_passes=sum(int(row["y_skip"]) for row in rows),
        verify_passes=sum(int(row["y_verify"]) for row in rows),
        helpful=sum(int(row["delta"]) == 1 for row in rows),
        harmful=sum(int(row["delta"]) == -1 for row in rows),
        unchanged=sum(int(row["delta"]) == 0 for row in rows),
        teacher_changes=sum(row["teacher_changed"] == "True" for row in rows),
        by_checkpoint=by_checkpoint,
    )


def rounded_panel(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_facecolor(WHITE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=NAVY, pad=14)
    if subtitle:
        ax.text(
            0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            fontsize=9,
            color=SLATE,
            va="bottom",
        )


def add_kpi_card(
    ax: plt.Axes,
    value: str,
    label: str,
    detail: str,
    accent: str,
) -> None:
    ax.set_axis_off()
    card = FancyBboxPatch(
        (0.02, 0.04),
        0.96,
        0.9,
        boxstyle="round,pad=0.018,rounding_size=0.04",
        facecolor=WHITE,
        edgecolor=GRID,
        linewidth=1.1,
        transform=ax.transAxes,
    )
    ax.add_patch(card)
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.04),
            0.025,
            0.9,
            boxstyle="round,pad=0.002,rounding_size=0.02",
            facecolor=accent,
            edgecolor=accent,
            transform=ax.transAxes,
        )
    )
    ax.text(0.09, 0.67, value, transform=ax.transAxes, fontsize=25, fontweight="bold", color=NAVY)
    ax.text(0.09, 0.39, label, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=SLATE)
    ax.text(0.09, 0.17, detail, transform=ax.transAxes, fontsize=8.7, color=MUTED)


def plot_dashboard(metrics: dict[str, RunMetrics]) -> Path:
    heldout = metrics["heldout_v2"]
    fig = plt.figure(figsize=(16, 14), facecolor=PAPER)
    grid = fig.add_gridspec(13, 12, left=0.045, right=0.97, top=0.96, bottom=0.045, hspace=1.25, wspace=0.85)

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.set_axis_off()
    fig.text(
        0.045,
        0.975,
        "Downstream-Aware Verification — Phase 1 Results",
        fontsize=25,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    fig.text(
        0.045,
        0.947,
        "Qwen2.5-Coder-7B child  •  Qwen2.5-Coder-32B teacher  •  HumanEval  •  24 Oct 2026",
        fontsize=11,
        color=SLATE,
        va="top",
    )

    kpi_axes = [fig.add_subplot(grid[1:3, start : start + 3]) for start in (0, 3, 6, 9)]
    add_kpi_card(
        kpi_axes[0], "140", "Held-out tasks attempted", "139 paired-complete • 1 generation failure", BLUE
    )
    add_kpi_card(kpi_axes[1], "417", "Paired interventions", "139 tasks × CP1/CP2/CP3", CYAN)
    add_kpi_card(kpi_axes[2], "+3.4 pp", "VERIFY vs SKIP", "84.7% vs 81.3% paired pass rate", GREEN)
    add_kpi_card(kpi_axes[3], "19 / 5", "Helpful / harmful", "+14 net successful outcomes", AMBER)

    pass_ax = fig.add_subplot(grid[3:7, :6])
    rounded_panel(pass_ax, "Pass rate by checkpoint", "Held-out set; each checkpoint has n=139 matched pairs")
    checkpoints = [1, 2, 3]
    x_positions = list(range(3))
    width = 0.34
    skip_rates = [
        100 * heldout.by_checkpoint[cp]["skip"] / heldout.by_checkpoint[cp]["n"] for cp in checkpoints
    ]
    verify_rates = [
        100 * heldout.by_checkpoint[cp]["verify"] / heldout.by_checkpoint[cp]["n"] for cp in checkpoints
    ]
    skip_bars = pass_ax.bar(
        [x - width / 2 for x in x_positions],
        skip_rates,
        width,
        label="SKIP",
        color=MUTED,
        zorder=3,
    )
    verify_bars = pass_ax.bar(
        [x + width / 2 for x in x_positions],
        verify_rates,
        width,
        label="VERIFY",
        color=BLUE,
        zorder=3,
    )
    pass_ax.set_xticks(x_positions, ["CP1 • Plan", "CP2 • Code", "CP3 • Final"])
    pass_ax.set_ylim(74, 89)
    pass_ax.set_ylabel("Pass rate (%)", color=SLATE)
    pass_ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    pass_ax.tick_params(colors=SLATE)
    pass_ax.legend(frameon=False, loc="upper left", ncol=2)
    for bars in (skip_bars, verify_bars):
        for bar in bars:
            pass_ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.35,
                f"{bar.get_height():.1f}%",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=NAVY,
            )

    effect_ax = fig.add_subplot(grid[3:7, 6:])
    rounded_panel(
        effect_ax, "Helpful vs harmful interventions", "Positive and negative outcome flips by checkpoint"
    )
    helpful = [heldout.by_checkpoint[cp]["helpful"] for cp in checkpoints]
    harmful = [-heldout.by_checkpoint[cp]["harmful"] for cp in checkpoints]
    effect_ax.barh(x_positions, helpful, color=GREEN, height=0.38, label="Helpful", zorder=3)
    effect_ax.barh(x_positions, harmful, color=RED, height=0.38, label="Harmful", zorder=3)
    effect_ax.axvline(0, color=NAVY, linewidth=0.9)
    effect_ax.set_yticks(x_positions, ["CP1 • Plan", "CP2 • Code", "CP3 • Final"])
    effect_ax.invert_yaxis()
    effect_ax.set_xlim(-4, 10)
    effect_ax.set_xlabel("Outcome flips", color=SLATE)
    effect_ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    effect_ax.tick_params(colors=SLATE)
    effect_ax.legend(frameon=False, loc="lower right", ncol=2)
    for index, (good, bad) in enumerate(zip(helpful, harmful, strict=True)):
        effect_ax.text(good + 0.2, index, f"+{good}", va="center", color=GREEN_DARK, fontweight="bold")
        effect_ax.text(bad - 0.2, index, f"{bad}", va="center", ha="right", color=RED_DARK, fontweight="bold")

    progress_ax = fig.add_subplot(grid[7:10, :6])
    rounded_panel(
        progress_ax,
        "Effect across experiment stages",
        "Net paired gain; R=3 repeats are not independent evidence",
    )
    stage_keys = ["dev_v1", "dev_v2", "dev_v2_r3", "heldout_v2"]
    stage_labels = ["Prompt v1\n24 dev", "Prompt v2\n24 dev", "v2, R=3\n24 dev", "Prompt v2\n139 held-out"]
    stage_values = [metrics[key].delta_pp for key in stage_keys]
    stage_colors = [MUTED, CYAN, CYAN, GREEN]
    bars = progress_ax.bar(range(4), stage_values, color=stage_colors, width=0.62, zorder=3)
    progress_ax.axhline(0, color=NAVY, linewidth=0.9)
    progress_ax.set_xticks(range(4), stage_labels)
    progress_ax.set_ylabel("VERIFY − SKIP (percentage points)", color=SLATE)
    progress_ax.set_ylim(-0.6, 4.2)
    progress_ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    progress_ax.tick_params(colors=SLATE)
    for bar, value in zip(bars, stage_values, strict=True):
        progress_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.16,
            f"{value:+.1f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=NAVY,
        )

    change_ax = fig.add_subplot(grid[7:10, 6:])
    rounded_panel(
        change_ax, "What happened when Teacher edited", "101 actual artifact changes in the held-out run"
    )
    segments = [
        heldout.helpful,
        heldout.unchanged - (heldout.pairs - heldout.teacher_changes),
        heldout.harmful,
    ]
    labels = ["Helpful", "No outcome change", "Harmful"]
    colors = [GREEN, MUTED, RED]
    left = 0
    for value, label, color in zip(segments, labels, colors, strict=True):
        change_ax.barh([0], [value], left=left, color=color, height=0.42, label=label)
        if value >= 5:
            change_ax.text(
                left + value / 2,
                0,
                f"{value}\n({100 * value / heldout.teacher_changes:.1f}%)",
                ha="center",
                va="center",
                color=WHITE if color != MUTED else NAVY,
                fontsize=9,
                fontweight="bold",
            )
        left += value
    change_ax.set_xlim(0, heldout.teacher_changes)
    change_ax.set_ylim(-0.7, 0.7)
    change_ax.set_yticks([])
    change_ax.set_xlabel("Teacher changes", color=SLATE)
    change_ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3)
    change_ax.text(
        0.01,
        0.92,
        "Critic correlation r = 0.223  •  134/139 tasks had ranking ties",
        transform=change_ax.transAxes,
        fontsize=9.2,
        color=SLATE,
        va="top",
    )

    cases_ax = fig.add_subplot(grid[10:13, :])
    cases_ax.set_axis_off()
    cases_ax.text(0, 1.04, "Concrete interventions from the logs", fontsize=14, fontweight="bold", color=NAVY)
    cases = [
        (
            0.0,
            GREEN,
            "HELPFUL • HumanEval/26 • CP1",
            "Plan: deduplicate values",
            "→ count frequencies; retain count == 1",
            "FAIL → PASS",
        ),
        (
            0.342,
            GREEN,
            "HELPFUL • HumanEval/93 • CP2",
            "shifted_vowels[char]",
            "→ shifted_vowels[char].swapcase()",
            "FAIL → PASS",
        ),
        (
            0.684,
            RED,
            "HARMFUL • HumanEval/137 • CP2",
            "return original input '1,5'",
            "→ returns normalized string '1.5'",
            "PASS → FAIL",
        ),
    ]
    for x_pos, color, heading, before, after, outcome in cases:
        card = FancyBboxPatch(
            (x_pos, 0.08),
            0.316,
            0.79,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=WHITE,
            edgecolor=GRID,
            linewidth=1.0,
            transform=cases_ax.transAxes,
        )
        cases_ax.add_patch(card)
        cases_ax.text(
            x_pos + 0.022,
            0.75,
            heading,
            transform=cases_ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            color=color,
        )
        cases_ax.text(
            x_pos + 0.022,
            0.53,
            before,
            transform=cases_ax.transAxes,
            fontsize=9.4,
            color=SLATE,
            family="monospace",
        )
        cases_ax.text(
            x_pos + 0.022,
            0.36,
            after,
            transform=cases_ax.transAxes,
            fontsize=9.4,
            color=NAVY,
            family="monospace",
            fontweight="bold",
        )
        cases_ax.text(
            x_pos + 0.022,
            0.16,
            outcome,
            transform=cases_ax.transAxes,
            fontsize=12,
            color=color,
            fontweight="bold",
        )

    fig.text(
        0.97,
        0.012,
        "Held-out run: 140 attempted, 139 paired-complete; HumanEval/38 failed during child JSON generation.",
        ha="right",
        fontsize=8.6,
        color=SLATE,
    )
    output = OUTPUT_DIR / "final_results_dashboard.png"
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def add_text_card(
    ax: plt.Axes,
    title: str,
    body: str,
    accent: str,
    *,
    monospace: bool = False,
) -> None:
    ax.set_axis_off()
    card = FancyBboxPatch(
        (0.01, 0.03),
        0.98,
        0.94,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=WHITE,
        edgecolor=GRID,
        linewidth=1.0,
        transform=ax.transAxes,
    )
    ax.add_patch(card)
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.03),
            0.018,
            0.94,
            boxstyle="round,pad=0.001,rounding_size=0.01",
            facecolor=accent,
            edgecolor=accent,
            transform=ax.transAxes,
        )
    )
    ax.text(0.07, 0.84, title, transform=ax.transAxes, fontsize=11, fontweight="bold", color=accent, va="top")
    ax.text(
        0.07,
        0.67,
        body,
        transform=ax.transAxes,
        fontsize=9.3,
        color=NAVY,
        va="top",
        linespacing=1.45,
        family="monospace" if monospace else "sans-serif",
    )


def plot_case_studies() -> Path:
    fig = plt.figure(figsize=(16, 10), facecolor=PAPER)
    grid = fig.add_gridspec(4, 4, left=0.04, right=0.97, top=0.91, bottom=0.06, hspace=0.42, wspace=0.35)
    fig.text(
        0.04,
        0.955,
        "Teacher interventions: what improved, and what broke",
        fontsize=23,
        fontweight="bold",
        color=NAVY,
    )
    fig.text(
        0.04, 0.92, "Verbatim logic extracted from paired intervention artifacts", fontsize=10.5, color=SLATE
    )

    rows = [
        {
            "accent": GREEN,
            "identity": "HELPFUL\nHumanEval/26 • CP1\nPlan verification",
            "before": "BEFORE — wrong contract\n\nKeep the first occurrence\nwhen value is not already\nin result.\n\nThis performs deduplication.",
            "after": "TEACHER CORRECTION\n\nCount every value first,\nthen retain values where\nfrequency == 1.\n\nThis removes all duplicates.",
            "effect": "DOWNSTREAM EFFECT\n\nCoder follows corrected plan.\n\nSKIP  0  →  VERIFY  1\nFAIL         PASS",
        },
        {
            "accent": GREEN,
            "identity": "HELPFUL\nHumanEval/93 • CP2\nCode verification",
            "before": "BEFORE — missed case swap\n\nresult.append(\n    shifted_vowels[char]\n)",
            "after": "TEACHER CORRECTION\n\nresult.append(\n    shifted_vowels[char]\n        .swapcase()\n)",
            "effect": "DOWNSTREAM EFFECT\n\nShifted vowels now obey the\nfull transformation contract.\n\nSKIP  0  →  VERIFY  1",
        },
        {
            "accent": RED,
            "identity": "HARMFUL\nHumanEval/137 • CP2\nCode verification",
            "before": "BEFORE — correct behavior\n\nCompare normalized floats,\nbut return the original input\na or b (for example '1,5').",
            "after": "TEACHER OVER-CORRECTION\n\nLabels a normalized value as\n'original_a', then returns\n'1.5' instead of '1,5'.",
            "effect": "DOWNSTREAM EFFECT\n\nRepresentation contract is\nbroken by normalization.\n\nSKIP  1  →  VERIFY  0\nPASS         FAIL",
        },
    ]

    for row_index, row in enumerate(rows, start=1):
        add_text_card(fig.add_subplot(grid[row_index, 0]), "CASE", "", row["accent"])
        identity_ax = fig.axes[-1]
        identity_ax.text(
            0.07,
            0.66,
            row["identity"],
            transform=identity_ax.transAxes,
            fontsize=13,
            fontweight="bold",
            color=row["accent"],
            va="top",
            linespacing=1.35,
        )
        add_text_card(
            fig.add_subplot(grid[row_index, 1]), "Observed artifact", row["before"], SLATE, monospace=True
        )
        add_text_card(
            fig.add_subplot(grid[row_index, 2]), "Teacher output", row["after"], row["accent"], monospace=True
        )
        add_text_card(fig.add_subplot(grid[row_index, 3]), "Measured outcome", row["effect"], row["accent"])

    header_labels = ["CASE", "BEFORE", "AFTER", "EVALUATION"]
    for column, label in enumerate(header_labels):
        header_ax = fig.add_subplot(grid[0, column])
        header_ax.set_axis_off()
        header_ax.text(0.02, 0.15, label, fontsize=10, fontweight="bold", color=SLATE)
        header_ax.plot([0.02, 0.98], [0.05, 0.05], color=GRID, linewidth=1.5, transform=header_ax.transAxes)

    output = OUTPUT_DIR / "teacher_intervention_case_studies.png"
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def write_summary_csv(metrics: dict[str, RunMetrics]) -> Path:
    output = OUTPUT_DIR / "experiment_summary.csv"
    labels = {
        "dev_v1": "Dev 24 / teacher prompt v1 / R=1",
        "dev_v2": "Dev 24 / teacher prompt v2 / R=1",
        "dev_v2_r3": "Dev 24 / teacher prompt v2 / R=3",
        "heldout_v2": "Held-out 140 attempted, 139 paired-complete / prompt v2 / R=1",
    }
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "run_id",
                "pairs",
                "skip_passes",
                "verify_passes",
                "helpful",
                "harmful",
                "unchanged",
                "net",
                "delta_percentage_points",
                "teacher_changes",
            ],
        )
        writer.writeheader()
        for key, label in labels.items():
            run = metrics[key]
            writer.writerow(
                {
                    "stage": label,
                    "run_id": run.run_id,
                    "pairs": run.pairs,
                    "skip_passes": run.skip_passes,
                    "verify_passes": run.verify_passes,
                    "helpful": run.helpful,
                    "harmful": run.harmful,
                    "unchanged": run.unchanged,
                    "net": run.net,
                    "delta_percentage_points": f"{run.delta_pp:.4f}",
                    "teacher_changes": run.teacher_changes,
                }
            )
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {key: load_metrics(run_id) for key, run_id in RUNS.items()}
    outputs = [plot_dashboard(metrics), plot_case_studies(), write_summary_csv(metrics)]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
