#!/usr/bin/env python3
"""Build the reproducible LiveCodeBench snapshot used by the dated report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAVY = "#0F172A"
SLATE = "#475569"
MUTED = "#94A3B8"
GRID = "#E2E8F0"
PAPER = "#F8FAFC"
BLUE = "#2563EB"
GREEN = "#10B981"
RED = "#EF4444"
AMBER = "#F59E0B"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_empirical_run(results_dir: Path) -> str:
    manifests = [row for row in read_jsonl(results_dir / "manifests.jsonl") if not row.get("mock_only")]
    if not manifests:
        raise ValueError(f"No empirical manifest in {results_dir}")
    return str(manifests[-1]["run_id"])


def style_axis(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=NAVY, pad=20)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=SLATE, va="bottom")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=SLATE)


def build_dashboard(
    results_dir: Path,
    output_dir: Path,
    run_id: str | None = None,
    output_name: str = "livecodebench_dashboard.png",
) -> Path:
    run_id = run_id or latest_empirical_run(results_dir)
    baselines = [row for row in read_jsonl(results_dir / "baseline.jsonl") if row.get("run_id") == run_id]
    pairs = [
        row
        for row in read_jsonl(results_dir / "intervention_pairs.jsonl")
        if row.get("run_id") == run_id and row.get("status") == "completed"
    ]
    invocations = [
        row
        for row in read_jsonl(results_dir / "llm_invocations.jsonl")
        if row.get("context", {}).get("run_id") == run_id
    ]
    events = [row for row in read_jsonl(results_dir / "events.jsonl") if row.get("run_id") == run_id]
    selected = json.loads((results_dir / "selected_tasks.json").read_text(encoding="utf-8"))["tasks"]
    metadata = {row["task_id"]: row for row in selected}

    planned_pairs = len(baselines) * 3
    completion = [sum(int(row["checkpoint"]) == cp for row in pairs) for cp in (1, 2, 3)]
    checkpoint_complete = [value == len(baselines) for value in completion]
    complete = bool(baselines) and all(checkpoint_complete)
    baseline_groups = ["Overall", "Easy", "Medium", "Hard"]
    baseline_rates = [100 * sum(int(row["passed"]) for row in baselines) / max(1, len(baselines))]
    for difficulty in ("easy", "medium", "hard"):
        rows = [row for row in baselines if metadata[row["task_id"]]["difficulty"] == difficulty]
        baseline_rates.append(100 * sum(int(row["passed"]) for row in rows) / max(1, len(rows)))

    cp_rows = [[row for row in pairs if int(row["checkpoint"]) == cp] for cp in (1, 2, 3)]
    skip_rates = [100 * sum(int(row["y_skip"]) for row in rows) / max(1, len(rows)) for rows in cp_rows]
    verify_rates = [100 * sum(int(row["y_verify"]) for row in rows) / max(1, len(rows)) for rows in cp_rows]
    helpful = [sum(int(row["delta"]) > 0 for row in rows) for rows in cp_rows]
    harmful = [sum(int(row["delta"]) < 0 for row in rows) for rows in cp_rows]
    unchanged = [sum(int(row["delta"]) == 0 for row in rows) for rows in cp_rows]
    teacher = [row for row in invocations if row.get("context", {}).get("role") == "teacher"]
    teacher_tokens = sum(
        int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0)) for row in teacher
    )
    failures = Counter(row.get("record_type") for row in events)

    fig = plt.figure(figsize=(16, 11), facecolor=PAPER)
    grid = fig.add_gridspec(2, 2, left=0.06, right=0.96, top=0.83, bottom=0.08, hspace=0.42, wspace=0.28)
    fig.text(
        0.06,
        0.955,
        f"LiveCodeBench Phase 3 — {'result' if complete else 'preliminary result'} snapshot",
        fontsize=24,
        fontweight="bold",
        color=NAVY,
    )
    fig.text(
        0.06,
        0.918,
        f"{len(baselines)} tasks • {len(pairs)}/{planned_pairs} pairs complete • run {run_id}",
        fontsize=11,
        color=SLATE,
    )
    fig.text(
        0.06,
        0.873,
        f"Baseline {sum(int(row['passed']) for row in baselines)}/{len(baselines)}   |   "
        f"Helpful {sum(helpful)}   Harmful {sum(harmful)}   No change {sum(unchanged)}   |   "
        f"Teacher {len(teacher)} calls / {teacher_tokens:,} tokens",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )

    baseline_ax = fig.add_subplot(grid[0, 0])
    style_axis(
        baseline_ax, "Baseline accuracy by difficulty", "Student workflow without Teacher verification"
    )
    colors = [BLUE, GREEN, AMBER, RED]
    bars = baseline_ax.bar(baseline_groups, baseline_rates, color=colors, width=0.66, zorder=3)
    baseline_ax.set_ylim(0, 65)
    baseline_ax.set_ylabel("Accuracy (%)", color=SLATE)
    for bar, value in zip(bars, baseline_rates, strict=True):
        baseline_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
            fontweight="bold",
            color=NAVY,
        )

    complete_ax = fig.add_subplot(grid[0, 1])
    style_axis(
        complete_ax,
        "Pair completeness audit",
        "All checkpoints complete" if complete else "Incomplete checkpoints remain provisional",
    )
    y = np.arange(3)
    audit_colors = [GREEN if value else AMBER for value in checkpoint_complete]
    bars = complete_ax.barh(y, completion, color=audit_colors, height=0.58, zorder=3)
    target = max(1, len(baselines))
    complete_ax.axvline(target, color=GREEN if complete else RED, linewidth=1.2, linestyle="--")
    complete_ax.set_yticks(y, ["CP1 • Plan", "CP2 • Code", "CP3 • Final"])
    complete_ax.set_xlim(0, target * 1.10)
    complete_ax.invert_yaxis()
    complete_ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    complete_ax.grid(axis="y", visible=False)
    for bar, value in zip(bars, completion, strict=True):
        complete_ax.text(
            value + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value}/{target}",
            va="center",
            fontweight="bold",
            color=NAVY,
        )

    accuracy_ax = fig.add_subplot(grid[1, 0])
    style_axis(
        accuracy_ax,
        "SKIP versus VERIFY accuracy",
        "Hatched bars are provisional" if not complete else "All checkpoint pairs are complete",
    )
    x = np.arange(3)
    width = 0.34
    skip_bars = accuracy_ax.bar(x - width / 2, skip_rates, width, color=MUTED, label="SKIP", zorder=3)
    verify_bars = accuracy_ax.bar(x + width / 2, verify_rates, width, color=BLUE, label="VERIFY", zorder=3)
    for index, is_complete in enumerate(checkpoint_complete):
        if not is_complete:
            for bar in (skip_bars[index], verify_bars[index]):
                bar.set_hatch("///")
                bar.set_alpha(0.65)
    checkpoint_labels = [f"CP{cp}{'' if checkpoint_complete[cp - 1] else '*'}" for cp in (1, 2, 3)]
    accuracy_ax.set_xticks(x, checkpoint_labels)
    accuracy_ax.set_ylim(0, 52)
    accuracy_ax.set_ylabel("Accuracy (%)", color=SLATE)
    accuracy_ax.legend(frameon=False, ncol=2, loc="upper left")
    for bars in (skip_bars, verify_bars):
        for bar in bars:
            accuracy_ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{bar.get_height():.1f}%",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=NAVY,
            )

    effect_ax = fig.add_subplot(grid[1, 1])
    style_axis(
        effect_ax,
        "Observed intervention outcomes",
        "Most completed verifications do not flip final correctness",
    )
    effect_ax.bar(x, unchanged, color=MUTED, width=0.62, label="No change", zorder=3)
    effect_ax.bar(x, helpful, bottom=unchanged, color=GREEN, width=0.62, label="Helpful", zorder=3)
    effect_ax.bar(
        x,
        harmful,
        bottom=np.array(unchanged) + np.array(helpful),
        color=RED,
        width=0.62,
        label="Harmful",
        zorder=3,
    )
    effect_ax.set_xticks(x, checkpoint_labels)
    effect_ax.set_ylabel("Completed pairs", color=SLATE)
    effect_ax.legend(frameon=False, ncol=3, loc="upper right")
    for index, total in enumerate(completion):
        effect_ax.text(
            index,
            total + 2,
            f"{helpful[index]} / {harmful[index]} / {unchanged[index]}",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=NAVY,
        )

    fig.text(
        0.96,
        0.025,
        (
            f"COMPLETE: {len(pairs)}/{planned_pairs} pairs • ready for downstream analysis"
            if complete
            else f"PRELIMINARY: {planned_pairs - len(pairs)} pairs missing • "
            f"paired failures={failures['paired_intervention_failure']} • do not train on this snapshot"
        ),
        ha="right",
        color=GREEN if complete else RED,
        fontsize=9.5,
        fontweight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / output_name
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    snapshot = output_dir / f"{Path(output_name).stem}_checkpoint_metrics.csv"
    with snapshot.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint",
                "pairs",
                "skip_accuracy",
                "verify_accuracy",
                "helpful",
                "harmful",
                "no_change",
            ],
        )
        writer.writeheader()
        for index, cp in enumerate((1, 2, 3)):
            writer.writerow(
                {
                    "checkpoint": cp,
                    "pairs": completion[index],
                    "skip_accuracy": skip_rates[index] / 100,
                    "verify_accuracy": verify_rates[index] / 100,
                    "helpful": helpful[index],
                    "harmful": harmful[index],
                    "no_change": unchanged[index],
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/livecodebench"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_report_241026"))
    parser.add_argument("--run-id")
    parser.add_argument("--output-name", default="livecodebench_dashboard.png")
    args = parser.parse_args()
    print(build_dashboard(args.results_dir, args.output_dir, args.run_id, args.output_name))


if __name__ == "__main__":
    main()
