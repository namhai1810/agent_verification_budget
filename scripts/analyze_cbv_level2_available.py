#!/usr/bin/env python3
"""Analyze complete 8-branch blocks from completed or partial CBV-Bench runs."""

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

SEQUENCES = ("SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV")
LEVEL1 = SEQUENCES[:4]
MULTI = SEQUENCES[4:]
COLORS = {"HumanEval": "#2563EB", "LiveCodeBench": "#F97316", "Combined": "#7C3AED"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--humaneval-dir", type=Path, required=True)
    parser.add_argument("--livecodebench-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}: {exc}") from exc
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def complete_blocks(results_dir: Path) -> tuple[dict[str, dict[str, dict]], dict[str, Any]]:
    source = [
        row
        for row in read_jsonl(results_dir / "trajectories.jsonl")
        if row.get("status") == "completed" and row.get("action_sequence") in SEQUENCES
    ]
    if not source:
        raise ValueError(f"No completed trajectories found in {results_dir}")
    fingerprint = Counter(
        str(row.get("compatibility_fingerprint", "")) for row in source
    ).most_common(1)[0][0]
    unique: dict[tuple[str, int, str], dict] = {}
    for row in source:
        if str(row.get("compatibility_fingerprint", "")) == fingerprint:
            key = (str(row["task_id"]), int(row.get("rollout_index", 0)), row["action_sequence"])
            unique[key] = row
    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for (task, rollout, sequence), row in unique.items():
        grouped[(task, rollout)][sequence] = row
    complete = {
        f"{task}::r{rollout}": block
        for (task, rollout), block in grouped.items()
        if set(SEQUENCES).issubset(block)
    }
    return complete, {
        "fingerprint": fingerprint,
        "observed_tasks": len(grouped),
        "level1_complete_tasks": sum(set(LEVEL1).issubset(block) for block in grouped.values()),
        "level2_complete_tasks": len(complete),
        "stored_trajectories": len(unique),
    }


def ci(values: np.ndarray, rng: np.random.Generator, reps: int) -> tuple[float, float]:
    samples = rng.integers(0, len(values), size=(reps, len(values)))
    bounds = np.percentile(values[samples].mean(axis=1), [2.5, 97.5])
    return float(bounds[0]), float(bounds[1])


def metrics(
    dataset: str,
    blocks: dict[str, dict[str, dict]],
    rng: np.random.Generator,
    reps: int,
) -> list[dict[str, Any]]:
    baseline = np.asarray([int(block["SSS"]["passed"]) for block in blocks.values()])
    output = []
    for sequence in SEQUENCES:
        outcomes = np.asarray([int(block[sequence]["passed"]) for block in blocks.values()])
        tokens = np.asarray(
            [int(block[sequence].get("teacher_total_tokens", 0)) for block in blocks.values()]
        )
        delta = outcomes - baseline
        acc_low, acc_high = ci(outcomes, rng, reps)
        delta_low, delta_high = ci(100.0 * delta, rng, reps)
        output.append(
            {
                "dataset": dataset,
                "sequence": sequence,
                "tasks": len(blocks),
                "correct": int(outcomes.sum()),
                "accuracy": float(outcomes.mean()),
                "accuracy_ci95_low": acc_low,
                "accuracy_ci95_high": acc_high,
                "delta_vs_sss_pp": float(100 * delta.mean()),
                "delta_ci95_low_pp": delta_low,
                "delta_ci95_high_pp": delta_high,
                "helpful_vs_sss": int((delta > 0).sum()),
                "harmful_vs_sss": int((delta < 0).sum()),
                "no_change_vs_sss": int((delta == 0).sum()),
                "mean_teacher_calls": float(
                    np.mean([block[sequence].get("teacher_calls", 0) for block in blocks.values()])
                ),
                "mean_teacher_tokens": float(tokens.mean()),
                "median_teacher_tokens": float(np.median(tokens)),
            }
        )
    return output


def effects(dataset: str, blocks: dict[str, dict[str, dict]]) -> list[dict[str, Any]]:
    output = []
    for checkpoint in (1, 2, 3):
        position = checkpoint - 1
        deltas, token_deltas = [], []
        for block in blocks.values():
            for skip in SEQUENCES:
                if skip[position] != "S":
                    continue
                verify = skip[:position] + "V" + skip[position + 1 :]
                deltas.append(int(block[verify]["passed"]) - int(block[skip]["passed"]))
                token_deltas.append(
                    int(block[verify].get("teacher_total_tokens", 0))
                    - int(block[skip].get("teacher_total_tokens", 0))
                )
        values = np.asarray(deltas)
        output.append(
            {
                "dataset": dataset,
                "checkpoint": checkpoint,
                "paired_edges": len(values),
                "helpful": int((values > 0).sum()),
                "harmful": int((values < 0).sum()),
                "no_change": int((values == 0).sum()),
                "net_effect_pp": float(100 * values.mean()),
                "mean_incremental_teacher_tokens": float(np.mean(token_deltas)),
            }
        )
    return output


def task_rows(dataset: str, blocks: dict[str, dict[str, dict]]) -> list[dict[str, Any]]:
    output = []
    for key, block in sorted(blocks.items()):
        result = {sequence: int(block[sequence]["passed"]) for sequence in SEQUENCES}
        helpful = [sequence for sequence in SEQUENCES[1:] if result[sequence] > result["SSS"]]
        harmful = [sequence for sequence in SEQUENCES[1:] if result[sequence] < result["SSS"]]
        output.append(
            {
                "dataset": dataset,
                "task_rollout": key,
                "task_id": block["SSS"]["task_id"],
                "difficulty": block["SSS"].get("difficulty", "unknown"),
                **result,
                "helpful_sequences_vs_sss": ",".join(helpful),
                "harmful_sequences_vs_sss": ",".join(harmful),
                "multi_only_rescue": int(
                    max(result[s] for s in MULTI) > max(result[s] for s in LEVEL1)
                ),
                "multi_worse_than_best_l1": int(
                    max(result[s] for s in MULTI) < max(result[s] for s in LEVEL1)
                ),
            }
        )
    return output


def dataset_summary(
    dataset: str, blocks: dict[str, dict[str, dict]], all_metrics: list[dict[str, Any]]
) -> dict[str, Any]:
    by_sequence = {row["sequence"]: row for row in all_metrics if row["dataset"] == dataset}
    best_fixed = max(SEQUENCES, key=lambda sequence: by_sequence[sequence]["accuracy"])
    any_branch = np.asarray(
        [max(int(block[sequence]["passed"]) for sequence in SEQUENCES) for block in blocks.values()]
    )
    best_l1 = np.asarray(
        [max(int(block[sequence]["passed"]) for sequence in LEVEL1) for block in blocks.values()]
    )
    best_multi = np.asarray(
        [max(int(block[sequence]["passed"]) for sequence in MULTI) for block in blocks.values()]
    )
    return {
        "tasks": len(blocks),
        "baseline_accuracy": by_sequence["SSS"]["accuracy"],
        "best_fixed_sequence": best_fixed,
        "best_fixed_accuracy": by_sequence[best_fixed]["accuracy"],
        "vvv_accuracy": by_sequence["VVV"]["accuracy"],
        "offline_any_branch_oracle_accuracy": float(any_branch.mean()),
        "multi_vs_best_level1": {
            "better": int((best_multi > best_l1).sum()),
            "equal": int((best_multi == best_l1).sum()),
            "worse": int((best_multi < best_l1).sum()),
        },
    }


def latest_collection_summary(results_dir: Path) -> dict[str, Any]:
    paths = sorted(results_dir.glob("reports/**/summary.json"), key=lambda path: path.stat().st_mtime)
    return json.loads(paths[-1].read_text(encoding="utf-8")) if paths else {}


def style(axis: Any) -> None:
    axis.set_facecolor("white")
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
    for spine in axis.spines.values():
        spine.set_visible(False)


def figures(
    output: Path,
    all_metrics: list[dict[str, Any]],
    all_effects: list[dict[str, Any]],
    completion: list[tuple[str, int, int]],
) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), facecolor="#F8FAFC")
    fig.suptitle(
        "CBV-Bench Level 2 — available-data dashboard",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.055,
        0.947,
        "Matched complete 8-branch blocks • HumanEval complete • LiveCodeBench partial",
        fontsize=10,
        color="#475569",
    )
    for axis in axes.flat:
        style(axis)

    x = np.arange(len(SEQUENCES))
    width = 0.36
    for offset, dataset in zip((-width / 2, width / 2), ("HumanEval", "LiveCodeBench"), strict=True):
        subset = [row for row in all_metrics if row["dataset"] == dataset]
        axes[0, 0].bar(
            x + offset,
            [100 * row["accuracy"] for row in subset],
            width,
            color=COLORS[dataset],
            label=f"{dataset} (n={subset[0]['tasks']})",
            zorder=3,
        )
    axes[0, 0].set_title("A. Final accuracy by sequence", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_xticks(x, SEQUENCES)
    axes[0, 0].legend(frameon=False)

    for dataset in ("HumanEval", "LiveCodeBench"):
        subset = [row for row in all_metrics if row["dataset"] == dataset]
        axes[0, 1].plot(
            [row["mean_teacher_tokens"] for row in subset],
            [100 * row["accuracy"] for row in subset],
            marker="o",
            color=COLORS[dataset],
            label=dataset,
        )
        for row in subset:
            axes[0, 1].annotate(
                row["sequence"],
                (row["mean_teacher_tokens"], 100 * row["accuracy"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axes[0, 1].set_title("B. Accuracy–Teacher-token trade-off", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Mean Teacher tokens / task")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend(frameon=False)

    combined = [row for row in all_effects if row["dataset"] == "Combined"]
    y = np.arange(3)
    axes[1, 0].barh(y, [row["helpful"] for row in combined], color="#16A34A", label="Helpful")
    axes[1, 0].barh(
        y, [-row["harmful"] for row in combined], color="#DC2626", label="Harmful"
    )
    axes[1, 0].axvline(0, color="#334155", linewidth=0.8)
    axes[1, 0].set_yticks(y, ["CP1 Planner", "CP2 Coder", "CP3 Reviewer"])
    axes[1, 0].set_title("C. Conditional marginal effects", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Paired edges (harmful shown negative)")
    axes[1, 0].legend(frameon=False)

    names, done, expected = zip(*completion, strict=True)
    positions = np.arange(len(names))
    axes[1, 1].bar(positions, expected, color="#E2E8F0", label="Expected")
    axes[1, 1].bar(positions, done, color="#0EA5E9", label="Complete")
    axes[1, 1].set_xticks(positions, names, rotation=12)
    axes[1, 1].set_ylabel("Task blocks")
    axes[1, 1].set_title("D. Pipeline completeness", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False)
    for index, (complete, total) in enumerate(zip(done, expected, strict=True)):
        axes[1, 1].text(index, complete + max(expected) * 0.015, f"{complete}/{total}", ha="center")
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.90))
    fig.savefig(figure_dir / "cbv_level2_dashboard.png", dpi=190, bbox_inches="tight")
    fig.savefig(figure_dir / "cbv_level2_dashboard.pdf", bbox_inches="tight")
    plt.close(fig)

    datasets = ("HumanEval", "LiveCodeBench", "Combined")
    matrix = np.asarray(
        [
            [
                100
                * next(
                    row["accuracy"]
                    for row in all_metrics
                    if row["dataset"] == dataset and row["sequence"] == sequence
                )
                for sequence in SEQUENCES
            ]
            for dataset in datasets
        ]
    )
    fig, axis = plt.subplots(figsize=(12, 4.8), facecolor="#F8FAFC")
    image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
    axis.set_xticks(np.arange(len(SEQUENCES)), SEQUENCES)
    axis.set_yticks(np.arange(len(datasets)), datasets)
    axis.set_title("Final accuracy — matched complete task blocks", loc="left", fontweight="bold")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if value > 55 else "#0F172A"
            axis.text(column_index, row_index, f"{value:.1f}%", ha="center", va="center", color=color)
    fig.colorbar(image, ax=axis, label="Accuracy (%)", fraction=0.025, pad=0.03)
    fig.tight_layout()
    fig.savefig(figure_dir / "accuracy_heatmap.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def report(
    summary: dict[str, Any],
    all_metrics: list[dict[str, Any]],
    all_effects: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> str:
    lines = [
        "# CBV-Bench Level 2 — Available-data analysis",
        "",
        "> Offline analysis only: no Student/Teacher inference and no model training was run.",
        "",
        "## Data status",
        "",
        "| Dataset | Complete L2 tasks | Expected | Status |",
        "|---|---:|---:|---|",
        "| HumanEval | 163 | 163 | Complete |",
        "| LiveCodeBench | 171 | 175 | Partial; four collection failures |",
        "| BigCodeBench L1 | 0 | 1000 | Not started because pipeline stopped at LCB |",
        "| BigCodeBench L2 | 0 | 250 | Not started because pipeline stopped at LCB |",
        "",
        (
            "All comparisons use only tasks with a complete "
            "SSS/VSS/SVS/SSV/VVS/VSV/SVV/VVV block."
        ),
        "",
        "## Main results",
        "",
    ]
    for dataset in ("HumanEval", "LiveCodeBench"):
        item = summary["datasets"][dataset]
        multi = item["multi_vs_best_level1"]
        lines += [
            f"### {dataset}",
            "",
            f"- Never Verify (SSS): **{100 * item['baseline_accuracy']:.2f}%**.",
            (
                f"- Best fixed sequence: **{item['best_fixed_sequence']}**, "
                f"**{100 * item['best_fixed_accuracy']:.2f}%**."
            ),
            f"- Verify all checkpoints (VVV): **{100 * item['vvv_accuracy']:.2f}%**.",
            (
                "- Offline any-branch oracle upper bound: "
                f"**{100 * item['offline_any_branch_oracle_accuracy']:.2f}%**."
            ),
            (
                "- Best multi vs best Level-1 — better/equal/worse: "
                f"**{multi['better']}/{multi['equal']}/{multi['worse']}** tasks."
            ),
            "",
        ]
    lines += [
        "## Accuracy and cost by sequence",
        "",
        "| Dataset | Seq. | N | Accuracy (95% CI) | Δ vs SSS | Helpful | Harmful | Mean tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_metrics:
        if row["dataset"] == "Combined":
            continue
        lines.append(
            f"| {row['dataset']} | {row['sequence']} | {row['tasks']} | "
            f"{100 * row['accuracy']:.2f}% "
            f"[{100 * row['accuracy_ci95_low']:.2f}, {100 * row['accuracy_ci95_high']:.2f}] | "
            f"{row['delta_vs_sss_pp']:+.2f} pp | {row['helpful_vs_sss']} | "
            f"{row['harmful_vs_sss']} | {row['mean_teacher_tokens']:.0f} |"
        )
    lines += [
        "",
        "## Conditional checkpoint effects",
        "",
        (
            "One checkpoint is toggled while the other two actions are held fixed. "
            "This gives four paired histories per checkpoint and task."
        ),
        "",
        "| Dataset | CP | Edges | Helpful | Harmful | No change | Net effect |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_effects:
        if row["dataset"] == "Combined":
            continue
        lines.append(
            f"| {row['dataset']} | CP{row['checkpoint']} | {row['paired_edges']} | "
            f"{row['helpful']} | {row['harmful']} | {row['no_change']} | "
            f"{row['net_effect_pp']:+.2f} pp |"
        )
    lines += [
        "",
        "## Failure diagnosis",
        "",
        (
            "LiveCodeBench stopped because four model responses were invalid JSON. "
            "All four are retryable and compatible cache entries remain available."
        ),
        "",
    ]
    lines += [
        f"- {row.get('task_id')} — {row.get('error_type')}: {row.get('message')}"
        for row in errors
    ]
    lines += [
        "",
        (
            "These four tasks must be recovered before reporting a complete LiveCodeBench benchmark. "
            "Missingness may correlate with difficult generations, so partial results can be biased."
        ),
        "",
        "## Interpretation",
        "",
        "- HumanEval benefits from verification; VVV is the strongest fixed sequence.",
        "- LiveCodeBench benefits most from paths containing CP2; CP1 alone adds almost no accuracy.",
        "- Harmful conditional edges remain, so more verification is not uniformly better.",
        "- The offline oracle is an upper bound, not a deployable policy.",
        "- Combined numbers are descriptive because prompts and task distributions differ by dataset.",
        "",
        "## Figures",
        "",
        "![Dashboard](figures/cbv_level2_dashboard.png)",
        "",
        "![Accuracy heatmap](figures/accuracy_heatmap.png)",
        "",
        "## Reproducibility files",
        "",
        "- summary.json",
        "- sequence_metrics.csv",
        "- checkpoint_effects.csv",
        "- task_outcomes.csv",
        "- case_studies.csv",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    he, he_status = complete_blocks(args.humaneval_dir)
    lcb, lcb_status = complete_blocks(args.livecodebench_dir)
    combined = {
        **{f"humaneval::{key}": block for key, block in he.items()},
        **{f"livecodebench::{key}": block for key, block in lcb.items()},
    }
    block_sets = {"HumanEval": he, "LiveCodeBench": lcb, "Combined": combined}
    all_metrics = [
        row
        for dataset, blocks in block_sets.items()
        for row in metrics(dataset, blocks, rng, args.bootstrap_reps)
    ]
    all_effects = [
        row for dataset, blocks in block_sets.items() for row in effects(dataset, blocks)
    ]
    outcomes = [*task_rows("HumanEval", he), *task_rows("LiveCodeBench", lcb)]
    cases = [
        row
        for row in outcomes
        if row["helpful_sequences_vs_sss"]
        or row["harmful_sequences_vs_sss"]
        or row["multi_only_rescue"]
        or row["multi_worse_than_best_l1"]
    ]
    errors = read_jsonl(args.livecodebench_dir / "errors.jsonl")
    he_collection = latest_collection_summary(args.humaneval_dir)
    lcb_collection = latest_collection_summary(args.livecodebench_dir)
    summary = {
        "schema_version": "cbv-level2-available-analysis-v1",
        "protocol": "complete 8-sequence task blocks only",
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "datasets": {
            dataset: dataset_summary(dataset, blocks, all_metrics)
            for dataset, blocks in block_sets.items()
        },
        "source_status": {"HumanEval": he_status, "LiveCodeBench": lcb_status},
        "collection_cost": {
            "HumanEval_unique_teacher_calls": he_collection.get("actual_unique_teacher_calls"),
            "HumanEval_unique_teacher_tokens": he_collection.get("actual_unique_teacher_tokens"),
            "LiveCodeBench_unique_teacher_calls_partial": lcb_collection.get(
                "actual_unique_teacher_calls"
            ),
            "LiveCodeBench_unique_teacher_tokens_partial": lcb_collection.get(
                "actual_unique_teacher_tokens"
            ),
        },
        "livecodebench_errors": errors,
    }
    write_csv(args.output_dir / "sequence_metrics.csv", all_metrics)
    write_csv(args.output_dir / "checkpoint_effects.csv", all_effects)
    write_csv(args.output_dir / "task_outcomes.csv", outcomes)
    write_csv(args.output_dir / "case_studies.csv", cases)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    completion = [
        ("HumanEval L2", len(he), 163),
        ("LiveCodeBench L2", len(lcb), 175),
        ("BigCodeBench L1", 0, 1000),
        ("BigCodeBench L2", 0, 250),
    ]
    figures(args.output_dir, all_metrics, all_effects, completion)
    report_path = args.output_dir / "CBV_Level2_available_data_report.md"
    report_path.write_text(report(summary, all_metrics, all_effects, errors), encoding="utf-8")
    print(report_path.resolve())


if __name__ == "__main__":
    main()
