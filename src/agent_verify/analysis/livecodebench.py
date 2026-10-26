"""LiveCodeBench H1 reporting with benchmark-separated summaries."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from agent_verify.logging import JsonlStore


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _task_bootstrap(
    values: dict[str, list[float]], seed: int, repetitions: int = 20_000
) -> tuple[float, float]:
    task_ids = sorted(values)
    if not task_ids:
        return float("nan"), float("nan")
    per_task = np.asarray([mean(values[task_id]) for task_id in task_ids], dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(per_task), size=(repetitions, len(per_task)))
    return tuple(float(value) for value in np.percentile(per_task[samples].mean(axis=1), [2.5, 97.5]))


def analyze_livecodebench(
    results_dir: Path,
    run_id: str,
    *,
    humaneval_results: Path | None = None,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        row
        for row in JsonlStore(results_dir / "intervention_pairs.jsonl").read_all()
        if row.get("run_id") == run_id and row.get("status") == "completed"
    ]
    baselines = [
        row
        for row in JsonlStore(results_dir / "baseline.jsonl").read_all()
        if row.get("run_id") == run_id and row.get("status") == "completed"
    ]
    invocations = {
        row["invocation_id"]: row
        for row in JsonlStore(results_dir / "llm_invocations.jsonl").read_all()
        if row.get("context", {}).get("run_id") == run_id
    }
    selected = json.loads((results_dir / "selected_tasks.json").read_text(encoding="utf-8"))["tasks"]
    task_meta = {row["task_id"]: row for row in selected}

    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint in (1, 2, 3):
        rows = [row for row in pairs if int(row["checkpoint"]) == checkpoint]
        by_task: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_task[str(row["task_id"])].append(float(row["delta"]))
        low, high = _task_bootstrap(by_task, seed=42 + checkpoint)
        checkpoint_rows.append(
            {
                "checkpoint": checkpoint,
                "pairs": len(rows),
                "tasks": len(by_task),
                "skip_accuracy": mean(float(row["y_skip"]) for row in rows) if rows else 0.0,
                "verify_accuracy": mean(float(row["y_verify"]) for row in rows) if rows else 0.0,
                "mean_delta": mean(float(row["delta"]) for row in rows) if rows else 0.0,
                "mean_delta_ci95_low": low,
                "mean_delta_ci95_high": high,
                "helpful": sum(int(row["delta"]) > 0 for row in rows),
                "harmful": sum(int(row["delta"]) < 0 for row in rows),
                "no_change": sum(int(row["delta"]) == 0 for row in rows),
                "teacher_calls": len(rows),
                "teacher_tokens": sum(
                    int(invocations.get(str(row.get("teacher_invocation_id")), {}).get("input_tokens", 0))
                    + int(invocations.get(str(row.get("teacher_invocation_id")), {}).get("output_tokens", 0))
                    for row in rows
                ),
                "teacher_latency_seconds": sum(
                    float(row.get("teacher_latency_seconds", 0.0)) for row in rows
                ),
                "teacher_cost_usd": sum(float(row.get("teacher_cost_usd", 0.0)) for row in rows),
            }
        )
    _write_csv(
        results_dir / "accuracy_by_checkpoint.csv",
        checkpoint_rows,
        list(checkpoint_rows[0]) if checkpoint_rows else ["checkpoint"],
    )

    outcome_rows: list[dict[str, Any]] = []
    for difficulty in ("easy", "medium", "hard", "all"):
        for checkpoint in (1, 2, 3):
            rows = [
                row
                for row in pairs
                if int(row["checkpoint"]) == checkpoint
                and (
                    difficulty == "all"
                    or task_meta.get(str(row["task_id"]), {}).get("difficulty") == difficulty
                )
            ]
            outcome_rows.append(
                {
                    "difficulty": difficulty,
                    "checkpoint": checkpoint,
                    "pairs": len(rows),
                    "helpful": sum(int(row["delta"]) > 0 for row in rows),
                    "harmful": sum(int(row["delta"]) < 0 for row in rows),
                    "no_change": sum(int(row["delta"]) == 0 for row in rows),
                }
            )
    _write_csv(
        results_dir / "helpful_harmful.csv",
        outcome_rows,
        ["difficulty", "checkpoint", "pairs", "helpful", "harmful", "no_change"],
    )

    baseline_by_difficulty: dict[str, list[int]] = defaultdict(list)
    for row in baselines:
        difficulty = str(task_meta.get(str(row["task_id"]), {}).get("difficulty", "unknown"))
        baseline_by_difficulty[difficulty].append(int(row["passed"]))
    helpful = sum(int(row["delta"]) > 0 for row in pairs)
    harmful = sum(int(row["delta"]) < 0 for row in pairs)
    no_change = sum(int(row["delta"]) == 0 for row in pairs)
    evaluation_attempts = [
        row for row in JsonlStore(results_dir / "evaluations.jsonl").read_all() if row.get("run_id") == run_id
    ]
    evaluations_by_id = {str(row["evaluation_id"]): row for row in evaluation_attempts}
    eval_status = Counter(row.get("status", "UNKNOWN") for row in evaluations_by_id.values())
    total_tokens = sum(
        int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0)) for row in invocations.values()
    )
    total_latency = sum(float(row.get("latency_seconds", 0.0)) for row in invocations.values())
    teacher_invocations = [
        row for row in invocations.values() if row.get("context", {}).get("role") == "teacher"
    ]
    teacher_tokens_all = sum(
        int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0)) for row in teacher_invocations
    )
    teacher_latency_all = sum(float(row.get("latency_seconds", 0.0)) for row in teacher_invocations)
    manifests = [
        row for row in JsonlStore(results_dir / "manifests.jsonl").read_all() if row.get("run_id") == run_id
    ]
    repetitions = (
        int(manifests[-1].get("config", {}).get("experiment", {}).get("repeated_pairs", 1))
        if manifests
        else 1
    )
    planned_pairs = len(baselines) * 3 * repetitions
    missing_pairs = max(0, planned_pairs - len(pairs))
    events = [
        row for row in JsonlStore(results_dir / "events.jsonl").read_all() if row.get("run_id") == run_id
    ]
    pair_failures = sum(row.get("record_type") == "paired_intervention_failure" for row in events)

    comparison_lines = ["HumanEval comparison unavailable or not requested."]
    if humaneval_results is not None and (humaneval_results / "h1_analysis.csv").exists():
        with (humaneval_results / "h1_analysis.csv").open(newline="", encoding="utf-8") as handle:
            human = list(csv.DictReader(handle))
        human_helpful = sum(int(row["delta"]) > 0 for row in human)
        human_harmful = sum(int(row["delta"]) < 0 for row in human)
        comparison_lines = [
            "| Benchmark | Completed pairs | Helpful | Harmful | Mean Delta |",
            "|---|---:|---:|---:|---:|",
            f"| LiveCodeBench | {len(pairs)} | {helpful} | {harmful} | "
            f"{mean(float(row['delta']) for row in pairs) if pairs else 0.0:+.4f} |",
            f"| HumanEval | {len(human)} | {human_helpful} | {human_harmful} | "
            f"{mean(float(row['delta']) for row in human) if human else 0.0:+.4f} |",
            "",
            "Metrics remain benchmark-separated; this table does not pool tasks or claim transfer.",
        ]

    difficulty_lines = [
        f"- {name}: {sum(values)}/{len(values)} ({100 * mean(values):.2f}%)"
        for name, values in sorted(baseline_by_difficulty.items())
        if values
    ]
    checkpoint_table = [
        "| CP | Pairs | Skip acc | Verify acc | Mean Delta [95% CI] | "
        "Helpful / Harmful / No change | Tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in checkpoint_rows:
        checkpoint_table.append(
            f"| {row['checkpoint']} | {row['pairs']} | {100 * row['skip_accuracy']:.2f}% | "
            f"{100 * row['verify_accuracy']:.2f}% | {row['mean_delta']:+.4f} "
            f"[{row['mean_delta_ci95_low']:+.4f}, {row['mean_delta_ci95_high']:+.4f}] | "
            f"{row['helpful']} / {row['harmful']} / {row['no_change']} | {row['teacher_tokens']:,} |"
        )
    report = "\n".join(
        [
            "# LiveCodeBench — H1 Downstream-Aware Verification",
            "",
            f"Run ID: `{run_id}`",
            f"Completed baselines: **{len(baselines)}**; completed pairs: **{len(pairs)}/{planned_pairs}**.",
            (
                f"> **PRELIMINARY / INCOMPLETE:** {missing_pairs} planned pairs are missing. "
                "Checkpoint-level estimates with incomplete coverage must not be treated as final."
                if missing_pairs
                else "> **COMPLETE:** All planned intervention pairs are present."
            ),
            "",
            "## Baseline accuracy",
            "",
            f"Overall: **{sum(int(row['passed']) for row in baselines)}/{len(baselines)} "
            f"({100 * mean(int(row['passed']) for row in baselines):.2f}%)**"
            if baselines
            else "No completed baseline.",
            *difficulty_lines,
            "",
            "## Verification effects",
            "",
            f"Helpful: **{helpful}**; Harmful: **{harmful}**; No change: **{no_change}**.",
            "",
            *checkpoint_table,
            "",
            "## Execution and cost audit",
            "",
            f"- Logged LLM calls: **{len(invocations)}**",
            f"- Total logged tokens: **{total_tokens:,}**",
            f"- Total model latency: **{total_latency:.1f}s**",
            f"- All attempted Teacher calls: **{len(teacher_invocations)}**, "
            f"**{teacher_tokens_all:,} tokens**, **{teacher_latency_all:.1f}s**",
            f"- Missing pairs / paired failures: **{missing_pairs} / {pair_failures}**",
            f"- Evaluator attempts / unique evaluation IDs: "
            f"**{len(evaluation_attempts)} / {len(evaluations_by_id)}**",
            f"- Evaluator statuses: `{dict(sorted(eval_status.items()))}`",
            "- EVALUATOR_ERROR is reported separately and is never converted into model FAIL.",
            "",
            "## HumanEval comparison",
            "",
            *comparison_lines,
            "",
            "## Protocol notes",
            "",
            "- Each pair starts from the same immutable checkpoint and uses matched downstream seeds.",
            "- At most one checkpoint is verified in a paired rollout.",
            "- Private tests are evaluator-only and absent from LLM/critic/checkpoint logs.",
            "- Confidence intervals are task-clustered bootstrap intervals.",
            "",
        ]
    )
    (results_dir / "h1_analysis.md").write_text(report, encoding="utf-8")
    summary = "\n".join(report.splitlines()[: min(45, len(report.splitlines()))]) + "\n"
    (results_dir / "summary.md").write_text(summary, encoding="utf-8")
