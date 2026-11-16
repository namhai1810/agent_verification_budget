# ruff: noqa: E501
"""Train-only synergy audit for the expanded 531-task CBV-Bench Level-2 set.

The audit is descriptive and read-only.  It combines the original 381 frozen
train groups with the 150 outcome-blind BigCodeBench train extensions.  No
validation/test trajectory is opened, and no model or LLM is invoked.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from agent_verify.analysis.cbv_v01_experiment import (
    load_frozen_partition,
    task_trajectories,
)
from agent_verify.analysis.darc_v04_experiment import _assert_disjoint, _load_groups, _write_csv
from agent_verify.analysis.darc_v05_synergy_audit import (
    MULTI_SEQUENCES,
    SEQUENCES,
    SINGLE_SEQUENCES,
    _tokens,
    analyze_partition,
)

COHORT_ORIGINAL = "original_381"
COHORT_EXTENSION = "extension_150"
COHORT_COMBINED = "combined_531"
DATASETS = ("humaneval", "livecodebench", "bigcodebench")
CATEGORIES = (
    "baseline_solved",
    "solvable_by_single_verify",
    "requires_multiple_verify",
    "unsolved_by_observed_trajectories",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _validate_complete(groups: dict[str, Any], *, cohort: str, expected: int) -> None:
    if len(groups) != expected:
        raise ValueError(f"Expected {expected} tasks in {cohort}, found {len(groups)}")
    for task_key, group in groups.items():
        missing = set(SEQUENCES) - set(group.rows)
        if missing:
            raise ValueError(f"{cohort}/{task_key} is missing {sorted(missing)}")


def load_expanded_train_groups(
    frozen_root: Path,
    extension_root: Path,
    selection_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load original and extension train groups without touching other splits."""

    original = _load_groups(frozen_root, "train")
    _validate_complete(original, cohort=COHORT_ORIGINAL, expected=381)

    selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
    records = list(selection.get("task_records", []))
    selected_ids = {str(row["task_id"]) for row in records}
    if len(records) != 150 or len(selected_ids) != 150:
        raise ValueError("Extension selection must contain 150 unique tasks")
    if selection.get("dataset") != "bigcodebench":
        raise ValueError("Extension selection must be BigCodeBench")
    if any(str(row.get("split")) != "train" for row in records):
        raise ValueError("Extension selection contains a non-train task")

    frozen_tasks, _train_rows = load_frozen_partition(frozen_root, "train")
    selected_keys = {f"bigcodebench::{task_id}" for task_id in selected_ids}
    missing_tasks = selected_keys - set(frozen_tasks)
    if missing_tasks:
        raise ValueError(f"Selected tasks missing from frozen train: {sorted(missing_tasks)[:3]}")
    if any(frozen_tasks[key].level2_complete for key in selected_keys):
        raise ValueError("Extension overlaps an original Level-2 train task")

    manifests = _read_jsonl(extension_root / "collection" / "manifests.jsonl")
    if not manifests:
        raise ValueError("Extension collection has no manifest")
    fingerprint = str(manifests[-1]["compatibility_fingerprint"])
    extension_rows = [
        row
        for row in _read_jsonl(extension_root / "collection" / "trajectories.jsonl")
        if str(row.get("compatibility_fingerprint")) == fingerprint
        and str(row.get("task_id")) in selected_ids
        and str(row.get("split")) == "train"
    ]
    extension = task_trajectories(
        {key: frozen_tasks[key] for key in selected_keys},
        extension_rows,
        level=2,
    )
    _validate_complete(extension, cohort=COHORT_EXTENSION, expected=150)
    _assert_disjoint(original, extension)
    return original, extension


def _cheapest_from(group: Any, candidates: Iterable[str]) -> str | None:
    successful = [sequence for sequence in candidates if bool(group.rows[sequence].get("passed"))]
    if not successful:
        return None
    return min(
        successful, key=lambda sequence: (_tokens(group.rows[sequence]), sequence.count("V"), sequence)
    )


def analyze_cohort(
    cohort: str,
    groups: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    trajectories, tasks, strict_rows, marginals, interactions = analyze_partition("train", groups)
    for collection in (trajectories, tasks, strict_rows, marginals, interactions):
        for row in collection:
            row["cohort"] = cohort

    for row in tasks:
        group = groups[str(row["task_key"])]
        cheapest = _cheapest_from(group, SEQUENCES)
        cheapest_single = _cheapest_from(group, SINGLE_SEQUENCES)
        cheapest_multi = _cheapest_from(group, MULTI_SEQUENCES)
        row.update(
            {
                # Includes SSS by design.  If SSS succeeds, its zero-token cost wins.
                "cheapest_success_sequence": cheapest or "",
                "cheapest_success_tokens": _tokens(group.rows[cheapest]) if cheapest else "",
                "cheapest_success_calls": cheapest.count("V") if cheapest else "",
                "cheapest_single_success_sequence": cheapest_single or "",
                "cheapest_single_success_tokens": (
                    _tokens(group.rows[cheapest_single]) if cheapest_single else ""
                ),
                "cheapest_multi_success_sequence": cheapest_multi or "",
                "cheapest_multi_success_tokens": (
                    _tokens(group.rows[cheapest_multi]) if cheapest_multi else ""
                ),
                "cheapest_multi_additional_tokens_vs_sss": (
                    _tokens(group.rows[cheapest_multi]) - _tokens(group.rows["SSS"]) if cheapest_multi else ""
                ),
                "cheapest_multi_premium_vs_single": (
                    _tokens(group.rows[cheapest_multi]) - _tokens(group.rows[cheapest_single])
                    if cheapest_multi and cheapest_single
                    else ""
                ),
            }
        )
    return trajectories, tasks, strict_rows, marginals, interactions


def _scopes(rows: list[dict[str, Any]]) -> Iterable[tuple[str, str, list[dict[str, Any]]]]:
    for cohort in (COHORT_COMBINED, COHORT_ORIGINAL, COHORT_EXTENSION):
        cohort_rows = rows if cohort == COHORT_COMBINED else [row for row in rows if row["cohort"] == cohort]
        if not cohort_rows:
            continue
        yield cohort, "ALL", cohort_rows
        for dataset in DATASETS:
            selected = [row for row in cohort_rows if row["dataset"] == dataset]
            if selected:
                yield cohort, dataset, selected


def task_summaries(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cohort, dataset, selected in _scopes(task_rows):
        counts = Counter(str(row["solvability_category"]) for row in selected)
        tasks = len(selected)
        output.append(
            {
                "cohort": cohort,
                "dataset": dataset,
                "tasks": tasks,
                **{category: counts[category] for category in CATEGORIES},
                "requires_multiple_rate": counts["requires_multiple_verify"] / tasks,
                "tasks_with_strict_synergy": sum(int(row["strict_synergy_count"] > 0) for row in selected),
                "strict_synergy_task_rate": sum(int(row["strict_synergy_count"] > 0) for row in selected)
                / tasks,
                "tasks_with_helpful_addition": sum(int(row["any_helpful_addition"]) for row in selected),
                "tasks_with_harmful_addition": sum(int(row["any_harmful_addition"]) for row in selected),
                "tasks_with_redundant_success_addition": sum(
                    int(row["any_redundant_success_addition"]) for row in selected
                ),
                "tasks_with_any_success": sum(bool(row["cheapest_success_sequence"]) for row in selected),
            }
        )
    return output


def sequence_summaries(trajectory_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cohort, dataset, scoped in _scopes(trajectory_rows):
        for sequence in SEQUENCES:
            selected = [row for row in scoped if row["sequence"] == sequence]
            successful = [row for row in selected if int(row["passed"]) == 1]
            tokens = np.asarray([int(row["teacher_tokens"]) for row in selected], dtype=float)
            output.append(
                {
                    "cohort": cohort,
                    "dataset": dataset,
                    "sequence": sequence,
                    "verify_calls": sequence.count("V"),
                    "tasks": len(selected),
                    "successes": len(successful),
                    "accuracy": len(successful) / len(selected),
                    "strict_synergy_cases": sum(int(row["strict_synergy"]) for row in selected),
                    "mean_teacher_tokens": float(np.mean(tokens)),
                    "median_teacher_tokens": float(np.median(tokens)),
                    "p95_teacher_tokens": float(np.quantile(tokens, 0.95)),
                    "mean_success_teacher_tokens": (
                        float(np.mean([int(row["teacher_tokens"]) for row in successful]))
                        if successful
                        else ""
                    ),
                }
            )
    return output


def marginal_summaries(marginals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cohort, dataset, scoped in _scopes(marginals):
        for checkpoint in (1, 2, 3):
            selected = [row for row in scoped if int(row["checkpoint"]) == checkpoint]
            counts = Counter(str(row["effect"]) for row in selected)
            total = len(selected)
            no_change = counts["redundant_success"] + counts["no_change_failure"]
            output.append(
                {
                    "cohort": cohort,
                    "dataset": dataset,
                    "checkpoint": checkpoint,
                    "tasks": len({row["task_key"] for row in selected}),
                    "paired_comparisons": total,
                    "helpful": counts["helpful"],
                    "harmful": counts["harmful"],
                    "no_change": no_change,
                    "redundant_success": counts["redundant_success"],
                    "no_change_failure": counts["no_change_failure"],
                    "helpful_rate": counts["helpful"] / total,
                    "harmful_rate": counts["harmful"] / total,
                    "net_effect_rate": (counts["helpful"] - counts["harmful"]) / total,
                    "mean_additional_teacher_tokens": float(
                        np.mean([int(row["token_delta"]) for row in selected])
                    ),
                    "median_additional_teacher_tokens": float(
                        np.median([int(row["token_delta"]) for row in selected])
                    ),
                }
            )
    return output


def interaction_summaries(interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cohort, dataset, scoped in _scopes(interactions):
        for pair in ("CP1×CP2", "CP1×CP3", "CP2×CP3"):
            selected = [row for row in scoped if row["checkpoint_pair"] == pair]
            output.append(
                {
                    "cohort": cohort,
                    "dataset": dataset,
                    "checkpoint_pair": pair,
                    "tasks": len({row["task_key"] for row in selected}),
                    "conditional_comparisons": len(selected),
                    "positive_interactions": sum(int(row["interaction"] > 0) for row in selected),
                    "negative_interactions": sum(int(row["interaction"] < 0) for row in selected),
                    "zero_interactions": sum(int(row["interaction"] == 0) for row in selected),
                    "strict_pair_synergy": sum(int(row["strict_pair_synergy"]) for row in selected),
                }
            )
    return output


def cheapest_summaries(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cohort, dataset, selected in _scopes(task_rows):
        counts = Counter(str(row["cheapest_success_sequence"] or "NONE") for row in selected)
        for sequence in (*SEQUENCES, "NONE"):
            matching = [
                row for row in selected if str(row["cheapest_success_sequence"] or "NONE") == sequence
            ]
            output.append(
                {
                    "cohort": cohort,
                    "dataset": dataset,
                    "sequence": sequence,
                    "tasks": counts[sequence],
                    "mean_actual_tokens": (
                        float(np.mean([int(row["cheapest_success_tokens"]) for row in matching]))
                        if matching and sequence != "NONE"
                        else ""
                    ),
                }
            )
    return output


def cohort_comparison(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = (
        ("all_original_vs_extension", "ALL"),
        ("bigcodebench_original_vs_extension", "bigcodebench"),
    )
    output: list[dict[str, Any]] = []
    for label, dataset in comparisons:
        sides: dict[str, list[dict[str, Any]]] = {}
        for cohort in (COHORT_ORIGINAL, COHORT_EXTENSION):
            sides[cohort] = [
                row
                for row in task_rows
                if row["cohort"] == cohort and (dataset == "ALL" or row["dataset"] == dataset)
            ]
        left, right = sides[COHORT_ORIGINAL], sides[COHORT_EXTENSION]
        if not left or not right:
            continue

        def rate(rows: list[dict[str, Any]], predicate: Any) -> float:
            return sum(bool(predicate(row)) for row in rows) / len(rows)

        metrics = {
            "baseline_solved_rate": lambda row: row["solvability_category"] == "baseline_solved",
            "single_verify_solved_rate": lambda row: row["solvability_category"]
            == "solvable_by_single_verify",
            "requires_multi_rate": lambda row: row["solvability_category"] == "requires_multiple_verify",
            "unsolved_rate": lambda row: row["solvability_category"] == "unsolved_by_observed_trajectories",
            "strict_synergy_task_rate": lambda row: int(row["strict_synergy_count"]) > 0,
            "any_helpful_addition_rate": lambda row: int(row["any_helpful_addition"]) == 1,
            "any_harmful_addition_rate": lambda row: int(row["any_harmful_addition"]) == 1,
        }
        for metric, predicate in metrics.items():
            old_value = rate(left, predicate)
            new_value = rate(right, predicate)
            output.append(
                {
                    "comparison": label,
                    "dataset": dataset,
                    "metric": metric,
                    "original_tasks": len(left),
                    "extension_tasks": len(right),
                    "original_value": old_value,
                    "extension_value": new_value,
                    "delta_extension_minus_original": new_value - old_value,
                }
            )
    return output


def _plots(
    output: Path,
    task_summary: list[dict[str, Any]],
    marginal_summary: list[dict[str, Any]],
    sequence_summary: list[dict[str, Any]],
) -> None:
    figure_root = output / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    navy, blue, orange, green, gray = "#17324D", "#3B6EA8", "#E67E22", "#2B8A66", "#DDE4EA"

    scopes = (
        (COHORT_ORIGINAL, "ALL", "Original 381"),
        (COHORT_EXTENSION, "ALL", "Extension 150"),
        (COHORT_ORIGINAL, "bigcodebench", "Original BCB 181"),
        (COHORT_EXTENSION, "bigcodebench", "Extension BCB 150"),
    )
    lookup = {(row["cohort"], row["dataset"]): row for row in task_summary}
    fig, axis = plt.subplots(figsize=(12.5, 5.7))
    bottoms = np.zeros(len(scopes))
    category_style = (
        ("baseline_solved", "SSS solved", blue),
        ("solvable_by_single_verify", "Single Verify", green),
        ("requires_multiple_verify", "Requires multi", orange),
        ("unsolved_by_observed_trajectories", "Unsolved", gray),
    )
    for category, label, color in category_style:
        values = np.asarray(
            [
                100 * int(lookup[(cohort, dataset)][category]) / int(lookup[(cohort, dataset)]["tasks"])
                for cohort, dataset, _ in scopes
            ]
        )
        axis.bar(np.arange(len(scopes)), values, bottom=bottoms, color=color, label=label)
        bottoms += values
    axis.set_xticks(np.arange(len(scopes)), [label for _cohort, _dataset, label in scopes])
    axis.set_ylabel("Tasks (%)")
    axis.set_ylim(0, 100)
    axis.set_title(
        "Task solvability: original vs Level-2 extension", loc="left", color=navy, fontweight="bold"
    )
    axis.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.11))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E4E9EE", linewidth=0.8)
    axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figure_root / "cohort_solvability.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    width = 0.34
    checkpoints = (1, 2, 3)
    for index, (cohort, label, color) in enumerate(
        ((COHORT_ORIGINAL, "Original 381", blue), (COHORT_EXTENSION, "Extension 150", orange))
    ):
        selected = [row for row in marginal_summary if row["cohort"] == cohort and row["dataset"] == "ALL"]
        values = [
            100 * next(row for row in selected if int(row["checkpoint"]) == cp)["net_effect_rate"]
            for cp in checkpoints
        ]
        axes[0].bar(np.arange(3) + (index - 0.5) * width, values, width, color=color, label=label)
    axes[0].axhline(0, color=navy, linewidth=1)
    axes[0].set_xticks(np.arange(3), ["CP1", "CP2", "CP3"])
    axes[0].set_ylabel("Helpful − Harmful (pp)")
    axes[0].set_title("Conditional marginal effect", loc="left", color=navy, fontweight="bold")
    axes[0].legend(frameon=False)

    for cohort, label, color, marker in (
        (COHORT_ORIGINAL, "Original 381", blue, "o"),
        (COHORT_EXTENSION, "Extension 150", orange, "s"),
    ):
        selected = [row for row in sequence_summary if row["cohort"] == cohort and row["dataset"] == "ALL"]
        values = [
            next(row for row in selected if row["sequence"] == sequence)["mean_teacher_tokens"]
            for sequence in SEQUENCES
        ]
        axes[1].plot(SEQUENCES, values, color=color, marker=marker, linewidth=2, label=label)
    axes[1].set_ylabel("Mean actual Teacher tokens/task")
    axes[1].set_title("Observed sequence cost", loc="left", color=navy, fontweight="bold")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E4E9EE", linewidth=0.8)
        axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figure_root / "marginal_effect_and_cost.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ]


def _write_report(
    output: Path,
    task_rows: list[dict[str, Any]],
    strict_rows: list[dict[str, Any]],
    task_summary: list[dict[str, Any]],
    sequence_summary: list[dict[str, Any]],
    marginal_summary: list[dict[str, Any]],
    interaction_summary: list[dict[str, Any]],
    cheapest_summary: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
) -> Path:
    display_scopes = (
        (COHORT_COMBINED, "ALL", "All train"),
        (COHORT_ORIGINAL, "ALL", "Original"),
        (COHORT_EXTENSION, "ALL", "Extension"),
        (COHORT_COMBINED, "humaneval", "HumanEval"),
        (COHORT_COMBINED, "livecodebench", "LiveCodeBench"),
        (COHORT_COMBINED, "bigcodebench", "BigCodeBench"),
        (COHORT_ORIGINAL, "bigcodebench", "Original BCB"),
        (COHORT_EXTENSION, "bigcodebench", "Extension BCB"),
    )
    task_lookup = {(row["cohort"], row["dataset"]): row for row in task_summary}
    combined = task_lookup[(COHORT_COMBINED, "ALL")]
    combined_marginal = [
        row for row in marginal_summary if row["cohort"] == COHORT_COMBINED and row["dataset"] == "ALL"
    ]
    effects_total = sum(int(row["paired_comparisons"]) for row in combined_marginal)
    helpful = sum(int(row["helpful"]) for row in combined_marginal)
    harmful = sum(int(row["harmful"]) for row in combined_marginal)
    no_change = sum(int(row["no_change"]) for row in combined_marginal)
    combined_cost = [
        row for row in sequence_summary if row["cohort"] == COHORT_COMBINED and row["dataset"] == "ALL"
    ]
    cheapest_all = [
        row
        for row in cheapest_summary
        if row["cohort"] == COHORT_COMBINED and row["dataset"] == "ALL" and int(row["tasks"]) > 0
    ]

    lines = [
        "# DARC v0.5.2 — Expanded Multi-Verification Synergy Audit",
        "",
        "> Training split only: 381 original + 150 extension tasks. No validation/frozen-test "
        "trajectory access, retraining, Student inference, or Teacher inference.",
        "",
        "## Scope and integrity",
        "",
        "- **531/531 tasks** have all eight trajectories: `SSS, VSS, SVS, SSV, VVS, VSV, SVV, VVV`.",
        "- The extension contributes **150 BigCodeBench train tasks / 1,200 paired trajectories**.",
        "- All sequence comparisons are paired within task; trajectories are never treated as independent samples.",
        "",
        "## Single-Verify versus Multi-Verify solvability",
        "",
        *_table(
            [
                "Scope",
                "Tasks",
                "SSS solved",
                "Single Verify",
                "Requires multi",
                "Unsolved",
                "Strict-synergy tasks",
            ],
            [
                [
                    label,
                    task_lookup[(cohort, dataset)]["tasks"],
                    task_lookup[(cohort, dataset)]["baseline_solved"],
                    task_lookup[(cohort, dataset)]["solvable_by_single_verify"],
                    task_lookup[(cohort, dataset)]["requires_multiple_verify"],
                    task_lookup[(cohort, dataset)]["unsolved_by_observed_trajectories"],
                    task_lookup[(cohort, dataset)]["tasks_with_strict_synergy"],
                ]
                for cohort, dataset, label in display_scopes
            ],
        ),
        "",
        "![Original versus extension task solvability](../figures/cohort_solvability.png)",
        "",
        f"Exactly **{combined['requires_multiple_verify']}/{combined['tasks']} tasks** strictly "
        "require multiple Verify calls under the eight observed branches: SSS and every "
        "single-Verify path fail, while at least one multi-Verify path succeeds.",
        "",
        "## Strict synergy cases",
        "",
        "A multi-Verify sequence is strict synergy only if it succeeds while `SSS` and every "
        "proper nonempty Verify subset of that exact sequence fail. This is stronger than "
        "ordinary multi-Verify success and is sequence-specific.",
        "",
    ]
    if strict_rows:
        lines.extend(
            _table(
                ["Cohort", "Dataset", "Task", "Sequence", "Tokens", "Failed subsets"],
                [
                    [
                        row["cohort"],
                        row["dataset"],
                        row["task_id"],
                        row["sequence"],
                        row["teacher_tokens"],
                        row["proper_nonempty_subsets"],
                    ]
                    for row in strict_rows
                ],
            )
        )
    else:
        lines.append("No strict-synergy cases were observed.")

    lines.extend(
        [
            "",
            "## Helpful, Harmful and No-change effects",
            "",
            f"Across **{effects_total:,}** shared-prefix paired comparisons, Verify is Helpful "
            f"in **{helpful:,} ({100 * helpful / effects_total:.2f}%)**, Harmful in "
            f"**{harmful:,} ({100 * harmful / effects_total:.2f}%)**, and No-change in "
            f"**{no_change:,} ({100 * no_change / effects_total:.2f}%)**.",
            "",
            *_table(
                [
                    "Scope",
                    "CP",
                    "Comparisons",
                    "Helpful",
                    "Harmful",
                    "No-change",
                    "Net effect",
                    "Mean extra tokens",
                ],
                [
                    [
                        f"{row['cohort']} / {row['dataset']}",
                        row["checkpoint"],
                        row["paired_comparisons"],
                        row["helpful"],
                        row["harmful"],
                        row["no_change"],
                        f"{100 * float(row['net_effect_rate']):+.2f} pp",
                        f"{float(row['mean_additional_teacher_tokens']):,.0f}",
                    ]
                    for row in marginal_summary
                    if (row["cohort"], row["dataset"])
                    in {
                        (COHORT_COMBINED, "ALL"),
                        (COHORT_COMBINED, "humaneval"),
                        (COHORT_COMBINED, "livecodebench"),
                        (COHORT_COMBINED, "bigcodebench"),
                        (COHORT_ORIGINAL, "ALL"),
                        (COHORT_EXTENSION, "ALL"),
                    }
                ],
            ),
            "",
            "![Marginal effects and actual Teacher-token costs](../figures/marginal_effect_and_cost.png)",
            "",
            "## Checkpoint interactions",
            "",
            *_table(
                ["Scope", "Pair", "Comparisons", "Positive", "Negative", "Zero", "Strict pair synergy"],
                [
                    [
                        f"{row['cohort']} / {row['dataset']}",
                        row["checkpoint_pair"],
                        row["conditional_comparisons"],
                        row["positive_interactions"],
                        row["negative_interactions"],
                        row["zero_interactions"],
                        row["strict_pair_synergy"],
                    ]
                    for row in interaction_summary
                    if row["cohort"] == COHORT_COMBINED
                    and row["dataset"] in {"ALL", "humaneval", "livecodebench", "bigcodebench"}
                ],
            ),
            "",
            "## Actual Teacher-token cost",
            "",
            *_table(
                ["Sequence", "Verify calls", "Accuracy", "Mean tokens", "Median tokens", "P95 tokens"],
                [
                    [
                        row["sequence"],
                        row["verify_calls"],
                        f"{100 * float(row['accuracy']):.2f}%",
                        f"{float(row['mean_teacher_tokens']):,.0f}",
                        f"{float(row['median_teacher_tokens']):,.0f}",
                        f"{float(row['p95_teacher_tokens']):,.0f}",
                    ]
                    for row in combined_cost
                ],
            ),
            "",
            "## Cheapest successful sequence (diagnostic oracle)",
            "",
            "`SSS` is explicitly included. Therefore every task already solved by `SSS` is "
            "assigned zero Teacher calls/tokens instead of being forced onto a verification path. "
            "This oracle uses observed outcomes and actual costs after the fact and must never be "
            "used as an online policy feature.",
            "",
            *_table(
                ["Sequence", "Tasks", "Mean actual tokens"],
                [
                    [
                        row["sequence"],
                        row["tasks"],
                        f"{float(row['mean_actual_tokens']):,.0f}"
                        if row["mean_actual_tokens"] != ""
                        else "—",
                    ]
                    for row in cheapest_all
                ],
            ),
            "",
            "## What did the 150-task extension add?",
            "",
            "The all-cohort comparison is partly confounded because the extension contains only "
            "BigCodeBench. The BigCodeBench-only comparison is the more defensible estimate of "
            "how the new task sample changes observed rates.",
            "",
            *_table(
                ["Comparison", "Metric", "Original", "Extension", "Delta"],
                [
                    [
                        row["comparison"],
                        row["metric"],
                        f"{100 * float(row['original_value']):.2f}%",
                        f"{100 * float(row['extension_value']):.2f}%",
                        f"{100 * float(row['delta_extension_minus_original']):+.2f} pp",
                    ]
                    for row in comparison
                ],
            ),
            "",
            "## Interpretation guardrails",
            "",
            "- `requires_multiple_verify` is an observed-branch statement, not proof that every possible intervention strategy needs multiple calls.",
            "- Strict synergy and checkpoint interactions are sparse task-level phenomena; report task counts, not trajectory-level significance.",
            "- Additional-token values use realized costs and are descriptive/oracle quantities, not online feasibility inputs.",
            "- Extension-vs-original differences can reflect task composition. Prefer the BigCodeBench-only comparison when discussing extension effects.",
            "- No validation or frozen-test outcome was read, so this audit can guide training hypotheses without contaminating model selection boundaries.",
            "",
            "## Artifacts",
            "",
            "- `tables/task_outcome_patterns.csv` and `task_solvability_summary.csv`",
            "- `tables/trajectory_outcomes.csv` and `sequence_summary.csv`",
            "- `tables/strict_synergy_cases.csv`",
            "- `tables/conditional_marginal_effects.csv` and `marginal_effect_summary.csv`",
            "- `tables/checkpoint_interactions.csv` and `interaction_summary.csv`",
            "- `tables/cheapest_success_summary.csv`",
            "- `tables/cohort_comparison.csv`",
            "- `synergy_audit_manifest.json`",
        ]
    )
    report = output / "report" / "darc_v052_multi_verification_synergy_audit.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_expanded_synergy_audit(
    *,
    frozen_root: Path,
    extension_root: Path,
    selection_manifest: Path,
    output: Path,
) -> Path:
    original, extension = load_expanded_train_groups(frozen_root, extension_root, selection_manifest)
    cohorts = ((COHORT_ORIGINAL, original), (COHORT_EXTENSION, extension))

    trajectory_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    marginals: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for cohort, groups in cohorts:
        trajectory, tasks, strict, marginal, interaction = analyze_cohort(cohort, groups)
        trajectory_rows.extend(trajectory)
        task_rows.extend(tasks)
        strict_rows.extend(strict)
        marginals.extend(marginal)
        interactions.extend(interaction)

    if len(task_rows) != 531 or len(trajectory_rows) != 531 * 8:
        raise AssertionError("Expanded audit is not exactly 531 tasks × 8 trajectories")

    task_summary = task_summaries(task_rows)
    sequence_summary = sequence_summaries(trajectory_rows)
    marginal_summary = marginal_summaries(marginals)
    interaction_summary = interaction_summaries(interactions)
    cheapest_summary = cheapest_summaries(task_rows)
    comparison = cohort_comparison(task_rows)

    table_root = output / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    _write_csv(table_root / "task_outcome_patterns.csv", task_rows)
    _write_csv(table_root / "task_solvability_summary.csv", task_summary)
    _write_csv(table_root / "trajectory_outcomes.csv", trajectory_rows)
    _write_csv(table_root / "sequence_summary.csv", sequence_summary)
    _write_csv(table_root / "strict_synergy_cases.csv", strict_rows or [{"note": "no strict synergy cases"}])
    _write_csv(table_root / "conditional_marginal_effects.csv", marginals)
    _write_csv(table_root / "marginal_effect_summary.csv", marginal_summary)
    _write_csv(table_root / "checkpoint_interactions.csv", interactions)
    _write_csv(table_root / "interaction_summary.csv", interaction_summary)
    _write_csv(table_root / "cheapest_success_summary.csv", cheapest_summary)
    _write_csv(table_root / "cohort_comparison.csv", comparison)

    _plots(output, task_summary, marginal_summary, sequence_summary)
    report = _write_report(
        output,
        task_rows,
        strict_rows,
        task_summary,
        sequence_summary,
        marginal_summary,
        interaction_summary,
        cheapest_summary,
        comparison,
    )
    manifest = {
        "schema_version": "darc-v052-expanded-synergy-audit-v1",
        "scope": "train-only",
        "frozen_root": str(frozen_root),
        "extension_root": str(extension_root),
        "selection_manifest": str(selection_manifest),
        "partitions_loaded": ["train"],
        "validation_outcomes_loaded": False,
        "frozen_test_outcomes_loaded": False,
        "tasks": {COHORT_ORIGINAL: 381, COHORT_EXTENSION: 150, COHORT_COMBINED: 531},
        "trajectories": 531 * 8,
        "sequences": list(SEQUENCES),
        "cheapest_success_includes_sss": True,
        "strict_synergy_definition": (
            "multi sequence succeeds while SSS and every proper nonempty Verify subset fail"
        ),
        "report": str(report),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "synergy_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
