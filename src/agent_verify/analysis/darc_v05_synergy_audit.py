"""Task-clustered multi-verification synergy audit for CBV-Bench Level 2.

This module is deliberately descriptive and read-only.  It loads only the
frozen train and validation partitions, never trains a model, and treats the
eight trajectories belonging to a task as paired counterfactual observations.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from agent_verify.analysis.cbv_v01_experiment import _passed
from agent_verify.analysis.darc_v04_experiment import (
    _assert_disjoint,
    _load_groups,
    _write_csv,
)

SEQUENCES = ("SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV")
SINGLE_SEQUENCES = ("VSS", "SVS", "SSV")
MULTI_SEQUENCES = ("VVS", "VSV", "SVV", "VVV")
PREFIXES = ("", "S", "V", "SS", "SV", "VS", "VV")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tokens(row: dict[str, Any]) -> int:
    return int(row.get("teacher_total_tokens", 0) or 0)


def _subset_sequence(positions: Iterable[int]) -> str:
    chosen = set(positions)
    return "".join("V" if index in chosen else "S" for index in range(3))


def proper_nonempty_verify_subsets(sequence: str) -> tuple[str, ...]:
    """Return every nonempty proper Verify subset of ``sequence``."""

    positions = [index for index, action in enumerate(sequence) if action == "V"]
    return tuple(
        _subset_sequence(subset)
        for size in range(1, len(positions))
        for subset in itertools.combinations(positions, size)
    )


def strict_synergy(group: Any, sequence: str) -> bool:
    """Whether a multi-Verify branch uniquely succeeds over all proper subsets."""

    if sequence not in MULTI_SEQUENCES:
        return False
    if _passed(group.rows[sequence]) != 1 or _passed(group.rows["SSS"]) != 0:
        return False
    subsets = proper_nonempty_verify_subsets(sequence)
    return bool(subsets) and all(_passed(group.rows[item]) == 0 for item in subsets)


def _cheapest_success(group: Any, *, require_verify: bool) -> str | None:
    candidates = [
        sequence
        for sequence in SEQUENCES
        if _passed(group.rows[sequence]) == 1 and (not require_verify or "V" in sequence)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (_tokens(group.rows[item]), item.count("V"), item))


def _effect(skip: int, verify: int) -> str:
    if verify > skip:
        return "helpful"
    if verify < skip:
        return "harmful"
    if verify == 1:
        return "redundant_success"
    return "no_change_failure"


def marginal_rows(split: str, task_key: str, group: Any) -> list[dict[str, Any]]:
    """Paired Verify-vs-Skip effects under every observable shared prefix.

    The downstream suffix is held fixed, so each comparison isolates the
    marginal effect of the current checkpoint under that history/continuation.
    """

    rows: list[dict[str, Any]] = []
    for checkpoint in (1, 2, 3):
        prefix_length = checkpoint - 1
        future_length = 3 - checkpoint
        prefixes = itertools.product("SV", repeat=prefix_length)
        for prefix_bits in prefixes:
            prefix = "".join(prefix_bits)
            for suffix_bits in itertools.product("SV", repeat=future_length):
                suffix = "".join(suffix_bits)
                skip_sequence = prefix + "S" + suffix
                verify_sequence = prefix + "V" + suffix
                y_skip = int(_passed(group.rows[skip_sequence]))
                y_verify = int(_passed(group.rows[verify_sequence]))
                skip_tokens = _tokens(group.rows[skip_sequence])
                verify_tokens = _tokens(group.rows[verify_sequence])
                rows.append(
                    {
                        "split": split,
                        "task_key": task_key,
                        "dataset": group.task.dataset,
                        "task_id": group.task.task_id,
                        "checkpoint": checkpoint,
                        "prefix": prefix or "START",
                        "future_suffix": suffix or "END",
                        "skip_sequence": skip_sequence,
                        "verify_sequence": verify_sequence,
                        "y_skip": y_skip,
                        "y_verify": y_verify,
                        "marginal_gain": y_verify - y_skip,
                        "effect": _effect(y_skip, y_verify),
                        "skip_tokens": skip_tokens,
                        "verify_tokens": verify_tokens,
                        "token_delta": verify_tokens - skip_tokens,
                    }
                )
    return rows


def interaction_rows(split: str, task_key: str, group: Any) -> list[dict[str, Any]]:
    """Pairwise checkpoint interactions conditional on the third action."""

    rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(range(3), 2):
        other = next(index for index in range(3) if index not in {left, right})
        for other_action in "SV":
            outcomes: dict[tuple[int, int], int] = {}
            sequences: dict[tuple[int, int], str] = {}
            for left_value, right_value in itertools.product((0, 1), repeat=2):
                actions = ["S", "S", "S"]
                actions[other] = other_action
                actions[left] = "V" if left_value else "S"
                actions[right] = "V" if right_value else "S"
                sequence = "".join(actions)
                sequences[(left_value, right_value)] = sequence
                outcomes[(left_value, right_value)] = int(_passed(group.rows[sequence]))
            interaction = outcomes[(1, 1)] - outcomes[(1, 0)] - outcomes[(0, 1)] + outcomes[(0, 0)]
            rows.append(
                {
                    "split": split,
                    "task_key": task_key,
                    "dataset": group.task.dataset,
                    "checkpoint_pair": f"CP{left + 1}×CP{right + 1}",
                    "condition_checkpoint": f"CP{other + 1}",
                    "condition_action": other_action,
                    "sequence_00": sequences[(0, 0)],
                    "sequence_10": sequences[(1, 0)],
                    "sequence_01": sequences[(0, 1)],
                    "sequence_11": sequences[(1, 1)],
                    "y00": outcomes[(0, 0)],
                    "y10": outcomes[(1, 0)],
                    "y01": outcomes[(0, 1)],
                    "y11": outcomes[(1, 1)],
                    "interaction": interaction,
                    "strict_pair_synergy": int(
                        outcomes[(1, 1)] == 1
                        and outcomes[(0, 0)] == 0
                        and outcomes[(1, 0)] == 0
                        and outcomes[(0, 1)] == 0
                    ),
                }
            )
    return rows


def _validate_groups(groups: dict[str, Any], split: str, expected_tasks: int) -> None:
    if len(groups) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} {split} tasks, found {len(groups)}")
    for task_key, group in groups.items():
        missing = set(SEQUENCES) - set(group.rows)
        if missing:
            raise ValueError(f"{split}/{task_key} is missing trajectories: {sorted(missing)}")


def analyze_partition(
    split: str,
    groups: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    trajectory_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    marginals: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for task_key, group in sorted(groups.items()):
        outcomes = {sequence: int(_passed(group.rows[sequence])) for sequence in SEQUENCES}
        strict_sequences = [sequence for sequence in MULTI_SEQUENCES if strict_synergy(group, sequence)]
        successful = [sequence for sequence in SEQUENCES if outcomes[sequence] == 1]
        successful_singles = [sequence for sequence in SINGLE_SEQUENCES if outcomes[sequence] == 1]
        successful_multi = [sequence for sequence in MULTI_SEQUENCES if outcomes[sequence] == 1]
        cheapest = _cheapest_success(group, require_verify=False)
        cheapest_verify = _cheapest_success(group, require_verify=True)
        task_marginals = marginal_rows(split, task_key, group)
        marginals.extend(task_marginals)
        interactions.extend(interaction_rows(split, task_key, group))

        if outcomes["SSS"] == 1:
            category = "baseline_solved"
        elif successful_singles:
            category = "solvable_by_single_verify"
        elif successful_multi:
            category = "requires_multiple_verify"
        else:
            category = "unsolved_by_observed_trajectories"

        task_rows.append(
            {
                "split": split,
                "task_key": task_key,
                "dataset": group.task.dataset,
                "task_id": group.task.task_id,
                "outcome_pattern": "".join(str(outcomes[item]) for item in SEQUENCES),
                **{f"outcome_{sequence}": outcomes[sequence] for sequence in SEQUENCES},
                "solvability_category": category,
                "successful_sequences": ",".join(successful),
                "successful_single_sequences": ",".join(successful_singles),
                "successful_multi_sequences": ",".join(successful_multi),
                "strict_synergy_sequences": ",".join(strict_sequences),
                "strict_synergy_count": len(strict_sequences),
                "cheapest_success_sequence": cheapest or "",
                "cheapest_success_tokens": _tokens(group.rows[cheapest]) if cheapest else "",
                "cheapest_success_verify_sequence": cheapest_verify or "",
                "cheapest_success_verify_tokens": (
                    _tokens(group.rows[cheapest_verify]) if cheapest_verify else ""
                ),
                "vvv_tokens": _tokens(group.rows["VVV"]),
                "any_helpful_addition": int(any(row["effect"] == "helpful" for row in task_marginals)),
                "any_harmful_addition": int(any(row["effect"] == "harmful" for row in task_marginals)),
                "any_redundant_success_addition": int(
                    any(row["effect"] == "redundant_success" for row in task_marginals)
                ),
            }
        )
        for sequence in SEQUENCES:
            is_strict = sequence in strict_sequences
            trajectory_rows.append(
                {
                    "split": split,
                    "task_key": task_key,
                    "dataset": group.task.dataset,
                    "task_id": group.task.task_id,
                    "sequence": sequence,
                    "verify_calls": sequence.count("V"),
                    "passed": outcomes[sequence],
                    "teacher_tokens": _tokens(group.rows[sequence]),
                    "strict_synergy": int(is_strict),
                    "proper_nonempty_subsets": ",".join(proper_nonempty_verify_subsets(sequence)),
                }
            )
            if is_strict:
                subsets = proper_nonempty_verify_subsets(sequence)
                strict_rows.append(
                    {
                        "split": split,
                        "task_key": task_key,
                        "dataset": group.task.dataset,
                        "task_id": group.task.task_id,
                        "sequence": sequence,
                        "teacher_tokens": _tokens(group.rows[sequence]),
                        "proper_nonempty_subsets": ",".join(subsets),
                        "subset_outcomes": ",".join(f"{item}:{outcomes[item]}" for item in subsets),
                        "outcome_pattern": "".join(str(outcomes[item]) for item in SEQUENCES),
                    }
                )
    return trajectory_rows, task_rows, strict_rows, marginals, interactions


def _task_summary(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "combined"):
        selected = task_rows if split == "combined" else [r for r in task_rows if r["split"] == split]
        categories = Counter(str(row["solvability_category"]) for row in selected)
        rows.append(
            {
                "split": split,
                "tasks": len(selected),
                "baseline_solved": categories["baseline_solved"],
                "solvable_by_single_verify": categories["solvable_by_single_verify"],
                "requires_multiple_verify": categories["requires_multiple_verify"],
                "unsolved_by_observed_trajectories": categories["unsolved_by_observed_trajectories"],
                "tasks_with_strict_synergy": sum(int(row["strict_synergy_count"] > 0) for row in selected),
                "tasks_with_helpful_addition": sum(int(row["any_helpful_addition"]) for row in selected),
                "tasks_with_harmful_addition": sum(int(row["any_harmful_addition"]) for row in selected),
                "tasks_with_redundant_success_addition": sum(
                    int(row["any_redundant_success_addition"]) for row in selected
                ),
            }
        )
    return rows


def _sequence_summary(trajectory_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "combined"):
        pool = trajectory_rows if split == "combined" else [r for r in trajectory_rows if r["split"] == split]
        for sequence in SEQUENCES:
            selected = [row for row in pool if row["sequence"] == sequence]
            successful = [row for row in selected if int(row["passed"]) == 1]
            rows.append(
                {
                    "split": split,
                    "sequence": sequence,
                    "tasks": len(selected),
                    "successes": len(successful),
                    "success_rate": len(successful) / max(len(selected), 1),
                    "strict_synergy_cases": sum(int(row["strict_synergy"]) for row in selected),
                    "mean_teacher_tokens": float(np.mean([int(row["teacher_tokens"]) for row in selected])),
                    "mean_success_teacher_tokens": (
                        float(np.mean([int(row["teacher_tokens"]) for row in successful]))
                        if successful
                        else ""
                    ),
                }
            )
    return rows


def _marginal_summary(marginals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "combined"):
        pool = marginals if split == "combined" else [r for r in marginals if r["split"] == split]
        for checkpoint in (1, 2, 3):
            selected = [row for row in pool if int(row["checkpoint"]) == checkpoint]
            counts = Counter(str(row["effect"]) for row in selected)
            rows.append(
                {
                    "split": split,
                    "checkpoint": checkpoint,
                    "tasks": len({row["task_key"] for row in selected}),
                    "paired_comparisons": len(selected),
                    "helpful": counts["helpful"],
                    "harmful": counts["harmful"],
                    "redundant_success": counts["redundant_success"],
                    "no_change_failure": counts["no_change_failure"],
                    "tasks_with_helpful": len(
                        {row["task_key"] for row in selected if row["effect"] == "helpful"}
                    ),
                    "tasks_with_harmful": len(
                        {row["task_key"] for row in selected if row["effect"] == "harmful"}
                    ),
                    "mean_token_delta": float(np.mean([int(row["token_delta"]) for row in selected])),
                }
            )
    return rows


def _interaction_summary(interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "combined"):
        pool = interactions if split == "combined" else [r for r in interactions if r["split"] == split]
        for pair in ("CP1×CP2", "CP1×CP3", "CP2×CP3"):
            selected = [row for row in pool if row["checkpoint_pair"] == pair]
            rows.append(
                {
                    "split": split,
                    "checkpoint_pair": pair,
                    "tasks": len({row["task_key"] for row in selected}),
                    "conditional_comparisons": len(selected),
                    "positive_interactions": sum(int(row["interaction"] > 0) for row in selected),
                    "negative_interactions": sum(int(row["interaction"] < 0) for row in selected),
                    "strict_pair_synergy": sum(int(row["strict_pair_synergy"]) for row in selected),
                }
            )
    return rows


def _cheapest_summary(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in task_rows:
        sequence = str(row["cheapest_success_verify_sequence"])
        if sequence:
            counter[(str(row["split"]), sequence)] += 1
            counter[("combined", sequence)] += 1
    return [
        {"split": split, "sequence": sequence, "tasks": counter[(split, sequence)]}
        for split in ("train", "validation", "combined")
        for sequence in SEQUENCES[1:]
    ]


def policy_selection_rows(
    trace_path: Path,
    validation_task_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing existing DARC decision trace: {trace_path}")
    task_lookup = {str(row["task_key"]): row for row in validation_task_rows}
    source = [
        row
        for row in _read_csv(trace_path)
        if row["analysis_family"] == "policy_improvement"
        and row["allowance_mode"] == "dynamic"
        and row["model_seed"] == row["order_seed"]
    ]
    details: list[dict[str, Any]] = []
    for row in source:
        task = task_lookup[str(row["task_key"])]
        sequence = str(row["sequence"])
        strict_sequences = set(filter(None, str(task["strict_synergy_sequences"]).split(",")))
        details.append(
            {
                "variant": row["variant"],
                "iteration": int(row["iteration"]),
                "seed": int(row["model_seed"]),
                "task_key": row["task_key"],
                "dataset": row["dataset"],
                "selected_sequence": sequence,
                "selected_success": int(row["final_correctness"]),
                "selected_teacher_tokens": int(row["teacher_tokens"]),
                "selected_calls": int(row["teacher_calls"]),
                "selected_multi_verify": int(sequence in MULTI_SEQUENCES),
                "task_has_strict_synergy": int(bool(strict_sequences)),
                "strict_synergy_sequences": ",".join(sorted(strict_sequences)),
                "selected_exact_strict_synergy": int(sequence in strict_sequences),
                "task_requires_multiple_verify": int(
                    task["solvability_category"] == "requires_multiple_verify"
                ),
                "cheapest_success_verify_sequence": task["cheapest_success_verify_sequence"],
                "cheapest_success_verify_tokens": task["cheapest_success_verify_tokens"],
            }
        )
    summary: list[dict[str, Any]] = []
    grouped: defaultdict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[(str(row["variant"]), int(row["iteration"]), int(row["seed"]))].append(row)
    for (variant, iteration, seed), rows in sorted(grouped.items()):
        synergy_tasks = [row for row in rows if int(row["task_has_strict_synergy"]) == 1]
        required_tasks = [row for row in rows if int(row["task_requires_multiple_verify"]) == 1]
        summary.append(
            {
                "variant": variant,
                "iteration": iteration,
                "seed": seed,
                "tasks": len(rows),
                "selected_multi_verify": sum(int(row["selected_multi_verify"]) for row in rows),
                "strict_synergy_task_opportunities": len(synergy_tasks),
                "selected_exact_strict_synergy": sum(
                    int(row["selected_exact_strict_synergy"]) for row in rows
                ),
                "strict_synergy_tasks_solved": sum(int(row["selected_success"]) for row in synergy_tasks),
                "requires_multi_task_opportunities": len(required_tasks),
                "requires_multi_tasks_solved": sum(int(row["selected_success"]) for row in required_tasks),
                "teacher_tokens": sum(int(row["selected_teacher_tokens"]) for row in rows),
            }
        )
    return details, summary


def _plots(
    output: Path,
    task_summary: list[dict[str, Any]],
    marginal_summary: list[dict[str, Any]],
    sequence_summary: list[dict[str, Any]],
    policy_summary: list[dict[str, Any]],
) -> None:
    figure_root = output / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    splits = ("train", "validation")
    categories = (
        "baseline_solved",
        "solvable_by_single_verify",
        "requires_multiple_verify",
        "unsolved_by_observed_trajectories",
    )
    labels = ("Baseline solved", "Single Verify", "Requires multi", "Unsolved")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    bottom = np.zeros(len(splits))
    for category, label in zip(categories, labels, strict=True):
        values = np.asarray(
            [int(next(row for row in task_summary if row["split"] == split)[category]) for split in splits]
        )
        axes[0].bar(splits, values, bottom=bottom, label=label)
        bottom += values
    axes[0].set(title="Task-level solvability", ylabel="Tasks")
    axes[0].legend(frameon=False)

    width = 0.25
    x = np.arange(3)
    for offset, split in enumerate(splits):
        selected = [
            next(
                row
                for row in marginal_summary
                if row["split"] == split and int(row["checkpoint"]) == checkpoint
            )
            for checkpoint in (1, 2, 3)
        ]
        axes[1].bar(
            x + (offset - 0.5) * width,
            [int(row["tasks_with_helpful"]) for row in selected],
            width,
            label=f"{split}: Helpful",
        )
        axes[1].bar(
            x + (offset - 0.5) * width,
            [-int(row["tasks_with_harmful"]) for row in selected],
            width,
            label=f"{split}: Harmful",
            alpha=0.65,
        )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(
        title="Tasks with conditional marginal effects",
        ylabel="Helpful (+) / Harmful (-) tasks",
        xticks=x,
        xticklabels=("CP1", "CP2", "CP3"),
    )
    axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_root / "synergy_task_marginals.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    combined = [row for row in sequence_summary if row["split"] == "combined"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].bar(
        [row["sequence"] for row in combined],
        [100 * float(row["success_rate"]) for row in combined],
        color="#4c78a8",
    )
    axes[0].set(title="Observed success by trajectory", ylabel="Success rate (%)")
    strict = [int(row["strict_synergy_cases"]) for row in combined]
    axes[1].bar([row["sequence"] for row in combined], strict, color="#59a14f")
    axes[1].set(title="Strict synergy cases", ylabel="Task–sequence cases")
    fig.tight_layout()
    fig.savefig(figure_root / "sequence_success_synergy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if policy_summary:
        labels = [f"{row['variant']} i{row['iteration']}\n{str(row['seed'])[-5:]}" for row in policy_summary]
        fig, ax = plt.subplots(figsize=(15, 5.5))
        ax.bar(
            labels,
            [int(row["selected_multi_verify"]) for row in policy_summary],
            label="Selected multi-Verify",
            color="#f28e2b",
        )
        ax.bar(
            labels,
            [int(row["selected_exact_strict_synergy"]) for row in policy_summary],
            label="Selected exact strict synergy",
            color="#59a14f",
        )
        ax.set(title="DARC v0.5 policy selection on validation", ylabel="Tasks")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_root / "policy_synergy_selection.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ]


def _write_report(
    output: Path,
    task_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    task_summary: list[dict[str, Any]],
    strict_rows: list[dict[str, Any]],
    marginal_summary: list[dict[str, Any]],
    interaction_summary: list[dict[str, Any]],
    cheapest_summary: list[dict[str, Any]],
    policy_summary: list[dict[str, Any]],
) -> Path:
    combined_tasks = next(row for row in task_summary if row["split"] == "combined")
    validation_tasks = next(row for row in task_summary if row["split"] == "validation")
    combined_marginals = [row for row in marginal_summary if row["split"] == "combined"]
    marginal_comparisons = sum(int(row["paired_comparisons"]) for row in combined_marginals)
    helpful_comparisons = sum(int(row["helpful"]) for row in combined_marginals)
    harmful_comparisons = sum(int(row["harmful"]) for row in combined_marginals)
    redundant_comparisons = sum(int(row["redundant_success"]) for row in combined_marginals)
    successful_multi = [
        row for row in trajectory_rows if row["sequence"] in MULTI_SEQUENCES and int(row["passed"]) == 1
    ]
    oracle_tasks = [row for row in task_rows if row["cheapest_success_verify_sequence"]]
    oracle_saving = float(
        np.mean([int(row["vvv_tokens"]) - int(row["cheapest_success_verify_tokens"]) for row in oracle_tasks])
    )
    lines = [
        "# DARC v0.5 — Multi-Verification Synergy Audit",
        "",
        "> Train/validation only; task-clustered descriptive audit. No model training, "
        "frozen-test access, or LLM inference.",
        "",
        "## Task-level solvability",
        "",
        *_markdown_table(
            ["Split", "Tasks", "Baseline", "Single Verify", "Requires multi", "Unsolved", "Strict synergy"],
            [
                [
                    row["split"],
                    row["tasks"],
                    row["baseline_solved"],
                    row["solvable_by_single_verify"],
                    row["requires_multiple_verify"],
                    row["unsolved_by_observed_trajectories"],
                    row["tasks_with_strict_synergy"],
                ]
                for row in task_summary
            ],
        ),
        "",
        "![Task solvability and marginal effects](../figures/synergy_task_marginals.png)",
        "",
        "## Key findings",
        "",
        f"- Strict synergy appears in **{combined_tasks['tasks_with_strict_synergy']}/"
        f"{combined_tasks['tasks']} tasks**; only **{combined_tasks['requires_multiple_verify']}** "
        "tasks cannot be solved by baseline or any single-Verify trajectory.",
        f"- Validation contains **{validation_tasks['requires_multiple_verify']}/"
        f"{validation_tasks['tasks']}** tasks requiring multi-Verify. Its one strict-synergy "
        "sequence belongs to a task that also has a successful single-Verify path.",
        f"- Of **{len(successful_multi)}** successful multi-Verify task–sequence branches, "
        f"only **{sum(int(row['strict_synergy']) for row in successful_multi)}** are strict "
        "synergy; the remainder are ordinary multi-Verify successes.",
        f"- Across {marginal_comparisons} paired shared-prefix comparisons, Verify is Helpful "
        f"in {helpful_comparisons} ({100 * helpful_comparisons / marginal_comparisons:.2f}%), "
        f"Harmful in {harmful_comparisons} ({100 * harmful_comparisons / marginal_comparisons:.2f}%), "
        f"and redundant on an already successful path in {redundant_comparisons} "
        f"({100 * redundant_comparisons / marginal_comparisons:.2f}%).",
        f"- The diagnostic cheapest-success oracle saves **{oracle_saving:,.0f} Teacher tokens** "
        "per task on average versus VVV among tasks with at least one successful Verify path.",
        "- Existing DARC v0.5 traces select no exact strict-synergy sequence. All runs still "
        "solve the sole validation strict-synergy task through an alternative successful path.",
        "",
        "## Strict synergy",
        "",
        "A sequence is strict synergy only when it succeeds while `SSS` and every proper "
        "nonempty Verify subset of that sequence fail. Ordinary multi-Verify success is not "
        "counted as synergy.",
        "",
    ]
    if strict_rows:
        lines.extend(
            _markdown_table(
                ["Split", "Task", "Sequence", "Tokens", "Failed proper subsets"],
                [
                    [
                        row["split"],
                        row["task_key"],
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
            "![Trajectory success and strict synergy](../figures/sequence_success_synergy.png)",
            "",
            "## Conditional marginal effects",
            "",
            "Each comparison holds the observed prefix and downstream action suffix fixed. "
            "Counts below are paired comparisons; task counts and task IDs remain available "
            "in the CSVs so trajectories are never treated as independent samples.",
            "",
            *_markdown_table(
                [
                    "Split",
                    "CP",
                    "Comparisons",
                    "Helpful",
                    "Harmful",
                    "Redundant success",
                    "No-change failure",
                ],
                [
                    [
                        row["split"],
                        row["checkpoint"],
                        row["paired_comparisons"],
                        row["helpful"],
                        row["harmful"],
                        row["redundant_success"],
                        row["no_change_failure"],
                    ]
                    for row in marginal_summary
                ],
            ),
            "",
            "## Checkpoint interactions",
            "",
            *_markdown_table(
                [
                    "Split",
                    "Pair",
                    "Conditional comparisons",
                    "Positive",
                    "Negative",
                    "Strict pair synergy",
                ],
                [
                    [
                        row["split"],
                        row["checkpoint_pair"],
                        row["conditional_comparisons"],
                        row["positive_interactions"],
                        row["negative_interactions"],
                        row["strict_pair_synergy"],
                    ]
                    for row in interaction_summary
                ],
            ),
            "",
            "## Cheapest successful verification sequence (diagnostic oracle)",
            "",
            "This uses observed outcomes and actual costs after the fact. It is an oracle for "
            "diagnosis only and is never an inference feature or deployable policy.",
            "",
            *_markdown_table(
                ["Split", "Sequence", "Tasks where cheapest successful Verify path"],
                [
                    [row["split"], row["sequence"], row["tasks"]]
                    for row in cheapest_summary
                    if int(row["tasks"]) > 0
                ],
            ),
            "",
            "## DARC v0.5 selection of synergy cases",
            "",
            "Existing validation decision traces are reused. Training tasks have no policy "
            "replay trace in this audit and are not fabricated.",
            "",
            *_markdown_table(
                [
                    "Variant",
                    "i",
                    "Seed",
                    "Multi selected",
                    "Strict opportunities",
                    "Exact strict selected",
                    "Strict tasks solved",
                    "Tokens",
                ],
                [
                    [
                        row["variant"],
                        row["iteration"],
                        row["seed"],
                        row["selected_multi_verify"],
                        row["strict_synergy_task_opportunities"],
                        row["selected_exact_strict_synergy"],
                        row["strict_synergy_tasks_solved"],
                        row["teacher_tokens"],
                    ]
                    for row in policy_summary
                ],
            ),
            "",
            "![Policy selection](../figures/policy_synergy_selection.png)",
            "",
            "## Interpretation guardrails",
            "",
            "- Eight trajectories from one task are paired counterfactual branches, not "
            "eight independent samples.",
            "- `requires_multiple_verify` means baseline and all three single-Verify paths "
            "fail while at least one observed multi-Verify path succeeds.",
            "- Strict synergy is sequence-specific and stronger: every proper nonempty "
            "subset of that sequence must fail.",
            "- Helpful/Harmful is conditional on both prior actions and the held-fixed future suffix.",
            "- Cheapest-success results use realized outcomes/costs and are diagnostic "
            "oracle quantities only.",
            "",
            "## Artifacts",
            "",
            "- `tables/task_outcome_patterns.csv`",
            "- `tables/task_solvability_summary.csv`",
            "- `tables/trajectory_outcomes.csv` and `sequence_summary.csv`",
            "- `tables/strict_synergy_cases.csv`",
            "- `tables/conditional_marginal_effects.csv` and `marginal_summary.csv`",
            "- `tables/checkpoint_interactions.csv` and `interaction_summary.csv`",
            "- `tables/cheapest_success_summary.csv`",
            "- `tables/policy_synergy_selection.csv` and `policy_synergy_summary.csv`",
            "- `synergy_audit_manifest.json`",
        ]
    )
    path = output / "report" / "darc_v05_multi_verification_synergy_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_synergy_audit(
    config_path: Path,
    *,
    decision_trace_path: Path,
    output: Path,
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["evaluation"]["partition"] != "validation":
        raise ValueError("Synergy audit requires the established validation protocol")
    if config["evaluation"]["frozen_test_access"] != "forbidden":
        raise ValueError("Frozen-test access must remain forbidden")

    frozen_root = Path(config["data"]["frozen_root"])
    train_groups = _load_groups(frozen_root, "train")
    validation_groups = _load_groups(frozen_root, "validation")
    _assert_disjoint(train_groups, validation_groups)
    _validate_groups(train_groups, "train", 381)
    _validate_groups(validation_groups, "validation", 98)

    all_trajectory_rows: list[dict[str, Any]] = []
    all_task_rows: list[dict[str, Any]] = []
    all_strict_rows: list[dict[str, Any]] = []
    all_marginals: list[dict[str, Any]] = []
    all_interactions: list[dict[str, Any]] = []
    for split, groups in (("train", train_groups), ("validation", validation_groups)):
        trajectories, tasks, strict, marginals, interactions = analyze_partition(split, groups)
        all_trajectory_rows.extend(trajectories)
        all_task_rows.extend(tasks)
        all_strict_rows.extend(strict)
        all_marginals.extend(marginals)
        all_interactions.extend(interactions)

    task_summary = _task_summary(all_task_rows)
    sequence_summary = _sequence_summary(all_trajectory_rows)
    marginal_summary = _marginal_summary(all_marginals)
    interaction_summary = _interaction_summary(all_interactions)
    cheapest_summary = _cheapest_summary(all_task_rows)
    validation_task_rows = [row for row in all_task_rows if row["split"] == "validation"]
    policy_details, policy_summary = policy_selection_rows(
        decision_trace_path,
        validation_task_rows,
    )

    table_root = output / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    _write_csv(table_root / "trajectory_outcomes.csv", all_trajectory_rows)
    _write_csv(table_root / "task_outcome_patterns.csv", all_task_rows)
    _write_csv(table_root / "task_solvability_summary.csv", task_summary)
    _write_csv(table_root / "sequence_summary.csv", sequence_summary)
    _write_csv(table_root / "strict_synergy_cases.csv", all_strict_rows)
    _write_csv(table_root / "conditional_marginal_effects.csv", all_marginals)
    _write_csv(table_root / "marginal_summary.csv", marginal_summary)
    _write_csv(table_root / "checkpoint_interactions.csv", all_interactions)
    _write_csv(table_root / "interaction_summary.csv", interaction_summary)
    _write_csv(table_root / "cheapest_success_summary.csv", cheapest_summary)
    _write_csv(table_root / "policy_synergy_selection.csv", policy_details)
    _write_csv(table_root / "policy_synergy_summary.csv", policy_summary)

    _plots(output, task_summary, marginal_summary, sequence_summary, policy_summary)
    report = _write_report(
        output,
        all_task_rows,
        all_trajectory_rows,
        task_summary,
        all_strict_rows,
        marginal_summary,
        interaction_summary,
        cheapest_summary,
        policy_summary,
    )
    manifest = {
        "schema_version": "darc-v05-synergy-audit-v1",
        "config": str(config_path),
        "frozen_root": str(frozen_root),
        "partitions_loaded": ["train", "validation"],
        "frozen_test_loaded": False,
        "tasks": {"train": len(train_groups), "validation": len(validation_groups)},
        "trajectories_per_task": 8,
        "sequence_order": list(SEQUENCES),
        "strict_synergy_definition": (
            "multi sequence succeeds; SSS and every proper nonempty Verify subset fail"
        ),
        "decision_trace": str(decision_trace_path),
        "report": str(report),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "synergy_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
