#!/usr/bin/env python3
"""Create an auditable cross-dataset CBV-Bench status and statistics report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


LEVEL1 = ("SSS", "VSS", "SVS", "SSV")
LEVEL2 = ("SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV")
SINGLE = {"CP1": "VSS", "CP2": "SVS", "CP3": "SSV"}


@dataclass(frozen=True)
class CollectionShard:
    results_dir: Path
    selection: Path


@dataclass(frozen=True)
class CollectionSpec:
    key: str
    label: str
    dataset: str
    level: int
    shards: tuple[CollectionShard, ...]


def build_specs(base_suffix: str, scale_suffix: str) -> tuple[CollectionSpec, ...]:
    return (
        CollectionSpec(
            "humaneval_l2", "HumanEval L2", "humaneval", 2,
            (CollectionShard(
                Path(f"results/cbv_bench_v2_{base_suffix}"),
                Path(
                    f"results/cbv_bench_v2_{base_suffix}/selections/"
                    "humaneval_existing_l2_163.json"
                ),
            ),),
        ),
        CollectionSpec(
            "livecodebench_l2", "LiveCodeBench L2", "livecodebench", 2,
            (CollectionShard(
                Path(f"results/cbv_bench_v2_livecodebench_{base_suffix}"),
                Path(
                    f"results/cbv_bench_v2_livecodebench_{base_suffix}/selections/"
                    "existing_l2_175.json"
                ),
            ),),
        ),
        CollectionSpec(
            "bigcodebench_l1", "BigCodeBench L1 (full)", "bigcodebench", 1,
            (
                CollectionShard(
                    Path(f"results/cbv_bench_v2_bigcodebench_{base_suffix}"),
                    Path(
                        f"results/cbv_bench_v2_bigcodebench_{base_suffix}/selections/"
                        "recovery_l1_200_v2.json"
                    ),
                ),
                CollectionShard(
                    Path(f"results/cbv_bench_v2_bigcodebench_{scale_suffix}"),
                    Path(
                        f"results/cbv_bench_scale_{scale_suffix}/selections/"
                        "level1_completed_scope.json"
                    ),
                ),
            ),
        ),
        CollectionSpec(
            "bigcodebench_l2", "BigCodeBench Natural L2 (full)", "bigcodebench", 2,
            (
                CollectionShard(
                    Path(f"results/cbv_bench_v2_bigcodebench_l2_recovery_{base_suffix}"),
                    Path(
                        f"results/cbv_bench_v2_bigcodebench_{base_suffix}/selections/"
                        "recovery_l2_25_v2.json"
                    ),
                ),
                CollectionShard(
                    Path(
                        "results/cbv_bench_v2_bigcodebench_natural_l2_extension_"
                        f"{scale_suffix}"
                    ),
                    Path(
                        f"results/cbv_bench_scale_{scale_suffix}/selections/"
                        "level2_natural_extension_225_recovered.json"
                    ),
                ),
            ),
        ),
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _latest(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No matching report artifact was found")
    return max(paths, key=lambda path: path.stat().st_mtime)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_collection(spec: CollectionSpec) -> dict[str, Any]:
    shard_validations: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    validation_paths: list[Path] = []
    summary_paths: list[Path] = []
    selected_ids: set[str] = set()
    task_records: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for shard in spec.shards:
        if not shard.selection.exists():
            raise FileNotFoundError(f"Missing selection manifest: {shard.selection}")
        validation_path = _latest(
            list(shard.results_dir.glob("validation/*/*/validation.json"))
        )
        summary_path = _latest(
            list(shard.results_dir.glob(f"reports/*/*_level{spec.level}/summary.json"))
        )
        validation = _json(validation_path)
        summary = _json(summary_path)
        selection = _json(shard.selection)
        shard_records = {
            str(row["task_id"]): row for row in selection["task_records"]
        }
        overlap = selected_ids.intersection(shard_records)
        if overlap:
            raise ValueError(
                f"Collection {spec.key} has overlapping shard task IDs: {sorted(overlap)[:5]}"
            )
        selected_ids.update(shard_records)
        task_records.update(shard_records)
        fingerprint = str(validation["compatibility_fingerprint"])
        rows.extend(
            row
            for row in _jsonl(shard.results_dir / "trajectories.jsonl")
            if row.get("compatibility_fingerprint") == fingerprint
            and row.get("dataset") == spec.dataset
            and str(row.get("task_id")) in shard_records
            and int(row.get("rollout_index", -1)) == 0
            and row.get("status") == "completed"
        )
        shard_validations.append(validation)
        shard_summaries.append(summary)
        validation_paths.append(validation_path)
        summary_paths.append(summary_path)

    missing_branches = {
        key: value
        for validation in shard_validations
        for key, value in validation.get("missing_branches", {}).items()
    }
    validation = {
        "valid": all(value.get("valid", False) for value in shard_validations),
        "logged_collection_errors": sum(
            int(value.get("logged_collection_errors", 0))
            for value in shard_validations
        ),
        "missing_branches": missing_branches,
    }
    multi = Counter()
    interactions = Counter()
    for summary in shard_summaries:
        multi.update(summary.get("multi_vs_best_single", {}))
        interactions.update(summary.get("interaction_labels", {}))
    summary = {
        "actual_unique_teacher_calls": sum(
            int(value.get("actual_unique_teacher_calls", 0)) for value in shard_summaries
        ),
        "actual_unique_teacher_tokens": sum(
            int(value.get("actual_unique_teacher_tokens", 0)) for value in shard_summaries
        ),
        "multi_vs_best_single": dict(multi),
        "interaction_labels": dict(interactions),
    }
    required = LEVEL2 if spec.level == 2 else LEVEL1
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_keys = 0
    for row in rows:
        key = (str(row["task_id"]), int(row["rollout_index"]))
        sequence = str(row["action_sequence"])
        duplicate_keys += sequence in grouped[key]
        grouped[key][sequence] = row
    complete = {
        key: values for key, values in grouped.items() if set(required).issubset(values)
    }
    level1_complete = {
        key: values for key, values in grouped.items() if set(LEVEL1).issubset(values)
    }

    sequence_rows = []
    for sequence in required:
        paired = [values[sequence] for values in complete.values()]
        sequence_rows.append(
            {
                "collection": spec.key,
                "collection_label": spec.label,
                "sequence": sequence,
                "paired_tasks": len(paired),
                "accuracy": mean(int(row["passed"]) for row in paired) if paired else float("nan"),
                "mean_teacher_calls": mean(int(row["teacher_calls"]) for row in paired)
                if paired else float("nan"),
                "mean_teacher_tokens": mean(int(row["teacher_total_tokens"]) for row in paired)
                if paired else float("nan"),
            }
        )
    by_sequence = {row["sequence"]: row for row in sequence_rows}
    baseline = by_sequence["SSS"]
    best_single = max(
        (by_sequence[sequence] for sequence in SINGLE.values()),
        key=lambda row: (row["accuracy"], -row["mean_teacher_tokens"]),
    )
    best_available = max(
        sequence_rows,
        key=lambda row: (row["accuracy"], -row["mean_teacher_tokens"]),
    )

    checkpoint_effects = []
    for checkpoint, sequence in SINGLE.items():
        paired_outcomes = [
            (int(values["SSS"]["passed"]), int(values[sequence]["passed"]))
            for values in complete.values()
        ]
        helpful = sum(skip == 0 and verify == 1 for skip, verify in paired_outcomes)
        harmful = sum(skip == 1 and verify == 0 for skip, verify in paired_outcomes)
        checkpoint_effects.append(
            {
                "collection": spec.key,
                "collection_label": spec.label,
                "checkpoint": checkpoint,
                "sequence": sequence,
                "n": len(paired_outcomes),
                "helpful": helpful,
                "harmful": harmful,
                "no_change": len(paired_outcomes) - helpful - harmful,
                "gain_pp": 100 * (by_sequence[sequence]["accuracy"] - baseline["accuracy"]),
            }
        )

    split_counts = Counter(str(row["split"]) for row in task_records.values())
    expected_trajectories = len(selected_ids) * len(required)
    return {
        "spec": spec,
        "validation": validation,
        "summary": summary,
        "task_records": task_records,
        "selected_ids": selected_ids,
        "split_counts": dict(split_counts),
        "trajectory_rows": rows,
        "observed_trajectories": len(rows),
        "expected_trajectories": expected_trajectories,
        "groups": grouped,
        "complete": complete,
        "level1_complete": level1_complete,
        "sequence_rows": sequence_rows,
        "by_sequence": by_sequence,
        "checkpoint_effects": checkpoint_effects,
        "baseline": baseline,
        "best_single": best_single,
        "best_available": best_available,
        "duplicate_keys": duplicate_keys,
        "validation_paths": validation_paths,
        "summary_paths": summary_paths,
    }


def build_dataset_coverage(
    collections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build non-overlapping Level-1/Level-2 coverage views by benchmark.

    Level 2 contains the four Level-1 branches.  For BigCodeBench, the 250-task
    Natural L2 pool is a subset of the 1,096-task L1 pool, so its L1 branches
    must not be added to the L1 count a second time.
    """
    by_key = {item["spec"].key: item for item in collections}
    rows: list[dict[str, Any]] = []
    for dataset, label, l1_key, l2_key in (
        ("humaneval", "HumanEval", "humaneval_l2", "humaneval_l2"),
        ("livecodebench", "LiveCodeBench", "livecodebench_l2", "livecodebench_l2"),
        ("bigcodebench", "BigCodeBench", "bigcodebench_l1", "bigcodebench_l2"),
    ):
        l1 = by_key[l1_key]
        l2 = by_key[l2_key]
        sequence_counts = {
            sequence: sum(sequence in values for values in l1["groups"].values())
            if sequence in LEVEL1
            else sum(sequence in values for values in l2["groups"].values())
            for sequence in LEVEL2
        }
        rows.append(
            {
                "dataset": dataset,
                "label": label,
                "level1_selected_tasks": len(l1["selected_ids"]),
                "level1_complete_tasks": len(l1["level1_complete"]),
                "level2_selected_tasks": len(l2["selected_ids"]),
                "level2_complete_tasks": len(l2["complete"]),
                **{f"{sequence}_complete_tasks": sequence_counts[sequence]
                   for sequence in LEVEL2},
            }
        )
    return rows


def _bootstrap_mean_ci(
    values: list[float], *, seed: int, repetitions: int = 10_000
) -> tuple[float, float]:
    """Task-clustered percentile CI; one entry is one task group."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return float("nan"), float("nan")
    if data.size == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    remaining = repetitions
    while remaining:
        batch = min(512, remaining)
        indices = rng.integers(0, data.size, size=(batch, data.size))
        estimates.append(data[indices].mean(axis=1))
        remaining -= batch
    samples = np.concatenate(estimates)
    low, high = np.percentile(samples, (2.5, 97.5))
    return float(low), float(high)


def _wilson_ci(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _mcnemar_exact_p(helpful: int, harmful: int) -> float:
    discordant = helpful + harmful
    if discordant == 0:
        return 1.0
    tail = min(helpful, harmful)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2 ** discordant))


def _analysis_sets(
    collections: list[dict[str, Any]],
) -> list[tuple[str, str, int, dict[tuple[str, int], dict[str, dict[str, Any]]], tuple[str, ...]]]:
    by_key = {item["spec"].key: item for item in collections}
    return [
        ("humaneval_l1", "HumanEval L1", 1, by_key["humaneval_l2"]["level1_complete"], LEVEL1),
        ("livecodebench_l1", "LiveCodeBench L1", 1, by_key["livecodebench_l2"]["level1_complete"], LEVEL1),
        ("bigcodebench_l1", "BigCodeBench L1", 1, by_key["bigcodebench_l1"]["level1_complete"], LEVEL1),
        ("humaneval_l2", "HumanEval L2", 2, by_key["humaneval_l2"]["complete"], LEVEL2),
        ("livecodebench_l2", "LiveCodeBench L2", 2, by_key["livecodebench_l2"]["complete"], LEVEL2),
        ("bigcodebench_l2", "BigCodeBench Natural L2", 2, by_key["bigcodebench_l2"]["complete"], LEVEL2),
    ]


def build_statistical_analysis(
    collections: list[dict[str, Any]],
    *,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 20261029,
) -> dict[str, list[dict[str, Any]]]:
    sequence_inference: list[dict[str, Any]] = []
    checkpoint_inference: list[dict[str, Any]] = []
    baseline_correctness: list[dict[str, Any]] = []
    difficulty_analysis: list[dict[str, Any]] = []
    seed_offset = 0

    for key, label, level, groups, sequences in _analysis_sets(collections):
        values = list(groups.values())
        baseline = [int(group["SSS"]["passed"]) for group in values]
        for sequence in sequences:
            outcomes = [int(group[sequence]["passed"]) for group in values]
            differences = [verify - skip for skip, verify in zip(baseline, outcomes)]
            accuracy_low, accuracy_high = _bootstrap_mean_ci(
                outcomes,
                seed=bootstrap_seed + seed_offset,
                repetitions=bootstrap_repetitions,
            )
            seed_offset += 1
            gain_low, gain_high = _bootstrap_mean_ci(
                differences,
                seed=bootstrap_seed + seed_offset,
                repetitions=bootstrap_repetitions,
            )
            seed_offset += 1
            helpful = sum(skip == 0 and verify == 1 for skip, verify in zip(baseline, outcomes))
            harmful = sum(skip == 1 and verify == 0 for skip, verify in zip(baseline, outcomes))
            sequence_inference.append(
                {
                    "analysis_set": key,
                    "label": label,
                    "level": level,
                    "sequence": sequence,
                    "task_clusters": len(values),
                    "accuracy": mean(outcomes),
                    "accuracy_ci95_low": accuracy_low,
                    "accuracy_ci95_high": accuracy_high,
                    "gain_pp_vs_sss": 100 * mean(differences),
                    "gain_ci95_low_pp": 100 * gain_low,
                    "gain_ci95_high_pp": 100 * gain_high,
                    "helpful": helpful,
                    "harmful": harmful,
                    "no_change": len(values) - helpful - harmful,
                    "mcnemar_exact_p": _mcnemar_exact_p(helpful, harmful),
                }
            )

        if level != 1:
            continue
        for checkpoint, sequence in SINGLE.items():
            outcomes = [int(group[sequence]["passed"]) for group in values]
            helpful = sum(skip == 0 and verify == 1 for skip, verify in zip(baseline, outcomes))
            harmful = sum(skip == 1 and verify == 0 for skip, verify in zip(baseline, outcomes))
            helpful_low, helpful_high = _wilson_ci(helpful, len(values))
            harmful_low, harmful_high = _wilson_ci(harmful, len(values))
            checkpoint_inference.append(
                {
                    "analysis_set": key,
                    "label": label,
                    "checkpoint": checkpoint,
                    "sequence": sequence,
                    "task_clusters": len(values),
                    "helpful": helpful,
                    "helpful_rate": helpful / len(values),
                    "helpful_ci95_low": helpful_low,
                    "helpful_ci95_high": helpful_high,
                    "harmful": harmful,
                    "harmful_rate": harmful / len(values),
                    "harmful_ci95_low": harmful_low,
                    "harmful_ci95_high": harmful_high,
                    "mcnemar_exact_p": _mcnemar_exact_p(helpful, harmful),
                }
            )
            baseline_failures = sum(value == 0 for value in baseline)
            baseline_passes = len(values) - baseline_failures
            rescue_low, rescue_high = _wilson_ci(helpful, baseline_failures)
            damage_low, damage_high = _wilson_ci(harmful, baseline_passes)
            baseline_correctness.append(
                {
                    "analysis_set": key,
                    "label": label,
                    "checkpoint": checkpoint,
                    "baseline_incorrect": baseline_failures,
                    "rescued": helpful,
                    "rescue_rate": helpful / baseline_failures if baseline_failures else float("nan"),
                    "rescue_ci95_low": rescue_low,
                    "rescue_ci95_high": rescue_high,
                    "baseline_correct": baseline_passes,
                    "damaged": harmful,
                    "damage_rate": harmful / baseline_passes if baseline_passes else float("nan"),
                    "damage_ci95_low": damage_low,
                    "damage_ci95_high": damage_high,
                }
            )

    for key, label, level, groups, sequences in _analysis_sets(collections):
        if not key.startswith("livecodebench"):
            continue
        grouped_by_difficulty: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
        for values in groups.values():
            grouped_by_difficulty[str(values["SSS"].get("difficulty", "unknown"))].append(values)
        for difficulty, values in sorted(grouped_by_difficulty.items()):
            baseline = [int(group["SSS"]["passed"]) for group in values]
            for sequence in sequences:
                outcomes = [int(group[sequence]["passed"]) for group in values]
                difficulty_analysis.append(
                    {
                        "analysis_set": key,
                        "label": label,
                        "level": level,
                        "difficulty": difficulty,
                        "sequence": sequence,
                        "task_clusters": len(values),
                        "accuracy": mean(outcomes),
                        "gain_pp_vs_sss": 100 * (mean(outcomes) - mean(baseline)),
                    }
                )

    interaction_analysis: list[dict[str, Any]] = []
    for item in collections:
        if item["spec"].level != 2:
            continue
        multi = item["summary"]["multi_vs_best_single"]
        interactions = item["summary"]["interaction_labels"]
        interaction_total = sum(int(interactions.get(name, 0)) for name in ("Positive", "Negative", "Zero"))
        interaction_analysis.append(
            {
                "collection": item["spec"].key,
                "label": item["spec"].label,
                "task_clusters": int(multi["comparable"]),
                "multi_better": int(multi["better"]),
                "multi_worse": int(multi["worse"]),
                "multi_equal": int(multi["equal"]),
                "positive_interactions": int(interactions.get("Positive", 0)),
                "negative_interactions": int(interactions.get("Negative", 0)),
                "zero_interactions": int(interactions.get("Zero", 0)),
                "interaction_contrasts": interaction_total,
            }
        )

    pareto_frontier: list[dict[str, Any]] = []
    for item in collections:
        rows = item["sequence_rows"]
        for row in rows:
            dominated = any(
                other["mean_teacher_tokens"] <= row["mean_teacher_tokens"]
                and other["accuracy"] >= row["accuracy"]
                and (
                    other["mean_teacher_tokens"] < row["mean_teacher_tokens"]
                    or other["accuracy"] > row["accuracy"]
                )
                for other in rows
            )
            pareto_frontier.append(
                {
                    "collection": item["spec"].key,
                    "label": item["spec"].label,
                    "sequence": row["sequence"],
                    "task_clusters": row["paired_tasks"],
                    "accuracy": row["accuracy"],
                    "mean_teacher_calls": row["mean_teacher_calls"],
                    "mean_teacher_tokens": row["mean_teacher_tokens"],
                    "strict_pareto": not dominated,
                }
            )

    return {
        "sequence_inference": sequence_inference,
        "checkpoint_inference": checkpoint_inference,
        "baseline_correctness": baseline_correctness,
        "difficulty_analysis": difficulty_analysis,
        "interaction_analysis": interaction_analysis,
        "pareto_frontier": pareto_frontier,
    }


def make_figures(
    collections: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    statistical: dict[str, list[dict[str, Any]]],
    output: Path,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#2563EB", "#F59E0B", "#16A34A", "#7C3AED"]
    labels = [item["spec"].label for item in collections]
    short = ["HE L2", "LCB L2", "BCB L1", "BCB L2"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle(
        "CBV-Bench: HumanEval + LiveCodeBench + BigCodeBench",
        fontsize=20, fontweight="bold", y=0.99,
    )

    axis = axes[0, 0]
    coverage_y = np.arange(len(coverage))
    l1_complete = [row["level1_complete_tasks"] for row in coverage]
    l2_complete = [row["level2_complete_tasks"] for row in coverage]
    axis.barh(coverage_y + 0.18, l1_complete, 0.34, color="#2563EB", label="Level 1")
    axis.barh(coverage_y - 0.18, l2_complete, 0.34, color="#8B5CF6", label="Level 2")
    axis.set(
        yticks=coverage_y,
        yticklabels=[row["label"] for row in coverage],
        xlabel="Complete tasks",
        title="Complete task coverage by level",
    )
    axis.invert_yaxis()
    axis.legend(fontsize=9)
    for index, (level1, level2) in enumerate(zip(l1_complete, l2_complete)):
        axis.text(level1 + 12, index + 0.18, f"{level1:,}", va="center", fontsize=9)
        axis.text(level2 + 12, index - 0.18, f"{level2:,}", va="center", fontsize=9)

    axis = axes[0, 1]
    expected = [item["expected_trajectories"] for item in collections]
    observed = [item["observed_trajectories"] for item in collections]
    x = np.arange(len(short))
    axis.bar(x - 0.18, expected, 0.36, label="Planned", color="#CBD5E1")
    axis.bar(x + 0.18, observed, 0.36, label="Observed", color=colors)
    axis.set(xticks=x, xticklabels=short, ylabel="Trajectories", title="Planned vs observed records")
    axis.legend()
    for index, value in enumerate(observed):
        axis.text(index + 0.18, value + max(expected) * 0.015, str(value), ha="center", fontsize=9)

    axis = axes[1, 0]
    baseline = [100 * item["baseline"]["accuracy"] for item in collections]
    best_single = [100 * item["best_single"]["accuracy"] for item in collections]
    best_any = [100 * item["best_available"]["accuracy"] for item in collections]
    axis.bar(x - 0.25, baseline, 0.25, label="SSS baseline", color="#94A3B8")
    axis.bar(x, best_single, 0.25, label="Best single verify", color="#0EA5E9")
    axis.bar(x + 0.25, best_any, 0.25, label="Best observed sequence", color="#22C55E")
    axis.set(xticks=x, xticklabels=short, ylabel="Paired final accuracy (%)", title="Verification accuracy gains")
    axis.legend(fontsize=9)

    axis = axes[1, 1]
    gains = np.array([
        [next(row["gain_pp"] for row in item["checkpoint_effects"] if row["checkpoint"] == cp) for cp in ("CP1", "CP2", "CP3")]
        for item in collections
    ])
    limit = max(1.0, float(np.nanmax(np.abs(gains))))
    image = axis.imshow(gains, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    axis.set(xticks=range(3), xticklabels=("CP1 Planner", "CP2 Coder", "CP3 Reviewer"),
             yticks=range(4), yticklabels=short, title="Single-verification gain over SSS (pp)")
    for row in range(gains.shape[0]):
        for column in range(gains.shape[1]):
            axis.text(column, row, f"{gains[row, column]:+.1f}", ha="center", va="center", fontweight="bold")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Percentage points")

    axis = axes[2, 0]
    y_labels, helpful, harmful = [], [], []
    for item, tag in zip(collections, short):
        for row in item["checkpoint_effects"]:
            y_labels.append(f"{tag} {row['checkpoint']}")
            helpful.append(row["helpful"])
            harmful.append(-row["harmful"])
    y = np.arange(len(y_labels))
    axis.barh(y, helpful, color="#16A34A", label="Helpful")
    axis.barh(y, harmful, color="#DC2626", label="Harmful")
    axis.axvline(0, color="#0F172A", linewidth=0.8)
    axis.set(yticks=y, yticklabels=y_labels, xlabel="Interventions (harmful shown negative)",
             title="Helpful vs harmful single verification")
    axis.invert_yaxis(); axis.legend(fontsize=9)

    axis = axes[2, 1]
    calls = [item["summary"]["actual_unique_teacher_calls"] for item in collections]
    tokens_m = [item["summary"]["actual_unique_teacher_tokens"] / 1_000_000 for item in collections]
    axis.bar(x - 0.18, calls, 0.36, color="#8B5CF6", label="Teacher calls")
    axis.set(xticks=x, xticklabels=short, ylabel="Teacher calls", title="Teacher collection cost")
    twin = axis.twinx()
    twin.plot(x + 0.18, tokens_m, color="#EF4444", marker="o", linewidth=2.5, label="Teacher tokens")
    twin.set_ylabel("Teacher tokens (millions)")
    handles, legends = axis.get_legend_handles_labels()
    handles2, legends2 = twin.get_legend_handles_labels()
    axis.legend(handles + handles2, legends + legends2, loc="upper right", fontsize=9)

    fig.text(
        0.5, 0.005,
        "Accuracy uses only fully paired task-rollouts. LiveCodeBench L2 is partial: 171/175 complete.",
        ha="center", fontsize=10, color="#475569",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    fig.savefig(figures / "cbv_bench_dashboard.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures / "cbv_bench_dashboard.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6.8), gridspec_kw={"width_ratios": (0.9, 1.35)}
    )
    fig.suptitle(
        "CBV-Bench dataset inventory: Level 1 and Level 2 coverage",
        fontsize=19, fontweight="bold", y=0.98,
    )
    axis = axes[0]
    y = np.arange(len(coverage))
    bars_l1 = axis.barh(
        y + 0.18, l1_complete, 0.34, color="#2563EB", label="Level 1 complete"
    )
    bars_l2 = axis.barh(
        y - 0.18, l2_complete, 0.34, color="#8B5CF6", label="Level 2 complete"
    )
    axis.set(
        yticks=y,
        yticklabels=[row["label"] for row in coverage],
        xlabel="Complete task groups",
        title="Usable paired task coverage",
    )
    axis.invert_yaxis()
    axis.legend(loc="upper right")
    axis.bar_label(bars_l1, labels=[f"{value:,}" for value in l1_complete], padding=4)
    axis.bar_label(bars_l2, labels=[f"{value:,}" for value in l2_complete], padding=4)

    axis = axes[1]
    matrix = np.array([
        [row[f"{sequence}_complete_tasks"] for sequence in LEVEL2]
        for row in coverage
    ])
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    axis.set(
        xticks=np.arange(len(LEVEL2)),
        xticklabels=LEVEL2,
        yticks=np.arange(len(coverage)),
        yticklabels=[row["label"] for row in coverage],
        xlabel="Verification sequence (CP1 → CP2 → CP3)",
        title="Completed task outcomes per sequence",
    )
    axis.axvline(3.5, color="#F97316", linewidth=2.2)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            color = "white" if value > 0.58 * matrix.max() else "#0F172A"
            axis.text(
                column_index, row_index, f"{value:,}", ha="center", va="center",
                color=color, fontweight="bold", fontsize=10,
            )
    axis.text(
        1.5, -0.74, "Level 1: baseline + single verification", ha="center",
        color="#1D4ED8", fontweight="bold", transform=axis.transData,
    )
    axis.text(
        5.5, -0.74, "Level 2: multi-verification branches", ha="center",
        color="#7C3AED", fontweight="bold", transform=axis.transData,
    )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Complete tasks")
    fig.text(
        0.5, 0.015,
        "BigCodeBench Natural L2 is a 250-task subset of the 1,096-task L1 pool; "
        "counts are coverage, not independent task totals.",
        ha="center", fontsize=10, color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(figures / "dataset_level_coverage.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "dataset_level_coverage.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=False)
    for axis, item, color in zip(axes.flat, collections, colors):
        rows = item["sequence_rows"]
        names = [row["sequence"] for row in rows]
        values = [100 * row["accuracy"] for row in rows]
        axis.plot(names, values, color=color, marker="o", linewidth=2.5)
        axis.axhline(100 * item["baseline"]["accuracy"], color="#64748B", linestyle="--", linewidth=1)
        axis.set(title=f"{item['spec'].label} (paired n={len(item['complete'])})",
                 xlabel="Verification sequence", ylabel="Final accuracy (%)")
        for name, value in zip(names, values):
            axis.annotate(f"{value:.1f}", (name, value), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    fig.suptitle("Paired final accuracy by verification sequence", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures / "accuracy_by_sequence.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures / "accuracy_by_sequence.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 7))
    for item, color in zip(collections, colors):
        rows = sorted(item["sequence_rows"], key=lambda row: row["mean_teacher_tokens"])
        tokens = [row["mean_teacher_tokens"] for row in rows]
        accuracy = [100 * row["accuracy"] for row in rows]
        axis.plot(tokens, accuracy, marker="o", linewidth=1.8, color=color, label=item["spec"].label)
        for row, x_value, y_value in zip(rows, tokens, accuracy):
            axis.annotate(row["sequence"], (x_value, y_value), textcoords="offset points", xytext=(4, 4), fontsize=7)
    axis.set(title="Observed accuracy–Teacher-token frontier", xlabel="Mean Teacher tokens / trajectory",
             ylabel="Paired final accuracy (%)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "accuracy_cost_frontier.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures / "accuracy_cost_frontier.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    level1_rows = [
        row for row in statistical["sequence_inference"]
        if row["level"] == 1 and row["sequence"] in SINGLE.values()
    ]
    checkpoint_index = {sequence: index for index, sequence in enumerate(SINGLE.values())}
    dataset_order = ("humaneval_l1", "livecodebench_l1", "bigcodebench_l1")
    dataset_colors = ("#2563EB", "#F59E0B", "#16A34A")
    offsets = (-0.18, 0.0, 0.18)
    axis = axes[0]
    for dataset_key, color, offset in zip(dataset_order, dataset_colors, offsets):
        rows = sorted(
            (row for row in level1_rows if row["analysis_set"] == dataset_key),
            key=lambda row: checkpoint_index[row["sequence"]],
        )
        x_values = np.arange(3) + offset
        gains = np.array([row["gain_pp_vs_sss"] for row in rows])
        low = gains - np.array([row["gain_ci95_low_pp"] for row in rows])
        high = np.array([row["gain_ci95_high_pp"] for row in rows]) - gains
        axis.errorbar(
            x_values, gains, yerr=np.vstack((low, high)), marker="o", capsize=4,
            linewidth=2, color=color, label=rows[0]["label"],
        )
    axis.axhline(0, color="#0F172A", linewidth=1)
    axis.set(
        xticks=np.arange(3), xticklabels=("CP1 Planner", "CP2 Coder", "CP3 Reviewer"),
        ylabel="Accuracy gain over SSS (pp)",
        title="Level 1 gain with task-clustered bootstrap 95% CI",
    )
    axis.legend()

    axis = axes[1]
    interaction_rows = statistical["interaction_analysis"]
    labels_multi = [row["label"].replace(" Natural", "") for row in interaction_rows]
    better = np.array([row["multi_better"] for row in interaction_rows])
    worse = np.array([row["multi_worse"] for row in interaction_rows])
    y_multi = np.arange(len(interaction_rows))
    axis.barh(y_multi, better, color="#16A34A", height=0.44, label="Multi better")
    axis.barh(y_multi, -worse, color="#DC2626", height=0.44, label="Multi worse")
    axis.axvline(0, color="#0F172A", linewidth=1.1)
    axis.set(
        yticks=y_multi, yticklabels=labels_multi, xlim=(-3.2, 8.2),
        xlabel="Tasks  ← multi worse | multi better →",
        title="Rare task-level value of multi-verification",
    )
    axis.invert_yaxis()
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=True
    )
    for index, row in enumerate(interaction_rows):
        axis.text(
            row["multi_better"] + 0.15, index,
            f"{row['multi_better']} ({100*row['multi_better']/row['task_clusters']:.1f}%)",
            va="center", color="#14532D", fontweight="bold", fontsize=10,
        )
        axis.text(
            -row["multi_worse"] - 0.15, index,
            f"{row['multi_worse']}", ha="right", va="center",
            color="#7F1D1D", fontweight="bold", fontsize=10,
        )
        axis.text(
            8.05, index,
            f"equal {row['multi_equal']} ({100*row['multi_equal']/row['task_clusters']:.1f}%)",
            ha="right", va="bottom", color="#475569", fontsize=9,
        )
    fig.suptitle("CBV-Bench statistical validation", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(figures / "statistical_validation.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "statistical_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    pareto_by_collection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in statistical["pareto_frontier"]:
        pareto_by_collection[row["collection"]].append(row)
    for axis, item, color in zip(axes.flat, collections, colors):
        rows = pareto_by_collection[item["spec"].key]
        axis.scatter(
            [row["mean_teacher_tokens"] for row in rows],
            [100 * row["accuracy"] for row in rows],
            color="#94A3B8", s=55, label="Dominated/observed", zorder=2,
        )
        frontier = sorted(
            (row for row in rows if row["strict_pareto"]),
            key=lambda row: row["mean_teacher_tokens"],
        )
        axis.plot(
            [row["mean_teacher_tokens"] for row in frontier],
            [100 * row["accuracy"] for row in frontier],
            color=color, marker="o", linewidth=2.5, markersize=7,
            label="Strict Pareto frontier", zorder=3,
        )
        for row in rows:
            axis.annotate(
                row["sequence"],
                (row["mean_teacher_tokens"], 100 * row["accuracy"]),
                textcoords="offset points", xytext=(4, 5), fontsize=8,
                fontweight="bold" if row["strict_pareto"] else "normal",
            )
        axis.set(
            title=f"{item['spec'].label} (n={len(item['complete'])})",
            xlabel="Mean Teacher tokens / task", ylabel="Paired final accuracy (%)",
        )
        axis.legend(fontsize=8)
    fig.suptitle(
        "Strict accuracy–cost Pareto frontiers",
        fontsize=18, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures / "strict_pareto_frontier.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "strict_pareto_frontier.pdf", bbox_inches="tight")
    plt.close(fig)


def markdown_report(
    collections: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    statistical: dict[str, list[dict[str, Any]]],
    totals: dict[str, Any],
) -> str:
    lines = [
        "# CBV-Bench — Báo cáo tổng hợp HumanEval, LiveCodeBench và BigCodeBench",
        "",
        "> Báo cáo chỉ phân tích dữ liệu đã thu thập; không chạy thêm LLM inference.",
        "",
        "![CBV-Bench dashboard](figures/cbv_bench_dashboard.png)",
        "",
        "## 1. Trạng thái collection",
        "",
        "| Collection | Valid | Selected tasks | Full-level complete | Trajectories | Completion | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in collections:
        lines.append(
            f"| {item['spec'].label} | {'Yes' if item['validation']['valid'] else '**No — partial**'} "
            f"| {len(item['selected_ids'])} | {len(item['complete'])} | "
            f"{item['observed_trajectories']}/{item['expected_trajectories']} | "
            f"{100 * len(item['complete']) / len(item['selected_ids']):.1f}% | "
            f"{item['validation']['logged_collection_errors']} |"
        )
    lines.extend(
        [
            "",
            f"- Unique selected task IDs: **{totals['unique_selected_tasks']}**.",
            f"- Collection task-rollouts (BigCodeBench L2 counted as a separate condition): **{totals['collection_task_rollouts']}**.",
            f"- Observed trajectories: **{totals['observed_trajectories']:,}/{totals['planned_trajectories']:,}** ({totals['trajectory_completion_pct']:.2f}%).",
            f"- Teacher usage: **{totals['teacher_calls']:,} calls / {totals['teacher_tokens']:,} tokens**.",
            "",
            "Kết luận trạng thái: **HumanEval L2 và hai collection BigCodeBench đã hoàn chỉnh; LiveCodeBench L2 vẫn giữ trạng thái partial đã audit.**",
            "",
            "### Coverage Level 1 và Level 2",
            "",
            "![Dataset coverage by level](figures/dataset_level_coverage.png)",
            "",
            "| Dataset | Level 1 complete | Level 1 target | Level 2 complete | Level 2 target |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in coverage:
        lines.append(
            f"| {row['label']} | {row['level1_complete_tasks']} | "
            f"{row['level1_selected_tasks']} | {row['level2_complete_tasks']} | "
            f"{row['level2_selected_tasks']} |"
        )
    lines.extend(
        [
            "",
            "Level 1 gồm `SSS/VSS/SVS/SSV`; Level 2 bổ sung `VVS/VSV/SVV/VVV`. "
            "BigCodeBench Natural L2 là subset 250 task của pool L1 1,096 task, không phải 250 task độc lập mới.",
            "",
            "### LiveCodeBench còn thiếu",
            "",
        ]
    )
    lcb = next(item for item in collections if item["spec"].key == "livecodebench_l2")
    for key, sequences in lcb["validation"]["missing_branches"].items():
        lines.append(f"- `{key}`: {', '.join(sequences)}")
    lines.extend(
        [
            "",
            f"Các thống kê Level 2 của LiveCodeBench chỉ dùng **{len(lcb['complete'])} task có đủ tám sequence**.",
            "",
            "## 2. Accuracy paired và chi phí",
            "",
            "| Collection | Paired n | SSS | Best single | Gain | Best observed | Gain | Mean Teacher tokens (best) |",
            "|---|---:|---:|---|---:|---|---:|---:|",
        ]
    )
    for item in collections:
        baseline = item["baseline"]
        single = item["best_single"]
        best = item["best_available"]
        lines.append(
            f"| {item['spec'].label} | {len(item['complete'])} | {100*baseline['accuracy']:.2f}% | "
            f"{single['sequence']} — {100*single['accuracy']:.2f}% | "
            f"{100*(single['accuracy']-baseline['accuracy']):+.2f} pp | "
            f"{best['sequence']} — {100*best['accuracy']:.2f}% | "
            f"{100*(best['accuracy']-baseline['accuracy']):+.2f} pp | "
            f"{best['mean_teacher_tokens']:.0f} |"
        )
    lines.extend(
        [
            "",
            "![Accuracy by sequence](figures/accuracy_by_sequence.png)",
            "",
            "![Accuracy cost frontier](figures/accuracy_cost_frontier.png)",
            "",
            "### Nhận xét",
            "",
            "- Checkpoint tốt nhất không cố định giữa dataset và mức branching.",
            "- Điều này ủng hộ việc học policy theo task/checkpoint thay vì hard-code một checkpoint chung.",
            "- BigCodeBench Natural L2 được báo cáo trên pool 250 task thay vì pilot 25 task trước đây.",
            "",
            "## 3. Statistical validation",
            "",
            "> Characterization dùng 10,000 task-clustered bootstrap resamples (seed 20261029). "
            "Đây chưa phải confirmatory test trên frozen holdout.",
            "",
            "![Statistical validation](figures/statistical_validation.png)",
            "",
            "![Strict Pareto frontier](figures/strict_pareto_frontier.png)",
            "",
            "| Level 1 set | Checkpoint | Gain (pp) | Bootstrap 95% CI | McNemar exact p | Helpful rate 95% CI | Harmful rate 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    sequence_lookup = {
        (row["analysis_set"], row["sequence"]): row
        for row in statistical["sequence_inference"]
    }
    for row in statistical["checkpoint_inference"]:
        inference = sequence_lookup[(row["analysis_set"], row["sequence"])]
        lines.append(
            f"| {row['label']} | {row['checkpoint']} | {inference['gain_pp_vs_sss']:+.2f} | "
            f"[{inference['gain_ci95_low_pp']:+.2f}, {inference['gain_ci95_high_pp']:+.2f}] | "
            f"{row['mcnemar_exact_p']:.4f} | "
            f"[{100*row['helpful_ci95_low']:.2f}%, {100*row['helpful_ci95_high']:.2f}%] | "
            f"[{100*row['harmful_ci95_low']:.2f}%, {100*row['harmful_ci95_high']:.2f}%] |"
        )
    lines.extend(
        [
            "",
            "### LiveCodeBench theo difficulty (Level 1)",
            "",
            "| Difficulty | Tasks | SSS | Best single | Gain |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    lcb_difficulty = [
        row for row in statistical["difficulty_analysis"]
        if row["analysis_set"] == "livecodebench_l1"
    ]
    for difficulty in sorted({row["difficulty"] for row in lcb_difficulty}):
        rows = [row for row in lcb_difficulty if row["difficulty"] == difficulty]
        baseline_row = next(row for row in rows if row["sequence"] == "SSS")
        best_row = max(
            (row for row in rows if row["sequence"] in SINGLE.values()),
            key=lambda row: row["accuracy"],
        )
        lines.append(
            f"| {difficulty} | {baseline_row['task_clusters']} | {100*baseline_row['accuracy']:.2f}% | "
            f"{best_row['sequence']} — {100*best_row['accuracy']:.2f}% | "
            f"{best_row['gain_pp_vs_sss']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "HumanEval và BigCodeBench hiện có difficulty=`unknown`; không suy diễn hoặc tự tạo nhãn độ khó.",
            "",
            "## 4. Helpful/Harmful theo checkpoint",
            "",
            "| Collection | Checkpoint | Gain vs SSS | Helpful | Harmful | No change |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in collections:
        for row in item["checkpoint_effects"]:
            lines.append(
                f"| {item['spec'].label} | {row['checkpoint']} | {row['gain_pp']:+.2f} pp | "
                f"{row['helpful']} | {row['harmful']} | {row['no_change']} |"
            )
    lines.extend(
        [
            "",
            "## 5. Multi-verification và interaction",
            "",
            "| Collection | Comparable tasks | Multi better than best single | Worse | Equal | Positive interactions | Negative | Zero |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in collections:
        if item["spec"].level != 2:
            continue
        multi = item["summary"]["multi_vs_best_single"]
        interactions = item["summary"]["interaction_labels"]
        lines.append(
            f"| {item['spec'].label} | {multi['comparable']} | {multi['better']} | "
            f"{multi['worse']} | {multi['equal']} | {interactions.get('Positive', 0)} | "
            f"{interactions.get('Negative', 0)} | {interactions.get('Zero', 0)} |"
        )
    lines.extend(
        [
            "",
            f"Trên **{totals['multi_comparable']}** task Level 2 hoàn chỉnh, multi-verification tốt hơn best single ở **{totals['multi_better']}** task, kém hơn ở **{totals['multi_worse']}**, và bằng nhau ở **{totals['multi_equal']}**. Vì vậy lợi ích bổ sung của verify nhiều lần hiện khá thưa và cần controller cân nhắc chi phí.",
            "",
            "## 6. Khả năng tái sử dụng cho training",
            "",
            f"- HumanEval L2: dùng {len(next(item for item in collections if item['spec'].key == 'humaneval_l2')['complete'])} task đầy đủ.",
            f"- LiveCodeBench L2: dùng {len(lcb['complete'])} task có đủ tám nhánh.",
            f"- BigCodeBench L1: dùng {len(next(item for item in collections if item['spec'].key == 'bigcodebench_l1')['complete'])} task đầy đủ.",
            f"- BigCodeBench Natural L2: dùng {len(next(item for item in collections if item['spec'].key == 'bigcodebench_l2')['complete'])} task đầy đủ.",
            "- BigCodeBench L2 là subset của BigCodeBench L1: split train/validation/test phải theo `task_id`, không được coi hai collection là task độc lập.",
            "- Khi train combined, chỉ lấy group có đủ counterfactual sequences cần thiết và giữ grouped split theo task để tránh leakage.",
            "",
            "## 7. Giới hạn khoa học",
            "",
            "- LiveCodeBench chưa phải full-valid dataset; không báo cáo 175/175 như đã hoàn tất.",
            "- So sánh accuracy trong báo cáo này là paired trên task đầy đủ, nhưng chưa kèm bootstrap confidence intervals.",
            "- Best observed sequence và multi-vs-best-single là phân tích offline, không phải policy có thể triển khai online.",
            "- Bốn prompt family khác nhau theo dataset; cross-dataset training phải giữ prompt/model metadata làm provenance.",
            "- Các task BigCodeBench bị loại do infrastructure attrition đã được audit riêng; không loại theo correctness label.",
            "",
            "## 8. Artifacts",
            "",
            "- `collection_summary.csv`: trạng thái và chi phí từng collection.",
            "- `dataset_level_coverage.csv`: coverage Level 1/Level 2 và số task theo từng sequence.",
            "- `sequence_accuracy_paired.csv`: accuracy/cost trên paired task groups.",
            "- `checkpoint_effects.csv`: helpful/harmful và gain theo checkpoint.",
            "- `sequence_inference.csv`: bootstrap CI và exact McNemar theo sequence.",
            "- `checkpoint_inference.csv`: Wilson CI cho Helpful/Harmful rates.",
            "- `baseline_correctness_analysis.csv`: rescue/damage theo baseline correctness.",
            "- `difficulty_analysis.csv`: breakdown theo difficulty khả dụng.",
            "- `interaction_analysis.csv`: multi-verification và interaction counts.",
            "- `strict_pareto_frontier.csv`: nhãn nondominated accuracy–cost points.",
            "- `summary.json`: số liệu machine-readable.",
            "- `figures/`: dashboard PNG/PDF và các biểu đồ chi tiết.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/cbv_bench_all_28102026"),
    )
    parser.add_argument("--base-date-suffix", default="26102026")
    parser.add_argument("--scale-date-suffix", default="28102026")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args.base_date_suffix, args.scale_date_suffix)
    collections = [load_collection(spec) for spec in specs]
    statistical = build_statistical_analysis(collections)

    unique_keys = {
        (item["spec"].dataset, task_id)
        for item in collections for task_id in item["selected_ids"]
    }
    level2 = [item for item in collections if item["spec"].level == 2]
    totals = {
        "schema_version": "cbv-cross-dataset-summary-v1",
        "all_collections_valid": all(item["validation"]["valid"] for item in collections),
        "unique_selected_tasks": len(unique_keys),
        "collection_task_rollouts": sum(len(item["selected_ids"]) for item in collections),
        "full_level_complete_task_rollouts": sum(len(item["complete"]) for item in collections),
        "planned_trajectories": sum(item["expected_trajectories"] for item in collections),
        "observed_trajectories": sum(item["observed_trajectories"] for item in collections),
        "teacher_calls": sum(int(item["summary"]["actual_unique_teacher_calls"]) for item in collections),
        "teacher_tokens": sum(int(item["summary"]["actual_unique_teacher_tokens"]) for item in collections),
        "multi_comparable": sum(int(item["summary"]["multi_vs_best_single"]["comparable"]) for item in level2),
        "multi_better": sum(int(item["summary"]["multi_vs_best_single"]["better"]) for item in level2),
        "multi_worse": sum(int(item["summary"]["multi_vs_best_single"]["worse"]) for item in level2),
        "multi_equal": sum(int(item["summary"]["multi_vs_best_single"]["equal"]) for item in level2),
    }
    totals["trajectory_completion_pct"] = (
        100 * totals["observed_trajectories"] / totals["planned_trajectories"]
    )

    collection_rows = []
    for item in collections:
        collection_rows.append(
            {
                "collection": item["spec"].key,
                "label": item["spec"].label,
                "dataset": item["spec"].dataset,
                "level": item["spec"].level,
                "validation_valid": item["validation"]["valid"],
                "selected_tasks": len(item["selected_ids"]),
                "full_level_complete_tasks": len(item["complete"]),
                "level1_complete_tasks": len(item["level1_complete"]),
                "planned_trajectories": item["expected_trajectories"],
                "observed_trajectories": item["observed_trajectories"],
                "completion_pct": 100 * len(item["complete"]) / len(item["selected_ids"]),
                "teacher_calls": item["summary"]["actual_unique_teacher_calls"],
                "teacher_tokens": item["summary"]["actual_unique_teacher_tokens"],
                "logged_errors": item["validation"]["logged_collection_errors"],
                "train_tasks": item["split_counts"].get("train", 0),
                "validation_tasks": item["split_counts"].get("validation", 0),
                "test_tasks": item["split_counts"].get("test", 0),
                "exploratory_tasks": item["split_counts"].get("exploratory", 0),
            }
        )
    sequence_rows = [row for item in collections for row in item["sequence_rows"]]
    checkpoint_rows = [row for item in collections for row in item["checkpoint_effects"]]
    coverage = build_dataset_coverage(collections)
    _write_csv(args.output_dir / "collection_summary.csv", collection_rows)
    _write_csv(args.output_dir / "sequence_accuracy_paired.csv", sequence_rows)
    _write_csv(args.output_dir / "checkpoint_effects.csv", checkpoint_rows)
    _write_csv(args.output_dir / "dataset_level_coverage.csv", coverage)
    _write_csv(args.output_dir / "sequence_inference.csv", statistical["sequence_inference"])
    _write_csv(args.output_dir / "checkpoint_inference.csv", statistical["checkpoint_inference"])
    _write_csv(
        args.output_dir / "baseline_correctness_analysis.csv",
        statistical["baseline_correctness"],
    )
    _write_csv(args.output_dir / "difficulty_analysis.csv", statistical["difficulty_analysis"])
    _write_csv(args.output_dir / "interaction_analysis.csv", statistical["interaction_analysis"])
    _write_csv(args.output_dir / "strict_pareto_frontier.csv", statistical["pareto_frontier"])
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                **totals,
                "statistical_protocol": {
                    "analysis_stage": "pre-freeze benchmark characterization",
                    "bootstrap_unit": "task_id cluster",
                    "bootstrap_repetitions": 10000,
                    "bootstrap_seed": 20261029,
                    "confidence_level": 0.95,
                    "paired_test": "two-sided exact McNemar",
                    "proportion_interval": "Wilson score",
                },
                "collections": collection_rows,
                "dataset_level_coverage": coverage,
                "livecodebench_missing_branches": next(
                    item["validation"]["missing_branches"]
                    for item in collections if item["spec"].key == "livecodebench_l2"
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    make_figures(collections, coverage, statistical, args.output_dir)
    report = markdown_report(collections, coverage, statistical, totals)
    report_path = args.output_dir / "CBV_Bench_all_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path.resolve())


if __name__ == "__main__":
    main()
