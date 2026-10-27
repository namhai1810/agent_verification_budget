"""Reproducible summaries and visualizations for CBV-Bench collections."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib.pyplot as plt

from agent_verify.cbv_bench.config import CBVConfig
from agent_verify.cbv_bench.datasets import project_path
from agent_verify.cbv_bench.labels import build_effect_records
from agent_verify.cbv_bench.sequences import LEVEL1_SEQUENCES, LEVEL2_SEQUENCES
from agent_verify.cbv_bench.validation import validate_cbv_data


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate_report(config: CBVConfig, project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[3]
    results = project_path(config.collection.results_dir, root)
    validation = validate_cbv_data(config, root)
    fingerprint = str(validation["compatibility_fingerprint"])
    allowed = {tuple(value.split("::", 1)) for value in validation["scope_task_keys"]}
    trajectories = [row for row in _read(results / "trajectories.jsonl")
                    if row.get("compatibility_fingerprint") == fingerprint
                    and (row.get("dataset"), row.get("task_id")) in allowed
                    and int(row.get("rollout_index", -1)) < config.collection.repeated_rollouts]
    transitions = [row for row in _read(results / "cache" / "transitions.jsonl")
                   if row.get("compatibility_fingerprint") == fingerprint
                   and (row.get("dataset"), row.get("task_id")) in allowed
                   and int(row.get("rollout_index", -1)) < config.collection.repeated_rollouts]
    roots = [row for row in _read(results / "cache" / "roots.jsonl")
             if row.get("compatibility_fingerprint") == fingerprint
             and (row.get("dataset"), row.get("task_id")) in allowed
             and int(row.get("rollout_index", -1)) < config.collection.repeated_rollouts]
    effects_path = build_effect_records(results, fingerprint, config.collection.level)
    effects = [
        row for row in _read(effects_path)
        if (row.get("dataset"), row.get("task_id")) in allowed
        and int(row.get("rollout_index", -1)) < config.collection.repeated_rollouts
    ]
    output = (
        results / "reports" / fingerprint
        / f"{config.fingerprint[:16]}_level{config.collection.level}"
    )
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for row in trajectories:
        by_sequence[row["action_sequence"]].append(row)
    sequence_rows = []
    for sequence in sorted(by_sequence):
        rows = by_sequence[sequence]
        sequence_rows.append({
            "action_sequence": sequence, "trajectories": len(rows),
            "accuracy": mean(int(row["passed"]) for row in rows),
            "mean_teacher_calls": mean(int(row["teacher_calls"]) for row in rows),
            "mean_teacher_tokens": mean(int(row["teacher_total_tokens"]) for row in rows),
            "median_teacher_tokens": median(int(row["teacher_total_tokens"]) for row in rows),
        })
    _write_csv(output / "accuracy_cost_by_sequence.csv", sequence_rows)

    difficulty_by_task = {
        (row["dataset"], row["task_id"]): row.get("difficulty", "unknown")
        for row in trajectories
    }
    l1 = [row for row in effects if row.get("effect_type") == "level1"]
    conditional = [row for row in effects if row.get("effect_type") == "conditional"]
    effect_rows = []
    for effect_type, rows in (("level1", l1), ("conditional", conditional)):
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row.get("dataset"),
                     difficulty_by_task.get((row.get("dataset"), row.get("task_id")), "unknown"),
                     row.get("checkpoint"))].append(row)
        for (dataset, difficulty, checkpoint), values in sorted(grouped.items()):
            counts = Counter(row["label"] for row in values)
            effect_rows.append({
                "effect_type": effect_type, "dataset": dataset, "difficulty": difficulty,
                "checkpoint": checkpoint,
                "n": len(values), "helpful": counts["Helpful"], "harmful": counts["Harmful"],
                "no_change": counts["NoChange"],
            })
    _write_csv(output / "intervention_effects.csv", effect_rows)

    grouped_trajectories: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in trajectories:
        grouped_trajectories[(row["dataset"], row["task_id"], row["rollout_index"])][
            row["action_sequence"]
        ] = row
    multi_better = multi_worse = comparable = 0
    examples_better, examples_harmful = [], []
    for key, rows in grouped_trajectories.items():
        if not set(LEVEL2_SEQUENCES).issubset(rows):
            continue
        comparable += 1
        best_l1 = max(int(rows[name]["passed"]) for name in LEVEL1_SEQUENCES)
        best_multi = max(int(rows[name]["passed"]) for name in ("VVS", "VSV", "SVV", "VVV"))
        multi_better += best_multi > best_l1
        multi_worse += best_multi < best_l1
        if best_multi > best_l1 and len(examples_better) < 10:
            examples_better.append({"dataset": key[0], "task_id": key[1], "rollout": key[2]})
    for row in effects:
        if row.get("label") == "Harmful" and len(examples_harmful) < 10:
            examples_harmful.append({"dataset": row["dataset"], "task_id": row["task_id"],
                                     "skip": row["skip_sequence"], "verify": row["verify_sequence"]})

    interaction_counts = Counter(
        row["label"] for row in effects
        if row.get("effect_type") in {"pairwise_interaction", "three_way_interaction"}
    )
    unique_teacher_edges = [row for row in transitions if row.get("action") == "V"]
    collection_tokens = sum(int(row.get("teacher_total_tokens", 0)) for row in unique_teacher_edges)
    collection_cost = (
        sum(float(row.get("cost_usd", 0.0)) for row in roots)
        + sum(float(row.get("teacher_cost_usd", 0.0)) + float(row.get("downstream_cost_usd", 0.0))
              for row in transitions)
    )
    l1_complete = sum(set(LEVEL1_SEQUENCES).issubset(rows) for rows in grouped_trajectories.values())
    l2_complete = sum(set(LEVEL2_SEQUENCES).issubset(rows) for rows in grouped_trajectories.values())
    summary = {
        "schema_version": "cbv-report-v2", "valid_full_dataset": validation["valid"],
        "model_pair_id": validation["model_pair_id"],
        "model_pair_fingerprint": validation["model_pair_fingerprint"],
        "student_model": validation["student_model"],
        "teacher_model": validation["teacher_model"],
        "prompt_version": validation["prompt_version"],
        "inference_config": validation["inference_config"],
        "requested_level": config.collection.level,
        "tasks": len({(row["dataset"], row["task_id"]) for row in trajectories}),
        "trajectories": len(trajectories), "intervention_records": len(unique_teacher_edges),
        "complete_task_rollouts": validation["complete_task_rollouts"],
        "expected_task_rollouts": validation["expected_task_rollouts"],
        "completion_rate": validation["completion_rate"],
        "level1_complete_task_rollouts": l1_complete,
        "level2_complete_task_rollouts": l2_complete,
        "actual_unique_teacher_calls": len(unique_teacher_edges),
        "actual_unique_teacher_tokens": collection_tokens,
        "actual_model_collection_cost_usd": collection_cost,
        "teacher_input_tokens": sum(int(row.get("teacher_input_tokens", 0)) for row in unique_teacher_edges),
        "teacher_output_tokens": sum(
            int(row.get("teacher_output_tokens", 0)) for row in unique_teacher_edges
        ),
        "multi_vs_best_single": {"comparable": comparable, "better": multi_better, "worse": multi_worse,
                                 "equal": comparable - multi_better - multi_worse},
        "interaction_labels": dict(interaction_counts),
        "logged_errors": validation["logged_collection_errors"],
        "missing_branch_groups": len(validation["missing_branches"]),
        "examples_multi_better": examples_better,
        "examples_harmful": examples_harmful,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    _plots(figures, sequence_rows, effect_rows, unique_teacher_edges)
    report = _markdown(summary, sequence_rows, effect_rows, validation)
    report_path = output / "CBV_Bench_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(figures: Path, sequence_rows: list[dict], effect_rows: list[dict],
           teacher_edges: list[dict]) -> None:
    if sequence_rows:
        fig, axis = plt.subplots(figsize=(10, 5))
        axis.bar([row["action_sequence"] for row in sequence_rows],
                 [100 * row["accuracy"] for row in sequence_rows], color="#3B82F6")
        axis.set(ylabel="Final accuracy (%)", xlabel="Action sequence", ylim=(0, 100),
                 title="CBV-Bench final correctness by verification sequence")
        fig.tight_layout()
        fig.savefig(figures / "accuracy_by_sequence.png", dpi=180)
        plt.close(fig)
    if effect_rows:
        labels = [
            f"{row['dataset']}/{row['difficulty']} CP{row['checkpoint']} {row['effect_type']}"
            for row in effect_rows
        ]
        helpful = [row["helpful"] for row in effect_rows]
        harmful = [-row["harmful"] for row in effect_rows]
        fig, axis = plt.subplots(figsize=(12, max(5, len(labels) * 0.45)))
        axis.barh(labels, helpful, color="#16A34A", label="Helpful")
        axis.barh(labels, harmful, color="#DC2626", label="Harmful")
        axis.axvline(0, color="black", linewidth=0.8); axis.legend()
        axis.set(title="Helpful vs harmful verification effects", xlabel="Count (harmful shown negative)")
        fig.tight_layout()
        fig.savefig(figures / "helpful_harmful.png", dpi=180)
        plt.close(fig)
    tokens = [int(row.get("teacher_total_tokens", 0)) for row in teacher_edges]
    if tokens:
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.hist(tokens, bins=min(30, max(5, len(set(tokens)))), color="#7C3AED", alpha=0.85)
        axis.set(title="Teacher tokens per unique verification edge", xlabel="Input + output tokens",
                 ylabel="Unique edges")
        fig.tight_layout()
        fig.savefig(figures / "teacher_token_distribution.png", dpi=180)
        plt.close(fig)


def _markdown(summary: dict, sequence_rows: list[dict], effect_rows: list[dict],
              validation: dict) -> str:
    lines = [
        "# CBV-Bench V2 collection report", "",
        "> This report describes collected counterfactual data. It does not train or evaluate C2/C3.", "",
        "## Model condition", "",
        f"- Model pair: **{summary['model_pair_id']}**",
        f"- Student: **{summary['student_model']['model_id']}**",
        f"- Teacher: **{summary['teacher_model']['model_id']}**",
        f"- Exact condition fingerprint: `{summary['model_pair_fingerprint']}`",
        "- Prompt versions: " + ", ".join(
            f"{role}=`{version}`" for role, version in sorted(summary["prompt_version"].items())
        ),
        "",
        "## Completion and collection cost", "",
        f"- Full validation passed: **{summary['valid_full_dataset']}**",
        f"- Tasks / trajectories: **{summary['tasks']} / {summary['trajectories']}**",
        (
            f"- Complete task-rollouts: **{summary['complete_task_rollouts']}/"
            f"{summary['expected_task_rollouts']}**"
        ),
        (
            f"- Level-1 / Level-2 complete task-rollouts: "
            f"**{summary['level1_complete_task_rollouts']} / "
            f"{summary['level2_complete_task_rollouts']}**"
        ),
        (
            f"- Unique Teacher calls / tokens: **{summary['actual_unique_teacher_calls']} / "
            f"{summary['actual_unique_teacher_tokens']}**"
        ),
        f"- Recorded model collection cost: **${summary['actual_model_collection_cost_usd']:.6f}**",
        (
            f"- Logged errors / missing groups: **{summary['logged_errors']} / "
            f"{summary['missing_branch_groups']}**"
        ),
        "",
        "## Accuracy and cost by action sequence", "",
        "| Sequence | N | Accuracy | Mean Teacher calls | Mean Teacher tokens |", "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['action_sequence']} | {row['trajectories']} | {100*row['accuracy']:.2f}% | "
        f"{row['mean_teacher_calls']:.2f} | {row['mean_teacher_tokens']:.1f} |"
        for row in sequence_rows
    )
    multi = summary["multi_vs_best_single"]
    lines.extend(["", "## Multi-intervention interaction", "",
                  f"Across **{multi['comparable']}** complete task-rollouts, the best multi-intervention "
                  f"branch was better/equal/worse than the best Level-1 branch in "
                  f"**{multi['better']}/{multi['equal']}/{multi['worse']}** cases.", "",
                  "Marginal effects are computed only between sibling branches with identical preceding "
                  "history and identical remaining actions; Level-1 gains are not summed.", "",
                  "## Observed cases (not synthetic)", ""])
    if summary["examples_multi_better"]:
        lines.extend(
            f"- Multi-intervention improved over every Level-1 branch: {row['dataset']} / "
            f"{row['task_id']} / rollout {row['rollout']}"
            for row in summary["examples_multi_better"]
        )
    else:
        lines.append("- No observed multi-over-single improvement in the complete data.")
    lines.extend(["", "### Harmful intervention edges", ""])
    if summary["examples_harmful"]:
        lines.extend(
            f"- {row['dataset']} / {row['task_id']}: {row['skip']} -> {row['verify']}"
            for row in summary["examples_harmful"]
        )
    else:
        lines.append("- No harmful edge observed in the complete data.")
    lines.extend(["", "## Figures", "", "![Accuracy](figures/accuracy_by_sequence.png)", "",
                  "![Effects](figures/helpful_harmful.png)", "",
                  "![Teacher tokens](figures/teacher_token_distribution.png)", "",
                  "## Validation caveats", ""])
    if validation["errors"]:
        lines.extend(f"- {value}" for value in validation["errors"])
    else:
        lines.append("- No structural validation error.")
    if not summary["valid_full_dataset"]:
        lines.append("- This run is incomplete and must not be reported as a full CBV-Bench dataset.")
    return "\n".join(lines) + "\n"
