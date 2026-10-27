#!/usr/bin/env python3
"""Create an auditable cross-dataset CBV-Bench status and statistics report."""

from __future__ import annotations

import argparse
import csv
import json
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
class CollectionSpec:
    key: str
    label: str
    dataset: str
    level: int
    results_dir: Path
    selection: Path


SPECS = (
    CollectionSpec(
        "humaneval_l2", "HumanEval L2", "humaneval", 2,
        Path("results/cbv_bench_v2_26102026"),
        Path("results/cbv_bench_v2_26102026/selections/humaneval_existing_l2_163.json"),
    ),
    CollectionSpec(
        "livecodebench_l2", "LiveCodeBench L2", "livecodebench", 2,
        Path("results/cbv_bench_v2_livecodebench_26102026"),
        Path("results/cbv_bench_v2_livecodebench_26102026/selections/existing_l2_175.json"),
    ),
    CollectionSpec(
        "bigcodebench_l1", "BigCodeBench L1", "bigcodebench", 1,
        Path("results/cbv_bench_v2_bigcodebench_26102026"),
        Path("results/cbv_bench_v2_bigcodebench_26102026/selections/recovery_l1_200_v2.json"),
    ),
    CollectionSpec(
        "bigcodebench_l2", "BigCodeBench L2", "bigcodebench", 2,
        Path("results/cbv_bench_v2_bigcodebench_l2_recovery_26102026"),
        Path("results/cbv_bench_v2_bigcodebench_26102026/selections/recovery_l2_25_v2.json"),
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
    validation_path = _latest(list(spec.results_dir.glob("validation/*/*/validation.json")))
    summary_path = _latest(
        list(spec.results_dir.glob(f"reports/*/*_level{spec.level}/summary.json"))
    )
    report_dir = summary_path.parent
    validation = _json(validation_path)
    summary = _json(summary_path)
    selection = _json(spec.selection)
    selected_ids = {str(row["task_id"]) for row in selection["task_records"]}
    fingerprint = str(validation["compatibility_fingerprint"])
    rows = [
        row
        for row in _jsonl(spec.results_dir / "trajectories.jsonl")
        if row.get("compatibility_fingerprint") == fingerprint
        and row.get("dataset") == spec.dataset
        and str(row.get("task_id")) in selected_ids
        and int(row.get("rollout_index", -1)) == 0
        and row.get("status") == "completed"
    ]
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

    split_counts = Counter(str(row["split"]) for row in selection["task_records"])
    expected_trajectories = len(selected_ids) * len(required)
    return {
        "spec": spec,
        "validation": validation,
        "summary": summary,
        "selection": selection,
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
        "validation_path": validation_path,
        "summary_path": summary_path,
    }


def make_figures(collections: list[dict[str, Any]], output: Path) -> None:
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
    completion = [100 * len(item["complete"]) / len(item["selected_ids"]) for item in collections]
    bars = axis.barh(short, completion, color=colors)
    axis.set(xlim=(0, 105), xlabel="Complete task-rollouts (%)", title="Collection completion")
    for bar, value, item in zip(bars, completion, collections):
        status = "VALID" if item["validation"]["valid"] else "PARTIAL"
        axis.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.1f}% {status}", va="center")

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


def markdown_report(collections: list[dict[str, Any]], totals: dict[str, Any]) -> str:
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
            "Kết luận trạng thái: **HumanEval L2, BigCodeBench L1 và BigCodeBench L2 hoàn chỉnh. LiveCodeBench L2 chưa hoàn chỉnh hoàn toàn.**",
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
            "Ba task có bốn nhánh multi-verify bị thiếu; một task thiếu cả tám nhánh. Vì vậy các thống kê Level 2 của LiveCodeBench bên dưới chỉ dùng **171 task có đủ tám sequence**.",
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
            "- Checkpoint tốt nhất không cố định: HumanEval ưu tiên CP3; LiveCodeBench và BigCodeBench L1 ưu tiên CP2; BigCodeBench L2 pilot ưu tiên CP1.",
            "- Điều này ủng hộ việc học policy theo task/checkpoint thay vì hard-code một checkpoint chung.",
            "- BigCodeBench L2 chỉ có 25 task, nên mức tăng lớn phải xem là kết quả pilot, chưa phải ước lượng ổn định.",
            "",
            "## 3. Helpful/Harmful theo checkpoint",
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
            "## 4. Multi-verification và interaction",
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
            "## 5. Khả năng tái sử dụng cho training",
            "",
            "- HumanEval L2: dùng trực tiếp 163 task đầy đủ.",
            "- LiveCodeBench: dùng 171 task cho Level 2; có thể dùng 174 task cho Level 1 nếu loader lọc theo completeness tương ứng.",
            "- BigCodeBench L1: dùng trực tiếp 200 task đầy đủ.",
            "- BigCodeBench L2: dùng trực tiếp 25 task đầy đủ.",
            "- BigCodeBench L2 là subset của BigCodeBench L1: split train/validation/test phải theo `task_id`, không được coi hai collection là task độc lập.",
            "- Khi train combined, chỉ lấy group có đủ counterfactual sequences cần thiết và giữ grouped split theo task để tránh leakage.",
            "",
            "## 6. Giới hạn khoa học",
            "",
            "- LiveCodeBench chưa phải full-valid dataset; không báo cáo 175/175 như đã hoàn tất.",
            "- So sánh accuracy trong báo cáo này là paired trên task đầy đủ, nhưng chưa kèm bootstrap confidence intervals.",
            "- Best observed sequence và multi-vs-best-single là phân tích offline, không phải policy có thể triển khai online.",
            "- Bốn prompt family khác nhau theo dataset; cross-dataset training phải giữ prompt/model metadata làm provenance.",
            "- Các task BigCodeBench bị loại do infrastructure attrition đã được audit riêng; không loại theo correctness label.",
            "",
            "## 7. Artifacts",
            "",
            "- `collection_summary.csv`: trạng thái và chi phí từng collection.",
            "- `sequence_accuracy_paired.csv`: accuracy/cost trên paired task groups.",
            "- `checkpoint_effects.csv`: helpful/harmful và gain theo checkpoint.",
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
        default=Path("results/cbv_bench_all_26102026"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    collections = [load_collection(spec) for spec in SPECS]

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
    _write_csv(args.output_dir / "collection_summary.csv", collection_rows)
    _write_csv(args.output_dir / "sequence_accuracy_paired.csv", sequence_rows)
    _write_csv(args.output_dir / "checkpoint_effects.csv", checkpoint_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                **totals,
                "collections": collection_rows,
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
    make_figures(collections, args.output_dir)
    report = markdown_report(collections, totals)
    report_path = args.output_dir / "CBV_Bench_all_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path.resolve())


if __name__ == "__main__":
    main()
