#!/usr/bin/env python3
"""Offline, task-clustered analysis of downstream-aware verification (H1).

This script only reads completed experiment artifacts. It never calls an LLM,
Teacher endpoint, evaluator, or training routine.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_RUN_ID = "h1_phase1_qwen25_gpu2-09092ebd5e-d03859f54f"
CHECKPOINT_NAMES = {1: "CP1 Planner", 2: "CP2 Coder", 3: "CP3 Reviewer"}
POLICY_ORDER = [
    "Never Verify",
    "Random Checkpoint",
    "Always Verify CP1",
    "Always Verify CP2",
    "Always Verify CP3",
    "Highest Local Error Risk",
    "Offline Oracle",
]


@dataclass(frozen=True)
class Cost:
    calls: float = 0.0
    tokens: float = 0.0
    latency: float = 0.0
    usd: float = 0.0


@dataclass
class Policy:
    name: str
    outcomes: np.ndarray
    deltas: np.ndarray
    calls: np.ndarray
    tokens: np.ndarray
    latency: np.ndarray
    usd: np.ndarray
    definition: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/h1_hypothesis_analysis"),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_261_024)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def teacher_cost(pair: dict[str, Any], invocations: dict[str, dict[str, Any]]) -> Cost:
    invocation = invocations.get(str(pair.get("teacher_invocation_id")), {})
    tokens = float(invocation.get("input_tokens", 0) or 0) + float(
        invocation.get("output_tokens", 0) or 0
    )
    return Cost(
        calls=1.0,
        tokens=tokens,
        latency=float(pair.get("teacher_latency_seconds", 0.0) or 0.0),
        usd=float(pair.get("teacher_cost_usd", 0.0) or 0.0),
    )


def validate_and_group(
    pair_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> tuple[list[str], dict[str, dict[int, dict[str, Any]]], dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in pair_rows:
        task_id = str(row["task_id"])
        checkpoint = int(row["checkpoint"])
        if checkpoint in grouped[task_id]:
            raise ValueError(
                f"Expected one repetition per task/checkpoint, found duplicate {task_id} CP{checkpoint}"
            )
        grouped[task_id][checkpoint] = row

    invalid = {task_id: sorted(rows) for task_id, rows in grouped.items() if set(rows) != {1, 2, 3}}
    if invalid:
        raise ValueError(f"Incomplete checkpoint triples: {invalid}")

    baseline_by_task = {str(row["task_id"]): row for row in baseline_rows}
    missing_baselines = sorted(set(grouped) - set(baseline_by_task))
    if missing_baselines:
        raise ValueError(f"Missing baselines: {missing_baselines}")

    for task_id, rows in grouped.items():
        skip_values = {int(row["y_skip"]) for row in rows.values()}
        if len(skip_values) != 1:
            raise ValueError(f"Y_skip differs across checkpoints for {task_id}: {skip_values}")

    task_ids = sorted(grouped, key=lambda value: int(value.split("/")[-1]))
    return task_ids, grouped, baseline_by_task


def build_task_rows(
    task_ids: list[str],
    grouped: dict[str, dict[int, dict[str, Any]]],
    invocations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        cps = grouped[task_id]
        baseline = int(cps[1]["y_skip"])
        risks = {cp: float(cps[cp]["local_risk_score"]) for cp in (1, 2, 3)}
        deltas = {cp: int(cps[cp]["delta"]) for cp in (1, 2, 3)}
        outcomes = {cp: int(cps[cp]["y_verify"]) for cp in (1, 2, 3)}
        costs = {cp: teacher_cost(cps[cp], invocations) for cp in (1, 2, 3)}

        max_risk = max(risks.values())
        risk_ties = [cp for cp in (1, 2, 3) if risks[cp] == max_risk]
        risk_cp = risk_ties[0]  # Explicit deterministic tie-break: earliest checkpoint.

        max_delta = max(deltas.values())
        oracle_candidates = [cp for cp in (1, 2, 3) if deltas[cp] == max_delta]
        oracle_cp: int | None = oracle_candidates[0] if max_delta > 0 else None
        oracle_y = outcomes[oracle_cp] if oracle_cp is not None else baseline
        oracle_delta = max_delta if max_delta > 0 else 0

        row: dict[str, Any] = {
            "task_id": task_id,
            "y_skip": baseline,
            "random_expected_y": float(np.mean(list(outcomes.values()))),
            "local_risk_selected_checkpoint": f"CP{risk_cp}",
            "local_risk_max_score": max_risk,
            "local_risk_max_tie_count": len(risk_ties),
            "local_risk_selected_y": outcomes[risk_cp],
            "local_risk_selected_delta": deltas[risk_cp],
            "oracle_selected_checkpoint": f"CP{oracle_cp}" if oracle_cp else "SKIP",
            "oracle_selected_y": oracle_y,
            "oracle_selected_delta": oracle_delta,
            "selection_diff": int(oracle_cp != risk_cp),
            "final_outcome_diff": int(outcomes[risk_cp] != oracle_y),
        }
        for cp in (1, 2, 3):
            row.update(
                {
                    f"cp{cp}_local_risk": risks[cp],
                    f"cp{cp}_y_skip": baseline,
                    f"cp{cp}_y_verify": outcomes[cp],
                    f"cp{cp}_delta": deltas[cp],
                    f"cp{cp}_teacher_changed": bool(cps[cp].get("teacher_changed", False)),
                    f"cp{cp}_teacher_calls": costs[cp].calls,
                    f"cp{cp}_teacher_tokens": costs[cp].tokens,
                    f"cp{cp}_teacher_latency_seconds": costs[cp].latency,
                    f"cp{cp}_teacher_cost_usd": costs[cp].usd,
                }
            )
        rows.append(row)
    return rows


def make_policy(
    name: str,
    outcomes: list[float],
    baseline: np.ndarray,
    costs: list[Cost],
    definition: str,
) -> Policy:
    y = np.asarray(outcomes, dtype=float)
    return Policy(
        name=name,
        outcomes=y,
        deltas=y - baseline,
        calls=np.asarray([cost.calls for cost in costs], dtype=float),
        tokens=np.asarray([cost.tokens for cost in costs], dtype=float),
        latency=np.asarray([cost.latency for cost in costs], dtype=float),
        usd=np.asarray([cost.usd for cost in costs], dtype=float),
        definition=definition,
    )


def build_policies(
    task_ids: list[str],
    grouped: dict[str, dict[int, dict[str, Any]]],
    invocations: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[Policy]]:
    baseline = np.asarray([int(grouped[task_id][1]["y_skip"]) for task_id in task_ids], dtype=float)
    zero_costs = [Cost() for _ in task_ids]
    policies = [
        make_policy(
            "Never Verify",
            baseline.tolist(),
            baseline,
            zero_costs,
            "Skip Teacher at all checkpoints.",
        )
    ]

    random_y: list[float] = []
    random_cost: list[Cost] = []
    for task_id in task_ids:
        cps = grouped[task_id]
        costs = [teacher_cost(cps[cp], invocations) for cp in (1, 2, 3)]
        random_y.append(float(np.mean([int(cps[cp]["y_verify"]) for cp in (1, 2, 3)])))
        random_cost.append(
            Cost(
                calls=1.0,
                tokens=float(np.mean([cost.tokens for cost in costs])),
                latency=float(np.mean([cost.latency for cost in costs])),
                usd=float(np.mean([cost.usd for cost in costs])),
            )
        )
    policies.append(
        make_policy(
            "Random Checkpoint",
            random_y,
            baseline,
            random_cost,
            "Uniform random choice among CP1/CP2/CP3; exact expectation, not one Monte Carlo draw.",
        )
    )

    for cp in (1, 2, 3):
        policies.append(
            make_policy(
                f"Always Verify CP{cp}",
                [int(grouped[task_id][cp]["y_verify"]) for task_id in task_ids],
                baseline,
                [teacher_cost(grouped[task_id][cp], invocations) for task_id in task_ids],
                f"Verify {CHECKPOINT_NAMES[cp]} for every task.",
            )
        )

    risk_outcomes: list[int] = []
    risk_costs: list[Cost] = []
    oracle_outcomes: list[int] = []
    oracle_costs: list[Cost] = []
    for task_id in task_ids:
        cps = grouped[task_id]
        max_risk = max(float(cps[cp]["local_risk_score"]) for cp in (1, 2, 3))
        risk_cp = next(cp for cp in (1, 2, 3) if float(cps[cp]["local_risk_score"]) == max_risk)
        risk_outcomes.append(int(cps[risk_cp]["y_verify"]))
        risk_costs.append(teacher_cost(cps[risk_cp], invocations))

        max_delta = max(int(cps[cp]["delta"]) for cp in (1, 2, 3))
        if max_delta > 0:
            oracle_cp = next(cp for cp in (1, 2, 3) if int(cps[cp]["delta"]) == max_delta)
            oracle_outcomes.append(int(cps[oracle_cp]["y_verify"]))
            oracle_costs.append(teacher_cost(cps[oracle_cp], invocations))
        else:
            oracle_outcomes.append(int(cps[1]["y_skip"]))
            oracle_costs.append(Cost())

    policies.extend(
        [
            make_policy(
                "Highest Local Error Risk",
                risk_outcomes,
                baseline,
                risk_costs,
                "Verify the maximum-risk checkpoint; ties choose CP1, then CP2, then CP3.",
            ),
            make_policy(
                "Offline Oracle",
                oracle_outcomes,
                baseline,
                oracle_costs,
                "Choose maximum observed positive Delta; ties choose earliest CP; otherwise skip.",
            ),
        ]
    )
    assert [policy.name for policy in policies] == POLICY_ORDER
    return baseline, policies


def clustered_bootstrap(
    baseline: np.ndarray,
    policies: list[Policy],
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    n_tasks = len(baseline)
    accuracy_samples = {policy.name: np.empty(repetitions) for policy in policies}
    improvement_samples = {policy.name: np.empty(repetitions) for policy in policies}
    batch_size = 500
    cursor = 0
    while cursor < repetitions:
        size = min(batch_size, repetitions - cursor)
        sampled = rng.integers(0, n_tasks, size=(size, n_tasks))
        sampled_baseline = baseline[sampled]
        for policy in policies:
            sampled_outcomes = policy.outcomes[sampled]
            accuracy_samples[policy.name][cursor : cursor + size] = sampled_outcomes.mean(axis=1)
            improvement_samples[policy.name][cursor : cursor + size] = (
                sampled_outcomes - sampled_baseline
            ).mean(axis=1)
        cursor += size
    return {
        policy.name: {
            "accuracy": percentile_ci(accuracy_samples[policy.name]),
            "improvement": percentile_ci(improvement_samples[policy.name]),
        }
        for policy in policies
    }


def correlation_bootstrap(
    task_ids: list[str],
    grouped: dict[str, dict[int, dict[str, Any]]],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    risk = np.asarray(
        [[float(grouped[task_id][cp]["local_risk_score"]) for cp in (1, 2, 3)] for task_id in task_ids]
    )
    delta = np.asarray(
        [[float(grouped[task_id][cp]["delta"]) for cp in (1, 2, 3)] for task_id in task_ids]
    )
    flat_risk = risk.ravel()
    flat_delta = delta.ravel()
    pearson = correlation(flat_risk, flat_delta)
    spearman = correlation(average_ranks(flat_risk), average_ranks(flat_delta))

    rng = np.random.default_rng(seed)
    pearson_samples = np.empty(repetitions)
    spearman_samples = np.empty(repetitions)
    for index in range(repetitions):
        sampled_tasks = rng.integers(0, len(task_ids), size=len(task_ids))
        x = risk[sampled_tasks].ravel()
        y = delta[sampled_tasks].ravel()
        pearson_samples[index] = correlation(x, y)
        spearman_samples[index] = correlation(average_ranks(x), average_ranks(y))
    return {
        "pearson": pearson,
        "pearson_ci": percentile_ci(pearson_samples),
        "spearman": spearman,
        "spearman_ci": percentile_ci(spearman_samples),
        "risk": flat_risk,
        "delta": flat_delta,
    }


def summary_rows(
    policies: list[Policy], bootstrap: dict[str, dict[str, tuple[float, float]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        accuracy_ci = bootstrap[policy.name]["accuracy"]
        improvement_ci = bootstrap[policy.name]["improvement"]
        rows.append(
            {
                "method": policy.name,
                "final_accuracy": float(policy.outcomes.mean()),
                "accuracy_ci95_low": accuracy_ci[0],
                "accuracy_ci95_high": accuracy_ci[1],
                "net_accuracy_improvement_pp": float(policy.deltas.mean() * 100),
                "improvement_ci95_low_pp": improvement_ci[0] * 100,
                "improvement_ci95_high_pp": improvement_ci[1] * 100,
                "helpful_interventions": float(np.maximum(policy.deltas, 0).sum()),
                "harmful_interventions": float(np.maximum(-policy.deltas, 0).sum()),
                "teacher_calls": float(policy.calls.sum()),
                "teacher_tokens": float(policy.tokens.sum()),
                "teacher_latency_seconds": float(policy.latency.sum()),
                "teacher_cost_usd": float(policy.usd.sum()),
                "policy_definition": policy.definition,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_risk_vs_gain(path: Path, correlation_stats: dict[str, Any], seed: int) -> None:
    risk = correlation_stats["risk"]
    delta = correlation_stats["delta"]
    rng = np.random.default_rng(seed)
    x_jitter = risk + rng.normal(0, 0.008, size=len(risk))
    y_jitter = delta + rng.normal(0, 0.035, size=len(delta))

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    colors = np.where(delta > 0, "#168A5B", np.where(delta < 0, "#D84A4A", "#9AA3AF"))
    ax.scatter(x_jitter, y_jitter, c=colors, alpha=0.48, s=32, edgecolors="none")
    for score in sorted(set(risk)):
        mask = risk == score
        mean_delta = float(delta[mask].mean())
        ax.scatter(score, mean_delta, marker="D", s=105, color="#173F5F", zorder=4)
        ax.annotate(
            f"n={int(mask.sum())}\nmean Δ={mean_delta:+.3f}",
            (score, mean_delta),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=8,
            color="#173F5F",
        )
    ax.axhline(0, color="#31363F", linewidth=1, linestyle="--")
    pearson_ci = correlation_stats["pearson_ci"]
    ax.text(
        0.99,
        0.03,
        f"Pearson r = {correlation_stats['pearson']:.3f}\n"
        f"task-clustered 95% CI [{pearson_ci[0]:.3f}, {pearson_ci[1]:.3f}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )
    ax.set(
        xlabel="Local Error Risk score",
        ylabel="Observed downstream gain Δ = Y_verify − Y_skip",
        title="Local risk is coarsely quantized and only weakly aligned with downstream gain",
        yticks=[-1, 0, 1],
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(path: Path, rows: list[dict[str, Any]]) -> None:
    names = [row["method"] for row in rows]
    accuracy = np.asarray([float(row["final_accuracy"]) * 100 for row in rows])
    low = np.asarray([float(row["accuracy_ci95_low"]) * 100 for row in rows])
    high = np.asarray([float(row["accuracy_ci95_high"]) * 100 for row in rows])
    error = np.vstack([accuracy - low, high - accuracy])
    colors = ["#6B7280", "#8B5CF6", "#57A0D3", "#3B82C4", "#1769AA", "#E07A2D", "#E0B12D"]

    fig, ax = plt.subplots(figsize=(11.2, 6.1))
    bars = ax.bar(range(len(names)), accuracy, color=colors, edgecolor="white", linewidth=0.8)
    ax.errorbar(
        range(len(names)),
        accuracy,
        yerr=error,
        fmt="none",
        ecolor="#1F2937",
        capsize=4,
        linewidth=1.2,
    )
    baseline = accuracy[0]
    ax.axhline(baseline, color="#4B5563", linestyle="--", linewidth=1.1, label="Never Verify")
    for bar, value in zip(bars, accuracy, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.45, f"{value:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(names)), [name.replace(" Verify ", "\nVerify ") for name in names], fontsize=9)
    ax.set_ylim(max(0, low.min() - 4), min(100, high.max() + 4))
    ax.set(ylabel="Final accuracy (%)", title="One-verification-per-task policy comparison")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def artifact_code(artifact: dict[str, Any]) -> str:
    if "code" in artifact:
        return str(artifact["code"]).rstrip()
    return json.dumps(artifact, ensure_ascii=False, indent=2)


def compact_code(code: str, max_lines: int = 24) -> str:
    lines = code.rstrip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + [f"# … {len(lines) - max_lines} more lines"])


def teacher_diff(pair: dict[str, Any], max_lines: int = 22) -> str:
    original = artifact_code(pair["original_artifact"]).splitlines()
    corrected = artifact_code(pair["corrected_artifact"]).splitlines()
    lines = list(
        difflib.unified_diff(original, corrected, fromfile="agent", tofile="teacher", lineterm="")
    )
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"… {len(lines) - max_lines} more diff lines"]
    return "\n".join(lines) if lines else "(Teacher giữ nguyên artifact.)"


def case_studies(grouped: dict[str, dict[int, dict[str, Any]]]) -> str:
    specs = [
        (
            "Local Risk cao nhưng Verify không giúp",
            "HumanEval/125",
            2,
            "Coder tạo lỗi cú pháp `</code>`. Teacher xóa phần thừa, nhưng Reviewer downstream lại tạo "
            "`txt.split(',').split()`, nên cả hai nhánh đều fail. Sửa cục bộ đúng nhưng không truyền thành "
            "cải thiện ở output cuối.",
        ),
        (
            "Local Risk thấp nhưng Verify cải thiện",
            "HumanEval/17",
            2,
            "Coder ném `ValueError` khi gặp dấu cách, dù input mẫu dùng dấu cách giữa các nốt. Teacher đổi "
            "nhánh này thành bỏ qua ký tự phân cách; thay đổi sống sót qua Reviewer và biến fail thành pass.",
        ),
        (
            "Teacher verification gây hại",
            "HumanEval/116",
            2,
            "Teacher hiểu câu mô tả “non-negative” là phải từ chối số âm và thêm `ValueError`, nhưng chính "
            "đề bài có ví dụ số âm. Artifact gốc pass; sửa đổi của Teacher làm nhánh VERIFY fail.",
        ),
        (
            "Downstream agent tự sửa lỗi không cần Teacher",
            "HumanEval/141",
            2,
            "Coder có lỗi cú pháp ở `split('.一起`. Teacher sửa đúng, nhưng ở nhánh SKIP Reviewer cũng độc lập "
            "sửa thành `split('.')`. Vì cả hai nhánh đều pass, giá trị downstream của lần Verify này bằng 0.",
        ),
        (
            "Risk-based selection khác Offline Oracle",
            "HumanEval/77",
            3,
            "Risk policy chọn CP1 (risk 0.20), nơi Verify không đổi kết quả. Oracle chọn CP3 (risk 0.15): "
            "Teacher sửa xử lý căn bậc ba cho số âm, đúng với ví dụ `iscube(-1)`, và biến fail thành pass.",
        ),
    ]
    sections: list[str] = []
    for index, (title, task_id, checkpoint, explanation) in enumerate(specs, start=1):
        pair = grouped[task_id][checkpoint]
        sections.extend(
            [
                f"### {index}. {title} — `{task_id}`, CP{checkpoint}",
                "",
                f"- Local Risk: **{float(pair['local_risk_score']):.2f}**; "
                f"`Y_skip={pair['y_skip']}`, `Y_verify={pair['y_verify']}`, `Delta={int(pair['delta']):+d}`.",
                f"- Teacher verdict: `{pair.get('teacher_verdict')}`; "
                f"artifact changed: `{bool(pair.get('teacher_changed'))}`.",
                f"- Diễn giải: {explanation}",
                "",
                "Output gốc của Agent:",
                "",
                "```python",
                compact_code(artifact_code(pair["original_artifact"])),
                "```",
                "",
                "Thay đổi của Teacher:",
                "",
                "```diff",
                teacher_diff(pair),
                "```",
                "",
            ]
        )
    return "\n".join(sections)


def fmt_ci(low: float, high: float, scale: float = 100.0) -> str:
    return f"[{low * scale:.2f}, {high * scale:.2f}]"


def build_report(
    run_id: str,
    task_rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    correlation_stats: dict[str, Any],
    grouped: dict[str, dict[int, dict[str, Any]]],
    bootstrap_reps: int,
    seed: int,
) -> str:
    n_tasks = len(task_rows)
    n_pairs = n_tasks * 3
    by_method = {row["method"]: row for row in summary}
    disagreements = sum(int(row["selection_diff"]) for row in task_rows)
    outcome_differences = sum(int(row["final_outcome_diff"]) for row in task_rows)

    any_risk_tie = sum(
        len({float(grouped[task_id][cp]["local_risk_score"]) for cp in (1, 2, 3)}) < 3
        for task_id in grouped
    )
    max_risk_tie = sum(int(row["local_risk_max_tie_count"]) > 1 for row in task_rows)
    max_delta_tie = sum(
        len(
            [
                cp
                for cp in (1, 2, 3)
                if int(grouped[task_id][cp]["delta"])
                == max(int(grouped[task_id][other]["delta"]) for other in (1, 2, 3))
            ]
        )
        > 1
        for task_id in grouped
    )
    legacy_union_tie = sum(
        (
            int(row["local_risk_max_tie_count"]) > 1
            or len(
                [
                    cp
                    for cp in (1, 2, 3)
                    if int(grouped[row["task_id"]][cp]["delta"])
                    == max(int(grouped[row["task_id"]][other]["delta"]) for other in (1, 2, 3))
                ]
            )
            > 1
        )
        for row in task_rows
    )

    risk_patterns = Counter(
        tuple(float(grouped[task_id][cp]["local_risk_score"]) for cp in (1, 2, 3))
        for task_id in grouped
    )
    earliest = by_method["Highest Local Error Risk"]
    latest_outcomes = []
    uniform_outcomes = []
    for task_id in sorted(grouped):
        cps = grouped[task_id]
        max_risk = max(float(cps[cp]["local_risk_score"]) for cp in (1, 2, 3))
        tied = [cp for cp in (1, 2, 3) if float(cps[cp]["local_risk_score"]) == max_risk]
        latest_outcomes.append(int(cps[tied[-1]]["y_verify"]))
        uniform_outcomes.append(float(np.mean([int(cps[cp]["y_verify"]) for cp in tied])))

    table_lines = [
        "| Method | Accuracy (95% CI) | Net Δ accuracy (pp, 95% CI) | Helpful | Harmful | Teacher calls | Tokens | Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        table_lines.append(
            f"| {row['method']} | {float(row['final_accuracy']) * 100:.2f}% "
            f"{fmt_ci(float(row['accuracy_ci95_low']), float(row['accuracy_ci95_high']))} | "
            f"{float(row['net_accuracy_improvement_pp']):+.2f} "
            f"[{float(row['improvement_ci95_low_pp']):+.2f}, {float(row['improvement_ci95_high_pp']):+.2f}] | "
            f"{float(row['helpful_interventions']):.2f} | {float(row['harmful_interventions']):.2f} | "
            f"{float(row['teacher_calls']):.1f} | {float(row['teacher_tokens']):,.0f} | "
            f"{float(row['teacher_latency_seconds']):,.1f}s |"
        )

    risk_row = by_method["Highest Local Error Risk"]
    oracle_row = by_method["Offline Oracle"]
    pearson_ci = correlation_stats["pearson_ci"]
    spearman_ci = correlation_stats["spearman_ci"]
    pattern_text = ", ".join(f"`{pattern}`: {count}" for pattern, count in risk_patterns.most_common())

    return "\n".join(
        [
            "# H1 Analysis — Downstream-Aware Verification",
            "",
            f"**Run:** `{run_id}`  ",
            f"**Sample:** {n_tasks} valid HumanEval tasks, {n_pairs} completed Verify/Skip pairs  ",
            "**Analysis mode:** offline only; no LLM inference, Teacher call, evaluation rerun, or training  ",
            f"**Uncertainty:** {bootstrap_reps:,} task-clustered bootstrap resamples, seed `{seed}`",
            "",
            "## Executive conclusion",
            "",
            f"Kết quả **ủng hộ H1 ở mức bằng chứng mô tả**, nhưng chưa chứng minh rằng Downstream Value có thể "
            f"được dự đoán khi triển khai. Highest Local Risk đạt **{float(risk_row['final_accuracy']) * 100:.2f}%**, "
            f"trong khi Never Verify đạt **{float(by_method['Never Verify']['final_accuracy']) * 100:.2f}%**. "
            f"Offline Oracle đạt **{float(oracle_row['final_accuracy']) * 100:.2f}%**, nhưng oracle nhìn thấy kết quả "
            "Verify/Skip sau thực nghiệm nên chỉ là upper bound phân tích, không phải policy deployable.",
            "",
            f"Local Risk và downstream gain chỉ tương quan yếu: Pearson `r={correlation_stats['pearson']:.3f}` "
            f"(task-clustered 95% CI `{fmt_ci(pearson_ci[0], pearson_ci[1], 1.0)}`); Spearman "
            f"`rho={correlation_stats['spearman']:.3f}` (95% CI "
            f"`{fmt_ci(spearman_ci[0], spearman_ci[1], 1.0)}`). Risk-based và oracle chọn khác nhau "
            f"**{disagreements}/{n_tasks}** task; trong đó final correctness thực sự khác ở "
            f"**{outcome_differences}/{n_tasks}** task.",
            "",
            "## Definitions and selection rules",
            "",
            "- `Y_skip`: final binary correctness when the checkpoint is not verified.",
            "- `Y_verify`: final binary correctness after Teacher verifies that checkpoint and the remaining agents run.",
            "- `Delta = Y_verify - Y_skip`: `+1` helpful, `0` no final effect, `-1` harmful.",
            "- **Highest Local Error Risk:** choose the maximum score; exact ties use deterministic order "
            "CP1 → CP2 → CP3.",
            "- **Offline Oracle:** choose the checkpoint with maximum positive observed Delta; positive ties use "
            "CP1 → CP2 → CP3; skip all if every Delta is non-positive.",
            "- **Random Checkpoint:** the table gives the exact expectation under a uniform 1/3 choice, not one "
            "random draw. Therefore helpful/harmful counts can be fractional.",
            "",
            "## Direct answers to H1 questions",
            "",
            f"1. **Selection disagreement:** {disagreements}/{n_tasks} tasks. Oracle skips whenever no checkpoint "
            "has positive observed gain, whereas the risk policy always spends one verification.",
            f"2. **Different final correctness:** {outcome_differences}/{n_tasks} tasks. A different action/checkpoint "
            "usually has no binary outcome effect because most Deltas are zero.",
            f"3. **Highest Local Risk accuracy:** {float(risk_row['final_accuracy']) * 100:.2f}% "
            f"(95% CI {fmt_ci(float(risk_row['accuracy_ci95_low']), float(risk_row['accuracy_ci95_high']))}).",
            f"4. **Offline Oracle accuracy:** {float(oracle_row['final_accuracy']) * 100:.2f}% "
            f"(95% CI {fmt_ci(float(oracle_row['accuracy_ci95_low']), float(oracle_row['accuracy_ci95_high']))}); "
            f"net gain {float(oracle_row['net_accuracy_improvement_pp']):+.2f} pp over Never Verify.",
            f"5. **Correlation:** Pearson `r={correlation_stats['pearson']:.3f}` and Spearman "
            f"`rho={correlation_stats['spearman']:.3f}`; both are weak and their clustered intervals should be "
            "used instead of treating 417 pairs as independent.",
            "6. **Counterexamples exist:** the case studies show both high-risk/no-benefit and "
            "low-risk/helpful pairs, plus a harmful intervention.",
            "",
            "## Baseline and policy comparison",
            "",
            *table_lines,
            "",
            "All USD costs are `$0.00` because both models were served locally. Token and latency totals quantify "
            "compute burden; latency is the sum of per-call wall times, not end-to-end parallel runtime.",
            "",
            "![Final accuracy by method](final_accuracy_by_method.png)",
            "",
            "## Audit of the 134/139 tie statement",
            "",
            f"The earlier **134/139** value is reproducible, but it is **not “134 tasks with risk-ranking "
            f"ties.”** The legacy analyzer increments its tie counter when either the maximum-risk set **or** the "
            f"maximum-Delta set has multiple checkpoints. That union is {legacy_union_tie}/{n_tasks}; it is driven "
            "mainly by many zero-Delta ties.",
            "",
            f"- Any duplicated risk score anywhere in the three-checkpoint ranking: **{any_risk_tie}/{n_tasks}**.",
            f"- Tie specifically at the **maximum risk**, which can alter Method A: **{max_risk_tie}/{n_tasks}**.",
            f"- Tie at maximum observed Delta: **{max_delta_tie}/{n_tasks}**.",
            f"- Observed risk patterns: {pattern_text}.",
            "",
            f"Under maximum-risk ties, earliest tie-breaking gives "
            f"**{float(earliest['final_accuracy']) * 100:.2f}%** accuracy; latest gives "
            f"**{np.mean(latest_outcomes) * 100:.2f}%**; uniform random among tied maxima has expected "
            f"accuracy **{np.mean(uniform_outcomes) * 100:.2f}%**. Thus tie-breaking can shift the result by "
            f"up to {(np.mean(latest_outcomes) - float(earliest['final_accuracy'])) * 100:.2f} pp here, but the "
            "larger problem is the critic's coarse score grid and limited discrimination—not 134 ambiguous "
            "maximum-risk choices.",
            "",
            "## Local Risk versus Downstream Gain",
            "",
            "![Local Risk versus downstream gain](local_risk_vs_downstream_gain.png)",
            "",
            "The risk score takes only four observed values (`0.15`, `0.20`, `0.35`, `0.99`) and is generated "
            "by the deterministic local heuristic critic; it was **not trained**. Therefore this analysis evaluates "
            "that heuristic's ranking signal, not a learned critic model.",
            "",
            "## Real case studies",
            "",
            case_studies(grouped),
            "## Scientific interpretation and recommendation",
            "",
            "1. **Does the evidence support H1?** Yes, provisionally: local risk has weak association with observed "
            "downstream gain, produces concrete ranking misses, and underperforms several simple checkpoint policies. "
            "The claim is support, not proof, and is specific to this workflow/model/dataset run.",
            "2. **Is Local Error Risk sufficient for checkpoint selection?** No for this experiment. Its scores are "
            "coarse, it does not encode downstream self-correction, and it cannot distinguish helpful from harmful "
            "Teacher edits reliably.",
            "3. **Is there evidence motivating a Downstream Verification Value predictor?** Yes: the gap between "
            "risk policy and oracle shows exploitable headroom, while the case studies identify learnable-looking "
            "signals such as downstream recoverability and intervention harm. However, oracle headroom alone does "
            "not establish predictability on unseen data.",
            "4. **Proceed to Phase 2?** Proceed only as a staged feasibility study: first define leakage-safe train/" 
            "validation/test splits and test whether an offline predictor beats simple baselines (especially Always "
            "CP3) on held-out tasks. Do not claim Phase 2 success or deploy a policy until that test passes.",
            "",
            "## Reproducibility and limitations",
            "",
            "- Bootstrap resampling unit is the task, preserving the dependence among its three checkpoints.",
            "- These are paired observational intervention outcomes under fixed seeds; provider kernels may still be "
            "nondeterministic, and each held-out pair has one repetition.",
            "- HumanEval/38 is absent because generation failed; conclusions apply to the 139 complete tasks.",
            "- Accuracy is binary unit-test correctness; it does not measure partial quality or security.",
            "- The oracle uses post-outcome information and is an analytical upper bound only.",
            "",
            "## Generated artifacts",
            "",
            "- `H1_task_comparison.csv`: one row per task with all CP risks, outcomes, Deltas, costs, and choices.",
            "- `H1_summary.csv`: policy metrics and task-clustered 95% confidence intervals.",
            "- `local_risk_vs_downstream_gain.png`: pair-level risk/gain structure with clustered correlation.",
            "- `final_accuracy_by_method.png`: policy accuracy comparison with clustered confidence intervals.",
            "- `analysis_manifest.json`: run ID, sample counts, tie audit, bootstrap settings, and output inventory.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 1:
        raise ValueError("--bootstrap-reps must be positive")

    pair_rows = [
        row
        for row in load_jsonl(args.results_dir / "intervention_pairs.jsonl")
        if row.get("run_id") == args.run_id and row.get("status") == "completed"
    ]
    baseline_rows = [
        row
        for row in load_jsonl(args.results_dir / "baseline.jsonl")
        if row.get("run_id") == args.run_id and row.get("status") == "completed"
    ]
    invocations = {
        str(row["invocation_id"]): row for row in load_jsonl(args.results_dir / "llm_invocations.jsonl")
    }
    if not pair_rows:
        raise ValueError(f"No completed intervention pairs for run {args.run_id}")

    task_ids, grouped, _ = validate_and_group(pair_rows, baseline_rows)
    task_rows = build_task_rows(task_ids, grouped, invocations)
    baseline, policies = build_policies(task_ids, grouped, invocations)
    bootstrap = clustered_bootstrap(baseline, policies, args.bootstrap_reps, args.seed)
    correlation_stats = correlation_bootstrap(
        task_ids, grouped, args.bootstrap_reps, args.seed + 1
    )
    summaries = summary_rows(policies, bootstrap)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "H1_task_comparison.csv", task_rows)
    write_csv(args.output_dir / "H1_summary.csv", summaries)
    plot_risk_vs_gain(
        args.output_dir / "local_risk_vs_downstream_gain.png",
        correlation_stats,
        args.seed + 2,
    )
    plot_accuracy(args.output_dir / "final_accuracy_by_method.png", summaries)
    report = build_report(
        args.run_id,
        task_rows,
        summaries,
        correlation_stats,
        grouped,
        args.bootstrap_reps,
        args.seed,
    )
    (args.output_dir / "H1_analysis.md").write_text(report, encoding="utf-8")

    max_risk_ties = sum(int(row["local_risk_max_tie_count"]) > 1 for row in task_rows)
    manifest = {
        "analysis_type": "offline_h1",
        "run_id": args.run_id,
        "tasks": len(task_ids),
        "pairs": len(pair_rows),
        "bootstrap": {
            "unit": "task",
            "repetitions": args.bootstrap_reps,
            "seed": args.seed,
            "interval": "percentile_95",
        },
        "tie_break": "earliest checkpoint: CP1, CP2, CP3",
        "maximum_risk_ties": max_risk_ties,
        "selection_disagreements": sum(int(row["selection_diff"]) for row in task_rows),
        "final_outcome_differences": sum(int(row["final_outcome_diff"]) for row in task_rows),
        "outputs": [
            "H1_analysis.md",
            "H1_task_comparison.csv",
            "H1_summary.csv",
            "local_risk_vs_downstream_gain.png",
            "final_accuracy_by_method.png",
        ],
        "inference_or_training_performed": False,
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote offline H1 analysis to: {args.output_dir.resolve()}")
    for row in summaries:
        print(
            f"  {row['method']:<26} accuracy={float(row['final_accuracy']) * 100:6.2f}% "
            f"net={float(row['net_accuracy_improvement_pp']):+6.2f} pp"
        )


if __name__ == "__main__":
    main()
