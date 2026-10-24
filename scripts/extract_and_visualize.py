#!/usr/bin/env python3
"""Flatten append-only experiment logs and build a compact visual dashboard."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


RAW_FILES = {
    "manifests": "manifests.jsonl",
    "baselines": "baseline.jsonl",
    "checkpoints": "checkpoint_states.jsonl",
    "pairs": "intervention_pairs.jsonl",
    "invocations": "llm_invocations.jsonl",
    "evaluations": "evaluations.jsonl",
    "events": "events.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def artifact_kind(artifact: dict[str, Any]) -> str:
    if "algorithm" in artifact:
        return "plan"
    if "review_summary" in artifact:
        return "review"
    if "code" in artifact:
        return "code"
    return "unknown"


def branch_group(branch: str) -> str:
    if branch == "baseline":
        return "baseline"
    if branch.endswith("_skip"):
        return "skip"
    if branch.endswith("_verify"):
        return "verify"
    return "other"


def record_run_id(row: dict[str, Any]) -> str | None:
    """Return the run id for both regular records and invocation envelopes."""
    return row.get("run_id") or row.get("context", {}).get("run_id")


def select_run(manifests: list[dict[str, Any]], requested: str | None) -> dict[str, Any]:
    if not manifests:
        raise ValueError("No manifests.jsonl records found")
    if requested:
        matches = [row for row in manifests if row.get("run_id") == requested]
        if not matches:
            raise ValueError(f"Unknown run_id: {requested}")
        return matches[-1]
    return manifests[-1]


def flatten_baselines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        risks = row.get("risk_by_checkpoint", {})
        code = row.get("final_implementation", "")
        item = {
            "run_id": row["run_id"],
            "task_id": row["task_id"],
            "status": row.get("status"),
            "rollout_seed": row.get("rollout_seed"),
            "passed": row.get("passed"),
            "evaluation_id": row.get("evaluation_id"),
            "agent_cost_usd": row.get("agent_cost_usd", 0.0),
            "agent_latency_seconds": row.get("agent_latency_seconds", 0.0),
            "final_code_lines": len(code.splitlines()),
            "final_code_chars": len(code),
            "final_implementation": code,
        }
        for checkpoint in (1, 2, 3):
            risk = risks.get(str(checkpoint), {})
            item[f"risk_cp{checkpoint}"] = risk.get("local_risk_score")
            item[f"risk_reason_cp{checkpoint}"] = risk.get("reasoning_summary")
            item[f"critic_tokens_cp{checkpoint}"] = risk.get("critic_tokens")
            item[f"critic_latency_cp{checkpoint}"] = risk.get("critic_latency_seconds")
        flattened.append(item)
    return flattened


def flatten_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        original = row.get("original_artifact", {})
        corrected = row.get("corrected_artifact", {})
        skip = row.get("skip", {})
        verify = row.get("verify", {})
        flattened.append(
            {
                "run_id": row["run_id"],
                "pair_id": row["pair_id"],
                "task_id": row["task_id"],
                "checkpoint": row["checkpoint"],
                "repetition": row["repetition"],
                "rollout_seed": row["rollout_seed"],
                "local_risk_score": row["local_risk_score"],
                "local_risk_reasoning": row.get("local_risk_reasoning", ""),
                "y_skip": row["y_skip"],
                "y_verify": row["y_verify"],
                "delta": row["delta"],
                "teacher_verdict": row.get("teacher_verdict"),
                "teacher_changed": row.get("teacher_changed"),
                "teacher_explanation": row.get("teacher_explanation"),
                "artifact_changed": compact_json(original) != compact_json(corrected),
                "teacher_cost_usd": row.get("teacher_cost_usd", 0.0),
                "teacher_latency_seconds": row.get("teacher_latency_seconds", 0.0),
                "skip_cost_usd": skip.get("cost_usd", 0.0),
                "verify_cost_usd": verify.get("cost_usd", 0.0),
                "skip_latency_seconds": skip.get("latency_seconds", 0.0),
                "verify_latency_seconds": verify.get("latency_seconds", 0.0),
                "skip_invocations": len(skip.get("invocation_ids", [])),
                "verify_invocations": len(verify.get("invocation_ids", [])),
                "skip_trajectory_states": len(skip.get("trajectory_state_ids", [])),
                "verify_trajectory_states": len(verify.get("trajectory_state_ids", [])),
                "original_artifact_json": compact_json(original),
                "corrected_artifact_json": compact_json(corrected),
                "skip_final_implementation": skip.get("final_implementation", ""),
                "verify_final_implementation": verify.get("final_implementation", ""),
            }
        )
    return flattened


def flatten_checkpoints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        artifact = row.get("current_artifact", {})
        code = artifact.get("code", "")
        model = row.get("llm_config", {})
        flattened.append(
            {
                "run_id": row["run_id"],
                "state_id": row["state_id"],
                "parent_state_id": row.get("parent_state_id"),
                "task_id": row["task_id"],
                "branch": row["branch"],
                "branch_group": branch_group(row["branch"]),
                "checkpoint": row["checkpoint_index"],
                "role": row["role"],
                "rollout_seed": row["rollout_seed"],
                "model_id": model.get("model_id"),
                "provider": model.get("provider"),
                "prompt_version": model.get("prompt_version"),
                "invocation_id": row.get("invocation_id"),
                "artifact_kind": artifact_kind(artifact),
                "artifact_chars": len(compact_json(artifact)),
                "code_lines": len(code.splitlines()) if code else 0,
                "preceding_artifact_count": len(row.get("preceding_artifacts", [])),
                "message_count": len(row.get("messages", [])),
                "tool_result_count": len(row.get("tool_results", [])),
                "artifact_json": compact_json(artifact),
            }
        )
    return flattened


def flatten_invocations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        context = row.get("context", {})
        model = row.get("llm_config", {})
        messages = row.get("messages", [])
        raw = row.get("raw_output") or ""
        flattened.append(
            {
                "run_id": context.get("run_id"),
                "invocation_id": row["invocation_id"],
                "task_id": context.get("task_id"),
                "checkpoint": context.get("checkpoint"),
                "role": context.get("role"),
                "branch": context.get("branch"),
                "branch_group": branch_group(context.get("branch", "")),
                "rollout_seed": context.get("seed"),
                "model_id": model.get("model_id"),
                "provider": model.get("provider"),
                "prompt_version": model.get("prompt_version"),
                "input_tokens": row.get("input_tokens", 0),
                "output_tokens": row.get("output_tokens", 0),
                "total_tokens": row.get("input_tokens", 0) + row.get("output_tokens", 0),
                "cost_usd": row.get("cost_usd", 0.0),
                "latency_seconds": row.get("latency_seconds", 0.0),
                "attempt_count": row.get("attempt_count", 0),
                "failed": row.get("failure") is not None,
                "failure": row.get("failure"),
                "prompt_hash": row.get("prompt_hash"),
                "message_count": len(messages),
                "prompt_chars": sum(len(message.get("content", "")) for message in messages),
                "raw_output_chars": len(raw),
            }
        )
    return flattened


def flatten_evaluations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": row["run_id"],
            "evaluation_id": row["evaluation_id"],
            "task_id": row["task_id"],
            "branch": row["branch"],
            "branch_group": branch_group(row["branch"]),
            "passed": row["passed"],
            "status": row["status"],
            "latency_seconds": row["latency_seconds"],
            "sandbox": row["sandbox"],
        }
        for row in rows
    ]


def summarize_models(invocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in invocations:
        grouped[(row["role"], row["model_id"], row["provider"])].append(row)
    summaries = []
    for (role, model_id, provider), rows in sorted(grouped.items()):
        summaries.append(
            {
                "role": role,
                "model_id": model_id,
                "provider": provider,
                "calls": len(rows),
                "input_tokens": sum(row["input_tokens"] for row in rows),
                "output_tokens": sum(row["output_tokens"] for row in rows),
                "total_tokens": sum(row["total_tokens"] for row in rows),
                "cost_usd": sum(row["cost_usd"] for row in rows),
                "mean_latency_seconds": mean(row["latency_seconds"] for row in rows),
                "failures": sum(bool(row["failed"]) for row in rows),
            }
        )
    return summaries


def build_dashboard(
    path: Path,
    run_id: str,
    mock_only: bool,
    counts: dict[str, int],
    baselines: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    invocations: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.suptitle(
        f"Phase 1 data overview — {run_id}\n{'MOCK VALIDATION' if mock_only else 'REAL MODEL RUN'}",
        fontsize=17,
        fontweight="bold",
    )

    ax = axes[0, 0]
    labels = ["baseline", "states", "pairs", "calls", "evals", "events"]
    values = [
        counts["baselines"],
        counts["checkpoints"],
        counts["pairs"],
        counts["invocations"],
        counts["evaluations"],
        counts["events"],
    ]
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.bar_label(bars)
    ax.set_title("Records in latest run")
    ax.tick_params(axis="x", rotation=25)

    role_order = ["planner", "coder", "reviewer", "local_critic", "teacher"]
    role_counts = Counter(row["role"] for row in invocations)
    ax = axes[0, 1]
    bars = ax.bar(role_order, [role_counts[role] for role in role_order], color="#72B7B2")
    ax.bar_label(bars)
    ax.set_title("Model/critic calls by role")
    ax.tick_params(axis="x", rotation=25)

    tasks = [row["task_id"] for row in baselines]
    short_tasks = [task.split("/")[-1] for task in tasks]
    risk_matrix = np.asarray(
        [[float(row.get(f"risk_cp{cp}") or 0.0) for cp in (1, 2, 3)] for row in baselines]
    )
    ax = axes[0, 2]
    image = ax.imshow(risk_matrix, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    for row_index in range(risk_matrix.shape[0]):
        for col_index in range(risk_matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{risk_matrix[row_index, col_index]:.2f}",
                ha="center",
                va="center",
            )
    ax.set(xticks=range(3), xticklabels=["CP1", "CP2", "CP3"])
    ax.set(yticks=range(len(tasks)), yticklabels=short_tasks)
    ax.set_title("Local risk by task/checkpoint")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    deltas_by_cell: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in pairs:
        deltas_by_cell[(row["task_id"], int(row["checkpoint"]))].append(float(row["delta"]))
    delta_lookup = {cell: mean(values) for cell, values in deltas_by_cell.items()}
    delta_matrix = np.asarray([[delta_lookup.get((task, cp), np.nan) for cp in (1, 2, 3)] for task in tasks])
    ax = axes[1, 0]
    image = ax.imshow(delta_matrix, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    for row_index in range(delta_matrix.shape[0]):
        for col_index in range(delta_matrix.shape[1]):
            value = delta_matrix[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                "N/A" if np.isnan(value) else f"{value:.0f}",
                ha="center",
                va="center",
            )
    ax.set(xticks=range(3), xticklabels=["CP1", "CP2", "CP3"])
    ax.set(yticks=range(len(tasks)), yticklabels=short_tasks)
    ax.set_title("Observed delta = Y_verify − Y_skip")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    input_tokens = [
        sum(row["input_tokens"] for row in invocations if row["role"] == role) for role in role_order
    ]
    output_tokens = [
        sum(row["output_tokens"] for row in invocations if row["role"] == role) for role in role_order
    ]
    ax.bar(role_order, input_tokens, label="input", color="#4C78A8")
    ax.bar(role_order, output_tokens, bottom=input_tokens, label="output", color="#F58518")
    ax.set_title("Estimated/logged tokens by role")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()

    ax = axes[1, 2]
    branch_order = ["baseline", "skip", "verify"]
    passed = [
        sum(row["branch_group"] == group and row["passed"] for row in evaluations) for group in branch_order
    ]
    failed = [
        sum(row["branch_group"] == group and not row["passed"] for row in evaluations)
        for group in branch_order
    ]
    ax.bar(branch_order, failed, label="failed", color="#E45756")
    ax.bar(branch_order, passed, bottom=failed, label="passed", color="#54A24B")
    for index, (failed_count, passed_count) in enumerate(zip(failed, passed, strict=True)):
        ax.text(index, failed_count + passed_count + 0.25, str(failed_count + passed_count), ha="center")
    ax.set_title("Sandbox evaluation outcomes")
    ax.legend()

    fig.text(
        0.5,
        0.01,
        "CP1=Planner, CP2=Coder, CP3=Reviewer. Mock outputs validate plumbing, not H1.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.94])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_markdown(
    run_id: str,
    manifest: dict[str, Any],
    counts: dict[str, int],
    baselines: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    model_summary: list[dict[str, Any]],
    all_run_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Extracted data overview",
        "",
        f"Run được visualize: `{run_id}`",
        f"Loại run: **{'MOCK VALIDATION' if manifest.get('mock_only') else 'REAL MODEL RUN'}**",
        "",
        "![Experiment dashboard](experiment_dashboard.png)",
        "",
        "## Raw data có gì?",
        "",
        "| Dataset/log | Records trong run | Nội dung chính |",
        "|---|---:|---|",
        f"| `baseline.jsonl` | {counts['baselines']} | Final code, baseline outcome, risk tại CP1/2/3 |",
        f"| `checkpoint_states.jsonl` | {counts['checkpoints']} | "
        "Artifact, messages, parent state, model config |",
        f"| `intervention_pairs.jsonl` | {counts['pairs']} | "
        "SKIP/VERIFY, delta, teacher change, trajectory refs |",
        f"| `llm_invocations.jsonl` | {counts['invocations']} | "
        "Prompt hash, token, cost, latency, attempts/failures |",
        f"| `evaluations.jsonl` | {counts['evaluations']} | Pass/fail, status, sandbox latency |",
        f"| `events.jsonl` | {counts['events']} | Preserved failures and consistency warnings |",
        "",
        "## Tasks và local risk",
        "",
        "| Task | Baseline pass | CP1 risk | CP2 risk | CP3 risk |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in baselines:
        lines.append(
            f"| `{row['task_id']}` | {int(bool(row['passed']))} | {row['risk_cp1']:.2f} | "
            f"{row['risk_cp2']:.2f} | {row['risk_cp3']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Paired interventions theo checkpoint",
            "",
            "| Checkpoint | Pairs | Helpful | Unchanged | Harmful | Mean risk | Mean delta |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in (1, 2, 3):
        selected = [row for row in pairs if int(row["checkpoint"]) == checkpoint]
        lines.append(
            f"| {checkpoint} | {len(selected)} | {sum(row['delta'] == 1 for row in selected)} | "
            f"{sum(row['delta'] == 0 for row in selected)} | "
            f"{sum(row['delta'] == -1 for row in selected)} | "
            f"{mean(row['local_risk_score'] for row in selected):.3f} | "
            f"{mean(row['delta'] for row in selected):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Model/critic usage",
            "",
            "| Role | Model | Provider | Calls | Total tokens | Cost USD | Mean latency (s) |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in model_summary:
        lines.append(
            f"| {row['role']} | `{row['model_id']}` | {row['provider']} | {row['calls']} | "
            f"{row['total_tokens']} | {row['cost_usd']:.6f} | {row['mean_latency_seconds']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Các run đang có trong append-only logs",
            "",
            "| Run ID | Type | Tasks | Pairs | Calls | Evaluations |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in all_run_rows:
        lines.append(
            f"| `{row['run_id']}` | {'mock' if row['mock_only'] else 'real'} | "
            f"{row['baselines']} | {row['pairs']} | {row['invocations']} | {row['evaluations']} |"
        )

    lines.extend(
        [
            "",
            "## File đã extract",
            "",
            "- `baseline_overview.csv`: một dòng cho mỗi task, gồm code cuối và risk từng checkpoint.",
            "- `intervention_overview.csv`: một dòng cho mỗi paired intervention, "
            "gồm artifact trước/sau và outcomes.",
            "- `checkpoint_overview.csv`: toàn bộ checkpoint state đã flatten.",
            "- `invocation_overview.csv`: token/cost/latency/model cho từng call.",
            "- `evaluation_overview.csv`: từng lần chạy sandbox.",
            "- `model_usage.csv`: aggregate theo role/model.",
            "- `all_runs_summary.csv`: inventory của mọi run trong append-only logs.",
            "- `overview_stats.json`: số liệu tổng hợp machine-readable.",
            "",
            "## Diễn giải run hiện tại",
            "",
        ]
    )
    if manifest.get("mock_only"):
        lines.append(
            "Đây là mock smoke test: coder tạo placeholder có `pass`, teacher giữ nguyên artifact. "
            "Vì vậy baseline/SKIP/VERIFY đều fail và delta bằng 0. Dataset này chứng minh pipeline, "
            "logging, restore và sandbox hoạt động; nó không phải bằng chứng khoa học về H1."
        )
    else:
        lines.append(
            "Đây là run model thật. Hãy đọc `summary.md` cùng uncertainty và sample size trước khi "
            "diễn giải H1."
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-id", help="Defaults to the last manifest record")
    parser.add_argument("--output-dir", help="Defaults to results/extracted/<run_id>")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    raw = {name: read_jsonl(results_dir / filename) for name, filename in RAW_FILES.items()}
    manifest = select_run(raw["manifests"], args.run_id)
    run_id = manifest["run_id"]
    current = {name: [row for row in rows if record_run_id(row) == run_id] for name, rows in raw.items()}
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_dir / "extracted" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    baselines = flatten_baselines(current["baselines"])
    pairs = flatten_pairs(current["pairs"])
    checkpoints = flatten_checkpoints(current["checkpoints"])
    invocations = flatten_invocations(current["invocations"])
    evaluations = flatten_evaluations(current["evaluations"])
    model_summary = summarize_models(invocations)
    counts = {name: len(rows) for name, rows in current.items()}

    baseline_fields = [
        "run_id",
        "task_id",
        "status",
        "rollout_seed",
        "passed",
        "evaluation_id",
        "agent_cost_usd",
        "agent_latency_seconds",
        "final_code_lines",
        "final_code_chars",
        "risk_cp1",
        "risk_reason_cp1",
        "critic_tokens_cp1",
        "critic_latency_cp1",
        "risk_cp2",
        "risk_reason_cp2",
        "critic_tokens_cp2",
        "critic_latency_cp2",
        "risk_cp3",
        "risk_reason_cp3",
        "critic_tokens_cp3",
        "critic_latency_cp3",
        "final_implementation",
    ]
    pair_fields = list(pairs[0]) if pairs else []
    checkpoint_fields = list(checkpoints[0]) if checkpoints else []
    invocation_fields = list(invocations[0]) if invocations else []
    evaluation_fields = list(evaluations[0]) if evaluations else []
    model_fields = list(model_summary[0]) if model_summary else []
    write_csv(output_dir / "baseline_overview.csv", baselines, baseline_fields)
    write_csv(output_dir / "intervention_overview.csv", pairs, pair_fields)
    write_csv(output_dir / "checkpoint_overview.csv", checkpoints, checkpoint_fields)
    write_csv(output_dir / "invocation_overview.csv", invocations, invocation_fields)
    write_csv(output_dir / "evaluation_overview.csv", evaluations, evaluation_fields)
    write_csv(output_dir / "model_usage.csv", model_summary, model_fields)

    all_run_rows = []
    for run_manifest in raw["manifests"]:
        candidate = run_manifest["run_id"]
        all_run_rows.append(
            {
                "run_id": candidate,
                "timestamp_utc": run_manifest.get("timestamp_utc"),
                "mock_only": run_manifest.get("mock_only", False),
                "software_sha256": run_manifest.get("software_sha256"),
                "baselines": sum(row.get("run_id") == candidate for row in raw["baselines"]),
                "pairs": sum(row.get("run_id") == candidate for row in raw["pairs"]),
                "checkpoints": sum(row.get("run_id") == candidate for row in raw["checkpoints"]),
                "invocations": sum(
                    row.get("context", {}).get("run_id") == candidate for row in raw["invocations"]
                ),
                "evaluations": sum(row.get("run_id") == candidate for row in raw["evaluations"]),
                "events": sum(row.get("run_id") == candidate for row in raw["events"]),
            }
        )
    write_csv(output_dir / "all_runs_summary.csv", all_run_rows, list(all_run_rows[0]))

    stats = {
        "run_id": run_id,
        "mock_only": manifest.get("mock_only", False),
        "task_ids": manifest.get("dataset", {}).get("task_ids", []),
        "record_counts": counts,
        "baseline_passes": sum(bool(row["passed"]) for row in baselines),
        "helpful_pairs": sum(row["delta"] == 1 for row in pairs),
        "unchanged_pairs": sum(row["delta"] == 0 for row in pairs),
        "harmful_pairs": sum(row["delta"] == -1 for row in pairs),
        "teacher_changes": sum(bool(row["teacher_changed"]) for row in pairs),
        "failure_events": sum(
            str(row.get("record_type", "")).endswith("_failure") for row in current["events"]
        ),
        "consistency_warnings": sum(
            row.get("record_type") == "consistency_warning" for row in current["events"]
        ),
        "total_tokens": sum(row["total_tokens"] for row in invocations),
        "total_cost_usd": sum(row["cost_usd"] for row in invocations),
        "model_usage": model_summary,
    }
    (output_dir / "overview_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_dashboard(
        output_dir / "experiment_dashboard.png",
        run_id,
        bool(manifest.get("mock_only")),
        counts,
        baselines,
        pairs,
        invocations,
        evaluations,
    )
    report = build_markdown(run_id, manifest, counts, baselines, pairs, model_summary, all_run_rows)
    (output_dir / "data_overview.md").write_text(report, encoding="utf-8")
    print(f"Extracted run: {run_id}")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
