#!/usr/bin/env python3
"""Evaluate complete code_lines from a finished thinking run without new LLM calls."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from agent_verify.analysis.execution_repair_pilot import (
    _evaluators,
    _load_private_cases,
    load_pilot_config,
    project_path,
    salvage_code_lines_response,
)
from agent_verify.cbv_bench.config import load_cbv_config
from agent_verify.evaluation.bigcodebench_runner import BigCodeBenchEvaluatorError
from agent_verify.evaluation.humaneval_runner import SandboxUnavailable
from agent_verify.logging import JsonlStore


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latest_attempt(invocation: dict[str, Any]) -> dict[str, Any]:
    attempts = invocation.get("attempts") or []
    return attempts[-1] if attempts else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/darc_qwen38_variant_b_thinking_30.yaml")
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("results/darc_qwen38_variant_b_thinking_30"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/darc_qwen38_variant_b_thinking_30_salvage"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = project_path(root, args.source)
    output = project_path(root, args.output)

    selected_payload = json.loads((source / "selection/selected_tasks.json").read_text())
    selected = selected_payload.get("tasks", selected_payload)
    invocations = read_jsonl(source / "run/teacher_invocations.jsonl")
    ledger = read_jsonl(source / "run/call_ledger.jsonl")
    if len(selected) != 30 or len(invocations) != len(selected):
        raise RuntimeError("source run is not the complete locked 30-task experiment")
    terminal = {
        row["task_key"]
        for row in ledger
        if row.get("status") in {"call_completed", "call_failed"}
    }
    expected = {row["task_key"] for row in selected}
    if terminal != expected:
        raise RuntimeError("not every Teacher call is terminal; refusing to load hidden evaluators")

    failures = {
        row["task_key"]: row.get("failure_category", "")
        for row in ledger
        if row.get("status") == "call_failed"
    }
    by_task = {row["context"]["task_id"]: row for row in invocations}
    rows: list[dict[str, Any]] = []
    salvageable: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []
    for selection in selected:
        key = selection["task_key"]
        invocation = by_task[key]
        attempt = latest_attempt(invocation)
        metadata = attempt.get("response_metadata") or {}
        base = {
            **selection,
            "source_failure_category": failures.get(key, ""),
            "teacher_input_tokens": int(invocation.get("input_tokens") or 0),
            "teacher_output_tokens": int(invocation.get("output_tokens") or 0),
            "teacher_latency_seconds": float(invocation.get("latency_seconds") or 0.0),
            "reasoning_tokens": int(metadata.get("reasoning_tokens") or 0),
        }
        if failures.get(key) != "response_schema_validation":
            rows.append({**base, "salvage_status": "not_salvageable"})
            continue
        try:
            repaired = salvage_code_lines_response(str(invocation.get("raw_output") or ""))
        except ValueError as exc:
            rows.append({**base, "salvage_status": "rejected", "salvage_error": str(exc)})
            continue
        salvageable.append((selection, repaired, base))

    config = load_pilot_config(project_path(root, args.config))
    cbv = load_cbv_config(project_path(root, config.runtime.cbv_config))
    # The private cases are loaded only after the terminal-call barrier above.
    private_cases = _load_private_cases(root, cbv, [item[0] for item in salvageable])
    evaluation_store = JsonlStore(output / "run/hidden_evaluations.jsonl")
    evaluators = _evaluators(root, cbv, evaluation_store)
    for selection, repaired, base in tqdm(
        salvageable, desc="Offline salvage evaluation", unit="task", dynamic_ncols=True
    ):
        key = selection["task_key"]
        try:
            result = evaluators[selection["dataset"]].evaluate(
                run_id="darc_qwen38_variant_b_thinking_30_salvage",
                case=private_cases[key],
                code=repaired.repaired_code,
                branch="VVV+execution_repair+offline_schema_salvage",
            )
            rows.append(
                {
                    **base,
                    "salvage_status": "evaluated",
                    "syntax_validated": True,
                    "repaired_code": repaired.repaired_code,
                    "repaired_passed": bool(result.passed),
                    "newly_rescued": bool(result.passed),
                    "evaluation_status": result.status,
                    "evaluation_latency_seconds": result.latency_seconds,
                }
            )
        except (BigCodeBenchEvaluatorError, SandboxUnavailable) as exc:
            rows.append(
                {
                    **base,
                    "salvage_status": "infrastructure_error",
                    "syntax_validated": True,
                    "repaired_code": repaired.repaired_code,
                    "evaluation_error": f"{type(exc).__name__}: {exc}",
                }
            )

    order = {row["task_key"]: index for index, row in enumerate(selected)}
    rows.sort(key=lambda row: order[row["task_key"]])
    write_csv(output / "tables/task_results.csv", rows)
    counts = Counter(row["salvage_status"] for row in rows)
    evaluated = [row for row in rows if row["salvage_status"] == "evaluated"]
    rescued = [row for row in evaluated if row.get("repaired_passed")]
    dataset_rows = []
    for dataset in ("humaneval", "livecodebench", "bigcodebench", "combined"):
        subset = rows if dataset == "combined" else [r for r in rows if r["dataset"] == dataset]
        dataset_rows.append(
            {
                "dataset": dataset,
                "tasks": len(subset),
                "salvageable": sum(r["source_failure_category"] == "response_schema_validation" for r in subset),
                "evaluated": sum(r["salvage_status"] == "evaluated" for r in subset),
                "rescued": sum(bool(r.get("repaired_passed")) for r in subset),
                "itt_success_rate": sum(bool(r.get("repaired_passed")) for r in subset) / len(subset),
            }
        )
    write_csv(output / "tables/dataset_summary.csv", dataset_rows)
    lines = [
        "# Variant B Thinking — Offline Schema Salvage",
        "",
        "No Teacher call was made. Only persisted complete JSON responses that contained "
        "`code_lines` but omitted `strategy`/`explanation` were reconstructed, syntax-checked, "
        "and evaluated. Truncated responses were not modified.",
        "",
        f"- Locked tasks: {len(rows)}",
        f"- Complete code-only responses: {len(salvageable)}",
        f"- Evaluated: {len(evaluated)}",
        f"- Rescued: {len(rescued)}/{len(rows)} ITT ({100*len(rescued)/len(rows):.2f}%)",
        f"- Conditional success among evaluated: {len(rescued)}/{len(evaluated) if evaluated else 0}",
        f"- Non-salvageable truncated/other responses: {counts['not_salvageable']}",
        "",
        "| Dataset | Tasks | Complete code | Evaluated | Rescued | ITT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['tasks']} | {row['salvageable']} | "
            f"{row['evaluated']} | {row['rescued']} | {100*row['itt_success_rate']:.2f}% |"
        )
    if rescued:
        lines.extend(["", "## Rescued tasks", ""])
        lines.extend(f"- `{row['task_key']}`" for row in rescued)
    report = output / "report/variant_b_thinking_salvage.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
