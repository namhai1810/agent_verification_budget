#!/usr/bin/env python3
"""Follow an append-only Phase 1 run and print a compact ETA progress line."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A writer may be between bytes while this monitor is polling.
                continue
    return rows


def process_is_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def current_software_hash() -> str:
    project_root = Path(__file__).resolve().parents[1]
    paths = list((project_root / "src").rglob("*.py"))
    paths.append(project_root / "pyproject.toml")
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def select_run(
    results_dir: Path,
    task_offset: int,
    task_count: int,
    repeated_pairs: int,
    run_id: str | None,
) -> dict[str, Any] | None:
    manifests = read_jsonl(results_dir / "manifests.jsonl")
    if run_id:
        matches = [row for row in manifests if row.get("run_id") == run_id]
    else:
        software_hash = current_software_hash()
        matches = [
            row
            for row in manifests
            if row.get("config", {}).get("experiment", {}).get("task_offset", 0) == task_offset
            and row.get("config", {}).get("experiment", {}).get("task_count") == task_count
            and row.get("config", {}).get("experiment", {}).get("repeated_pairs") == repeated_pairs
            and not row.get("mock_only", False)
            and row.get("software_sha256") == software_hash
        ]
    return matches[-1] if matches else None


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "calculating"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def expected_role_counts(task_count: int, repeated_pairs: int) -> dict[str, int]:
    return {
        "planner": task_count,
        "coder": task_count * (1 + 2 * repeated_pairs),
        "reviewer": task_count * (1 + 4 * repeated_pairs),
        "teacher": task_count * 3 * repeated_pairs,
        "local_critic": task_count * 3,
    }


def estimate_eta(
    invocations: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    expected_roles: dict[str, int],
    expected_evaluations: int,
) -> float | None:
    role_latencies: dict[str, list[float]] = defaultdict(list)
    role_done: dict[str, int] = defaultdict(int)
    for row in invocations:
        role = row.get("context", {}).get("role", "unknown")
        role_done[role] += 1
        latency = float(row.get("latency_seconds", 0.0))
        if latency > 0:
            role_latencies[role].append(latency)

    child_samples = [
        latency for role in ("planner", "coder", "reviewer") for latency in role_latencies.get(role, [])
    ]
    child_fallback = mean(child_samples) if child_samples else 8.0
    defaults = {
        "planner": child_fallback,
        "coder": child_fallback,
        "reviewer": child_fallback,
        "teacher": child_fallback * 2.5,
        "local_critic": 0.01,
    }

    eta = 0.0
    for role, expected in expected_roles.items():
        remaining = max(0, expected - role_done[role])
        average = mean(role_latencies[role]) if role_latencies[role] else defaults[role]
        eta += remaining * average

    evaluation_latencies = [
        float(row.get("latency_seconds", 0.0))
        for row in evaluations
        if float(row.get("latency_seconds", 0.0)) > 0
    ]
    evaluation_average = mean(evaluation_latencies) if evaluation_latencies else 0.25
    eta += max(0, expected_evaluations - len(evaluations)) * evaluation_average
    return eta


def snapshot(
    results_dir: Path,
    run_id: str,
    task_count: int,
    repeated_pairs: int,
    started: float,
) -> tuple[str, bool]:
    def matching(filename: str) -> list[dict[str, Any]]:
        return [
            row
            for row in read_jsonl(results_dir / filename)
            if (row.get("run_id") or row.get("context", {}).get("run_id")) == run_id
        ]

    baselines = [row for row in matching("baseline.jsonl") if row.get("status") == "completed"]
    pairs = [row for row in matching("intervention_pairs.jsonl") if row.get("status") == "completed"]
    invocations = matching("llm_invocations.jsonl")
    evaluations = matching("evaluations.jsonl")
    events = matching("events.jsonl")
    failures = sum(str(row.get("record_type", "")).endswith("_failure") for row in events)
    warnings = sum(row.get("record_type") == "consistency_warning" for row in events)

    expected_roles = expected_role_counts(task_count, repeated_pairs)
    expected_calls = sum(expected_roles.values())
    expected_pairs = task_count * 3 * repeated_pairs
    expected_evaluations = task_count * (1 + 6 * repeated_pairs)
    completed_calls = min(len(invocations), expected_calls)
    ratio = completed_calls / expected_calls if expected_calls else 1.0
    width = 28
    filled = min(width, int(width * ratio))
    bar = "█" * filled + "░" * (width - filled)
    eta = estimate_eta(invocations, evaluations, expected_roles, expected_evaluations)
    elapsed = time.monotonic() - started

    current = "waiting for first call"
    if invocations:
        context = invocations[-1].get("context", {})
        current = f"{context.get('task_id', '?')} / {context.get('role', '?')}"

    line = (
        f"[{bar}] {ratio * 100:5.1f}% | calls {len(invocations)}/{expected_calls} | "
        f"baseline {len(baselines)}/{task_count} | pairs {len(pairs)}/{expected_pairs} | "
        f"eval {len(evaluations)}/{expected_evaluations} | errors {failures} | warnings {warnings} | "
        f"elapsed {format_duration(elapsed)} | ETA ~{format_duration(eta)} | {current}"
    )
    complete = len(baselines) >= task_count and len(pairs) >= expected_pairs
    return line, complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--repeated-pairs", type=int, default=1)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    started = time.monotonic()
    manifest = select_run(
        results_dir,
        args.task_offset,
        args.task_count,
        args.repeated_pairs,
        args.run_id,
    )
    while manifest is None:
        if not process_is_alive(args.pid):
            print("Experiment exited before writing a matching manifest.", file=sys.stderr)
            return 1
        print("Waiting for run manifest...", end="\r", flush=True)
        time.sleep(args.poll_seconds)
        manifest = select_run(
            results_dir,
            args.task_offset,
            args.task_count,
            args.repeated_pairs,
            args.run_id,
        )

    run_id = manifest["run_id"]
    print(f"Monitoring run: {run_id}")
    previous_length = 0
    while True:
        line, complete = snapshot(
            results_dir,
            run_id,
            args.task_count,
            args.repeated_pairs,
            started,
        )
        padding = " " * max(0, previous_length - len(line))
        print(f"\r{line}{padding}", end="", flush=True)
        previous_length = len(line)
        if args.once or complete or not process_is_alive(args.pid):
            print()
            if not complete and not args.once:
                print("Process ended before all planned records completed; inspect the console log/events.")
            return 0 if complete or args.once else 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
