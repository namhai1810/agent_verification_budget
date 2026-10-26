"""Leakage-safe Phase-2 data loading and feature extraction."""

from __future__ import annotations

import ast
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CHECKPOINT_ROLES = {1: "planner", 2: "coder", 3: "reviewer"}
FORBIDDEN_FEATURE_NAMES = {
    "task_id",
    "y_skip",
    "y_verify",
    "delta",
    "corrected_artifact",
    "teacher_explanation",
    "teacher_verdict",
    "final_implementation",
    "future_artifact",
}


@dataclass(frozen=True)
class PairExample:
    """One pre-verification checkpoint with offline labels kept separately."""

    task_id: str
    checkpoint: int
    role: str
    text: str
    metadata: tuple[float, ...]
    metadata_names: tuple[str, ...]
    y_skip: int
    y_verify: int
    delta: int
    local_risk: float
    teacher_changed: bool
    teacher_tokens: int
    teacher_latency_seconds: float
    teacher_cost_usd: float
    original_artifact: dict[str, Any]
    teacher_explanation: str


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


def load_public_tasks(dataset_path: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    with gzip.open(dataset_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            tasks[str(row["task_id"])] = {
                "prompt": str(row["prompt"]),
                "entry_point": str(row["entry_point"]),
            }
    return tasks


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def artifact_code(artifact: dict[str, Any]) -> str:
    return str(artifact.get("code", ""))


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 + max((_ast_depth(child) for child in children), default=0)


def static_code_features(code: str, entry_point: str) -> dict[str, float]:
    features = {
        "has_code": float(bool(code.strip())),
        "syntax_ok": 0.0,
        "syntax_error": 0.0,
        "entry_point_present": 0.0,
        "function_count": 0.0,
        "branch_count": 0.0,
        "loop_count": 0.0,
        "return_count": 0.0,
        "raise_count": 0.0,
        "pass_count": 0.0,
        "call_count": 0.0,
        "ast_depth": 0.0,
    }
    if not code.strip():
        return features
    try:
        tree = ast.parse(code)
    except SyntaxError:
        features["syntax_error"] = 1.0
        return features
    nodes = list(ast.walk(tree))
    function_names = [node.name for node in nodes if isinstance(node, ast.FunctionDef)]
    features.update(
        {
            "syntax_ok": 1.0,
            "entry_point_present": float(entry_point in function_names),
            "function_count": float(len(function_names)),
            "branch_count": float(
                sum(isinstance(node, (ast.If, ast.IfExp, ast.Match)) for node in nodes)
            ),
            "loop_count": float(sum(isinstance(node, (ast.For, ast.While)) for node in nodes)),
            "return_count": float(sum(isinstance(node, ast.Return) for node in nodes)),
            "raise_count": float(sum(isinstance(node, ast.Raise) for node in nodes)),
            "pass_count": float(sum(isinstance(node, ast.Pass) for node in nodes)),
            "call_count": float(sum(isinstance(node, ast.Call) for node in nodes)),
            "ast_depth": float(_ast_depth(tree)),
        }
    )
    return features


def build_observable_features(
    *,
    prompt: str,
    entry_point: str,
    checkpoint: int,
    role: str,
    current_artifact: dict[str, Any],
    preceding_artifacts: list[dict[str, Any]],
    local_risk: float,
) -> tuple[str, tuple[float, ...], tuple[str, ...]]:
    """Build only features available immediately before a verification decision."""

    current_json = canonical_json(current_artifact)
    previous_json = canonical_json(preceding_artifacts)
    code = artifact_code(current_artifact)
    static = static_code_features(code, entry_point)

    numeric: dict[str, float] = {
        "checkpoint": float(checkpoint),
        "role_planner": float(role == "planner"),
        "role_coder": float(role == "coder"),
        "role_reviewer": float(role == "reviewer"),
        "local_risk": float(local_risk),
        "downstream_stages_remaining": float(3 - checkpoint),
        "remaining_verification_budget": 1.0,
        "task_chars": float(len(prompt)),
        "task_lines": float(len(prompt.splitlines())),
        "current_artifact_chars": float(len(current_json)),
        "current_artifact_lines": float(len(current_json.splitlines())),
        "current_artifact_keys": float(len(current_artifact)),
        "previous_artifact_count": float(len(preceding_artifacts)),
        "previous_artifact_chars": float(len(previous_json)),
        "plan_algorithm_steps": float(len(current_artifact.get("algorithm", []))),
        "plan_edge_cases": float(len(current_artifact.get("edge_cases", []))),
        "review_issue_count": float(len(current_artifact.get("issues_found", []))),
        "review_changed": float(bool(current_artifact.get("changed", False))),
        "code_chars": float(len(code)),
        "code_lines": float(len(code.splitlines())),
        "non_ascii_chars": float(sum(ord(char) > 127 for char in current_json)),
    }
    numeric.update(static)
    if FORBIDDEN_FEATURE_NAMES.intersection(numeric):
        raise AssertionError("A forbidden post-treatment field entered numeric features")

    text = "\n".join(
        [
            "[PROBLEM]",
            prompt,
            "[ROLE]",
            role,
            "[CURRENT_ARTIFACT]",
            current_json,
            "[PREVIOUS_ARTIFACTS]",
            previous_json,
        ]
    )
    names = tuple(numeric)
    values = tuple(numeric[name] for name in names)
    return text, values, names


def load_phase1_examples(
    *,
    results_dir: Path,
    dataset_path: Path,
    run_id: str,
) -> list[PairExample]:
    public_tasks = load_public_tasks(dataset_path)
    pairs = [
        row
        for row in load_jsonl(results_dir / "intervention_pairs.jsonl")
        if row.get("run_id") == run_id and row.get("status") == "completed"
    ]
    baselines = [
        row
        for row in load_jsonl(results_dir / "baseline.jsonl")
        if row.get("run_id") == run_id and row.get("status") == "completed"
    ]
    invocations = {
        str(row["invocation_id"]): row
        for row in load_jsonl(results_dir / "llm_invocations.jsonl")
    }
    baseline_states: dict[tuple[str, int], dict[str, Any]] = {}
    for baseline in baselines:
        for state in baseline["checkpoint_states"]:
            baseline_states[(str(baseline["task_id"]), int(state["checkpoint_index"]))] = state

    if len(pairs) != 417:
        raise ValueError(f"Expected 417 completed pairs for {run_id}, found {len(pairs)}")
    task_counts: dict[str, int] = {}
    examples: list[PairExample] = []
    metadata_names: tuple[str, ...] | None = None
    for pair in pairs:
        task_id = str(pair["task_id"])
        checkpoint = int(pair["checkpoint"])
        task_counts[task_id] = task_counts.get(task_id, 0) + 1
        state = baseline_states.get((task_id, checkpoint))
        if state is None:
            raise ValueError(f"Missing baseline state for {task_id} CP{checkpoint}")
        if state["state_id"] != pair["original_checkpoint_state_id"]:
            raise ValueError(f"State mismatch for {task_id} CP{checkpoint}")
        task = public_tasks[task_id]
        current_artifact = dict(pair["original_artifact"])
        text, metadata, names = build_observable_features(
            prompt=task["prompt"],
            entry_point=task["entry_point"],
            checkpoint=checkpoint,
            role=CHECKPOINT_ROLES[checkpoint],
            current_artifact=current_artifact,
            preceding_artifacts=list(state.get("preceding_artifacts", [])),
            local_risk=float(pair["local_risk_score"]),
        )
        if task_id in text:
            raise AssertionError(f"task_id leaked into model text for {task_id}")
        if metadata_names is None:
            metadata_names = names
        elif metadata_names != names:
            raise AssertionError("Inconsistent metadata schema")

        invocation = invocations.get(str(pair.get("teacher_invocation_id")), {})
        teacher_tokens = int(invocation.get("input_tokens", 0) or 0) + int(
            invocation.get("output_tokens", 0) or 0
        )
        examples.append(
            PairExample(
                task_id=task_id,
                checkpoint=checkpoint,
                role=CHECKPOINT_ROLES[checkpoint],
                text=text,
                metadata=metadata,
                metadata_names=names,
                y_skip=int(pair["y_skip"]),
                y_verify=int(pair["y_verify"]),
                delta=int(pair["delta"]),
                local_risk=float(pair["local_risk_score"]),
                teacher_changed=bool(pair.get("teacher_changed", False)),
                teacher_tokens=teacher_tokens,
                teacher_latency_seconds=float(pair.get("teacher_latency_seconds", 0.0) or 0.0),
                teacher_cost_usd=float(pair.get("teacher_cost_usd", 0.0) or 0.0),
                original_artifact=current_artifact,
                teacher_explanation=str(pair.get("teacher_explanation", "")),
            )
        )

    if len(task_counts) != 139 or set(task_counts.values()) != {3}:
        raise ValueError(f"Expected 139 complete task triples, got counts={set(task_counts.values())}")
    examples.sort(key=lambda row: (int(row.task_id.split("/")[-1]), row.checkpoint))
    return examples


def feature_matrix(examples: list[PairExample]) -> np.ndarray:
    return np.asarray([example.metadata for example in examples], dtype=float)


def assert_group_isolation(
    examples: list[PairExample], train_indices: np.ndarray, test_indices: np.ndarray
) -> None:
    train_tasks = {examples[int(index)].task_id for index in train_indices}
    test_tasks = {examples[int(index)].task_id for index in test_indices}
    overlap = train_tasks.intersection(test_tasks)
    if overlap:
        raise AssertionError(f"Task leakage across folds: {sorted(overlap)}")
