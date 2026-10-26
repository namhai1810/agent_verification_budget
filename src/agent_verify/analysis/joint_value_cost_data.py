"""Leakage-safe data assembly for joint verification value and Teacher cost learning."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from agent_verify.analysis.phase2_data import (
    CHECKPOINT_ROLES,
    FORBIDDEN_FEATURE_NAMES,
    PairExample,
    build_observable_features,
    load_jsonl,
    load_public_tasks,
)
from agent_verify.analysis.phase3_training import load_livecodebench_examples

VALUE_CLASSES = ("harmful", "no_change", "helpful")
VALUE_TO_INDEX = {name: index for index, name in enumerate(VALUE_CLASSES)}


@dataclass(frozen=True)
class JointExample:
    """One pre-verification state with value and Teacher-only cost labels."""

    task_id: str
    dataset: str
    difficulty: str
    checkpoint: int
    role: str
    text: str
    metadata: tuple[float, ...]
    metadata_names: tuple[str, ...]
    y_skip: int
    y_verify: int
    delta: int
    value_label: int
    local_risk: float
    teacher_input_tokens: int
    teacher_output_tokens: int
    teacher_total_tokens: int

    @property
    def group_key(self) -> str:
        return f"{self.dataset}::{self.task_id}"


@dataclass(frozen=True)
class FrozenSplit:
    train_group_keys: frozenset[str]
    test_group_keys: frozenset[str]
    manifest_path: Path
    assignments_path: Path
    exploratory: bool = True


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def value_label(y_skip: int, y_verify: int) -> int:
    if (y_skip, y_verify) == (0, 1):
        return VALUE_TO_INDEX["helpful"]
    if (y_skip, y_verify) == (1, 0):
        return VALUE_TO_INDEX["harmful"]
    if y_skip == y_verify and y_skip in (0, 1):
        return VALUE_TO_INDEX["no_change"]
    raise ValueError(f"Invalid paired outcomes: y_skip={y_skip}, y_verify={y_verify}")


def _latest_run_id(results_dir: Path) -> str:
    manifests = load_jsonl(results_dir / "manifests.jsonl")
    if not manifests:
        raise ValueError(f"No run manifests in {results_dir}")
    return str(manifests[-1]["run_id"])


def _completed_pairs(results_dir: Path, run_id: str) -> list[dict[str, Any]]:
    return [
        row
        for row in load_jsonl(results_dir / "intervention_pairs.jsonl")
        if row.get("run_id") == run_id and row.get("status") == "completed"
    ]


def _teacher_token_labels(
    results_dir: Path,
    run_id: str,
) -> dict[tuple[str, int], tuple[int, int, int]]:
    """Join intervention rows to Teacher invocations and reject ambiguous accounting."""

    invocations = {
        str(row["invocation_id"]): row for row in load_jsonl(results_dir / "llm_invocations.jsonl")
    }
    labels: dict[tuple[str, int], tuple[int, int, int]] = {}
    for pair in _completed_pairs(results_dir, run_id):
        key = (str(pair["task_id"]), int(pair["checkpoint"]))
        if key in labels:
            raise ValueError(f"Duplicate completed intervention pair for {key} in {run_id}")
        invocation_id = str(pair.get("teacher_invocation_id", ""))
        invocation = invocations.get(invocation_id)
        if invocation is None:
            raise ValueError(f"Missing Teacher invocation {invocation_id!r} for {key}")
        input_tokens = invocation.get("input_tokens")
        output_tokens = invocation.get("output_tokens")
        if input_tokens is None or output_tokens is None:
            raise ValueError(f"Missing Teacher token counts for {key}")
        input_tokens = int(input_tokens)
        output_tokens = int(output_tokens)
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError(f"Negative Teacher token count for {key}")
        total_tokens = input_tokens + output_tokens
        labels[key] = (input_tokens, output_tokens, total_tokens)
    return labels


def _augment_example(
    pair: PairExample,
    *,
    dataset: str,
    difficulty: str,
    token_labels: tuple[int, int, int],
) -> JointExample:
    dataset = dataset.lower()
    difficulty = difficulty.lower() or "unspecified"
    domain_names = (
        "dataset_humaneval",
        "dataset_livecodebench",
        "difficulty_easy",
        "difficulty_medium",
        "difficulty_hard",
        "difficulty_unspecified",
    )
    domain_values = (
        float(dataset == "humaneval"),
        float(dataset == "livecodebench"),
        float(difficulty == "easy"),
        float(difficulty == "medium"),
        float(difficulty == "hard"),
        float(difficulty not in {"easy", "medium", "hard"}),
    )
    names = (*pair.metadata_names, *domain_names)
    if FORBIDDEN_FEATURE_NAMES.intersection(names):
        raise AssertionError("Post-treatment field entered Joint Value-Cost features")
    text = "\n".join(
        [
            f"[DATASET]\n{dataset}",
            f"[DIFFICULTY]\n{difficulty}",
            pair.text,
        ]
    )
    if pair.task_id in text:
        raise AssertionError(f"task_id leaked into model text for {pair.task_id}")
    input_tokens, output_tokens, total_tokens = token_labels
    if total_tokens != pair.teacher_tokens:
        raise ValueError(
            f"Teacher token mismatch for {pair.task_id} CP{pair.checkpoint}: "
            f"invocation={total_tokens}, PairExample={pair.teacher_tokens}"
        )
    return JointExample(
        task_id=pair.task_id,
        dataset=dataset,
        difficulty=difficulty,
        checkpoint=pair.checkpoint,
        role=pair.role,
        text=text,
        metadata=(*pair.metadata, *domain_values),
        metadata_names=names,
        y_skip=pair.y_skip,
        y_verify=pair.y_verify,
        delta=pair.delta,
        value_label=value_label(pair.y_skip, pair.y_verify),
        local_risk=pair.local_risk,
        teacher_input_tokens=input_tokens,
        teacher_output_tokens=output_tokens,
        teacher_total_tokens=total_tokens,
    )


def _load_humaneval(config: dict[str, Any]) -> list[JointExample]:
    source = config["data"]["humaneval"]
    results_dir = Path(source["results_dir"])
    run_id = str(source["run_id"])
    public_tasks = load_public_tasks(Path(source["dataset_path"]))
    raw_pairs = _completed_pairs(results_dir, run_id)
    baselines = [
        row
        for row in load_jsonl(results_dir / "baseline.jsonl")
        if row.get("run_id") == run_id and row.get("status") == "completed"
    ]
    states = {
        (str(baseline["task_id"]), int(state["checkpoint_index"])): state
        for baseline in baselines
        for state in baseline["checkpoint_states"]
    }
    pairs: list[PairExample] = []
    feature_names: tuple[str, ...] | None = None
    invocations = {
        str(row["invocation_id"]): row
        for row in load_jsonl(results_dir / "llm_invocations.jsonl")
    }
    for raw in raw_pairs:
        task_id = str(raw["task_id"])
        checkpoint = int(raw["checkpoint"])
        if task_id not in public_tasks or (task_id, checkpoint) not in states:
            raise ValueError(f"Missing HumanEval task/state for {task_id} CP{checkpoint}")
        state = states[(task_id, checkpoint)]
        if state["state_id"] != raw["original_checkpoint_state_id"]:
            raise ValueError(f"State mismatch for {task_id} CP{checkpoint}")
        current_artifact = dict(raw["original_artifact"])
        text, metadata, names = build_observable_features(
            prompt=public_tasks[task_id]["prompt"],
            entry_point=public_tasks[task_id]["entry_point"],
            checkpoint=checkpoint,
            role=CHECKPOINT_ROLES[checkpoint],
            current_artifact=current_artifact,
            preceding_artifacts=list(state.get("preceding_artifacts", [])),
            local_risk=float(raw["local_risk_score"]),
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Inconsistent HumanEval feature schema")
        invocation = invocations.get(str(raw.get("teacher_invocation_id")), {})
        input_tokens = int(invocation.get("input_tokens", 0) or 0)
        output_tokens = int(invocation.get("output_tokens", 0) or 0)
        pairs.append(
            PairExample(
                task_id=task_id,
                checkpoint=checkpoint,
                role=CHECKPOINT_ROLES[checkpoint],
                text=text,
                metadata=metadata,
                metadata_names=names,
                y_skip=int(raw["y_skip"]),
                y_verify=int(raw["y_verify"]),
                delta=int(raw["delta"]),
                local_risk=float(raw["local_risk_score"]),
                teacher_changed=bool(raw.get("teacher_changed", False)),
                teacher_tokens=input_tokens + output_tokens,
                teacher_latency_seconds=float(raw.get("teacher_latency_seconds", 0.0) or 0.0),
                teacher_cost_usd=float(raw.get("teacher_cost_usd", 0.0) or 0.0),
                original_artifact=current_artifact,
                teacher_explanation=str(raw.get("teacher_explanation", "")),
            )
        )
    costs = _teacher_token_labels(results_dir, run_id)
    return [
        _augment_example(
            pair,
            dataset="humaneval",
            difficulty="unspecified",
            token_labels=costs[(pair.task_id, pair.checkpoint)],
        )
        for pair in pairs
    ]


def source_exclusions(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Report, rather than silently hide, incomplete task trajectories."""

    exclusions: list[dict[str, Any]] = []
    sources = [config["data"]["humaneval"], *config["data"]["livecodebench"]]
    for source in sources:
        results_dir = Path(source["results_dir"])
        run_id = str(source.get("run_id") or _latest_run_id(results_dir))
        counts: dict[str, int] = {}
        for pair in _completed_pairs(results_dir, run_id):
            task_id = str(pair["task_id"])
            counts[task_id] = counts.get(task_id, 0) + 1
        for task_id, count in sorted(counts.items()):
            if count != 3:
                exclusions.append(
                    {
                        "results_dir": str(results_dir),
                        "run_id": run_id,
                        "task_id": task_id,
                        "completed_pairs": count,
                        "reason": "incomplete CP1/CP2/CP3 trajectory; excluded as a whole task",
                    }
                )
    return exclusions


def _load_livecodebench(config: dict[str, Any]) -> list[JointExample]:
    examples: list[JointExample] = []
    seen_tasks: set[str] = set()
    for source in config["data"]["livecodebench"]:
        results_dir = Path(source["results_dir"])
        run_id = str(source.get("run_id") or _latest_run_id(results_dir))
        pairs, tasks, _dropped = load_livecodebench_examples(
            results_dir,
            run_id,
            drop_incomplete_tasks=True,
        )
        overlap = seen_tasks.intersection(tasks)
        if overlap:
            raise ValueError(f"Duplicate LiveCodeBench tasks across sources: {sorted(overlap)[:5]}")
        seen_tasks.update(tasks)
        costs = _teacher_token_labels(results_dir, run_id)
        for pair in pairs:
            difficulty = str(tasks[pair.task_id].get("difficulty", "unspecified"))
            examples.append(
                _augment_example(
                    pair,
                    dataset="livecodebench",
                    difficulty=difficulty,
                    token_labels=costs[(pair.task_id, pair.checkpoint)],
                )
            )
    return examples


def load_joint_examples(config: dict[str, Any]) -> list[JointExample]:
    examples = [*_load_humaneval(config), *_load_livecodebench(config)]
    examples.sort(key=lambda row: (row.dataset, row.task_id, row.checkpoint))
    validate_examples(examples)
    return examples


def validate_examples(examples: list[JointExample]) -> None:
    if not examples:
        raise ValueError("No Joint Value-Cost examples were loaded")
    keys: set[tuple[str, str, int]] = set()
    groups: dict[str, list[JointExample]] = {}
    schema = examples[0].metadata_names
    for example in examples:
        key = (example.dataset, example.task_id, example.checkpoint)
        if key in keys:
            raise ValueError(f"Duplicate sample: {key}")
        keys.add(key)
        groups.setdefault(example.group_key, []).append(example)
        if example.metadata_names != schema:
            raise ValueError("Inconsistent feature schema")
        forbidden = FORBIDDEN_FEATURE_NAMES.intersection(example.metadata_names)
        if forbidden:
            raise ValueError(
                f"Post-treatment feature names present for {key}: {sorted(forbidden)}"
            )
        if example.teacher_total_tokens != (
            example.teacher_input_tokens + example.teacher_output_tokens
        ):
            raise ValueError(f"Invalid Teacher token accounting for {key}")
        if example.teacher_total_tokens < 0:
            raise ValueError(f"Negative Teacher cost for {key}")
        if example.delta != example.y_verify - example.y_skip:
            raise ValueError(f"Invalid Delta for {key}")
        if example.value_label != value_label(example.y_skip, example.y_verify):
            raise ValueError(f"Invalid value label for {key}")
        if example.task_id in example.text:
            raise ValueError(f"task_id present in feature text for {key}")
    incomplete = {
        group: sorted(row.checkpoint for row in rows)
        for group, rows in groups.items()
        if sorted(row.checkpoint for row in rows) != [1, 2, 3]
    }
    if incomplete:
        raise ValueError(f"Every task must have CP1/CP2/CP3; incomplete={incomplete}")


def load_frozen_split(config: dict[str, Any], examples: list[JointExample]) -> FrozenSplit:
    split_config = config["frozen_split"]
    manifest_path = Path(split_config["manifest_path"])
    assignments_path = Path(split_config["assignments_path"])
    if not manifest_path.exists() or not assignments_path.exists():
        raise FileNotFoundError(
            "The original frozen split is required; refusing to generate a replacement split"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with assignments_path.open(newline="", encoding="utf-8") as handle:
        assignments = list(csv.DictReader(handle))
    assignment_map: dict[str, str] = {}
    assignment_datasets: dict[str, str] = {}
    for row in assignments:
        task_id = str(row["task_id"])
        if task_id in assignment_map:
            raise ValueError(f"Duplicate frozen split assignment for {task_id}")
        assignment_map[task_id] = str(row["split"])
        assignment_datasets[task_id] = str(row["benchmark"])
    universe = {example.task_id for example in examples}
    if universe != set(assignment_map):
        missing = sorted(universe.difference(assignment_map))
        extra = sorted(set(assignment_map).difference(universe))
        raise ValueError(f"Frozen split universe mismatch; missing={missing[:5]}, extra={extra[:5]}")
    manifest_test = set(map(str, manifest.get("task_ids", [])))
    assignment_test = {task_id for task_id, split in assignment_map.items() if split == "test"}
    if manifest_test != assignment_test:
        raise ValueError("fixed_test_tasks.json and split_assignments.csv disagree")
    task_dataset = {example.task_id: example.dataset for example in examples}
    for task_id in universe:
        if assignment_datasets[task_id] != task_dataset[task_id]:
            raise ValueError(f"Dataset mismatch in frozen assignment for {task_id}")
    train_keys = frozenset(
        f"{task_dataset[task_id]}::{task_id}"
        for task_id, split in assignment_map.items()
        if split == "train"
    )
    test_keys = frozenset(
        f"{task_dataset[task_id]}::{task_id}"
        for task_id, split in assignment_map.items()
        if split == "test"
    )
    if train_keys.intersection(test_keys):
        raise AssertionError("Frozen train/test group overlap")
    return FrozenSplit(train_keys, test_keys, manifest_path, assignments_path)


def filter_examples(
    examples: Iterable[JointExample],
    split: FrozenSplit,
    *,
    partition: str,
    datasets: set[str] | None = None,
) -> list[JointExample]:
    if partition not in {"train", "test"}:
        raise ValueError("partition must be train or test")
    active = split.train_group_keys if partition == "train" else split.test_group_keys
    return [
        row
        for row in examples
        if row.group_key in active and (datasets is None or row.dataset in datasets)
    ]


def dataset_fingerprint(examples: Iterable[JointExample]) -> str:
    rows = [
        (
            row.dataset,
            row.task_id,
            row.checkpoint,
            row.y_skip,
            row.y_verify,
            row.teacher_input_tokens,
            row.teacher_output_tokens,
        )
        for row in examples
    ]
    return hashlib.sha256(
        json.dumps(sorted(rows), separators=(",", ":")).encode()
    ).hexdigest()


def audit_rows(examples: list[JointExample], split: FrozenSplit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition, keys in (
        ("train", split.train_group_keys),
        ("test", split.test_group_keys),
    ):
        for dataset in ("humaneval", "livecodebench", "combined"):
            subset = [
                row
                for row in examples
                if row.group_key in keys and (dataset == "combined" or row.dataset == dataset)
            ]
            if not subset:
                continue
            rows.append(
                {
                    "partition": partition,
                    "dataset": dataset,
                    "tasks": len({row.group_key for row in subset}),
                    "pairs": len(subset),
                    "helpful": sum(row.value_label == VALUE_TO_INDEX["helpful"] for row in subset),
                    "harmful": sum(row.value_label == VALUE_TO_INDEX["harmful"] for row in subset),
                    "no_change": sum(
                        row.value_label == VALUE_TO_INDEX["no_change"] for row in subset
                    ),
                    "teacher_input_tokens": sum(row.teacher_input_tokens for row in subset),
                    "teacher_output_tokens": sum(row.teacher_output_tokens for row in subset),
                    "teacher_total_tokens": sum(row.teacher_total_tokens for row in subset),
                }
            )
    return rows
