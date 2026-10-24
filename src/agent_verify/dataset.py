from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path

from agent_verify.config import ExperimentConfig
from agent_verify.logging import hash_file
from agent_verify.schemas import PrivateEvaluationCase, PublicTask


@dataclass(frozen=True)
class DatasetSelection:
    public_tasks: list[PublicTask]
    private_cases: dict[str, PrivateEvaluationCase]
    metadata: dict[str, object]


def load_humaneval(config: ExperimentConfig) -> DatasetSelection:
    dataset_path = config.dataset.path
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parents[2] / dataset_path
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        raise RuntimeError(f"Pinned HumanEval dataset is missing: {dataset_path}")
    actual_hash = hash_file(dataset_path)
    if actual_hash != config.dataset.sha256:
        raise RuntimeError(f"HumanEval SHA-256 mismatch: expected {config.dataset.sha256}, got {actual_hash}")
    problems = read_raw_humaneval(dataset_path)
    task_ids = sorted(problems, key=lambda value: int(value.split("/")[-1]))
    if config.task_selection == "shuffled":
        random.Random(config.seed).shuffle(task_ids)
    selected_ids = task_ids[config.task_offset : config.task_offset + config.task_count]

    public_tasks: list[PublicTask] = []
    private_cases: dict[str, PrivateEvaluationCase] = {}
    for task_id in selected_ids:
        row = problems[task_id]
        # canonical_solution is deliberately not copied into either object.
        public_tasks.append(PublicTask(task_id=task_id, prompt=row["prompt"], entry_point=row["entry_point"]))
        private_cases[task_id] = PrivateEvaluationCase(
            task_id=task_id,
            test=row["test"],
            entry_point=row["entry_point"],
        )

    metadata: dict[str, object] = {
        "name": "openai_humaneval",
        "upstream_repository": "https://github.com/openai/human-eval",
        "upstream_commit": config.dataset.upstream_commit,
        "data_path": str(dataset_path),
        "data_sha256": actual_hash,
        "task_ids": selected_ids,
        "selection": config.task_selection,
        "selection_seed": config.seed,
        "task_offset": config.task_offset,
    }
    return DatasetSelection(public_tasks, private_cases, metadata)


def read_raw_humaneval(path: str | Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with gzip.open(Path(path), mode="rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                records[row["task_id"]] = row
    return records
