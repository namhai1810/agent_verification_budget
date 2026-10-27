"""Reproducible public/private dataset adapters for CBV-Bench."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_verify.cbv_bench.cache import sha256_json
from agent_verify.cbv_bench.config import CBVConfig, DatasetSourceConfig
from agent_verify.cbv_bench.schemas import TaskRecord
from agent_verify.dataset import read_raw_humaneval
from agent_verify.livecodebench_dataset import (
    DATASET_CONFIG as LCB_DATASET_CONFIG,
    DATASET_REVISION as LCB_DATASET_REVISION,
    OFFICIAL_EVALUATOR_COMMIT as LCB_EVALUATOR_COMMIT,
    dataset_content_hash,
    load_source_rows,
    row_to_private_case,
    row_to_public_task,
)
from agent_verify.logging import hash_file
from agent_verify.schemas import PrivateEvaluationCase, PublicTask, utc_now


@dataclass(frozen=True)
class BenchmarkSelection:
    spec: DatasetSourceConfig
    public_tasks: list[PublicTask]
    private_cases: dict[str, Any]
    task_records: dict[str, TaskRecord]
    source_metadata: dict[str, Any]


def project_path(path: Path | None, project_root: Path) -> Path:
    if path is None:
        raise ValueError("Required dataset path is not configured")
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _stable_split(task_id: str, config: CBVConfig, exploratory: set[str]) -> tuple[str, str]:
    if task_id in exploratory:
        return "exploratory", "task was used in development or a previous exploratory experiment"
    digest = hashlib.sha256(f"{config.selection.split_seed}:{task_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    train_edge = config.selection.train_fraction
    validation_edge = train_edge + config.selection.validation_fraction
    if fraction < train_edge:
        return "train", ""
    if fraction < validation_edge:
        return "validation", ""
    return "test", ""


def _exploratory_ids(config: CBVConfig, project_root: Path) -> set[str]:
    ids: set[str] = set()
    for configured in config.selection.exploratory_task_files:
        path = project_path(configured, project_root)
        if not path.exists():
            raise FileNotFoundError(f"Exploratory task file does not exist: {path}")
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            ids.update(str(row["task_id"]) for row in rows if row.get("task_id"))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload.get("task_ids", payload) if isinstance(payload, dict) else payload
            ids.update(str(value) for value in values)
    return ids


def _select(candidates: list[tuple[PublicTask, Any, dict[str, Any]]], spec: DatasetSourceConfig,
            seed: int, excluded: set[str]) -> list[tuple[PublicTask, Any, dict[str, Any]]]:
    eligible = [item for item in candidates if item[0].task_id not in excluded]
    if spec.minimum_release_date:
        eligible = [
            item for item in eligible
            if str(item[2].get("release_date", "")) >= spec.minimum_release_date
        ]
    if spec.maximum_release_date:
        eligible = [
            item for item in eligible
            if str(item[2].get("release_date", "")) <= spec.maximum_release_date
        ]
    if spec.difficulties:
        allowed = {value.lower() for value in spec.difficulties}
        eligible = [item for item in eligible if str(item[2].get("difficulty", "unknown")).lower() in allowed]
    eligible.sort(key=lambda item: item[0].task_id)
    random.Random(f"cbv:{seed}:{spec.name}:{spec.version}").shuffle(eligible)
    end = spec.task_offset + spec.task_count
    if end > len(eligible):
        raise ValueError(
            f"{spec.name}: requested slice [{spec.task_offset}:{end}] but only "
            f"{len(eligible)} tasks are eligible"
        )
    if not spec.difficulty_weights:
        return eligible[spec.task_offset:end]
    positive = {key.lower(): float(value) for key, value in spec.difficulty_weights.items() if value > 0}
    if not positive:
        raise ValueError("difficulty_weights must contain at least one positive weight")
    total_weight = sum(positive.values())
    exact = {key: end * value / total_weight for key, value in positive.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    for key in sorted(positive, key=lambda value: (exact[value] - quotas[value], value), reverse=True):
        if sum(quotas.values()) >= end:
            break
        quotas[key] += 1
    selected = []
    selected_ids = set()
    for difficulty, quota in sorted(quotas.items()):
        group = [item for item in eligible
                 if str(item[2].get("difficulty", "unknown")).lower() == difficulty]
        for item in group[:quota]:
            selected.append(item)
            selected_ids.add(item[0].task_id)
    if len(selected) < end:
        selected.extend(item for item in eligible if item[0].task_id not in selected_ids) 
    selected = selected[:end]
    random.Random(f"cbv-order:{seed}:{spec.name}:{spec.version}").shuffle(selected)
    return selected[spec.task_offset:end]


def _humaneval_candidates(spec: DatasetSourceConfig, project_root: Path) -> tuple[list, dict]:
    path = project_path(spec.data_path, project_root)
    actual_sha = hash_file(path)
    if spec.sha256 and actual_sha != spec.sha256:
        raise ValueError(f"HumanEval SHA-256 mismatch: expected {spec.sha256}, got {actual_sha}")
    candidates = []
    for task_id, row in read_raw_humaneval(path).items():
        public = PublicTask(task_id=task_id, prompt=row["prompt"], entry_point=row["entry_point"])
        private = PrivateEvaluationCase(task_id=task_id, test=row["test"], entry_point=row["entry_point"])
        candidates.append((public, private, {"difficulty": "unknown", "release_date": ""}))
    return candidates, {"path": str(path), "sha256": actual_sha, "upstream_version": spec.version}


def _livecodebench_candidates(spec: DatasetSourceConfig, _: Path) -> tuple[list, dict]:
    if spec.revision != LCB_DATASET_REVISION or spec.config_name != LCB_DATASET_CONFIG:
        raise ValueError(
            "LiveCodeBench config must match the repository-pinned dataset revision/config"
        )
    if spec.official_evaluator_commit and spec.official_evaluator_commit != LCB_EVALUATOR_COMMIT:
        raise ValueError("LiveCodeBench official evaluator commit does not match the pinned runner")
    rows, source_info = load_source_rows()
    candidates = []
    for row in rows:
        public = row_to_public_task(row)
        candidates.append((public, row_to_private_case(row), {
            "difficulty": public.difficulty,
            "release_date": public.contest_date,
            "platform": public.platform,
            "title": public.title,
        }))
    return candidates, {**source_info, "dataset_sha256": dataset_content_hash(rows),
                        "revision": spec.revision, "config_name": spec.config_name}


def _bigcodebench_candidates(spec: DatasetSourceConfig, _: Path) -> tuple[list, dict]:
    from agent_verify.cbv_bench.bigcodebench import load_bigcodebench_candidates

    return load_bigcodebench_candidates(spec)


_LOADERS = {"humaneval": _humaneval_candidates, "livecodebench": _livecodebench_candidates,
            "bigcodebench": _bigcodebench_candidates}


def prepare_selection_manifest(config: CBVConfig, spec: DatasetSourceConfig, project_root: Path, *,
                               mode: str, known_task_ids: set[str] | None = None,
                               force: bool = False) -> dict[str, Any]:
    """Freeze outcome-blind selection metadata; no LLM or evaluator is called."""
    manifest_path = project_path(spec.selection_manifest, project_root)
    if manifest_path.exists() and not force:
        raise FileExistsError(f"Selection manifest already exists: {manifest_path}; use --force after review")
    candidates, source_metadata = _LOADERS[spec.name](spec, project_root)
    known = known_task_ids or set()
    if mode == "new":
        excluded = known
    elif mode == "existing":
        if not known:
            raise ValueError("mode=existing requires task IDs in existing_results_dir")
        excluded = {item[0].task_id for item in candidates}.difference(known)
    else:
        excluded = set()
    selected = _select(candidates, spec, config.selection.seed, excluded)
    exploratory = _exploratory_ids(config, project_root)
    exploratory.update(known)
    records = []
    for rank, (task, _private, metadata) in enumerate(selected):
        split, reason = _stable_split(task.task_id, config, exploratory)
        records.append(TaskRecord(
            dataset=spec.name, dataset_version=spec.version, task_id=task.task_id,
            difficulty=str(metadata.get("difficulty", "unknown")),
            release_date=str(metadata.get("release_date", "")), source_metadata=metadata,
            public_task=task.model_dump(mode="json"),
            public_task_sha256=sha256_json(task.model_dump(mode="json")),
            selection_seed=config.selection.seed, selection_rank=rank, split=split,
            exploratory_reason=reason,
        ).model_dump(mode="json"))
    payload = {
        "schema_version": "cbv-selection-v2", "created_at_utc": utc_now(),
        "dataset": spec.name, "dataset_version": spec.version, "mode": mode,
        "selection_seed": config.selection.seed,
        "selection_protocol": "metadata-only; no evaluation outcome used",
        "source_metadata": source_metadata, "task_records": records,
    }
    payload["selection_sha256"] = sha256_json(payload)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_selection(config: CBVConfig, spec: DatasetSourceConfig,
                   project_root: Path) -> BenchmarkSelection:
    manifest_path = project_path(spec.selection_manifest, project_root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = payload.pop("selection_sha256")
    if sha256_json(payload) != expected_hash:
        raise ValueError(f"Selection manifest hash mismatch: {manifest_path}")
    payload["selection_sha256"] = expected_hash
    if payload["dataset"] != spec.name or payload["dataset_version"] != spec.version:
        raise ValueError("Selection manifest dataset/version does not match active config")
    candidates, source_metadata = _LOADERS[spec.name](spec, project_root)
    by_id = {task.task_id: (task, private) for task, private, _ in candidates}
    records = {row["task_id"]: TaskRecord.model_validate(row) for row in payload["task_records"]}
    missing = set(records).difference(by_id)
    if missing:
        raise ValueError(f"Pinned source no longer contains selected tasks: {sorted(missing)[:5]}")
    public_tasks, private_cases = [], {}
    for task_id, record in records.items():
        task, private = by_id[task_id]
        if sha256_json(task.model_dump(mode="json")) != record.public_task_sha256:
            raise ValueError(f"Public task content drift for {task_id}")
        public_tasks.append(task)
        private_cases[task_id] = private
    return BenchmarkSelection(spec, public_tasks, private_cases, records, source_metadata)


def discover_collected_task_ids(paths: list[Path], project_root: Path) -> set[str]:
    ids: set[str] = set()
    for configured in paths:
        path = project_path(configured, project_root)
        for name in ("trajectories.jsonl", "baseline.jsonl", "intervention_pairs.jsonl"):
            candidate = path / name
            if candidate.exists():
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    if line.strip() and json.loads(line).get("task_id"):
                        ids.add(str(json.loads(line)["task_id"]))
    return ids
