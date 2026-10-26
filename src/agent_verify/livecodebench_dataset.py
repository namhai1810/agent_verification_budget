"""Pinned LiveCodeBench v6 loading, deterministic selection, and privacy separation."""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
import random
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_verify.dataset import DatasetSelection
from agent_verify.logging import canonical_json
from agent_verify.schemas import LiveCodeBenchPrivateCase, LiveCodeBenchPublicTask, utc_now

DATASET_ID = "livecodebench/code_generation_lite"
DATASET_CONFIG = "v6"
DATASET_SPLIT = "test"
DATASET_REVISION = "175d0ac20500fd7aa0813f7a2e6ae7ecad3c76c9"
OFFICIAL_REPOSITORY = "https://github.com/LiveCodeBench/LiveCodeBench"
OFFICIAL_EVALUATOR_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
MIN_CONTEST_DATE = "2025-01-01"
DEFAULT_TASK_COUNT = 100
DEFAULT_SELECTION_SEED = 42

EXPECTED_COLUMNS = {
    "question_title",
    "question_content",
    "platform",
    "question_id",
    "contest_id",
    "contest_date",
    "starter_code",
    "difficulty",
    "public_test_cases",
    "private_test_cases",
    "metadata",
}


class LiveCodeBenchDatasetError(RuntimeError):
    """Raised when the pinned dataset cannot satisfy the frozen protocol."""


def _json_value(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    if isinstance(value, str):
        return json.loads(value)
    return value


def decode_private_tests(value: Any) -> list[dict[str, Any]]:
    """Decode both plain JSON and the compressed encoding used by LiveCodeBench."""

    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise LiveCodeBenchDatasetError("private_test_cases must be a JSON/compressed string")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        try:
            unpacked = pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8"))))
            decoded = json.loads(unpacked) if isinstance(unpacked, str) else unpacked
        except Exception as exc:
            raise LiveCodeBenchDatasetError("Could not decode private_test_cases") from exc
    if not isinstance(decoded, list):
        raise LiveCodeBenchDatasetError("Decoded private_test_cases is not a list")
    return decoded


def _date(value: Any) -> datetime:
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise LiveCodeBenchDatasetError(f"Invalid contest_date: {value!r}") from exc


def _difficulty(value: Any) -> str:
    normalized = str(value).strip().lower()
    aliases = {"med": "medium", "introductory": "easy", "interview": "medium", "competition": "hard"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"easy", "medium", "hard"}:
        raise LiveCodeBenchDatasetError(f"Unsupported difficulty: {value!r}")
    return normalized


def validate_row(row: dict[str, Any]) -> None:
    missing = EXPECTED_COLUMNS.difference(row)
    if missing:
        raise LiveCodeBenchDatasetError(f"Dataset row is missing columns: {sorted(missing)}")
    if not str(row["question_id"]).strip() or not str(row["question_content"]).strip():
        raise LiveCodeBenchDatasetError("question_id and question_content must be non-empty")
    _date(row["contest_date"])
    _difficulty(row["difficulty"])
    _json_value(row["public_test_cases"], [])
    decode_private_tests(row["private_test_cases"])
    _json_value(row["metadata"], {})


def load_source_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Download only the incremental v6 config and return its fully validated rows."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise LiveCodeBenchDatasetError(
            "Missing dependency `datasets`; run `conda env update -n agent_verify -f environment.yml`."
        ) from exc
    # Load the pinned parquet directly. The same upstream revision also contains
    # a legacy dataset script, which datasets>=4 intentionally refuses to run.
    parquet_uri = (
        f"hf://datasets/{DATASET_ID}@{DATASET_REVISION}/{DATASET_CONFIG}/test-00000-of-00001.parquet"
    )
    dataset = load_dataset(
        "parquet",
        data_files={DATASET_SPLIT: parquet_uri},
        split=DATASET_SPLIT,
    )
    columns = set(dataset.column_names)
    if columns != EXPECTED_COLUMNS:
        raise LiveCodeBenchDatasetError(
            f"Pinned schema mismatch: expected {sorted(EXPECTED_COLUMNS)}, got {sorted(columns)}"
        )
    rows = [dict(row) for row in dataset]
    for row in rows:
        validate_row(row)
    info = {
        "dataset_fingerprint": getattr(dataset, "_fingerprint", ""),
        "source_rows": len(rows),
        "columns": sorted(columns),
    }
    return rows, info


def dataset_content_hash(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: str(value["question_id"])):
        digest.update(canonical_json(row).encode("utf-8"))
    return digest.hexdigest()


def select_rows(
    rows: list[dict[str, Any]],
    *,
    task_count: int = DEFAULT_TASK_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    min_contest_date: str = MIN_CONTEST_DATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select deterministically, preferring Easy/Medium and round-robin platform diversity."""

    cutoff = _date(min_contest_date)
    eligible = [row for row in rows if _date(row["contest_date"]) >= cutoff]
    if len(eligible) < task_count:
        raise LiveCodeBenchDatasetError(
            f"Only {len(eligible)} tasks satisfy contest_date >= {min_contest_date}; need {task_count}."
        )

    preferred = [row for row in eligible if _difficulty(row["difficulty"]) in {"easy", "medium"}]
    hard = [row for row in eligible if _difficulty(row["difficulty"]) == "hard"]

    def take_diverse(
        candidates: list[dict[str, Any]],
        needed: int,
        stage: str,
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            grouped[(_difficulty(row["difficulty"]), str(row["platform"]).lower())].append(row)
        for key, values in grouped.items():
            values.sort(key=lambda row: str(row["question_id"]))
            random.Random(f"{seed}:{stage}:{key[0]}:{key[1]}").shuffle(values)
        group_keys = sorted(grouped, key=lambda key: (key[0], key[1]))
        random.Random(f"{seed}:{stage}").shuffle(group_keys)
        taken: list[dict[str, Any]] = []
        while len(taken) < needed:
            progressed = False
            for key in group_keys:
                if grouped[key] and len(taken) < needed:
                    taken.append(grouped[key].pop())
                    progressed = True
            if not progressed:
                break
        return taken

    preferred_count = min(task_count, len(preferred))
    selected = take_diverse(preferred, preferred_count, "preferred")
    if len(selected) < task_count:
        selected.extend(take_diverse(hard, task_count - len(selected), "hard"))
    if len(selected) != task_count:
        raise LiveCodeBenchDatasetError(f"Selection produced {len(selected)} tasks, expected {task_count}")

    stats = {
        "eligible_count": len(eligible),
        "eligible_difficulty": dict(
            sorted(Counter(_difficulty(row["difficulty"]) for row in eligible).items())
        ),
        "selected_difficulty": dict(
            sorted(Counter(_difficulty(row["difficulty"]) for row in selected).items())
        ),
        "selected_platform": dict(sorted(Counter(str(row["platform"]).lower() for row in selected).items())),
        "hard_supplemented": sum(_difficulty(row["difficulty"]) == "hard" for row in selected),
    }
    return selected, stats


def _format_instruction(metadata: dict[str, Any], starter_code: str) -> tuple[str, str, str]:
    func_name = str(metadata.get("func_name") or "")
    if func_name:
        starter = f"\n\nSTARTER CODE:\n{starter_code}" if starter_code.strip() else ""
        return (
            "function",
            func_name,
            "Return a complete Python solution implementing the required function/class interface. "
            "Do not read stdin unless the problem explicitly requires it." + starter,
        )
    return (
        "stdin",
        "",
        "Return a complete Python program that reads from standard input and writes the answer "
        "to standard output.",
    )


def row_to_public_task(row: dict[str, Any]) -> LiveCodeBenchPublicTask:
    metadata = _json_value(row["metadata"], {})
    mode, entry_point, instruction = _format_instruction(metadata, str(row.get("starter_code") or ""))
    task_id = f"LiveCodeBench/{row['question_id']}"
    prompt = (
        f"TITLE: {row['question_title']}\n\n"
        f"{row['question_content'].strip()}\n\n"
        f"SUBMISSION FORMAT:\n{instruction}"
    )
    return LiveCodeBenchPublicTask(
        task_id=task_id,
        prompt=prompt,
        entry_point=entry_point,
        benchmark="livecodebench",
        title=str(row["question_title"]),
        difficulty=_difficulty(row["difficulty"]),
        platform=str(row["platform"]).lower(),
        contest_date=str(row["contest_date"]),
        starter_code=str(row.get("starter_code") or ""),
        input_mode=mode,
    )


def row_to_private_case(row: dict[str, Any]) -> LiveCodeBenchPrivateCase:
    public_tests = _json_value(row["public_test_cases"], [])
    private_tests = decode_private_tests(row["private_test_cases"])
    metadata = _json_value(row["metadata"], {})
    tests = list(public_tests) + list(private_tests)
    evaluation_sample = {
        "inputs": [test["input"] for test in tests],
        "outputs": [test["output"] for test in tests],
        "fn_name": metadata.get("func_name"),
    }
    return LiveCodeBenchPrivateCase(
        task_id=f"LiveCodeBench/{row['question_id']}",
        input_output=json.dumps(evaluation_sample, ensure_ascii=False),
        test_count=len(tests),
    )


def _public_record(row: dict[str, Any]) -> dict[str, Any]:
    task = row_to_public_task(row)
    public_tests = _json_value(row["public_test_cases"], [])
    return {
        **task.model_dump(mode="json"),
        "question_id": str(row["question_id"]),
        "contest_id": str(row["contest_id"]),
        "public_test_count": len(public_tests),
    }


def prepare_locked_selection(
    output_dir: Path,
    *,
    task_count: int = DEFAULT_TASK_COUNT,
    seed: int = DEFAULT_SELECTION_SEED,
    force: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_tasks.json"
    manifest_path = output_dir / "dataset_manifest.json"
    if (selected_path.exists() or manifest_path.exists()) and not force:
        if not selected_path.exists() or not manifest_path.exists():
            raise LiveCodeBenchDatasetError("Selection lock is incomplete; use --force after inspection")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(existing.get("task_count", -1)) != task_count or int(
            existing.get("selection_seed", -1)
        ) != seed:
            raise LiveCodeBenchDatasetError(
                "Existing selection lock uses a different task_count/seed; inspect it before --force"
            )
        return existing

    rows, source_info = load_source_rows()
    selected, stats = select_rows(rows, task_count=task_count, seed=seed)
    public_records = [_public_record(row) for row in selected]
    selection_hash = hashlib.sha256(canonical_json(public_records).encode("utf-8")).hexdigest()
    source_hash = dataset_content_hash(rows)
    selected_payload = {
        "schema_version": 1,
        "selection_seed": seed,
        "task_count": task_count,
        "selection_sha256": selection_hash,
        "tasks": public_records,
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "dataset_id": DATASET_ID,
        "config_name": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "revision": DATASET_REVISION,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_evaluator_commit": OFFICIAL_EVALUATOR_COMMIT,
        "minimum_contest_date": MIN_CONTEST_DATE,
        "selection_seed": seed,
        "task_count": task_count,
        "task_ids": [record["task_id"] for record in public_records],
        "dataset_sha256": source_hash,
        "selection_sha256": selection_hash,
        **source_info,
        **stats,
    }
    selected_path.write_text(
        json.dumps(selected_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_locked_selection(
    output_dir: Path,
    *,
    task_count: int | None = None,
) -> DatasetSelection:
    selected_path = output_dir / "selected_tasks.json"
    manifest_path = output_dir / "dataset_manifest.json"
    if not selected_path.exists() or not manifest_path.exists():
        raise LiveCodeBenchDatasetError(
            "Locked task files are missing. Run scripts/run_livecodebench_pilot.sh --prepare-only first."
        )
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    public_records = selected_payload["tasks"]
    actual_selection_hash = hashlib.sha256(canonical_json(public_records).encode("utf-8")).hexdigest()
    if actual_selection_hash != manifest["selection_sha256"]:
        raise LiveCodeBenchDatasetError("selected_tasks.json hash does not match dataset_manifest.json")

    rows, _ = load_source_rows()
    if dataset_content_hash(rows) != manifest["dataset_sha256"]:
        raise LiveCodeBenchDatasetError("Pinned LiveCodeBench dataset content hash changed")
    by_id = {f"LiveCodeBench/{row['question_id']}": row for row in rows}
    locked_ids = list(manifest["task_ids"])
    count = len(locked_ids) if task_count is None else task_count
    if count < 1 or count > len(locked_ids):
        raise LiveCodeBenchDatasetError(f"task_count must be in [1, {len(locked_ids)}]")
    selected_ids = locked_ids[:count]
    missing = [task_id for task_id in selected_ids if task_id not in by_id]
    if missing:
        raise LiveCodeBenchDatasetError(f"Locked tasks missing from pinned dataset: {missing[:5]}")
    public_tasks = [row_to_public_task(by_id[task_id]) for task_id in selected_ids]
    private_cases = {task_id: row_to_private_case(by_id[task_id]) for task_id in selected_ids}
    return DatasetSelection(public_tasks=public_tasks, private_cases=private_cases, metadata=manifest)


