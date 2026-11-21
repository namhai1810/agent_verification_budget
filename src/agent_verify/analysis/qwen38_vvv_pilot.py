"""Qwen3.8 VVV-only checkpoint-verification pilot.

The pilot starts from the cached Qwen2.5 Student planner artifact, then executes
exactly one branch-consistent path::

    cached plan -> Verify(CP1) -> Coder -> Verify(CP2) -> Reviewer -> Verify(CP3)

Only training tasks are eligible. Cached SSS and Qwen2.5-VVV outcomes are read for
selection/comparison; neither outcome nor evaluator feedback is included in any LLM
message.  Per-task/checkpoint snapshots are written atomically so resume can continue
without intentionally repeating completed Teacher calls.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from pydantic import Field, model_validator
from tqdm.auto import tqdm

from agent_verify.agents.fixed import CoderAgent, ReviewerAgent
from agent_verify.analysis.darc_v052_synergy_audit import load_expanded_train_groups
from agent_verify.budget import BudgetManager
from agent_verify.cbv_bench.cache import sha256_json
from agent_verify.cbv_bench.config import load_cbv_config
from agent_verify.cbv_bench.datasets import BenchmarkSelection, load_selection
from agent_verify.cbv_bench.tree import PrivacyGuard, deterministic_seed
from agent_verify.config import (
    BudgetConfig,
    EvaluationConfig,
    ModelConfig,
    ProviderConfig,
    StrictModel,
)
from agent_verify.evaluation.bigcodebench_runner import BigCodeBenchRunner
from agent_verify.evaluation.humaneval_runner import HumanEvalRunner
from agent_verify.evaluation.livecodebench_runner import LiveCodeBenchRunner
from agent_verify.llm import LLMClient
from agent_verify.logging import JsonlStore, content_id, hash_file
from agent_verify.schemas import (
    CheckpointState,
    CodeArtifact,
    InvocationContext,
    PlanArtifact,
    PublicTask,
    ReviewArtifact,
    utc_now,
)
from agent_verify.verification.teacher import TeacherVerifier

COHORT_UNSOLVED = "old_oracle_unsolvable"
COHORT_RESCUED = "verification_rescued"
COHORT_SSS = "sss_correct"
COHORTS = (COHORT_UNSOLVED, COHORT_RESCUED, COHORT_SSS)
DATASETS = ("humaneval", "livecodebench", "bigcodebench")


class DataConfig(StrictModel):
    frozen_root: Path
    extension_root: Path
    extension_selection_manifest: Path
    outcome_table: Path
    split: Literal["train"] = "train"


class SelectionConfig(StrictModel):
    seed: int = 42
    old_oracle_unsolvable: int = Field(default=20, ge=1)
    verification_rescued: int = Field(default=10, ge=1)
    sss_correct: int = Field(default=10, ge=1)

    @property
    def total(self) -> int:
        return self.old_oracle_unsolvable + self.verification_rescued + self.sss_correct


class RuntimeConfig(StrictModel):
    output_root: Path
    run_id: str
    progress: bool = True
    student_base_url: str
    teacher_base_url: str
    api_key_env: str = "VLLM_API_KEY"
    timeout_seconds: float = Field(default=900.0, gt=0)
    evaluator_timeout_seconds: int = Field(default=12, ge=1)
    evaluator_memory_mb: int = Field(default=2048, ge=64)
    evaluator_max_processes: int = Field(default=1, ge=1)
    dataset_configs: dict[str, Path]

    @model_validator(mode="after")
    def all_dataset_configs(self) -> RuntimeConfig:
        if set(self.dataset_configs) != set(DATASETS):
            raise ValueError(f"runtime.dataset_configs must contain exactly {DATASETS}")
        return self


class StudentConfig(StrictModel):
    model_id: str


class TeacherConfig(StrictModel):
    model_id: str
    family: str
    quantization: str
    parameter_count_b: float
    prompt_version: str
    inference_seed: int
    temperature: float
    top_p: float
    top_k: int
    max_output_tokens: int
    thinking_enabled: bool
    reasoning_parser: str
    thinking_token_budget: int
    include_reasoning: bool = True
    retries: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_thinking(self) -> TeacherConfig:
        if not self.thinking_enabled:
            raise ValueError("VVV40 requires thinking_enabled=true")
        if self.reasoning_parser != "qwen3":
            raise ValueError("VVV40 requires the qwen3 reasoning parser")
        if self.thinking_token_budget != 3500:
            raise ValueError("VVV40 locks thinking_token_budget=3500")
        return self


class SmokeConfig(StrictModel):
    total_tasks: int = Field(default=2, ge=1)


class PilotConfig(StrictModel):
    schema_version: Literal["darc-qwen38-vvv40-v1"]
    data: DataConfig
    selection: SelectionConfig
    runtime: RuntimeConfig
    student: StudentConfig
    teacher: TeacherConfig
    smoke: SmokeConfig = Field(default_factory=SmokeConfig)


def load_config(path: Path) -> PilotConfig:
    return PilotConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    fields = sorted({key for row in rows for key in row}) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stable_rank(seed: int, cohort: str, task_key: str) -> str:
    return hashlib.sha256(f"qwen38-vvv40:{seed}:{cohort}:{task_key}".encode()).hexdigest()


def classify_outcome_row(row: dict[str, str]) -> str:
    outcomes = [
        int(row[f"outcome_{sequence}"])
        for sequence in ("SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV")
    ]
    if outcomes[0] == 1:
        return COHORT_SSS
    if any(outcomes[1:]):
        return COHORT_RESCUED
    return COHORT_UNSOLVED


def select_tasks(rows: list[dict[str, str]], selection: SelectionConfig) -> list[dict[str, Any]]:
    """Select three disjoint, seeded cohorts from exactly 531 train L2 groups."""

    if len(rows) != 531 or len({row["task_key"] for row in rows}) != 531:
        raise ValueError("Outcome table must contain exactly 531 unique task rows")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Outcome table contains a non-training task")
    pools: dict[str, list[dict[str, str]]] = {name: [] for name in COHORTS}
    for row in rows:
        pools[classify_outcome_row(row)].append(row)
    requested = {
        COHORT_UNSOLVED: selection.old_oracle_unsolvable,
        COHORT_RESCUED: selection.verification_rescued,
        COHORT_SSS: selection.sss_correct,
    }
    selected: list[dict[str, Any]] = []
    for cohort in COHORTS:
        ranked = sorted(pools[cohort], key=lambda row: _stable_rank(selection.seed, cohort, row["task_key"]))
        if len(ranked) < requested[cohort]:
            raise ValueError(f"Cohort {cohort} has only {len(ranked)} eligible tasks")
        for within_rank, row in enumerate(ranked[: requested[cohort]]):
            selected.append(
                {
                    **row,
                    "selection_cohort": cohort,
                    "selection_seed": selection.seed,
                    "cohort_rank": within_rank,
                }
            )
    if len(selected) != selection.total or len({row["task_key"] for row in selected}) != selection.total:
        raise AssertionError("Selection is not distinct or has the wrong size")
    return selected


def _task_records(root: Path, frozen_root: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(_resolve(root, frozen_root) / "train" / "tasks.jsonl")
    return {f"{row['dataset']}::{row['task_id']}": row for row in rows}


def _selection_manifest_payload(
    dataset: str,
    selected: list[dict[str, Any]],
    task_records: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    records = []
    for rank, selected_row in enumerate(selected):
        source = task_records[selected_row["task_key"]]
        records.append(
            {
                "schema_version": "cbv-bench-v2",
                "record_type": "cbv_task",
                "dataset": dataset,
                "dataset_version": source["dataset_version"],
                "task_id": source["task_id"],
                "difficulty": source.get("difficulty", "unknown"),
                "release_date": source.get("source_metadata", {}).get("release_date", ""),
                "source_metadata": source.get("source_metadata", {}),
                "public_task": source["public_task"],
                "public_task_sha256": source["public_task_sha256"],
                "selection_seed": seed,
                "selection_rank": rank,
                "split": "train",
                "exploratory_reason": "",
            }
        )
    payload = {
        "schema_version": "cbv-selection-v2",
        "created_at_utc": utc_now(),
        "dataset": dataset,
        "dataset_version": records[0]["dataset_version"] if records else "",
        "mode": "qwen38-vvv40-seeded-outcome-strata",
        "selection_seed": seed,
        "selection_protocol": (
            "Training-only seeded SHA-256 ordering inside three locked old-Qwen2.5 "
            "outcome strata; no Qwen3.8 outcome is used"
        ),
        "source_metadata": {"cohorts": Counter(row["selection_cohort"] for row in selected)},
        "task_records": records,
    }
    payload["selection_sha256"] = sha256_json(payload)
    return payload


def prepare_selection(config: PilotConfig, root: Path) -> tuple[list[dict[str, Any]], Path]:
    outcome_path = _resolve(root, config.data.outcome_table)
    with outcome_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_tasks(rows, config.selection)
    output = _resolve(root, config.runtime.output_root)
    selection_dir = output / "selection"
    master_path = selection_dir / "selected_tasks.json"
    source_hash = hash_file(outcome_path)
    manifest = {
        "schema_version": "darc-qwen38-vvv40-selection-v1",
        "selection_seed": config.selection.seed,
        "source_outcome_table": str(config.data.outcome_table),
        "source_outcome_sha256": source_hash,
        "tasks": selected,
        "counts": dict(Counter(row["selection_cohort"] for row in selected)),
        "dataset_counts": dict(Counter(row["dataset"] for row in selected)),
    }
    manifest["selection_sha256"] = sha256_json(manifest)
    if master_path.exists():
        existing = json.loads(master_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Locked VVV40 selection differs from deterministic regeneration")
    else:
        _atomic_json(master_path, manifest)

    records = _task_records(root, config.data.frozen_root)
    for dataset in DATASETS:
        dataset_rows = [row for row in selected if row["dataset"] == dataset]
        payload = _selection_manifest_payload(dataset, dataset_rows, records, config.selection.seed)
        path = selection_dir / f"{dataset}.json"
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            # created_at is intentionally frozen at the first materialization.
            payload["created_at_utc"] = current.get("created_at_utc", payload["created_at_utc"])
            payload["selection_sha256"] = sha256_json(
                {key: value for key, value in payload.items() if key != "selection_sha256"}
            )
            if current != payload:
                raise ValueError(f"Locked dataset selection changed: {path}")
        else:
            _atomic_json(path, payload)
    return selected, master_path


def _load_groups(config: PilotConfig, root: Path) -> dict[str, Any]:
    original, extension = load_expanded_train_groups(
        _resolve(root, config.data.frozen_root),
        _resolve(root, config.data.extension_root),
        _resolve(root, config.data.extension_selection_manifest),
    )
    groups = {**original, **extension}
    if len(groups) != 531:
        raise ValueError(f"Expected 531 expanded train groups, found {len(groups)}")
    return groups


def _load_benchmark_selections(
    config: PilotConfig,
    root: Path,
    selected: list[dict[str, Any]],
) -> dict[str, BenchmarkSelection]:
    output = _resolve(root, config.runtime.output_root)
    selections: dict[str, BenchmarkSelection] = {}
    for dataset in DATASETS:
        count = sum(row["dataset"] == dataset for row in selected)
        if count == 0:
            continue
        base = load_cbv_config(_resolve(root, config.runtime.dataset_configs[dataset]))
        spec = next(item for item in base.datasets if item.name == dataset)
        spec = spec.model_copy(
            update={
                "enabled": True,
                "selection_manifest": output / "selection" / f"{dataset}.json",
                "task_count": count,
            }
        )
        selections[dataset] = load_selection(base, spec, root)
    return selections


def _evaluation_config(config: PilotConfig) -> EvaluationConfig:
    return EvaluationConfig(
        sandbox="landlock_seccomp",
        timeout_seconds=config.runtime.evaluator_timeout_seconds,
        memory_mb=config.runtime.evaluator_memory_mb,
        max_processes=config.runtime.evaluator_max_processes,
    )


def _evaluator(
    dataset: str,
    selection: BenchmarkSelection,
    config: PilotConfig,
    store: JsonlStore,
    root: Path,
):
    evaluation = _evaluation_config(config)
    if dataset == "humaneval":
        return HumanEvalRunner(evaluation, store)
    if dataset == "livecodebench":
        official = selection.spec.official_evaluator_path
        path = official if official and official.is_absolute() else root / str(official)
        return LiveCodeBenchRunner(evaluation, store, path)
    return BigCodeBenchRunner(evaluation, store)


def _model_from_cached(payload: dict[str, Any], config: PilotConfig) -> ModelConfig:
    model = ModelConfig.model_validate(payload)
    if model.model_id != config.student.model_id:
        raise ValueError(f"Unexpected cached Student model: {model.model_id}")
    return model.model_copy(
        update={
            "base_url": config.runtime.student_base_url,
            "api_key_env": config.runtime.api_key_env,
        }
    )


def _teacher_model(config: PilotConfig) -> ModelConfig:
    teacher = config.teacher
    return ModelConfig(
        provider="openai_compatible",
        model_id=teacher.model_id,
        base_url=config.runtime.teacher_base_url,
        api_key_env=config.runtime.api_key_env,
        temperature=teacher.temperature,
        top_p=teacher.top_p,
        top_k=teacher.top_k,
        max_output_tokens=teacher.max_output_tokens,
        seed=teacher.inference_seed,
        prompt_version=teacher.prompt_version,
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
    )


def _provider(config: PilotConfig) -> ProviderConfig:
    return ProviderConfig(
        base_url=config.runtime.student_base_url,
        api_key_env=config.runtime.api_key_env,
        timeout_seconds=config.runtime.timeout_seconds,
        max_retries=config.teacher.retries,
        response_format="json_schema",
    )


def _api_models(base_url: str, api_key: str) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.get(base_url.rstrip("/") + "/models", headers=headers, timeout=10)
    response.raise_for_status()
    return [str(item["id"]) for item in response.json().get("data", [])]


def check_apis(config: PilotConfig) -> dict[str, Any]:
    api_key = os.environ.get(config.runtime.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"{config.runtime.api_key_env} is not set")
    student_models = _api_models(config.runtime.student_base_url, api_key)
    teacher_models = _api_models(config.runtime.teacher_base_url, api_key)
    ready = {
        "student_base_url": config.runtime.student_base_url,
        "teacher_base_url": config.runtime.teacher_base_url,
        "student_models": student_models,
        "teacher_models": teacher_models,
        "student_ready": config.student.model_id in student_models,
        "teacher_ready": config.teacher.model_id in teacher_models,
    }
    if not ready["student_ready"] or not ready["teacher_ready"]:
        raise RuntimeError(f"Model API mismatch: {ready}")
    return ready


def _state(
    *,
    run_id: str,
    task: PublicTask,
    checkpoint: int,
    branch: str,
    rollout_seed: int,
    preceding: list[dict[str, Any]],
    current: dict[str, Any],
    model: ModelConfig,
    invocation_id: str,
    parent_state_id: str | None,
    messages: list[Any],
) -> CheckpointState:
    payload = {
        "run_id": run_id,
        "task_id": task.task_id,
        "checkpoint": checkpoint,
        "branch": branch,
        "preceding": preceding,
        "current": current,
        "parent": parent_state_id,
    }
    return CheckpointState(
        state_id=content_id("q38_vvv_state", payload),
        run_id=run_id,
        task_id=task.task_id,
        checkpoint_index=checkpoint,
        role={1: "planner", 2: "coder", 3: "reviewer"}[checkpoint],
        branch=f"qwen38-vvv:{branch}",
        rollout_seed=rollout_seed,
        parent_state_id=parent_state_id,
        preceding_artifacts=preceding,
        current_artifact=current,
        messages=messages,
        llm_config=model,
        invocation_id=invocation_id,
    )


def _invocation(store: JsonlStore, invocation_id: str) -> dict[str, Any]:
    for row in reversed(store.read_all()):
        if row.get("invocation_id") == invocation_id:
            return row
    raise ValueError(f"Missing invocation record {invocation_id}")


def _reasoning_tokens(record: dict[str, Any]) -> int:
    total = 0
    for attempt in record.get("attempts", []):
        value = (attempt.get("response_metadata") or {}).get("reasoning_tokens")
        if value is not None:
            total += int(value)
    return total


def _teacher_metrics(output: Any, store: JsonlStore) -> dict[str, Any]:
    record = _invocation(store, output.invocation_id)
    return {
        "verdict": output.result.verdict,
        "changed": bool(output.result.changed),
        "explanation": output.result.explanation,
        "corrected_artifact": output.corrected_artifact.model_dump(mode="json"),
        "invocation_id": output.invocation_id,
        "input_tokens": int(record["input_tokens"]),
        "output_tokens": int(record["output_tokens"]),
        "actual_total_tokens": int(record["input_tokens"]) + int(record["output_tokens"]),
        "reasoning_tokens": _reasoning_tokens(record),
        "latency_seconds": float(record["latency_seconds"]),
    }


def _save_progress(path: Path, progress: dict[str, Any]) -> None:
    progress["updated_at_utc"] = utc_now()
    _atomic_json(path, progress)


def _student_recovery(
    store: JsonlStore,
    *,
    task_id: str,
    role: str,
) -> dict[str, Any] | None:
    """Return an auditable anti-repetition override after proven truncation only."""

    failures = []
    for row in store.read_all():
        context = row.get("context", {})
        if context.get("task_id") != task_id or context.get("role") != role:
            continue
        failure = str(row.get("failure") or "")
        if "completion truncated: finish_reason=length" in failure:
            failures.append(row)
    if not failures:
        return None
    count = len(failures)
    # Keep the first anti-repetition adjustment mild. Reviewer truncations also
    # switch immediately to the semantically equivalent code_lines wire schema:
    # observed failures can otherwise spend the whole token budget repeating
    # escaped newlines/backticks inside one JSON string. Successful tasks and
    # first attempts remain untouched. If the same task/role truncates again,
    # escalate the repetition penalty using only proven matching failures.
    repetition_penalty = 1.30 if count >= 3 else 1.20 if count >= 2 else 1.08
    return {
        "reason": "prior_deterministic_student_repetition_truncation",
        "prior_truncations": count,
        "repetition_penalty": repetition_penalty,
        "recovery_strength": (
            "emergency" if count >= 3 else "aggressive" if count >= 2 else "mild"
        ),
        "line_safe_output": role == "reviewer",
        "temperature_unchanged": True,
        "max_output_tokens_unchanged": True,
    }


def _task_file(output: Path, task_key: str) -> Path:
    safe = hashlib.sha256(task_key.encode()).hexdigest()[:16]
    return output / "task_checkpoints" / f"{safe}.json"


def _run_task(
    *,
    selected_row: dict[str, Any],
    group: Any,
    task: PublicTask,
    private_case: Any,
    evaluator: Any,
    config: PilotConfig,
    client: LLMClient,
    invocations: JsonlStore,
    events: JsonlStore,
    output: Path,
    run_id: str,
) -> dict[str, Any]:
    task_key = selected_row["task_key"]
    path = _task_file(output, task_key)
    if path.exists():
        progress = json.loads(path.read_text(encoding="utf-8"))
        if progress.get("status") == "completed":
            return progress
    else:
        progress = {
            "schema_version": "darc-qwen38-vvv40-task-v1",
            "status": "running",
            "run_id": run_id,
            "task_key": task_key,
            "dataset": selected_row["dataset"],
            "task_id": selected_row["task_id"],
            "selection_cohort": selected_row["selection_cohort"],
            "old_sss_passed": bool(int(selected_row["outcome_SSS"])),
            "old_qwen25_vvv_passed": bool(int(selected_row["outcome_VVV"])),
            "old_qwen25_vvv_tokens": int(selected_row["vvv_tokens"]),
            "action_sequence": "VVV",
            "checkpoints": {},
        }
        _save_progress(path, progress)

    cached_sss = group.rows["SSS"]
    rollout_seed = int(cached_sss["rollout_seed"])
    planner_model = _model_from_cached(cached_sss["student_models"]["planner"], config)
    coder_model = _model_from_cached(cached_sss["student_models"]["coder"], config)
    reviewer_model = _model_from_cached(cached_sss["student_models"]["reviewer"], config)
    teacher_model = _teacher_model(config)
    teacher = TeacherVerifier(
        client,
        teacher_model,
        events,
        request_overrides={
            "thinking_token_budget": config.teacher.thinking_token_budget,
            "include_reasoning": config.teacher.include_reasoning,
        },
    )
    coder_recovery = _student_recovery(invocations, task_id=task.task_id, role="coder")
    reviewer_recovery = _student_recovery(invocations, task_id=task.task_id, role="reviewer")
    coder = CoderAgent(
        client,
        coder_model,
        request_overrides=(
            {"repetition_penalty": coder_recovery["repetition_penalty"]}
            if coder_recovery
            else None
        ),
    )
    reviewer = ReviewerAgent(
        client,
        reviewer_model,
        events,
        request_overrides=(
            {"repetition_penalty": reviewer_recovery["repetition_penalty"]}
            if reviewer_recovery
            else None
        ),
        line_safe_output=bool(
            reviewer_recovery and reviewer_recovery.get("line_safe_output")
        ),
    )
    if coder_recovery or reviewer_recovery:
        progress["student_recovery"] = {
            "coder": coder_recovery,
            "reviewer": reviewer_recovery,
        }
        _save_progress(path, progress)

    plan = PlanArtifact.model_validate(cached_sss["artifact_snapshots"][0])
    state1 = _state(
        run_id=run_id,
        task=task,
        checkpoint=1,
        branch="root",
        rollout_seed=rollout_seed,
        preceding=[],
        current=plan.model_dump(mode="json"),
        model=planner_model,
        invocation_id=f"cached:{cached_sss['checkpoint_state_ids'][0]}",
        parent_state_id=None,
        messages=[],
    )
    if "CP1" not in progress["checkpoints"]:
        context = InvocationContext(
            task_id=task.task_id,
            checkpoint=1,
            role="teacher",
            branch="qwen38-vvv:V",
            run_id=run_id,
            seed=deterministic_seed(rollout_seed, 1, teacher_model.seed, "V"),
        )
        cp1 = teacher.verify(task, state1, plan, context)
        record = _invocation(invocations, cp1.invocation_id)
        PrivacyGuard.assert_messages_public(record["messages"], private_case)
        progress["checkpoints"]["CP1"] = _teacher_metrics(cp1, invocations)
        _save_progress(path, progress)
    plan = PlanArtifact.model_validate(progress["checkpoints"]["CP1"]["corrected_artifact"])

    if "coder_after_CP1" not in progress:
        context = InvocationContext(
            task_id=task.task_id,
            checkpoint=2,
            role="coder",
            branch="qwen38-vvv:V",
            run_id=run_id,
            seed=deterministic_seed(rollout_seed, 2, coder_model.seed, "V"),
        )
        coder_output = coder.run(task, plan, context)
        PrivacyGuard.assert_messages_public(coder_output.messages, private_case)
        progress["coder_after_CP1"] = {
            "artifact": coder_output.artifact.model_dump(mode="json"),
            "invocation_id": coder_output.llm_result.record.invocation_id,
            "input_tokens": coder_output.llm_result.record.input_tokens,
            "output_tokens": coder_output.llm_result.record.output_tokens,
            "latency_seconds": coder_output.llm_result.record.latency_seconds,
            "recovery": coder_recovery,
        }
        _save_progress(path, progress)
    code = CodeArtifact.model_validate(progress["coder_after_CP1"]["artifact"])
    state2 = _state(
        run_id=run_id,
        task=task,
        checkpoint=2,
        branch="V",
        rollout_seed=rollout_seed,
        preceding=[plan.model_dump(mode="json")],
        current=code.model_dump(mode="json"),
        model=coder_model,
        invocation_id=progress["coder_after_CP1"]["invocation_id"],
        parent_state_id=state1.state_id,
        messages=[],
    )
    if "CP2" not in progress["checkpoints"]:
        context = InvocationContext(
            task_id=task.task_id,
            checkpoint=2,
            role="teacher",
            branch="qwen38-vvv:VV",
            run_id=run_id,
            seed=deterministic_seed(rollout_seed, 2, teacher_model.seed, "VV"),
        )
        cp2 = teacher.verify(task, state2, code, context)
        record = _invocation(invocations, cp2.invocation_id)
        PrivacyGuard.assert_messages_public(record["messages"], private_case)
        progress["checkpoints"]["CP2"] = _teacher_metrics(cp2, invocations)
        _save_progress(path, progress)
    code = CodeArtifact.model_validate(progress["checkpoints"]["CP2"]["corrected_artifact"])

    if "reviewer_after_CP2" not in progress:
        context = InvocationContext(
            task_id=task.task_id,
            checkpoint=3,
            role="reviewer",
            branch="qwen38-vvv:VV",
            run_id=run_id,
            seed=deterministic_seed(rollout_seed, 3, reviewer_model.seed, "VV"),
        )
        reviewer_output = reviewer.run(task, plan, code, context)
        PrivacyGuard.assert_messages_public(reviewer_output.messages, private_case)
        progress["reviewer_after_CP2"] = {
            "artifact": reviewer_output.artifact.model_dump(mode="json"),
            "invocation_id": reviewer_output.llm_result.record.invocation_id,
            "input_tokens": reviewer_output.llm_result.record.input_tokens,
            "output_tokens": reviewer_output.llm_result.record.output_tokens,
            "latency_seconds": reviewer_output.llm_result.record.latency_seconds,
            "recovery": reviewer_recovery,
        }
        _save_progress(path, progress)
    review = ReviewArtifact.model_validate(progress["reviewer_after_CP2"]["artifact"])
    state3 = _state(
        run_id=run_id,
        task=task,
        checkpoint=3,
        branch="VV",
        rollout_seed=rollout_seed,
        preceding=[plan.model_dump(mode="json"), code.model_dump(mode="json")],
        current=review.model_dump(mode="json"),
        model=reviewer_model,
        invocation_id=progress["reviewer_after_CP2"]["invocation_id"],
        parent_state_id=state2.state_id,
        messages=[],
    )
    if "CP3" not in progress["checkpoints"]:
        context = InvocationContext(
            task_id=task.task_id,
            checkpoint=3,
            role="teacher",
            branch="qwen38-vvv:VVV",
            run_id=run_id,
            seed=deterministic_seed(rollout_seed, 3, teacher_model.seed, "VVV"),
        )
        cp3 = teacher.verify(task, state3, review, context)
        record = _invocation(invocations, cp3.invocation_id)
        PrivacyGuard.assert_messages_public(record["messages"], private_case)
        progress["checkpoints"]["CP3"] = _teacher_metrics(cp3, invocations)
        _save_progress(path, progress)
    final = ReviewArtifact.model_validate(progress["checkpoints"]["CP3"]["corrected_artifact"])

    if "evaluation" not in progress:
        evaluation = evaluator.evaluate(
            run_id=run_id,
            case=private_case,
            code=final.code,
            branch="qwen38-vvv:VVV",
        )
        progress["evaluation"] = evaluation.model_dump(mode="json")
        progress["qwen38_vvv_passed"] = bool(evaluation.passed)
        _save_progress(path, progress)

    progress["teacher_calls"] = 3
    progress["teacher_input_tokens"] = sum(
        int(progress["checkpoints"][cp]["input_tokens"]) for cp in ("CP1", "CP2", "CP3")
    )
    progress["teacher_output_tokens"] = sum(
        int(progress["checkpoints"][cp]["output_tokens"]) for cp in ("CP1", "CP2", "CP3")
    )
    progress["teacher_reasoning_tokens"] = sum(
        int(progress["checkpoints"][cp]["reasoning_tokens"]) for cp in ("CP1", "CP2", "CP3")
    )
    progress["teacher_actual_total_tokens"] = (
        progress["teacher_input_tokens"] + progress["teacher_output_tokens"]
    )
    progress["teacher_latency_seconds"] = sum(
        float(progress["checkpoints"][cp]["latency_seconds"]) for cp in ("CP1", "CP2", "CP3")
    )
    progress["student_downstream_tokens"] = sum(
        int(progress[key]["input_tokens"]) + int(progress[key]["output_tokens"])
        for key in ("coder_after_CP1", "reviewer_after_CP2")
    )
    progress["student_downstream_latency_seconds"] = sum(
        float(progress[key]["latency_seconds"]) for key in ("coder_after_CP1", "reviewer_after_CP2")
    )
    progress["vs_sss"] = (
        "rescued"
        if progress["qwen38_vvv_passed"] and not progress["old_sss_passed"]
        else "harmed"
        if progress["old_sss_passed"] and not progress["qwen38_vvv_passed"]
        else "same_correct"
        if progress["qwen38_vvv_passed"]
        else "same_wrong"
    )
    progress["vs_old_qwen25_vvv"] = (
        "win"
        if progress["qwen38_vvv_passed"] and not progress["old_qwen25_vvv_passed"]
        else "loss"
        if progress["old_qwen25_vvv_passed"] and not progress["qwen38_vvv_passed"]
        else "tie_correct"
        if progress["qwen38_vvv_passed"]
        else "tie_wrong"
    )
    progress["status"] = "completed"
    _save_progress(path, progress)
    return progress


def _flatten_task(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_key": row["task_key"],
        "dataset": row["dataset"],
        "task_id": row["task_id"],
        "selection_cohort": row["selection_cohort"],
        "old_sss_passed": int(row["old_sss_passed"]),
        "old_qwen25_vvv_passed": int(row["old_qwen25_vvv_passed"]),
        "qwen38_vvv_passed": int(row["qwen38_vvv_passed"]),
        "vs_sss": row["vs_sss"],
        "vs_old_qwen25_vvv": row["vs_old_qwen25_vvv"],
        "old_qwen25_vvv_tokens": row["old_qwen25_vvv_tokens"],
        "teacher_calls": row["teacher_calls"],
        "teacher_input_tokens": row["teacher_input_tokens"],
        "teacher_output_tokens": row["teacher_output_tokens"],
        "teacher_reasoning_tokens": row["teacher_reasoning_tokens"],
        "teacher_actual_total_tokens": row["teacher_actual_total_tokens"],
        "teacher_latency_seconds": row["teacher_latency_seconds"],
        "student_downstream_tokens": row["student_downstream_tokens"],
        "student_downstream_latency_seconds": row["student_downstream_latency_seconds"],
        "student_recovery_used": int(bool(row.get("student_recovery"))),
        "coder_recovery": json.dumps(
            row.get("student_recovery", {}).get("coder"), sort_keys=True
        ),
        "reviewer_recovery": json.dumps(
            row.get("student_recovery", {}).get("reviewer"), sort_keys=True
        ),
    }


def _checkpoint_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for checkpoint in ("CP1", "CP2", "CP3"):
            item = result["checkpoints"][checkpoint]
            rows.append(
                {
                    "task_key": result["task_key"],
                    "dataset": result["dataset"],
                    "selection_cohort": result["selection_cohort"],
                    "checkpoint": checkpoint,
                    "verdict": item["verdict"],
                    "changed": int(item["changed"]),
                    "input_tokens": item["input_tokens"],
                    "output_tokens": item["output_tokens"],
                    "reasoning_tokens": item["reasoning_tokens"],
                    "actual_total_tokens": item["actual_total_tokens"],
                    "latency_seconds": item["latency_seconds"],
                    "final_passed": int(result["qwen38_vvv_passed"]),
                }
            )
    return rows


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = len(results)
    passed = sum(bool(row["qwen38_vvv_passed"]) for row in results)
    checkpoint_rows = _checkpoint_rows(results)
    checkpoints = {}
    for checkpoint in ("CP1", "CP2", "CP3"):
        rows = [row for row in checkpoint_rows if row["checkpoint"] == checkpoint]
        checkpoints[checkpoint] = {
            "calls": len(rows),
            "changed": sum(row["changed"] for row in rows),
            "actual_total_tokens": sum(int(row["actual_total_tokens"]) for row in rows),
            "reasoning_tokens": sum(int(row["reasoning_tokens"]) for row in rows),
            "latency_seconds": sum(float(row["latency_seconds"]) for row in rows),
            "verdicts": dict(Counter(row["verdict"] for row in rows)),
        }
    return {
        "completed_tasks": completed,
        "qwen38_vvv_passed": passed,
        "qwen38_vvv_accuracy": passed / completed if completed else 0.0,
        "old_sss_passed": sum(bool(row["old_sss_passed"]) for row in results),
        "old_qwen25_vvv_passed": sum(bool(row["old_qwen25_vvv_passed"]) for row in results),
        "vs_sss": dict(Counter(row["vs_sss"] for row in results)),
        "vs_old_qwen25_vvv": dict(Counter(row["vs_old_qwen25_vvv"] for row in results)),
        "teacher_calls": sum(int(row["teacher_calls"]) for row in results),
        "teacher_actual_total_tokens": sum(int(row["teacher_actual_total_tokens"]) for row in results),
        "teacher_reasoning_tokens": sum(int(row["teacher_reasoning_tokens"]) for row in results),
        "teacher_latency_seconds": sum(float(row["teacher_latency_seconds"]) for row in results),
        "student_recovery_tasks": [
            row["task_key"] for row in results if row.get("student_recovery")
        ],
        "checkpoint_summary": checkpoints,
        "by_dataset": {
            dataset: {
                "tasks": len(rows := [row for row in results if row["dataset"] == dataset]),
                "passed": sum(bool(row["qwen38_vvv_passed"]) for row in rows),
                "rescued_vs_sss": sum(row["vs_sss"] == "rescued" for row in rows),
                "harmed_vs_sss": sum(row["vs_sss"] == "harmed" for row in rows),
            }
            for dataset in DATASETS
        },
        "by_cohort": {
            cohort: {
                "tasks": len(rows := [row for row in results if row["selection_cohort"] == cohort]),
                "passed": sum(bool(row["qwen38_vvv_passed"]) for row in rows),
                "old_sss_passed": sum(bool(row["old_sss_passed"]) for row in rows),
                "old_qwen25_vvv_passed": sum(bool(row["old_qwen25_vvv_passed"]) for row in rows),
            }
            for cohort in COHORTS
        },
    }


def _write_report(output: Path, summary: dict[str, Any], mode: str) -> Path:
    by_cohort = summary["by_cohort"]
    by_dataset = summary["by_dataset"]
    lines = [
        "# Qwen3.8 VVV Verification Pilot",
        "",
        f"> Mode: `{mode}`. Training tasks only. One branch-consistent `VVV` path per task.",
        "",
        "## Overall comparison",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Completed tasks | {summary['completed_tasks']} |",
        f"| Cached SSS correct | {summary['old_sss_passed']} |",
        f"| Cached Qwen2.5 VVV correct | {summary['old_qwen25_vvv_passed']} |",
        f"| Qwen3.8 VVV correct | **{summary['qwen38_vvv_passed']}** |",
        f"| Qwen3.8 VVV accuracy | {summary['qwen38_vvv_accuracy']:.2%} |",
        f"| Teacher calls | {summary['teacher_calls']} |",
        f"| Actual Teacher tokens | {summary['teacher_actual_total_tokens']:,} |",
        f"| Reasoning tokens | {summary['teacher_reasoning_tokens']:,} |",
        f"| Teacher latency | {summary['teacher_latency_seconds'] / 60:.1f} min |",
        f"| Student anti-repetition recovery tasks | {len(summary['student_recovery_tasks'])} |",
        "",
        "## Locked selection strata",
        "",
        "| Cohort | Tasks | SSS | old Qwen2.5 VVV | Qwen3.8 VVV |",
        "|---|---:|---:|---:|---:|",
    ]
    for cohort in COHORTS:
        row = by_cohort[cohort]
        lines.append(
            f"| {cohort} | {row['tasks']} | {row['old_sss_passed']} | "
            f"{row['old_qwen25_vvv_passed']} | **{row['passed']}** |"
        )
    lines.extend(
        [
            "",
            "## Dataset breakdown",
            "",
            "| Dataset | Tasks | Passed | Rescued vs SSS | Harmed vs SSS |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        row = by_dataset[dataset]
        lines.append(
            f"| {dataset} | {row['tasks']} | {row['passed']} | "
            f"{row['rescued_vs_sss']} | {row['harmed_vs_sss']} |"
        )
    lines.extend(
        [
            "",
            "## Checkpoint accounting",
            "",
            "| Checkpoint | Calls | Changed | Actual tokens | Reasoning tokens | Latency (min) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in ("CP1", "CP2", "CP3"):
        row = summary["checkpoint_summary"][checkpoint]
        lines.append(
            f"| {checkpoint} | {row['calls']} | {row['changed']} | "
            f"{row['actual_total_tokens']:,} | {row['reasoning_tokens']:,} | "
            f"{row['latency_seconds'] / 60:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "`changed` is an artifact-difference diagnostic, not a causal Helpful label. "
            "Because only Qwen3.8 `VVV` is generated, checkpoint-specific causal effects cannot "
            "be identified without additional Qwen3.8 counterfactual branches. Final rescue/harm "
            "is paired against cached SSS; model comparison is paired against cached Qwen2.5 VVV.",
            "",
            "No hidden test, evaluator result, or execution feedback was included "
            "in Teacher or Student prompts.",
            "",
            "Student recovery is activated only after a logged deterministic length truncation. "
            "It preserves the model, prompt, temperature, seed, and output-token budget, while "
            "adding a logged vLLM repetition penalty. Recovered tasks remain identifiable in "
            "`task_results.csv` and should be reported as a sensitivity subset.",
        ]
    )
    report = output / "report" / "darc_qwen38_vvv40_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def dry_run(config: PilotConfig, root: Path) -> dict[str, Any]:
    selected, manifest = prepare_selection(config, root)
    payload = {
        "mode": "dry-run",
        "tasks": len(selected),
        "cohorts": dict(Counter(row["selection_cohort"] for row in selected)),
        "datasets": dict(Counter(row["dataset"] for row in selected)),
        "trajectory": "VVV",
        "teacher_calls": len(selected) * 3,
        "student_planner_calls": 0,
        "student_downstream_calls": len(selected) * 2,
        "selection_manifest": str(manifest),
        "output_root": str(config.runtime.output_root),
    }
    _atomic_json(_resolve(root, config.runtime.output_root) / "dry_run.json", payload)
    return payload


def run(
    config: PilotConfig,
    root: Path,
    *,
    mode: Literal["smoke", "full"],
    resume: bool,
) -> dict[str, Any]:
    selected, _manifest = prepare_selection(config, root)
    if mode == "smoke":
        # Interleave strata so smoke does not accidentally test only one cohort.
        interleaved = []
        for rank in range(
            max(
                config.selection.old_oracle_unsolvable,
                config.selection.verification_rescued,
                config.selection.sss_correct,
            )
        ):
            for cohort in COHORTS:
                matches = [row for row in selected if row["selection_cohort"] == cohort]
                if rank < len(matches):
                    interleaved.append(matches[rank])
        selected = interleaved[: config.smoke.total_tasks]
    output = _resolve(root, config.runtime.output_root) / ("smoke" if mode == "smoke" else "")
    output.mkdir(parents=True, exist_ok=True)
    existing = (
        list((output / "task_checkpoints").glob("*.json")) if (output / "task_checkpoints").exists() else []
    )
    if existing and not resume:
        raise FileExistsError(f"Existing incremental results found under {output}; rerun with --resume")

    groups = _load_groups(config, root)
    benchmark_selections = _load_benchmark_selections(config, root, selected)
    public: dict[str, PublicTask] = {}
    private: dict[str, Any] = {}
    for dataset, selection in benchmark_selections.items():
        for task in selection.public_tasks:
            key = f"{dataset}::{task.task_id}"
            public[key] = task
            private[key] = selection.private_cases[task.task_id]

    invocations = JsonlStore(output / "llm_invocations.jsonl")
    events = JsonlStore(output / "events.jsonl")
    evaluations = JsonlStore(output / "evaluations.jsonl")
    errors = JsonlStore(output / "errors.jsonl")
    provider = _provider(config)
    client = LLMClient(
        provider,
        BudgetManager(BudgetConfig(max_spend_usd=0.0, allow_paid_calls=True), provider),
        invocations,
    )
    evaluators = {
        dataset: _evaluator(dataset, selection, config, evaluations, root)
        for dataset, selection in benchmark_selections.items()
    }
    for dataset, evaluator in evaluators.items():
        evaluator.check_sandbox()
        if hasattr(evaluator, "check_official_evaluator"):
            evaluator.check_official_evaluator()
        if hasattr(evaluator, "check_dependencies"):
            evaluator.check_dependencies(benchmark_selections[dataset].public_tasks)

    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in selected:
        path = _task_file(output, row["task_key"])
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("status") == "completed":
                results.append(current)
                continue
        pending.append(row)
    bar = tqdm(
        pending,
        initial=len(results),
        total=len(selected),
        desc=f"Qwen3.8 VVV40 ({mode})",
        unit="task",
        disable=not config.runtime.progress,
        dynamic_ncols=True,
    )
    failures = 0
    started = time.perf_counter()
    for row in bar:
        try:
            result = _run_task(
                selected_row=row,
                group=groups[row["task_key"]],
                task=public[row["task_key"]],
                private_case=private[row["task_key"]],
                evaluator=evaluators[row["dataset"]],
                config=config,
                client=client,
                invocations=invocations,
                events=events,
                output=output,
                run_id=f"{config.runtime.run_id}-{mode}",
            )
            results.append(result)
        except Exception as exc:  # Persist and continue; resume can retry the incomplete stage.
            failures += 1
            errors.append(
                {
                    "record_type": "qwen38_vvv40_error",
                    "timestamp_utc": utc_now(),
                    "mode": mode,
                    "task_key": row["task_key"],
                    "dataset": row["dataset"],
                    "task_id": row["task_id"],
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        total_tokens = sum(int(item.get("teacher_actual_total_tokens", 0)) for item in results)
        bar.set_postfix(
            completed=len(results),
            failures=failures,
            passed=sum(bool(item.get("qwen38_vvv_passed")) for item in results),
            teacher_tokens=total_tokens,
        )
    bar.close()

    task_rows = [_flatten_task(row) for row in results]
    checkpoint_rows = _checkpoint_rows(results)
    _atomic_csv(output / "tables" / "task_results.csv", task_rows)
    _atomic_csv(output / "tables" / "checkpoint_results.csv", checkpoint_rows)
    summary = _summarize(results)
    summary.update(
        {
            "schema_version": "darc-qwen38-vvv40-summary-v1",
            "mode": mode,
            "requested_tasks": len(selected),
            "failed_tasks": failures,
            "elapsed_seconds": time.perf_counter() - started,
            "selection_seed": config.selection.seed,
            "thinking_token_budget": config.teacher.thinking_token_budget,
            "max_output_tokens": config.teacher.max_output_tokens,
        }
    )
    _atomic_json(output / "summary.json", summary)
    report = _write_report(output, summary, mode)
    manifest = {
        "schema_version": "darc-qwen38-vvv40-run-v1",
        "status": "complete" if len(results) == len(selected) and failures == 0 else "incomplete",
        "mode": mode,
        "run_id": f"{config.runtime.run_id}-{mode}",
        "selection_seed": config.selection.seed,
        "tasks": len(selected),
        "completed": len(results),
        "failures": failures,
        "student_model": config.student.model_id,
        "teacher_model": config.teacher.model_id,
        "teacher_endpoint": config.runtime.teacher_base_url,
        "student_endpoint": config.runtime.student_base_url,
        "action_sequence": "VVV",
        "thinking_token_budget": config.teacher.thinking_token_budget,
        "hidden_feedback_exposed": False,
        "report": str(report),
    }
    _atomic_json(output / "run_manifest.json", manifest)
    if manifest["status"] != "complete":
        raise RuntimeError(
            f"VVV40 {mode} incomplete: {len(results)}/{len(selected)} tasks; "
            f"inspect {output / 'errors.jsonl'} and resume"
        )
    return {"summary": summary, "report": str(report), "output": str(output)}
