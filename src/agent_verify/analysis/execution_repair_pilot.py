"""Execution-guided one-call Teacher repair pilot on frozen Level-2 train tasks."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from pydantic import Field, field_validator, model_validator
from tqdm import tqdm

from agent_verify.analysis.cbv_v01_experiment import (
    ALL_SEQUENCES,
    load_frozen_partition,
    task_trajectories,
)
from agent_verify.budget import BudgetManager
from agent_verify.cbv_bench.bigcodebench import load_bigcodebench_candidates
from agent_verify.cbv_bench.config import load_cbv_config
from agent_verify.config import StrictModel
from agent_verify.dataset import read_raw_humaneval
from agent_verify.evaluation.bigcodebench_runner import (
    BigCodeBenchEvaluatorError,
    BigCodeBenchRunner,
)
from agent_verify.evaluation.humaneval_runner import HumanEvalRunner, SandboxUnavailable
from agent_verify.evaluation.livecodebench_runner import LiveCodeBenchRunner
from agent_verify.livecodebench_dataset import (
    load_source_rows,
    row_to_private_case,
)
from agent_verify.llm import LLMClient
from agent_verify.logging import JsonlStore, canonical_json, content_id
from agent_verify.schemas import (
    InvocationContext,
    Message,
    PrivateEvaluationCase,
    PublicTask,
    utc_now,
)

DATASETS = ("humaneval", "livecodebench", "bigcodebench")
FORBIDDEN_TEACHER_FIELDS = (
    "private_test_cases",
    "hidden_test",
    "canonical_solution",
    "hidden evaluation",
)


class DataSettings(StrictModel):
    frozen_root: Path
    split: Literal["train"] = "train"
    level: Literal[2] = 2
    sequences: list[str]


class SelectionSettings(StrictModel):
    seed: int
    total_tasks: int = Field(ge=3)
    tasks_per_dataset: dict[str, int]
    require_all_sequences_failed: bool = True
    reference_manifest: Path | None = None
    required_manifest: Path | None = None
    exclude_manifest: Path | None = None
    # Optional infrastructure follow-up subset. The full deterministic selection
    # is reconstructed and validated against reference_manifest before filtering.
    explicit_task_keys: list[str] | None = None


class RuntimeSettings(StrictModel):
    cbv_config: Path
    output_root: Path
    public_timeout_seconds: int = Field(ge=1)
    public_memory_mb: int = Field(ge=64)
    public_max_processes: int = Field(ge=1)
    max_feedback_characters: int = Field(ge=500)
    max_artifact_characters: int = Field(ge=500)
    progress: bool = True
    run_id: str = "darc_execution_repair_pilot"
    include_artifact_context: bool = True
    include_vvv_final_code: bool = True
    include_public_feedback: bool = True
    reference_public_feedback: Path | None = None
    comparison_task_results: Path | None = None
    comparison_baseline_label: str = "Variant A"
    candidate_label: str = "Variant B"


class TeacherSettings(StrictModel):
    prompt_version: str
    exactly_one_call: Literal[True] = True
    retries: Literal[0] = 0
    inference_seed: int
    max_output_tokens: int = Field(default=3200, ge=512, le=8192)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    thinking_enabled: bool = False
    reasoning_parser: str | None = None
    thinking_token_budget: int | None = Field(default=None, ge=0)
    reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh", "max"
    ] | None = None
    include_reasoning: bool = True
    response_format: Literal["json_schema", "json_object", "none"] | None = None
    model_id: str | None = None
    base_url: str | None = None
    system_prompt_path: Path | None = None

    @model_validator(mode="after")
    def valid_reasoning_control(self) -> TeacherSettings:
        if self.thinking_token_budget is not None and self.reasoning_effort is not None:
            raise ValueError(
                "thinking_token_budget and reasoning_effort are alternative controls; "
                "configure exactly one"
            )
        if self.thinking_token_budget is not None and not self.thinking_enabled:
            raise ValueError("thinking_token_budget requires thinking_enabled=true")
        return self


class SmokeSettings(StrictModel):
    tasks_per_dataset: int = Field(default=1, ge=1)
    total_tasks: int | None = Field(default=None, ge=1)


class PilotConfig(StrictModel):
    schema_version: Literal[
        "darc-execution-repair-pilot-v1",
        "darc-execution-repair-pilot-v2",
    ]
    data: DataSettings
    selection: SelectionSettings
    runtime: RuntimeSettings
    teacher: TeacherSettings
    smoke: SmokeSettings


class RepairResponse(StrictModel):
    strategy: Literal["targeted_correction", "complete_rewrite"] = "targeted_correction"
    code_lines: list[str] = Field(min_length=1)
    explanation: str = "Teacher returned code without optional descriptive metadata."

    @field_validator("code_lines")
    @classmethod
    def validate_code_lines(cls, value: list[str]) -> list[str]:
        """Keep leading whitespace while enforcing one source line per JSON item."""
        if any("\n" in line or "\r" in line for line in value):
            raise ValueError("each code_lines item must contain exactly one source line")
        if any(line.strip().startswith("```") for line in value):
            raise ValueError("code_lines must not contain Markdown fences")
        if not any(line.strip() for line in value):
            raise ValueError("code_lines must contain executable source")
        return value

    @model_validator(mode="after")
    def validate_python_syntax(self) -> RepairResponse:
        try:
            ast.parse(self.repaired_code)
        except (SyntaxError, ValueError) as exc:
            location = ""
            if isinstance(exc, SyntaxError):
                location = f" at line {exc.lineno}, column {exc.offset}"
            raise ValueError(f"repaired code is not valid Python{location}: {exc.msg}") from exc
        return self

    @property
    def repaired_code(self) -> str:
        """Reconstruct source without stripping indentation or blank lines."""
        return "\n".join(self.code_lines).rstrip("\n") + "\n"


def salvage_code_lines_response(raw_output: str) -> RepairResponse:
    """Recover a complete code-only response without relaxing the normal API contract.

    This helper is intentionally offline-only.  It accepts an already persisted raw
    response whose JSON object contains ``code_lines`` but omitted the descriptive
    ``strategy``/``explanation`` fields.  The reconstructed object still passes the
    same one-line transport checks and ``ast.parse`` syntax gate as a normal response.
    It must never be used to recover a truncated response or to trigger another call.
    """
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"raw response is not complete JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("raw response must be a JSON object")
    if set(payload).difference({"strategy", "code_lines", "explanation"}):
        raise ValueError("raw response contains unexpected fields")
    if "code_lines" not in payload:
        raise ValueError("raw response does not contain code_lines")
    return RepairResponse(
        strategy=payload.get("strategy", "targeted_correction"),
        code_lines=payload["code_lines"],
        explanation=payload.get(
            "explanation",
            "Offline salvage: Teacher omitted optional descriptive metadata.",
        ),
    )


def load_pilot_config(path: str | Path) -> PilotConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return PilotConfig.model_validate(payload)


def project_path(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _stable_order(keys: list[str], seed: int, dataset: str) -> list[str]:
    return sorted(
        keys,
        key=lambda key: hashlib.sha256(f"repair:{seed}:{dataset}:{key}".encode()).hexdigest(),
    )


def select_unsolved_tasks(
    groups: dict[str, Any], tasks_per_dataset: dict[str, int], seed: int
) -> list[dict[str, Any]]:
    """Select deterministic train-only strata whose eight observed paths all fail."""
    candidates: dict[str, list[str]] = defaultdict(list)
    for key, group in groups.items():
        if set(group.rows) != set(ALL_SEQUENCES):
            continue
        if all(not bool(group.rows[sequence].get("passed")) for sequence in ALL_SEQUENCES):
            candidates[group.task.dataset].append(key)
    selected: list[dict[str, Any]] = []
    for dataset in DATASETS:
        count = int(tasks_per_dataset.get(dataset, 0))
        ordered = _stable_order(candidates.get(dataset, []), seed, dataset)
        if len(ordered) < count:
            raise ValueError(
                f"{dataset}: requested {count} all-eight-unsolved train tasks, found {len(ordered)}"
            )
        for rank, key in enumerate(ordered[:count]):
            group = groups[key]
            vvv = group.rows["VVV"]
            selected.append(
                {
                    "dataset": dataset,
                    "task_key": key,
                    "task_id": group.task.task_id,
                    "stratum_rank": rank,
                    "sss_passed": bool(group.rows["SSS"].get("passed")),
                    "vvv_passed": bool(vvv.get("passed")),
                    "vvv_trajectory_id": str(vvv.get("trajectory_id", "")),
                    "vvv_code_sha256": hashlib.sha256(
                        str(vvv.get("final_code", "")).encode()
                    ).hexdigest(),
                }
            )
    if len(selected) != sum(tasks_per_dataset.values()):
        raise AssertionError("Stratified selection count mismatch")
    return selected


def _selection_payload(config: PilotConfig, selected: list[dict[str, Any]]) -> dict[str, Any]:
    core = {
        "schema_version": "darc-execution-repair-selection-v1",
        "split": "train",
        "selection_seed": config.selection.seed,
        "selection_protocol": "all-eight-observed-trajectories-failed; stable hash within dataset",
        "tasks": selected,
    }
    return {**core, "selection_sha256": hashlib.sha256(canonical_json(core).encode()).hexdigest()}


def write_or_validate_selection(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Existing selection manifest differs: {path}")
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_reference_selection(
    selected: list[dict[str, Any]], reference_path: Path
) -> None:
    """Require an exact task/artifact match with a prior registered pilot."""
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = reference.get("tasks")
    if not isinstance(expected, list):
        raise ValueError(f"Reference selection has no task list: {reference_path}")
    fields = (
        "dataset",
        "task_key",
        "task_id",
        "stratum_rank",
        "sss_passed",
        "vvv_passed",
        "vvv_trajectory_id",
        "vvv_code_sha256",
    )
    projection = lambda rows: [  # noqa: E731 - compact deterministic projection
        {field: row.get(field) for field in fields} for row in rows
    ]
    if projection(selected) != projection(expected):
        raise ValueError(
            "Selected tasks/artifacts differ from the registered reference manifest: "
            f"{reference_path}"
        )


def validate_required_selection(
    selected: list[dict[str, Any]], required_path: Path
) -> None:
    """Require every registered task/artifact to occur in a larger selection."""
    required = json.loads(required_path.read_text(encoding="utf-8")).get("tasks")
    if not isinstance(required, list):
        raise ValueError(f"Required selection has no task list: {required_path}")
    fields = (
        "dataset",
        "task_key",
        "task_id",
        "stratum_rank",
        "sss_passed",
        "vvv_passed",
        "vvv_trajectory_id",
        "vvv_code_sha256",
    )
    project = lambda row: tuple(row.get(field) for field in fields)  # noqa: E731
    available = {project(row) for row in selected}
    missing = [row.get("task_key") for row in required if project(row) not in available]
    if missing:
        raise ValueError(
            "Larger selection does not preserve every required task/artifact: "
            f"{sorted(missing)}"
        )


def _json_value(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return value


def load_livecodebench_public_tests(task_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Return a public-only projection; private test fields are never retained."""
    rows, _ = load_source_rows()
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = f"LiveCodeBench/{row['question_id']}"
        if task_id not in task_ids:
            continue
        metadata = _json_value(row.get("metadata"), {})
        output[task_id] = {
            "public_tests": list(_json_value(row.get("public_test_cases"), [])),
            "fn_name": metadata.get("func_name"),
        }
    missing = task_ids.difference(output)
    if missing:
        raise ValueError(f"Missing LiveCodeBench public rows: {sorted(missing)}")
    return output


def _set_public_limits(memory_mb: int, timeout_seconds: int, max_processes: int) -> None:
    memory = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (timeout_seconds, timeout_seconds + 1))
    # RLIMIT_NPROC is user-wide and counts threads, so setting it to one breaks
    # trusted numerical/plotting imports before seccomp is active. Preserve a
    # small thread allowance above the user's current count; seccomp separately
    # blocks fork/vfork and process-form clone in the worker.
    current_tasks = 0
    uid = os.getuid()
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = status_path.read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        owner = next((line for line in fields if line.startswith("Uid:")), "")
        if not owner or int(owner.split()[1]) != uid:
            continue
        threads = next((line for line in fields if line.startswith("Threads:")), "Threads:\t1")
        current_tasks += int(threads.split()[1])
    process_limit = current_tasks + max(16, max_processes)
    _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if hard != resource.RLIM_INFINITY:
        process_limit = min(process_limit, hard)
    resource.setrlimit(resource.RLIMIT_NPROC, (process_limit, process_limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))


def run_public_feedback(
    *,
    root: Path,
    config: PilotConfig,
    dataset: str,
    task: Any,
    code: str,
    public_projection: dict[str, Any] | None,
    official_repo: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset": dataset,
        "task_id": task.task_id,
        "prompt": task.prompt,
        "entry_point": task.entry_point,
        "code": code,
    }
    if dataset == "livecodebench":
        payload.update(public_projection or {})
        payload["official_repo"] = str(official_repo)
        payload["per_test_timeout"] = min(6, config.runtime.public_timeout_seconds)
    worker = root / "src/agent_verify/evaluation/execution_repair_worker.py"
    # Matplotlib and similar libraries legitimately need a cache directory.
    # Landlock still denies writes everywhere except this disposable scratch root.
    with tempfile.TemporaryDirectory(prefix="darc-public-work-") as scratch_name:
        scratch = Path(scratch_name).resolve()
        payload["sandbox_work_dir"] = str(scratch)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="darc-public-repair-",
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False)
            payload_path = Path(handle.name)
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(worker), str(payload_path)],
                capture_output=True,
                text=True,
                timeout=config.runtime.public_timeout_seconds + 8,
                check=False,
                preexec_fn=lambda: _set_public_limits(
                    config.runtime.public_memory_mb,
                    config.runtime.public_timeout_seconds + 2,
                    config.runtime.public_max_processes,
                ),
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": str(scratch),
                    "TMPDIR": str(scratch),
                    "TMP": str(scratch),
                    "TEMP": str(scratch),
                    "MPLCONFIGDIR": str(scratch),
                    "XDG_CACHE_HOME": str(scratch),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "OMP_NUM_THREADS": "1",
                },
            )
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if completed.returncode != 0 or not lines:
                return {
                    "available": True,
                    "passed": False,
                    "status": f"PUBLIC_WORKER_EXIT:{completed.returncode}",
                    "feedback": (completed.stderr or completed.stdout)[-3000:],
                    "test_count": 0,
                    "latency_seconds": 0.0,
                }
            result = json.loads(lines[-1])
            result["feedback"] = str(result.get("feedback", ""))[
                : config.runtime.max_feedback_characters
            ]
            return result
        finally:
            payload_path.unlink(missing_ok=True)


def _compact_artifacts(vvv_row: dict[str, Any], limit: int) -> str:
    snapshots = list(vvv_row.get("artifact_snapshots", []))
    compact: list[dict[str, Any]] = []
    for checkpoint, artifact in enumerate(snapshots, start=1):
        if not isinstance(artifact, dict):
            continue
        item = dict(artifact)
        if checkpoint in {2, 3}:
            item.pop("code", None)  # final code is supplied once, separately
        compact.append({"checkpoint": checkpoint, "artifact": item})
    text = json.dumps(compact, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def build_repair_messages(
    *,
    task: Any,
    dataset: str,
    vvv_row: dict[str, Any],
    public_feedback: dict[str, Any],
    max_artifact_characters: int = 6000,
    include_artifact_context: bool = True,
    include_vvv_final_code: bool = True,
    include_public_feedback: bool = True,
    system_prompt: str | None = None,
) -> list[Message]:
    default_system = (
        "You are an execution-guided code-repair Teacher in a controlled research pilot. "
        "You receive only the public problem, cached VVV artifacts/final code, and feedback from "
        "public/sample tests or a public smoke check. Hidden tests, hidden correctness labels, the "
        "canonical solution, and hidden-evaluator feedback are unavailable. Diagnose the actual "
        "defect from the supplied evidence. You may make a targeted correction or completely rewrite "
        "the implementation. Preserve the required interface and return self-contained Python code. "
        "Do not hardcode examples. Return only JSON matching the requested schema. The repaired "
        "program MUST be returned as code_lines: a JSON array with exactly one Python source line "
        "per item. Preserve leading spaces in every indented line. Do not use Markdown fences, do "
        "not place multiple source lines in one item, and keep imports concise."
    )
    system = system_prompt.strip() if system_prompt is not None else default_system
    if not system:
        raise ValueError("Teacher system prompt must not be empty")
    artifact_context = ""
    if include_artifact_context:
        artifact_context = (
            "\n\nCACHED VVV ARTIFACT CONTEXT:\n"
            f"{_compact_artifacts(vvv_row, max_artifact_characters)}"
        )
    sections = [
        f"TASK_ID: {task.task_id}\nBENCHMARK: {dataset}\nENTRY_POINT: "
        f"{task.entry_point or 'stdin/stdout program'}",
        f"PUBLIC PROBLEM:\n{task.prompt}",
    ]
    if artifact_context:
        sections.append(artifact_context.strip())
    if include_vvv_final_code:
        sections.append(
            f"CACHED VVV FINAL CODE:\n```python\n{vvv_row.get('final_code', '')}\n```"
        )
    if include_public_feedback:
        sections.append(
            f"PUBLIC EXECUTION STATUS: {public_feedback.get('status')}\n"
            f"PUBLIC EXECUTION FEEDBACK:\n{public_feedback.get('feedback', '')}"
        )
    user = "\n\n".join(sections)
    messages = [Message(role="system", content=system), Message(role="user", content=user)]
    assert_teacher_prompt_is_public(messages)
    return messages


PUBLIC_FEEDBACK_FIELDS = {
    "available",
    "dataset",
    "failed_count",
    "feedback",
    "latency_seconds",
    "passed",
    "status",
    "task_key",
    "test_count",
}


def load_reference_public_feedback(
    path: Path, selected: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], str]:
    """Load a public-only feedback cache and lock it to the selected task set."""
    rows: dict[str, dict[str, Any]] = {}
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        extra = set(row).difference(PUBLIC_FEEDBACK_FIELDS)
        if extra:
            raise ValueError(
                f"Reference public feedback contains non-public/unknown fields at line "
                f"{line_number}: {sorted(extra)}"
            )
        missing = {"task_key", "dataset", "status", "feedback"}.difference(row)
        if missing:
            raise ValueError(
                f"Reference public feedback is missing fields at line {line_number}: "
                f"{sorted(missing)}"
            )
        key = str(row["task_key"])
        if key in rows:
            raise ValueError(f"Duplicate reference public feedback for {key}")
        rows[key] = row

    expected = {row["task_key"] for row in selected}
    if set(rows) != expected:
        raise ValueError(
            "Reference public feedback task set differs from the locked selection: "
            f"missing={sorted(expected.difference(rows))}, "
            f"extra={sorted(set(rows).difference(expected))}"
        )
    for selection in selected:
        row = rows[selection["task_key"]]
        if row.get("dataset") != selection["dataset"]:
            raise ValueError(
                f"Reference feedback dataset mismatch for {selection['task_key']}: "
                f"{row.get('dataset')} != {selection['dataset']}"
            )
    digest = hashlib.sha256(
        canonical_json([rows[key] for key in sorted(rows)]).encode()
    ).hexdigest()
    return rows, digest


def assert_teacher_prompt_is_public(messages: list[Message]) -> None:
    lowered = "\n".join(message.content for message in messages).lower()
    # The system policy may name forbidden concepts while stating they are unavailable.
    user_text = "\n".join(message.content for message in messages if message.role == "user").lower()
    violations = [field for field in FORBIDDEN_TEACHER_FIELDS if field in user_text]
    if violations:
        raise ValueError(f"Teacher-visible payload contains forbidden fields: {violations}")
    if not lowered.strip():
        raise ValueError("Teacher prompt is empty")


def call_disposition(
    task_key: str,
    ledger_rows: list[dict[str, Any]],
    invocation_rows: list[dict[str, Any]],
) -> Literal["call", "recover", "terminal_failed", "ambiguous"]:
    task_events = [row for row in ledger_rows if row.get("task_key") == task_key]
    invocations = [
        row
        for row in invocation_rows
        if row.get("context", {}).get("task_id") == task_key
        and row.get("context", {}).get("role") == "execution_repair"
    ]
    if invocations:
        return "recover" if invocations[-1].get("parsed_output") else "terminal_failed"
    if any(row.get("status") == "call_started" for row in task_events):
        return "ambiguous"
    return "call"


def classify_response_failure(exc: Exception, invocation: dict[str, Any] | None) -> str:
    """Give response/transport failures explicit diagnostic categories."""
    message = f"{type(exc).__name__}: {exc}".lower()
    if "finish_reason=length" in message or "completion truncated" in message:
        return "response_truncated"
    if "reasoning content leaked" in message or "after reasoning separation" in message:
        return "response_reasoning_separation_failure"
    if "unterminated string" in message or "jsondecodeerror" in message:
        if invocation and int(invocation.get("output_tokens", 0)) >= int(
            invocation.get("llm_config", {}).get("max_output_tokens", 0)
        ):
            return "response_truncated_invalid_json"
        return "response_invalid_json"
    if "not valid python" in message or "syntax" in message or "indent" in message:
        return "response_invalid_python"
    if "validationerror" in message:
        return "response_schema_validation"
    return "teacher_invocation_failure"


def _task_public(task_row: Any) -> PublicTask:
    public = dict(task_row.get("public_task", {}))
    benchmark = public.get("benchmark")
    if benchmark == "livecodebench":
        from agent_verify.schemas import LiveCodeBenchPublicTask

        return LiveCodeBenchPublicTask.model_validate(public)
    if benchmark == "bigcodebench":
        from agent_verify.schemas import BigCodeBenchPublicTask

        return BigCodeBenchPublicTask.model_validate(public)
    return PublicTask.model_validate(public)


def load_raw_frozen_task_rows(frozen_root: Path, split: str) -> dict[str, dict[str, Any]]:
    """Load raw public task records from the same explicit frozen partition."""
    task_path = frozen_root / split / "tasks.jsonl"
    rows = JsonlStore(task_path).read_all()
    if not rows:
        raise FileNotFoundError(f"Frozen task partition is empty or missing: {task_path}")
    output = {
        f"{str(row['dataset']).lower()}::{row['task_id']}": row
        for row in rows
    }
    if len(output) != len(rows):
        raise ValueError(f"Duplicate frozen task keys in {task_path}")
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _teacher_ready(model: Any, provider: Any) -> dict[str, Any]:
    base_url = (model.base_url or provider.base_url).rstrip("/")
    api_key_env = model.api_key_env or provider.api_key_env
    api_key = os.environ.get(api_key_env, "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = httpx.get(base_url + "/models", headers=headers, timeout=5.0)
    response.raise_for_status()
    body = response.json()
    ids = [str(item.get("id")) for item in body.get("data", [])]
    return {"base_url": base_url, "expected_model": model.model_id, "models": ids,
            "ready": model.model_id in ids}


def _pilot_teacher_model(config: PilotConfig, cbv: Any) -> Any:
    updates: dict[str, Any] = {
        "prompt_version": config.teacher.prompt_version,
        "max_output_tokens": config.teacher.max_output_tokens,
    }
    if config.teacher.model_id:
        updates["model_id"] = config.teacher.model_id
    if config.teacher.base_url:
        updates["base_url"] = config.teacher.base_url
    if getattr(config.teacher, "temperature", None) is not None:
        updates["temperature"] = config.teacher.temperature
    if getattr(config.teacher, "top_p", None) is not None:
        updates["top_p"] = config.teacher.top_p
    if getattr(config.teacher, "top_k", None) is not None:
        updates["top_k"] = config.teacher.top_k
    return cbv.models.teacher.model_copy(update=updates)


def _pilot_teacher_and_readiness(config: PilotConfig, cbv: Any) -> tuple[Any, dict[str, Any]]:
    """Resolve the effective Pilot Teacher before checking the served model ID."""
    teacher_model = _pilot_teacher_model(config, cbv)
    return teacher_model, _teacher_ready(teacher_model, cbv.provider)


def _teacher_request_overrides(settings: TeacherSettings) -> dict[str, Any]:
    """Build explicit reasoning controls; absent keys are never sent silently."""
    overrides: dict[str, Any] = {"include_reasoning": settings.include_reasoning}
    if settings.thinking_token_budget is not None:
        overrides["thinking_token_budget"] = settings.thinking_token_budget
    elif settings.reasoning_effort is not None:
        overrides["reasoning_effort"] = settings.reasoning_effort
    return overrides


def check_teacher(config_path: str | Path, root: Path) -> dict[str, Any]:
    config = load_pilot_config(config_path)
    cbv = load_cbv_config(project_path(root, config.runtime.cbv_config))
    _, ready = _pilot_teacher_and_readiness(config, cbv)
    return ready


def _load_private_cases(root: Path, cbv: Any, selected: list[dict[str, Any]]) -> dict[str, Any]:
    """Load evaluator-only cases after every Teacher call has reached a terminal state."""
    wanted = {row["dataset"]: set() for row in selected}
    for row in selected:
        wanted.setdefault(row["dataset"], set()).add(row["task_id"])
    private: dict[str, Any] = {}
    human_spec = next(item for item in cbv.datasets if item.name == "humaneval")
    raw_human = read_raw_humaneval(project_path(root, human_spec.data_path))
    for task_id in wanted.get("humaneval", set()):
        row = raw_human[task_id]
        private[f"humaneval::{task_id}"] = PrivateEvaluationCase(
            task_id=task_id, test=row["test"], entry_point=row["entry_point"]
        )
    if wanted.get("livecodebench"):
        rows, _ = load_source_rows()
        by_id = {f"LiveCodeBench/{row['question_id']}": row for row in rows}
        for task_id in wanted["livecodebench"]:
            private[f"livecodebench::{task_id}"] = row_to_private_case(by_id[task_id])
    if wanted.get("bigcodebench"):
        spec = next(item for item in cbv.datasets if item.name == "bigcodebench")
        candidates, _ = load_bigcodebench_candidates(spec)
        by_id = {task.task_id: case for task, case, _ in candidates}
        for task_id in wanted["bigcodebench"]:
            private[f"bigcodebench::{task_id}"] = by_id[task_id]
    missing = {row["task_key"] for row in selected}.difference(private)
    if missing:
        raise ValueError(f"Missing private evaluator cases: {sorted(missing)}")
    return private


def _evaluators(root: Path, cbv: Any, evaluation_store: JsonlStore) -> dict[str, Any]:
    lcb_spec = next(item for item in cbv.datasets if item.name == "livecodebench")
    return {
        "humaneval": HumanEvalRunner(cbv.evaluation, evaluation_store),
        "livecodebench": LiveCodeBenchRunner(
            cbv.evaluation, evaluation_store, project_path(root, lcb_spec.official_evaluator_path)
        ),
        "bigcodebench": BigCodeBenchRunner(cbv.evaluation, evaluation_store),
    }


def _csv_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def paired_variant_rows(
    baseline_path: Path,
    selected: list[dict[str, Any]],
    variant_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build paired A/B outcomes after the candidate Teacher phase is complete."""
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        baseline_rows = {row["task_key"]: row for row in csv.DictReader(handle)}
    variant_rows = {row["task_key"]: row for row in variant_results}
    expected = {row["task_key"] for row in selected}
    baseline_keys = set(baseline_rows)
    if not (expected.issubset(baseline_keys) or baseline_keys.issubset(expected)):
        raise ValueError("Baseline and candidate selections are not nested")
    paired_keys = expected.intersection(baseline_keys)
    if not paired_keys:
        raise ValueError("Baseline and candidate have no shared tasks")
    if set(variant_rows) != expected:
        raise ValueError("Candidate task results do not match its locked selection")

    paired: list[dict[str, Any]] = []
    for selection in selected:
        key = selection["task_key"]
        if key not in paired_keys:
            continue
        baseline = baseline_rows[key]
        variant = variant_rows[key]
        a_pass = _csv_bool(baseline.get("repaired_passed"))
        b_pass = bool(variant.get("repaired_passed"))
        if b_pass and not a_pass:
            outcome = "B_win"
        elif a_pass and not b_pass:
            outcome = "A_win"
        elif a_pass and b_pass:
            outcome = "tie_rescued"
        else:
            outcome = "tie_failed"
        paired.append(
            {
                "task_key": key,
                "dataset": selection["dataset"],
                "task_id": selection["task_id"],
                "paired_outcome": outcome,
                "variant_a_rescued": a_pass,
                "variant_b_rescued": b_pass,
                "variant_a_teacher_status": baseline.get("teacher_status", ""),
                "variant_b_teacher_status": variant.get("teacher_status", ""),
                "variant_a_response_valid": baseline.get("teacher_status") == "completed",
                "variant_b_response_valid": variant.get("teacher_status") == "completed",
                "variant_a_teacher_tokens": int(baseline.get("teacher_total_tokens") or 0),
                "variant_b_teacher_tokens": int(variant.get("teacher_total_tokens") or 0),
                "variant_a_latency_seconds": float(
                    baseline.get("teacher_latency_seconds") or 0.0
                ),
                "variant_b_latency_seconds": float(
                    variant.get("teacher_latency_seconds") or 0.0
                ),
            }
        )
    return paired


def _exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _report(
    output_root: Path,
    selected: list[dict[str, Any]],
    results: list[dict[str, Any]],
    teacher_model_id: str,
    *,
    include_artifact_context: bool,
    include_vvv_final_code: bool,
    include_public_feedback: bool,
    comparison: list[dict[str, Any]] | None = None,
    comparison_baseline_label: str = "Variant A",
    candidate_label: str = "Variant B",
) -> None:
    table_root = output_root / "tables"
    _write_csv(table_root / "task_results.csv", results)
    summary: list[dict[str, Any]] = []
    for dataset in (*DATASETS, "combined"):
        rows = results if dataset == "combined" else [row for row in results if row["dataset"] == dataset]
        if not rows:
            continue
        summary.append(
            {
                "dataset": dataset,
                "tasks": len(rows),
                "completed_repairs": sum(row.get("teacher_status") == "completed" for row in rows),
                "syntax_valid_repairs": sum(bool(row.get("syntax_validated")) for row in rows),
                "response_failures": sum(row.get("teacher_status") != "completed" for row in rows),
                "evaluated_tasks": sum(
                    "repaired_passed" in row and not row.get("evaluation_infrastructure_error")
                    for row in rows
                ),
                "infrastructure_errors": sum(
                    bool(row.get("evaluation_infrastructure_error")) for row in rows
                ),
                "newly_rescued": sum(bool(row.get("repaired_passed")) for row in rows),
                "success_rate": sum(bool(row.get("repaired_passed")) for row in rows) / len(rows),
                "teacher_input_tokens": sum(int(row.get("teacher_input_tokens", 0)) for row in rows),
                "teacher_output_tokens": sum(int(row.get("teacher_output_tokens", 0)) for row in rows),
                "teacher_total_tokens": sum(int(row.get("teacher_total_tokens", 0)) for row in rows),
                "teacher_actual_total_tokens": sum(
                    int(row.get("teacher_actual_total_tokens", row.get("teacher_total_tokens", 0)))
                    for row in rows
                ),
                "teacher_latency_seconds": sum(float(row.get("teacher_latency_seconds", 0)) for row in rows),
                "reasoning_responses": sum(
                    bool(row.get("reasoning_content_present")) for row in rows
                ),
                "reasoning_tokens": sum(int(row.get("reasoning_tokens", 0)) for row in rows),
                "reasoning_content_characters": sum(
                    int(row.get("reasoning_content_characters", 0)) for row in rows
                ),
                "truncated_responses": sum(
                    str(row.get("failure_category", "")).startswith("response_truncated")
                    for row in rows
                ),
                "evaluation_latency_seconds": sum(
                    float(row.get("evaluation_latency_seconds", 0)) for row in rows
                ),
            }
        )
    _write_csv(table_root / "dataset_summary.csv", summary)
    if comparison is not None:
        _write_csv(table_root / "paired_variant_comparison.csv", comparison)
    combined = next((row for row in summary if row["dataset"] == "combined"), None)
    headline = ""
    if combined is not None:
        headline = (
            f"**Headline:** {combined['newly_rescued']}/{combined['tasks']} tasks rescued "
            f"({100 * combined['success_rate']:.2f}%)."
        )
    visible_inputs = ["the public problem"]
    if include_vvv_final_code:
        visible_inputs.append("cached VVV final code")
    if include_public_feedback:
        visible_inputs.append("public/sample execution feedback")
    if include_artifact_context:
        visible_inputs.append("Planner/Coder/Reviewer artifacts")
    teacher_inputs = ", ".join(visible_inputs)
    lines = [
        "# Execution-Guided Teacher Repair Pilot",
        "",
        "This pilot uses frozen Level-2 **training tasks only**. The Teacher received only "
        f"{teacher_inputs}. Hidden "
        "tests and hidden evaluation feedback were loaded only after all Teacher calls became "
        "terminal. No Student or DARC training was run.",
        "",
        f"**Teacher:** `{teacher_model_id}`.",
        "",
        "## Result",
        "",
        headline,
        "",
        "| Dataset | Tasks | Syntax-valid | Evaluated | Rescued | ITT success | Teacher tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['tasks']} | {row['syntax_valid_repairs']} | "
            f"{row['evaluated_tasks']} | {row['newly_rescued']} | "
            f"{100 * row['success_rate']:.2f}% | {row['teacher_actual_total_tokens']:,} |"
        )
    if combined is not None and (
        combined["reasoning_responses"] or combined["truncated_responses"]
    ):
        lines.extend(
            [
                "",
                f"- Parsed reasoning was present for {combined['reasoning_responses']}/"
                f"{combined['tasks']} responses and was kept separate from final JSON.",
                f"- Reported reasoning tokens: {combined['reasoning_tokens']:,}; "
                f"reasoning characters (content not stored): "
                f"{combined['reasoning_content_characters']:,}.",
                f"- Actual API tokens (prompt + completion, including failed calls): "
                f"{combined['teacher_actual_total_tokens']:,}.",
                f"- Truncated responses rejected before evaluation: "
                f"{combined['truncated_responses']}.",
            ]
        )
    if comparison is not None:
        wins = sum(row["paired_outcome"] == "B_win" for row in comparison)
        losses = sum(row["paired_outcome"] == "A_win" for row in comparison)
        ties_rescued = sum(row["paired_outcome"] == "tie_rescued" for row in comparison)
        a_tokens = sum(int(row["variant_a_teacher_tokens"]) for row in comparison)
        b_tokens = sum(int(row["variant_b_teacher_tokens"]) for row in comparison)
        a_latency = sum(float(row["variant_a_latency_seconds"]) for row in comparison)
        b_latency = sum(float(row["variant_b_latency_seconds"]) for row in comparison)
        lines.extend(
            [
                "",
                f"## Paired comparison: {comparison_baseline_label} vs {candidate_label}",
                "",
                f"{comparison_baseline_label} outcomes were loaded only after every "
                f"{candidate_label} Teacher call became "
                "terminal and hidden evaluation completed; they were never inference inputs.",
                "",
                "| Dataset | A/B rescued | B wins/losses | A/B valid | A/B tokens | A/B latency (s) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for dataset in (*DATASETS, "combined"):
            rows = comparison if dataset == "combined" else [
                row for row in comparison if row["dataset"] == dataset
            ]
            lines.append(
                f"| {dataset} | {sum(bool(row['variant_a_rescued']) for row in rows)}/"
                f"{sum(bool(row['variant_b_rescued']) for row in rows)} | "
                f"{sum(row['paired_outcome'] == 'B_win' for row in rows)}/"
                f"{sum(row['paired_outcome'] == 'A_win' for row in rows)} | "
                f"{sum(bool(row['variant_a_response_valid']) for row in rows)}/"
                f"{sum(bool(row['variant_b_response_valid']) for row in rows)} | "
                f"{sum(int(row['variant_a_teacher_tokens']) for row in rows):,}/"
                f"{sum(int(row['variant_b_teacher_tokens']) for row in rows):,} | "
                f"{sum(float(row['variant_a_latency_seconds']) for row in rows):.2f}/"
                f"{sum(float(row['variant_b_latency_seconds']) for row in rows):.2f} |"
            )
        b_rescued = [row["task_id"] for row in comparison if row["variant_b_rescued"]]
        b_win_tasks = [row["task_id"] for row in comparison if row["paired_outcome"] == "B_win"]
        b_loss_tasks = [row["task_id"] for row in comparison if row["paired_outcome"] == "A_win"]
        lines.extend(
            [
                "",
                f"- Paired outcome: B wins={wins}, B losses={losses}, both rescue={ties_rescued}; "
                f"exact McNemar p={_exact_mcnemar_p(wins, losses):.4f}.",
                f"- Response validity: A={sum(bool(row['variant_a_response_valid']) for row in comparison)}/"
                f"{len(comparison)}, B={sum(bool(row['variant_b_response_valid']) for row in comparison)}/"
                f"{len(comparison)}.",
                f"- Teacher tokens: A={a_tokens:,}, B={b_tokens:,} ({b_tokens - a_tokens:+,}).",
                f"- Teacher latency: A={a_latency:.2f}s, B={b_latency:.2f}s "
                f"({b_latency - a_latency:+.2f}s).",
                f"- {candidate_label} rescued tasks: "
                f"{', '.join(b_rescued) if b_rescued else 'none'}.",
                f"- Candidate-only wins: {', '.join(b_win_tasks) if b_win_tasks else 'none'}.",
                f"- Candidate losses versus baseline: "
                f"{', '.join(b_loss_tasks) if b_loss_tasks else 'none'}.",
            ]
        )
    lines.extend(
        [
            "",
            "Every selected task had cached `SSS=0` and `VVV=0`; therefore every repaired pass is "
            "a newly rescued task. Repaired correctness is an evaluator-only outcome and is never "
            "Teacher-visible.",
            "",
            "The primary success rate is intention-to-treat (all selected tasks in the denominator). "
            "Responses rejected by the syntax gate are response failures, not model-incorrect hidden "
            "evaluations. Evaluator infrastructure errors are reported separately and are never "
            "converted into benchmark failures.",
            "",
            "## Integrity",
            "",
            f"- Selected tasks: {len(selected)}; result rows: {len(results)}.",
            "- Teacher attempts per selected task are bounded at one; failed or ambiguous calls are "
            "terminal and are never retried automatically.",
            "- Frozen validation/test partitions were not loaded.",
            "- Original CBV trajectories and DARC checkpoints were not modified.",
            "",
            "## Artifacts",
            "",
            "- `selection/selected_tasks.json`",
            "- `run/public_feedback.jsonl`",
            "- `run/teacher_invocations.jsonl`",
            "- `run/hidden_evaluations.jsonl`",
            "- `tables/task_results.csv` and `dataset_summary.csv`",
            "- `tables/paired_variant_comparison.csv` (when a cached baseline is configured)",
        ]
    )
    report_path = output_root / "report/darc_execution_repair_pilot.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pilot(
    config_path: str | Path,
    *,
    root: Path,
    mode: Literal["dry-run", "smoke", "full"],
    resume: bool,
) -> Path:
    config = load_pilot_config(config_path)
    frozen_root = project_path(root, config.data.frozen_root)
    tasks, trajectory_rows = load_frozen_partition(frozen_root, "train")
    groups = task_trajectories(tasks, trajectory_rows, level=2)
    selected_pool = select_unsolved_tasks(
        groups, config.selection.tasks_per_dataset, config.selection.seed
    )
    if config.selection.reference_manifest is not None:
        validate_reference_selection(
            selected_pool,
            project_path(root, config.selection.reference_manifest),
        )
    if config.selection.required_manifest is not None:
        validate_required_selection(
            selected_pool,
            project_path(root, config.selection.required_manifest),
        )
    selected = selected_pool
    if config.selection.explicit_task_keys is not None:
        requested = config.selection.explicit_task_keys
        if len(requested) != len(set(requested)):
            raise ValueError("selection.explicit_task_keys contains duplicates")
        by_key = {row["task_key"]: row for row in selected_pool}
        missing = sorted(set(requested).difference(by_key))
        if missing:
            raise ValueError(f"Explicit follow-up tasks are outside locked selection: {missing}")
        selected = [by_key[key] for key in requested]
    if config.selection.exclude_manifest is not None:
        exclusion_path = project_path(root, config.selection.exclude_manifest)
        exclusion_rows = json.loads(exclusion_path.read_text(encoding="utf-8")).get("tasks")
        if not isinstance(exclusion_rows, list):
            raise ValueError(f"Exclusion manifest has no task list: {exclusion_path}")
        excluded = {str(row.get("task_key")) for row in exclusion_rows}
        selected_keys = {row["task_key"] for row in selected}
        missing = sorted(excluded.difference(selected_keys))
        if missing:
            raise ValueError(f"Excluded tasks are outside deterministic selection: {missing}")
        selected = [row for row in selected if row["task_key"] not in excluded]
    system_prompt: str | None = None
    system_prompt_sha256: str | None = None
    if config.teacher.system_prompt_path is not None:
        prompt_path = project_path(root, config.teacher.system_prompt_path)
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise ValueError(f"Teacher system prompt is empty: {prompt_path}")
        system_prompt_sha256 = hashlib.sha256(system_prompt.encode()).hexdigest()
    raw_task_rows = load_raw_frozen_task_rows(frozen_root, config.data.split)
    bigcodebench_public_tasks = [
        _task_public(raw_task_rows[row["task_key"]])
        for row in selected
        if row["dataset"] == "bigcodebench"
    ]
    # Dry-run is a real dependency preflight, so an overnight run cannot fail
    # only after some expensive Teacher calls have already completed.
    BigCodeBenchRunner.check_dependencies(bigcodebench_public_tasks)
    reference_feedback: dict[str, dict[str, Any]] | None = None
    reference_feedback_sha256: str | None = None
    if (
        not config.runtime.include_public_feedback
        and config.runtime.reference_public_feedback is not None
    ):
        raise ValueError(
            "reference_public_feedback must be unset when include_public_feedback=false"
        )
    if config.runtime.reference_public_feedback is not None:
        reference_feedback, reference_feedback_sha256 = load_reference_public_feedback(
            project_path(root, config.runtime.reference_public_feedback), selected
        )
    output_root = project_path(root, config.runtime.output_root)
    selection_path = output_root / "selection/selected_tasks.json"
    write_or_validate_selection(selection_path, _selection_payload(config, selected))
    plan = {
        "schema_version": "darc-execution-repair-plan-v1",
        "mode": mode,
        "selected_tasks": len(selected),
        "teacher_calls": len(selected),
        "teacher_calls_per_task": 1,
        "datasets": dict(Counter(row["dataset"] for row in selected)),
        "frozen_split": "train",
        "validation_or_test_loaded": False,
        "include_artifact_context": config.runtime.include_artifact_context,
        "include_vvv_final_code": config.runtime.include_vvv_final_code,
        "include_public_feedback": config.runtime.include_public_feedback,
        "teacher_system_prompt": (
            str(project_path(root, config.teacher.system_prompt_path))
            if config.teacher.system_prompt_path is not None
            else None
        ),
        "teacher_system_prompt_sha256": system_prompt_sha256,
        "thinking_enabled": config.teacher.thinking_enabled,
        "reasoning_parser": config.teacher.reasoning_parser,
        "thinking_token_budget": config.teacher.thinking_token_budget,
        "reasoning_effort": config.teacher.reasoning_effort,
        "include_reasoning": config.teacher.include_reasoning,
        "response_format": config.teacher.response_format,
        "reference_public_feedback": (
            str(project_path(root, config.runtime.reference_public_feedback))
            if config.runtime.reference_public_feedback is not None
            else None
        ),
        "reference_public_feedback_sha256": reference_feedback_sha256,
        "comparison_outcomes_locked_until_post_teacher_phase": True,
    }
    (output_root / "planning").mkdir(parents=True, exist_ok=True)
    (output_root / "planning/plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    if mode == "dry-run":
        return output_root / "planning/plan.json"

    active = selected
    if mode == "smoke":
        if config.smoke.total_tasks is not None:
            active = selected[: config.smoke.total_tasks]
        else:
            active = []
            for dataset in DATASETS:
                active.extend(
                    [row for row in selected if row["dataset"] == dataset][
                        : config.smoke.tasks_per_dataset
                    ]
                )
    # Smoke is a resumable prefix of the full run. Sharing this ledger prevents
    # smoke tasks from receiving a second Teacher call during --full --resume.
    run_root = output_root / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    if not resume and any((run_root / name).exists() for name in (
        "call_ledger.jsonl", "teacher_invocations.jsonl", "task_results.jsonl"
    )):
        raise FileExistsError(f"{run_root} already contains a run; use --resume")

    cbv = load_cbv_config(project_path(root, config.runtime.cbv_config))
    teacher_model, ready = _pilot_teacher_and_readiness(config, cbv)
    if not ready["ready"]:
        raise RuntimeError(f"Teacher endpoint is not ready: {ready}")
    provider_updates: dict[str, Any] = {"max_retries": 0}
    if config.teacher.timeout_seconds is not None:
        provider_updates["timeout_seconds"] = config.teacher.timeout_seconds
    if config.teacher.response_format is not None:
        provider_updates["response_format"] = config.teacher.response_format
    provider = cbv.provider.model_copy(update=provider_updates)
    invocation_store = JsonlStore(run_root / "teacher_invocations.jsonl")
    ledger_store = JsonlStore(run_root / "call_ledger.jsonl")
    feedback_store = JsonlStore(run_root / "public_feedback.jsonl")
    result_store = JsonlStore(run_root / "task_results.jsonl")
    budget = BudgetManager(cbv.budget, provider)
    client = LLMClient(provider, budget, invocation_store)

    public_by_task: dict[str, dict[str, Any]] = {}
    lcb_ids = {
        row["task_id"]
        for row in active
        if row["dataset"] == "livecodebench" and config.runtime.include_public_feedback
    }
    if lcb_ids:
        public_by_task = load_livecodebench_public_tests(lcb_ids)
    lcb_spec = next(item for item in cbv.datasets if item.name == "livecodebench")
    official_repo = project_path(root, lcb_spec.official_evaluator_path)
    existing_feedback = {row["task_key"]: row for row in feedback_store.read_all()}
    existing_results = {row["task_key"]: row for row in result_store.read_all()}

    progress = tqdm(
        active,
        desc="Execution repair · Teacher",
        unit="task",
        disable=not config.runtime.progress,
        dynamic_ncols=True,
    )
    for index, selection in enumerate(progress):
        key = selection["task_key"]
        if key in existing_results and existing_results[key].get("teacher_status") in {
            "completed", "failed", "ambiguous"
        }:
            continue
        group = groups[key]
        public_task = _task_public(raw_task_rows[key])
        feedback = existing_feedback.get(key)
        if not config.runtime.include_public_feedback:
            omitted_feedback = {
                "task_key": key,
                "dataset": selection["dataset"],
                "available": False,
                "passed": None,
                "status": "OMITTED_PROBLEM_ONLY",
                "feedback": "",
                "test_count": 0,
                "failed_count": 0,
                "latency_seconds": 0.0,
            }
            if feedback is not None and feedback != omitted_feedback:
                raise ValueError(f"Resumed omitted feedback marker differs for {key}")
            if feedback is None:
                feedback_store.append(omitted_feedback)
                existing_feedback[key] = omitted_feedback
            feedback = omitted_feedback
        elif reference_feedback is not None:
            locked_feedback = reference_feedback[key]
            if feedback is not None and feedback != locked_feedback:
                raise ValueError(f"Resumed public feedback differs from locked Variant A for {key}")
            if feedback is None:
                feedback_store.append(locked_feedback)
                existing_feedback[key] = locked_feedback
            feedback = locked_feedback
        elif feedback is None:
            body = run_public_feedback(
                root=root,
                config=config,
                dataset=selection["dataset"],
                task=public_task,
                code=str(group.rows["VVV"].get("final_code", "")),
                public_projection=public_by_task.get(selection["task_id"]),
                official_repo=official_repo,
            )
            feedback = {"task_key": key, "dataset": selection["dataset"], **body}
            feedback_store.append(feedback)
            existing_feedback[key] = feedback

        ledger_rows = ledger_store.read_all()
        invocation_rows = invocation_store.read_all()
        disposition = call_disposition(key, ledger_rows, invocation_rows)
        if disposition == "ambiguous":
            row = {
                **selection,
                "teacher_status": "ambiguous",
                "error": "call_started exists without durable invocation; refusing automatic retry",
            }
            result_store.append(row)
            existing_results[key] = row
            continue
        if disposition == "terminal_failed":
            row = {**selection, "teacher_status": "failed", "error": "one call already failed"}
            result_store.append(row)
            existing_results[key] = row
            continue
        if disposition == "recover":
            invocation = [
                row for row in invocation_rows
                if row.get("context", {}).get("task_id") == key
                and row.get("context", {}).get("role") == "execution_repair"
            ][-1]
            repaired = RepairResponse.model_validate(invocation["parsed_output"])
        else:
            call_id = content_id("repair_call", {"task_key": key, "selection": selection_path.name})
            ledger_store.append(
                {"task_key": key, "call_id": call_id, "status": "call_started", "at": utc_now()}
            )
            messages = build_repair_messages(
                task=public_task,
                dataset=selection["dataset"],
                vvv_row=group.rows["VVV"],
                public_feedback=feedback,
                max_artifact_characters=config.runtime.max_artifact_characters,
                include_artifact_context=config.runtime.include_artifact_context,
                include_vvv_final_code=config.runtime.include_vvv_final_code,
                include_public_feedback=config.runtime.include_public_feedback,
                system_prompt=system_prompt,
            )
            cached_vvv_code = str(group.rows["VVV"].get("final_code", ""))
            try:
                llm_result = client.invoke(
                    model=teacher_model,
                    messages=messages,
                    output_schema=RepairResponse,
                    context=InvocationContext(
                        task_id=key,
                        checkpoint=3,
                        role="execution_repair",
                        branch="VVV+E",
                        run_id=config.runtime.run_id,
                        seed=config.teacher.inference_seed + index,
                    ),
                    mock_factory=lambda code=cached_vvv_code: RepairResponse(
                        strategy="targeted_correction",
                        code_lines=code.rstrip("\n").split("\n"),
                        explanation="Mock repair preserves cached VVV code.",
                    ),
                    request_overrides=_teacher_request_overrides(config.teacher),
                )
                repaired = RepairResponse.model_validate(llm_result.value.model_dump())
                ledger_store.append(
                    {"task_key": key, "call_id": call_id, "status": "call_completed",
                     "invocation_id": llm_result.record.invocation_id, "at": utc_now()}
                )
            except Exception as exc:
                invocation = next(
                    (
                        item
                        for item in reversed(invocation_store.read_all())
                        if item.get("context", {}).get("task_id") == key
                        and item.get("context", {}).get("role") == "execution_repair"
                    ),
                    None,
                )
                failure_category = classify_response_failure(exc, invocation)
                invocation_metadata = (
                    invocation.get("attempts", [{}])[-1].get("response_metadata", {})
                    if invocation
                    else {}
                )
                ledger_store.append(
                    {"task_key": key, "call_id": call_id, "status": "call_failed",
                     "failure_category": failure_category,
                     "error": f"{type(exc).__name__}: {exc}", "at": utc_now()}
                )
                row = {
                    **selection,
                    "teacher_status": "failed",
                    "failure_category": failure_category,
                    "error": str(exc),
                    "teacher_input_tokens": int(invocation.get("input_tokens", 0))
                    if invocation
                    else 0,
                    "teacher_output_tokens": int(invocation.get("output_tokens", 0))
                    if invocation
                    else 0,
                    "teacher_total_tokens": (
                        int(invocation.get("input_tokens", 0))
                        + int(invocation.get("output_tokens", 0))
                        if invocation
                        else 0
                    ),
                    "teacher_actual_total_tokens": (
                        int(invocation.get("input_tokens", 0))
                        + int(invocation.get("output_tokens", 0))
                        if invocation
                        else 0
                    ),
                    "teacher_latency_seconds": float(invocation.get("latency_seconds", 0.0))
                    if invocation
                    else 0.0,
                    "invocation_id": invocation.get("invocation_id") if invocation else None,
                    "finish_reason": invocation_metadata.get("finish_reason"),
                    "reasoning_content_present": invocation_metadata.get(
                        "reasoning_content_present", False
                    ),
                    "reasoning_content_characters": int(
                        invocation_metadata.get("reasoning_content_characters") or 0
                    ),
                    "reasoning_tokens": int(invocation_metadata.get("reasoning_tokens") or 0),
                }
                result_store.append(row)
                existing_results[key] = row
                continue
        invocation = [
            row for row in invocation_store.read_all()
            if row.get("context", {}).get("task_id") == key
            and row.get("context", {}).get("role") == "execution_repair"
        ][-1]
        invocation_metadata = invocation.get("attempts", [{}])[-1].get(
            "response_metadata", {}
        )
        row = {
            **selection,
            "teacher_status": "completed",
            "strategy": repaired.strategy,
            "repaired_code": repaired.repaired_code,
            "repaired_code_lines": repaired.code_lines,
            "syntax_validated": True,
            "repair_explanation": repaired.explanation,
            "public_status": feedback.get("status"),
            "public_passed": feedback.get("passed"),
            "teacher_input_tokens": int(invocation.get("input_tokens", 0)),
            "teacher_output_tokens": int(invocation.get("output_tokens", 0)),
            "teacher_total_tokens": int(invocation.get("input_tokens", 0))
            + int(invocation.get("output_tokens", 0)),
            "teacher_actual_total_tokens": int(invocation.get("input_tokens", 0))
            + int(invocation.get("output_tokens", 0)),
            "teacher_latency_seconds": float(invocation.get("latency_seconds", 0.0)),
            "invocation_id": invocation.get("invocation_id"),
            "finish_reason": invocation_metadata.get("finish_reason"),
            "reasoning_content_present": invocation_metadata.get(
                "reasoning_content_present", False
            ),
            "reasoning_content_characters": int(
                invocation_metadata.get("reasoning_content_characters") or 0
            ),
            "reasoning_tokens": int(invocation_metadata.get("reasoning_tokens") or 0),
        }
        result_store.append(row)
        existing_results[key] = row
        progress.set_postfix(
            calls=sum(item.get("teacher_status") == "completed" for item in existing_results.values()),
            tokens=sum(int(item.get("teacher_total_tokens", 0)) for item in existing_results.values()),
        )

    # Strict phase barrier: never load private cases while any selected Teacher call is pending.
    existing_results = {row["task_key"]: row for row in result_store.read_all()}
    pending = [row["task_key"] for row in active if row["task_key"] not in existing_results]
    if pending:
        raise RuntimeError(f"Teacher phase incomplete; hidden evaluation remains locked: {pending}")

    private_cases = _load_private_cases(root, cbv, active)
    evaluation_store = JsonlStore(run_root / "hidden_evaluations.jsonl")
    evaluators = _evaluators(root, cbv, evaluation_store)
    completed_evaluations = {
        row.get("task_key")
        for row in result_store.read_all()
        if "repaired_passed" in row and not row.get("evaluation_infrastructure_error")
    }
    eval_progress = tqdm(
        active,
        desc="Execution repair · hidden evaluation",
        unit="task",
        disable=not config.runtime.progress,
        dynamic_ncols=True,
    )
    for selection in eval_progress:
        key = selection["task_key"]
        if key in completed_evaluations:
            continue
        base = existing_results[key]
        if base.get("teacher_status") != "completed":
            continue
        # Defense in depth: durable rows are syntax-checked again immediately
        # before they can reach any benchmark evaluator.
        try:
            ast.parse(str(base["repaired_code"]))
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError(f"syntax gate failed before evaluation for {key}: {exc}") from exc
        evaluator = evaluators[selection["dataset"]]
        try:
            result = evaluator.evaluate(
                run_id=config.runtime.run_id,
                case=private_cases[key],
                code=str(base["repaired_code"]),
                branch="VVV+execution_repair",
            )
        except (BigCodeBenchEvaluatorError, SandboxUnavailable) as exc:
            result_store.append(
                {
                    **base,
                    "repaired_passed": None,
                    "newly_rescued": None,
                    "evaluation_status": f"INFRASTRUCTURE_ERROR:{type(exc).__name__}",
                    "evaluation_error": str(exc),
                    "evaluation_infrastructure_error": True,
                }
            )
            continue
        result_store.append(
            {
                **base,
                "repaired_passed": bool(result.passed),
                "newly_rescued": bool(result.passed),
                "evaluation_status": result.status,
                "evaluation_latency_seconds": result.latency_seconds,
            }
        )
        eval_progress.set_postfix(
            rescued=sum(
                bool(row.get("repaired_passed")) for row in result_store.read_all()
                if "repaired_passed" in row
            )
        )

    latest: dict[str, dict[str, Any]] = {}
    for row in result_store.read_all():
        latest[row["task_key"]] = row
    final_rows = [latest[row["task_key"]] for row in active]
    if mode == "full":
        comparison = None
        if config.runtime.comparison_task_results is not None:
            # This is intentionally after the Teacher phase barrier and hidden evaluation.
            # Cached Variant A outcomes can never influence Variant B prompts or actions.
            comparison = paired_variant_rows(
                project_path(root, config.runtime.comparison_task_results),
                selected,
                final_rows,
            )
        _report(
            output_root,
            selected,
            final_rows,
            teacher_model.model_id,
            include_artifact_context=config.runtime.include_artifact_context,
            include_vvv_final_code=config.runtime.include_vvv_final_code,
            include_public_feedback=config.runtime.include_public_feedback,
            comparison=comparison,
            comparison_baseline_label=config.runtime.comparison_baseline_label,
            candidate_label=config.runtime.candidate_label,
        )
        manifest = {
            "schema_version": "darc-execution-repair-result-v1",
            "mode": mode,
            "tasks": len(final_rows),
            "split": "train",
            "frozen_validation_test_loaded": False,
            "student_inference": False,
            "teacher_calls_max_per_task": 1,
            "teacher_retries": 0,
            "teacher_model_id": teacher_model.model_id,
            "teacher_prompt_version": teacher_model.prompt_version,
            "teacher_temperature": teacher_model.temperature,
            "teacher_top_p": teacher_model.top_p,
            "teacher_top_k": teacher_model.top_k,
            "teacher_max_output_tokens": teacher_model.max_output_tokens,
            "thinking_enabled": config.teacher.thinking_enabled,
            "reasoning_parser": config.teacher.reasoning_parser,
            "thinking_token_budget": config.teacher.thinking_token_budget,
            "reasoning_effort": config.teacher.reasoning_effort,
            "include_reasoning": config.teacher.include_reasoning,
            "response_format": provider.response_format,
            "teacher_system_prompt_sha256": system_prompt_sha256,
            "include_artifact_context": config.runtime.include_artifact_context,
            "include_vvv_final_code": config.runtime.include_vvv_final_code,
            "include_public_feedback": config.runtime.include_public_feedback,
            "reference_public_feedback_sha256": reference_feedback_sha256,
            "reference_selection_manifest": (
                str(project_path(root, config.selection.reference_manifest))
                if config.selection.reference_manifest is not None
                else None
            ),
            "comparison_task_results": (
                str(project_path(root, config.runtime.comparison_task_results))
                if config.runtime.comparison_task_results is not None
                else None
            ),
            "report": str(output_root / "report/darc_execution_repair_pilot.md"),
        }
        (output_root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return output_root / "report/darc_execution_repair_pilot.md"
    return run_root / "task_results.jsonl"
