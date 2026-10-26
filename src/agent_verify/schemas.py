from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import Field, field_validator

from agent_verify.config import ModelConfig, StrictModel


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PublicTask(StrictModel):
    task_id: str
    prompt: str
    entry_point: str


class LiveCodeBenchPublicTask(PublicTask):
    benchmark: Literal["livecodebench"] = "livecodebench"
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    platform: str
    contest_date: str
    starter_code: str = ""
    input_mode: Literal["function", "stdin"]


class PrivateEvaluationCase(StrictModel):
    """Gold-only data. This type must never enter a model-facing method."""

    task_id: str
    test: str
    entry_point: str


class LiveCodeBenchPrivateCase(StrictModel):
    """Evaluator-only LiveCodeBench sample; never pass this object to an LLM-facing method."""

    task_id: str
    input_output: str
    test_count: int
    official_schema_version: str = "code_generation_lite-v6"


class PlanArtifact(StrictModel):
    summary: str
    algorithm: list[str] = Field(min_length=1)
    edge_cases: list[str]
    complexity: str

    @field_validator("summary", "complexity")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class CodeArtifact(StrictModel):
    code: str
    implementation_notes: str = ""

    @field_validator("code")
    @classmethod
    def code_must_be_nonempty(cls, value: str) -> str:
        value = value.strip()
        if value.startswith("```"):
            lines = value.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            value = "\n".join(lines)
        if not value:
            raise ValueError("code must not be blank")
        return value + ("\n" if not value.endswith("\n") else "")


class ReviewArtifact(StrictModel):
    code: str
    review_summary: str
    issues_found: list[str] = Field(default_factory=list)
    changed: bool

    @field_validator("code")
    @classmethod
    def code_must_be_nonempty(cls, value: str) -> str:
        return CodeArtifact.code_must_be_nonempty(value)


Artifact = PlanArtifact | CodeArtifact | ReviewArtifact


class VerificationResult(StrictModel):
    verdict: Literal["correct", "incorrect", "uncertain"]
    corrected_artifact: Artifact
    explanation: str
    changed: bool


class PlanVerificationResult(StrictModel):
    verdict: Literal["correct", "incorrect", "uncertain"]
    corrected_artifact: PlanArtifact
    explanation: str
    changed: bool


class CodeVerificationResult(StrictModel):
    verdict: Literal["correct", "incorrect", "uncertain"]
    corrected_artifact: CodeArtifact
    explanation: str
    changed: bool


class ReviewVerificationResult(StrictModel):
    verdict: Literal["correct", "incorrect", "uncertain"]
    corrected_artifact: ReviewArtifact
    explanation: str
    changed: bool


class RiskEstimate(StrictModel):
    local_risk_score: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = ""


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str


class InvocationContext(StrictModel):
    task_id: str
    checkpoint: int | None
    role: str
    branch: str
    run_id: str
    seed: int


class InvocationRecord(StrictModel):
    record_type: Literal["llm_invocation"] = "llm_invocation"
    invocation_id: str
    timestamp_utc: str = Field(default_factory=utc_now)
    context: InvocationContext
    llm_config: ModelConfig
    prompt_hash: str
    messages: list[Message]
    raw_output: str | None
    parsed_output: dict[str, Any] | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float
    attempt_count: int
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    failure: str | None = None


class CheckpointState(StrictModel):
    record_type: Literal["checkpoint_state"] = "checkpoint_state"
    state_id: str
    timestamp_utc: str = Field(default_factory=utc_now)
    run_id: str
    task_id: str
    checkpoint_index: Literal[1, 2, 3]
    role: Literal["planner", "coder", "reviewer"]
    branch: str
    rollout_seed: int
    parent_state_id: str | None = None
    preceding_artifacts: list[dict[str, Any]]
    current_artifact: dict[str, Any]
    messages: list[Message]
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    llm_config: ModelConfig
    invocation_id: str


class EvaluationResult(StrictModel):
    record_type: Literal["evaluation"] = "evaluation"
    evaluation_id: str
    timestamp_utc: str = Field(default_factory=utc_now)
    run_id: str
    task_id: str
    branch: str
    passed: bool
    status: str
    latency_seconds: float
    sandbox: str


class WorkflowResult(StrictModel):
    final_code: str
    checkpoints: list[CheckpointState]
    invocation_ids: list[str]
    model_cost_usd: float
    model_latency_seconds: float
