"""Versioned machine-readable records emitted by CBV-Bench V2."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from agent_verify.cbv_bench.config import ModelIdentityConfig
from agent_verify.config import ModelConfig, StrictModel
from agent_verify.schemas import EvaluationResult, Message, utc_now


class ModelPairSnapshot(StrictModel):
    """Self-contained, versioned metadata for one Student--Teacher condition."""

    schema_version: Literal["cbv-model-pair-v1"] = "cbv-model-pair-v1"
    model_pair_id: str
    model_pair_fingerprint: str
    student_model: ModelIdentityConfig
    teacher_model: ModelIdentityConfig
    student_role_model_ids: dict[str, str]
    prompt_version: dict[str, str]
    inference_config: dict[str, dict[str, Any]]
    runtime_config: dict[str, Any]
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskRecord(StrictModel):
    schema_version: Literal["cbv-bench-v2"] = "cbv-bench-v2"
    record_type: Literal["cbv_task"] = "cbv_task"
    dataset: str
    dataset_version: str
    task_id: str
    difficulty: str = "unknown"
    release_date: str = ""
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    public_task: dict[str, Any]
    public_task_sha256: str
    selection_seed: int
    selection_rank: int
    split: Literal["train", "validation", "test", "exploratory"]
    exploratory_reason: str = ""


class TransitionRecord(StrictModel):
    schema_version: Literal["cbv-bench-v2"] = "cbv-bench-v2"
    record_type: Literal["cbv_transition"] = "cbv_transition"
    timestamp_utc: str = Field(default_factory=utc_now)
    transition_id: str
    cache_key: str
    compatibility_fingerprint: str
    model_pair_id: str
    model_pair_fingerprint: str
    run_id: str
    dataset: str
    task_id: str
    rollout_index: int
    rollout_seed: int
    checkpoint: Literal[1, 2, 3]
    prefix: str
    action: Literal["S", "V"]
    parent_prefix: str
    parent_state_id: str | None
    input_state_id: str
    output_state_id: str
    input_artifact: dict[str, Any]
    effective_artifact: dict[str, Any]
    output_artifact: dict[str, Any]
    preceding_artifacts: list[dict[str, Any]]
    teacher_model: ModelConfig | None = None
    teacher_verdict: str | None = None
    teacher_changed: bool | None = None
    teacher_explanation: str | None = None
    teacher_invocation_id: str | None = None
    teacher_messages: list[Message] = Field(default_factory=list)
    teacher_raw_output: str | None = None
    teacher_input_tokens: int = 0
    teacher_output_tokens: int = 0
    teacher_total_tokens: int = 0
    teacher_cost_usd: float = 0.0
    teacher_latency_seconds: float = 0.0
    downstream_invocation_id: str | None = None
    downstream_model: ModelConfig | None = None
    downstream_input_tokens: int = 0
    downstream_output_tokens: int = 0
    downstream_cost_usd: float = 0.0
    downstream_latency_seconds: float = 0.0
    status: Literal["completed", "failed", "budget_blocked"] = "completed"
    error: str | None = None
    source: Literal["generated", "cache", "legacy_level1"] = "generated"
    provenance: dict[str, Any] = Field(default_factory=dict)


class TrajectoryRecord(StrictModel):
    schema_version: Literal["cbv-bench-v2"] = "cbv-bench-v2"
    record_type: Literal["cbv_trajectory"] = "cbv_trajectory"
    timestamp_utc: str = Field(default_factory=utc_now)
    trajectory_id: str
    branch_id: str
    parent_branch_id: str | None = None
    cache_key: str
    compatibility_fingerprint: str
    model_pair_id: str
    model_pair_fingerprint: str
    model_pair: ModelPairSnapshot
    run_id: str
    dataset: str
    dataset_version: str
    task_id: str
    difficulty: str = "unknown"
    split: Literal["train", "validation", "test", "exploratory"]
    level: Literal[1, 2]
    workflow_version: str
    student_models: dict[str, ModelConfig]
    teacher_model: ModelConfig
    prompt_versions: dict[str, str]
    action_sequence: str
    actions: list[Literal["S", "V"]]
    rollout_index: int
    rollout_seed: int
    transition_ids: list[str]
    checkpoint_state_ids: list[str]
    artifact_snapshots: list[dict[str, Any]]
    final_code: str
    evaluation: EvaluationResult
    passed: bool
    evaluator_status: str
    teacher_calls: int
    teacher_input_tokens: int
    teacher_output_tokens: int
    teacher_total_tokens: int
    teacher_cost_usd: float
    student_input_tokens: int
    student_output_tokens: int
    student_cost_usd: float
    total_latency_seconds: float
    status: Literal["completed", "failed", "budget_blocked"] = "completed"
    error: str | None = None
    source: Literal["generated", "cache", "legacy_level1"] = "generated"
    provenance: dict[str, Any] = Field(default_factory=dict)


class EffectRecord(StrictModel):
    schema_version: Literal["cbv-bench-v2"] = "cbv-bench-v2"
    record_type: Literal["cbv_effect"] = "cbv_effect"
    model_pair_id: str
    model_pair_fingerprint: str
    dataset: str
    task_id: str
    rollout_index: int
    level: Literal[1, 2]
    effect_type: Literal["level1", "conditional", "pairwise_interaction", "three_way_interaction"]
    checkpoint: int | None = None
    preceding_history: str = ""
    fixed_other_actions: str = ""
    skip_sequence: str = ""
    verify_sequence: str = ""
    y_skip: int | None = None
    y_verify: int | None = None
    gain: int
    label: Literal["Helpful", "Harmful", "NoChange", "Positive", "Negative", "Zero"]
    teacher_token_delta: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorRecord(StrictModel):
    schema_version: Literal["cbv-bench-v2"] = "cbv-bench-v2"
    record_type: Literal["cbv_error"] = "cbv_error"
    timestamp_utc: str = Field(default_factory=utc_now)
    run_id: str
    compatibility_fingerprint: str
    model_pair_id: str
    model_pair_fingerprint: str
    dataset: str
    task_id: str
    rollout_index: int
    prefix: str = ""
    sequence: str = ""
    stage: str
    error_type: str
    message: str
    retryable: bool


class RunManifest(StrictModel):
    schema_version: Literal["cbv-bench-v2"] = "cbv-bench-v2"
    record_type: Literal["cbv_manifest"] = "cbv_manifest"
    created_at_utc: str = Field(default_factory=utc_now)
    run_id: str
    status: Literal["planned", "running", "complete", "incomplete"]
    requested_level: Literal[1, 2]
    required_sequences: list[str]
    config_fingerprint: str
    compatibility_fingerprint: str
    model_pair_id: str
    model_pair_fingerprint: str
    model_pair: ModelPairSnapshot
    software_sha256: str
    config: dict[str, Any]
    selection_manifest_sha256: str
    counts: dict[str, int] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
