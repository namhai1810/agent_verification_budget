from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from agent_verify.config import ModelConfig
from agent_verify.llm import LLMClient
from agent_verify.logging import JsonlStore, canonical_json
from agent_verify.schemas import (
    CheckpointState,
    CodeArtifact,
    CodeVerificationResult,
    InvocationContext,
    Message,
    PlanArtifact,
    PlanVerificationResult,
    PublicTask,
    ReviewArtifact,
    ReviewVerificationResult,
    VerificationResult,
    utc_now,
)

ArtifactT = TypeVar("ArtifactT", bound=BaseModel)


_CHECKPOINT_GUIDANCE: dict[int, str] = {
    1: (
        "You are verifying the Planner artifact. Check that the plan faithfully captures the "
        "problem contract, uses a feasible algorithm, covers specification-relevant edge cases, "
        "and states a defensible complexity. Do not write executable code. Correct only plan "
        "defects likely to affect the downstream Coder or Reviewer. After this intervention, the "
        "Coder and then the Reviewer will run."
    ),
    2: (
        "You are verifying the Coder artifact. Audit the code independently against the public "
        "problem: required entry point and signature, exact return-value and type semantics, "
        "literal handling of unusual problem-specific rules, public examples, boundary cases, "
        "syntax, and runtime safety. Treat the preceding plan as untrusted context rather than "
        "authority. Preserve original input types and representations when the specification "
        "requires them. Make the smallest correction that resolves an identifiable defect. After "
        "this intervention, the Reviewer will run."
    ),
    3: (
        "You are verifying the Reviewer's final artifact. This is the last intervention and its "
        "code will be submitted directly with no downstream agent available to repair it. Re-derive "
        "the contract independently from the public problem; do not trust the review summary, issue "
        "list, plan, or their confidence over the specification. Audit the final code for the "
        "required entry point and signature, exact return-value and type semantics, literal handling "
        "of unusual problem-specific rules, public examples, boundary cases, syntax, and runtime "
        "safety. Make only a minimal, high-confidence correction."
    ),
}


def submission_contract(task: PublicTask) -> str:
    if (
        getattr(task, "benchmark", "humaneval") == "livecodebench"
        and getattr(task, "input_mode", "function") == "stdin"
    ):
        return (
            "This is a LiveCodeBench stdin/stdout submission. The final artifact must be a complete "
            "Python program that reads stdin, writes only the required stdout, and does not hardcode "
            "public examples. Hidden tests are unavailable."
        )
    if getattr(task, "benchmark", "humaneval") == "livecodebench":
        return (
            "This is a LiveCodeBench functional submission. Preserve the required function/class and "
            "starter-code interface. Hidden tests are unavailable."
        )
    if getattr(task, "benchmark", "humaneval") == "bigcodebench":
        return (
            "This is a BigCodeBench function submission. Preserve the public entry point and "
            "library contract. Hidden tests and the canonical solution are unavailable."
        )
    return "This is a HumanEval function submission. Preserve the required entry point and signature."


def checkpoint_guidance(checkpoint_index: int) -> str:
    """Return the frozen, checkpoint-specific TeacherVerifier policy."""
    try:
        return _CHECKPOINT_GUIDANCE[checkpoint_index]
    except KeyError as exc:
        raise ValueError(f"Unsupported checkpoint index: {checkpoint_index}") from exc


def artifact_schema(checkpoint_index: int) -> type[BaseModel]:
    return {1: PlanArtifact, 2: CodeArtifact, 3: ReviewArtifact}[checkpoint_index]


def verification_response_schema(checkpoint_index: int) -> type[BaseModel]:
    """Return a checkpoint-specific envelope for constrained JSON decoding."""
    return {
        1: PlanVerificationResult,
        2: CodeVerificationResult,
        3: ReviewVerificationResult,
    }[checkpoint_index]


@dataclass(frozen=True)
class TeacherOutput:
    result: VerificationResult
    corrected_artifact: BaseModel
    invocation_id: str
    cost_usd: float
    latency_seconds: float


class TeacherVerifier:
    def __init__(
        self,
        client: LLMClient,
        model: ModelConfig,
        event_log: JsonlStore,
    ) -> None:
        self.client = client
        self.model = model
        self.event_log = event_log

    def verify(
        self,
        task: PublicTask,
        checkpoint_state: CheckpointState,
        original_artifact: BaseModel,
        context: InvocationContext,
    ) -> TeacherOutput:
        expected_schema = artifact_schema(checkpoint_state.checkpoint_index)
        response_model = verification_response_schema(checkpoint_state.checkpoint_index)
        response_schema = response_model.model_json_schema()
        guidance = checkpoint_guidance(checkpoint_state.checkpoint_index)
        messages = [
            Message(
                role="system",
                content=(
                    "You are the frozen strong TeacherVerifier in a controlled checkpoint-intervention "
                    "experiment. First derive the task contract independently from PROBLEM and its public "
                    "examples. Then inspect CURRENT_ARTIFACT for substantive correctness defects. Treat "
                    "PRECEDING_ARTIFACTS as potentially fallible context, never as ground truth. Follow "
                    "the literal task specification even when it differs from a familiar textbook "
                    "problem, and do not invent requirements for underspecified inputs. Preserve the "
                    "artifact exactly when it is correct; prefer no change over a speculative or "
                    "style-only rewrite. Correct only identifiable mistakes and make the smallest "
                    "sufficient correction. You have no hidden tests, canonical solution, execution "
                    "feedback, or permission to change workflow/model settings.\n\n"
                    "SUBMISSION CONTRACT:\n"
                    + submission_contract(task)
                    + "\n\nCHECKPOINT-SPECIFIC DUTY:\n"
                    + guidance
                    + "\n\ncorrected_artifact must match ARTIFACT_SCHEMA. Set changed=true if and "
                    "only if corrected_artifact differs from CURRENT_ARTIFACT. Return only JSON matching "
                    "RESPONSE_SCHEMA.\nRESPONSE_SCHEMA:\n" + json.dumps(response_schema, sort_keys=True)
                ),
            ),
            Message(
                role="user",
                content=(
                    f"TASK_ID: {task.task_id}\n"
                    f"BENCHMARK: {getattr(task, 'benchmark', 'humaneval')}\n"
                    f"INPUT_MODE: {getattr(task, 'input_mode', 'function')}\n"
                    f"CHECKPOINT: {checkpoint_state.checkpoint_index}\n"
                    f"AGENT_ROLE: {checkpoint_state.role}\n"
                    f"ENTRY_POINT: {task.entry_point or 'N/A (stdin/stdout program)'}\n\n"
                    f"PROBLEM:\n{task.prompt}\n\n"
                    f"PRECEDING_ARTIFACTS:\n{json.dumps(checkpoint_state.preceding_artifacts)}\n\n"
                    f"CURRENT_ARTIFACT:\n{original_artifact.model_dump_json()}\n\n"
                    f"ARTIFACT_SCHEMA:\n{json.dumps(expected_schema.model_json_schema(), sort_keys=True)}"
                ),
            ),
        ]
        started = time.perf_counter()
        try:
            llm_result = self.client.invoke(
                model=self.model,
                messages=messages,
                output_schema=response_model,
                context=context,
                mock_factory=lambda: response_model(
                    verdict="uncertain",
                    corrected_artifact=original_artifact,
                    explanation="Mock teacher preserves the artifact; no semantic verification occurred.",
                    changed=False,
                ),
            )
            result = VerificationResult.model_validate(llm_result.value.model_dump())
            corrected = expected_schema.model_validate(result.corrected_artifact.model_dump())
            actually_changed = canonical_json(corrected) != canonical_json(original_artifact)
            if result.changed != actually_changed:
                self.event_log.append(
                    {
                        "record_type": "consistency_warning",
                        "timestamp_utc": utc_now(),
                        "run_id": context.run_id,
                        "task_id": context.task_id,
                        "checkpoint": checkpoint_state.checkpoint_index,
                        "branch": context.branch,
                        "role": context.role,
                        "invocation_id": llm_result.record.invocation_id,
                        "field": "changed",
                        "declared_value": result.changed,
                        "derived_value": actually_changed,
                        "resolution": "used_derived_value_from_artifact_diff",
                    }
                )
                result = result.model_copy(update={"changed": actually_changed})
            return TeacherOutput(
                result=result,
                corrected_artifact=corrected,
                invocation_id=llm_result.record.invocation_id,
                cost_usd=llm_result.record.cost_usd,
                latency_seconds=llm_result.record.latency_seconds,
            )
        except (ValidationError, ValueError, RuntimeError) as exc:
            self.event_log.append(
                {
                    "record_type": "teacher_verification_failure",
                    "task_id": task.task_id,
                    "checkpoint": checkpoint_state.checkpoint_index,
                    "branch": context.branch,
                    "run_id": context.run_id,
                    "original_artifact": original_artifact.model_dump(mode="json"),
                    "failure": f"{type(exc).__name__}: {exc}",
                    "latency_seconds": time.perf_counter() - started,
                }
            )
            raise
