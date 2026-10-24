from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel

from agent_verify.config import ModelConfig
from agent_verify.llm import LLMClient, LLMResult
from agent_verify.logging import JsonlStore
from agent_verify.schemas import (
    CodeArtifact,
    InvocationContext,
    Message,
    PlanArtifact,
    PublicTask,
    ReviewArtifact,
    utc_now,
)


def _schema_rule(schema: type[BaseModel]) -> str:
    return (
        "Return only one JSON object matching this JSON Schema exactly. Do not use Markdown:\n"
        + json.dumps(schema.model_json_schema(), sort_keys=True)
    )


@dataclass(frozen=True)
class AgentOutput:
    artifact: BaseModel
    llm_result: LLMResult
    messages: list[Message]


class PlannerAgent:
    role = "planner"
    checkpoint = 1

    def __init__(self, client: LLMClient, model: ModelConfig) -> None:
        self.client = client
        self.model = model

    def run(self, task: PublicTask, context: InvocationContext) -> AgentOutput:
        messages = [
            Message(
                role="system",
                content=(
                    "You are the Planner in a frozen sequential coding workflow. Produce a concise "
                    "implementation plan, algorithm, edge cases, and complexity. Do not write the "
                    "final implementation or any executable code. " + _schema_rule(PlanArtifact)
                ),
            ),
            Message(
                role="user",
                content=f"TASK_ID: {task.task_id}\nENTRY_POINT: {task.entry_point}\n\n{task.prompt}",
            ),
        ]
        result = self.client.invoke(
            model=self.model,
            messages=messages,
            output_schema=PlanArtifact,
            context=context,
            mock_factory=lambda: PlanArtifact(
                summary=f"Plan a direct implementation of {task.entry_point} from its specification.",
                algorithm=[
                    "Read the documented input/output contract.",
                    "Implement the required transformation directly.",
                    "Return a value of the documented type.",
                ],
                edge_cases=["Empty or minimal inputs", "Boundary values described by the task"],
                complexity="Use a straightforward implementation appropriate to the input size.",
            ),
        )
        return AgentOutput(result.value, result, messages)


class CoderAgent:
    role = "coder"
    checkpoint = 2

    def __init__(self, client: LLMClient, model: ModelConfig) -> None:
        self.client = client
        self.model = model

    def run(
        self,
        task: PublicTask,
        plan: PlanArtifact,
        context: InvocationContext,
    ) -> AgentOutput:
        messages = [
            Message(
                role="system",
                content=(
                    "You are the Coder in a frozen sequential coding workflow. Implement the HumanEval "
                    "task as a complete self-contained Python source file. Preserve the required function "
                    "name and signature. Do not include tests or Markdown fences. "
                    + _schema_rule(CodeArtifact)
                ),
            ),
            Message(
                role="user",
                content=(
                    f"TASK_ID: {task.task_id}\nENTRY_POINT: {task.entry_point}\n\n"
                    f"PROBLEM:\n{task.prompt}\n\nPLAN:\n{plan.model_dump_json(indent=2)}"
                ),
            ),
        ]
        result = self.client.invoke(
            model=self.model,
            messages=messages,
            output_schema=CodeArtifact,
            context=context,
            mock_factory=lambda: CodeArtifact(
                code=task.prompt + "pass\n",
                implementation_notes=(
                    "Deterministic mock placeholder; it is intentionally not a research model output."
                ),
            ),
        )
        return AgentOutput(result.value, result, messages)


class ReviewerAgent:
    role = "reviewer"
    checkpoint = 3

    def __init__(
        self,
        client: LLMClient,
        model: ModelConfig,
        event_log: JsonlStore | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.event_log = event_log

    def run(
        self,
        task: PublicTask,
        plan: PlanArtifact,
        code: CodeArtifact,
        context: InvocationContext,
    ) -> AgentOutput:
        messages = [
            Message(
                role="system",
                content=(
                    "You are the Reviewer in a frozen sequential coding workflow. Inspect logic, syntax, "
                    "edge cases, and consistency with the plan. Return the complete final Python source, "
                    "correcting it only when justified. Set changed=true if and only if the returned "
                    "code differs from CODE after ignoring leading/trailing whitespace. Do not include "
                    "tests or Markdown fences. " + _schema_rule(ReviewArtifact)
                ),
            ),
            Message(
                role="user",
                content=(
                    f"TASK_ID: {task.task_id}\nENTRY_POINT: {task.entry_point}\n\n"
                    f"PROBLEM:\n{task.prompt}\n\nPLAN:\n{plan.model_dump_json(indent=2)}\n\n"
                    f"CODE:\n{code.code}"
                ),
            ),
        ]
        result = self.client.invoke(
            model=self.model,
            messages=messages,
            output_schema=ReviewArtifact,
            context=context,
            mock_factory=lambda: ReviewArtifact(
                code=code.code,
                review_summary="Mock reviewer preserved the placeholder unchanged.",
                issues_found=["No semantic review is performed by the mock backend."],
                changed=False,
            ),
        )
        artifact = ReviewArtifact.model_validate(result.value.model_dump())
        actually_changed = artifact.code.strip() != code.code.strip()
        if artifact.changed != actually_changed:
            if self.event_log is not None:
                self.event_log.append(
                    {
                        "record_type": "consistency_warning",
                        "timestamp_utc": utc_now(),
                        "run_id": context.run_id,
                        "task_id": context.task_id,
                        "checkpoint": context.checkpoint,
                        "branch": context.branch,
                        "role": context.role,
                        "invocation_id": result.record.invocation_id,
                        "field": "changed",
                        "declared_value": artifact.changed,
                        "derived_value": actually_changed,
                        "resolution": "used_derived_value_from_code_diff",
                    }
                )
            artifact = artifact.model_copy(update={"changed": actually_changed})
        return AgentOutput(artifact, result, messages)
