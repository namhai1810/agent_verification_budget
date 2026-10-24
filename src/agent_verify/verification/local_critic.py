from __future__ import annotations

import ast
import hashlib
import time
import uuid

from pydantic import BaseModel

from agent_verify.budget import BudgetManager
from agent_verify.config import ModelConfig
from agent_verify.llm import LLMClient
from agent_verify.logging import JsonlStore, canonical_json
from agent_verify.schemas import (
    CodeArtifact,
    InvocationContext,
    InvocationRecord,
    Message,
    PlanArtifact,
    PublicTask,
    ReviewArtifact,
    RiskEstimate,
)


class LocalCritic:
    def __init__(
        self,
        client: LLMClient,
        model: ModelConfig,
        invocation_log: JsonlStore,
    ) -> None:
        self.client = client
        self.model = model
        self.invocation_log = invocation_log

    def estimate(
        self,
        task: PublicTask,
        role: str,
        artifact: BaseModel,
        preceding_artifacts: list[dict],
        context: InvocationContext,
    ) -> tuple[RiskEstimate, InvocationRecord]:
        messages = [
            Message(
                role="system",
                content=(
                    "Estimate the chance in [0,1] that the current artifact has a meaningful local "
                    "error. Use only the supplied problem and artifacts; do not infer hidden tests or "
                    "downstream outcomes. Return JSON with local_risk_score and reasoning_summary."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"TASK_ID: {task.task_id}\nROLE: {role}\nPROBLEM:\n{task.prompt}\n\n"
                    f"PRECEDING:\n{canonical_json(preceding_artifacts)}\n\n"
                    f"CURRENT:\n{artifact.model_dump_json()}"
                ),
            ),
        ]
        if self.model.provider != "heuristic":
            result = self.client.invoke(
                model=self.model,
                messages=messages,
                output_schema=RiskEstimate,
                context=context,
                mock_factory=lambda: self._heuristic(task, artifact),
            )
            return RiskEstimate.model_validate(result.value.model_dump()), result.record

        started = time.perf_counter()
        estimate = self._heuristic(task, artifact)
        serialized = canonical_json([message.model_dump() for message in messages])
        record = InvocationRecord(
            invocation_id=f"inv_{uuid.uuid4().hex}",
            context=context,
            llm_config=self.model,
            prompt_hash=hashlib.sha256(serialized.encode()).hexdigest(),
            messages=messages,
            raw_output=estimate.model_dump_json(),
            parsed_output=estimate.model_dump(mode="json"),
            input_tokens=BudgetManager.estimate_tokens(serialized),
            output_tokens=BudgetManager.estimate_tokens(estimate.model_dump_json()),
            cost_usd=0.0,
            latency_seconds=time.perf_counter() - started,
            attempt_count=1,
            attempts=[
                {
                    "attempt": 1,
                    "status": "success",
                    "raw_output": estimate.model_dump_json(),
                    "cost_usd": 0.0,
                }
            ],
        )
        self.invocation_log.append(record)
        return estimate, record

    @staticmethod
    def _heuristic(task: PublicTask, artifact: BaseModel) -> RiskEstimate:
        if isinstance(artifact, PlanArtifact):
            risk = 0.20
            reasons = ["Structured plan exists."]
            if len(artifact.algorithm) < 2:
                risk += 0.25
                reasons.append("Algorithm is unusually short.")
            if not artifact.edge_cases:
                risk += 0.20
                reasons.append("No edge cases are listed.")
        elif isinstance(artifact, (CodeArtifact, ReviewArtifact)):
            code = artifact.code
            risk = 0.15
            reasons = []
            try:
                tree = ast.parse(code)
                functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
                if task.entry_point not in functions:
                    risk += 0.65
                    reasons.append("Required entry point is absent.")
                if any(isinstance(node, ast.Pass) for node in ast.walk(tree)):
                    risk += 0.55
                    reasons.append("Implementation contains pass.")
                if "NotImplementedError" in code:
                    risk += 0.55
                    reasons.append("Implementation raises NotImplementedError.")
                if len(code.splitlines()) < 3:
                    risk += 0.20
                    reasons.append("Implementation is unusually short.")
            except SyntaxError:
                risk = 0.99
                reasons.append("Python syntax parsing failed.")
            if not reasons:
                reasons.append("No obvious structural defect was found; semantics remain untested.")
        else:
            risk = 0.50
            reasons = ["Unknown artifact type."]
        return RiskEstimate(
            local_risk_score=min(1.0, round(risk, 4)),
            reasoning_summary=" ".join(reasons),
        )
