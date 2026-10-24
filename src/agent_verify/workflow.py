from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from agent_verify.agents.fixed import CoderAgent, PlannerAgent, ReviewerAgent
from agent_verify.config import ModelsConfig
from agent_verify.logging import JsonlStore, content_id
from agent_verify.schemas import (
    CheckpointState,
    CodeArtifact,
    InvocationContext,
    PlanArtifact,
    PublicTask,
    ReviewArtifact,
    WorkflowResult,
)


@dataclass(frozen=True)
class ArtifactChain:
    plan: PlanArtifact
    code: CodeArtifact | None = None
    review: ReviewArtifact | None = None


class FixedWorkflow:
    def __init__(
        self,
        planner: PlannerAgent,
        coder: CoderAgent,
        reviewer: ReviewerAgent,
        models: ModelsConfig,
        checkpoint_log: JsonlStore,
    ) -> None:
        self.planner = planner
        self.coder = coder
        self.reviewer = reviewer
        self.models = models
        self.checkpoint_log = checkpoint_log

    @staticmethod
    def _seed(rollout_seed: int, checkpoint: int, configured_seed: int) -> int:
        return (rollout_seed * 1009 + checkpoint * 9176 + configured_seed) % 2_147_483_647

    def run_baseline(
        self,
        task: PublicTask,
        run_id: str,
        rollout_seed: int,
    ) -> WorkflowResult:
        checkpoints: list[CheckpointState] = []
        invocation_ids: list[str] = []
        cost = 0.0
        latency = 0.0

        planner_context = self._context(task, run_id, rollout_seed, 1, "planner", "baseline")
        planner_output = self.planner.run(task, planner_context)
        plan = PlanArtifact.model_validate(planner_output.artifact.model_dump())
        state1 = self._checkpoint(task, run_id, rollout_seed, "baseline", 1, [], plan, planner_output, None)
        checkpoints.append(state1)
        invocation_ids.append(planner_output.llm_result.record.invocation_id)
        cost += planner_output.llm_result.record.cost_usd
        latency += planner_output.llm_result.record.latency_seconds

        coder_context = self._context(task, run_id, rollout_seed, 2, "coder", "baseline")
        coder_output = self.coder.run(task, plan, coder_context)
        code = CodeArtifact.model_validate(coder_output.artifact.model_dump())
        state2 = self._checkpoint(
            task,
            run_id,
            rollout_seed,
            "baseline",
            2,
            [plan],
            code,
            coder_output,
            state1.state_id,
        )
        checkpoints.append(state2)
        invocation_ids.append(coder_output.llm_result.record.invocation_id)
        cost += coder_output.llm_result.record.cost_usd
        latency += coder_output.llm_result.record.latency_seconds

        reviewer_context = self._context(task, run_id, rollout_seed, 3, "reviewer", "baseline")
        reviewer_output = self.reviewer.run(task, plan, code, reviewer_context)
        review = ReviewArtifact.model_validate(reviewer_output.artifact.model_dump())
        state3 = self._checkpoint(
            task,
            run_id,
            rollout_seed,
            "baseline",
            3,
            [plan, code],
            review,
            reviewer_output,
            state2.state_id,
        )
        checkpoints.append(state3)
        invocation_ids.append(reviewer_output.llm_result.record.invocation_id)
        cost += reviewer_output.llm_result.record.cost_usd
        latency += reviewer_output.llm_result.record.latency_seconds
        return WorkflowResult(
            final_code=review.code,
            checkpoints=checkpoints,
            invocation_ids=invocation_ids,
            model_cost_usd=cost,
            model_latency_seconds=latency,
        )

    def continue_from(
        self,
        *,
        task: PublicTask,
        restored_state: CheckpointState,
        artifact: BaseModel,
        run_id: str,
        rollout_seed: int,
        branch: str,
    ) -> WorkflowResult:
        checkpoint = restored_state.checkpoint_index
        preceding = restored_state.preceding_artifacts
        if checkpoint == 1:
            chain = ArtifactChain(plan=PlanArtifact.model_validate(artifact.model_dump()))
        elif checkpoint == 2:
            chain = ArtifactChain(
                plan=PlanArtifact.model_validate(preceding[0]),
                code=CodeArtifact.model_validate(artifact.model_dump()),
            )
        else:
            review = ReviewArtifact.model_validate(artifact.model_dump())
            return WorkflowResult(
                final_code=review.code,
                checkpoints=[],
                invocation_ids=[],
                model_cost_usd=0.0,
                model_latency_seconds=0.0,
            )

        states: list[CheckpointState] = []
        invocations: list[str] = []
        cost = 0.0
        latency = 0.0
        parent_id = restored_state.state_id
        if checkpoint < 2:
            context = self._context(task, run_id, rollout_seed, 2, "coder", branch)
            output = self.coder.run(task, chain.plan, context)
            code = CodeArtifact.model_validate(output.artifact.model_dump())
            state = self._checkpoint(
                task,
                run_id,
                rollout_seed,
                branch,
                2,
                [chain.plan],
                code,
                output,
                parent_id,
            )
            states.append(state)
            invocations.append(output.llm_result.record.invocation_id)
            cost += output.llm_result.record.cost_usd
            latency += output.llm_result.record.latency_seconds
            chain = ArtifactChain(plan=chain.plan, code=code)
            parent_id = state.state_id

        assert chain.code is not None
        context = self._context(task, run_id, rollout_seed, 3, "reviewer", branch)
        output = self.reviewer.run(task, chain.plan, chain.code, context)
        review = ReviewArtifact.model_validate(output.artifact.model_dump())
        state = self._checkpoint(
            task,
            run_id,
            rollout_seed,
            branch,
            3,
            [chain.plan, chain.code],
            review,
            output,
            parent_id,
        )
        states.append(state)
        invocations.append(output.llm_result.record.invocation_id)
        cost += output.llm_result.record.cost_usd
        latency += output.llm_result.record.latency_seconds
        return WorkflowResult(
            final_code=review.code,
            checkpoints=states,
            invocation_ids=invocations,
            model_cost_usd=cost,
            model_latency_seconds=latency,
        )

    def _context(
        self,
        task: PublicTask,
        run_id: str,
        rollout_seed: int,
        checkpoint: int,
        role: str,
        branch: str,
    ) -> InvocationContext:
        model = getattr(self.models, role)
        return InvocationContext(
            task_id=task.task_id,
            checkpoint=checkpoint,
            role=role,
            branch=branch,
            run_id=run_id,
            seed=self._seed(rollout_seed, checkpoint, model.seed),
        )

    def _checkpoint(
        self,
        task: PublicTask,
        run_id: str,
        rollout_seed: int,
        branch: str,
        checkpoint_index: int,
        preceding_artifacts: list[BaseModel],
        artifact: BaseModel,
        output,
        parent_state_id: str | None,
    ) -> CheckpointState:
        role = {1: "planner", 2: "coder", 3: "reviewer"}[checkpoint_index]
        identity = {
            "run_id": run_id,
            "task_id": task.task_id,
            "branch": branch,
            "rollout_seed": rollout_seed,
            "checkpoint": checkpoint_index,
            "parent": parent_state_id,
            "artifact": artifact.model_dump(mode="json"),
        }
        state = CheckpointState(
            state_id=content_id("state", identity),
            run_id=run_id,
            task_id=task.task_id,
            checkpoint_index=checkpoint_index,
            role=role,
            branch=branch,
            rollout_seed=rollout_seed,
            parent_state_id=parent_state_id,
            preceding_artifacts=[item.model_dump(mode="json") for item in preceding_artifacts],
            current_artifact=artifact.model_dump(mode="json"),
            messages=output.messages,
            llm_config=getattr(self.models, role),
            invocation_id=output.llm_result.record.invocation_id,
        )
        self.checkpoint_log.append(state)
        return state
