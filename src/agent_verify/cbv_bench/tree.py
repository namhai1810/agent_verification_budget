"""Causal shared-prefix execution tree for the frozen three-agent workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from agent_verify.agents.fixed import CoderAgent, PlannerAgent, ReviewerAgent
from agent_verify.budget import BudgetManager
from agent_verify.cbv_bench.cache import ExactCache, cache_key, sha256_json
from agent_verify.cbv_bench.resource_budget import CollectionBudget
from agent_verify.cbv_bench.schemas import ModelPairSnapshot, TransitionRecord
from agent_verify.config import ModelsConfig
from agent_verify.logging import JsonlStore, content_id
from agent_verify.schemas import (
    CheckpointState,
    CodeArtifact,
    InvocationContext,
    InvocationRecord,
    PlanArtifact,
    PublicTask,
    ReviewArtifact,
)
from agent_verify.verification.teacher import TeacherVerifier, artifact_schema


@dataclass(frozen=True)
class RootNode:
    state: CheckpointState
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float
    cache_key: str


@dataclass(frozen=True)
class EdgeOutcome:
    transition: TransitionRecord
    next_state: CheckpointState | None
    terminal_artifact: ReviewArtifact | None


def deterministic_seed(rollout_seed: int, checkpoint: int, role_seed: int, prefix: str) -> int:
    digest = hashlib.sha256(f"{rollout_seed}:{checkpoint}:{role_seed}:{prefix}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 2_147_483_647


class PrivacyGuard:
    """Fail closed if evaluator-only payloads appear in model messages."""

    @staticmethod
    def assert_messages_public(messages: list, private_case: Any) -> None:
        serialized = json.dumps(
            [
                message.model_dump(mode="json") if hasattr(message, "model_dump") else message
                for message in messages
            ],
            ensure_ascii=False,
        )
        forbidden = []
        for field in ("test", "input_output"):
            value = getattr(private_case, field, "")
            if isinstance(value, str) and len(value.strip()) >= 16:
                forbidden.append((field, value))
        for field, value in forbidden:
            if value in serialized:
                raise RuntimeError(f"Privacy violation: evaluator-only {field} entered model messages")


class BranchingExecutionTree:
    def __init__(
        self,
        *,
        run_id: str,
        dataset: str,
        compatibility: str,
        models: ModelsConfig,
        planner: PlannerAgent,
        coder: CoderAgent,
        reviewer: ReviewerAgent,
        teacher: TeacherVerifier,
        invocation_log: JsonlStore,
        state_log: JsonlStore,
        root_cache: ExactCache,
        transition_cache: ExactCache,
        collection_budget: CollectionBudget,
        model_pair: ModelPairSnapshot,
        teacher_safety_factor: float,
        teacher_max_attempts: int,
    ) -> None:
        self.run_id = run_id
        self.dataset = dataset
        self.compatibility = compatibility
        self.models = models
        self.planner = planner
        self.coder = coder
        self.reviewer = reviewer
        self.teacher = teacher
        self.invocation_log = invocation_log
        self.state_log = state_log
        self.root_cache = root_cache
        self.transition_cache = transition_cache
        self.collection_budget = collection_budget
        self.model_pair = model_pair
        self.teacher_safety_factor = teacher_safety_factor
        self.teacher_max_attempts = teacher_max_attempts
        self.states = {
            row["state_id"]: CheckpointState.model_validate(row)
            for row in state_log.read_all()
            if row.get("record_type") == "checkpoint_state"
        }

    def root(self, task: PublicTask, private_case: Any, rollout_index: int,
             rollout_seed: int) -> RootNode:
        identity = {
            "dataset": self.dataset,
            "task_sha256": sha256_json(task.model_dump(mode="json")),
            "rollout_index": rollout_index,
            "rollout_seed": rollout_seed,
            "role": "planner",
        }
        key = cache_key("root", self.compatibility, identity)
        cached = self.root_cache.get(key)
        if cached is not None:
            state = self.states.get(str(cached["state_id"]))
            if state is None:
                raise ValueError(f"Root cache references missing state {cached['state_id']}")
            return RootNode(state, int(cached["input_tokens"]), int(cached["output_tokens"]),
                            float(cached["cost_usd"]), float(cached["latency_seconds"]), key)
        context = self._context(task, rollout_seed, 1, "planner", "root", self.models.planner.seed)
        output = self.planner.run(task, context)
        PrivacyGuard.assert_messages_public(output.messages, private_case)
        artifact = PlanArtifact.model_validate(output.artifact.model_dump())
        state = self._state(task, rollout_seed, "root", 1, [], artifact, output.messages,
                            self.models.planner, output.llm_result.record.invocation_id, None)
        record = output.llm_result.record
        row = {
            "record_type": "cbv_root", "cache_key": key,
            "compatibility_fingerprint": self.compatibility, "status": "completed",
            "model_pair_id": self.model_pair.model_pair_id,
            "model_pair_fingerprint": self.model_pair.model_pair_fingerprint,
            "dataset": self.dataset, "task_id": task.task_id, "rollout_index": rollout_index,
            "rollout_seed": rollout_seed, "state_id": state.state_id,
            "invocation_id": record.invocation_id,
            "model": record.llm_config.model_dump(mode="json"),
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens, "cost_usd": record.cost_usd,
            "latency_seconds": record.latency_seconds,
        }
        self.root_cache.put(row)
        return RootNode(state, record.input_tokens, record.output_tokens, record.cost_usd,
                        record.latency_seconds, key)

    def edge(self, *, task: PublicTask, private_case: Any, input_state: CheckpointState,
             action: str, prefix: str, rollout_index: int, rollout_seed: int) -> EdgeOutcome:
        checkpoint = len(prefix)
        if checkpoint != input_state.checkpoint_index:
            raise ValueError(
                f"Temporal mismatch: prefix {prefix} acts at CP{checkpoint}, "
                f"state is CP{input_state.checkpoint_index}"
            )
        identity = {
            "dataset": self.dataset, "task_id": task.task_id, "rollout_index": rollout_index,
            "rollout_seed": rollout_seed, "prefix": prefix, "action": action,
            "input_state_id": input_state.state_id,
        }
        key = cache_key("transition", self.compatibility, identity)
        cached = self.transition_cache.get(key)
        if cached is not None:
            transition = TransitionRecord.model_validate(cached)
            if checkpoint < 3:
                state = self.states.get(transition.output_state_id)
                if state is None:
                    raise ValueError(
                        f"Transition cache references missing state {transition.output_state_id}"
                    )
                return EdgeOutcome(transition, state, None)
            return EdgeOutcome(
                transition,
                None,
                ReviewArtifact.model_validate(transition.output_artifact),
            )

        schema = artifact_schema(checkpoint)
        original = schema.model_validate(input_state.current_artifact)
        effective: BaseModel = original
        teacher_record: InvocationRecord | None = None
        teacher_result = None
        reservation = None
        if action == "V":
            estimated_prompt = BudgetManager.estimate_tokens(
                task.prompt + json.dumps(input_state.preceding_artifacts) + original.model_dump_json()
            )
            estimated_upper = int(
                (estimated_prompt + self.models.teacher.max_output_tokens)
                * self.teacher_safety_factor
                * self.teacher_max_attempts
            )
            reservation = self.collection_budget.reserve(max(1, estimated_upper))
            context = self._context(task, rollout_seed, checkpoint, "teacher", prefix,
                                    self.models.teacher.seed)
            try:
                teacher_result = self.teacher.verify(task, input_state, original, context)
                teacher_record = self._invocation(teacher_result.invocation_id)
                PrivacyGuard.assert_messages_public(teacher_record.messages, private_case)
                actual_tokens = teacher_record.input_tokens + teacher_record.output_tokens
                self.collection_budget.settle(reservation, actual_tokens)
                reservation = None
                effective = teacher_result.corrected_artifact
            except Exception:
                failed_record = self._latest_teacher_invocation(task.task_id, prefix, context.seed)
                if failed_record is not None:
                    actual_tokens = failed_record.input_tokens + failed_record.output_tokens
                    self.collection_budget.settle(reservation, actual_tokens)
                    reservation = None
                raise
            finally:
                if reservation is not None:
                    self.collection_budget.cancel(reservation)

        transition_id = content_id("cbv_transition", {"cache_key": key, "prefix": prefix})
        downstream_record: InvocationRecord | None = None
        next_state: CheckpointState | None = None
        terminal: ReviewArtifact | None = None
        if checkpoint == 1:
            plan = PlanArtifact.model_validate(effective.model_dump())
            context = self._context(task, rollout_seed, 2, "coder", prefix, self.models.coder.seed)
            output = self.coder.run(task, plan, context)
            PrivacyGuard.assert_messages_public(output.messages, private_case)
            code = CodeArtifact.model_validate(output.artifact.model_dump())
            downstream_record = output.llm_result.record
            next_state = self._state(task, rollout_seed, prefix, 2, [plan], code, output.messages,
                                     self.models.coder, downstream_record.invocation_id, transition_id)
            output_state_id, output_artifact = next_state.state_id, code.model_dump(mode="json")
        elif checkpoint == 2:
            plan = PlanArtifact.model_validate(input_state.preceding_artifacts[0])
            code = CodeArtifact.model_validate(effective.model_dump())
            context = self._context(task, rollout_seed, 3, "reviewer", prefix,
                                    self.models.reviewer.seed)
            output = self.reviewer.run(task, plan, code, context)
            PrivacyGuard.assert_messages_public(output.messages, private_case)
            review = ReviewArtifact.model_validate(output.artifact.model_dump())
            downstream_record = output.llm_result.record
            next_state = self._state(task, rollout_seed, prefix, 3, [plan, code], review,
                                     output.messages, self.models.reviewer,
                                     downstream_record.invocation_id, transition_id)
            output_state_id, output_artifact = next_state.state_id, review.model_dump(mode="json")
        else:
            terminal = ReviewArtifact.model_validate(effective.model_dump())
            output_artifact = terminal.model_dump(mode="json")
            output_state_id = content_id("terminal", {"transition_id": transition_id,
                                                       "artifact": output_artifact})

        transition = TransitionRecord(
            transition_id=transition_id, cache_key=key,
            compatibility_fingerprint=self.compatibility, run_id=self.run_id,
            model_pair_id=self.model_pair.model_pair_id,
            model_pair_fingerprint=self.model_pair.model_pair_fingerprint,
            dataset=self.dataset, task_id=task.task_id, rollout_index=rollout_index,
            rollout_seed=rollout_seed, checkpoint=checkpoint, prefix=prefix,
            action=action, parent_prefix=prefix[:-1], parent_state_id=input_state.parent_state_id,
            input_state_id=input_state.state_id, output_state_id=output_state_id,
            input_artifact=original.model_dump(mode="json"),
            effective_artifact=effective.model_dump(mode="json"), output_artifact=output_artifact,
            preceding_artifacts=input_state.preceding_artifacts,
            teacher_model=teacher_record.llm_config if teacher_record else None,
            teacher_verdict=teacher_result.result.verdict if teacher_result else None,
            teacher_changed=teacher_result.result.changed if teacher_result else None,
            teacher_explanation=teacher_result.result.explanation if teacher_result else None,
            teacher_invocation_id=teacher_record.invocation_id if teacher_record else None,
            teacher_messages=teacher_record.messages if teacher_record else [],
            teacher_raw_output=teacher_record.raw_output if teacher_record else None,
            teacher_input_tokens=teacher_record.input_tokens if teacher_record else 0,
            teacher_output_tokens=teacher_record.output_tokens if teacher_record else 0,
            teacher_total_tokens=(teacher_record.input_tokens + teacher_record.output_tokens)
            if teacher_record else 0,
            teacher_cost_usd=teacher_record.cost_usd if teacher_record else 0.0,
            teacher_latency_seconds=teacher_record.latency_seconds if teacher_record else 0.0,
            downstream_invocation_id=downstream_record.invocation_id if downstream_record else None,
            downstream_model=downstream_record.llm_config if downstream_record else None,
            downstream_input_tokens=downstream_record.input_tokens if downstream_record else 0,
            downstream_output_tokens=downstream_record.output_tokens if downstream_record else 0,
            downstream_cost_usd=downstream_record.cost_usd if downstream_record else 0.0,
            downstream_latency_seconds=downstream_record.latency_seconds if downstream_record else 0.0,
        )
        self.transition_cache.put(transition.model_dump(mode="json"))
        return EdgeOutcome(transition, next_state, terminal)

    def _context(self, task: PublicTask, rollout_seed: int, checkpoint: int, role: str,
                 prefix: str, role_seed: int) -> InvocationContext:
        return InvocationContext(
            task_id=task.task_id, checkpoint=checkpoint, role=role, branch=f"cbv:{prefix}",
            run_id=self.run_id,
            seed=deterministic_seed(rollout_seed, checkpoint, role_seed, prefix),
        )

    def _state(self, task: PublicTask, rollout_seed: int, branch: str, checkpoint: int,
               preceding: list[BaseModel], artifact: BaseModel, messages: list,
               model, invocation_id: str, parent_state_id: str | None) -> CheckpointState:
        identity = {
            "compatibility": self.compatibility, "task_id": task.task_id, "branch": branch,
            "rollout_seed": rollout_seed, "checkpoint": checkpoint, "parent": parent_state_id,
            "preceding": [item.model_dump(mode="json") for item in preceding],
            "artifact": artifact.model_dump(mode="json"),
        }
        state = CheckpointState(
            state_id=content_id("state", identity), run_id=self.run_id, task_id=task.task_id,
            checkpoint_index=checkpoint, role={1: "planner", 2: "coder", 3: "reviewer"}[checkpoint],
            branch=f"cbv:{branch}", rollout_seed=rollout_seed, parent_state_id=parent_state_id,
            preceding_artifacts=[item.model_dump(mode="json") for item in preceding],
            current_artifact=artifact.model_dump(mode="json"), messages=messages,
            llm_config=model, invocation_id=invocation_id,
        )
        if state.state_id not in self.states:
            self.state_log.append(state)
            self.states[state.state_id] = state
        return state

    def _latest_teacher_invocation(self, task_id: str, prefix: str,
                                   expected_seed: int) -> InvocationRecord | None:
        for row in reversed(self.invocation_log.read_all()):
            context = row.get("context", {})
            if (
                context.get("run_id") == self.run_id
                and context.get("task_id") == task_id
                and context.get("role") == "teacher"
                and context.get("branch") == f"cbv:{prefix}"
                and int(context.get("seed", -1)) == expected_seed
            ):
                return InvocationRecord.model_validate(row)
        return None

    def _invocation(self, invocation_id: str) -> InvocationRecord:
        for row in reversed(self.invocation_log.read_all()):
            if row.get("invocation_id") == invocation_id:
                return InvocationRecord.model_validate(row)
        raise ValueError(f"Invocation record not found: {invocation_id}")
