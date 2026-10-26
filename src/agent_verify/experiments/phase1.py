from __future__ import annotations

import platform
import sys
from pathlib import Path

from agent_verify.agents.fixed import CoderAgent, PlannerAgent, ReviewerAgent
from agent_verify.analysis.analyze_h1 import analyze_h1
from agent_verify.budget import BudgetManager
from agent_verify.config import AppConfig
from agent_verify.dataset import load_humaneval
from agent_verify.evaluation.humaneval_runner import HumanEvalRunner
from agent_verify.llm import LLMClient
from agent_verify.logging import JsonlStore, content_id, hash_paths
from agent_verify.schemas import CheckpointState, InvocationContext, PublicTask, utc_now
from agent_verify.verification.local_critic import LocalCritic
from agent_verify.verification.teacher import TeacherVerifier, artifact_schema
from agent_verify.workflow import FixedWorkflow


class Phase1Experiment:
    def __init__(
        self,
        config: AppConfig,
        project_root: Path | None = None,
        run_id_override: str | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root or Path(__file__).resolve().parents[3]
        results_dir = config.experiment.results_dir
        self.results_dir = (
            results_dir if results_dir.is_absolute() else (self.project_root / results_dir)
        ).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.invocations = JsonlStore(self.results_dir / "llm_invocations.jsonl")
        self.checkpoints = JsonlStore(self.results_dir / "checkpoint_states.jsonl")
        self.evaluations = JsonlStore(self.results_dir / "evaluations.jsonl")
        self.baselines = JsonlStore(self.results_dir / "baseline.jsonl")
        self.pairs = JsonlStore(self.results_dir / "intervention_pairs.jsonl")
        self.events = JsonlStore(self.results_dir / "events.jsonl")
        self.manifests = JsonlStore(self.results_dir / "manifests.jsonl")

        source_paths = list((self.project_root / "src").rglob("*.py"))
        source_paths.extend([self.project_root / "pyproject.toml"])
        self.software_hash = hash_paths(source_paths)
        generated_run_id = f"{config.experiment.name}-{config.fingerprint[:10]}-{self.software_hash[:10]}"
        self.resumed_manifest: dict | None = None
        if run_id_override is not None:
            if not config.experiment.resume:
                raise ValueError("run_id_override requires experiment.resume=true")
            matches = [
                row
                for row in self.manifests.read_all()
                if row.get("record_type") == "manifest" and row.get("run_id") == run_id_override
            ]
            if not matches:
                raise ValueError(f"Cannot resume unknown run_id: {run_id_override}")
            manifest = matches[-1]
            if manifest.get("config_fingerprint") != config.fingerprint:
                raise ValueError("Resume run config fingerprint does not match the current config")
            self.run_id = run_id_override
            self.resumed_manifest = manifest
        else:
            self.run_id = generated_run_id

        spent = sum(
            float(row.get("cost_usd", 0.0))
            for row in self.invocations.read_all()
            if row.get("context", {}).get("run_id") == self.run_id
        )
        budget = BudgetManager(config.budget, config.provider, already_spent_usd=spent)
        client = LLMClient(config.provider, budget, self.invocations)
        self.workflow = FixedWorkflow(
            PlannerAgent(client, config.models.planner),
            CoderAgent(client, config.models.coder),
            ReviewerAgent(client, config.models.reviewer, self.events),
            config.models,
            self.checkpoints,
        )
        self.teacher = TeacherVerifier(client, config.models.teacher, self.events)
        self.critic = LocalCritic(client, config.models.critic, self.invocations)
        self.evaluator = HumanEvalRunner(config.evaluation, self.evaluations)

    def run(self) -> str:
        # Fail closed before any generation if generated code cannot be isolated.
        self.evaluator.check_sandbox()
        dataset = load_humaneval(self.config.experiment)
        self._write_manifest(dataset.metadata)

        baseline_by_task = {
            row["task_id"]: row
            for row in self.baselines.read_all()
            if row.get("run_id") == self.run_id and row.get("status") == "completed"
        }
        for task_index, task in enumerate(dataset.public_tasks):
            if task.task_id not in baseline_by_task:
                try:
                    row = self._run_baseline(task, dataset.private_cases[task.task_id], task_index)
                    baseline_by_task[task.task_id] = row
                except Exception as exc:
                    self._failure("baseline_failure", task.task_id, exc)
                    continue

        completed_pair_ids = {
            row["pair_id"]
            for row in self.pairs.read_all()
            if row.get("run_id") == self.run_id and row.get("status") == "completed"
        }
        for task_index, task in enumerate(dataset.public_tasks):
            baseline = baseline_by_task.get(task.task_id)
            if not baseline:
                continue
            state_rows = baseline["checkpoint_states"]
            for checkpoint_index in (1, 2, 3):
                state = CheckpointState.model_validate(state_rows[checkpoint_index - 1])
                for repetition in range(self.config.experiment.repeated_pairs):
                    pair_key = {
                        "run_id": self.run_id,
                        "task_id": task.task_id,
                        "checkpoint": checkpoint_index,
                        "repetition": repetition,
                    }
                    pair_id = content_id("pair", pair_key)
                    if pair_id in completed_pair_ids:
                        continue
                    try:
                        self._run_pair(
                            task=task,
                            case=dataset.private_cases[task.task_id],
                            state=state,
                            baseline=baseline,
                            task_index=task_index,
                            repetition=repetition,
                            pair_id=pair_id,
                        )
                        completed_pair_ids.add(pair_id)
                    except Exception as exc:
                        self._failure(
                            "paired_intervention_failure",
                            task.task_id,
                            exc,
                            checkpoint=checkpoint_index,
                            repetition=repetition,
                            pair_id=pair_id,
                        )

        analyze_h1(self.results_dir, self.run_id)
        return self.run_id

    def _run_baseline(self, task: PublicTask, case, task_index: int) -> dict:
        rollout_seed = self.config.experiment.seed + task_index * 10_000
        workflow_result = self.workflow.run_baseline(task, self.run_id, rollout_seed)
        risk_by_checkpoint: dict[str, dict] = {}
        critic_invocations: list[str] = []
        for state in workflow_result.checkpoints:
            schema = artifact_schema(state.checkpoint_index)
            artifact = schema.model_validate(state.current_artifact)
            context = InvocationContext(
                task_id=task.task_id,
                checkpoint=state.checkpoint_index,
                role="local_critic",
                branch="baseline",
                run_id=self.run_id,
                seed=(rollout_seed + self.config.models.critic.seed + state.checkpoint_index),
            )
            estimate, record = self.critic.estimate(
                task,
                state.role,
                artifact,
                state.preceding_artifacts,
                context,
            )
            risk_by_checkpoint[str(state.checkpoint_index)] = {
                **estimate.model_dump(mode="json"),
                "critic_invocation_id": record.invocation_id,
                "critic_tokens": record.input_tokens + record.output_tokens,
                "critic_cost_usd": record.cost_usd,
                "critic_latency_seconds": record.latency_seconds,
            }
            critic_invocations.append(record.invocation_id)

        evaluation = self.evaluator.evaluate(
            run_id=self.run_id,
            case=case,
            code=workflow_result.final_code,
            branch="baseline",
        )
        identity = {"run_id": self.run_id, "task_id": task.task_id}
        row = {
            "record_type": "baseline",
            "baseline_id": content_id("baseline", identity),
            "timestamp_utc": utc_now(),
            "status": "completed",
            "run_id": self.run_id,
            "task_id": task.task_id,
            "rollout_seed": rollout_seed,
            "checkpoint_state_ids": [state.state_id for state in workflow_result.checkpoints],
            # Embedded snapshots make restoration auditable and robust to interrupted indexing.
            "checkpoint_states": [state.model_dump(mode="json") for state in workflow_result.checkpoints],
            "final_implementation": workflow_result.final_code,
            "evaluation_id": evaluation.evaluation_id,
            "passed": evaluation.passed,
            "risk_by_checkpoint": risk_by_checkpoint,
            "agent_invocation_ids": workflow_result.invocation_ids,
            "critic_invocation_ids": critic_invocations,
            "agent_cost_usd": workflow_result.model_cost_usd,
            "agent_latency_seconds": workflow_result.model_latency_seconds,
        }
        self.baselines.append(row)
        return row

    def _run_pair(
        self,
        *,
        task: PublicTask,
        case,
        state: CheckpointState,
        baseline: dict,
        task_index: int,
        repetition: int,
        pair_id: str,
    ) -> None:
        checkpoint = state.checkpoint_index
        schema = artifact_schema(checkpoint)
        original = schema.model_validate(state.current_artifact)
        rollout_seed = self.config.experiment.seed + task_index * 10_000 + checkpoint * 100 + repetition
        skip_branch = f"cp{checkpoint}_rep{repetition}_skip"
        verify_branch = f"cp{checkpoint}_rep{repetition}_verify"

        skip_result = self.workflow.continue_from(
            task=task,
            restored_state=state,
            artifact=original,
            run_id=self.run_id,
            rollout_seed=rollout_seed,
            branch=skip_branch,
        )
        skip_eval = self.evaluator.evaluate(
            run_id=self.run_id,
            case=case,
            code=skip_result.final_code,
            branch=skip_branch,
        )

        teacher_context = InvocationContext(
            task_id=task.task_id,
            checkpoint=checkpoint,
            role="teacher",
            branch=verify_branch,
            run_id=self.run_id,
            seed=FixedWorkflow._seed(rollout_seed, checkpoint, self.config.models.teacher.seed),
        )
        teacher_output = self.teacher.verify(task, state, original, teacher_context)
        verify_result = self.workflow.continue_from(
            task=task,
            restored_state=state,
            artifact=teacher_output.corrected_artifact,
            run_id=self.run_id,
            rollout_seed=rollout_seed,
            branch=verify_branch,
        )
        verify_eval = self.evaluator.evaluate(
            run_id=self.run_id,
            case=case,
            code=verify_result.final_code,
            branch=verify_branch,
        )

        y_skip, y_verify = int(skip_eval.passed), int(verify_eval.passed)
        risk = baseline["risk_by_checkpoint"][str(checkpoint)]
        row = {
            "record_type": "intervention_pair",
            "pair_id": pair_id,
            "timestamp_utc": utc_now(),
            "status": "completed",
            "run_id": self.run_id,
            "task_id": task.task_id,
            "checkpoint": checkpoint,
            "repetition": repetition,
            "rollout_seed": rollout_seed,
            "matched_randomness": True,
            "nondeterminism_note": (
                "Identical configured seeds and decoding settings were used downstream. "
                "Provider kernels/services may still be nondeterministic."
            ),
            "original_checkpoint_state_id": state.state_id,
            "original_artifact": original.model_dump(mode="json"),
            "corrected_artifact": teacher_output.corrected_artifact.model_dump(mode="json"),
            "local_risk_score": risk["local_risk_score"],
            "local_risk_reasoning": risk["reasoning_summary"],
            "y_skip": y_skip,
            "y_verify": y_verify,
            "delta": y_verify - y_skip,
            "teacher_verdict": teacher_output.result.verdict,
            "teacher_changed": teacher_output.result.changed,
            "teacher_explanation": teacher_output.result.explanation,
            "teacher_invocation_id": teacher_output.invocation_id,
            "teacher_cost_usd": teacher_output.cost_usd,
            "teacher_latency_seconds": teacher_output.latency_seconds,
            "skip": {
                "evaluation_id": skip_eval.evaluation_id,
                "final_implementation": skip_result.final_code,
                "trajectory_state_ids": [state.state_id]
                + [item.state_id for item in skip_result.checkpoints],
                "invocation_ids": skip_result.invocation_ids,
                "cost_usd": skip_result.model_cost_usd,
                "latency_seconds": skip_result.model_latency_seconds,
            },
            "verify": {
                "evaluation_id": verify_eval.evaluation_id,
                "final_implementation": verify_result.final_code,
                "trajectory_state_ids": [state.state_id]
                + [item.state_id for item in verify_result.checkpoints],
                "invocation_ids": [teacher_output.invocation_id] + verify_result.invocation_ids,
                "cost_usd": teacher_output.cost_usd + verify_result.model_cost_usd,
                "latency_seconds": (teacher_output.latency_seconds + verify_result.model_latency_seconds),
            },
        }
        self.pairs.append(row)

    def _write_manifest(self, dataset_metadata: dict[str, object]) -> None:
        existing = {
            row.get("run_id") for row in self.manifests.read_all() if row.get("record_type") == "manifest"
        }
        if self.run_id in existing:
            return
        self.manifests.append(
            {
                "record_type": "manifest",
                "timestamp_utc": utc_now(),
                "run_id": self.run_id,
                "config_fingerprint": self.config.fingerprint,
                "software_sha256": self.software_hash,
                "python": sys.version,
                "platform": platform.platform(),
                "config": self.config.model_dump(mode="json"),
                "dataset": dataset_metadata,
                "mock_only": all(
                    getattr(self.config.models, role).provider in {"mock", "heuristic"}
                    for role in ("planner", "coder", "reviewer", "teacher", "critic")
                ),
            }
        )

    def _failure(self, record_type: str, task_id: str, exc: Exception, **extra) -> None:
        self.events.append(
            {
                "record_type": record_type,
                "timestamp_utc": utc_now(),
                "run_id": self.run_id,
                "task_id": task_id,
                "failure": f"{type(exc).__name__}: {exc}",
                **extra,
            }
        )
