"""CBV-Bench collection orchestration; deliberately independent of model training."""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from agent_verify.agents.fixed import CoderAgent, PlannerAgent, ReviewerAgent
from agent_verify.budget import BudgetManager
from agent_verify.cbv_bench.cache import ExactCache, cache_key, compatibility_fingerprint
from agent_verify.cbv_bench.config import CBVConfig
from agent_verify.cbv_bench.datasets import BenchmarkSelection, load_selection, project_path
from agent_verify.cbv_bench.labels import build_effect_records
from agent_verify.cbv_bench.legacy import LegacyImportError, import_legacy_level1
from agent_verify.cbv_bench.model_metadata import build_model_pair_snapshot
from agent_verify.cbv_bench.resource_budget import CollectionBudget, CollectionBudgetExhausted
from agent_verify.cbv_bench.schemas import ErrorRecord, RunManifest, TrajectoryRecord
from agent_verify.cbv_bench.sequences import required_prefixes, required_sequences
from agent_verify.cbv_bench.tree import BranchingExecutionTree
from agent_verify.evaluation.bigcodebench_runner import BigCodeBenchRunner
from agent_verify.evaluation.humaneval_runner import HumanEvalRunner
from agent_verify.evaluation.livecodebench_runner import LiveCodeBenchRunner
from agent_verify.llm import LLMClient
from agent_verify.logging import JsonlStore, content_id, hash_paths
from agent_verify.verification.teacher import TeacherVerifier


class CBVCollector:
    def __init__(self, config: CBVConfig, project_root: Path | None = None) -> None:
        self.config = config
        self.project_root = project_root or Path(__file__).resolve().parents[3]
        self.results_dir = project_path(config.collection.results_dir, self.project_root)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_pair = build_model_pair_snapshot(config)
        if not config.collection.resume and any(
            (self.results_dir / name).exists()
            for name in ("trajectories.jsonl", "manifests.jsonl", "llm_invocations.jsonl")
        ):
            raise FileExistsError(
                f"CBV results already exist at {self.results_dir}; enable resume or choose a new directory"
            )
        source_paths = list((self.project_root / "src").rglob("*.py"))
        source_paths.extend([self.project_root / "pyproject.toml"])
        self.software_sha256 = hash_paths(source_paths)
        generation_payload = {
            "model_pair": self.model_pair.model_dump(mode="json"),
            "models": config.models.model_dump(mode="json"),
            "provider": config.provider.model_dump(mode="json"),
            "evaluation": config.evaluation.model_dump(mode="json"),
            "workflow": "fixed-planner-coder-reviewer-v1",
            "teacher_semantics": "existing-teacher-verifier",
        }
        generation_fingerprint = content_id("generation", generation_payload)
        dataset_versions = {item.name: item.version for item in config.datasets if item.enabled}
        self.compatibility = compatibility_fingerprint(
            config_fingerprint=generation_fingerprint,
            software_sha256=self.software_sha256,
            dataset_versions=dataset_versions,
        )
        self.run_id = content_id("cbv_run", {
            "config": config.fingerprint, "software": self.software_sha256,
            "compatibility": self.compatibility,
        })
        self.invocations = JsonlStore(self.results_dir / "llm_invocations.jsonl")
        self.states = JsonlStore(self.results_dir / "checkpoint_states.jsonl")
        self.evaluations = JsonlStore(self.results_dir / "evaluations.jsonl")
        self.errors = JsonlStore(self.results_dir / "errors.jsonl")
        self.events = JsonlStore(self.results_dir / "events.jsonl")
        self.manifests = JsonlStore(self.results_dir / "manifests.jsonl")
        self.effects = JsonlStore(self.results_dir / "effects.jsonl")
        self.root_cache = ExactCache(self.results_dir / "cache" / "roots.jsonl", self.compatibility)
        self.transition_cache = ExactCache(
            self.results_dir / "cache" / "transitions.jsonl", self.compatibility
        )
        self.trajectory_cache = ExactCache(
            self.results_dir / "trajectories.jsonl", self.compatibility
        )
        spent = sum(
            float(row.get("cost_usd", 0.0)) for row in self.invocations.read_all()
            if row.get("context", {}).get("run_id") == self.run_id
        )
        api_budget = BudgetManager(config.budget, config.provider, already_spent_usd=spent)
        client = LLMClient(config.provider, api_budget, self.invocations)
        self.planner = PlannerAgent(client, config.models.planner)
        self.coder = CoderAgent(client, config.models.coder)
        self.reviewer = ReviewerAgent(client, config.models.reviewer, self.events)
        self.teacher = TeacherVerifier(client, config.models.teacher, self.events)
        prior_transitions = [
            row for row in self.transition_cache.store.read_all()
            if row.get("run_id") == self.run_id and row.get("action") == "V"
            and row.get("status") == "completed" and row.get("source") != "legacy_level1"
        ]
        self.collection_budget = CollectionBudget(
            max_calls=config.collection.max_teacher_calls,
            max_tokens=config.collection.max_teacher_tokens,
            used_calls=len(prior_transitions),
            used_tokens=sum(int(row.get("teacher_total_tokens", 0)) for row in prior_transitions),
        )

    def collect(self) -> str:
        selections = [
            load_selection(self.config, spec, self.project_root)
            for spec in self.config.datasets if spec.enabled
        ]
        self._manifest("running", selections)
        evaluators = {selection.spec.name: self._evaluator(selection) for selection in selections}
        for selection in selections:
            evaluator = evaluators[selection.spec.name]
            evaluator.check_sandbox()
            if hasattr(evaluator, "check_official_evaluator"):
                evaluator.check_official_evaluator()
            if hasattr(evaluator, "check_dependencies"):
                evaluator.check_dependencies(selection.public_tasks)
        expected = sum(len(item.public_tasks) for item in selections)
        expected *= self.config.collection.repeated_rollouts
        expected *= len(required_sequences(self.config.collection.level))
        completed_before = self._completed_requested_count(selections)
        progress = tqdm(
            total=expected, initial=completed_before, desc="CBV-Bench trajectories",
            unit="trajectory", disable=not self.config.collection.progress,
        )
        budget_blocked = False
        failure_count = 0
        progress.set_postfix(
            teacher_calls=self.collection_budget.used_calls,
            teacher_tokens=self.collection_budget.used_tokens,
            errors=failure_count,
        )
        try:
            for selection in selections:
                tasks = selection.public_tasks
                for batch_start in range(0, len(tasks), self.config.collection.batch_size):
                    batch = tasks[batch_start:batch_start + self.config.collection.batch_size]
                    for task_index, task in enumerate(batch, start=batch_start):
                        for rollout_index in range(self.config.collection.repeated_rollouts):
                            try:
                                added = self._collect_task_rollout(
                                    selection, task, task_index, rollout_index,
                                    evaluators[selection.spec.name],
                                )
                                progress.update(added)
                                progress.set_postfix(
                                    teacher_calls=self.collection_budget.used_calls,
                                    teacher_tokens=self.collection_budget.used_tokens,
                                    errors=failure_count,
                                )
                            except CollectionBudgetExhausted as exc:
                                budget_blocked = True
                                failure_count += 1
                                self._error(selection, task.task_id, rollout_index, "budget", exc, False)
                                progress.set_postfix(
                                    teacher_calls=self.collection_budget.used_calls,
                                    teacher_tokens=self.collection_budget.used_tokens,
                                    errors=failure_count,
                                )
                                break
                            except Exception as exc:
                                failure_count += 1
                                self._error(
                                    selection, task.task_id, rollout_index, "collection", exc, True
                                )
                                progress.set_postfix(
                                    teacher_calls=self.collection_budget.used_calls,
                                    teacher_tokens=self.collection_budget.used_tokens,
                                    errors=failure_count,
                                )
                        if budget_blocked:
                            break
                    if budget_blocked:
                        break
                if budget_blocked:
                    break
        finally:
            progress.close()
        build_effect_records(self.results_dir, self.compatibility, self.config.collection.level)
        complete = self._completed_requested_count(selections) == expected
        self._manifest("complete" if complete else "incomplete", selections)
        if not complete:
            raise RuntimeError(
                "CBV-Bench collection is incomplete; inspect errors.jsonl and validate before export"
            )
        return self.run_id

    def _collect_task_rollout(self, selection: BenchmarkSelection, task, task_index: int,
                              rollout_index: int, evaluator) -> int:
        task_record = selection.task_records[task.task_id]
        rollout_seed = self.config.collection.rollout_seed + task_index * 100_003 + rollout_index
        requested = required_sequences(self.config.collection.level)
        def trajectory_key(sequence: str) -> str:
            return cache_key("trajectory", self.compatibility, {
                "dataset": selection.spec.name, "task_id": task.task_id,
                "rollout_index": rollout_index, "rollout_seed": rollout_seed,
                "sequence": sequence,
            })
        completed_before = sum(
            self.trajectory_cache.get(trajectory_key(sequence)) is not None for sequence in requested
        )
        if (
            self.config.collection.reuse_legacy_level1
            and selection.spec.legacy_result_dirs
            and rollout_index == 0
        ):
            for configured_legacy_dir in selection.spec.legacy_result_dirs:
                try:
                    imported = import_legacy_level1(
                        legacy_dir=project_path(configured_legacy_dir, self.project_root),
                        dataset=selection.spec.name,
                        dataset_version=selection.spec.version,
                        task=task,
                        task_record=task_record,
                        logical_rollout_seed=rollout_seed,
                        run_id=self.run_id,
                        compatibility=self.compatibility,
                        config=self.config,
                        root_cache=self.root_cache,
                        transition_cache=self.transition_cache,
                        trajectory_cache=self.trajectory_cache,
                        state_log=self.states,
                    )
                    if imported:
                        break
                except (LegacyImportError, KeyError, IndexError, ValueError) as exc:
                    self._error(
                        selection, task.task_id, rollout_index,
                        f"legacy_cache_rejected:{configured_legacy_dir}", exc, False,
                    )
        tree = BranchingExecutionTree(
            run_id=self.run_id, dataset=selection.spec.name, compatibility=self.compatibility,
            models=self.config.models, planner=self.planner, coder=self.coder,
            reviewer=self.reviewer, teacher=self.teacher, invocation_log=self.invocations,
            state_log=self.states, root_cache=self.root_cache,
            transition_cache=self.transition_cache, collection_budget=self.collection_budget,
            model_pair=self.model_pair,
            teacher_safety_factor=self.config.collection.teacher_prompt_token_safety_factor,
            teacher_max_attempts=self.config.provider.max_retries + 1,
        )
        private_case = selection.private_cases[task.task_id]
        root = tree.root(task, private_case, rollout_index, rollout_seed)
        node_states = {"": root.state}
        transitions = {}
        terminals = {}
        for prefix in required_prefixes(self.config.collection.level):
            parent = prefix[:-1]
            input_state = node_states.get(parent)
            if input_state is None:
                raise RuntimeError(f"Missing causal parent state for prefix {prefix}")
            edge = tree.edge(
                task=task, private_case=private_case, input_state=input_state,
                action=prefix[-1], prefix=prefix, rollout_index=rollout_index,
                rollout_seed=rollout_seed,
            )
            transitions[prefix] = edge.transition
            if edge.next_state is not None:
                node_states[prefix] = edge.next_state
            if edge.terminal_artifact is not None:
                terminals[prefix] = edge.terminal_artifact

        for sequence in requested:
            identity = {
                "dataset": selection.spec.name, "task_id": task.task_id,
                "rollout_index": rollout_index, "rollout_seed": rollout_seed,
                "sequence": sequence,
            }
            key = cache_key("trajectory", self.compatibility, identity)
            if self.trajectory_cache.get(key) is not None:
                continue
            edges = [transitions[sequence[:depth]] for depth in (1, 2, 3)]
            final = terminals[sequence]
            evaluation = evaluator.evaluate(
                run_id=self.run_id, case=private_case, code=final.code,
                branch=f"cbv:{sequence}:r{rollout_index}",
            )
            teacher_edges = [edge for edge in edges if edge.action == "V"]
            row = TrajectoryRecord(
                trajectory_id=content_id("trajectory", {"cache_key": key}),
                branch_id=content_id("branch", identity),
                parent_branch_id=content_id("branch_prefix", {**identity, "prefix": sequence[:2]}),
                cache_key=key, compatibility_fingerprint=self.compatibility,
                model_pair_id=self.model_pair.model_pair_id,
                model_pair_fingerprint=self.model_pair.model_pair_fingerprint,
                model_pair=self.model_pair,
                run_id=self.run_id, dataset=selection.spec.name,
                dataset_version=selection.spec.version, task_id=task.task_id,
                difficulty=task_record.difficulty, split=task_record.split,
                level=self.config.collection.level,
                workflow_version="fixed-planner-coder-reviewer-v1",
                student_models={
                    "planner": self.config.models.planner,
                    "coder": self.config.models.coder,
                    "reviewer": self.config.models.reviewer,
                },
                teacher_model=self.config.models.teacher,
                prompt_versions={
                    role: getattr(self.config.models, role).prompt_version
                    for role in ("planner", "coder", "reviewer", "teacher")
                },
                action_sequence=sequence,
                actions=list(sequence), rollout_index=rollout_index, rollout_seed=rollout_seed,
                transition_ids=[edge.transition_id for edge in edges],
                checkpoint_state_ids=[root.state.state_id] +
                [edge.output_state_id for edge in edges[:2]],
                artifact_snapshots=[edge.effective_artifact for edge in edges],
                final_code=final.code, evaluation=evaluation, passed=evaluation.passed,
                evaluator_status=evaluation.status, teacher_calls=len(teacher_edges),
                teacher_input_tokens=sum(edge.teacher_input_tokens for edge in teacher_edges),
                teacher_output_tokens=sum(edge.teacher_output_tokens for edge in teacher_edges),
                teacher_total_tokens=sum(edge.teacher_total_tokens for edge in teacher_edges),
                teacher_cost_usd=sum(edge.teacher_cost_usd for edge in teacher_edges),
                student_input_tokens=root.input_tokens +
                sum(edge.downstream_input_tokens for edge in edges),
                student_output_tokens=root.output_tokens +
                sum(edge.downstream_output_tokens for edge in edges),
                student_cost_usd=root.cost_usd + sum(edge.downstream_cost_usd for edge in edges),
                total_latency_seconds=root.latency_seconds +
                sum(edge.teacher_latency_seconds + edge.downstream_latency_seconds for edge in edges),
            )
            self.trajectory_cache.put(row.model_dump(mode="json"))
        completed_after = sum(
            self.trajectory_cache.get(trajectory_key(sequence)) is not None for sequence in requested
        )
        return completed_after - completed_before

    def _evaluator(self, selection: BenchmarkSelection):
        if selection.spec.name == "humaneval":
            return HumanEvalRunner(self.config.evaluation, self.evaluations)
        if selection.spec.name == "livecodebench":
            path = project_path(selection.spec.official_evaluator_path, self.project_root)
            return LiveCodeBenchRunner(self.config.evaluation, self.evaluations, path)
        return BigCodeBenchRunner(self.config.evaluation, self.evaluations)

    def _completed_requested_count(self, selections: list[BenchmarkSelection]) -> int:
        allowed = {(item.spec.name, task.task_id) for item in selections for task in item.public_tasks}
        sequences = set(required_sequences(self.config.collection.level))
        return sum(
            1 for row in self.trajectory_cache.store.read_all()
            if row.get("compatibility_fingerprint") == self.compatibility
            and (row.get("dataset"), row.get("task_id")) in allowed
            and row.get("action_sequence") in sequences and row.get("status") == "completed"
            and int(row.get("rollout_index", -1)) < self.config.collection.repeated_rollouts
        )

    def _error(self, selection: BenchmarkSelection, task_id: str, rollout_index: int,
               stage: str, exc: Exception, retryable: bool) -> None:
        self.errors.append(ErrorRecord(
            run_id=self.run_id, compatibility_fingerprint=self.compatibility,
            model_pair_id=self.model_pair.model_pair_id,
            model_pair_fingerprint=self.model_pair.model_pair_fingerprint,
            dataset=selection.spec.name, task_id=task_id,
            rollout_index=rollout_index, stage=stage, error_type=type(exc).__name__,
            message=str(exc), retryable=retryable,
        ))

    def _manifest(self, status: str, selections: list[BenchmarkSelection]) -> None:
        selection_hash = content_id("selection", {
            item.spec.name: [record.model_dump(mode="json") for record in item.task_records.values()]
            for item in selections
        })
        completed = self._completed_requested_count(selections)
        manifest = RunManifest(
            run_id=self.run_id, status=status, requested_level=self.config.collection.level,
            required_sequences=list(required_sequences(self.config.collection.level)),
            config_fingerprint=self.config.fingerprint,
            compatibility_fingerprint=self.compatibility,
            model_pair_id=self.model_pair.model_pair_id,
            model_pair_fingerprint=self.model_pair.model_pair_fingerprint,
            model_pair=self.model_pair,
            software_sha256=self.software_sha256,
            config=self.config.model_dump(mode="json"),
            selection_manifest_sha256=selection_hash,
            counts={"completed_trajectories": completed,
                    "teacher_calls_used": self.collection_budget.used_calls,
                    "teacher_tokens_used": self.collection_budget.used_tokens},
            limitations=[
                (
                    "BigCodeBench bridge runs official task tests in isolation but does not claim "
                    "leaderboard equivalence."
                ),
                "Provider kernels may remain nondeterministic despite fixed request seeds.",
            ],
        )
        self.manifests.append(manifest)
