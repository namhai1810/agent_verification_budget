"""LiveCodeBench Phase-3 experiment using the frozen Phase-1 intervention workflow."""

from __future__ import annotations

import json
from pathlib import Path

from agent_verify.analysis.livecodebench import analyze_livecodebench
from agent_verify.budget import BudgetError
from agent_verify.dataset import DatasetSelection
from agent_verify.evaluation.livecodebench_runner import (
    EvaluatorInfrastructureError,
    LiveCodeBenchRunner,
)
from agent_verify.experiments.phase1 import Phase1Experiment
from agent_verify.livecodebench_dataset import (
    DATASET_CONFIG,
    DATASET_REVISION,
    DEFAULT_SELECTION_SEED,
    MIN_CONTEST_DATE,
    OFFICIAL_EVALUATOR_COMMIT,
    load_locked_selection,
)
from agent_verify.logging import content_id
from agent_verify.schemas import CheckpointState, utc_now


class LiveCodeBenchExperiment(Phase1Experiment):
    """Benchmark adapter; baseline/pair logic remains inherited and unchanged."""

    def __init__(
        self,
        config,
        project_root: Path | None = None,
        resume_run_id: str | None = None,
    ) -> None:
        dataset_config = config.experiment.dataset
        if dataset_config.name != "livecodebench_code_generation_lite":
            raise ValueError("LiveCodeBenchExperiment requires a LiveCodeBench dataset config")
        expected = {
            "config_name": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "minimum_contest_date": MIN_CONTEST_DATE,
            "selection_seed": DEFAULT_SELECTION_SEED,
            "official_evaluator_commit": OFFICIAL_EVALUATOR_COMMIT,
        }
        mismatches = {
            key: (getattr(dataset_config, key), value)
            for key, value in expected.items()
            if getattr(dataset_config, key) != value
        }
        if mismatches:
            raise ValueError(f"LiveCodeBench protocol config mismatch: {mismatches}")
        super().__init__(config, project_root, run_id_override=resume_run_id)
        self.resume_run_id = resume_run_id
        evaluator_path = config.experiment.dataset.official_evaluator_path
        if not evaluator_path.is_absolute():
            evaluator_path = self.project_root / evaluator_path
        self.evaluator = LiveCodeBenchRunner(config.evaluation, self.evaluations, evaluator_path)

    def run(self) -> str:
        # Both checks occur before the first model invocation.
        self.evaluator.check_official_evaluator()
        self.evaluator.check_sandbox()
        self._write_recovery_audit()
        selection = load_locked_selection(
            self.results_dir,
            task_count=self.config.experiment.task_offset + self.config.experiment.task_count,
        )
        start = self.config.experiment.task_offset
        stop = start + self.config.experiment.task_count
        tasks = selection.public_tasks[start:stop]
        dataset = DatasetSelection(
            public_tasks=tasks,
            private_cases={task.task_id: selection.private_cases[task.task_id] for task in tasks},
            metadata={
                **selection.metadata,
                "active_task_ids": [task.task_id for task in tasks],
                "task_offset": start,
                "task_count": len(tasks),
            },
        )
        self._write_manifest(dataset.metadata)
        self._write_run_config(dataset.metadata)

        baseline_by_task = {
            row["task_id"]: row
            for row in self.baselines.read_all()
            if row.get("run_id") == self.run_id and row.get("status") == "completed"
        }
        for task_index, task in enumerate(dataset.public_tasks, start=start):
            if task.task_id not in baseline_by_task:
                try:
                    row = self._run_baseline(task, dataset.private_cases[task.task_id], task_index)
                    baseline_by_task[task.task_id] = row
                except (BudgetError, EvaluatorInfrastructureError) as exc:
                    self._failure("baseline_failure", task.task_id, exc)
                    raise
                except Exception as exc:
                    self._failure("baseline_failure", task.task_id, exc)

        completed_pair_ids = {
            row["pair_id"]
            for row in self.pairs.read_all()
            if row.get("run_id") == self.run_id and row.get("status") == "completed"
        }
        for task_index, task in enumerate(dataset.public_tasks, start=start):
            baseline = baseline_by_task.get(task.task_id)
            if not baseline:
                continue
            for checkpoint_index in (1, 2, 3):
                state = CheckpointState.model_validate(baseline["checkpoint_states"][checkpoint_index - 1])
                for repetition in range(self.config.experiment.repeated_pairs):
                    pair_id = content_id(
                        "pair",
                        {
                            "run_id": self.run_id,
                            "task_id": task.task_id,
                            "checkpoint": checkpoint_index,
                            "repetition": repetition,
                        },
                    )
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
                    except (BudgetError, EvaluatorInfrastructureError) as exc:
                        self._failure(
                            "paired_intervention_failure",
                            task.task_id,
                            exc,
                            checkpoint=checkpoint_index,
                            repetition=repetition,
                            pair_id=pair_id,
                        )
                        raise
                    except Exception as exc:
                        self._failure(
                            "paired_intervention_failure",
                            task.task_id,
                            exc,
                            checkpoint=checkpoint_index,
                            repetition=repetition,
                            pair_id=pair_id,
                        )

        analyze_livecodebench(self.results_dir, self.run_id, humaneval_results=self.project_root / "results")
        return self.run_id

    def _write_recovery_audit(self) -> None:
        if self.resumed_manifest is None:
            return
        previous_hash = str(self.resumed_manifest.get("software_sha256", ""))
        existing = [
            row
            for row in self.events.read_all()
            if row.get("record_type") == "run_recovery"
            and row.get("run_id") == self.run_id
            and row.get("recovery_software_sha256") == self.software_hash
        ]
        if existing:
            return
        self.events.append(
            {
                "record_type": "run_recovery",
                "timestamp_utc": utc_now(),
                "run_id": self.run_id,
                "original_software_sha256": previous_hash,
                "recovery_software_sha256": self.software_hash,
                "config_fingerprint": self.config.fingerprint,
                "reason": "Resume only incomplete records after an audited code-only recovery fix.",
            }
        )

    def _write_run_config(self, dataset_metadata: dict) -> None:
        payload = {
            "run_id": self.run_id,
            "config_fingerprint": self.config.fingerprint,
            "config": self.config.model_dump(mode="json"),
            "dataset": dataset_metadata,
            "models": {
                role: getattr(self.config.models, role).model_dump(mode="json")
                for role in ("planner", "coder", "reviewer", "teacher", "critic")
            },
            "protocol": {
                "workflow": "Planner -> CP1 -> Coder -> CP2 -> Reviewer -> CP3",
                "paired_interventions": ["CP1", "CP2", "CP3"],
                "maximum_verifications_per_pair": 1,
                "private_tests_model_visible": False,
            },
        }
        path = self.results_dir / "run_config.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("run_id") == self.run_id:
                return
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
