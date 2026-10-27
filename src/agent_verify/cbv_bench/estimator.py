"""Preflight upper-bound estimates; this module never invokes a model."""

from __future__ import annotations

import json
from pathlib import Path

from agent_verify.budget import BudgetManager
from agent_verify.cbv_bench.config import CBVConfig
from agent_verify.cbv_bench.datasets import load_selection, project_path
from agent_verify.cbv_bench.model_metadata import build_model_pair_snapshot
from agent_verify.cbv_bench.sequences import required_prefixes, required_sequences, teacher_call_count


def estimate_collection(config: CBVConfig, project_root: Path | None = None) -> dict:
    root = project_root or Path(__file__).resolve().parents[3]
    selections = [load_selection(config, spec, root) for spec in config.datasets if spec.enabled]
    task_count = sum(len(item.public_tasks) for item in selections)
    rollouts = task_count * config.collection.repeated_rollouts
    level = config.collection.level
    prefix_counts = {
        checkpoint: sum(len(prefix) == checkpoint for prefix in required_prefixes(level))
        for checkpoint in (1, 2, 3)
    }
    verify_counts = {
        checkpoint: sum(len(prefix) == checkpoint and prefix.endswith("V")
                        for prefix in required_prefixes(level))
        for checkpoint in (1, 2, 3)
    }
    models = config.models
    model_pair = build_model_pair_snapshot(config)
    estimated_teacher_tokens = 0
    estimated_teacher_input = 0
    for selection in selections:
        for task in selection.public_tasks:
            public_tokens = BudgetManager.estimate_tokens(task.prompt)
            input_upper = {
                1: public_tokens + models.planner.max_output_tokens + 900,
                2: public_tokens + models.planner.max_output_tokens + models.coder.max_output_tokens + 900,
                3: public_tokens + models.planner.max_output_tokens + models.coder.max_output_tokens
                + models.reviewer.max_output_tokens + 900,
            }
            for checkpoint in (1, 2, 3):
                calls = verify_counts[checkpoint] * config.collection.repeated_rollouts
                estimated_teacher_input += calls * input_upper[checkpoint]
                estimated_teacher_tokens += calls * (
                    input_upper[checkpoint] + models.teacher.max_output_tokens
                )
    estimated_teacher_tokens = int(
        estimated_teacher_tokens
        * config.collection.teacher_prompt_token_safety_factor
        * (config.provider.max_retries + 1)
    )
    teacher_calls = rollouts * teacher_call_count(level)
    trajectories = rollouts * len(required_sequences(level))
    student_calls = rollouts * (1 + prefix_counts[1] + prefix_counts[2])
    retry_factor = config.provider.max_retries + 1
    estimated_teacher_cost = (
        estimated_teacher_input * config.collection.teacher_prompt_token_safety_factor
        * retry_factor * models.teacher.input_cost_per_million
        + teacher_calls * models.teacher.max_output_tokens * retry_factor
        * models.teacher.output_cost_per_million
    ) / 1_000_000
    estimate = {
        "schema_version": "cbv-estimate-v2", "level": level, "tasks": task_count,
        "model_pair_id": model_pair.model_pair_id,
        "model_pair_fingerprint": model_pair.model_pair_fingerprint,
        "student_model": model_pair.student_model.model_dump(mode="json"),
        "teacher_model": model_pair.teacher_model.model_dump(mode="json"),
        "prompt_version": model_pair.prompt_version,
        "inference_config": model_pair.inference_config,
        "task_rollouts": rollouts, "trajectories": trajectories,
        "unique_student_calls": student_calls, "unique_teacher_calls": teacher_calls,
        "evaluations": trajectories, "teacher_calls_by_checkpoint_per_task": verify_counts,
        "estimated_teacher_token_upper_bound": estimated_teacher_tokens,
        "estimated_teacher_cost_usd": estimated_teacher_cost,
        "assumptions": [
            "Shared prefixes are executed once per task/rollout.",
            (
                "Token estimate uses configured maximum output lengths plus a conservative "
                "prompt/artifact envelope."
            ),
            (
                "Persistent compatible cache hits can reduce new calls; this headline estimate "
                "assumes a cold cache."
            ),
            "Provider retries and evaluator time are not included in token estimates.",
        ],
    }
    output = project_path(config.collection.results_dir, root) / "estimates"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"level{level}_{task_count}tasks.json"
    path.write_text(json.dumps(estimate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return estimate
