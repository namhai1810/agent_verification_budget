"""Budget-aware counterfactual replay for Joint and Separate value-cost models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from agent_verify.analysis.joint_value_cost_data import (
    FrozenSplit,
    JointExample,
    filter_examples,
)
from agent_verify.analysis.joint_value_cost_experiment import (
    MODEL_KINDS,
    TRAINING_DATASETS,
    read_csv,
    write_csv,
)
from agent_verify.analysis.joint_value_cost_models import ValueCostPrediction
from agent_verify.analysis.joint_value_cost_policy import (
    baseline_candidates,
    learned_candidates,
    legacy_candidates_from_rows,
    replay_candidates,
    replay_summary,
    token_budgets,
)


def _prediction_from_rows(
    examples: list[JointExample],
    rows: list[dict[str, str]],
    *,
    train_mode: str,
    model_kind: str,
    scope: str,
) -> ValueCostPrediction:
    selected = {
        (row["dataset"], row["task_id"], int(row["checkpoint"])): row
        for row in rows
        if row["train_mode"] == train_mode
        and row["model"] == model_kind
        and row["scope"] == scope
    }
    expected_keys = {(row.dataset, row.task_id, row.checkpoint) for row in examples}
    if set(selected) != expected_keys:
        raise ValueError(
            f"Prediction universe mismatch for {train_mode}/{model_kind}/{scope}"
        )
    probabilities = np.asarray(
        [
            [
                float(selected[(row.dataset, row.task_id, row.checkpoint)]["p_harmful"]),
                float(selected[(row.dataset, row.task_id, row.checkpoint)]["p_no_change"]),
                float(selected[(row.dataset, row.task_id, row.checkpoint)]["p_helpful"]),
            ]
            for row in examples
        ]
    )
    expected_cost = np.asarray(
        [
            float(selected[(row.dataset, row.task_id, row.checkpoint)]["predicted_total_tokens"])
            for row in examples
        ]
    )
    upper_cost = np.asarray(
        [
            float(selected[(row.dataset, row.task_id, row.checkpoint)]["predicted_upper_tokens"])
            for row in examples
        ]
    )
    return ValueCostPrediction(
        probabilities,
        probabilities[:, 2] - probabilities[:, 0],
        expected_cost,
        upper_cost,
    )


def _pareto_flags(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        peers = [
            other
            for other in rows
            if other["scope"] == row["scope"]
            and other["admission_mode"] == row["admission_mode"]
            and other["token_budget"] == row["token_budget"]
        ]
        dominated = any(
            float(other["accuracy"]) >= float(row["accuracy"])
            and int(other["teacher_total_tokens"]) <= int(row["teacher_total_tokens"])
            and (
                float(other["accuracy"]) > float(row["accuracy"])
                or int(other["teacher_total_tokens"]) < int(row["teacher_total_tokens"])
            )
            for other in peers
        )
        row["pareto_efficient"] = not dominated


def _legacy_rows(config: dict[str, Any], scope: str) -> list[dict[str, str]]:
    path = Path(config["legacy_baselines"][scope]["compact_predictions"])
    if not path.exists():
        raise FileNotFoundError(f"Missing legacy prediction file: {path}")
    return read_csv(path)


def run_budget_evaluation(
    config: dict[str, Any],
    *,
    examples: list[JointExample],
    split: FrozenSplit,
    output_root: Path,
) -> None:
    prediction_path = output_root / "evaluation" / "predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError("Run cross-domain evaluation before budget evaluation")
    prediction_rows = read_csv(prediction_path)
    scopes = {
        "humaneval": filter_examples(
            examples, split, partition="test", datasets={"humaneval"}
        ),
        "livecodebench": filter_examples(
            examples, split, partition="test", datasets={"livecodebench"}
        ),
        "combined": filter_examples(examples, split, partition="test"),
    }
    combined_train = filter_examples(examples, split, partition="train")
    seed = int(config["experiment"]["seed"])
    repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    policy_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    budget_quantiles = [
        float(value) for value in config["policy"]["budget_quantiles"]
    ]
    replay_count_per_budget = 9 + len(TRAINING_DATASETS) * len(MODEL_KINDS) * 2
    total_replays = sum(
        len(
            token_budgets(
                combined_train,
                len({row.group_key for row in test}),
                budget_quantiles,
            )
        )
        * replay_count_per_budget
        for test in scopes.values()
    )
    replay_progress = tqdm(
        total=total_replays,
        desc="budget-aware replay",
        unit="policy",
        dynamic_ncols=True,
    )
    for scope_index, (scope, test) in enumerate(scopes.items()):
        budgets = token_budgets(
            combined_train,
            len({row.group_key for row in test}),
            budget_quantiles,
        )
        baseline_specs = [
            "Never Verify",
            "Always CP1",
            "Always CP2",
            "Always CP3",
            "Random Verification",
            "Highest Local Risk",
            "Offline Oracle",
        ]
        baseline_maps = {
            name: baseline_candidates(test, name, seed=seed) for name in baseline_specs
        }
        legacy = _legacy_rows(config, scope)
        baseline_maps["Legacy Logistic"] = legacy_candidates_from_rows(
            test, legacy, model_name="Ours Logistic"
        )
        baseline_maps["Legacy Boosting"] = legacy_candidates_from_rows(
            test, legacy, model_name="Ours Boosting"
        )
        for budget_index, budget in enumerate(budgets):
            for policy_name, candidates in baseline_maps.items():
                replay = replay_candidates(
                    test,
                    candidates,
                    policy=policy_name,
                    token_budget=budget,
                    seed=seed,
                    admission_mode="retrospective_hard_cap",
                )
                policy_rows.append(
                    {
                        "scope": scope,
                        "train_mode": "baseline",
                        "model": policy_name,
                        **replay_summary(
                            test,
                            replay,
                            bootstrap_repetitions=repetitions,
                            seed=seed + scope_index * 10_000 + budget_index,
                        ),
                    }
                )
                replay_progress.set_postfix(
                    scope=scope, budget=budget, policy=policy_name
                )
                replay_progress.update()
            for train_mode in TRAINING_DATASETS:
                selections = json.loads(
                    (output_root / "training" / train_mode / "selected_models.json").read_text(
                        encoding="utf-8"
                    )
                )
                for model_kind in MODEL_KINDS:
                    prediction = _prediction_from_rows(
                        test,
                        prediction_rows,
                        train_mode=train_mode,
                        model_kind=model_kind,
                        scope=scope,
                    )
                    threshold = float(selections[model_kind]["gain_per_1k_threshold"])
                    candidates = learned_candidates(
                        test,
                        prediction,
                        gain_per_1k_threshold=threshold,
                    )
                    for admission_mode in ("expected_cost", "retrospective_hard_cap"):
                        replay = replay_candidates(
                            test,
                            candidates,
                            policy=f"{train_mode}:{model_kind}",
                            token_budget=budget,
                            seed=seed,
                            admission_mode=admission_mode,
                            predicted_costs=prediction.expected_total_tokens,
                        )
                        policy_rows.append(
                            {
                                "scope": scope,
                                "train_mode": train_mode,
                                "model": model_kind,
                                **replay_summary(
                                    test,
                                    replay,
                                    bootstrap_repetitions=repetitions,
                                    seed=(
                                        seed
                                        + scope_index * 100_000
                                        + budget_index * 100
                                        + (0 if model_kind == "joint" else 1)
                                    ),
                                ),
                            }
                        )
                        for task_id, selected_index, checkpoint in zip(
                            replay.task_ids,
                            replay.selected_indices,
                            replay.selected_checkpoints,
                            strict=True,
                        ):
                            decision_rows.append(
                                {
                                    "scope": scope,
                                    "train_mode": train_mode,
                                    "model": model_kind,
                                    "admission_mode": admission_mode,
                                    "token_budget": budget,
                                    "task_group": task_id,
                                    "selected_checkpoint": int(checkpoint),
                                    "selected_index": int(selected_index),
                                }
                            )
                        replay_progress.set_postfix(
                            scope=scope,
                            budget=budget,
                            policy=f"{train_mode}:{model_kind}:{admission_mode}",
                        )
                        replay_progress.update()
    replay_progress.close()
    _pareto_flags(policy_rows)
    budget_dir = output_root / "budget_evaluation"
    write_csv(budget_dir / "policy_results.csv", policy_rows)
    write_csv(budget_dir / "policy_decisions.csv", decision_rows)
    (budget_dir / "budget_manifest.json").write_text(
        json.dumps(
            {
                "max_teacher_calls_per_task": 1,
                "expected_cost_admission": (
                    "deployable predicted-cost admission; actual token budget may be exceeded"
                ),
                "retrospective_hard_cap": (
                    "uses observed cost only to reject cap violations; accounting upper bound, "
                    "not deployable foresight"
                ),
                "sequential_constraint": (
                    "first eligible CP1/CP2/CP3; no later checkpoint consulted before current decision"
                ),
                "threshold_selection": "training-only grouped OOF predictions",
                "frozen_test_status": "exploratory_previously_observed",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
