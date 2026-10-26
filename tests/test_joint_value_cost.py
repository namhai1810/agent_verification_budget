from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from agent_verify.analysis.joint_value_cost_data import (
    JointExample,
    load_frozen_split,
    validate_examples,
    value_label,
)
from agent_verify.analysis.joint_value_cost_models import (
    JointValueCostModel,
    ValueCostPrediction,
)
from agent_verify.analysis.joint_value_cost_policy import (
    learned_candidates,
    replay_candidates,
)


def make_example(
    task: int,
    checkpoint: int,
    *,
    dataset: str = "humaneval",
    y_skip: int = 0,
    y_verify: int = 0,
    tokens: int = 100,
) -> JointExample:
    task_id = f"HumanEval/{task}" if dataset == "humaneval" else f"LiveCodeBench/{task}"
    return JointExample(
        task_id=task_id,
        dataset=dataset,
        difficulty="unspecified" if dataset == "humaneval" else "easy",
        checkpoint=checkpoint,
        role={1: "planner", 2: "coder", 3: "reviewer"}[checkpoint],
        text=f"[PROBLEM] synthetic content {task % 3} [CP] {checkpoint}",
        metadata=(float(checkpoint), float(dataset == "livecodebench")),
        metadata_names=("checkpoint", "dataset_livecodebench"),
        y_skip=y_skip,
        y_verify=y_verify,
        delta=y_verify - y_skip,
        value_label=value_label(y_skip, y_verify),
        local_risk=0.1 * checkpoint,
        teacher_input_tokens=tokens - 20,
        teacher_output_tokens=20,
        teacher_total_tokens=tokens,
    )


def test_value_labels_cover_helpful_harmful_and_no_change() -> None:
    assert value_label(0, 1) == 2
    assert value_label(1, 0) == 0
    assert value_label(0, 0) == 1
    assert value_label(1, 1) == 1
    with pytest.raises(ValueError):
        value_label(0, 2)


def test_integrity_rejects_bad_cost_accounting() -> None:
    rows = [make_example(0, checkpoint) for checkpoint in (1, 2, 3)]
    bad = JointExample(
        **{
            **rows[0].__dict__,
            "teacher_total_tokens": 999,
        }
    )
    with pytest.raises(ValueError, match="token accounting"):
        validate_examples([bad, rows[1], rows[2]])


def test_integrity_rejects_post_treatment_feature_names() -> None:
    rows = [make_example(0, checkpoint) for checkpoint in (1, 2, 3)]
    leaked = JointExample(
        **{
            **rows[0].__dict__,
            "metadata_names": ("checkpoint", "y_verify"),
        }
    )
    with pytest.raises(ValueError, match="Post-treatment"):
        validate_examples([leaked, rows[1], rows[2]])


def test_all_checkpoints_of_task_must_stay_together() -> None:
    rows = [make_example(0, checkpoint) for checkpoint in (1, 2, 3)]
    with pytest.raises(ValueError, match="CP1/CP2/CP3"):
        validate_examples(rows[:2])


def test_frozen_split_reuses_exact_task_manifest(tmp_path) -> None:
    examples = [make_example(task, checkpoint) for task in range(2) for checkpoint in (1, 2, 3)]
    manifest = tmp_path / "fixed_test_tasks.json"
    assignments = tmp_path / "split_assignments.csv"
    manifest.write_text(json.dumps({"task_ids": ["HumanEval/1"]}), encoding="utf-8")
    with assignments.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task_id", "benchmark", "difficulty", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "HumanEval/0",
                "benchmark": "humaneval",
                "difficulty": "unspecified",
                "split": "train",
            }
        )
        writer.writerow(
            {
                "task_id": "HumanEval/1",
                "benchmark": "humaneval",
                "difficulty": "unspecified",
                "split": "test",
            }
        )
    split = load_frozen_split(
        {
            "frozen_split": {
                "manifest_path": str(manifest),
                "assignments_path": str(assignments),
            }
        },
        examples,
    )
    assert split.train_group_keys == frozenset({"humaneval::HumanEval/0"})
    assert split.test_group_keys == frozenset({"humaneval::HumanEval/1"})


def test_sequential_policy_stops_at_first_eligible_checkpoint() -> None:
    examples = [make_example(0, checkpoint) for checkpoint in (1, 2, 3)]
    probabilities = np.asarray(
        [
            [0.10, 0.85, 0.05],
            [0.05, 0.75, 0.20],
            [0.01, 0.49, 0.50],
        ]
    )
    prediction = ValueCostPrediction(
        value_probabilities=probabilities,
        expected_gain=probabilities[:, 2] - probabilities[:, 0],
        expected_total_tokens=np.asarray([100.0, 100.0, 100.0]),
        upper_total_tokens=np.asarray([120.0, 120.0, 120.0]),
    )
    candidates = learned_candidates(examples, prediction, gain_per_1k_threshold=1.0)
    assert list(candidates.values()) == [1]


def test_expected_cost_admission_can_overshoot_but_hard_replay_cannot() -> None:
    examples = [make_example(0, checkpoint, y_verify=1, tokens=100) for checkpoint in (1, 2, 3)]
    candidates = {examples[0].group_key: 0}
    predicted = np.asarray([5.0, 5.0, 5.0])
    expected = replay_candidates(
        examples,
        candidates,
        policy="joint",
        token_budget=50,
        seed=1,
        admission_mode="expected_cost",
        predicted_costs=predicted,
    )
    hard = replay_candidates(
        examples,
        candidates,
        policy="joint",
        token_budget=50,
        seed=1,
        admission_mode="retrospective_hard_cap",
        predicted_costs=predicted,
    )
    assert expected.actual_total_tokens == 100
    assert expected.selected_checkpoints.tolist() == [1]
    assert hard.actual_total_tokens == 0
    assert hard.selected_checkpoints.tolist() == [0]


def test_joint_model_has_shared_value_and_cost_heads() -> None:
    model = JointValueCostModel(
        {
            "hidden_dim": 8,
            "max_features": 20,
            "svd_components": 2,
            "learning_rate": 0.01,
            "epochs": 1,
            "batch_size": 4,
            "lambda_cost": 0.5,
        },
        seed=7,
    )
    model._initialize(input_dim=5)
    assert model.weights["shared_w"].shape == (5, 8)
    assert model.weights["value_w"].shape == (8, 3)
    assert model.weights["cost_w"].shape == (8, 2)
