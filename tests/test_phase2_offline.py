from __future__ import annotations

import numpy as np

from agent_verify.analysis.phase2_data import (
    FORBIDDEN_FEATURE_NAMES,
    PairExample,
    build_observable_features,
)
from agent_verify.analysis.phase2_models import group_folds, simulate_sequential_policy


def make_example(task: int, checkpoint: int, *, delta: int = 0) -> PairExample:
    y_skip = int(task % 2 == 0)
    y_verify = y_skip + delta
    return PairExample(
        task_id=f"HumanEval/{task}",
        checkpoint=checkpoint,
        role={1: "planner", 2: "coder", 3: "reviewer"}[checkpoint],
        text=f"problem number {task} without identifier field",
        metadata=(float(checkpoint),),
        metadata_names=("checkpoint",),
        y_skip=y_skip,
        y_verify=y_verify,
        delta=delta,
        local_risk=0.2,
        teacher_changed=False,
        teacher_tokens=10,
        teacher_latency_seconds=1.0,
        teacher_cost_usd=0.0,
        original_artifact={"code": "def f():\n    return 1\n"},
        teacher_explanation="",
    )


def test_feature_builder_excludes_post_treatment_fields() -> None:
    text, values, names = build_observable_features(
        prompt="def f(x): return x",
        entry_point="f",
        checkpoint=2,
        role="coder",
        current_artifact={"code": "def f(x):\n    return x\n"},
        preceding_artifacts=[{"summary": "identity"}],
        local_risk=0.15,
    )
    assert len(values) == len(names)
    assert not FORBIDDEN_FEATURE_NAMES.intersection(names)
    assert "y_verify" not in text
    assert "corrected_artifact" not in text


def test_group_folds_keep_task_checkpoints_together() -> None:
    examples = [make_example(task, checkpoint) for task in range(10) for checkpoint in (1, 2, 3)]
    for train, test in group_folds(examples, n_splits=5):
        train_tasks = {examples[int(index)].task_id for index in train}
        test_tasks = {examples[int(index)].task_id for index in test}
        assert train_tasks.isdisjoint(test_tasks)
        assert all(sum(examples[int(index)].task_id == task for index in test) == 3 for task in test_tasks)


def test_sequential_controller_uses_at_most_one_call_and_stops_early() -> None:
    examples = [make_example(task, checkpoint) for task in range(2) for checkpoint in (1, 2, 3)]
    predictions = np.asarray([0.2, 0.9, 0.9, -0.1, 0.3, 0.8])
    _, selected, checkpoints = simulate_sequential_policy(examples, predictions, thresholds=0.1)
    assert selected.tolist() == [0, 4]
    assert checkpoints.tolist() == [1, 2]
    assert len(selected) == 2
