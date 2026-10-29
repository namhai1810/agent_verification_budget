from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from agent_verify.analysis.darc_data import (
    DARCExample,
    dataset_fingerprint,
    feature_fingerprint,
    validate_darc_examples,
)
from agent_verify.analysis.darc_experiment import replay_darc
from agent_verify.analysis.darc_model import DARCPrediction


def _examples() -> list[DARCExample]:
    rows: list[DARCExample] = []
    for task in ("task-a", "task-b"):
        for checkpoint in (1, 2, 3):
            rows.append(
                DARCExample(
                    group_key=f"demo::{task}",
                    task_id=task,
                    dataset="humaneval",
                    difficulty="unknown",
                    checkpoint=checkpoint,
                    task_text="public task",
                    artifact_text=f"artifact cp{checkpoint}",
                    history_text="observed prefix only",
                    current_role_text=f"role cp{checkpoint}",
                    downstream_role_texts=("static downstream role",),
                    metadata=(float(checkpoint),),
                    metadata_names=("checkpoint",),
                    y_skip=0,
                    y_verify=int(checkpoint == 2),
                    delta=int(checkpoint == 2),
                    value_label=2 if checkpoint == 2 else 1,
                    teacher_total_tokens=5,
                )
            )
    return rows


def test_embedding_cache_identity_excludes_labels_and_costs() -> None:
    examples = _examples()
    changed = [
        replace(row, y_verify=1, delta=1, value_label=2, teacher_total_tokens=999)
        for row in examples
    ]
    assert feature_fingerprint(examples) == feature_fingerprint(changed)
    assert dataset_fingerprint(examples) != dataset_fingerprint(changed)


def test_post_treatment_metadata_is_rejected() -> None:
    examples = _examples()
    examples[0] = replace(examples[0], metadata_names=("y_verify",))
    with pytest.raises(ValueError, match="Forbidden"):
        validate_darc_examples(examples)


def test_replay_uses_predicted_budget_and_at_most_one_call_per_task() -> None:
    examples = _examples()
    probabilities = np.tile(np.asarray([[0.1, 0.2, 0.7]]), (len(examples), 1))
    prediction = DARCPrediction(
        probabilities=probabilities,
        advantage=np.full(len(examples), 0.6),
        advantage_uncertainty=np.zeros(len(examples)),
        expected_total_tokens=np.full(len(examples), 10.0),
        upper_total_tokens=np.full(len(examples), 12.0),
        latency_seconds=0.0,
    )
    replay, _trace = replay_darc(
        examples,
        prediction,
        policy="test",
        margin=0.0,
        uncertainty_beta=0.0,
        cost_mode="expected",
        token_budget=10,
        budget_fraction=1.0,
        seed=7,
        scope="combined",
    )
    assert replay.calls == 1
    assert sum(checkpoint > 0 for checkpoint in replay.checkpoints) == 1
    assert replay.predicted_tokens == 10.0
