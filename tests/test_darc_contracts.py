from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from agent_verify.analysis.darc_data import (
    DARCExample,
    EncodedDARC,
    chunk_spans,
    dataset_fingerprint,
    downstream_dependency_texts,
    feature_fingerprint,
    validate_darc_examples,
)
from agent_verify.analysis.darc_experiment import (
    _assert_disjoint,
    _assert_hard_call_cap,
    replay_darc,
)
from agent_verify.analysis.darc_model import DARCPrediction, _network_class, train_ensemble


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


def _prediction(examples: list[DARCExample], predicted_cost: float) -> DARCPrediction:
    probabilities = np.tile(np.asarray([[0.1, 0.2, 0.7]]), (len(examples), 1))
    return DARCPrediction(
        probabilities=probabilities,
        advantage=np.full(len(examples), 0.6),
        advantage_uncertainty=np.zeros(len(examples)),
        expected_total_tokens=np.full(len(examples), predicted_cost),
        upper_total_tokens=np.full(len(examples), predicted_cost + 1.0),
        latency_seconds=0.0,
    )


def test_predicted_budget_reports_actual_overshoot_without_hiding_it() -> None:
    examples = [replace(row, teacher_total_tokens=7) for row in _examples()]
    replay, _trace, audit = replay_darc(
        examples,
        _prediction(examples, 5.0),
        policy="test",
        margin=0.0,
        uncertainty_beta=0.0,
        cost_mode="expected",
        score_mode="advantage",
        budget_accounting="predicted_compatible",
        hard_call_token_cap=10,
        token_budget=10,
        budget_fraction=1.0,
        seed=7,
        scope="combined",
    )
    assert replay.calls == 2
    assert replay.predicted_tokens == 10.0
    assert replay.actual_tokens == 14
    assert audit["actual_budget_overshoot_tokens"] == 4
    assert audit["predicted_budget_overshoot_tokens"] == 0.0


def test_hard_cap_prevents_actual_budget_overshoot_and_one_call_per_task() -> None:
    examples = [replace(row, teacher_total_tokens=7) for row in _examples()]
    replay, _trace, audit = replay_darc(
        examples,
        _prediction(examples, 5.0),
        policy="test",
        margin=0.0,
        uncertainty_beta=0.0,
        cost_mode="expected",
        score_mode="advantage",
        budget_accounting="hard_cap",
        hard_call_token_cap=7,
        token_budget=10,
        budget_fraction=1.0,
        seed=7,
        scope="combined",
    )
    assert replay.calls == 1
    assert sum(checkpoint > 0 for checkpoint in replay.checkpoints) == 1
    assert replay.actual_tokens == 7
    assert replay.actual_tokens <= replay.token_budget
    assert replay.predicted_tokens == 5.0
    assert audit["actual_budget_overshoot_tokens"] == 0
    assert audit["reservation_tokens_sum"] == 7


def test_hard_cap_rejects_an_invalid_per_call_upper_bound() -> None:
    examples = [replace(row, teacher_total_tokens=8) for row in _examples()]
    with pytest.raises(RuntimeError, match="exceeds configured hard cap"):
        replay_darc(
            examples,
            _prediction(examples, 2.0),
            policy="test",
            margin=0.0,
            uncertainty_beta=0.0,
            cost_mode="expected",
            score_mode="advantage",
            budget_accounting="hard_cap",
            hard_call_token_cap=7,
            token_budget=14,
            budget_fraction=1.0,
            seed=7,
            scope="combined",
        )


def test_dataset_hard_cap_and_task_split_guards() -> None:
    examples = _examples()
    _assert_hard_call_cap(examples, split="train", hard_call_token_cap=5)
    with pytest.raises(ValueError, match="not a valid upper bound"):
        _assert_hard_call_cap(examples, split="train", hard_call_token_cap=4)
    with pytest.raises(AssertionError, match="task leakage"):
        _assert_disjoint({"train": examples, "validation": examples})


def test_dependency_views_are_task_and_artifact_conditioned() -> None:
    first = downstream_dependency_texts(
        checkpoint=2,
        task_prompt="Return the sum.",
        entry_point="solve",
        current_artifact={"code": "return a + b"},
    )
    second = downstream_dependency_texts(
        checkpoint=2,
        task_prompt="Return the product.",
        entry_point="solve",
        current_artifact={"code": "return a * b"},
    )
    assert first != second
    assert "return a + b" in first[0]
    assert "Reviewer depends directly" in first[0]


def test_chunk_plan_keeps_head_and_tail_and_respects_limit() -> None:
    spans = chunk_spans(2_000, capacity=510, stride=480, max_chunks=3)
    assert spans[0][0] == 0
    assert spans[-1][1] == 2_000
    assert len(spans) <= 3


def test_downstream_controls_have_distinct_information_paths() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(11)
    network = _network_class()
    params = {"hidden_dim": 8, "dropout": 0.0, "attention_heads": 2}
    batch = {
        name: torch.randn(1, 4)
        for name in ("task", "artifact", "history", "current_role")
    }
    batch.update(
        {
            "downstream": torch.randn(1, 2, 4),
            "downstream_mask": torch.ones(1, 2, dtype=torch.bool),
            "metadata": torch.randn(1, 2),
            "checkpoint": torch.zeros(1, dtype=torch.long),
        }
    )
    changed_downstream = dict(batch)
    changed_downstream["downstream"] = batch["downstream"] + 10.0
    changed_checkpoint = dict(batch)
    changed_checkpoint["checkpoint"] = torch.full((1,), 2, dtype=torch.long)

    none = network(4, 2, params, "none").eval()
    checkpoint = network(4, 2, params, "checkpoint_only").eval()
    semantic = network(4, 2, params, "semantic_attention").eval()
    with torch.no_grad():
        none_base = none(batch)[0]
        assert torch.allclose(none_base, none(changed_downstream)[0])
        assert torch.allclose(none_base, none(changed_checkpoint)[0])
        checkpoint_base = checkpoint(batch)[0]
        assert torch.allclose(checkpoint_base, checkpoint(changed_downstream)[0])
        assert not torch.allclose(checkpoint_base, checkpoint(changed_checkpoint)[0])
        assert not torch.allclose(semantic(batch)[0], semantic(changed_downstream)[0])


def test_config_isolates_downstream_and_loss_ablations() -> None:
    config = yaml.safe_load(Path("configs/darc_v01.yaml").read_text(encoding="utf-8"))
    observed = {
        (
            row["name"],
            row["downstream_mode"],
            tuple(row.get("loss_candidates", [])),
            bool(row.get("paired_ranking")),
        )
        for row in config["ablations"]
        if "reuse_model_from" not in row
    }
    expected = {
        ("semantic_no_downstream", "none", ("natural_ce",), False),
        ("checkpoint_only_control", "checkpoint_only", ("natural_ce",), False),
        ("semantic_attention_natural", "semantic_attention", ("natural_ce",), False),
        ("semantic_attention_balanced", "semantic_attention", ("balanced_ce",), False),
        ("semantic_attention_focal", "semantic_attention", ("focal",), False),
        ("semantic_attention_ce_ranking", "semantic_attention", ("natural_ce",), True),
        ("semantic_attention_balanced_ranking", "semantic_attention", ("balanced_ce",), True),
    }
    assert observed == expected


def test_training_progress_history_records_losses_and_early_stopping_state(
    tmp_path: Path,
) -> None:
    examples = _examples()
    rng = np.random.default_rng(17)
    size = len(examples)
    encoded = EncodedDARC(
        state_keys=tuple(row.state_key for row in examples),
        task=rng.normal(size=(size, 4)).astype(np.float32),
        artifact=rng.normal(size=(size, 4)).astype(np.float32),
        history=rng.normal(size=(size, 4)).astype(np.float32),
        current_role=rng.normal(size=(size, 4)).astype(np.float32),
        downstream=rng.normal(size=(size, 2, 4)).astype(np.float32),
        downstream_mask=np.ones((size, 2), dtype=bool),
        metadata=rng.normal(size=(size, 2)).astype(np.float32),
        checkpoint=np.asarray([row.checkpoint - 1 for row in examples], dtype=np.int64),
    )
    model = train_ensemble(
        examples,
        encoded,
        examples,
        encoded,
        params={
            "hidden_dim": 8,
            "dropout": 0.0,
            "attention_heads": 2,
            "ensemble_members": 1,
            "ensemble_task_fraction": 1.0,
            "epochs": 2,
            "patience": 2,
            "learning_rate": 1e-3,
        },
        downstream_mode="none",
        loss_mode="natural_ce",
        paired_ranking=False,
        seed=19,
        device_name="cpu",
        deterministic=True,
        description="progress-contract",
        progress={"enabled": False},
    )
    history = model.members[0].history
    assert len(history) == 2
    assert {
        "epoch",
        "total_loss",
        "value_loss",
        "cost_mean_loss",
        "cost_quantile_loss",
        "ranking_loss",
        "validation_nll",
        "best_validation_nll",
        "stale_epochs",
    } <= set(history[0])
    model_path = tmp_path / "model.pt"
    model.save(model_path)
    restored = type(model).load(model_path)
    assert restored.members[0].history == history
