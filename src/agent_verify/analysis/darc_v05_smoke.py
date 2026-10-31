"""Small synthetic smoke test for DARC v0.5 (no benchmark data or encoder)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from agent_verify.analysis.darc_data import EncodedDARC
from agent_verify.analysis.darc_v04_data import PREFIXES, MultiHistoryState
from agent_verify.analysis.darc_v05_experiment import NullWriter, replay_v05
from agent_verify.analysis.darc_v05_training import (
    ConservativePolicy,
    assign_task_folds,
    generate_policy_targets,
    train_crossfit_iteration,
    validate_target_coverage,
)


def _groups(count: int) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for task in range(count):
        key = f"synthetic::{task}"
        rows: dict[str, dict[str, Any]] = {}
        for mask in range(8):
            sequence = "".join("V" if mask & (1 << index) else "S" for index in range(3))
            # Non-trivial interactions: some tasks need CP1+CP2, some benefit only
            # at CP3, and some are harmed by any verification.
            if task % 3 == 0:
                passed = int(sequence[0:2] == "VV")
            elif task % 3 == 1:
                passed = int(sequence[2] == "V")
            else:
                passed = int(sequence == "SSS")
            rows[sequence] = {
                "passed": passed,
                "teacher_total_tokens": 100 * sequence.count("V"),
            }
        groups[key] = SimpleNamespace(
            task=SimpleNamespace(dataset="synthetic", task_id=str(task)), rows=rows
        )
    return groups


def _states(groups: dict[str, Any]) -> tuple[list[MultiHistoryState], dict[tuple[str, str], int]]:
    states: list[MultiHistoryState] = []
    lookup: dict[tuple[str, str], int] = {}
    for key in sorted(groups):
        for prefix in PREFIXES:
            lookup[(key, prefix)] = len(states)
            states.append(
                MultiHistoryState(
                    task_key=key,
                    dataset="synthetic",
                    task_id=key.rsplit("::", 1)[-1],
                    prefix=prefix,
                    checkpoint=len(prefix) + 1,
                    encoder_example=None,  # type: ignore[arg-type]
                )
            )
    return states, lookup


def _encoded(states: list[MultiHistoryState], seed: int) -> EncodedDARC:
    rng = np.random.default_rng(seed)
    count, embedding, metadata = len(states), 16, 6
    def array(*shape: int) -> np.ndarray:
        return rng.normal(size=shape).astype(np.float32)
    return EncodedDARC(
        state_keys=tuple(state.state_key for state in states),
        task=array(count, embedding),
        artifact=array(count, embedding),
        history=array(count, embedding),
        current_role=array(count, embedding),
        downstream=array(count, 3, embedding),
        downstream_mask=np.ones((count, 3), dtype=bool),
        metadata=array(count, metadata),
        checkpoint=np.asarray([state.checkpoint for state in states]),
    )


def run_synthetic_smoke(config: dict[str, Any], *, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    train_groups = _groups(6)
    validation_groups = {
        key.replace("synthetic::", "synthetic-val::"): value
        for key, value in _groups(3).items()
    }
    # Keep task metadata keys consistent with their dictionary keys.
    train_states, train_lookup = _states(train_groups)
    validation_states, validation_lookup = _states(validation_groups)
    train_encoded = _encoded(train_states, 7)
    validation_encoded = _encoded(validation_states, 11)
    allowances = [0.0, 200.0, 400.0]
    reference_budget = 400.0
    hard_cap = 120
    def always_skip(_state: MultiHistoryState, _allowance: int):
        return lambda _observation: "S"
    train_targets = generate_policy_targets(
        train_states,
        train_groups,
        train_lookup,
        allowances=allowances,
        reference_budget=reference_budget,
        hard_call_token_cap=hard_cap,
        continuation_factory=always_skip,
        target_mode="policy_aware",
    )
    validation_targets = generate_policy_targets(
        validation_states,
        validation_groups,
        validation_lookup,
        allowances=allowances,
        reference_budget=reference_budget,
        hard_call_token_cap=hard_cap,
        continuation_factory=always_skip,
        target_mode="policy_aware",
    )
    train_diagnostics = validate_target_coverage(
        train_targets, name="synthetic-smoke/train"
    )
    validation_diagnostics = validate_target_coverage(
        validation_targets, name="synthetic-smoke/validation"
    )
    params = dict(config["model"])
    params.update(
        {
            "hidden_dim": 16,
            "attention_heads": 4,
            "dropout": 0.0,
            "epochs": int(config["smoke"]["epochs"]),
            "patience": int(config["smoke"]["epochs"]),
            "batch_size": 64,
        }
    )
    folds = int(config["smoke"]["folds"])
    model = train_crossfit_iteration(
        train_states,
        train_encoded,
        train_targets,
        validation_states,
        validation_encoded,
        validation_targets,
        task_folds=assign_task_folds(list(train_groups), folds, 17),
        folds=folds,
        members_per_fold=1,
        iteration=0,
        params=params,
        reference_budget=reference_budget,
        v04_initialization=None,
        previous=None,
        seed=19,
        deterministic=True,
        device_name="cpu",
        writer=NullWriter(),
        variant="synthetic_smoke",
        progress={"tqdm": True, "mininterval_seconds": 0.1},
    )
    checkpoint = output / "synthetic_checkpoint.pt"
    model.save(checkpoint)
    prediction = model.predict(
        validation_encoded, validation_states, allowance=200.0, device_name="cpu"
    )
    policy = ConservativePolicy(0.01, 1.0, 0.5, 0.5, 1.0, 0.0)
    replay, traces = replay_v05(
        validation_groups,
        validation_lookup,
        [prediction, prediction, prediction],
        allowances,
        policy,
        reference_budget=reference_budget,
        token_budget=600,
        budget_fraction=0.5,
        hard_call_token_cap=hard_cap,
        seed=23,
        label="synthetic_smoke",
    )
    if replay.actual_tokens > replay.token_budget:
        raise AssertionError("Synthetic smoke replay overshot budget")
    payload = {
        "schema_version": "darc-v0.5-smoke-v1",
        "synthetic_only": True,
        "train_tasks": len(train_groups),
        "validation_tasks": len(validation_groups),
        "members": len(model.members),
        "checkpoint": str(checkpoint),
        "accuracy": float(replay.outcomes.mean()),
        "teacher_tokens": replay.actual_tokens,
        "teacher_calls": replay.calls,
        "trace_rows": len(traces),
        "train_verify_target_coverage": train_diagnostics[
            "verify_supervision_rate_positive"
        ],
        "validation_verify_target_coverage": validation_diagnostics[
            "verify_supervision_rate_positive"
        ],
        "passed": True,
    }
    result = output / "smoke_result.json"
    result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = output / "smoke_report.md"
    report.write_text(
        "# DARC v0.5 synthetic smoke\n\n"
        "No real CBV data, frozen test, semantic encoder, Student, or Teacher was used.\n\n"
        f"- Cross-fitted members: {len(model.members)}\n"
        f"- Strict budget: {replay.actual_tokens}/{replay.token_budget} tokens\n"
        f"- Result: PASS\n",
        encoding="utf-8",
    )
    return report

