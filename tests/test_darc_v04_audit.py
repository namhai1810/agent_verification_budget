from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import yaml

from agent_verify.analysis.cbv_v01_experiment import ALL_SEQUENCES, FrozenTask, TaskTrajectories
from agent_verify.analysis.darc_v04_audit import (
    aggregate_records,
    replay_c3_strict,
    replay_fixed_strict,
    run,
)
from agent_verify.analysis.darc_v04_experiment import select_development_candidate


def _group(task_id: str = "audit") -> TaskTrajectories:
    task = FrozenTask(
        task_id=task_id,
        dataset="humaneval",
        difficulty="unknown",
        prompt="Return one.",
        entry_point="f",
        level1_complete=True,
        level2_complete=True,
    )
    rows = {}
    for sequence in ALL_SEQUENCES:
        rows[sequence] = {
            "status": "completed",
            "passed": sequence.count("V") >= 2,
            "teacher_total_tokens": 5 * sequence.count("V"),
            "teacher_calls": sequence.count("V"),
            "artifact_snapshots": [
                {"plan": sequence[0]},
                {"code": sequence[:2]},
                {"review": sequence},
            ],
        }
    return TaskTrajectories(task=task, rows=rows, source_level=2)


class _AlwaysVerifyC3:
    @staticmethod
    def predict(_group: TaskTrajectories, _prefix: str, _allowance: float):
        return 0.0, 1.0, 1.0

    @staticmethod
    def _verify_cost(group: TaskTrajectories, prefix: str) -> int:
        before = prefix + "S" * (3 - len(prefix))
        after = prefix + "V" + "S" * (2 - len(prefix))
        return int(group.rows[after]["teacher_total_tokens"] - group.rows[before]["teacher_total_tokens"])


def test_development_selection_is_not_hard_coded_to_full() -> None:
    candidates = [
        {"variant": "full_pretrained", "mean_accuracy": 0.50, "harmful_actions": 1, "actual_tokens": 100, "calls": 20, "token_budget": 200},
        {"variant": "myopic_targets", "mean_accuracy": 0.51, "harmful_actions": 1, "actual_tokens": 60, "calls": 20, "token_budget": 200},
    ]
    selected, rows = select_development_candidate(
        candidates,
        reference_variant="full_pretrained",
        token_utilization_penalty=0.005,
        max_harmful_rate=0.10,
        harmful_rate_tolerance_over_reference=0.01,
    )
    assert selected == "myopic_targets"
    assert next(row for row in rows if row["selected"])["variant"] == "myopic_targets"


def test_c3_and_fixed_replay_share_strict_actual_token_accounting() -> None:
    group = _group()
    groups = {group.task.group_key: group}
    c3, c3_trace = replay_c3_strict(
        groups,
        _AlwaysVerifyC3(),
        token_budget=10,
        budget_fraction=1.0,
        hard_call_token_cap=5,
        seed=7,
        label="C3",
    )
    fixed, fixed_trace = replay_fixed_strict(
        groups,
        "VVV",
        token_budget=10,
        budget_fraction=1.0,
        hard_call_token_cap=5,
        seed=7,
    )
    assert c3.actual_tokens == fixed.actual_tokens == 10
    assert c3.calls == fixed.calls == 2
    assert max(row["remaining_budget_before_action"] for row in c3_trace) == 10
    assert sum(row["actual_cost_after_action"] for row in fixed_trace) == 10


def test_aggregate_reports_between_seed_variance_and_task_bootstrap() -> None:
    group_a = _group("a")
    group_b = _group("b")
    groups = {group_a.task.group_key: group_a, group_b.task.group_key: group_b}
    records = []
    for seed in (11, 13):
        replay, traces = replay_fixed_strict(
            groups,
            "VVV",
            token_budget=30,
            budget_fraction=1.0,
            hard_call_token_cap=5,
            seed=seed,
        )
        records.append(
            {
                "seed": seed,
                "policy": replay.policy,
                "fraction": 1.0,
                "replay": replay,
                "summary": {
                    "accuracy": float(replay.outcomes.mean()),
                    "teacher_tokens": replay.actual_tokens,
                    "teacher_calls": replay.calls,
                    "helpful": 0,
                    "harmful": 0,
                    "no_change": replay.calls,
                    "budget_overshoot_tokens": 0,
                },
            }
        )
    row = aggregate_records(records, bootstrap=50, seed=3)[0]
    assert row["seeds"] == 2
    assert row["max_budget_overshoot"] == 0
    assert 0.0 <= row["accuracy_ci95_low"] <= row["accuracy_ci95_high"] <= 1.0


def test_audit_config_uses_identical_seed_set_and_never_loads_test() -> None:
    config = yaml.safe_load(Path("configs/darc_v04_audit.yaml").read_text())
    assert len(config["reproducibility"]["seeds"]) >= 3
    assert config["evaluation"]["partition"] == "validation"
    assert config["evaluation"]["frozen_test_access"] == "forbidden"
    assert config["selection"]["reference_variant"] == "full_pretrained"
    source = inspect.getsource(run)
    assert '_load_groups(frozen_root, "test")' not in source
    assert "seed=experiment_seed" in source
    assert "seed + offset" not in source
    assert 'selected_variant = "full_pretrained"' not in source

