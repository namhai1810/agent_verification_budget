from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import yaml

from agent_verify.analysis.cbv_v01_experiment import (
    ALL_SEQUENCES,
    FrozenTask,
    TaskTrajectories,
)
from agent_verify.analysis.darc_v04_data import build_states, build_targets
from agent_verify.analysis.darc_v04_experiment import (
    SequentialPolicy,
    replay_v04,
    run,
)
from agent_verify.analysis.darc_v04_model import V04Prediction


def _group(task_id: str = "demo") -> TaskTrajectories:
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
        # Only histories with a verified final reviewer pass in this toy DAG.
        passed = sequence.endswith("V")
        rows[sequence] = {
            "status": "completed",
            "passed": passed,
            "teacher_total_tokens": 5 * sequence.count("V"),
            "teacher_input_tokens": 3 * sequence.count("V"),
            "teacher_output_tokens": 2 * sequence.count("V"),
            "artifact_snapshots": [
                {"plan": sequence[0]},
                {"code": sequence[:2]},
                {"code": sequence, "changed": sequence[2] == "V"},
            ],
        }
    return TaskTrajectories(task=task, rows=rows, source_level=2)


def test_level2_states_cover_all_shared_prefixes_without_future_actions() -> None:
    group = _group()
    states, lookup = build_states({group.task.group_key: group})
    assert len(states) == 7
    assert set(prefix for _key, prefix in lookup) == {"", "S", "V", "SS", "SV", "VS", "VV"}
    for state in states:
        assert "future" not in state.encoder_example.history_text.lower()
        assert len(state.prefix) == state.checkpoint - 1


def test_counterfactual_targets_use_only_valid_shared_prefix_descendants() -> None:
    group = _group()
    key = group.task.group_key
    states, lookup = build_states({key: group})
    targets = build_targets(
        states,
        {key: group},
        allowances=[0.0, 15.0],
        reference_budget=15.0,
        counterfactual=True,
    )
    start = lookup[(key, "")]
    first = np.flatnonzero((targets.state_indices == start) & (targets.allowance == 0))[0]
    full = np.flatnonzero((targets.state_indices == start) & (targets.allowance == 15))[0]
    assert not bool(targets.verify_mask[first])
    assert bool(targets.verify_mask[full])
    # At full allowance either first action can still reach the helpful CP3 branch.
    assert targets.q_skip[full] == targets.q_verify[full] == 1.0


def test_sequential_replay_has_a_strict_actual_token_budget() -> None:
    group = _group()
    key = group.task.group_key
    states, lookup = build_states({key: group})
    count = len(states)
    prediction = V04Prediction(
        q_skip=np.zeros(count),
        q_verify=np.ones(count),
        q_advantage=np.ones(count),
        uncertainty=np.zeros(count),
        effect_probabilities=np.tile(np.asarray([[0.0, 0.0, 1.0]]), (count, 1)),
        expected_cost=np.full(count, 5.0),
    )
    replay, _ = replay_v04(
        {key: group},
        states,
        lookup,
        [prediction],
        [15.0],
        SequentialPolicy(0.0, 0.0, 0.0, 0.0),
        token_budget=10,
        budget_fraction=1.0,
        hard_call_token_cap=5,
        seed=3,
        label="strict",
    )
    assert replay.actual_tokens <= replay.token_budget
    assert replay.calls == 2


def test_v04_config_and_source_lock_test_after_selection() -> None:
    config = yaml.safe_load(Path("configs/darc_v04.yaml").read_text())
    assert config["model"]["ensemble_members"] == 3
    assert config["evaluation"]["frozen_test_access"] == "locked_after_selection"
    variants = {row["name"]: row for row in config["variants"]}
    assert variants["full_pretrained"]["pretrained"] is True
    assert variants["full_scratch"]["pretrained"] is False
    assert variants["no_history"]["use_history"] is False
    assert variants["myopic_targets"]["counterfactual_targets"] is False
    source = inspect.getsource(run)
    manifest_write = source.index('output / "selection_manifest.json"')
    test_load = source.index('_load_groups(frozen_root, "test")')
    assert manifest_write < test_load

