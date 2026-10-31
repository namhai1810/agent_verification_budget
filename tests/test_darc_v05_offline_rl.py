from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from agent_verify.analysis.darc_v04_data import PREFIXES, MultiHistoryState
from agent_verify.analysis.darc_v05_env import (
    CBVLevel2OfflineEnv,
    InfeasibleOfflineAction,
    rollout_forced_action,
)
from agent_verify.analysis.darc_v05_experiment import TensorBoardRunRouter
from agent_verify.analysis.darc_v05_model import V05Prediction
from agent_verify.analysis.darc_v05_training import (
    ConservativePolicy,
    assign_task_folds,
    generate_policy_targets,
    policy_action,
    validate_target_coverage,
)


def _group(outcomes: dict[str, int] | None = None) -> SimpleNamespace:
    outcomes = outcomes or {sequence: int(sequence == "VVS") for sequence in _sequences()}
    rows = {
        sequence: {
            "passed": outcomes[sequence],
            "teacher_total_tokens": 100 * sequence.count("V"),
        }
        for sequence in _sequences()
    }
    return SimpleNamespace(
        task=SimpleNamespace(dataset="synthetic", task_id="0"), rows=rows
    )


def _sequences() -> tuple[str, ...]:
    return ("SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV")


def _states() -> tuple[list[MultiHistoryState], dict[tuple[str, str], int]]:
    states = [
        MultiHistoryState("task", "synthetic", "0", prefix, len(prefix) + 1, None)  # type: ignore[arg-type]
        for prefix in PREFIXES
    ]
    return states, {("task", state.prefix): index for index, state in enumerate(states)}


def _prediction(count: int, *, advantage: float, cost: float = 80.0) -> V05Prediction:
    q_success = np.column_stack((np.full(count, 0.2), np.full(count, 0.2 + advantage)))
    q_cost = np.column_stack((np.zeros(count), np.full(count, cost)))
    return V05Prediction(
        q_success=q_success,
        q_cost=q_cost,
        advantage=np.full(count, advantage),
        advantage_uncertainty=np.zeros(count),
        additional_cost=np.full(count, cost),
        additional_cost_uncertainty=np.zeros(count),
        effect_probabilities=np.tile(np.asarray([[0.0, 0.5, 0.5]]), (count, 1)),
        immediate_cost=np.full(count, cost),
        immediate_cost_uncertainty=np.zeros(count),
    )


def test_environment_follows_branch_and_charges_cost_after_action() -> None:
    states, lookup = _states()
    environment = CBVLevel2OfflineEnv(
        {"task": _group()}, lookup, hard_call_token_cap=120
    )
    observation = environment.reset("task", 240)
    assert observation.prefix == ""
    first = environment.step("V")
    assert first.incremental_teacher_tokens == 100
    assert first.next_observation is not None
    assert first.next_observation.prefix == "V"
    second = environment.step("V")
    third = environment.step("S")
    assert second.incremental_teacher_tokens == 100
    assert third.terminal_correctness == 1
    assert environment.prefix == "VVS"


def test_environment_hard_reserve_rejects_verify_without_peeking_at_actual_cost() -> None:
    _states_list, lookup = _states()
    environment = CBVLevel2OfflineEnv(
        {"task": _group()}, lookup, hard_call_token_cap=120
    )
    environment.reset("task", 119)
    with pytest.raises(InfeasibleOfflineAction):
        environment.step("V")


def test_small_per_task_allowance_does_not_mask_global_feasibility() -> None:
    _states_list, lookup = _states()
    environment = CBVLevel2OfflineEnv(
        {"task": _group()}, lookup, hard_call_token_cap=120
    )
    observation = environment.reset(
        "task",
        240,
        per_task_allowance=10.0,
    )
    assert observation.per_task_allowance == 10.0
    assert observation.global_remaining_budget == 240
    assert environment.feasible("V", predicted_call_tokens=80.0)
    transition = environment.step("V")
    assert transition.incremental_teacher_tokens == 100
    assert transition.next_observation is not None
    assert transition.next_observation.global_remaining_budget == 140


def test_policy_aware_target_can_learn_vvs_without_oracle_max() -> None:
    states, lookup = _states()
    groups = {"task": _group()}

    def continuation(_state: MultiHistoryState, _allowance: int):
        # Fixed observable continuation VV -> S. It is not chosen by inspecting
        # the task outcome inside target construction.
        return lambda observation: "V" if observation.checkpoint == 2 else "S"

    targets = generate_policy_targets(
        states,
        groups,
        lookup,
        allowances=[240.0],
        reference_budget=240.0,
        hard_call_token_cap=120,
        continuation_factory=continuation,
        target_mode="policy_aware",
    )
    start = int(np.flatnonzero(targets.state_indices == lookup[("task", "")])[0])
    assert tuple(targets.q_success[start]) == (0.0, 1.0)
    myopic = generate_policy_targets(
        states,
        groups,
        lookup,
        allowances=[240.0],
        reference_budget=240.0,
        hard_call_token_cap=120,
        continuation_factory=continuation,
        target_mode="myopic",
    )
    assert tuple(myopic.q_success[start]) == (0.0, 0.0)


def test_conservative_score_penalizes_uncertainty_harm_and_cost() -> None:
    base = _prediction(1, advantage=0.2, cost=100.0)
    risky = V05Prediction(
        **{
            **base.__dict__,
            "advantage_uncertainty": np.asarray([0.2]),
            "additional_cost_uncertainty": np.asarray([100.0]),
            "effect_probabilities": np.asarray([[0.4, 0.5, 0.1]]),
        }
    )
    policy = ConservativePolicy(0.1, 1.0, 1.0, 1.0, 1.0, 0.0)
    assert policy.score(base, 0, allowance=500, reference_budget=500) > policy.score(
        risky, 0, allowance=500, reference_budget=500
    )


def test_policy_feasibility_uses_prediction_and_hard_reserve() -> None:
    prediction = _prediction(1, advantage=1.0, cost=80.0)
    policy = ConservativePolicy(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    low = SimpleNamespace(
        global_remaining_budget=119,
        per_task_allowance=10.0,
        state_index=0,
    )
    high = SimpleNamespace(
        global_remaining_budget=120,
        per_task_allowance=10.0,
        state_index=0,
    )
    assert policy_action(
        low,
        predictions=[prediction],
        allowances=[120.0],
        policy=policy,
        reference_budget=120.0,
        hard_call_token_cap=120,
    ) == "S"
    assert policy_action(
        high,
        predictions=[prediction],
        allowances=[120.0],
        policy=policy,
        reference_budget=120.0,
        hard_call_token_cap=120,
    ) == "V"


def test_crossfit_assignment_is_reproducible_and_task_level() -> None:
    keys = [f"task-{index}" for index in range(12)]
    first = assign_task_folds(keys, 3, 42)
    second = assign_task_folds(list(reversed(keys)), 3, 42)
    assert first == second
    assert set(first) == set(keys)
    assert set(first.values()) == {0, 1, 2}


def test_forced_rollout_never_selects_best_descendant_implicitly() -> None:
    _states_list, lookup = _states()
    environment = CBVLevel2OfflineEnv(
        {"task": _group()}, lookup, hard_call_token_cap=120
    )
    result = rollout_forced_action(
        environment,
        task_key="task",
        prefix="",
        global_budget=240,
        per_task_allowance=10.0,
        forced_action="V",
        continuation_policy=lambda _observation: "S",
    )
    assert result.sequence == "VSS"
    assert result.final_correctness == 0


def test_positive_allowances_receive_verify_target_supervision() -> None:
    states, lookup = _states()
    # Exact regression for the real run: every non-zero allowance is smaller
    # than the global hard-call reserve, but is not itself a feasibility gate.
    allowances = [0.0, 1228.0, 2456.0, 3684.0, 4912.0]
    targets = generate_policy_targets(
        states,
        {"task": _group()},
        lookup,
        allowances=allowances,
        reference_budget=4912.0,
        hard_call_token_cap=8192,
        continuation_factory=lambda _state, _allowance: (
            lambda _observation: "S"
        ),
        target_mode="policy_aware",
    )
    positive = targets.allowance > 0
    assert positive.any()
    assert targets.action_mask[positive, 1].all()
    diagnostics = validate_target_coverage(targets, name="synthetic")
    assert diagnostics["verify_supervision_rate_positive"] == 1.0


def test_target_diagnostics_fail_fast_when_all_verify_targets_are_masked() -> None:
    states, lookup = _states()
    targets = generate_policy_targets(
        states,
        {"task": _group()},
        lookup,
        allowances=[10.0],
        reference_budget=10.0,
        hard_call_token_cap=120,
        continuation_factory=lambda _state, _allowance: (
            lambda _observation: "S"
        ),
        target_mode="policy_aware",
    )
    broken = replace(targets, action_mask=np.zeros_like(targets.action_mask))
    with pytest.raises(RuntimeError, match="all Verify targets are masked"):
        validate_target_coverage(broken, name="broken")


def test_tensorboard_router_uses_runs_instead_of_unique_metric_cards(tmp_path) -> None:
    class FakeWriter:
        def __init__(self, log_dir: str) -> None:
            self.log_dir = log_dir

        def close(self) -> None:
            pass

    router = TensorBoardRunRouter(tmp_path, FakeWriter)
    first = router.for_run("conservative_dynamic/seed_1/fold_0")
    same = router.for_run("conservative_dynamic/seed_1/fold_0")
    other = router.for_run("conservative_dynamic/seed_1/fold_1")
    assert first is same
    assert first is not other
    assert len(router.writers) == 2

