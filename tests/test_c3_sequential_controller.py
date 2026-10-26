from __future__ import annotations

import numpy as np

from agent_verify.analysis.c3_controller import (
    BudgetContext,
    ControllerObservation,
    FittedSequentialController,
    controller_feature_vector,
    group_observations,
    verify_utility_target,
)
from agent_verify.analysis.c3_experiment import _group_folds, replay_controller
from agent_verify.analysis.joint_value_cost_data import JointExample, value_label


def make_observation(
    task: int,
    checkpoint: int,
    *,
    predicted_cost: float = 100.0,
    actual_input: int = 80,
    actual_output: int = 20,
    y_skip: int = 0,
    y_verify: int = 1,
) -> ControllerObservation:
    example = JointExample(
        task_id=f"HumanEval/{task}",
        dataset="humaneval",
        difficulty="unspecified",
        checkpoint=checkpoint,
        role={1: "planner", 2: "coder", 3: "reviewer"}[checkpoint],
        text=f"current observable state cp={checkpoint}",
        metadata=(float(checkpoint), 0.2),
        metadata_names=("checkpoint", "local_risk"),
        y_skip=y_skip,
        y_verify=y_verify,
        delta=y_verify - y_skip,
        value_label=value_label(y_skip, y_verify),
        local_risk=0.2,
        teacher_input_tokens=actual_input,
        teacher_output_tokens=actual_output,
        teacher_total_tokens=actual_input + actual_output,
    )
    return ControllerObservation(
        example=example,
        hidden_state=np.asarray([0.1, -0.2]),
        p_harmful=0.1,
        p_no_change=0.6,
        p_helpful=0.3,
        expected_gain=0.2,
        predicted_cost=predicted_cost,
        predicted_upper_cost=predicted_cost * 1.2,
    )


class FixedActionController(FittedSequentialController):
    def __init__(self, action_values_by_checkpoint: dict[int, tuple[float, float]]):
        super().__init__({}, seed=1, use_continuation=True)
        self.action_values_by_checkpoint = action_values_by_checkpoint

    def action_values(
        self,
        observation: ControllerObservation,
        context: BudgetContext,
    ) -> tuple[float, float]:
        return self.action_values_by_checkpoint[observation.example.checkpoint]


def test_controller_features_exclude_outcomes_and_actual_teacher_cost() -> None:
    first = make_observation(
        0,
        1,
        actual_input=80,
        actual_output=20,
        y_skip=0,
        y_verify=1,
    )
    changed_labels = make_observation(
        0,
        1,
        actual_input=800,
        actual_output=200,
        y_skip=1,
        y_verify=0,
    )
    context = BudgetContext(200.0, 0.5)
    np.testing.assert_allclose(
        controller_feature_vector(first, context),
        controller_feature_vector(changed_labels, context),
    )


def test_verify_target_uses_observed_single_intervention_and_cost_only_as_target() -> None:
    observation = make_observation(0, 1, actual_input=80, actual_output=20)
    target = verify_utility_target(
        observation,
        BudgetContext(200.0, 1.0),
        cost_penalty_scale=0.1,
        reference_cost=100.0,
    )
    assert target == 0.95


def test_temporal_feature_for_cp1_does_not_depend_on_future_artifacts() -> None:
    current = make_observation(0, 1)
    future_a = make_observation(0, 2)
    future_b = ControllerObservation(
        **{
            **future_a.__dict__,
            "hidden_state": np.asarray([999.0, -999.0]),
            "expected_gain": -1.0,
        }
    )
    context = BudgetContext(150.0, 0.75)
    before = controller_feature_vector(current, context)
    _unused = controller_feature_vector(future_a, context)
    after = controller_feature_vector(current, context)
    _unused_changed = controller_feature_vector(future_b, context)
    np.testing.assert_allclose(before, after)


def test_predicted_cost_controls_online_feasibility_not_actual_cost() -> None:
    rows = [
        make_observation(
            0,
            checkpoint,
            predicted_cost=500.0,
            actual_input=8,
            actual_output=2,
        )
        for checkpoint in (1, 2, 3)
    ]
    controller = FixedActionController(
        {1: (1.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 0.0)}
    )
    decision = controller.decide(
        rows,
        remaining_budget=100.0,
        initial_budget=100.0,
        tasks_remaining=1,
    )
    assert decision.checkpoint == 0


def test_sequential_decision_waits_then_verifies_once() -> None:
    rows = [make_observation(0, checkpoint) for checkpoint in (1, 2, 3)]
    controller = FixedActionController(
        {1: (0.1, 0.8), 2: (0.9, 0.2), 3: (1.0, 0.0)}
    )
    decision = controller.decide(
        rows,
        remaining_budget=500.0,
        initial_budget=500.0,
        tasks_remaining=1,
    )
    assert decision.checkpoint == 2
    assert [row["verify"] for row in decision.trace] == [False, True]


def test_replay_accounts_teacher_input_and_output_and_one_call() -> None:
    observations = [
        make_observation(
            0,
            checkpoint,
            predicted_cost=100.0,
            actual_input=80,
            actual_output=20,
        )
        for checkpoint in (1, 2, 3)
    ]
    group_observations(observations)
    controller = FixedActionController(
        {1: (0.1, 0.8), 2: (0.9, 0.2), 3: (1.0, 0.0)}
    )
    replay, decisions = replay_controller(
        observations,
        controller,
        policy_name="test",
        token_budget=500,
        seed=1,
        admission_mode="expected_cost",
    )
    assert replay.actual_input_tokens == 80
    assert replay.actual_output_tokens == 20
    assert replay.actual_total_tokens == 100
    assert replay.selected_checkpoints.tolist() == [2]
    assert sum(int(row["admitted"]) for row in decisions) == 1


def test_grouped_cv_never_splits_checkpoints_from_the_same_task() -> None:
    observations = [
        make_observation(task, checkpoint)
        for task in range(4)
        for checkpoint in (1, 2, 3)
    ]
    for train_indices, validation_indices in _group_folds(observations, folds=2):
        train_groups = {observations[int(index)].group_key for index in train_indices}
        validation_groups = {
            observations[int(index)].group_key for index in validation_indices
        }
        assert train_groups.isdisjoint(validation_groups)


def test_retrospective_hard_cap_is_separate_from_predicted_admission() -> None:
    observations = [
        make_observation(
            0, checkpoint, predicted_cost=50.0, actual_input=160, actual_output=40
        )
        for checkpoint in (1, 2, 3)
    ]
    controller = FixedActionController(
        {1: (1.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 0.0)}
    )
    expected, _ = replay_controller(
        observations,
        controller,
        policy_name="expected",
        token_budget=100,
        seed=1,
        admission_mode="expected_cost",
    )
    hard, _ = replay_controller(
        observations,
        controller,
        policy_name="hard",
        token_budget=100,
        seed=1,
        admission_mode="retrospective_hard_cap",
    )
    assert expected.actual_total_tokens == 200
    assert hard.actual_total_tokens == 0
    assert hard.selected_checkpoints.tolist() == [0]
