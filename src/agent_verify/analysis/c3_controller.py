"""Budget-conditioned fitted-Q controller for single-intervention verification.

The controller treats the pretrained Joint Value-Cost model as a frozen state
encoder. It learns Q(verify now) and Q(wait) backwards from CP3 to CP1. Future
checkpoint states are used only to construct fitted continuation targets during
training; inference receives one current checkpoint state at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor

from agent_verify.analysis.joint_value_cost_data import JointExample
from agent_verify.analysis.joint_value_cost_models import ValueCostPrediction


@dataclass(frozen=True)
class BudgetContext:
    """Deployable budget information available at the current decision."""

    budget_per_remaining_task: float
    remaining_fraction: float


@dataclass(frozen=True)
class ControllerObservation:
    """One current checkpoint plus frozen upstream predictions."""

    example: JointExample
    hidden_state: np.ndarray
    p_harmful: float
    p_no_change: float
    p_helpful: float
    expected_gain: float
    predicted_cost: float
    predicted_upper_cost: float

    @property
    def group_key(self) -> str:
        return self.example.group_key


@dataclass(frozen=True)
class CheckpointAction:
    checkpoint: int
    verify: bool
    predicted_feasible: bool
    predicted_cost: float
    q_verify: float
    q_wait: float


@dataclass(frozen=True)
class ControllerDecision:
    selected_index: int
    checkpoint: int
    predicted_cost: float
    q_verify: float
    q_wait: float
    trace: tuple[dict[str, float | int | bool], ...]


class RobustQRegressor:
    """Gradient boosting with a constant-target fallback."""

    def __init__(self, params: dict[str, Any], seed: int) -> None:
        self.params = dict(params)
        self.seed = seed
        self.model: GradientBoostingRegressor | DummyRegressor | None = None

    def fit(self, matrix: np.ndarray, target: np.ndarray) -> "RobustQRegressor":
        if len(target) == 0:
            raise ValueError("Cannot fit a Q regressor without targets")
        if float(np.ptp(target)) < 1e-12:
            self.model = DummyRegressor(strategy="constant", constant=float(target[0]))
        else:
            self.model = GradientBoostingRegressor(
                loss="huber",
                n_estimators=int(self.params.get("n_estimators", 120)),
                learning_rate=float(self.params.get("learning_rate", 0.03)),
                max_depth=int(self.params.get("max_depth", 2)),
                min_samples_leaf=int(self.params.get("min_samples_leaf", 10)),
                subsample=float(self.params.get("subsample", 0.9)),
                random_state=self.seed,
            )
        self.model.fit(matrix, target)
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Q regressor has not been fitted")
        return np.asarray(self.model.predict(matrix), dtype=float)


def extract_observations(
    examples: list[JointExample],
    joint_model: Any,
) -> list[ControllerObservation]:
    """Create current-state observations from a frozen pretrained Joint model."""

    matrix = joint_model.encoder.transform(examples)
    hidden, _raw_probabilities, _raw_costs = joint_model._forward(matrix)
    prediction: ValueCostPrediction = joint_model.predict(examples)
    observations: list[ControllerObservation] = []
    for index, example in enumerate(examples):
        probabilities = prediction.value_probabilities[index]
        observations.append(
            ControllerObservation(
                example=example,
                hidden_state=np.asarray(hidden[index], dtype=float),
                p_harmful=float(probabilities[0]),
                p_no_change=float(probabilities[1]),
                p_helpful=float(probabilities[2]),
                expected_gain=float(prediction.expected_gain[index]),
                predicted_cost=float(prediction.expected_total_tokens[index]),
                predicted_upper_cost=float(prediction.upper_total_tokens[index]),
            )
        )
    return observations


def group_observations(
    observations: Iterable[ControllerObservation],
) -> dict[str, list[ControllerObservation]]:
    grouped: dict[str, list[ControllerObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.group_key, []).append(observation)
    for group_key, rows in grouped.items():
        rows.sort(key=lambda row: row.example.checkpoint)
        checkpoints = [row.example.checkpoint for row in rows]
        if checkpoints != [1, 2, 3]:
            raise ValueError(f"Incomplete controller trajectory for {group_key}: {checkpoints}")
        baseline = {row.example.y_skip for row in rows}
        if len(baseline) != 1:
            raise ValueError(f"Inconsistent Y_skip for {group_key}")
    return grouped


def controller_feature_vector(
    observation: ControllerObservation,
    context: BudgetContext,
) -> np.ndarray:
    """Use only the current observable state, frozen predictions, and current budget."""

    cost = max(observation.predicted_cost, 1.0)
    allowance = max(context.budget_per_remaining_task, 0.0)
    checkpoint_one_hot = [
        float(observation.example.checkpoint == checkpoint) for checkpoint in (1, 2, 3)
    ]
    explicit = np.asarray(
        [
            *checkpoint_one_hot,
            observation.example.local_risk,
            observation.p_harmful,
            observation.p_no_change,
            observation.p_helpful,
            observation.expected_gain,
            np.log1p(cost),
            np.log1p(max(observation.predicted_upper_cost, 0.0)),
            np.log1p(allowance),
            allowance / cost,
            cost / max(allowance, 1.0),
            float(cost <= allowance),
            float(np.clip(context.remaining_fraction, 0.0, 1.0)),
        ],
        dtype=float,
    )
    return np.concatenate([observation.hidden_state, explicit])


def verify_utility_target(
    observation: ControllerObservation,
    context: BudgetContext,
    *,
    cost_penalty_scale: float,
    reference_cost: float,
) -> float:
    """Observed single-intervention correctness minus a budget-dependent cost price."""

    denominator = max(context.budget_per_remaining_task, reference_cost, 1.0)
    cost_penalty = (
        float(cost_penalty_scale)
        * observation.example.teacher_total_tokens
        / denominator
    )
    return float(observation.example.y_verify) - cost_penalty


class FittedSequentialController:
    """Checkpoint-wise Q models trained CP3 -> CP2 -> CP1."""

    def __init__(
        self,
        params: dict[str, Any],
        *,
        seed: int,
        use_continuation: bool,
        min_action_margin: float = 0.0,
    ) -> None:
        self.params = dict(params)
        self.seed = seed
        self.use_continuation = use_continuation
        self.min_action_margin = float(min_action_margin)
        self.reference_cost = 1.0
        self.verify_models: dict[int, RobustQRegressor] = {}
        self.wait_models: dict[int, RobustQRegressor] = {}

    def _matrix(
        self,
        observations: list[ControllerObservation],
        contexts: list[BudgetContext],
    ) -> np.ndarray:
        return np.vstack(
            [
                controller_feature_vector(observation, context)
                for observation, context in zip(observations, contexts, strict=True)
            ]
        )

    def action_values(
        self,
        observation: ControllerObservation,
        context: BudgetContext,
    ) -> tuple[float, float]:
        checkpoint = observation.example.checkpoint
        if checkpoint not in self.verify_models or checkpoint not in self.wait_models:
            raise RuntimeError(f"Checkpoint CP{checkpoint} models are unavailable")
        matrix = controller_feature_vector(observation, context)[None, :]
        q_verify = float(self.verify_models[checkpoint].predict(matrix)[0])
        q_wait = float(self.wait_models[checkpoint].predict(matrix)[0])
        return q_verify, q_wait

    def fit(
        self,
        observations: list[ControllerObservation],
        *,
        budget_levels: list[float],
        remaining_fractions: list[float],
    ) -> "FittedSequentialController":
        grouped = group_observations(observations)
        self.reference_cost = float(
            np.median([row.example.teacher_total_tokens for row in observations])
        )
        contexts = [
            BudgetContext(float(budget), float(fraction))
            for budget in budget_levels
            for fraction in remaining_fractions
        ]
        cost_scale = float(self.params.get("cost_penalty_scale", 0.07))

        for checkpoint in (3, 2, 1):
            stage_observations: list[ControllerObservation] = []
            stage_contexts: list[BudgetContext] = []
            verify_targets: list[float] = []
            wait_targets: list[float] = []
            for task_rows in grouped.values():
                current = task_rows[checkpoint - 1]
                for context in contexts:
                    stage_observations.append(current)
                    stage_contexts.append(context)
                    verify_targets.append(
                        verify_utility_target(
                            current,
                            context,
                            cost_penalty_scale=cost_scale,
                            reference_cost=self.reference_cost,
                        )
                    )
                    if checkpoint == 3 or not self.use_continuation:
                        wait_targets.append(float(current.example.y_skip))
                        continue
                    future = task_rows[checkpoint]
                    q_verify, q_wait = self.action_values(future, context)
                    future_feasible = future.predicted_cost <= max(
                        context.budget_per_remaining_task, 0.0
                    )
                    continuation = max(q_wait, q_verify) if future_feasible else q_wait
                    wait_targets.append(float(continuation))

            matrix = self._matrix(stage_observations, stage_contexts)
            self.verify_models[checkpoint] = RobustQRegressor(
                self.params, self.seed + checkpoint * 101
            ).fit(matrix, np.asarray(verify_targets, dtype=float))
            self.wait_models[checkpoint] = RobustQRegressor(
                self.params, self.seed + checkpoint * 101 + 1
            ).fit(matrix, np.asarray(wait_targets, dtype=float))
        return self

    @staticmethod
    def budget_context(
        *,
        remaining_budget: float,
        initial_budget: float,
        tasks_remaining: int,
    ) -> BudgetContext:
        return BudgetContext(
            budget_per_remaining_task=max(float(remaining_budget), 0.0)
            / max(int(tasks_remaining), 1),
            remaining_fraction=(
                max(float(remaining_budget), 0.0) / max(float(initial_budget), 1.0)
            ),
        )

    def decide_current(
        self,
        observation: ControllerObservation,
        *,
        remaining_budget: float,
        initial_budget: float,
        tasks_remaining: int,
    ) -> CheckpointAction:
        """Choose from one current checkpoint without accepting future state."""

        context = self.budget_context(
            remaining_budget=remaining_budget,
            initial_budget=initial_budget,
            tasks_remaining=tasks_remaining,
        )
        q_verify, q_wait = self.action_values(observation, context)
        feasible = observation.predicted_cost <= max(float(remaining_budget), 0.0)
        verify = feasible and q_verify > q_wait + self.min_action_margin
        return CheckpointAction(
            checkpoint=observation.example.checkpoint,
            verify=verify,
            predicted_feasible=feasible,
            predicted_cost=observation.predicted_cost,
            q_verify=q_verify,
            q_wait=q_wait,
        )

    def decide(
        self,
        task_observations: list[ControllerObservation],
        *,
        remaining_budget: float,
        initial_budget: float,
        tasks_remaining: int,
    ) -> ControllerDecision:
        """Offline sequential driver; each action receives only its current row."""

        rows = sorted(task_observations, key=lambda row: row.example.checkpoint)
        if [row.example.checkpoint for row in rows] != [1, 2, 3]:
            raise ValueError("Sequential inference requires CP1, CP2, and CP3")
        trace: list[dict[str, float | int | bool]] = []
        final_action: CheckpointAction | None = None
        for index, observation in enumerate(rows):
            action = self.decide_current(
                observation,
                remaining_budget=remaining_budget,
                initial_budget=initial_budget,
                tasks_remaining=tasks_remaining,
            )
            final_action = action
            trace.append(
                {
                    "checkpoint": action.checkpoint,
                    "q_verify": action.q_verify,
                    "q_wait": action.q_wait,
                    "predicted_cost": action.predicted_cost,
                    "predicted_feasible": action.predicted_feasible,
                    "verify": action.verify,
                }
            )
            if action.verify:
                return ControllerDecision(
                    selected_index=index,
                    checkpoint=action.checkpoint,
                    predicted_cost=action.predicted_cost,
                    q_verify=action.q_verify,
                    q_wait=action.q_wait,
                    trace=tuple(trace),
                )
        if final_action is None:
            raise AssertionError("A controller trajectory cannot be empty")
        return ControllerDecision(
            selected_index=-1,
            checkpoint=0,
            predicted_cost=0.0,
            q_verify=final_action.q_verify,
            q_wait=final_action.q_wait,
            trace=tuple(trace),
        )
