"""Branch-consistent finite-horizon offline environment for CBV-Bench Level-2."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from agent_verify.analysis.cbv_v01_experiment import _passed
from agent_verify.analysis.darc_v04_data import MultiHistoryState, verify_cost

Action = Literal["S", "V"]


class InfeasibleOfflineAction(RuntimeError):
    """Raised when Verify cannot be safely admitted under the hard-call reserve."""


@dataclass(frozen=True)
class OfflineObservation:
    task_key: str
    prefix: str
    checkpoint: int
    global_remaining_budget: int
    per_task_allowance: float
    state_index: int


@dataclass(frozen=True)
class OfflineTransition:
    observation: OfflineObservation
    action: Action
    next_observation: OfflineObservation | None
    incremental_teacher_tokens: int
    terminal_correctness: int | None
    done: bool


@dataclass(frozen=True)
class RolloutResult:
    sequence: str
    final_correctness: int
    cumulative_teacher_tokens: int
    transitions: tuple[OfflineTransition, ...]


class CBVLevel2OfflineEnv:
    """Replays only one of the eight recorded shared-prefix trajectories.

    The policy receives :class:`OfflineObservation`, which contains a lookup into
    the separately constructed observable state.  Outcome and realized call cost
    are returned only after an action; neither is part of the observation.
    """

    def __init__(
        self,
        groups: dict[str, Any],
        state_lookup: dict[tuple[str, str], int],
        *,
        hard_call_token_cap: int,
    ) -> None:
        self.groups = groups
        self.state_lookup = state_lookup
        self.hard_call_token_cap = int(hard_call_token_cap)
        self.task_key: str | None = None
        self.prefix = ""
        self.global_remaining_budget = 0
        self.per_task_allowance = 0.0
        self.done = True

    def reset(
        self,
        task_key: str,
        global_remaining_budget: int,
        *,
        per_task_allowance: float | None = None,
        prefix: str = "",
    ) -> OfflineObservation:
        if task_key not in self.groups:
            raise KeyError(task_key)
        if prefix not in {"", "S", "V", "SS", "SV", "VS", "VV"}:
            raise ValueError(f"Invalid observable prefix {prefix!r}")
        if global_remaining_budget < 0:
            raise ValueError("Global remaining budget must be non-negative")
        if per_task_allowance is not None and per_task_allowance < 0:
            raise ValueError("Per-task allowance must be non-negative")
        self.task_key = task_key
        self.prefix = prefix
        self.global_remaining_budget = int(global_remaining_budget)
        self.per_task_allowance = float(
            global_remaining_budget
            if per_task_allowance is None
            else per_task_allowance
        )
        self.done = False
        return self.observation()

    def set_per_task_allowance(self, allowance: float) -> None:
        """Update the observable shadow allowance without changing global budget."""

        if allowance < 0:
            raise ValueError("Per-task allowance must be non-negative")
        self.per_task_allowance = float(allowance)

    def observation(self) -> OfflineObservation:
        if self.done or self.task_key is None:
            raise RuntimeError("Environment is terminal; call reset")
        return OfflineObservation(
            task_key=self.task_key,
            prefix=self.prefix,
            checkpoint=len(self.prefix) + 1,
            global_remaining_budget=self.global_remaining_budget,
            per_task_allowance=self.per_task_allowance,
            state_index=self.state_lookup[(self.task_key, self.prefix)],
        )

    def feasible(self, action: Action, *, predicted_call_tokens: float | None = None) -> bool:
        if action == "S":
            return True
        if self.global_remaining_budget < self.hard_call_token_cap:
            return False
        return (
            predicted_call_tokens is None
            or predicted_call_tokens <= self.global_remaining_budget
        )

    def step(self, action: Action) -> OfflineTransition:
        if action not in ("S", "V"):
            raise ValueError(f"Unknown action {action!r}")
        observation = self.observation()
        group = self.groups[observation.task_key]
        cost = 0
        if action == "V":
            if self.global_remaining_budget < self.hard_call_token_cap:
                raise InfeasibleOfflineAction("Verify lacks the hard-call budget reserve")
            # This realized label is obtained only after the action is fixed.
            cost = int(verify_cost(group, observation.prefix))
            if cost > self.hard_call_token_cap:
                raise InfeasibleOfflineAction(
                    f"Recorded call {cost} exceeds hard cap {self.hard_call_token_cap}"
                )
            if cost > self.global_remaining_budget:
                raise InfeasibleOfflineAction("Recorded call exceeds remaining actual budget")
            self.global_remaining_budget -= cost
            # Completed-call cost is observable after the action.  It may update
            # the shadow allowance for continuation, but never action admission
            # before the call.
            self.per_task_allowance = max(self.per_task_allowance - cost, 0.0)
        self.prefix += action
        self.done = len(self.prefix) == 3
        terminal = int(_passed(group.rows[self.prefix])) if self.done else None
        next_observation = None if self.done else self.observation()
        return OfflineTransition(
            observation=observation,
            action=action,
            next_observation=next_observation,
            incremental_teacher_tokens=cost,
            terminal_correctness=terminal,
            done=self.done,
        )


ContinuationPolicy = Callable[[OfflineObservation], Action]


def rollout_forced_action(
    env: CBVLevel2OfflineEnv,
    *,
    task_key: str,
    prefix: str,
    global_budget: int,
    per_task_allowance: float,
    forced_action: Action,
    continuation_policy: ContinuationPolicy,
) -> RolloutResult:
    """Force one action, then follow an observable continuation policy.

    There is intentionally no maximization over recorded descendants.  Once an
    action is chosen, the environment follows exactly that branch.
    """

    observation = env.reset(
        task_key,
        global_budget,
        per_task_allowance=per_task_allowance,
        prefix=prefix,
    )
    if not env.feasible(forced_action):
        raise InfeasibleOfflineAction(
            f"Forced {forced_action} infeasible at {task_key}:{prefix or 'START'}"
        )
    transitions = [env.step(forced_action)]
    while not transitions[-1].done:
        observation = transitions[-1].next_observation
        assert observation is not None
        action = continuation_policy(observation)
        if action == "V" and not env.feasible(action):
            action = "S"
        transitions.append(env.step(action))
    return RolloutResult(
        sequence=env.prefix,
        final_correctness=int(transitions[-1].terminal_correctness or 0),
        cumulative_teacher_tokens=sum(row.incremental_teacher_tokens for row in transitions),
        transitions=tuple(transitions),
    )


def state_by_index(states: list[MultiHistoryState]) -> dict[int, MultiHistoryState]:
    return {index: state for index, state in enumerate(states)}

