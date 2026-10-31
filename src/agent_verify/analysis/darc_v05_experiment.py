"""Validation-only DARC v0.5 conservative offline-RL experiment.

The module deliberately never loads the frozen test partition.  It evaluates
every controller with the same task order, hard-call reserve, and actual-token
ledger.  Recorded outcomes and costs are visible to target generation and
post-action accounting, never to the online policy.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm

from agent_verify.analysis.cbv_v01_experiment import (
    MultiHistoryController,
    ReplayResult,
    _passed,
    _stable_order,
)
from agent_verify.analysis.darc_data import EncodedDARC, encode_or_load
from agent_verify.analysis.darc_v04_audit import (
    _seed_summary,
    _trace_counts,
    aggregate_records,
    paired_comparison,
    replay_c3_strict,
    replay_fixed_strict,
)
from agent_verify.analysis.darc_v04_data import (
    MultiHistoryState,
    audit_level2,
    build_states,
)
from agent_verify.analysis.darc_v04_experiment import (
    SequentialPolicy,
    _assert_disjoint,
    _load_groups,
    _precompute,
    _write_csv,
    replay_v04,
    select_development_candidate,
)
from agent_verify.analysis.darc_v04_model import DARCv04Ensemble, V04Prediction
from agent_verify.analysis.darc_v05_env import CBVLevel2OfflineEnv, OfflineObservation
from agent_verify.analysis.darc_v05_model import DARCv05Ensemble, V05Prediction
from agent_verify.analysis.darc_v05_training import (
    ConservativePolicy,
    assign_task_folds,
    generate_policy_targets,
    policy_action,
    policy_grid,
    train_crossfit_iteration,
    validate_target_coverage,
)


class NullWriter:
    """TensorBoard-compatible no-op writer used only by synthetic smoke tests."""

    def add_scalar(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def for_run(self, _name: str) -> NullWriter:
        return self

    def close(self) -> None:
        pass


class TensorBoardRunRouter:
    """Keep metric tags stable and put experimental dimensions in run names.

    TensorBoard creates one card per scalar tag.  Routing fold/seed/variant to
    sub-runs avoids hundreds of nearly identical cards while preserving every
    individual learning curve in the run selector.
    """

    def __init__(self, root: Path, writer_class: Any) -> None:
        self.root = root
        self.writer_class = writer_class
        self.writers: dict[str, Any] = {}

    @staticmethod
    def _clean(name: str) -> str:
        return name.replace(" · ", "/").replace(" ", "_").replace("@", "_")

    def for_run(self, name: str) -> Any:
        clean = self._clean(name).strip("/")
        if clean not in self.writers:
            self.writers[clean] = self.writer_class(log_dir=str(self.root / clean))
        return self.writers[clean]

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()


def _writer(path: Path, *, allow_missing: bool) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter

        return TensorBoardRunRouter(path, SummaryWriter)
    except (ImportError, ModuleNotFoundError) as error:
        if allow_missing:
            return NullWriter()
        raise RuntimeError(
            "TensorBoard is mandatory for full DARC v0.5 training. Install the "
            "updated environment/requirements before running the full pipeline."
        ) from error


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _selected_v04_policy(root: Path, seed: int, variant: str) -> SequentialPolicy:
    label = f"DARC v0.4 · {variant}"
    rows = [
        row
        for row in _read_csv(root / "tables" / "policy_grid_results.csv")
        if int(row["seed"]) == seed
        and row["variant"] == label
        and str(row["selected"]).lower() in {"1", "true"}
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one selected v0.4 policy for {seed}/{variant}")
    row = rows[0]
    return SequentialPolicy(
        harmful_weight=float(row["harmful_weight"]),
        uncertainty_beta=float(row["uncertainty_beta"]),
        cost_lambda=float(row["cost_lambda"]),
        decision_margin=float(row["decision_margin"]),
    )


def _v04_continuation_factory(
    predictions: list[V04Prediction],
    allowances: list[float],
    policy: SequentialPolicy,
    *,
    reference_budget: float,
    hard_cap: int,
) -> Callable[[MultiHistoryState, int], Callable[[OfflineObservation], str]]:
    def factory(_state: MultiHistoryState, _allowance: int) -> Callable[[OfflineObservation], str]:
        def continuation(observation: OfflineObservation) -> str:
            prediction = predictions[
                int(
                    np.argmin(
                        np.abs(
                            np.asarray(allowances)
                            - observation.per_task_allowance
                        )
                    )
                )
            ]
            index = observation.state_index
            expected_cost = max(float(prediction.expected_cost[index]), 1.0)
            utility = (
                float(prediction.q_advantage[index])
                - policy.harmful_weight * float(prediction.effect_probabilities[index, 0])
                - policy.uncertainty_beta * float(prediction.uncertainty[index])
                - policy.cost_lambda
                * expected_cost
                / max(float(observation.per_task_allowance), 1.0)
            )
            feasible = (
                observation.global_remaining_budget >= hard_cap
                and expected_cost <= observation.global_remaining_budget
            )
            return "V" if feasible and utility > policy.decision_margin else "S"

        return continuation

    return factory


def _v05_continuation_factory(
    predictions_by_fold: dict[int, list[V05Prediction]],
    allowances: list[float],
    task_folds: dict[str, int],
    policy: ConservativePolicy,
    *,
    reference_budget: float,
    hard_cap: int,
) -> Callable[[MultiHistoryState, int], Callable[[OfflineObservation], str]]:
    def factory(state: MultiHistoryState, _allowance: int) -> Callable[[OfflineObservation], str]:
        fold = task_folds[state.task_key]

        def continuation(observation: OfflineObservation) -> str:
            return policy_action(
                observation,
                predictions=predictions_by_fold[fold],
                allowances=allowances,
                policy=policy,
                reference_budget=reference_budget,
                hard_call_token_cap=hard_cap,
            )

        return continuation

    return factory


def _precompute_v05(
    model: DARCv05Ensemble,
    encoded: EncodedDARC,
    states: list[MultiHistoryState],
    allowances: list[float],
    *,
    device: str,
    description: str,
) -> list[V05Prediction]:
    return [
        model.predict(encoded, states, allowance=value, device_name=device)
        for value in tqdm(allowances, desc=description, unit="budget", leave=False)
    ]


def replay_v05(
    groups: dict[str, Any],
    state_lookup: dict[tuple[str, str], int],
    predictions: list[V05Prediction],
    allowances: list[float],
    policy: ConservativePolicy,
    *,
    reference_budget: float,
    token_budget: int,
    budget_fraction: float,
    hard_call_token_cap: int,
    seed: int,
    label: str,
) -> tuple[ReplayResult, list[dict[str, Any]]]:
    """Strict global-budget replay using no realized future information online."""

    keys = _stable_order(groups, seed)
    environment = CBVLevel2OfflineEnv(
        groups, state_lookup, hard_call_token_cap=hard_call_token_cap
    )
    actual_spent = calls = 0
    predicted_spent = 0.0
    outcomes: list[float] = []
    baselines: list[float] = []
    masks: list[int] = []
    traces: list[dict[str, Any]] = []
    for order, key in enumerate(keys):
        remaining = max(token_budget - actual_spent, 0)
        task_allowance = remaining / max(len(keys) - order, 1)
        observation = environment.reset(
            key,
            remaining,
            per_task_allowance=task_allowance,
        )
        mask = 0
        while True:
            task_allowance = remaining / max(len(keys) - order, 1)
            environment.set_per_task_allowance(task_allowance)
            observation = environment.observation()
            action = policy_action(
                observation,
                predictions=predictions,
                allowances=allowances,
                policy=policy,
                reference_budget=reference_budget,
                hard_call_token_cap=hard_call_token_cap,
            )
            prediction = predictions[
                int(np.argmin(np.abs(np.asarray(allowances) - task_allowance)))
            ]
            index = observation.state_index
            predicted_cost = max(float(prediction.immediate_cost[index]), 1.0)
            direct_skip = observation.prefix + "S" + "S" * (2 - len(observation.prefix))
            direct_verify = observation.prefix + "V" + "S" * (2 - len(observation.prefix))
            delta = int(
                _passed(groups[key].rows[direct_verify])
                - _passed(groups[key].rows[direct_skip])
            )
            before = actual_spent
            transition = environment.step(action)
            actual_spent += transition.incremental_teacher_tokens
            remaining = max(token_budget - actual_spent, 0)
            if actual_spent > token_budget:
                raise AssertionError("DARC v0.5 exceeded the strict actual-token budget")
            if action == "V":
                predicted_spent += predicted_cost
                calls += 1
                mask |= 1 << (observation.checkpoint - 1)
            score = policy.score(
                prediction,
                index,
                allowance=task_allowance,
                reference_budget=reference_budget,
            )
            traces.append(
                {
                    "policy": label,
                    "seed": seed,
                    "budget_fraction": budget_fraction,
                    "task_key": key,
                    "dataset": groups[key].task.dataset,
                    "checkpoint": observation.checkpoint,
                    "history": observation.prefix or "START",
                    "action": action,
                    "remaining_global_budget_before_action": token_budget - before,
                    "per_task_allowance": task_allowance,
                    "predicted_immediate_cost": predicted_cost,
                    "actual_cost_after_action": transition.incremental_teacher_tokens,
                    "predicted_advantage": float(prediction.advantage[index]),
                    "advantage_uncertainty": float(
                        prediction.advantage_uncertainty[index]
                    ),
                    "predicted_harmful": float(
                        prediction.effect_probabilities[index, 0]
                    ),
                    "decision_score": score,
                    "direct_actual_delta_audit_only": delta,
                    "helpful_selected": int(action == "V" and delta > 0),
                    "harmful_selected": int(action == "V" and delta < 0),
                    "no_change_selected": int(action == "V" and delta == 0),
                }
            )
            if transition.done:
                outcomes.append(float(transition.terminal_correctness or 0))
                break
            assert transition.next_observation is not None
            observation = transition.next_observation
        baselines.append(float(_passed(groups[key].rows["SSS"])))
        masks.append(mask)
    return (
        ReplayResult(
            label,
            "C-multi-strict-v05",
            "combined",
            "combined",
            budget_fraction,
            token_budget,
            tuple(keys),
            np.asarray(outcomes),
            np.asarray(baselines),
            np.asarray(masks),
            calls,
            actual_spent,
            predicted_spent,
        ),
        traces,
    )


def _tune_v05_policy(
    groups: dict[str, Any],
    lookup: dict[tuple[str, str], int],
    predictions: list[V05Prediction],
    allowances: list[float],
    policies: list[ConservativePolicy],
    *,
    fractions: list[float],
    reference_budget: float,
    hard_cap: int,
    seed: int,
    label: str,
    token_penalty: float,
    max_harmful_rate: float,
) -> tuple[ConservativePolicy, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    total_budget = sum(
        int(round(fraction * reference_budget * len(groups)))
        for fraction in fractions
        if fraction > 0
    )
    for candidate, policy in enumerate(
        tqdm(policies, desc=f"Tune {label}", unit="policy", leave=False)
    ):
        accuracies: list[float] = []
        tokens = calls = helpful = harmful = no_change = 0
        for fraction in fractions:
            if fraction <= 0:
                continue
            replay, traces = replay_v05(
                groups,
                lookup,
                predictions,
                allowances,
                policy,
                reference_budget=reference_budget,
                token_budget=int(round(fraction * reference_budget * len(groups))),
                budget_fraction=fraction,
                hard_call_token_cap=hard_cap,
                seed=seed,
                label=label,
            )
            h, r, n = _trace_counts(traces)
            accuracies.append(float(replay.outcomes.mean()))
            tokens += replay.actual_tokens
            calls += replay.calls
            helpful += h
            harmful += r
            no_change += n
        accuracy = float(np.mean(accuracies))
        utilization = tokens / max(total_budget, 1)
        harmful_rate = harmful / max(calls, 1)
        rows.append(
            {
                "seed": seed,
                "variant": label,
                "candidate": candidate,
                **asdict(policy),
                "mean_validation_accuracy": accuracy,
                "actual_tokens": tokens,
                "token_budget": total_budget,
                "token_utilization": utilization,
                "calls": calls,
                "helpful": helpful,
                "harmful": harmful,
                "no_change": no_change,
                "harmful_rate": harmful_rate,
                "risk_eligible": harmful_rate <= max_harmful_rate + 1e-12,
                "selection_score": accuracy - token_penalty * utilization,
            }
        )
    eligible = [row for row in rows if bool(row["risk_eligible"])]
    if not eligible:
        minimum = min(float(row["harmful_rate"]) for row in rows)
        eligible = [row for row in rows if float(row["harmful_rate"]) == minimum]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["mean_validation_accuracy"]),
            -float(row["actual_tokens"]),
            -int(row["candidate"]),
        ),
    )
    for row in rows:
        row["selected"] = int(row is selected)
    fields = ConservativePolicy.__dataclass_fields__
    return ConservativePolicy(**{key: selected[key] for key in fields}), rows


def _record(
    records: list[dict[str, Any]],
    replay: ReplayResult,
    traces: list[dict[str, Any]],
    *,
    seed: int,
) -> None:
    summary = _seed_summary(replay, traces, seed=seed)
    records.append(
        {
            "policy": replay.policy,
            "fraction": replay.budget_fraction,
            "seed": seed,
            "accuracy": summary["accuracy"],
            "summary": summary,
            "replay": replay,
        }
    )


def _append_target_diagnostics(
    rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    *,
    split: str,
    variant: str,
    seed: int,
    iteration: int,
) -> None:
    """Flatten fail-fast target coverage diagnostics for reproducible audit."""

    for allowance, bucket in diagnostics["by_allowance"].items():
        rows.append(
            {
                "split": split,
                "variant": variant,
                "seed": seed,
                "iteration": iteration,
                "allowance": float(allowance),
                "rows": int(bucket["rows"]),
                "verify_supervised": int(bucket["verify_supervised"]),
                "verify_coverage": float(bucket["coverage"]),
                "positive_budget_rows_total": int(
                    diagnostics["positive_budget_rows"]
                ),
                "verify_supervised_rows_total": int(
                    diagnostics["verify_supervised_rows"]
                ),
                "verify_supervision_rate_positive": float(
                    diagnostics["verify_supervision_rate_positive"]
                ),
            }
        )


def _evaluate_v05(
    records: list[dict[str, Any]],
    traces_out: list[dict[str, Any]],
    *,
    groups: dict[str, Any],
    lookup: dict[tuple[str, str], int],
    predictions: list[V05Prediction],
    allowances: list[float],
    policy: ConservativePolicy,
    fractions: list[float],
    reference_budget: float,
    hard_cap: int,
    seed: int,
    label: str,
    iteration: int,
    writer: Any,
) -> None:
    policy_writer = writer.for_run(
        f"{label}/seed_{seed}/policy_evaluation"
    )
    for fraction in fractions:
        budget = int(round(fraction * reference_budget * len(groups)))
        replay, traces = replay_v05(
            groups,
            lookup,
            predictions,
            allowances,
            policy,
            reference_budget=reference_budget,
            token_budget=budget,
            budget_fraction=fraction,
            hard_call_token_cap=hard_cap,
            seed=seed,
            label=label,
        )
        for row in traces:
            row["iteration"] = iteration
        traces_out.extend(traces)
        _record(records, replay, traces, seed=seed)
        h, r, n = _trace_counts(traces)
        budget_step = int(round(100 * fraction))
        policy_writer.add_scalar(
            "policy/accuracy", float(replay.outcomes.mean()), budget_step
        )
        policy_writer.add_scalar("policy/actual_teacher_tokens", replay.actual_tokens, budget_step)
        policy_writer.add_scalar("policy/teacher_calls", replay.calls, budget_step)
        policy_writer.add_scalar(
            "policy/budget_utilization", replay.actual_tokens / max(budget, 1), budget_step
        )
        policy_writer.add_scalar("policy/helpful", h, budget_step)
        policy_writer.add_scalar("policy/harmful", r, budget_step)
        policy_writer.add_scalar("policy/no_change", n, budget_step)
        policy_writer.add_scalar(
            "policy/ensemble_advantage_uncertainty",
            float(np.mean(predictions[-1].advantage_uncertainty)),
            budget_step,
        )
        policy_writer.add_scalar(
            "policy/improvement_over_never",
            float(replay.outcomes.mean() - replay.baseline.mean()),
            budget_step,
        )


def _baseline_records(
    groups: dict[str, Any],
    states: list[MultiHistoryState],
    lookup: dict[tuple[str, str], int],
    encoded: EncodedDARC,
    *,
    config: dict[str, Any],
    reference_budget: float,
    allowances: list[float],
    seeds: list[int],
    fractions: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    traces_out: list[dict[str, Any]] = []
    hard_cap = int(config["offline_rl"]["hard_call_token_cap"])
    device = str(config["semantic_encoder"]["device"])
    v04_root = Path(config["data"]["darc_v04_audit"])
    c3_root = Path(config["data"]["c3_results"])
    for seed in tqdm(seeds, desc="Strict baseline replay", unit="seed"):
        for variant in ("myopic_targets", "full_pretrained"):
            model = DARCv04Ensemble.load(v04_root / "models" / f"seed_{seed}" / f"{variant}.pt")
            predictions = _precompute(model, encoded, states, allowances, device=device)
            policy = _selected_v04_policy(v04_root, seed, variant)
            label = f"DARC v0.4 · {variant}"
            for fraction in fractions:
                replay, traces = replay_v04(
                    groups,
                    states,
                    lookup,
                    predictions,
                    allowances,
                    policy,
                    token_budget=int(round(fraction * reference_budget * len(groups))),
                    budget_fraction=fraction,
                    hard_call_token_cap=hard_cap,
                    seed=seed,
                    label=label,
                )
                _record(records, replay, traces, seed=seed)
                traces_out.extend(traces)
        for filename, label in (
            ("multi_myopic.joblib", "C3 MultiHistory Myopic"),
            ("multi_continuation.joblib", "C3 MultiHistory Continuation"),
        ):
            controller: MultiHistoryController = joblib.load(
                c3_root / "models" / "combined" / filename
            )
            for fraction in fractions:
                replay, traces = replay_c3_strict(
                    groups,
                    controller,
                    token_budget=int(round(fraction * reference_budget * len(groups))),
                    budget_fraction=fraction,
                    hard_call_token_cap=hard_cap,
                    seed=seed,
                    label=label,
                )
                _record(records, replay, traces, seed=seed)
                traces_out.extend(traces)
        for sequence in config["evaluation"]["include_fixed_sequences"]:
            for fraction in fractions:
                replay, traces = replay_fixed_strict(
                    groups,
                    str(sequence),
                    token_budget=int(round(fraction * reference_budget * len(groups))),
                    budget_fraction=fraction,
                    hard_call_token_cap=hard_cap,
                    seed=seed,
                )
                _record(records, replay, traces, seed=seed)
                traces_out.extend(traces)
    return records, traces_out


def _figures(
    output: Path,
    training: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for variant in sorted({str(row["variant"]) for row in training}):
        rows = [
            row
            for row in training
            if row["variant"] == variant and int(row["fold"]) == 0 and int(row["member"]) == 0
        ]
        if not rows:
            continue
        x = [int(row["epoch"]) for row in rows]
        axes[0, 0].plot(x, [float(row["total_loss"]) for row in rows], label=variant)
        axes[0, 0].plot(
            x,
            [float(row["validation_total_loss"]) for row in rows],
            linestyle="--",
        )
        axes[0, 1].plot(x, [float(row["validation_q_success_loss"]) for row in rows], label=variant)
    for policy in sorted({str(row["policy"]) for row in aggregate}):
        points = sorted(
            [row for row in aggregate if row["policy"] == policy],
            key=lambda row: float(row["teacher_tokens_mean"]),
        )
        if points:
            axes[1, 0].plot(
                [float(row["teacher_tokens_mean"]) / 1000 for row in points],
                [100 * float(row["accuracy_mean"]) for row in points],
                marker="o",
                label=policy,
            )
    v05_traces = [row for row in traces if str(row["policy"]).startswith("DARC v0.5")]
    labels = ["Helpful", "Harmful", "No-change"]
    values = [
        sum(int(row.get("helpful_selected", 0)) for row in v05_traces),
        sum(int(row.get("harmful_selected", 0)) for row in v05_traces),
        sum(int(row.get("no_change_selected", 0)) for row in v05_traces),
    ]
    axes[1, 1].bar(labels, values, color=["#269E6A", "#D64F5F", "#93A1B3"])
    axes[0, 0].set_title("Train (solid) vs validation (dashed) loss")
    axes[0, 1].set_title("Validation Q-success loss")
    axes[1, 0].set_title("Validation accuracy–actual-token frontier")
    axes[1, 1].set_title("Selected verification behavior")
    axes[1, 0].set_xlabel("Actual Teacher tokens (thousands)")
    axes[1, 0].set_ylabel("Accuracy (%)")
    for axis in axes.flat:
        axis.grid(color="#E4E8EE", linewidth=0.7)
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[1, 0].legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "darc_v05_diagnostics.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "darc_v05_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)

    # Individual paper/report assets make each mandatory diagnostic reusable.
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    for variant in sorted({str(row["variant"]) for row in training}):
        rows = [
            row
            for row in training
            if row["variant"] == variant
            and int(row["fold"]) == 0
            and int(row["member"]) == 0
        ]
        axis.plot(
            [int(row["epoch"]) for row in rows],
            [float(row["total_loss"]) for row in rows],
            label=f"{variant} train",
        )
        axis.plot(
            [int(row["epoch"]) for row in rows],
            [float(row["validation_total_loss"]) for row in rows],
            linestyle="--",
            label=f"{variant} validation",
        )
    axis.set(title="Training convergence", xlabel="Epoch", ylabel="Loss")
    axis.grid(color="#E4E8EE")
    axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "training_loss.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    iteration_loss: dict[tuple[str, int], list[float]] = {}
    for row in training:
        iteration_loss.setdefault(
            (str(row["variant"]), int(row["iteration"])), []
        ).append(float(row["validation_q_success_loss"]))
    for variant in sorted({key[0] for key in iteration_loss}):
        points = sorted(
            (iteration, float(np.mean(values)))
            for (name, iteration), values in iteration_loss.items()
            if name == variant
        )
        axis.plot(*zip(*points, strict=True), marker="o", label=variant)
    axis.set(
        title="Q-success loss across policy-improvement iterations",
        xlabel="Policy iteration",
        ylabel="Validation Q-success loss",
    )
    axis.grid(color="#E4E8EE")
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "q_loss_by_iteration.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    for policy in sorted(
        {str(row["policy"]) for row in aggregate if str(row["policy"]).startswith("DARC v0.5")}
    ):
        rows = [
            row
            for row in aggregate
            if row["policy"] == policy and float(row["budget_fraction"]) == 1.0
        ]
        if rows:
            iteration = int(policy.rsplit("i", 1)[-1])
            axis.scatter(iteration, 100 * float(rows[0]["accuracy_mean"]), label=policy)
    axis.set(
        title="Validation accuracy across policy iterations",
        xlabel="Policy iteration",
        ylabel="Accuracy (%)",
    )
    axis.grid(color="#E4E8EE")
    axis.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "validation_accuracy_by_iteration.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.0, 5.5))
    for policy in sorted({str(row["policy"]) for row in aggregate}):
        rows = sorted(
            [row for row in aggregate if row["policy"] == policy],
            key=lambda row: float(row["teacher_tokens_mean"]),
        )
        axis.plot(
            [float(row["teacher_tokens_mean"]) / 1000 for row in rows],
            [100 * float(row["accuracy_mean"]) for row in rows],
            marker="o",
            label=policy,
        )
    axis.set(
        title="Strict validation accuracy–cost frontier",
        xlabel="Actual Teacher tokens (thousands)",
        ylabel="Accuracy (%)",
    )
    axis.grid(color="#E4E8EE")
    axis.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "accuracy_vs_actual_tokens.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, values, color=["#269E6A", "#D64F5F", "#93A1B3"])
    uncertainties = [
        float(row["advantage_uncertainty"])
        for row in v05_traces
        if "advantage_uncertainty" in row
    ]
    axes[1].hist(uncertainties, bins=20, color="#527AA3", alpha=0.9)
    axes[0].set_title("Selected verification outcomes")
    axes[1].set_title("Ensemble advantage uncertainty")
    axes[1].set_xlabel("Ensemble standard deviation")
    for axis in axes:
        axis.grid(color="#E4E8EE", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(figures / "verification_behavior_uncertainty.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _report(
    output: Path,
    config: dict[str, Any],
    aggregate: list[dict[str, Any]],
    selected: str,
    audit: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> Path:
    full = sorted(
        [row for row in aggregate if float(row["budget_fraction"]) == 1.0],
        key=lambda row: float(row["accuracy_mean"]),
        reverse=True,
    )
    lines = [
        "# DARC v0.5 — Conservative Budget-Aware Offline RL",
        "",
        (
            "> Validation-only development report. The frozen test partition was "
            "never loaded. No Student/Teacher inference was performed."
        ),
        "",
        "## Method",
        "",
        (
            "For each observable shared prefix and budget bucket, policy evaluation "
            "forces Skip or Verify and follows the current observable continuation "
            "policy. Targets are recorded final correctness and cumulative future "
            "Teacher tokens along that single branch. There is no maximum over "
            "task-specific descendants. After initialization, fold-specific models "
            "that excluded the target task generate train targets."
        ),
        "",
        (
            "The online score is an empirical lower-confidence advantage minus "
            "harmful-risk and dynamic token-price penalties. Predicted immediate cost "
            "plus a hard-call reserve admits the action; realized cost is charged only "
            "after the action is fixed. This is conservative empirically, not a formal "
            "statistical coverage guarantee."
        ),
        "",
        "## Data audit",
        "",
        "| Split | Dataset | Tasks | 8-branch complete | Token-valid |",
        "|---|---|---:|---:|---:|",
    ]
    for row in audit:
        lines.append(
            f"| {row['split']} | {row['dataset']} | {row['tasks']} | "
            f"{row['sequence_complete']} | {row['token_valid']} |"
        )
    lines.extend(
        [
            "",
            (
                f"Development-selected candidate: **{selected}**. Selection uses "
                "validation accuracy minus the predeclared actual-token utilization "
                "penalty, subject to the harmful-rate gate."
            ),
            "",
            "## Full-budget validation results",
            "",
            (
                "| Policy | Accuracy mean (95% task CI) | Actual tokens | Calls | "
                "Helpful / Harmful / No-change |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in full:
        lines.append(
            f"| {row['policy']} | {100*float(row['accuracy_mean']):.2f}% "
            f"[{100*float(row['accuracy_ci95_low']):.2f}, "
            f"{100*float(row['accuracy_ci95_high']):.2f}] | "
            f"{float(row['teacher_tokens_mean']):,.0f} | "
            f"{float(row['teacher_calls_mean']):.1f} | "
            f"{float(row['helpful_mean']):.1f} / "
            f"{float(row['harmful_mean']):.1f} / "
            f"{float(row['no_change_mean']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Candidate − reference | Accuracy Δ pp (95% CI) | p | Token Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['candidate']} − {row['reference']} | "
            f"{float(row['accuracy_difference_pp']):+.2f} "
            f"[{float(row['difference_ci95_low_pp']):+.2f}, "
            f"{float(row['difference_ci95_high_pp']):+.2f}] | "
            f"{float(row['paired_permutation_p']):.4f} | "
            f"{float(row['token_difference_mean']):+,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            (
                "- The first policy-evaluation iteration is initialized from the "
                "existing v0.4 Myopic fit; later iterations use out-of-fold "
                "continuation predictions."
            ),
            (
                "- Level-2 supplies complete branches for this fixed three-checkpoint "
                "workflow only; it cannot simulate unrecorded actions, truncated "
                "calls, or another model pair."
            ),
            (
                "- Policy selection and reporting share validation data, so "
                "conclusions are development evidence, not confirmatory test claims."
            ),
            (
                "- Hard-cap reservation is conservative: it may reject a call whose "
                "unknown realized cost would have fit."
            ),
            "",
            "![DARC v0.5 diagnostics](../figures/darc_v05_diagnostics.png)",
            "",
            "## Reproducible artifacts",
            "",
            "- `checkpoints/<variant>/seed_<seed>/iteration_<k>.pt`",
            (
                "- `tables/training_history.csv`, `policy_grid.csv`, "
                "`target_coverage.csv`, `seed_results.csv`, `aggregate_results.csv`, "
                "`paired_comparisons.csv`, `decision_trace.csv`"
            ),
            "- `tensorboard/` and `figures/`",
            "- `selection_manifest.json` (explicitly records that frozen test was not loaded)",
        ]
    )
    path = output / "report" / "darc_v05_offline_rl_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(config_path: Path, *, smoke: bool = False) -> Path:
    config = yaml.safe_load(config_path.read_text())
    if config["evaluation"]["partition"] != "validation":
        raise ValueError("DARC v0.5 is validation-only")
    if config["evaluation"]["frozen_test_access"] != "forbidden":
        raise ValueError("Frozen test access must be forbidden")
    if smoke:
        return run_smoke(config)
    output = Path(config["output"]["root"])
    for child in ("logs", "checkpoints", "tables", "figures", "report", "tensorboard"):
        (output / child).mkdir(parents=True, exist_ok=True)
    writer = _writer(output / "tensorboard" / "concise", allow_missing=False)
    start = time.time()
    frozen_root = Path(config["data"]["frozen_root"])
    train_groups = _load_groups(frozen_root, "train")
    validation_groups = _load_groups(frozen_root, "validation")
    _assert_disjoint(train_groups, validation_groups)
    train_states, train_lookup = build_states(train_groups)
    validation_states, validation_lookup = build_states(validation_groups)
    cache = Path(config["semantic_encoder"]["cache_dir"])
    train_encoded, train_cache = encode_or_load(
        [state.encoder_example for state in train_states],
        split="darc_v05_train",
        config=config["semantic_encoder"],
        cache_dir=cache,
    )
    validation_encoded, validation_cache = encode_or_load(
        [state.encoder_example for state in validation_states],
        split="darc_v05_validation",
        config=config["semantic_encoder"],
        cache_dir=cache,
    )
    reference_budget = float(
        np.median(
            [
                int(group.rows["VVV"].get("teacher_total_tokens", 0) or 0)
                for group in train_groups.values()
            ]
        )
    )
    fractions = [float(value) for value in config["offline_rl"]["budget_fractions"]]
    allowances = [fraction * reference_budget for fraction in fractions]
    hard_cap = int(config["offline_rl"]["hard_call_token_cap"])
    seeds = [int(value) for value in config["reproducibility"]["seeds"]]
    folds = int(config["offline_rl"]["crossfit_folds"])
    iterations = int(config["offline_rl"]["policy_iterations"])
    device = str(config["semantic_encoder"]["device"])
    all_records: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    target_diagnostic_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    v04_root = Path(config["data"]["darc_v04_audit"])
    for variant in tqdm(config["variants"], desc="DARC v0.5 ablations", unit="variant"):
        name = str(variant["name"])
        for seed in seeds:
            folds_by_task = assign_task_folds(
                list(train_groups), folds, int(config["reproducibility"]["crossfit_seed"])
            )
            validation_fold = {key: 0 for key in validation_groups}
            initial = DARCv04Ensemble.load(
                v04_root / "models" / f"seed_{seed}" / "myopic_targets.pt"
            )
            initial_policy = _selected_v04_policy(v04_root, seed, "myopic_targets")
            initial_train_predictions = _precompute(
                initial, train_encoded, train_states, allowances, device=device
            )
            initial_validation_predictions = _precompute(
                initial, validation_encoded, validation_states, allowances, device=device
            )
            train_factory: Any = _v04_continuation_factory(
                initial_train_predictions,
                allowances,
                initial_policy,
                reference_budget=reference_budget,
                hard_cap=hard_cap,
            )
            validation_factory: Any = _v04_continuation_factory(
                initial_validation_predictions,
                allowances,
                initial_policy,
                reference_budget=reference_budget,
                hard_cap=hard_cap,
            )
            previous: DARCv05Ensemble | None = None
            for iteration in range(iterations):
                train_targets = generate_policy_targets(
                    train_states,
                    train_groups,
                    train_lookup,
                    allowances=allowances,
                    reference_budget=reference_budget,
                    hard_call_token_cap=hard_cap,
                    continuation_factory=train_factory,
                    target_mode=str(variant["target_mode"]),
                )
                validation_targets = generate_policy_targets(
                    validation_states,
                    validation_groups,
                    validation_lookup,
                    allowances=allowances,
                    reference_budget=reference_budget,
                    hard_call_token_cap=hard_cap,
                    continuation_factory=validation_factory,
                    target_mode=str(variant["target_mode"]),
                )
                train_diagnostics = validate_target_coverage(
                    train_targets,
                    name=f"train/{name}/seed_{seed}/iteration_{iteration}",
                )
                validation_diagnostics = validate_target_coverage(
                    validation_targets,
                    name=f"validation/{name}/seed_{seed}/iteration_{iteration}",
                )
                _append_target_diagnostics(
                    target_diagnostic_rows,
                    train_diagnostics,
                    split="train",
                    variant=name,
                    seed=seed,
                    iteration=iteration,
                )
                _append_target_diagnostics(
                    target_diagnostic_rows,
                    validation_diagnostics,
                    split="validation",
                    variant=name,
                    seed=seed,
                    iteration=iteration,
                )
                tqdm.write(
                    "Verify target coverage "
                    f"{name} seed={seed} iteration={iteration}: "
                    f"train={train_diagnostics['verify_supervision_rate_positive']:.1%}, "
                    "validation="
                    f"{validation_diagnostics['verify_supervision_rate_positive']:.1%}"
                )
                model = train_crossfit_iteration(
                    train_states,
                    train_encoded,
                    train_targets,
                    validation_states,
                    validation_encoded,
                    validation_targets,
                    task_folds=folds_by_task,
                    folds=folds,
                    members_per_fold=int(config["offline_rl"]["members_per_fold"]),
                    iteration=iteration,
                    params=config["model"],
                    reference_budget=reference_budget,
                    v04_initialization=initial,
                    previous=previous,
                    seed=seed,
                    deterministic=bool(config["reproducibility"]["deterministic_torch"]),
                    device_name=device,
                    writer=writer,
                    variant=name,
                    progress=config["monitoring"],
                )
                checkpoint = output / "checkpoints" / name / f"seed_{seed}" / f"iteration_{iteration}.pt"
                model.save(checkpoint)
                for member in model.members:
                    training_rows.extend(dict(row) for row in member.history)
                validation_predictions = _precompute_v05(
                    model,
                    validation_encoded,
                    validation_states,
                    allowances,
                    device=device,
                    description=f"Predict {name} s{seed} i{iteration}",
                )
                label = f"DARC v0.5 · {name} · i{iteration}"
                selected_policy, grid = _tune_v05_policy(
                    validation_groups,
                    validation_lookup,
                    validation_predictions,
                    allowances,
                    policy_grid(config["policy_grid"], variant),
                    fractions=fractions,
                    reference_budget=reference_budget,
                    hard_cap=hard_cap,
                    seed=seed,
                    label=label,
                    token_penalty=float(config["selection"]["token_utilization_penalty"]),
                    max_harmful_rate=float(config["selection"]["max_harmful_rate"]),
                )
                for row in grid:
                    row["iteration"] = iteration
                policy_rows.extend(grid)
                before = len(all_records)
                _evaluate_v05(
                    all_records,
                    all_traces,
                    groups=validation_groups,
                    lookup=validation_lookup,
                    predictions=validation_predictions,
                    allowances=allowances,
                    policy=selected_policy,
                    fractions=fractions,
                    reference_budget=reference_budget,
                    hard_cap=hard_cap,
                    seed=seed,
                    label=label,
                    iteration=iteration,
                    writer=writer,
                )
                summaries = [row["summary"] for row in all_records[before:]]
                nonzero = [row for row in summaries if float(row["budget_fraction"]) > 0]
                candidates.append(
                    {
                        "variant": f"{name}@i{iteration}",
                        "mean_accuracy": float(np.mean([row["accuracy"] for row in nonzero])),
                        "harmful_actions": sum(int(row["harmful"]) for row in nonzero),
                        "actual_tokens": sum(int(row["teacher_tokens"]) for row in nonzero),
                        "calls": sum(int(row["teacher_calls"]) for row in nonzero),
                        "token_budget": sum(int(row["token_budget"]) for row in nonzero),
                    }
                )
                # The next iteration's train continuation is out-of-fold.
                fold_predictions: dict[int, list[V05Prediction]] = {}
                for fold in range(folds):
                    fold_predictions[fold] = _precompute_v05(
                        model.members_for_fold(fold),
                        train_encoded,
                        train_states,
                        allowances,
                        device=device,
                        description=f"OOF targets {name} s{seed} f{fold}",
                    )
                train_factory = _v05_continuation_factory(
                    fold_predictions,
                    allowances,
                    folds_by_task,
                    selected_policy,
                    reference_budget=reference_budget,
                    hard_cap=hard_cap,
                )
                validation_factory = _v05_continuation_factory(
                    {0: validation_predictions},
                    allowances,
                    validation_fold,
                    selected_policy,
                    reference_budget=reference_budget,
                    hard_cap=hard_cap,
                )
                previous = model
    baseline_records, baseline_traces = _baseline_records(
        validation_groups,
        validation_states,
        validation_lookup,
        validation_encoded,
        config=config,
        reference_budget=reference_budget,
        allowances=allowances,
        seeds=seeds,
        fractions=fractions,
    )
    all_records.extend(baseline_records)
    all_traces.extend(baseline_traces)
    aggregated_candidates: list[dict[str, Any]] = []
    for candidate_name in sorted({str(row["variant"]) for row in candidates}):
        rows = [row for row in candidates if row["variant"] == candidate_name]
        aggregated_candidates.append(
            {
                "variant": candidate_name,
                "mean_accuracy": float(np.mean([row["mean_accuracy"] for row in rows])),
                "harmful_actions": sum(int(row["harmful_actions"]) for row in rows),
                "actual_tokens": sum(int(row["actual_tokens"]) for row in rows),
                "calls": sum(int(row["calls"]) for row in rows),
                "token_budget": sum(int(row["token_budget"]) for row in rows),
                "seeds": len(rows),
            }
        )
    selected, selection_rows = select_development_candidate(
        aggregated_candidates,
        reference_variant=f"{config['selection']['reference_variant']}@i{iterations - 1}",
        token_utilization_penalty=float(config["selection"]["token_utilization_penalty"]),
        max_harmful_rate=float(config["selection"]["max_harmful_rate"]),
        harmful_rate_tolerance_over_reference=float(
            config["selection"]["harmful_rate_tolerance_over_reference"]
        ),
    )
    aggregate = aggregate_records(
        all_records,
        bootstrap=int(config["reproducibility"]["bootstrap_repetitions"]),
        seed=seeds[0],
    )
    selected_label = f"DARC v0.5 · {selected.replace('@', ' · ')}"
    comparisons: list[dict[str, Any]] = []
    available = {row["policy"] for row in all_records}
    for reference in (
        "DARC v0.4 · myopic_targets",
        "DARC v0.4 · full_pretrained",
        "C3 MultiHistory Myopic",
        "C3 MultiHistory Continuation",
    ):
        if selected_label in available and reference in available:
            comparisons.append(
                paired_comparison(
                    all_records,
                    candidate=selected_label,
                    reference=reference,
                    fraction=1.0,
                    bootstrap=int(config["reproducibility"]["bootstrap_repetitions"]),
                    permutations=int(config["reproducibility"]["permutation_repetitions"]),
                    seed=seeds[0],
                )
            )
    seed_rows = [row["summary"] for row in all_records]
    _write_csv(output / "tables" / "training_history.csv", training_rows)
    _write_csv(output / "tables" / "target_coverage.csv", target_diagnostic_rows)
    _write_csv(output / "tables" / "policy_grid.csv", policy_rows)
    _write_csv(output / "tables" / "seed_results.csv", seed_rows)
    _write_csv(output / "tables" / "aggregate_results.csv", aggregate)
    _write_csv(output / "tables" / "candidate_selection.csv", selection_rows)
    _write_csv(output / "tables" / "decision_trace.csv", all_traces)
    if comparisons:
        _write_csv(output / "tables" / "paired_comparisons.csv", comparisons)
    audit = audit_level2({"train": train_groups, "validation": validation_groups})
    _write_csv(output / "tables" / "data_audit.csv", audit)
    inherited_leakage_path = v04_root / "leakage_audit.json"
    inherited_leakage = json.loads(inherited_leakage_path.read_text(encoding="utf-8"))
    leakage_audit = {
        "schema_version": "darc-v0.5-leakage-audit-v1",
        "all_required_checks_pass": bool(
            inherited_leakage.get("all_required_checks_pass", False)
        ),
        "inherited_level1_level2_checks": inherited_leakage,
        "new_checks": [
            {
                "check": "train_validation_overlap",
                "value": len(set(train_groups) & set(validation_groups)),
                "passed": not bool(set(train_groups) & set(validation_groups)),
            },
            {
                "check": "frozen_test_loaded",
                "value": False,
                "passed": True,
            },
            {
                "check": "policy_targets_use_descendant_max",
                "value": False,
                "passed": True,
                "detail": "Forced action followed by one fixed observable continuation policy.",
            },
            {
                "check": "online_actual_future_cost_feature",
                "value": False,
                "passed": True,
                "detail": "Predicted call cost admits action; realized cost is charged afterward.",
            },
        ],
    }
    (output / "leakage_audit.json").write_text(
        json.dumps(leakage_audit, indent=2) + "\n", encoding="utf-8"
    )
    _figures(output, training_rows, aggregate, all_traces)
    manifest = {
        "schema_version": "darc-v0.5-selection-manifest-v1",
        "partition_loaded": "validation",
        "frozen_test_loaded": False,
        "selected_candidate": selected,
        "seeds": seeds,
        "crossfit_folds": folds,
        "policy_iterations": iterations,
        "reference_budget_per_task": reference_budget,
        "hard_call_token_cap": hard_cap,
        "budget_semantics": {
            "global_remaining_budget": "strict actual-token ledger used for action admission",
            "per_task_allowance": "observable allocation/shadow-price feature; not a hard feasibility gate",
            "hard_call_reserve": "conservative reserve checked against the global ledger only",
        },
        "verify_target_coverage_check": "passed",
        "selection_criteria": config["selection"],
        "embedding_cache": {"train": train_cache, "validation": validation_cache},
        "elapsed_seconds": time.time() - start,
    }
    (output / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    report = _report(output, config, aggregate, selected, audit, comparisons)
    writer.close()
    return report


def run_smoke(config: dict[str, Any]) -> Path:
    """Run a tiny synthetic environment/model contract check, never real data."""

    from agent_verify.analysis.darc_v05_smoke import run_synthetic_smoke

    output = Path(config["output"]["root"]) / "smoke"
    return run_synthetic_smoke(config, output=output)

