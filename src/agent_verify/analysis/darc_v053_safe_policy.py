"""DARC v0.5.3 baseline-anchored safe budgeted policy improvement.

This module never trains or mutates DARC neural weights.  It loads the original
v0.5 Myopic-i0 ensemble, obtains out-of-fold predictions on training tasks,
fits checkpoint-local probability calibrators, and selects conservative action
switching rules exclusively from cross-fitted training replay.  Validation is
used once for reporting after every policy parameter has been fixed.

The implementation is SPIBB-inspired: the original v0.5 action is the default
and a switch is admitted only when a calibrated confidence interval lies on
the other side of the checkpoint-specific decision margin.  It does not claim
a formal high-probability safety guarantee.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from agent_verify.analysis.cbv_v01_experiment import ReplayResult, _passed, _stable_order
from agent_verify.analysis.darc_data import EncodedDARC, encode_or_load
from agent_verify.analysis.darc_v04_audit import (
    _seed_summary,
    aggregate_records,
    paired_comparison,
)
from agent_verify.analysis.darc_v04_data import MultiHistoryState, build_states
from agent_verify.analysis.darc_v04_experiment import (
    _assert_disjoint,
    _load_groups,
    _write_csv,
)
from agent_verify.analysis.darc_v05_env import CBVLevel2OfflineEnv, OfflineObservation
from agent_verify.analysis.darc_v05_experiment import _precompute_v05, replay_v05
from agent_verify.analysis.darc_v05_model import DARCv05Ensemble, V05Prediction
from agent_verify.analysis.darc_v05_training import (
    ConservativePolicy,
    assign_task_folds,
    nearest_prediction,
    policy_action,
)

ABLATION_A = "A · Original v0.5 Myopic i0"
ABLATION_B = "B · Checkpoint calibration"
ABLATION_C = "C · Baseline-anchored safe improvement"
ABLATION_D = "D · Safe improvement + dynamic shadow price"
ABLATIONS = (ABLATION_A, ABLATION_B, ABLATION_C, ABLATION_D)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_original_policy(config: dict[str, Any], seed: int) -> ConservativePolicy:
    root = Path(config["baseline"]["results_root"])
    rows = [
        row
        for row in _read_csv(root / config["baseline"]["policy_table"])
        if int(row["seed"]) == seed
        and row["variant"] == config["baseline"]["policy_label"]
        and str(row["selected"]).lower() in {"1", "true"}
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one selected original policy for seed {seed}, got {len(rows)}")
    row = rows[0]
    fields = ConservativePolicy.__dataclass_fields__
    values: dict[str, Any] = {}
    for name, field in fields.items():
        value = row[name]
        if field.type is bool or name in {"conservative", "dynamic_token_price", "uncertainty_penalty"}:
            values[name] = str(value).lower() in {"1", "true"}
        else:
            values[name] = float(value)
    return ConservativePolicy(**values)


@dataclass(frozen=True)
class ConstantClassifier:
    probabilities: tuple[float, float, float]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray(self.probabilities, dtype=float), (len(features), 1))


@dataclass
class CheckpointPlattCalibrator:
    """Natural-frequency multinomial Platt calibration for one checkpoint."""

    checkpoint: int
    scaler: StandardScaler | None
    classifier: LogisticRegression | ConstantClassifier
    observed_classes: tuple[int, ...]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features) if self.scaler is not None else features
        partial = self.classifier.predict_proba(scaled)
        if isinstance(self.classifier, ConstantClassifier):
            return partial
        output = np.zeros((len(features), 3), dtype=float)
        for column, label in enumerate(self.classifier.classes_):
            output[:, int(label)] = partial[:, column]
        output /= np.clip(output.sum(axis=1, keepdims=True), 1e-12, None)
        return output


def calibration_features(
    prediction: V05Prediction,
    states: list[MultiHistoryState],
    allowance: float,
    reference_budget: float,
) -> np.ndarray:
    """Observable prediction features; contains no outcomes or future state."""

    return np.column_stack(
        (
            prediction.q_success[:, 0],
            prediction.q_success[:, 1],
            prediction.advantage,
            prediction.advantage_uncertainty,
            prediction.effect_probabilities,
            prediction.additional_cost / max(reference_budget, 1.0),
            prediction.additional_cost_uncertainty / max(reference_budget, 1.0),
            prediction.immediate_cost / max(reference_budget, 1.0),
            np.full(len(states), allowance / max(reference_budget, 1.0)),
        )
    ).astype(np.float64)


def _effect_labels(states: list[MultiHistoryState]) -> np.ndarray:
    return np.asarray([row.encoder_example.value_label for row in states], dtype=np.int64)


def fit_checkpoint_calibrators(
    predictions: list[V05Prediction],
    states: list[MultiHistoryState],
    allowances: list[float],
    *,
    reference_budget: float,
    regularization_c: float,
    max_iter: int,
    task_filter: set[str] | None = None,
) -> dict[int, CheckpointPlattCalibrator]:
    labels = _effect_labels(states)
    calibrators: dict[int, CheckpointPlattCalibrator] = {}
    positive_allowances = [i for i, value in enumerate(allowances) if value > 0]
    if not positive_allowances:
        raise ValueError("Calibration requires at least one positive allowance")
    for checkpoint in (1, 2, 3):
        state_mask = np.asarray(
            [
                row.checkpoint == checkpoint and (task_filter is None or row.task_key in task_filter)
                for row in states
            ],
            dtype=bool,
        )
        feature_rows: list[np.ndarray] = []
        label_rows: list[np.ndarray] = []
        for allowance_index in positive_allowances:
            feature_rows.append(
                calibration_features(
                    predictions[allowance_index], states, allowances[allowance_index], reference_budget
                )[state_mask]
            )
            label_rows.append(labels[state_mask])
        x = np.concatenate(feature_rows)
        y = np.concatenate(label_rows)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            probabilities = np.zeros(3, dtype=float)
            probabilities[int(classes[0])] = 1.0
            calibrators[checkpoint] = CheckpointPlattCalibrator(
                checkpoint, None, ConstantClassifier(tuple(probabilities)), tuple(map(int, classes))
            )
            continue
        scaler = StandardScaler().fit(x)
        classifier = LogisticRegression(
            C=regularization_c,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=0,
        ).fit(scaler.transform(x), y)
        calibrators[checkpoint] = CheckpointPlattCalibrator(
            checkpoint, scaler, classifier, tuple(map(int, classes))
        )
    return calibrators


def apply_calibrators(
    predictions: list[V05Prediction],
    states: list[MultiHistoryState],
    allowances: list[float],
    calibrators: dict[int, CheckpointPlattCalibrator],
    *,
    reference_budget: float,
) -> list[V05Prediction]:
    output: list[V05Prediction] = []
    checkpoints = np.asarray([row.checkpoint for row in states])
    for prediction, allowance in zip(predictions, allowances, strict=True):
        features = calibration_features(prediction, states, allowance, reference_budget)
        probabilities = np.zeros_like(prediction.effect_probabilities, dtype=float)
        for checkpoint, calibrator in calibrators.items():
            mask = checkpoints == checkpoint
            probabilities[mask] = calibrator.predict_proba(features[mask])
        calibrated_advantage = probabilities[:, 2] - probabilities[:, 0]
        output.append(replace(prediction, effect_probabilities=probabilities, advantage=calibrated_advantage))
    return output


def cross_calibrated_oof_predictions(
    predictions: list[V05Prediction],
    states: list[MultiHistoryState],
    allowances: list[float],
    task_folds: dict[str, int],
    *,
    folds: int,
    reference_budget: float,
    regularization_c: float,
    max_iter: int,
) -> list[V05Prediction]:
    """Fit calibration without letting a task calibrate its own OOF prediction."""

    rows = [
        replace(
            prediction,
            effect_probabilities=np.zeros_like(prediction.effect_probabilities),
            advantage=np.zeros_like(prediction.advantage),
        )
        for prediction in predictions
    ]
    for fold in range(folds):
        fit_tasks = {key for key, value in task_folds.items() if value != fold}
        calibrators = fit_checkpoint_calibrators(
            predictions,
            states,
            allowances,
            reference_budget=reference_budget,
            regularization_c=regularization_c,
            max_iter=max_iter,
            task_filter=fit_tasks,
        )
        calibrated = apply_calibrators(
            predictions, states, allowances, calibrators, reference_budget=reference_budget
        )
        mask = np.asarray([task_folds[row.task_key] == fold for row in states])
        for allowance_index in range(len(allowances)):
            rows[allowance_index].effect_probabilities[mask] = calibrated[
                allowance_index
            ].effect_probabilities[mask]
            rows[allowance_index].advantage[mask] = calibrated[allowance_index].advantage[mask]
    return rows


def oof_predictions(
    model: DARCv05Ensemble,
    encoded: EncodedDARC,
    states: list[MultiHistoryState],
    allowances: list[float],
    task_folds: dict[str, int],
    *,
    folds: int,
    device: str,
    description: str,
) -> list[V05Prediction]:
    """Assemble one held-out-member neural prediction for every train task."""

    combined: list[V05Prediction] | None = None
    for fold in tqdm(range(folds), desc=description, unit="fold", dynamic_ncols=True, leave=False):
        fold_predictions = _precompute_v05(
            model.members_for_fold(fold),
            encoded,
            states,
            allowances,
            device=device,
            description=f"OOF fold {fold}",
        )
        mask = np.asarray([task_folds[row.task_key] == fold for row in states])
        if combined is None:
            combined = [
                V05Prediction(
                    **{name: np.zeros_like(getattr(row, name)) for name in V05Prediction.__dataclass_fields__}
                )
                for row in fold_predictions
            ]
        for allowance_index, prediction in enumerate(fold_predictions):
            for name in V05Prediction.__dataclass_fields__:
                getattr(combined[allowance_index], name)[mask] = getattr(prediction, name)[mask]
    assert combined is not None
    return combined


@dataclass(frozen=True)
class SafeImprovementPolicy:
    checkpoint_margins: tuple[float, float, float]
    uncertainty_beta: float
    switch_margin: float
    harmful_weight: float
    shadow_price: float
    dynamic_shadow_price: bool
    price_elasticity: float = 1.0

    def interval(
        self,
        prediction: V05Prediction,
        index: int,
        *,
        checkpoint: int,
        allowance: float,
        reference_budget: float,
    ) -> tuple[float, float, float, float]:
        cost_upper = float(prediction.additional_cost[index]) + float(
            prediction.additional_cost_uncertainty[index]
        )
        pressure = 1.0
        if self.dynamic_shadow_price:
            pressure = (reference_budget / max(allowance, 1.0)) ** self.price_elasticity
        net = (
            float(prediction.advantage[index])
            - self.harmful_weight * float(prediction.effect_probabilities[index, 0])
            - self.shadow_price * pressure * cost_upper / max(reference_budget, 1.0)
        )
        radius = self.uncertainty_beta * float(prediction.advantage_uncertainty[index])
        margin = self.checkpoint_margins[checkpoint - 1]
        return net - radius, net + radius, net, margin


def safe_improved_action(
    observation: OfflineObservation,
    *,
    baseline_action: str,
    prediction: V05Prediction,
    policy: SafeImprovementPolicy,
    reference_budget: float,
    hard_call_token_cap: int,
) -> tuple[str, dict[str, float | int | str]]:
    index = observation.state_index
    lower, upper, net, margin = policy.interval(
        prediction,
        index,
        checkpoint=observation.checkpoint,
        allowance=observation.per_task_allowance,
        reference_budget=reference_budget,
    )
    predicted_call_upper = float(prediction.immediate_cost[index]) + float(
        prediction.immediate_cost_uncertainty[index]
    )
    feasible = (
        observation.global_remaining_budget >= hard_call_token_cap
        and predicted_call_upper <= observation.global_remaining_budget
    )
    action = baseline_action
    reason = "retain_baseline_uncertain"
    if baseline_action == "S" and feasible and lower > margin + policy.switch_margin:
        action, reason = "V", "confident_skip_to_verify"
    elif baseline_action == "V" and upper < margin - policy.switch_margin:
        action, reason = "S", "confident_verify_to_skip"
    elif baseline_action == "V" and not feasible:
        action, reason = "S", "infeasible_verify"
    return action, {
        "safe_net_advantage": net,
        "safe_lower_bound": lower,
        "safe_upper_bound": upper,
        "checkpoint_margin": margin,
        "predicted_call_upper": predicted_call_upper,
        "switch_reason": reason,
        "switched": int(action != baseline_action),
    }


def replay_v053(
    groups: dict[str, Any],
    lookup: dict[tuple[str, str], int],
    raw_predictions: list[V05Prediction],
    calibrated_predictions: list[V05Prediction],
    allowances: list[float],
    baseline_policy: ConservativePolicy,
    *,
    variant: str,
    safe_policy: SafeImprovementPolicy | None,
    reference_budget: float,
    token_budget: int,
    budget_fraction: float,
    hard_call_token_cap: int,
    seed: int,
) -> tuple[ReplayResult, list[dict[str, Any]]]:
    keys = _stable_order(groups, seed)
    env = CBVLevel2OfflineEnv(groups, lookup, hard_call_token_cap=hard_call_token_cap)
    actual_spent = calls = 0
    predicted_spent = 0.0
    outcomes: list[float] = []
    baselines: list[float] = []
    masks: list[int] = []
    traces: list[dict[str, Any]] = []
    for order, key in enumerate(keys):
        remaining = max(token_budget - actual_spent, 0)
        allowance = remaining / max(len(keys) - order, 1)
        env.reset(key, remaining, per_task_allowance=allowance)
        mask = 0
        while True:
            allowance = remaining / max(len(keys) - order, 1)
            env.set_per_task_allowance(allowance)
            observation = env.observation()
            raw = nearest_prediction(raw_predictions, allowances, allowance)
            calibrated = nearest_prediction(calibrated_predictions, allowances, allowance)
            baseline_action = policy_action(
                observation,
                predictions=raw_predictions,
                allowances=allowances,
                policy=baseline_policy,
                reference_budget=reference_budget,
                hard_call_token_cap=hard_call_token_cap,
            )
            diagnostics: dict[str, Any] = {
                "switch_reason": "not_applicable",
                "switched": 0,
                "safe_net_advantage": math.nan,
                "safe_lower_bound": math.nan,
                "safe_upper_bound": math.nan,
                "checkpoint_margin": math.nan,
            }
            if variant == ABLATION_B:
                action = policy_action(
                    observation,
                    predictions=calibrated_predictions,
                    allowances=allowances,
                    policy=baseline_policy,
                    reference_budget=reference_budget,
                    hard_call_token_cap=hard_call_token_cap,
                )
                diagnostics.update(
                    switch_reason="calibrated_baseline_rule",
                    switched=int(action != baseline_action),
                )
            elif variant in {ABLATION_C, ABLATION_D}:
                if safe_policy is None:
                    raise ValueError(f"{variant} requires a safe policy")
                action, diagnostics = safe_improved_action(
                    observation,
                    baseline_action=baseline_action,
                    prediction=calibrated,
                    policy=safe_policy,
                    reference_budget=reference_budget,
                    hard_call_token_cap=hard_call_token_cap,
                )
            else:
                action = baseline_action
            index = observation.state_index
            predicted_cost = max(float(calibrated.immediate_cost[index]), 1.0)
            direct_skip = observation.prefix + "S" + "S" * (2 - len(observation.prefix))
            direct_verify = observation.prefix + "V" + "S" * (2 - len(observation.prefix))
            delta = int(_passed(groups[key].rows[direct_verify]) - _passed(groups[key].rows[direct_skip]))
            before = actual_spent
            transition = env.step(action)
            actual_spent += transition.incremental_teacher_tokens
            remaining = max(token_budget - actual_spent, 0)
            if actual_spent > token_budget:
                raise AssertionError("DARC v0.5.3 exceeded strict actual-token budget")
            if action == "V":
                calls += 1
                predicted_spent += predicted_cost
                mask |= 1 << (observation.checkpoint - 1)
            traces.append(
                {
                    "policy": variant,
                    "seed": seed,
                    "budget_fraction": budget_fraction,
                    "task_key": key,
                    "dataset": groups[key].task.dataset,
                    "checkpoint": observation.checkpoint,
                    "history": observation.prefix or "START",
                    "baseline_action": baseline_action,
                    "action": action,
                    "remaining_global_budget_before_action": token_budget - before,
                    "per_task_allowance": allowance,
                    "raw_advantage": float(raw.advantage[index]),
                    "calibrated_advantage": float(calibrated.advantage[index]),
                    "advantage_uncertainty": float(raw.advantage_uncertainty[index]),
                    "predicted_harmful": float(calibrated.effect_probabilities[index, 0]),
                    "predicted_no_change": float(calibrated.effect_probabilities[index, 1]),
                    "predicted_helpful": float(calibrated.effect_probabilities[index, 2]),
                    "predicted_immediate_cost": predicted_cost,
                    "actual_cost_after_action": transition.incremental_teacher_tokens,
                    "direct_actual_delta_audit_only": delta,
                    "helpful_selected": int(action == "V" and delta > 0),
                    "harmful_selected": int(action == "V" and delta < 0),
                    "no_change_selected": int(action == "V" and delta == 0),
                    **diagnostics,
                }
            )
            if transition.done:
                outcomes.append(float(transition.terminal_correctness or 0))
                break
        baselines.append(float(_passed(groups[key].rows["SSS"])))
        masks.append(mask)
    return ReplayResult(
        variant,
        "C-multi-strict-v053",
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
    ), traces


def safe_policy_grid(config: dict[str, Any], *, dynamic: bool) -> list[SafeImprovementPolicy]:
    grid = config["safe_policy_grid"]
    elasticities = grid["dynamic_price_elasticity"] if dynamic else [1.0]
    rows = []
    for margins, beta, switch, harmful, price, elasticity in itertools.product(
        grid["checkpoint_margin_sets"],
        grid["uncertainty_beta"],
        grid["switch_margin"],
        grid["harmful_weight"],
        grid["shadow_price"],
        elasticities,
    ):
        rows.append(
            SafeImprovementPolicy(
                tuple(map(float, margins)),
                float(beta),
                float(switch),
                float(harmful),
                float(price),
                dynamic,
                float(elasticity),
            )
        )
    return rows


def _trace_counts(traces: list[dict[str, Any]]) -> tuple[int, int, int]:
    return (
        sum(int(row["helpful_selected"]) for row in traces),
        sum(int(row["harmful_selected"]) for row in traces),
        sum(int(row["no_change_selected"]) for row in traces),
    )


def tune_safe_policy(
    groups: dict[str, Any],
    lookup: dict[tuple[str, str], int],
    raw_predictions: list[V05Prediction],
    calibrated_predictions: list[V05Prediction],
    allowances: list[float],
    baseline_policy: ConservativePolicy,
    candidates: list[SafeImprovementPolicy],
    *,
    variant: str,
    fractions: list[float],
    reference_budget: float,
    hard_cap: int,
    seed: int,
    selection: dict[str, Any],
) -> tuple[SafeImprovementPolicy, list[dict[str, Any]]]:
    active_fractions = [value for value in fractions if value > 0]
    baseline_metrics = []
    for fraction in active_fractions:
        replay, traces = replay_v053(
            groups,
            lookup,
            raw_predictions,
            calibrated_predictions,
            allowances,
            baseline_policy,
            variant=ABLATION_A,
            safe_policy=None,
            reference_budget=reference_budget,
            token_budget=int(round(fraction * reference_budget * len(groups))),
            budget_fraction=fraction,
            hard_call_token_cap=hard_cap,
            seed=seed,
        )
        baseline_metrics.append((float(replay.outcomes.mean()), replay.actual_tokens, *_trace_counts(traces)))
    baseline_accuracy = float(np.mean([row[0] for row in baseline_metrics]))
    baseline_harmful_rate = sum(row[3] for row in baseline_metrics) / max(
        sum(sum(row[2:5]) for row in baseline_metrics), 1
    )
    total_budget = sum(int(round(f * reference_budget * len(groups))) for f in active_fractions)
    rows: list[dict[str, Any]] = []
    progress = tqdm(candidates, desc=f"{variant} train-OOF grid", unit="candidate", dynamic_ncols=True)
    for candidate_id, policy in enumerate(progress):
        accuracies: list[float] = []
        tokens = calls = helpful = harmful = no_change = switches = 0
        for fraction in active_fractions:
            replay, traces = replay_v053(
                groups,
                lookup,
                raw_predictions,
                calibrated_predictions,
                allowances,
                baseline_policy,
                variant=variant,
                safe_policy=policy,
                reference_budget=reference_budget,
                token_budget=int(round(fraction * reference_budget * len(groups))),
                budget_fraction=fraction,
                hard_call_token_cap=hard_cap,
                seed=seed,
            )
            accuracies.append(float(replay.outcomes.mean()))
            tokens += replay.actual_tokens
            calls += replay.calls
            h, r, n = _trace_counts(traces)
            helpful += h
            harmful += r
            no_change += n
            switches += sum(int(row["switched"]) for row in traces)
        accuracy = float(np.mean(accuracies))
        harmful_rate = harmful / max(calls, 1)
        utilization = tokens / max(total_budget, 1)
        accuracy_floor = baseline_accuracy - float(selection["maximum_accuracy_regression_pp"]) / 100.0
        risk_limit = min(
            float(selection["maximum_harmful_rate"]),
            baseline_harmful_rate + float(selection["harmful_rate_tolerance_over_baseline"]),
        )
        eligible = accuracy >= accuracy_floor - 1e-12 and harmful_rate <= risk_limit + 1e-12
        score = accuracy - float(selection["token_utilization_penalty"]) * utilization
        rows.append(
            {
                "seed": seed,
                "variant": variant,
                "candidate": candidate_id,
                **asdict(policy),
                "mean_train_oof_accuracy": accuracy,
                "baseline_train_oof_accuracy": baseline_accuracy,
                "actual_tokens": tokens,
                "token_budget": total_budget,
                "token_utilization": utilization,
                "calls": calls,
                "helpful": helpful,
                "harmful": harmful,
                "no_change": no_change,
                "harmful_rate": harmful_rate,
                "switches": switches,
                "eligible": int(eligible),
                "selection_score": score,
            }
        )
        progress.set_postfix(acc=f"{accuracy:.3f}", tok=tokens, eligible=int(eligible))
    eligible_rows = [row for row in rows if int(row["eligible"])]
    if not eligible_rows:
        # Explicitly fall back to the unchanged baseline.  A very large switch
        # margin makes both confidence tests impossible; this is safer than
        # silently selecting the least-bad policy from an ineligible grid.
        fallback = SafeImprovementPolicy(
            checkpoint_margins=(0.0, 0.0, 0.0),
            uncertainty_beta=1.0,
            switch_margin=1.0e6,
            harmful_weight=1.0,
            shadow_price=0.0,
            dynamic_shadow_price=variant == ABLATION_D,
            price_elasticity=1.0,
        )
        fallback_row = {
            "seed": seed,
            "variant": variant,
            "candidate": len(rows),
            **asdict(fallback),
            "mean_train_oof_accuracy": baseline_accuracy,
            "baseline_train_oof_accuracy": baseline_accuracy,
            "actual_tokens": sum(row[1] for row in baseline_metrics),
            "token_budget": total_budget,
            "token_utilization": sum(row[1] for row in baseline_metrics) / max(total_budget, 1),
            "calls": sum(sum(row[2:5]) for row in baseline_metrics),
            "helpful": sum(row[2] for row in baseline_metrics),
            "harmful": sum(row[3] for row in baseline_metrics),
            "no_change": sum(row[4] for row in baseline_metrics),
            "harmful_rate": baseline_harmful_rate,
            "switches": 0,
            "eligible": 1,
            "selection_score": baseline_accuracy
            - float(selection["token_utilization_penalty"])
            * sum(row[1] for row in baseline_metrics)
            / max(total_budget, 1),
            "fallback_to_baseline": 1,
        }
        rows.append(fallback_row)
        eligible_rows = [fallback_row]
    selected = max(
        eligible_rows,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["mean_train_oof_accuracy"]),
            -float(row["actual_tokens"]),
            -int(row["candidate"]),
        ),
    )
    for row in rows:
        row["selected"] = int(row is selected)
    fields = SafeImprovementPolicy.__dataclass_fields__
    values = {name: selected[name] for name in fields}
    values["checkpoint_margins"] = tuple(values["checkpoint_margins"])
    return SafeImprovementPolicy(**values), rows


def ranking_metrics(
    prediction: V05Prediction,
    states: list[MultiHistoryState],
    *,
    split: str,
    seed: int,
    recall_k: list[int],
) -> list[dict[str, Any]]:
    labels = (_effect_labels(states) == 2).astype(int)
    rows: list[dict[str, Any]] = []
    for checkpoint in (0, 1, 2, 3):
        mask = (
            np.ones(len(states), dtype=bool)
            if checkpoint == 0
            else np.asarray([row.checkpoint == checkpoint for row in states])
        )
        y = labels[mask]
        score = prediction.advantage[mask]
        positives = int(y.sum())
        auprc = float(average_precision_score(y, score)) if positives and positives < len(y) else math.nan
        base = {
            "split": split,
            "seed": seed,
            "checkpoint": "all" if checkpoint == 0 else f"CP{checkpoint}",
            "rows": int(len(y)),
            "helpful": positives,
            "helpful_auprc": auprc,
        }
        order = np.argsort(-score)
        for k in recall_k:
            selected = order[: min(k, len(order))]
            base[f"recall_at_{k}"] = float(y[selected].sum() / max(positives, 1))
        rows.append(base)
    return rows


def verify_baseline_reproduction(
    actual_rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    source = [
        row
        for row in _read_csv(Path(config["baseline"]["results_root"]) / config["baseline"]["result_table"])
        if row["policy"] == config["baseline"]["policy_label"]
    ]
    expected = {(int(row["seed"]), float(row["budget_fraction"])): row for row in source}
    output = []
    for row in actual_rows:
        key = (int(row["seed"]), float(row["budget_fraction"]))
        old = expected[key]
        match = (
            abs(float(row["accuracy"]) - float(old["accuracy"])) < 1e-12
            and int(row["teacher_tokens"]) == int(old["teacher_tokens"])
            and int(row["teacher_calls"]) == int(old["teacher_calls"])
        )
        output.append(
            {
                "seed": key[0],
                "budget_fraction": key[1],
                "expected_accuracy": float(old["accuracy"]),
                "actual_accuracy": row["accuracy"],
                "expected_tokens": int(old["teacher_tokens"]),
                "actual_tokens": row["teacher_tokens"],
                "expected_calls": int(old["teacher_calls"]),
                "actual_calls": row["teacher_calls"],
                "exact_match": int(match),
            }
        )
    if config["baseline"].get("require_exact_reproduction", True) and not all(
        row["exact_match"] for row in output
    ):
        failures = [row for row in output if not row["exact_match"]]
        raise RuntimeError(f"Baseline reproduction gate failed: {failures[:3]}")
    return output


def _record(
    records: list[dict[str, Any]], replay: ReplayResult, traces: list[dict[str, Any]], seed: int
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


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row["summary"] for row in records]


def _action_distribution(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["policy"], row["seed"], row["budget_fraction"], row["checkpoint"]) for row in traces})
    output = []
    for policy, seed, fraction, checkpoint in keys:
        selected = [
            row
            for row in traces
            if (row["policy"], row["seed"], row["budget_fraction"], row["checkpoint"])
            == (policy, seed, fraction, checkpoint)
        ]
        verifies = sum(row["action"] == "V" for row in selected)
        switches = sum(int(row.get("switched", 0)) for row in selected)
        output.append(
            {
                "policy": policy,
                "seed": seed,
                "budget_fraction": fraction,
                "checkpoint": checkpoint,
                "decisions": len(selected),
                "verify": verifies,
                "skip": len(selected) - verifies,
                "verify_rate": verifies / max(len(selected), 1),
                "switches_from_baseline": switches,
            }
        )
    return output


def _task_changes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in records:
        if row["policy"] == ABLATION_A:
            continue
        references = [
            r
            for r in records
            if r["policy"] == ABLATION_A and r["seed"] == row["seed"] and r["fraction"] == row["fraction"]
        ]
        if len(references) != 1:
            continue
        reference = references[0]["replay"]
        candidate = row["replay"]
        base = dict(zip(reference.task_keys, reference.outcomes, strict=True))
        for key, value in zip(candidate.task_keys, candidate.outcomes, strict=True):
            delta = int(value - base[key])
            if delta:
                output.append(
                    {
                        "policy": row["policy"],
                        "seed": row["seed"],
                        "budget_fraction": row["fraction"],
                        "task_key": key,
                        "baseline_correct": int(base[key]),
                        "candidate_correct": int(value),
                        "change": "win" if delta > 0 else "loss",
                    }
                )
    return output


def _plot_frontier(aggregate: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = {ABLATION_A: "#6b7280", ABLATION_B: "#60a5fa", ABLATION_C: "#10b981", ABLATION_D: "#ef4444"}
    for policy in ABLATIONS:
        rows = sorted(
            [row for row in aggregate if row["policy"] == policy], key=lambda row: row["teacher_tokens_mean"]
        )
        if not rows:
            continue
        ax.plot(
            [row["teacher_tokens_mean"] for row in rows],
            [100 * row["accuracy_mean"] for row in rows],
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=policy,
            color=colors[policy],
        )
    ax.set_xlabel("Actual Teacher tokens (mean across seeds)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("DARC v0.5.3: strict-budget accuracy–cost frontier")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_report(
    path: Path,
    aggregate: list[dict[str, Any]],
    reproduction: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> None:
    full = sorted(
        [row for row in aggregate if float(row["budget_fraction"]) == 1.0],
        key=lambda row: (-float(row["accuracy_mean"]), float(row["teacher_tokens_mean"])),
    )
    lines = [
        "# DARC v0.5.3 — Baseline-Anchored Safe Budgeted Policy Improvement",
        "",
        "## Protocol",
        "",
        "The original DARC v0.5 Myopic iteration-0 neural checkpoints are loaded read-only. "
        "No UniXcoder or DARC parameter is retrained. Checkpoint-local Platt calibration and "
        "safe-switch parameters are fitted/selected only from training out-of-fold predictions. "
        "The unchanged 98-task validation split is used only for final reporting; "
        "frozen test is never loaded.",
        "",
        "Baseline reproduction gate: "
        f"**{'PASS' if all(row['exact_match'] for row in reproduction) else 'FAIL'}** "
        f"({sum(row['exact_match'] for row in reproduction)}/{len(reproduction)} seed-budget rows exact).",
        "",
        "## Full-budget validation results",
        "",
        "| Variant | Accuracy | Actual tokens | Calls | Helpful | Harmful | No-change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in full:
        lines.append(
            f"| {row['policy']} | {100 * row['accuracy_mean']:.2f}% | {row['teacher_tokens_mean']:.0f} | "
            f"{row['teacher_calls_mean']:.1f} | {row['helpful_mean']:.1f} | "
            f"{row['harmful_mean']:.1f} | {row['no_change_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "![Accuracy–token frontier](../figures/accuracy_token_frontier.png)",
            "",
            "## Interpretation boundary",
            "",
            "This is conservative empirical policy improvement, inspired by SPIBB. The uncertainty "
            "gate is calibrated empirically and is **not** a formal safety bound. If B/C/D do not "
            "improve A at matched cost, the original baseline remains the recommended controller.",
            "",
            "## Selected training-only policies",
            "",
        ]
    )
    for row in selected_rows:
        lines.append(
            f"- Seed {row['seed']} / {row['variant']}: candidate {row['candidate']}, "
            f"train-OOF accuracy {100 * float(row['mean_train_oof_accuracy']):.2f}%, "
            f"tokens {int(row['actual_tokens'])}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "darc-v0.5.3-safe-policy-v1":
        raise ValueError("Unexpected DARC v0.5.3 config schema")
    if config["data"].get("frozen_test_access") != "forbidden":
        raise ValueError("Frozen-test access must remain forbidden")
    return config


def _preflight(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["baseline"]["results_root"])
    checkpoint_rows = []
    for seed in config["reproducibility"]["seeds"]:
        path = root / config["baseline"]["checkpoint_template"].format(seed=seed)
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint_rows.append(
            {"seed": seed, "path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        )
        load_original_policy(config, seed)
    train = _load_groups(Path(config["data"]["frozen_root"]), config["data"]["split_train"])
    validation = _load_groups(Path(config["data"]["frozen_root"]), config["data"]["split_validation"])
    _assert_disjoint(train, validation)
    return {
        "schema_version": "darc-v0.5.3-preflight-v1",
        "train_tasks": len(train),
        "validation_tasks": len(validation),
        "frozen_test_loaded": False,
        "checkpoints": checkpoint_rows,
        "ablations": list(ABLATIONS),
        "budget_fractions": config["evaluation"]["budget_fractions"],
    }


def run(config_path: Path, *, mode: str, resume: bool) -> Path:
    config = _load_config(config_path)
    output = Path(config["output"]["root"])
    for name in ("tables", "figures", "report", "logs", "incremental", "calibration"):
        (output / name).mkdir(parents=True, exist_ok=True)
    preflight = _preflight(config)
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if mode == "dry-run":
        print(json.dumps(preflight, indent=2))
        return output / "preflight.json"

    data_root = Path(config["data"]["frozen_root"])
    train_groups = _load_groups(data_root, config["data"]["split_train"])
    validation_groups = _load_groups(data_root, config["data"]["split_validation"])
    _assert_disjoint(train_groups, validation_groups)
    if mode == "smoke":
        count = int(config["smoke"]["tasks"])
        train_groups = dict(list(sorted(train_groups.items()))[:count])
        validation_groups = dict(list(sorted(validation_groups.items()))[:count])

    train_states, train_lookup = build_states(train_groups)
    validation_states, validation_lookup = build_states(validation_groups)
    encoder_config = dict(config["semantic_encoder"])
    cache_dir = Path(encoder_config.pop("cache_dir"))
    train_encoded, train_manifest = encode_or_load(
        [row.encoder_example for row in train_states],
        # Keep the original split name so a feature-identical full run reuses
        # the immutable v0.5 embedding cache instead of encoding UniXcoder again.
        split="darc_v05_train",
        config=encoder_config,
        cache_dir=cache_dir,
    )
    validation_encoded, validation_manifest = encode_or_load(
        [row.encoder_example for row in validation_states],
        split="darc_v05_validation",
        config=encoder_config,
        cache_dir=cache_dir,
    )
    (output / "embedding_manifests.json").write_text(
        json.dumps({"train": train_manifest, "validation": validation_manifest}, indent=2), encoding="utf-8"
    )

    evaluation = config["evaluation"]
    fractions = list(
        map(float, evaluation["budget_fractions"] if mode == "full" else config["smoke"]["budget_fractions"])
    )
    reference_budget = float(evaluation["reference_budget_per_task"])
    allowances = [fraction * reference_budget for fraction in fractions]
    hard_cap = int(evaluation["hard_call_token_cap"])
    folds = int(config["reproducibility"]["crossfit_folds"])
    if mode == "smoke":
        folds = min(2, len(train_groups))
    device = str(evaluation.get("device", "auto"))

    all_records: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    all_grids: list[dict[str, Any]] = []
    all_ranking: list[dict[str, Any]] = []
    checkpoint_hashes_before = {row["seed"]: row["sha256"] for row in preflight["checkpoints"]}

    seed_bar = tqdm(
        config["reproducibility"]["seeds"], desc="DARC v0.5.3 seeds", unit="seed", dynamic_ncols=True
    )
    for seed in seed_bar:
        # Smoke artifacts must never be mistaken for completed full-run seeds.
        incremental = output / "incremental" / f"{mode}_seed_{seed}.joblib"
        if resume and incremental.exists():
            payload = joblib.load(incremental)
            all_records.extend(payload["records"])
            all_traces.extend(payload["traces"])
            all_grids.extend(payload["grids"])
            all_ranking.extend(payload["ranking"])
            seed_bar.set_postfix(seed=seed, resumed=1)
            continue
        checkpoint = Path(config["baseline"]["results_root"]) / config["baseline"][
            "checkpoint_template"
        ].format(seed=seed)
        model = DARCv05Ensemble.load(checkpoint)
        baseline_policy = load_original_policy(config, seed)
        task_folds = assign_task_folds(
            list(train_groups), folds, int(config["reproducibility"]["crossfit_seed"])
        )
        train_raw = oof_predictions(
            model,
            train_encoded,
            train_states,
            allowances,
            task_folds,
            folds=folds,
            device=device,
            description=f"seed {seed} OOF neural",
        )
        validation_raw = _precompute_v05(
            model,
            validation_encoded,
            validation_states,
            allowances,
            device=device,
            description=f"seed {seed} validation neural",
        )
        calibration = config["calibration"]
        train_calibrated = cross_calibrated_oof_predictions(
            train_raw,
            train_states,
            allowances,
            task_folds,
            folds=folds,
            reference_budget=reference_budget,
            regularization_c=float(calibration["regularization_c"]),
            max_iter=int(calibration["max_iter"]),
        )
        final_calibrators = fit_checkpoint_calibrators(
            train_raw,
            train_states,
            allowances,
            reference_budget=reference_budget,
            regularization_c=float(calibration["regularization_c"]),
            max_iter=int(calibration["max_iter"]),
        )
        joblib.dump(final_calibrators, output / "calibration" / f"seed_{seed}.joblib")
        validation_calibrated = apply_calibrators(
            validation_raw,
            validation_states,
            allowances,
            final_calibrators,
            reference_budget=reference_budget,
        )
        all_ranking.extend(
            ranking_metrics(
                train_calibrated[-1],
                train_states,
                split="train_oof",
                seed=seed,
                recall_k=list(map(int, evaluation["recall_k"])),
            )
        )
        all_ranking.extend(
            ranking_metrics(
                validation_calibrated[-1],
                validation_states,
                split="validation_report_only",
                seed=seed,
                recall_k=list(map(int, evaluation["recall_k"])),
            )
        )

        selected_policies: dict[str, SafeImprovementPolicy] = {}
        if mode == "smoke":
            selected_policies[ABLATION_C] = safe_policy_grid(config, dynamic=False)[0]
            selected_policies[ABLATION_D] = safe_policy_grid(config, dynamic=True)[0]
        else:
            for variant, dynamic in ((ABLATION_C, False), (ABLATION_D, True)):
                selected, grid_rows = tune_safe_policy(
                    train_groups,
                    train_lookup,
                    train_raw,
                    train_calibrated,
                    allowances,
                    baseline_policy,
                    safe_policy_grid(config, dynamic=dynamic),
                    variant=variant,
                    fractions=fractions,
                    reference_budget=reference_budget,
                    hard_cap=hard_cap,
                    seed=seed,
                    selection=config["selection"],
                )
                selected_policies[variant] = selected
                all_grids.extend(grid_rows)

        seed_records: list[dict[str, Any]] = []
        seed_traces: list[dict[str, Any]] = []
        budget_bar = tqdm(
            fractions, desc=f"seed {seed} validation budgets", unit="budget", dynamic_ncols=True, leave=False
        )
        for fraction in budget_bar:
            token_budget = int(round(fraction * reference_budget * len(validation_groups)))
            # A uses the unchanged replay function to make reproduction auditable.
            replay_a, traces_a = replay_v05(
                validation_groups,
                validation_lookup,
                validation_raw,
                allowances,
                baseline_policy,
                reference_budget=reference_budget,
                token_budget=token_budget,
                budget_fraction=fraction,
                hard_call_token_cap=hard_cap,
                seed=seed,
                label=ABLATION_A,
            )
            _record(seed_records, replay_a, traces_a, seed)
            seed_traces.extend(traces_a)
            for variant in (ABLATION_B, ABLATION_C, ABLATION_D):
                replay, traces = replay_v053(
                    validation_groups,
                    validation_lookup,
                    validation_raw,
                    validation_calibrated,
                    allowances,
                    baseline_policy,
                    variant=variant,
                    safe_policy=selected_policies.get(variant),
                    reference_budget=reference_budget,
                    token_budget=token_budget,
                    budget_fraction=fraction,
                    hard_call_token_cap=hard_cap,
                    seed=seed,
                )
                _record(seed_records, replay, traces, seed)
                seed_traces.extend(traces)
            latest = [row for row in seed_records if row["fraction"] == fraction]
            budget_bar.set_postfix(
                budget=f"{fraction:.0%}",
                A=f"{latest[0]['accuracy']:.3f}",
                best=f"{max(row['accuracy'] for row in latest):.3f}",
            )
        if checkpoint_hashes_before[seed] != _sha256(checkpoint):
            raise AssertionError(f"Pretrained checkpoint was modified: {checkpoint}")
        payload = {
            "records": seed_records,
            "traces": seed_traces,
            "grids": [row for row in all_grids if int(row["seed"]) == seed],
            "ranking": [row for row in all_ranking if int(row["seed"]) == seed],
        }
        joblib.dump(payload, incremental)
        all_records.extend(seed_records)
        all_traces.extend(seed_traces)
        best_full = max(
            row["accuracy"]
            for row in seed_records
            if row["fraction"] == fractions[-1]
        )
        seed_bar.set_postfix(seed=seed, full_best=f"{best_full:.3f}")

    summary_rows = _summary_rows(all_records)
    baseline_rows = [row for row in summary_rows if row["policy"] == ABLATION_A]
    reproduction = [] if mode == "smoke" else verify_baseline_reproduction(baseline_rows, config)
    if mode == "smoke":
        reproduction = [
            {"seed": row["seed"], "budget_fraction": row["budget_fraction"], "exact_match": 1}
            for row in baseline_rows
        ]
    aggregate = aggregate_records(
        all_records,
        bootstrap=min(
            int(config["reproducibility"]["bootstrap_repetitions"]), 100 if mode == "smoke" else 10**9
        ),
        seed=int(config["reproducibility"]["crossfit_seed"]),
    )
    comparisons = []
    if mode == "full":
        for variant in (ABLATION_B, ABLATION_C, ABLATION_D):
            for fraction in fractions:
                comparisons.append(
                    paired_comparison(
                        all_records,
                        candidate=variant,
                        reference=ABLATION_A,
                        fraction=fraction,
                        bootstrap=int(config["reproducibility"]["bootstrap_repetitions"]),
                        permutations=int(config["reproducibility"]["permutation_repetitions"]),
                        seed=int(config["reproducibility"]["crossfit_seed"]),
                    )
                )
    tables = output / "tables"
    _write_csv(tables / "seed_results.csv", summary_rows)
    _write_csv(tables / "aggregate_results.csv", aggregate)
    _write_csv(tables / "decision_traces.csv", all_traces)
    _write_csv(tables / "action_distribution.csv", _action_distribution(all_traces))
    _write_csv(tables / "ranking_metrics.csv", all_ranking)
    _write_csv(tables / "baseline_reproduction.csv", reproduction)
    changes = _task_changes(all_records)
    if changes:
        _write_csv(tables / "task_wins_losses.csv", changes)
    if all_grids:
        _write_csv(tables / "training_oof_policy_grid.csv", all_grids)
    if comparisons:
        _write_csv(tables / "paired_comparisons.csv", comparisons)
    _plot_frontier(aggregate, output / "figures" / "accuracy_token_frontier.png")
    selected_rows = [row for row in all_grids if int(row.get("selected", 0))]
    report = output / "report" / "darc_v053_safe_policy_report.md"
    _write_report(report, aggregate, reproduction, selected_rows)
    manifest = {
        "schema_version": "darc-v0.5.3-result-v1",
        "mode": mode,
        "frozen_test_loaded": False,
        "train_tasks": len(train_groups),
        "validation_tasks": len(validation_groups),
        "report": str(report),
        "checkpoint_sha256_before_after_equal": True,
    }
    (output / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/darc_v053_safe_policy.yaml"))
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    mode = "dry-run" if args.dry_run else "smoke" if args.smoke else "full"
    started = time.monotonic()
    result = run(args.config, mode=mode, resume=args.resume)
    print(f"Completed {mode} in {time.monotonic() - started:.1f}s")
    print(f"Output: {result}")


if __name__ == "__main__":
    main()
