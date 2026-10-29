"""End-to-end offline DARC v0.1 training, validation selection, test replay, and reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm

from agent_verify.analysis.c3_controller import extract_observations
from agent_verify.analysis.c3_experiment import replay_controller
from agent_verify.analysis.cbv_v01_experiment import (
    DATASETS,
    ReplayResult,
    _bootstrap_summary,
    _budget_values,
    _stable_order,
    build_level1_examples,
    load_frozen_partition,
    task_trajectories,
)
from agent_verify.analysis.darc_data import (
    DARCExample,
    EncodedDARC,
    build_darc_examples,
    class_counts,
    dataset_fingerprint,
    encode_or_load,
)
from agent_verify.analysis.darc_model import (
    DARCEnsemble,
    DARCPrediction,
    calibrate_temperature,
    candidate_score,
    parameter_count,
    prediction_metrics,
    train_ensemble,
)


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _absolute(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _load_split(frozen_root: Path, split: str) -> tuple[list[DARCExample], Any]:
    tasks, rows = load_frozen_partition(frozen_root, split)
    groups = task_trajectories(tasks, rows, level=1)
    return build_darc_examples(groups), groups


def _assert_disjoint(partitions: dict[str, list[DARCExample]]) -> None:
    keys = {split: {row.group_key for row in examples} for split, examples in partitions.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = keys.get(left, set()) & keys.get(right, set())
        if overlap:
            raise AssertionError(f"DARC task leakage between {left}/{right}: {sorted(overlap)[:5]}")


def _assert_hard_call_cap(
    examples: list[DARCExample], *, split: str, hard_call_token_cap: int
) -> None:
    violations = [row for row in examples if row.teacher_total_tokens > hard_call_token_cap]
    if violations:
        sample = violations[0]
        raise ValueError(
            f"Configured hard_call_token_cap={hard_call_token_cap} is not a valid upper bound "
            f"for {split}; {sample.state_key} used {sample.teacher_total_tokens} tokens"
        )


def _split_rows(partitions: dict[str, list[DARCExample]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, examples in partitions.items():
        for dataset in (*DATASETS, "combined"):
            subset = examples if dataset == "combined" else [row for row in examples if row.dataset == dataset]
            if not subset:
                continue
            counts = class_counts(subset)
            rows.append(
                {
                    "split": split,
                    "scope": dataset,
                    "tasks": len({row.group_key for row in subset}),
                    "states": len(subset),
                    **counts,
                    "helpful_rate": counts["helpful"] / len(subset),
                    "harmful_rate": counts["harmful"] / len(subset),
                    "no_change_rate": counts["no_change"] / len(subset),
                }
            )
    return rows


def _prediction_subset(prediction: DARCPrediction, indices: np.ndarray) -> DARCPrediction:
    return DARCPrediction(
        prediction.probabilities[indices],
        prediction.advantage[indices],
        prediction.advantage_uncertainty[indices],
        prediction.expected_total_tokens[indices],
        prediction.upper_total_tokens[indices],
        prediction.latency_seconds * len(indices) / max(len(prediction.advantage), 1),
    )


def _group_indices(examples: list[DARCExample]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(examples):
        grouped.setdefault(row.group_key, []).append(index)
    for indices in grouped.values():
        indices.sort(key=lambda index: examples[index].checkpoint)
    return grouped


def replay_darc(
    examples: list[DARCExample],
    prediction: DARCPrediction,
    *,
    policy: str,
    margin: float,
    uncertainty_beta: float,
    cost_mode: str,
    score_mode: str,
    budget_accounting: str,
    hard_call_token_cap: int,
    token_budget: int,
    budget_fraction: float,
    seed: int,
    scope: str,
) -> tuple[ReplayResult, list[dict[str, Any]], dict[str, Any]]:
    if cost_mode not in {"expected", "upper"}:
        raise ValueError(f"Unknown DARC cost admission mode: {cost_mode}")
    if score_mode not in {"advantage", "advantage_per_1k"}:
        raise ValueError(f"Unknown DARC score mode: {score_mode}")
    if budget_accounting not in {"predicted_compatible", "hard_cap"}:
        raise ValueError(f"Unknown budget accounting mode: {budget_accounting}")
    if hard_call_token_cap <= 0:
        raise ValueError("hard_call_token_cap must be positive")
    costs = (
        prediction.expected_total_tokens
        if cost_mode == "expected"
        else prediction.upper_total_tokens
    )
    conservative = prediction.advantage - uncertainty_beta * prediction.advantage_uncertainty
    grouped = _group_indices(examples)
    task_keys = _stable_order(grouped, seed)
    predicted_remaining = float(token_budget)
    outcomes: list[float] = []
    baselines: list[float] = []
    checkpoints: list[int] = []
    actual_tokens = 0
    predicted_tokens = 0.0
    traces: list[dict[str, Any]] = []
    for key in task_keys:
        chosen: int | None = None
        for index in grouped[key]:
            predicted_cost = max(float(costs[index]), 1.0)
            decision_score = float(conservative[index])
            if score_mode == "advantage_per_1k":
                decision_score = 1000.0 * decision_score / predicted_cost
            if budget_accounting == "hard_cap":
                feasible = actual_tokens + hard_call_token_cap <= token_budget
                admission_reservation = float(hard_call_token_cap)
            else:
                feasible = predicted_cost <= predicted_remaining
                admission_reservation = predicted_cost
            verify = decision_score > float(margin) and feasible
            traces.append(
                {
                    "task_key": key,
                    "dataset": examples[index].dataset,
                    "checkpoint": examples[index].checkpoint,
                    "budget_fraction": budget_fraction,
                    "prob_harmful": float(prediction.probabilities[index, 0]),
                    "prob_no_change": float(prediction.probabilities[index, 1]),
                    "prob_helpful": float(prediction.probabilities[index, 2]),
                    "predicted_advantage": float(prediction.advantage[index]),
                    "uncertainty": float(prediction.advantage_uncertainty[index]),
                    "conservative_advantage": float(conservative[index]),
                    "decision_score": decision_score,
                    "score_mode": score_mode,
                    "predicted_cost": predicted_cost,
                    "admission_reservation": admission_reservation,
                    "predicted_remaining_before": predicted_remaining,
                    "actual_tokens_before": actual_tokens,
                    "budget_accounting": budget_accounting,
                    "feasible": feasible,
                    "verify": verify,
                    "actual_delta": examples[index].delta,
                }
            )
            if verify:
                chosen = index
                break
        base = float(examples[grouped[key][0]].y_skip)
        baselines.append(base)
        if chosen is None:
            outcomes.append(base)
            checkpoints.append(0)
            continue
        predicted = max(float(costs[chosen]), 0.0)
        actual = int(examples[chosen].teacher_total_tokens)
        if budget_accounting == "hard_cap" and actual > hard_call_token_cap:
            raise RuntimeError(
                f"Observed Teacher call cost {actual} exceeds configured hard cap "
                f"{hard_call_token_cap} for {examples[chosen].state_key}"
            )
        predicted_remaining -= predicted
        predicted_tokens += predicted
        actual_tokens += actual
        outcomes.append(float(examples[chosen].y_verify))
        checkpoints.append(examples[chosen].checkpoint)
    selected = np.asarray(checkpoints)
    replay = ReplayResult(
        policy=policy,
        protocol="A-single",
        train_mode="combined",
        scope=scope,
        budget_fraction=budget_fraction,
        token_budget=token_budget,
        task_keys=tuple(task_keys),
        outcomes=np.asarray(outcomes),
        baseline=np.asarray(baselines),
        checkpoints=selected,
        calls=int(np.sum(selected > 0)),
        actual_tokens=actual_tokens,
        predicted_tokens=predicted_tokens,
    )
    audit = {
        "budget_accounting": budget_accounting,
        "score_mode": score_mode,
        "cost_mode": cost_mode,
        "hard_call_token_cap": hard_call_token_cap,
        "actual_budget_overshoot_tokens": max(actual_tokens - token_budget, 0),
        "predicted_budget_overshoot_tokens": max(predicted_tokens - token_budget, 0.0),
        "reservation_tokens_sum": int(np.sum(selected > 0)) * hard_call_token_cap
        if budget_accounting == "hard_cap"
        else predicted_tokens,
    }
    if budget_accounting == "hard_cap" and actual_tokens > token_budget:
        raise AssertionError("Hard-cap DARC replay exceeded the actual Teacher-token budget")
    return replay, traces, audit


def _validation_policy_selection(
    examples: list[DARCExample],
    prediction: DARCPrediction,
    train_examples: list[DARCExample],
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fractions = [float(value) for value in config["calibration"]["budget_fractions"] if float(value) > 0]
    rows: list[dict[str, Any]] = []
    best: tuple[float, float, float, float] | None = None
    selected: dict[str, Any] | None = None
    joint_like = _as_joint_cost_examples(train_examples)
    budgets = dict(_budget_values(joint_like, len(_group_indices(examples))))
    total = (
        len(config["calibration"]["uncertainty_beta_grid"])
        * len(config["calibration"]["margin_grid"])
        * len(config["calibration"]["cost_admission_grid"])
        * len(config["calibration"]["score_mode_grid"])
    )
    budget_accounting = str(config["evaluation"]["darc_budget_accounting"])
    hard_call_token_cap = int(config["evaluation"]["hard_call_token_cap"])
    for beta in tqdm(
        config["calibration"]["uncertainty_beta_grid"],
        desc="DARC validation margins",
        unit="beta",
        dynamic_ncols=True,
    ):
        for margin in config["calibration"]["margin_grid"]:
            for cost_mode in config["calibration"]["cost_admission_grid"]:
                for score_mode in config["calibration"]["score_mode_grid"]:
                    results = [
                        replay_darc(
                            examples,
                            prediction,
                            policy="DARC validation",
                            margin=float(margin),
                            uncertainty_beta=float(beta),
                            cost_mode=str(cost_mode),
                            score_mode=str(score_mode),
                            budget_accounting=budget_accounting,
                            hard_call_token_cap=hard_call_token_cap,
                            token_budget=int(budgets[fraction]),
                            budget_fraction=fraction,
                            seed=seed,
                            scope="combined",
                        )
                        for fraction in fractions
                    ]
                    replays = [result[0] for result in results]
                    audits = [result[2] for result in results]
                    mean_accuracy = float(np.mean([row.outcomes.mean() for row in replays]))
                    mean_calls = float(np.mean([row.calls for row in replays]))
                    mean_tokens = float(np.mean([row.actual_tokens for row in replays]))
                    harmful = int(
                        sum(np.sum((row.outcomes - row.baseline) < 0) for row in replays)
                    )
                    row = {
                        "uncertainty_beta": float(beta),
                        "margin": float(margin),
                        "cost_mode": str(cost_mode),
                        "score_mode": str(score_mode),
                        "budget_accounting": budget_accounting,
                        "hard_call_token_cap": hard_call_token_cap,
                        "mean_validation_accuracy": mean_accuracy,
                        "mean_teacher_calls": mean_calls,
                        "mean_actual_tokens": mean_tokens,
                        "max_actual_overshoot": max(
                            int(audit["actual_budget_overshoot_tokens"]) for audit in audits
                        ),
                        "harmful_interventions": harmful,
                    }
                    rows.append(row)
                    # Accuracy is primary. Ties select fewer harmful outcomes, fewer tokens,
                    # and finally the larger conservative margin. All labels are validation-only.
                    key = (mean_accuracy, -harmful, -mean_tokens, float(margin))
                    if best is None or key > best:
                        best = key
                        selected = row
    if selected is None or len(rows) != total:
        raise RuntimeError("Incomplete DARC validation policy search")
    return selected, rows


def _as_joint_cost_examples(examples: list[DARCExample]) -> list[Any]:
    """Small adapter because the established budget grid only reads token labels."""

    class CostRow:
        def __init__(self, cost: int) -> None:
            self.teacher_total_tokens = cost

    return [CostRow(row.teacher_total_tokens) for row in examples]


def _select_and_train_models(
    train_examples: list[DARCExample],
    train_encoded: EncodedDARC,
    validation_examples: list[DARCExample],
    validation_encoded: EncodedDARC,
    config: dict[str, Any],
    output: Path,
    *,
    resume: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    registry: dict[str, dict[str, Any]] = {}
    selection_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    model_params = dict(config["model"])
    seed = int(config["reproducibility"]["seed"])
    device_name = str(config["semantic_encoder"].get("device", "auto"))
    deterministic = bool(config["reproducibility"].get("deterministic_torch", True))
    progress_config = dict(config.get("progress", {}))
    model_root = output / "models"
    ablation_progress = tqdm(
        enumerate(config["ablations"]),
        total=len(config["ablations"]),
        desc="DARC ablations",
        unit="ablation",
        dynamic_ncols=True,
        disable=not bool(progress_config.get("enabled", True)),
        mininterval=float(progress_config.get("mininterval_seconds", 1.0)),
    )
    for ablation_index, ablation in ablation_progress:
        name = str(ablation["name"])
        ablation_progress.set_postfix(name=name, stage="prepare")
        if "reuse_model_from" in ablation:
            source = str(ablation["reuse_model_from"])
            if source not in registry:
                raise ValueError(f"DARC ablation {name} references unavailable {source}")
            source_path = Path(registry[source]["model_path"])
            target_path = model_root / name / "selected.pt"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists() or not resume:
                shutil.copy2(source_path, target_path)
            model = DARCEnsemble.load(target_path)
            model.temperature = 1.0
            if bool(ablation.get("calibrate")):
                ablation_progress.set_postfix(name=name, stage="calibrate")
                labels = np.asarray([row.value_label for row in validation_examples], dtype=int)
                temperature, rows = calibrate_temperature(
                    model,
                    validation_encoded,
                    labels,
                    [float(value) for value in config["calibration"]["temperature_grid"]],
                    device_name=device_name,
                )
                calibration_rows.extend({"ablation": name, **row} for row in rows)
                model.save(target_path)
            else:
                temperature = 1.0
            prediction = model.predict(validation_encoded, device_name=device_name)
            metrics = prediction_metrics(
                validation_examples,
                prediction,
                bins=int(config["calibration"].get("ece_bins", 10)),
            )
            policy, policy_rows = _validation_policy_selection(
                validation_examples, prediction, train_examples, config, seed=seed
            )
            _write_csv(output / "tables" / "validation_policy_grid.csv", policy_rows)
            registry[name] = {
                "model": model,
                "model_path": str(target_path),
                "display_name": str(ablation["display_name"]),
                "temperature": temperature,
                "policy": policy,
                "validation_metrics": metrics,
            }
            ablation_progress.set_postfix(name=name, stage="complete")
            continue

        candidates: list[dict[str, Any]] = []
        for candidate_index, loss_mode in enumerate(ablation.get("loss_candidates", ["natural_ce"])):
            candidate_dir = model_root / name / f"candidate_{candidate_index}_{loss_mode}"
            model_path = candidate_dir / "model.pt"
            metrics_path = candidate_dir / "validation_metrics.json"
            if resume and model_path.exists() and metrics_path.exists():
                ablation_progress.set_postfix(name=name, stage="resume-cache")
                model = DARCEnsemble.load(model_path)
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                ablation_progress.set_postfix(name=name, stage=f"train-{loss_mode}")
                model = train_ensemble(
                    train_examples,
                    train_encoded,
                    validation_examples,
                    validation_encoded,
                    params=model_params,
                    downstream_mode=str(ablation.get("downstream_mode", "none")),
                    loss_mode=str(loss_mode),
                    paired_ranking=bool(ablation.get("paired_ranking")),
                    seed=seed + ablation_index * 10007 + candidate_index * 503,
                    device_name=device_name,
                    deterministic=deterministic,
                    description=f"{name}/{loss_mode}",
                    progress=progress_config,
                )
                prediction = model.predict(validation_encoded, device_name=device_name)
                metrics = prediction_metrics(
                    validation_examples,
                    prediction,
                    bins=int(config["calibration"].get("ece_bins", 10)),
                )
                metrics["selection_score"] = candidate_score(metrics)
                model.save(model_path)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            score = float(metrics.get("selection_score", candidate_score(metrics)))
            row = {
                "ablation": name,
                "candidate": candidate_index,
                "loss_mode": loss_mode,
                "selection_score": score,
                **metrics,
            }
            selection_rows.append(row)
            candidates.append({"model": model, "model_path": model_path, "metrics": metrics, "score": score})
        selected = max(enumerate(candidates), key=lambda item: (item[1]["score"], -item[0]))[1]
        selected_path = model_root / name / "selected.pt"
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        if not selected_path.exists() or not resume:
            shutil.copy2(selected["model_path"], selected_path)
        model = DARCEnsemble.load(selected_path)
        registry[name] = {
            "model": model,
            "model_path": str(selected_path),
            "display_name": str(ablation["display_name"]),
            "temperature": 1.0,
            "policy": {
                "margin": 0.0,
                "uncertainty_beta": 0.0,
                "cost_mode": "expected",
                "score_mode": "advantage",
                "budget_accounting": "predicted_compatible",
                "hard_call_token_cap": int(config["evaluation"]["hard_call_token_cap"]),
            },
            "validation_metrics": selected["metrics"],
        }
        ablation_progress.set_postfix(name=name, stage="complete")
    ablation_progress.close()
    return registry, selection_rows, calibration_rows


def _training_history_rows(registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ablation, entry in registry.items():
        model: DARCEnsemble = entry["model"]
        for member_index, member in enumerate(model.members, start=1):
            for epoch in member.history:
                rows.append(
                    {
                        "ablation": ablation,
                        "display_name": entry["display_name"],
                        "loss_mode": model.loss_mode,
                        "downstream_mode": model.downstream_mode,
                        "member": member_index,
                        "selected_best_epoch": member.best_epoch,
                        **epoch,
                    }
                )
    return rows


def _metric_rows(
    split: str,
    examples: list[DARCExample],
    registry: dict[str, dict[str, Any]],
    predictions: dict[str, DARCPrediction],
    *,
    bins: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, entry in registry.items():
        for dataset in (*DATASETS, "combined"):
            for checkpoint in (0, 1, 2, 3):
                indices = np.asarray(
                    [
                        index
                        for index, row in enumerate(examples)
                        if (dataset == "combined" or row.dataset == dataset)
                        and (checkpoint == 0 or row.checkpoint == checkpoint)
                    ],
                    dtype=int,
                )
                if not len(indices):
                    continue
                subset = [examples[int(index)] for index in indices]
                metrics = prediction_metrics(
                    subset, _prediction_subset(predictions[name], indices), bins=bins
                )
                rows.append(
                    {
                        "split": split,
                        "ablation": name,
                        "display_name": entry["display_name"],
                        "scope": dataset,
                        "checkpoint": "all" if checkpoint == 0 else checkpoint,
                        "states": len(indices),
                        **metrics,
                    }
                )
    return rows


def _baseline_rows(baseline_root: Path, allowed: set[str]) -> list[dict[str, Any]]:
    path = baseline_root / "tables" / "policy_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Existing CBV baseline table is required: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row.get("protocol") == "A-single"
        and row.get("train_mode") in {"combined", "none"}
        and row.get("policy") in allowed
    ]
    learned = {
        "Joint + Threshold",
        "Separate + Threshold",
        "Legacy Compact Boosting",
        "C3 Myopic",
    }
    for row in selected:
        row["budget_accounting"] = (
            "predicted_compatible" if row.get("policy") in learned else "offline_actual_cost"
        )
        row["actual_budget_overshoot_tokens"] = int(
            float(row.get("budget_overshoot_tokens", 0) or 0)
        )
        row["predicted_budget_overshoot_tokens"] = max(
            float(row.get("predicted_teacher_tokens", 0) or 0)
            - float(row.get("token_budget", 0) or 0),
            0.0,
        )
        row["hard_call_token_cap"] = ""
        row["score_mode"] = "legacy"
        row["cost_mode"] = "legacy"
    return selected


def _evaluate_policies(
    examples: list[DARCExample],
    train_examples: list[DARCExample],
    registry: dict[str, dict[str, Any]],
    predictions: dict[str, DARCPrediction],
    config: dict[str, Any],
    *,
    seed: int,
    bootstrap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    fractions = [float(value) for value in config["calibration"]["budget_fractions"]]
    train_cost_rows = _as_joint_cost_examples(train_examples)
    for scope in config["evaluation"]["scopes"]:
        subset_indices = np.asarray(
            [index for index, row in enumerate(examples) if scope == "combined" or row.dataset == scope],
            dtype=int,
        )
        if not len(subset_indices):
            continue
        subset = [examples[int(index)] for index in subset_indices]
        budgets = dict(_budget_values(train_cost_rows, len(_group_indices(subset))))
        for name, entry in registry.items():
            prediction = _prediction_subset(predictions[name], subset_indices)
            policy = entry["policy"]
            for fraction in fractions:
                replay, traces, audit = replay_darc(
                    subset,
                    prediction,
                    policy=entry["display_name"],
                    margin=float(policy["margin"]),
                    uncertainty_beta=float(policy["uncertainty_beta"]),
                    cost_mode=str(policy["cost_mode"]),
                    score_mode=str(policy["score_mode"]),
                    budget_accounting=str(policy["budget_accounting"]),
                    hard_call_token_cap=int(policy["hard_call_token_cap"]),
                    token_budget=int(budgets[fraction]),
                    budget_fraction=fraction,
                    seed=seed,
                    scope=str(scope),
                )
                summary = _bootstrap_summary(replay, seed=seed, repetitions=bootstrap)
                summary.update(audit)
                rows.append(summary)
                if scope == "combined" and fraction == 1.0:
                    trace_rows.extend({"ablation": name, **trace} for trace in traces)
    return rows, trace_rows


def _calibration_bins(
    split: str,
    examples: list[DARCExample],
    prediction: DARCPrediction,
    bins: int,
) -> list[dict[str, Any]]:
    target = np.asarray([row.value_label == 2 for row in examples], dtype=float)
    probability = prediction.probabilities[:, 2]
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if not np.any(selected):
            continue
        rows.append(
            {
                "split": split,
                "bin": index,
                "lower": edges[index],
                "upper": edges[index + 1],
                "states": int(selected.sum()),
                "mean_predicted_helpful": float(probability[selected].mean()),
                "observed_helpful_rate": float(target[selected].mean()),
            }
        )
    return rows


def _failure_rows(
    examples: list[DARCExample], full_budget_trace: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Summarize selected/missed counterfactuals without changing the policy."""

    grouped: dict[str, list[DARCExample]] = {}
    for row in examples:
        grouped.setdefault(row.group_key, []).append(row)
    selected: dict[str, int] = {
        str(row["task_key"]): int(row["checkpoint"])
        for row in full_budget_trace
        if bool(row["verify"])
    }
    task_rows: list[dict[str, Any]] = []
    for key, states in grouped.items():
        states.sort(key=lambda row: row.checkpoint)
        chosen_cp = selected.get(key, 0)
        chosen = next((row for row in states if row.checkpoint == chosen_cp), None)
        chosen_delta = 0 if chosen is None else chosen.delta
        best_delta = max(0, *(row.delta for row in states))
        task_rows.append(
            {
                "task_key": key,
                "dataset": states[0].dataset,
                "selected_checkpoint": chosen_cp,
                "selected_delta": chosen_delta,
                "best_available_delta": best_delta,
                "helpful_selected": int(chosen_delta > 0),
                "harmful_selected": int(chosen_delta < 0),
                "wasted_no_change": int(chosen is not None and chosen_delta == 0),
                "missed_helpful": int(best_delta > 0 and chosen_delta <= 0),
                "incorrect_checkpoint": int(chosen is not None and chosen_delta < best_delta),
            }
        )
    summaries: list[dict[str, Any]] = []
    for dataset in (*DATASETS, "combined"):
        subset = task_rows if dataset == "combined" else [row for row in task_rows if row["dataset"] == dataset]
        if not subset:
            continue
        summaries.append(
            {
                "scope": dataset,
                "tasks": len(subset),
                "teacher_calls": sum(int(row["selected_checkpoint"] > 0) for row in subset),
                "helpful_selected": sum(row["helpful_selected"] for row in subset),
                "harmful_selected": sum(row["harmful_selected"] for row in subset),
                "wasted_no_change": sum(row["wasted_no_change"] for row in subset),
                "missed_helpful": sum(row["missed_helpful"] for row in subset),
                "incorrect_checkpoint": sum(row["incorrect_checkpoint"] for row in subset),
                "cp1_calls": sum(row["selected_checkpoint"] == 1 for row in subset),
                "cp2_calls": sum(row["selected_checkpoint"] == 2 for row in subset),
                "cp3_calls": sum(row["selected_checkpoint"] == 3 for row in subset),
            }
        )
    return summaries


def _coverage_rows(caches: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, manifest in caches.items():
        for field, metrics in manifest.get("coverage_by_field", {}).items():
            rows.append({"split": split, "field": field, **metrics})
    return rows


def _make_figures(
    output: Path,
    metric_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    test = [
        row
        for row in metric_rows
        if row["split"] == "test" and row["scope"] == "combined" and row["checkpoint"] == "all"
    ]
    labels = [row["display_name"].replace(" + ", "\n+") for row in test]
    x = np.arange(len(test))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    axes[0].bar(x, [row["helpful_auprc"] for row in test], color="#2A9D8F")
    axes[0].set_title("Helpful AUPRC")
    axes[1].bar(x, [row["gain_pearson"] for row in test], color="#457B9D")
    axes[1].set_title("Predicted vs actual gain")
    axes[2].bar(x, [row["helpful_ece"] for row in test], color="#E76F51")
    axes[2].set_title("Helpful ECE (lower is better)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("DARC cumulative ablations — frozen test (descriptive, not tuned)")
    fig.tight_layout()
    fig.savefig(figure_dir / "darc_ablation_metrics.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "darc_ablation_metrics.pdf", bbox_inches="tight")
    plt.close(fig)
    combined = [row for row in policy_rows if row.get("scope") == "combined"]
    selected_names = {
        "DARC Full",
        "Joint + Threshold",
        "C3 Myopic",
        "Always CP2",
        "Never Verify",
        "Offline Oracle",
    }
    fig, axis = plt.subplots(figsize=(9.2, 6.0))
    for policy in sorted({row["policy"] for row in combined if row["policy"] in selected_names}):
        points = sorted(
            [row for row in combined if row["policy"] == policy],
            key=lambda row: float(row["actual_teacher_tokens"]),
        )
        if not points:
            continue
        axis.plot(
            [float(row["actual_teacher_tokens"]) for row in points],
            [100.0 * float(row["accuracy"]) for row in points],
            marker="o",
            label=policy,
        )
    axis.set_xlabel("Actual Teacher tokens")
    axis.set_ylabel("Final task accuracy (%)")
    axis.set_title("Matched-budget accuracy–cost frontier (Protocol A)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "darc_accuracy_cost.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "darc_accuracy_cost.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    for split, marker in (("validation", "o"), ("test", "s")):
        points = [row for row in calibration_rows if row["split"] == split]
        if points:
            axis.plot(
                [row["mean_predicted_helpful"] for row in points],
                [row["observed_helpful_rate"] for row in points],
                marker=marker,
                label=split,
            )
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.6)
    axis.set_xlabel("Predicted P(Helpful)")
    axis.set_ylabel("Observed Helpful rate")
    axis.set_title("DARC full reliability")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "darc_helpful_calibration.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "darc_helpful_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def _measure_baseline_overhead(
    baseline_root: Path,
    train_groups: Any,
    test_groups: Any,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Time saved offline baseline controllers; never invokes Student or Teacher."""

    rows: list[dict[str, Any]] = []
    joint_path = baseline_root / "models" / "combined" / "joint.joblib"
    c3_path = baseline_root / "models" / "combined" / "c3_myopic.joblib"
    if not joint_path.exists() or not c3_path.exists():
        return rows
    train = build_level1_examples(train_groups)
    test = build_level1_examples(test_groups)
    joint = joblib.load(joint_path)
    started = time.perf_counter()
    joint.predict(test)
    elapsed = time.perf_counter() - started
    rows.append(
        {
            "component": "Joint MLP + feature encoder",
            "split": "test",
            "states": len(test),
            "seconds": elapsed,
            "milliseconds_per_state": 1000.0 * elapsed / max(len(test), 1),
            "notes": "Saved sklearn/NumPy baseline; CPU wall time; excludes Teacher",
        }
    )
    controller = joblib.load(c3_path)
    budget = _budget_values(train, len({row.group_key for row in test}))[-1][1]
    started = time.perf_counter()
    observations = extract_observations(test, joint)
    replay_controller(
        observations,
        controller,
        policy_name="C3 Myopic overhead",
        token_budget=budget,
        seed=seed,
        admission_mode="expected_cost",
    )
    elapsed = time.perf_counter() - started
    rows.append(
        {
            "component": "C3 Myopic end-to-end controller",
            "split": "test",
            "states": len(test),
            "seconds": elapsed,
            "milliseconds_per_state": 1000.0 * elapsed / max(len(test), 1),
            "notes": "Includes Joint feature prediction + C3 replay; CPU; excludes Teacher",
        }
    )
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def _write_report(
    output: Path,
    config: dict[str, Any],
    split_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]],
    overhead_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> Path:
    report = output / "report" / "darc_v01_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    test_metrics = {
        row["display_name"]: row
        for row in metric_rows
        if row["split"] == "test" and row["scope"] == "combined" and row["checkpoint"] == "all"
    }
    full_budget = [
        row
        for row in policy_rows
        if row.get("scope") == "combined" and float(row.get("budget_fraction", 0)) == 1.0
    ]
    desired_order = [
        *[entry["display_name"] for entry in registry.values()],
        "Joint + Threshold",
        "C3 Myopic",
        "Always CP2",
        "Never Verify",
        "Offline Oracle",
    ]
    selected_policy = {str(row["policy"]): row for row in full_budget}
    lines = [
        "# DARC v0.1 — Downstream-Aware Risk-Calibrated Controller",
        "",
        "> Offline CBV v0.1 experiment. Training/model selection use train+validation only; frozen test is loaded only after every DARC choice is fixed. However, this test was already inspected in the earlier C3 failure analysis, so DARC test results are exploratory rather than a new confirmatory claim.",
        "",
        "## Architecture",
        "",
        "```mermaid",
        "flowchart LR",
        "  T[Public task] --> E[Frozen semantic encoder]",
        "  A[Current artifact] --> E",
        "  H[Observed history] --> E",
        "  R[Current role] --> E",
        "  D[Static downstream roles/dependencies] --> E",
        "  E --> S[State fusion]",
        "  E --> AT[Lightweight downstream attention]",
        "  S --> AT",
        "  AT --> V[Helpful / No-change / Harmful head]",
        "  AT --> C[Expected / upper-cost head]",
        "  V --> U[Ensemble uncertainty + calibrated advantage]",
        "  C --> P[Budget-feasible conservative policy]",
        "  U --> P",
        "```",
        "",
        f"- Frozen encoder: `{config['semantic_encoder']['model_id']}` at revision `{config['semantic_encoder'].get('revision', 'main')}`.",
        "- Intervention advantage: `P(Helpful) - P(Harmful)`.",
        "- Conservative score: `advantage - beta × ensemble_std`, optionally normalized per 1k predicted tokens.",
        f"- Hard budget: DARC Full reserves `{config['evaluation']['hard_call_token_cap']}` tokens before every call and refunds unused capacity after observed usage; actual spend therefore cannot exceed the cap.",
        "- Protocol: maximum one Teacher verification per task. No fitted continuation/RL is used.",
        "",
        "## Leakage and protocol audit",
        "",
        "| Split | Tasks | States | Helpful | Harmful | No-change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in split_rows:
        if row["scope"] == "combined":
            lines.append(
                f"| {row['split']} | {row['tasks']} | {row['states']} | {row['helpful']} | {row['harmful']} | {row['no_change']} |"
            )
    lines.extend(
        [
            "",
            "Inputs contain only the public task, current SSS-prefix artifact, preceding observed artifacts, checkpoint/role, static downstream dependencies, and pre-treatment metadata. Labels, actual Teacher cost, future artifacts, final outcomes, and oracle choices are excluded from inference features.",
            "",
            "## Predictive ablations",
            "",
            "| Ablation | Helpful AUPRC | Harmful AUPRC | Gain Pearson | Helpful ECE | Cost rel. error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in [entry["display_name"] for entry in registry.values()]:
        row = test_metrics.get(name)
        if row:
            lines.append(
                f"| {name} | {_fmt(row['helpful_auprc'])} | {_fmt(row['harmful_auprc'])} | {_fmt(row['gain_pearson'])} | {_fmt(row['helpful_ece'])} | {_fmt(row['cost_mean_relative_error'])} |"
            )
    lines.extend(
        [
            "",
            "![DARC ablations](../figures/darc_ablation_metrics.png)",
            "",
            "## Full-budget policy comparison",
            "",
            "| Policy | Accounting | Accuracy | Gain vs Never (pp) | Calls | Actual tokens | Overshoot | Helpful | Harmful |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in desired_order:
        row = selected_policy.get(name)
        if row:
            lines.append(
                f"| {name} | {row.get('budget_accounting', 'legacy')} | {100 * float(row['accuracy']):.2f}% | {_fmt(row['gain_vs_never_pp'], 2)} | {row['teacher_calls']} | {int(float(row['actual_teacher_tokens'])):,} | {int(float(row.get('actual_budget_overshoot_tokens', 0))):,} | {row['helpful_selected']} | {row['harmful_selected']} |"
            )
    lines.extend(
        [
            "",
            "![Accuracy cost](../figures/darc_accuracy_cost.png)",
            "",
            "![Helpful calibration](../figures/darc_helpful_calibration.png)",
            "",
            "## DARC full failure analysis",
            "",
            "| Scope | Calls | Helpful | Harmful | Wasted/no-change | Missed Helpful | Wrong checkpoint | CP1 / CP2 / CP3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in failure_rows:
        lines.append(
            f"| {row['scope']} | {row['teacher_calls']} | {row['helpful_selected']} | {row['harmful_selected']} | {row['wasted_no_change']} | {row['missed_helpful']} | {row['incorrect_checkpoint']} | {row['cp1_calls']} / {row['cp2_calls']} / {row['cp3_calls']} |"
        )
    lines.extend(
        [
            "",
            "## Semantic input coverage",
            "",
            "Long inputs are chunked into at most the configured number of 512-token windows and mean-pooled. Coverage is measured before model training.",
            "",
            "| Split | Field | Texts | Mean tokens | P95 | >1 window | Mean retained | Minimum retained | Chunks |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in coverage_rows:
        lines.append(
            f"| {row['split']} | {row['field']} | {row['texts']} | {_fmt(row['mean_tokens'], 1)} | {_fmt(row['p95_tokens'], 1)} | {row['over_single_window']} | {100 * float(row['mean_retained_ratio']):.1f}% | {100 * float(row['min_retained_ratio']):.1f}% | {row['encoded_chunks']} |"
        )
    lines.extend(
        [
            "",
            "## Controller overhead",
            "",
            "| Component | Split | States | Seconds | ms/state | Notes |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in overhead_rows:
        lines.append(
            f"| {row['component']} | {row['split']} | {row['states']} | {_fmt(row['seconds'], 4)} | {_fmt(row['milliseconds_per_state'], 3)} | {row['notes']} |"
        )
    full = selected_policy.get("DARC Full")
    joint = selected_policy.get("Joint + Threshold")
    c3 = selected_policy.get("C3 Myopic")
    lines.extend(["", "## Evidence-based interpretation", ""])
    if full and joint and c3:
        darc_accuracy = float(full["accuracy"])
        reference = max(float(joint["accuracy"]), float(c3["accuracy"]))
        difference = 100.0 * (darc_accuracy - reference)
        lines.append(
            f"- On the frozen test at the full nominal budget, DARC differs from the stronger of Joint MLP/C3 Myopic by **{difference:+.2f} pp**. This is an observed estimate, not proof of superiority."
        )
        lines.append(
            f"- DARC uses {int(float(full['actual_teacher_tokens'])):,} actual Teacher tokens and {full['teacher_calls']} calls; comparisons should use the entire frontier and paired confidence intervals rather than accuracy alone."
        )
    lines.extend(
        [
            "- Improvements in AUPRC without policy improvement would mean representation/ranking improved but conservative admission or calibration remains limiting.",
            "- Downstream attention is meaningful only if A3 beats both A1 (no downstream module) and A2 (checkpoint-only capacity control); the architecture does not assume attention must help.",
            "- The frozen test was already consumed by the preceding C3/failure-analysis work. DARC does not tune on it, but its results remain exploratory because the method was motivated after inspecting that test. A newly frozen holdout is required for confirmation.",
            "",
            "## Failure analysis and next steps",
            "",
            "1. Inspect `tables/darc_full_decision_trace.csv` for missed Helpful, selected Harmful, and no-change calls by checkpoint/dataset.",
            "2. If calibration improves but AUPRC remains low, prioritize better paired representations or new training tasks—not a more aggressive margin.",
            "3. If A3 improves ranking but A4 loses accuracy, retune only on validation or grouped train CV; do not tune against this test.",
            "4. Only after DARC demonstrates stable immediate-value gains should continuation or multi-verification control be revisited.",
            "",
            "## Artifacts",
            "",
            "- `selection_manifest.json`: immutable validation-selected specification before test loading.",
            "- `tables/model_selection.csv`: loss candidate selection.",
            "- `tables/training_history.csv`: per-member epoch losses, validation NLL, best epoch, and early-stopping state.",
            "- `tables/validation_policy_grid.csv`: conservative margin/beta/cost search.",
            "- `tables/prediction_metrics.csv`: validation/test metrics by dataset/checkpoint.",
            "- `tables/policy_results.csv`: DARC and existing baseline budget frontiers.",
            "- `tables/darc_full_decision_trace.csv`: decision-level audit.",
            "- `tables/darc_full_failure_summary.csv`: Helpful/Harmful/wasted/missed error counts.",
            "- `tables/semantic_coverage.csv`: per-field token lengths, chunk counts, and retained coverage.",
            "- `figures/`: PNG/PDF paper-ready figures.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_darc(
    config_path: Path,
    *,
    project_root: Path,
    resume: bool = True,
) -> Path:
    config = load_config(config_path)
    frozen_root = _absolute(project_root, config["data"]["frozen_root"])
    baseline_root = _absolute(project_root, config["data"]["baseline_results"])
    output = _absolute(project_root, config["output"]["root"])
    cache_dir = _absolute(project_root, config["semantic_encoder"]["cache_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)

    # Stage 1: test trajectories are intentionally not read here.
    train_examples, train_groups = _load_split(frozen_root, "train")
    validation_examples, _validation_groups = _load_split(frozen_root, "validation")
    partitions: dict[str, list[DARCExample]] = {
        "train": train_examples,
        "validation": validation_examples,
    }
    _assert_disjoint(partitions)
    hard_call_token_cap = int(config["evaluation"]["hard_call_token_cap"])
    _assert_hard_call_cap(
        train_examples, split="train", hard_call_token_cap=hard_call_token_cap
    )
    _assert_hard_call_cap(
        validation_examples,
        split="validation",
        hard_call_token_cap=hard_call_token_cap,
    )
    train_encoded, train_cache = encode_or_load(
        train_examples,
        split="train",
        config=config["semantic_encoder"],
        cache_dir=cache_dir,
    )
    validation_encoded, validation_cache = encode_or_load(
        validation_examples,
        split="validation",
        config=config["semantic_encoder"],
        cache_dir=cache_dir,
    )
    registry, model_selection, temperature_rows = _select_and_train_models(
        train_examples,
        train_encoded,
        validation_examples,
        validation_encoded,
        config,
        output,
        resume=resume,
    )
    training_history = _training_history_rows(registry)
    validation_predictions = {
        name: entry["model"].predict(
            validation_encoded, device_name=str(config["semantic_encoder"].get("device", "auto"))
        )
        for name, entry in registry.items()
    }
    bins = int(config["calibration"].get("ece_bins", 10))
    metric_rows = _metric_rows(
        "validation", validation_examples, registry, validation_predictions, bins=bins
    )

    # Persist the complete selection before test data is opened.
    selection_payload = {
        "schema_version": "darc-selection-v1",
        "config_sha256": _json_hash(config),
        "train_fingerprint": dataset_fingerprint(train_examples),
        "validation_fingerprint": dataset_fingerprint(validation_examples),
        "test_loaded_during_selection": False,
        "selected": {
            name: {
                "model_path": entry["model_path"],
                "temperature": entry["temperature"],
                "policy": entry["policy"],
                "loss_mode": entry["model"].loss_mode,
                "downstream_mode": entry["model"].downstream_mode,
            }
            for name, entry in registry.items()
        },
    }
    selection_path = output / "selection_manifest.json"
    selection_path.write_text(json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8")
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()

    # Stage 2: one-shot frozen-test encoding/evaluation after every choice is fixed.
    test_examples, test_groups = _load_split(frozen_root, "test")
    partitions["test"] = test_examples
    _assert_disjoint(partitions)
    _assert_hard_call_cap(
        test_examples, split="test", hard_call_token_cap=hard_call_token_cap
    )
    if hashlib.sha256(selection_path.read_bytes()).hexdigest() != selection_sha:
        raise AssertionError("DARC selection changed after test loading")
    test_encoded, test_cache = encode_or_load(
        test_examples,
        split="test",
        config=config["semantic_encoder"],
        cache_dir=cache_dir,
    )
    device_name = str(config["semantic_encoder"].get("device", "auto"))
    test_predictions = {
        name: entry["model"].predict(test_encoded, device_name=device_name)
        for name, entry in registry.items()
    }
    metric_rows.extend(_metric_rows("test", test_examples, registry, test_predictions, bins=bins))
    seed = int(config["reproducibility"]["seed"])
    bootstrap = int(config["reproducibility"]["bootstrap_repetitions"])
    darc_policy_rows, decision_traces = _evaluate_policies(
        test_examples,
        train_examples,
        registry,
        test_predictions,
        config,
        seed=seed,
        bootstrap=bootstrap,
    )
    baseline_policy_rows = _baseline_rows(
        baseline_root, set(config["evaluation"]["baseline_policies"])
    )
    policy_rows = [*darc_policy_rows, *baseline_policy_rows]
    full_trace = [row for row in decision_traces if row["ablation"] == "darc_full"]
    failure_rows = _failure_rows(test_examples, full_trace)
    calibration_bins = [
        *_calibration_bins(
            "validation",
            validation_examples,
            validation_predictions["darc_full"],
            bins,
        ),
        *_calibration_bins("test", test_examples, test_predictions["darc_full"], bins),
    ]
    split_rows = _split_rows(partitions)
    cache_manifests = {
        "train": train_cache,
        "validation": validation_cache,
        "test": test_cache,
    }
    coverage_rows = _coverage_rows(cache_manifests)
    overhead_rows: list[dict[str, Any]] = []
    for split, examples, cache in (
        ("train", train_examples, train_cache),
        ("validation", validation_examples, validation_cache),
        ("test", test_examples, test_cache),
    ):
        cold_seconds = float(cache.get("build_seconds", 0.0))
        overhead_rows.append(
            {
                "component": "Frozen encoder cold build",
                "split": split,
                "states": len(examples),
                "seconds": cold_seconds,
                "milliseconds_per_state": 1000.0 * cold_seconds / max(len(examples), 1),
                "notes": "Measured when cache was created; includes model load, tokenization and all chunks",
            }
        )
        if cache.get("cache_hit"):
            warm_seconds = float(cache.get("cache_load_seconds", 0.0))
            overhead_rows.append(
                {
                    "component": "Frozen encoder cache load",
                    "split": split,
                    "states": len(examples),
                    "seconds": warm_seconds,
                    "milliseconds_per_state": 1000.0 * warm_seconds / max(len(examples), 1),
                    "notes": "Current-run NPZ load; no encoder forward pass",
                }
            )
    for name, prediction in test_predictions.items():
        seconds = float(prediction.latency_seconds)
        overhead_rows.append(
            {
                "component": registry[name]["display_name"] + " neural heads",
                "split": "test",
                "states": len(test_examples),
                "seconds": seconds,
                "milliseconds_per_state": 1000.0 * seconds / max(len(test_examples), 1),
                "notes": f"{len(registry[name]['model'].members)}-member ensemble; excludes Teacher",
            }
        )
    overhead_rows.extend(
        _measure_baseline_overhead(
            baseline_root, train_groups, test_groups, seed=seed
        )
    )

    _write_csv(output / "tables" / "dataset_audit.csv", split_rows)
    _write_csv(output / "tables" / "model_selection.csv", model_selection)
    _write_csv(output / "tables" / "training_history.csv", training_history)
    _write_csv(output / "tables" / "temperature_grid.csv", temperature_rows)
    _write_csv(output / "tables" / "prediction_metrics.csv", metric_rows)
    _write_csv(output / "tables" / "policy_results.csv", policy_rows)
    _write_csv(output / "tables" / "darc_full_decision_trace.csv", full_trace)
    _write_csv(output / "tables" / "darc_full_failure_summary.csv", failure_rows)
    _write_csv(output / "tables" / "calibration_bins.csv", calibration_bins)
    _write_csv(output / "tables" / "controller_overhead.csv", overhead_rows)
    _write_csv(output / "tables" / "semantic_coverage.csv", coverage_rows)
    _make_figures(output, metric_rows, policy_rows, calibration_bins)
    report = _write_report(
        output,
        config,
        split_rows,
        metric_rows,
        policy_rows,
        registry,
        overhead_rows,
        failure_rows,
        coverage_rows,
    )
    manifest = {
        "schema_version": "darc-v0.1-experiment-v1",
        "config_path": str(config_path),
        "config_sha256": _json_hash(config),
        "frozen_root": str(frozen_root),
        "baseline_root": str(baseline_root),
        "selection_manifest_sha256": selection_sha,
        "split_fingerprints": {
            split: dataset_fingerprint(examples) for split, examples in partitions.items()
        },
        "split_disjoint": True,
        "test_use": "exploratory only; loaded after selection freeze but previously consumed by C3 analysis",
        "student_teacher_inference": False,
        "max_teacher_calls_per_task": 1,
        "parameter_counts": {
            name: parameter_count(entry["model"]) for name, entry in registry.items()
        },
        "report": str(report),
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "report": str(report),
        "selection_manifest": str(selection_path),
        "tables": str(output / "tables"),
        "figures": str(output / "figures"),
        "darc_full_validation_policy": registry["darc_full"]["policy"],
        "darc_full_temperature": registry["darc_full"]["temperature"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return report
