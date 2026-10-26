"""Training and offline evaluation orchestration for Joint Value-Cost learning."""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

from agent_verify.analysis.joint_value_cost_data import (
    VALUE_CLASSES,
    VALUE_TO_INDEX,
    FrozenSplit,
    JointExample,
    audit_rows,
    dataset_fingerprint,
    filter_examples,
    load_frozen_split,
    load_joint_examples,
    source_exclusions,
)
from agent_verify.analysis.joint_value_cost_models import (
    JointValueCostModel,
    SeparateValueCostModel,
    ValueCostPrediction,
    calibrate_probabilities,
    choose_temperature,
    training_sample_weights,
)
from agent_verify.analysis.joint_value_cost_policy import tune_gain_per_cost_threshold

TRAINING_DATASETS = {
    "humaneval": {"humaneval"},
    "livecodebench": {"livecodebench"},
    "combined": {"humaneval", "livecodebench"},
}
MODEL_KINDS = ("joint", "separate")


@dataclass(frozen=True)
class CandidateResult:
    model_kind: str
    candidate_index: int
    params: dict[str, Any]
    selection_score: float
    temperature: float
    gain_per_1k_threshold: float
    metrics: dict[str, float]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _model(model_kind: str, params: dict[str, Any], seed: int):
    if model_kind == "joint":
        return JointValueCostModel(params, seed)
    if model_kind == "separate":
        return SeparateValueCostModel(params, seed)
    raise ValueError(f"Unknown model kind: {model_kind}")


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(average_precision_score(labels, scores))
        if np.any(labels == 1)
        else float("nan")
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (scores >= lower) & (scores < upper if upper < 1.0 else scores <= upper)
        if np.any(mask):
            total += float(mask.mean()) * abs(float(scores[mask].mean() - labels[mask].mean()))
    return total


def prediction_metrics(
    examples: list[JointExample], prediction: ValueCostPrediction
) -> dict[str, float]:
    labels = np.asarray([row.value_label for row in examples], dtype=int)
    delta = np.asarray([row.delta for row in examples], dtype=float)
    actual_cost = np.asarray([row.teacher_total_tokens for row in examples], dtype=float)
    probabilities = np.clip(prediction.value_probabilities, 1e-9, 1.0)
    helpful = (labels == VALUE_TO_INDEX["helpful"]).astype(int)
    harmful = (labels == VALUE_TO_INDEX["harmful"]).astype(int)
    absolute_cost_error = np.abs(prediction.expected_total_tokens - actual_cost)
    relative_cost_error = absolute_cost_error / np.maximum(actual_cost, 1.0)
    one_hot = np.eye(len(VALUE_CLASSES))[labels]
    return {
        "value_log_loss": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "value_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "helpful_auprc": _safe_ap(helpful, probabilities[:, VALUE_TO_INDEX["helpful"]]),
        "harmful_auprc": _safe_ap(harmful, probabilities[:, VALUE_TO_INDEX["harmful"]]),
        "helpful_ece": _ece(helpful, probabilities[:, VALUE_TO_INDEX["helpful"]]),
        "harmful_ece": _ece(harmful, probabilities[:, VALUE_TO_INDEX["harmful"]]),
        "gain_pearson": _correlation(prediction.expected_gain, delta),
        "cost_mae": float(absolute_cost_error.mean()),
        "cost_median_absolute_error": float(np.median(absolute_cost_error)),
        "cost_mean_relative_error": float(relative_cost_error.mean()),
        "cost_median_relative_error": float(np.median(relative_cost_error)),
        "cost_upper_coverage": float(np.mean(actual_cost <= prediction.upper_total_tokens)),
        "cost_mean_actual": float(actual_cost.mean()),
        "cost_mean_predicted": float(prediction.expected_total_tokens.mean()),
    }


def reliability_rows(
    examples: list[JointExample],
    prediction: ValueCostPrediction,
    *,
    model_kind: str,
    train_mode: str,
    scope: str,
    bins: int = 10,
) -> list[dict[str, Any]]:
    labels = np.asarray([row.value_label for row in examples], dtype=int)
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for target in ("helpful", "harmful"):
        target_index = VALUE_TO_INDEX[target]
        observed = (labels == target_index).astype(float)
        scores = prediction.value_probabilities[:, target_index]
        for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            mask = (scores >= lower) & (scores < upper if upper < 1.0 else scores <= upper)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "train_mode": train_mode,
                    "model": model_kind,
                    "scope": scope,
                    "target": target,
                    "bin": bin_index,
                    "count": int(mask.sum()),
                    "mean_predicted": float(scores[mask].mean()),
                    "observed_rate": float(observed[mask].mean()),
                }
            )
    return rows


def _group_folds(examples: list[JointExample], folds: int):
    groups = np.asarray([row.group_key for row in examples])
    splitter = GroupKFold(n_splits=folds)
    for train_indices, validation_indices in splitter.split(np.zeros(len(examples)), groups=groups):
        train_groups = {examples[int(index)].group_key for index in train_indices}
        validation_groups = {examples[int(index)].group_key for index in validation_indices}
        if train_groups.intersection(validation_groups):
            raise AssertionError("Task leakage inside GroupKFold")
        yield train_indices, validation_indices


def oof_prediction(
    examples: list[JointExample],
    *,
    model_kind: str,
    params: dict[str, Any],
    folds: int,
    seed: int,
    progress_label: str | None = None,
) -> ValueCostPrediction:
    probabilities = np.empty((len(examples), len(VALUE_CLASSES)), dtype=float)
    expected_cost = np.empty(len(examples), dtype=float)
    upper_cost = np.empty(len(examples), dtype=float)
    fold_iterator = tqdm(
        list(_group_folds(examples, folds)),
        total=folds,
        desc=progress_label or f"{model_kind} grouped CV",
        unit="fold",
        dynamic_ncols=True,
        leave=False,
    )
    for fold, (train_indices, validation_indices) in enumerate(fold_iterator):
        train = [examples[int(index)] for index in train_indices]
        validation = [examples[int(index)] for index in validation_indices]
        predictor = _model(model_kind, params, seed + fold * 10_007).fit(train)
        prediction = predictor.predict(validation)
        probabilities[validation_indices] = prediction.value_probabilities
        expected_cost[validation_indices] = prediction.expected_total_tokens
        upper_cost[validation_indices] = prediction.upper_total_tokens
    gain = probabilities[:, VALUE_TO_INDEX["helpful"]] - probabilities[:, VALUE_TO_INDEX["harmful"]]
    return ValueCostPrediction(probabilities, gain, expected_cost, upper_cost)


def _selection_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    harmful = metrics["harmful_auprc"]
    correlation = metrics["gain_pearson"]
    return (
        weights["helpful_auprc"] * metrics["helpful_auprc"]
        + weights["harmful_auprc"] * (0.0 if np.isnan(harmful) else harmful)
        + weights["gain_correlation"] * (0.0 if np.isnan(correlation) else correlation)
        - weights["value_log_loss"] * metrics["value_log_loss"]
        - weights["cost_relative_error"] * metrics["cost_mean_relative_error"]
    )


def select_candidate(
    examples: list[JointExample],
    *,
    model_kind: str,
    candidates: list[dict[str, Any]],
    training_config: dict[str, Any],
    policy_config: dict[str, Any],
    seed: int,
) -> tuple[CandidateResult, ValueCostPrediction, list[dict[str, Any]]]:
    labels = np.asarray([row.value_label for row in examples], dtype=int)
    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, int] | None = None
    best_result: CandidateResult | None = None
    best_prediction: ValueCostPrediction | None = None
    candidate_iterator = tqdm(
        enumerate(candidates),
        total=len(candidates),
        desc=f"{model_kind} candidates",
        unit="candidate",
        dynamic_ncols=True,
    )
    for candidate_index, params in candidate_iterator:
        oof = oof_prediction(
            examples,
            model_kind=model_kind,
            params=params,
            folds=int(training_config["group_folds"]),
            seed=seed + candidate_index * 100_003,
            progress_label=(
                f"{model_kind} candidate {candidate_index + 1}/{len(candidates)}"
            ),
        )
        weights = training_sample_weights(
            examples,
            class_balanced=bool(params.get("class_balanced", True)),
            domain_balanced=bool(params.get("domain_balanced", False)),
        )
        temperature = choose_temperature(
            oof.value_probabilities,
            labels,
            [float(value) for value in training_config["temperature_grid"]],
            weights,
        )
        calibrated = calibrate_probabilities(oof.value_probabilities, temperature)
        calibrated_prediction = ValueCostPrediction(
            calibrated,
            calibrated[:, VALUE_TO_INDEX["helpful"]]
            - calibrated[:, VALUE_TO_INDEX["harmful"]],
            oof.expected_total_tokens,
            oof.upper_total_tokens,
        )
        metrics = prediction_metrics(examples, calibrated_prediction)
        threshold = tune_gain_per_cost_threshold(
            examples,
            calibrated_prediction,
            thresholds=[float(value) for value in policy_config["gain_per_1k_thresholds"]],
            budget_quantiles=[float(value) for value in policy_config["budget_quantiles"]],
            seed=seed + candidate_index,
        )
        score = _selection_score(metrics, training_config["selection_weights"])
        result = CandidateResult(
            model_kind=model_kind,
            candidate_index=candidate_index,
            params=dict(params),
            selection_score=score,
            temperature=temperature,
            gain_per_1k_threshold=threshold,
            metrics=metrics,
        )
        rows.append(
            {
                "model": model_kind,
                "candidate_index": candidate_index,
                "params_json": json.dumps(params, sort_keys=True),
                "selection_score": score,
                "temperature": temperature,
                "gain_per_1k_threshold": threshold,
                **metrics,
            }
        )
        candidate_iterator.set_postfix(
            score=f"{score:.3f}",
            helpful_ap=f"{metrics['helpful_auprc']:.3f}",
            cost_rel=f"{metrics['cost_mean_relative_error']:.3f}",
        )
        key = (score, -metrics["cost_mean_relative_error"], -candidate_index)
        if best_key is None or key > best_key:
            best_key = key
            best_result = result
            best_prediction = calibrated_prediction
    assert best_result is not None and best_prediction is not None
    return best_result, best_prediction, rows


def _prediction_rows(
    examples: list[JointExample],
    prediction: ValueCostPrediction,
    *,
    train_mode: str,
    model_kind: str,
    scope: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, example in enumerate(examples):
        rows.append(
            {
                "train_mode": train_mode,
                "model": model_kind,
                "scope": scope,
                "dataset": example.dataset,
                "difficulty": example.difficulty,
                "task_id": example.task_id,
                "checkpoint": example.checkpoint,
                "role": example.role,
                "p_harmful": float(prediction.value_probabilities[index, 0]),
                "p_no_change": float(prediction.value_probabilities[index, 1]),
                "p_helpful": float(prediction.value_probabilities[index, 2]),
                "expected_gain": float(prediction.expected_gain[index]),
                "predicted_total_tokens": float(prediction.expected_total_tokens[index]),
                "predicted_upper_tokens": float(prediction.upper_total_tokens[index]),
                "y_skip": example.y_skip,
                "y_verify": example.y_verify,
                "delta": example.delta,
                "value_label": VALUE_CLASSES[example.value_label],
                "teacher_input_tokens": example.teacher_input_tokens,
                "teacher_output_tokens": example.teacher_output_tokens,
                "teacher_total_tokens": example.teacher_total_tokens,
                "local_risk": example.local_risk,
            }
        )
    return rows


def prepare_data(
    config: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[list[JointExample], FrozenSplit]:
    examples = load_joint_examples(config)
    split = load_frozen_split(config, examples)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = audit_rows(examples, split)
    write_csv(output_dir / "split_summary.csv", summary)
    exclusions = source_exclusions(config)
    if exclusions:
        write_csv(output_dir / "excluded_incomplete_tasks.csv", exclusions)
    index_rows = [
        {
            "dataset": row.dataset,
            "difficulty": row.difficulty,
            "task_id": row.task_id,
            "checkpoint": row.checkpoint,
            "partition": "test" if row.group_key in split.test_group_keys else "train",
            "value_label": VALUE_CLASSES[row.value_label],
            "delta": row.delta,
            "teacher_input_tokens": row.teacher_input_tokens,
            "teacher_output_tokens": row.teacher_output_tokens,
            "teacher_total_tokens": row.teacher_total_tokens,
        }
        for row in examples
    ]
    write_csv(output_dir / "sample_index.csv", index_rows)
    audit = {
        "schema_version": 1,
        "dataset_fingerprint": dataset_fingerprint(examples),
        "tasks": len({row.group_key for row in examples}),
        "pairs": len(examples),
        "rows_per_task": sorted(
            {
                sum(other.group_key == row.group_key for other in examples)
                for row in examples
            }
        ),
        "train_test_group_overlap": sorted(
            split.train_group_keys.intersection(split.test_group_keys)
        ),
        "frozen_manifest": str(split.manifest_path),
        "frozen_assignments": str(split.assignments_path),
        "frozen_test_status": "exploratory_previously_observed",
        "feature_fit_scope": "training folds only",
        "task_id_used_as_feature": False,
        "post_treatment_features_used": False,
        "teacher_cost_definition": "Teacher input_tokens + output_tokens only",
        "excluded_incomplete_tasks": exclusions,
        "valid_complete_samples_silently_dropped": False,
    }
    (output_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return examples, split


def train_mode(
    config: dict[str, Any],
    *,
    mode: str,
    output_root: Path,
    resume: bool,
) -> None:
    if mode not in TRAINING_DATASETS:
        raise ValueError(f"Unknown training mode: {mode}")
    examples, split = prepare_data(config, output_dir=output_root / "data")
    train = filter_examples(
        examples,
        split,
        partition="train",
        datasets=TRAINING_DATASETS[mode],
    )
    mode_dir = output_root / "training" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    expected = [mode_dir / f"{kind}_model.joblib" for kind in MODEL_KINDS]
    if resume and all(path.exists() for path in expected) and (mode_dir / "training_manifest.json").exists():
        tqdm.write(f"[{mode}] already complete; resume skipped training")
        return
    training_config = config["training"]
    seed = int(config["experiment"]["seed"])
    selection_rows: list[dict[str, Any]] = []
    selected_payload: dict[str, Any] = {}
    oof_rows: list[dict[str, Any]] = []
    tqdm.write(
        f"[{mode}] training {len({row.group_key for row in train})} tasks, "
        f"{len(train)} checkpoint pairs"
    )
    for offset, model_kind in enumerate(MODEL_KINDS):
        result, oof, candidate_rows = select_candidate(
            train,
            model_kind=model_kind,
            candidates=[dict(row) for row in config["models"][model_kind]["candidates"]],
            training_config=training_config,
            policy_config=config["policy"],
            seed=seed + offset * 1_000_003,
        )
        selection_rows.extend(candidate_rows)
        predictor = _model(model_kind, result.params, seed + offset * 2_000_003).fit(
            train,
            progress_desc=f"{mode} {model_kind} final fit",
        )
        predictor.temperature = result.temperature
        joblib.dump(predictor, mode_dir / f"{model_kind}_model.joblib", compress=3)
        selected_payload[model_kind] = asdict(result)
        oof_rows.extend(
            _prediction_rows(
                train,
                oof,
                train_mode=mode,
                model_kind=model_kind,
                scope="train_oof",
            )
        )
    write_csv(mode_dir / "candidate_selection.csv", selection_rows)
    write_csv(mode_dir / "oof_predictions.csv", oof_rows)
    (mode_dir / "selected_models.json").write_text(
        json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "experiment": "joint_value_cost_learning",
        "training_mode": mode,
        "datasets": sorted(TRAINING_DATASETS[mode]),
        "train_tasks": len({row.group_key for row in train}),
        "train_pairs": len(train),
        "value_counts": {
            label: sum(row.value_label == index for row in train)
            for index, label in enumerate(VALUE_CLASSES)
        },
        "dataset_fingerprint": dataset_fingerprint(examples),
        "frozen_test_used_for_selection": False,
        "inner_cv": f"GroupKFold(dataset::task_id), {training_config['group_folds']} folds",
        "domain_balancing_available": True,
        "seed": seed,
        "llm_inference_performed": False,
        "teacher_calls_performed": False,
        "evaluator_calls_performed": False,
        "python": sys.version,
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
    }
    (mode_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def evaluate_all(config: dict[str, Any], *, output_root: Path) -> None:
    examples, split = prepare_data(config, output_dir=output_root / "data")
    scopes = {
        "humaneval": filter_examples(
            examples, split, partition="test", datasets={"humaneval"}
        ),
        "livecodebench": filter_examples(
            examples, split, partition="test", datasets={"livecodebench"}
        ),
        "combined": filter_examples(examples, split, partition="test"),
    }
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    evaluation_progress = tqdm(
        total=len(TRAINING_DATASETS) * len(MODEL_KINDS) * len(scopes),
        desc="cross-domain evaluation",
        unit="model-scope",
        dynamic_ncols=True,
    )
    for train_mode, training_datasets in TRAINING_DATASETS.items():
        mode_dir = output_root / "training" / train_mode
        for model_kind in MODEL_KINDS:
            model_path = mode_dir / f"{model_kind}_model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Missing {model_path}; train {train_mode} before evaluation"
                )
            predictor = joblib.load(model_path)
            for scope, test in scopes.items():
                prediction = predictor.predict(test)
                cross_domain = scope != "combined" and scope not in training_datasets
                common = {
                    "train_mode": train_mode,
                    "model": model_kind,
                    "scope": scope,
                    "checkpoint": "all",
                    "test_tasks": len({row.group_key for row in test}),
                    "test_pairs": len(test),
                    "cross_domain": cross_domain,
                    "evaluation_status": "exploratory_previously_observed_frozen_test",
                }
                metrics_rows.append({**common, **prediction_metrics(test, prediction)})
                for checkpoint in (1, 2, 3):
                    indices = [index for index, row in enumerate(test) if row.checkpoint == checkpoint]
                    subset = [test[index] for index in indices]
                    subset_prediction = ValueCostPrediction(
                        prediction.value_probabilities[indices],
                        prediction.expected_gain[indices],
                        prediction.expected_total_tokens[indices],
                        prediction.upper_total_tokens[indices],
                    )
                    metrics_rows.append(
                        {
                            **common,
                            "checkpoint": f"CP{checkpoint}",
                            "test_tasks": len(subset),
                            "test_pairs": len(subset),
                            **prediction_metrics(subset, subset_prediction),
                        }
                    )
                prediction_rows.extend(
                    _prediction_rows(
                        test,
                        prediction,
                        train_mode=train_mode,
                        model_kind=model_kind,
                        scope=scope,
                    )
                )
                calibration_rows.extend(
                    reliability_rows(
                        test,
                        prediction,
                        model_kind=model_kind,
                        train_mode=train_mode,
                        scope=scope,
                    )
                )
                evaluation_progress.set_postfix(
                    train=train_mode, model=model_kind, test=scope
                )
                evaluation_progress.update()
    evaluation_progress.close()
    evaluation_dir = output_root / "evaluation"
    write_csv(evaluation_dir / "prediction_metrics.csv", metrics_rows)
    write_csv(evaluation_dir / "predictions.csv", prediction_rows)
    write_csv(evaluation_dir / "value_calibration.csv", calibration_rows)
    (evaluation_dir / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "status": "exploratory_previously_observed_frozen_test",
                "frozen_test_reused": True,
                "winner_selection_allowed": False,
                "dataset_fingerprint": dataset_fingerprint(examples),
                "test_tasks": len(split.test_group_keys),
                "llm_inference_performed": False,
                "teacher_calls_performed": False,
                "evaluator_calls_performed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
