"""Train and evaluate leakage-safe downstream verification value predictors."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from agent_verify.analysis.phase2_data import PairExample, load_phase1_examples
from agent_verify.analysis.phase2_models import (
    TwoHeadPredictor,
    group_folds,
    select_hyperparameters,
    simulate_sequential_policy,
)

DEFAULT_RUN_ID = "h1_phase1_qwen25_gpu2-09092ebd5e-d03859f54f"
MODEL_DISPLAY = {
    "metadata_logistic": "Metadata-only Logistic",
    "semantic_logistic": "Ours Logistic",
    "semantic_boosting": "Ours Boosting",
}
POLICY_ORDER = [
    "Never Verify",
    "Always Verify CP1",
    "Always Verify CP2",
    "Always Verify CP3",
    "Random Checkpoint",
    "Local Risk Controller",
    "Highest Risk (offline lookahead)",
    "Metadata-only Logistic",
    "Ours Logistic",
    "Ours Boosting",
    "Offline Oracle",
]


@dataclass
class OOFResult:
    model_name: str
    q_skip: np.ndarray
    q_verify: np.ndarray
    fold: np.ndarray
    threshold: np.ndarray
    selected: np.ndarray
    selections: list[dict[str, Any]]


@dataclass
class PolicyResult:
    name: str
    outcomes: np.ndarray
    calls: np.ndarray
    tokens: np.ndarray
    latency: np.ndarray
    usd: np.ndarray
    helpful: np.ndarray
    harmful: np.ndarray
    unchanged: np.ndarray
    selected_checkpoint: np.ndarray
    definition: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dataset", type=Path, default=Path("data/HumanEval.jsonl.gz"))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_261_024)
    return parser.parse_args()


def task_order(examples: list[PairExample]) -> list[str]:
    return list(dict.fromkeys(example.task_id for example in examples))


def train_nested_oof(
    examples: list[PairExample],
    *,
    model_name: str,
    output_dir: Path,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> OOFResult:
    q_skip = np.empty(len(examples), dtype=float)
    q_verify = np.empty(len(examples), dtype=float)
    fold_ids = np.empty(len(examples), dtype=int)
    thresholds = np.empty(len(examples), dtype=float)
    selections: list[dict[str, Any]] = []
    checkpoint_dir = output_dir / "model_checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_indices, test_indices) in enumerate(group_folds(examples, outer_folds)):
        train = [examples[int(index)] for index in train_indices]
        test = [examples[int(index)] for index in test_indices]
        selection = select_hyperparameters(
            train,
            model_name,
            inner_splits=inner_folds,
            seed=seed + fold * 10_007,
        )
        predictor = TwoHeadPredictor(
            model_name,
            selection.params,
            seed + fold * 100_003,
        ).fit(train)
        skip_prediction, verify_prediction = predictor.predict(test)
        q_skip[test_indices] = skip_prediction
        q_verify[test_indices] = verify_prediction
        fold_ids[test_indices] = fold
        thresholds[test_indices] = selection.threshold
        joblib.dump(predictor, checkpoint_dir / f"fold_{fold}.joblib", compress=3)
        fold_record = {
            "fold": fold,
            "train_tasks": len({example.task_id for example in train}),
            "test_tasks": len({example.task_id for example in test}),
            **asdict(selection),
        }
        selections.append(fold_record)
        (checkpoint_dir / f"fold_{fold}.json").write_text(
            json.dumps(fold_record, indent=2) + "\n", encoding="utf-8"
        )

    _, selected, _ = simulate_sequential_policy(
        examples, q_verify - q_skip, thresholds
    )
    selected_mask = np.zeros(len(examples), dtype=int)
    for index in selected:
        if index >= 0:
            selected_mask[index] = 1

    final_selection = select_hyperparameters(
        examples,
        model_name,
        inner_splits=outer_folds,
        seed=seed + 900_001,
    )
    final_predictor = TwoHeadPredictor(
        model_name,
        final_selection.params,
        seed + 900_101,
    ).fit(examples)
    joblib.dump(final_predictor, checkpoint_dir / "final_model.joblib", compress=3)
    final_record = {"scope": "all_139_tasks_not_used_for_oof_metrics", **asdict(final_selection)}
    (checkpoint_dir / "final_model.json").write_text(
        json.dumps(final_record, indent=2) + "\n", encoding="utf-8"
    )
    return OOFResult(
        model_name=model_name,
        q_skip=q_skip,
        q_verify=q_verify,
        fold=fold_ids,
        threshold=thresholds,
        selected=selected_mask,
        selections=selections,
    )


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(labels, scores)) if np.any(labels == 1) else float("nan")


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def predictor_metric_rows(
    examples: list[PairExample], results: dict[str, OOFResult]
) -> list[dict[str, Any]]:
    y_skip = np.asarray([example.y_skip for example in examples], dtype=int)
    y_verify = np.asarray([example.y_verify for example in examples], dtype=int)
    delta = y_verify - y_skip
    helpful = (delta > 0).astype(int)
    harmful = (delta < 0).astype(int)
    rows: list[dict[str, Any]] = []
    risk = np.asarray([example.local_risk for example in examples], dtype=float)
    rows.append(
        {
            "model": "Local Risk heuristic",
            "q_skip_brier": "",
            "q_verify_brier": "",
            "joint_log_loss": "",
            "delta_mae": float(np.mean(np.abs(risk - delta))),
            "delta_mse": float(np.mean((risk - delta) ** 2)),
            "delta_pearson": _correlation(risk, delta),
            "delta_spearman": _correlation(_rank(risk), _rank(delta.astype(float))),
            "helpful_auroc": _safe_auc(helpful, risk),
            "helpful_average_precision": _safe_ap(helpful, risk),
            "harmful_auroc": _safe_auc(harmful, -risk),
            "evaluation": "observed heuristic score; no training",
        }
    )
    for model_name, result in results.items():
        predicted_delta = result.q_verify - result.q_skip
        clipped = np.clip(np.concatenate([result.q_skip, result.q_verify]), 1e-7, 1 - 1e-7)
        labels = np.concatenate([y_skip, y_verify])
        rows.append(
            {
                "model": MODEL_DISPLAY[model_name],
                "q_skip_brier": brier_score_loss(y_skip, result.q_skip),
                "q_verify_brier": brier_score_loss(y_verify, result.q_verify),
                "joint_log_loss": log_loss(labels, clipped, labels=[0, 1]),
                "delta_mae": float(np.mean(np.abs(predicted_delta - delta))),
                "delta_mse": float(np.mean((predicted_delta - delta) ** 2)),
                "delta_pearson": _correlation(predicted_delta, delta),
                "delta_spearman": _correlation(
                    _rank(predicted_delta), _rank(delta.astype(float))
                ),
                "helpful_auroc": _safe_auc(helpful, predicted_delta),
                "helpful_average_precision": _safe_ap(helpful, predicted_delta),
                "harmful_auroc": _safe_auc(harmful, -predicted_delta),
                "evaluation": "5-fold task-grouped out-of-fold",
            }
        )
    return rows


def _selected_policy(
    name: str,
    examples: list[PairExample],
    selected_indices: np.ndarray,
    definition: str,
) -> PolicyResult:
    tasks = task_order(examples)
    grouped: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(example.task_id, []).append(index)
    if len(selected_indices) != len(tasks):
        raise ValueError("One selected index is required per task")
    outcomes = []
    calls = []
    tokens = []
    latency = []
    usd = []
    helpful = []
    harmful = []
    unchanged = []
    checkpoints = []
    for task_id, selected in zip(tasks, selected_indices, strict=True):
        indices = grouped[task_id]
        baseline = examples[indices[0]].y_skip
        if selected < 0:
            outcomes.append(baseline)
            calls.append(0.0)
            tokens.append(0.0)
            latency.append(0.0)
            usd.append(0.0)
            helpful.append(0.0)
            harmful.append(0.0)
            unchanged.append(0.0)
            checkpoints.append(0)
            continue
        example = examples[int(selected)]
        if example.task_id != task_id:
            raise AssertionError("Selected index belongs to another task")
        outcomes.append(example.y_verify)
        calls.append(1.0)
        tokens.append(float(example.teacher_tokens))
        latency.append(example.teacher_latency_seconds)
        usd.append(example.teacher_cost_usd)
        helpful.append(float(example.delta > 0))
        harmful.append(float(example.delta < 0))
        unchanged.append(float(example.delta == 0))
        checkpoints.append(example.checkpoint)
    return PolicyResult(
        name=name,
        outcomes=np.asarray(outcomes, dtype=float),
        calls=np.asarray(calls),
        tokens=np.asarray(tokens),
        latency=np.asarray(latency),
        usd=np.asarray(usd),
        helpful=np.asarray(helpful),
        harmful=np.asarray(harmful),
        unchanged=np.asarray(unchanged),
        selected_checkpoint=np.asarray(checkpoints, dtype=int),
        definition=definition,
    )


def build_policies(
    examples: list[PairExample], oof_results: dict[str, OOFResult]
) -> tuple[np.ndarray, list[PolicyResult]]:
    tasks = task_order(examples)
    grouped: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(example.task_id, []).append(index)
    baseline = np.asarray([examples[grouped[task_id][0]].y_skip for task_id in tasks], dtype=float)
    never = _selected_policy(
        "Never Verify",
        examples,
        np.full(len(tasks), -1, dtype=int),
        "Skip all checkpoints.",
    )
    policies = [never]

    for checkpoint in (1, 2, 3):
        selected = np.asarray(
            [next(i for i in grouped[task_id] if examples[i].checkpoint == checkpoint) for task_id in tasks]
        )
        policies.append(
            _selected_policy(
                f"Always Verify CP{checkpoint}",
                examples,
                selected,
                f"Verify CP{checkpoint} for every task.",
            )
        )

    random_outcomes = []
    random_tokens = []
    random_latency = []
    random_usd = []
    random_helpful = []
    random_harmful = []
    random_unchanged = []
    for task_id in tasks:
        rows = [examples[index] for index in grouped[task_id]]
        random_outcomes.append(np.mean([row.y_verify for row in rows]))
        random_tokens.append(np.mean([row.teacher_tokens for row in rows]))
        random_latency.append(np.mean([row.teacher_latency_seconds for row in rows]))
        random_usd.append(np.mean([row.teacher_cost_usd for row in rows]))
        random_helpful.append(np.mean([row.delta > 0 for row in rows]))
        random_harmful.append(np.mean([row.delta < 0 for row in rows]))
        random_unchanged.append(np.mean([row.delta == 0 for row in rows]))
    policies.append(
        PolicyResult(
            name="Random Checkpoint",
            outcomes=np.asarray(random_outcomes),
            calls=np.ones(len(tasks)),
            tokens=np.asarray(random_tokens),
            latency=np.asarray(random_latency),
            usd=np.asarray(random_usd),
            helpful=np.asarray(random_helpful),
            harmful=np.asarray(random_harmful),
            unchanged=np.asarray(random_unchanged),
            selected_checkpoint=np.zeros(len(tasks), dtype=int),
            definition="Exact expectation for a uniform CP1/CP2/CP3 choice.",
        )
    )

    local_selected = []
    lookahead_selected = []
    oracle_selected = []
    for task_id in tasks:
        indices = sorted(grouped[task_id], key=lambda index: examples[index].checkpoint)
        local_selected.append(next((index for index in indices if examples[index].local_risk >= 0.50), -1))
        max_risk = max(examples[index].local_risk for index in indices)
        lookahead_selected.append(next(index for index in indices if examples[index].local_risk == max_risk))
        positive = [index for index in indices if examples[index].delta > 0]
        oracle_selected.append(positive[0] if positive else -1)
    policies.extend(
        [
            _selected_policy(
                "Local Risk Controller",
                examples,
                np.asarray(local_selected),
                "Online: verify first checkpoint with heuristic risk >= 0.50; otherwise skip all.",
            ),
            _selected_policy(
                "Highest Risk (offline lookahead)",
                examples,
                np.asarray(lookahead_selected),
                "Diagnostic only: inspect all three risks, choose maximum, ties earliest.",
            ),
        ]
    )

    for model_name in ("metadata_logistic", "semantic_logistic", "semantic_boosting"):
        result = oof_results[model_name]
        _, selected, _ = simulate_sequential_policy(
            examples,
            result.q_verify - result.q_skip,
            result.threshold,
        )
        policies.append(
            _selected_policy(
                MODEL_DISPLAY[model_name],
                examples,
                selected,
                "Online sequential controller using task-grouped OOF predictions and fold-specific threshold.",
            )
        )

    policies.append(
        _selected_policy(
            "Offline Oracle",
            examples,
            np.asarray(oracle_selected),
            "Upper bound: first checkpoint with observed positive Delta; skip if none.",
        )
    )
    assert [policy.name for policy in policies] == POLICY_ORDER
    return baseline, policies


def bootstrap_policy_metrics(
    baseline: np.ndarray,
    policies: list[PolicyResult],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    n_tasks = len(baseline)
    samples = rng.integers(0, n_tasks, size=(repetitions, n_tasks))
    by_name = {policy.name: policy for policy in policies}
    cp3 = by_name["Always Verify CP3"].outcomes
    risk = by_name["Local Risk Controller"].outcomes
    output: dict[str, dict[str, tuple[float, float]]] = {}
    for policy in policies:
        outcome = policy.outcomes[samples]
        metrics = {
            "accuracy": outcome.mean(axis=1),
            "vs_never": (outcome - baseline[samples]).mean(axis=1),
            "vs_cp3": (outcome - cp3[samples]).mean(axis=1),
            "vs_local_risk": (outcome - risk[samples]).mean(axis=1),
        }
        output[policy.name] = {
            name: tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))
            for name, values in metrics.items()
        }
    return output


def policy_rows(
    baseline: np.ndarray,
    policies: list[PolicyResult],
    bootstrap: dict[str, dict[str, tuple[float, float]]],
) -> list[dict[str, Any]]:
    by_name = {policy.name: policy for policy in policies}
    cp3 = by_name["Always Verify CP3"].outcomes
    risk = by_name["Local Risk Controller"].outcomes
    rows = []
    for policy in policies:
        intervals = bootstrap[policy.name]
        rows.append(
            {
                "method": policy.name,
                "accuracy": float(policy.outcomes.mean()),
                "accuracy_ci95_low": intervals["accuracy"][0],
                "accuracy_ci95_high": intervals["accuracy"][1],
                "gain_vs_never_pp": float(np.mean(policy.outcomes - baseline) * 100),
                "gain_ci95_low_pp": intervals["vs_never"][0] * 100,
                "gain_ci95_high_pp": intervals["vs_never"][1] * 100,
                "difference_vs_cp3_pp": float(np.mean(policy.outcomes - cp3) * 100),
                "difference_vs_cp3_ci95_low_pp": intervals["vs_cp3"][0] * 100,
                "difference_vs_cp3_ci95_high_pp": intervals["vs_cp3"][1] * 100,
                "difference_vs_local_risk_pp": float(np.mean(policy.outcomes - risk) * 100),
                "difference_vs_local_risk_ci95_low_pp": intervals["vs_local_risk"][0] * 100,
                "difference_vs_local_risk_ci95_high_pp": intervals["vs_local_risk"][1] * 100,
                "average_teacher_calls_per_task": float(policy.calls.mean()),
                "teacher_calls": float(policy.calls.sum()),
                "teacher_tokens": float(policy.tokens.sum()),
                "teacher_latency_seconds": float(policy.latency.sum()),
                "total_cost_usd": float(policy.usd.sum()),
                "helpful": float(policy.helpful.sum()),
                "harmful": float(policy.harmful.sum()),
                "no_change": float(policy.unchanged.sum()),
                "skip_all_rate": float(np.mean(policy.calls == 0)),
                "definition": policy.definition,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_oof_predictions(
    path: Path,
    examples: list[PairExample],
    results: dict[str, OOFResult],
) -> None:
    rows: list[dict[str, Any]] = []
    for model_name, result in results.items():
        for index, example in enumerate(examples):
            rows.append(
                {
                    "model": MODEL_DISPLAY[model_name],
                    "fold": int(result.fold[index]),
                    "task_id": example.task_id,
                    "checkpoint": example.checkpoint,
                    "role": example.role,
                    "q_skip_pred": float(result.q_skip[index]),
                    "q_verify_pred": float(result.q_verify[index]),
                    "delta_pred": float(result.q_verify[index] - result.q_skip[index]),
                    "fold_threshold": float(result.threshold[index]),
                    "selected_by_controller": int(result.selected[index]),
                    "y_skip": example.y_skip,
                    "y_verify": example.y_verify,
                    "delta_observed": example.delta,
                    "local_risk": example.local_risk,
                }
            )
    write_csv(path, rows)


def plot_accuracy_vs_cost(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(rows)))
    for color, row in zip(colors, rows, strict=True):
        x = float(row["teacher_tokens"]) / 1_000
        y = float(row["accuracy"]) * 100
        size = 55 + 150 * float(row["average_teacher_calls_per_task"])
        ax.scatter(x, y, s=size, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(str(row["method"]), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(
        xlabel="Total Teacher tokens (thousands)",
        ylabel="Final accuracy (%)",
        title="Accuracy–verification cost trade-off (139 OOF tasks)",
    )
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_policy_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["method"]) for row in rows]
    accuracy = np.asarray([float(row["accuracy"]) * 100 for row in rows])
    low = np.asarray([float(row["accuracy_ci95_low"]) * 100 for row in rows])
    high = np.asarray([float(row["accuracy_ci95_high"]) * 100 for row in rows])
    calls = np.asarray([float(row["average_teacher_calls_per_task"]) for row in rows])
    y = np.arange(len(labels))

    fig, (ax_accuracy, ax_calls) = plt.subplots(1, 2, figsize=(14.5, 7.6), gridspec_kw={"width_ratios": [1.5, 1]})
    colors = ["#9CA3AF" if "Ours" not in label else "#1769AA" for label in labels]
    ax_accuracy.barh(y, accuracy, color=colors)
    ax_accuracy.errorbar(
        accuracy,
        y,
        xerr=np.vstack([accuracy - low, high - accuracy]),
        fmt="none",
        ecolor="#1F2937",
        capsize=3,
    )
    ax_accuracy.set_yticks(y, labels)
    ax_accuracy.invert_yaxis()
    ax_accuracy.set_xlabel("Final accuracy (%)")
    ax_accuracy.set_title("Task-clustered 95% CIs")
    ax_accuracy.grid(axis="x", alpha=0.2)

    ax_calls.barh(y, calls, color="#E07A2D")
    ax_calls.set_yticks(y, [""] * len(labels))
    ax_calls.invert_yaxis()
    ax_calls.set_xlim(0, 1.05)
    ax_calls.set_xlabel("Teacher calls / task")
    ax_calls.set_title("Realized verification budget")
    ax_calls.grid(axis="x", alpha=0.2)
    fig.suptitle("Phase 2 offline policy evaluation", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _format_float(value: Any, digits: int = 3) -> str:
    if value == "" or value is None:
        return "—"
    number = float(value)
    return "—" if not np.isfinite(number) else f"{number:.{digits}f}"


def failure_case_lines(
    examples: list[PairExample], results: dict[str, OOFResult]
) -> list[str]:
    lines: list[str] = []
    tasks = task_order(examples)
    for model_name in ("semantic_logistic", "semantic_boosting"):
        result = results[model_name]
        predicted_delta = result.q_verify - result.q_skip
        _, selected, _ = simulate_sequential_policy(examples, predicted_delta, result.threshold)
        harmful = [index for index in selected if index >= 0 and examples[index].delta < 0]
        selected_set = {index for index in selected if index >= 0}
        missed = []
        for task_id in tasks:
            task_indices = [i for i, example in enumerate(examples) if example.task_id == task_id]
            if any(examples[i].delta > 0 for i in task_indices) and selected_set.isdisjoint(task_indices):
                missed.append(max(task_indices, key=lambda i: predicted_delta[i]))
        lines.extend([f"### {MODEL_DISPLAY[model_name]}", ""])
        if harmful:
            index = harmful[0]
            example = examples[index]
            lines.append(
                f"- Harmful selection: `{example.task_id}` CP{example.checkpoint}, "
                f"predicted Delta `{predicted_delta[index]:+.3f}`, observed Delta `{example.delta:+d}`. "
                f"Teacher note: {example.teacher_explanation}"
            )
        else:
            lines.append("- No harmful selected intervention in the OOF policy.")
        if missed:
            index = missed[0]
            example = examples[index]
            lines.append(
                f"- Missed helpful task: `{example.task_id}` (best predicted CP{example.checkpoint} "
                f"Delta `{predicted_delta[index]:+.3f}`); at least one recorded checkpoint had Delta `+1`, "
                "but the controller skipped all three."
            )
        else:
            lines.append("- No task with a positive recorded intervention was fully skipped.")
        lines.append("")
    return lines


def build_report(
    *,
    run_id: str,
    examples: list[PairExample],
    predictor_rows: list[dict[str, Any]],
    policy_summary: list[dict[str, Any]],
    oof_results: dict[str, OOFResult],
    bootstrap_reps: int,
    seed: int,
) -> str:
    policy_by_name = {row["method"]: row for row in policy_summary}
    model_by_name = {row["model"]: row for row in predictor_rows}
    table = [
        "| Method | Accuracy | Gain | Teacher Calls | Total Cost | Helpful / Harmful / No Change | Skip all |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy_summary:
        table.append(
            f"| {row['method']} | {float(row['accuracy']) * 100:.2f}% "
            f"[{float(row['accuracy_ci95_low']) * 100:.2f}, {float(row['accuracy_ci95_high']) * 100:.2f}] | "
            f"{float(row['gain_vs_never_pp']):+.2f} pp | {float(row['teacher_calls']):.1f} | "
            f"${float(row['total_cost_usd']):.2f} / {float(row['teacher_tokens']):,.0f} tok | "
            f"{float(row['helpful']):.2f} / {float(row['harmful']):.2f} / {float(row['no_change']):.2f} | "
            f"{float(row['skip_all_rate']) * 100:.1f}% |"
        )

    predictor_table = [
        "| Predictor | Delta Pearson | Delta Spearman | Helpful AUROC | Helpful AP | Q-skip Brier | Q-verify Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in predictor_rows:
        predictor_table.append(
            f"| {row['model']} | {_format_float(row['delta_pearson'])} | "
            f"{_format_float(row['delta_spearman'])} | {_format_float(row['helpful_auroc'])} | "
            f"{_format_float(row['helpful_average_precision'])} | {_format_float(row['q_skip_brier'])} | "
            f"{_format_float(row['q_verify_brier'])} |"
        )

    fold_lines = []
    for model_name, result in oof_results.items():
        thresholds = ", ".join(
            f"F{record['fold']}={record['threshold']:+.3f}" for record in result.selections
        )
        fold_lines.append(f"- **{MODEL_DISPLAY[model_name]}:** {thresholds}")

    logistic = policy_by_name["Ours Logistic"]
    boosting = policy_by_name["Ours Boosting"]
    cp3 = policy_by_name["Always Verify CP3"]
    local = policy_by_name["Local Risk Controller"]
    oracle = policy_by_name["Offline Oracle"]
    best_ours = max([logistic, boosting], key=lambda row: float(row["accuracy"]))
    local_predictor = model_by_name["Local Risk heuristic"]
    logistic_predictor = model_by_name["Ours Logistic"]
    boosting_predictor = model_by_name["Ours Boosting"]

    return "\n".join(
        [
            "# Phase 2 — Downstream Verification Value Predictor",
            "",
            f"**Phase-1 run:** `{run_id}`  ",
            "**Data:** 139 HumanEval tasks, 417 pre-verification checkpoint examples  ",
            "**Evaluation:** nested 5-fold GroupKFold by task; 4-fold inner selection  ",
            f"**Uncertainty:** {bootstrap_reps:,} task-level paired bootstrap samples, seed `{seed}`  ",
            "**Execution:** offline only; no Student/Teacher inference and no HumanEval evaluator calls",
            "",
            "## Executive result",
            "",
            f"Best learned controller in this run is **{best_ours['method']}** at "
            f"**{float(best_ours['accuracy']) * 100:.2f}%** accuracy with "
            f"**{float(best_ours['average_teacher_calls_per_task']):.3f}** Teacher calls/task. "
            f"Always CP3 reaches **{float(cp3['accuracy']) * 100:.2f}%**, while the causal Local Risk "
            f"Controller reaches **{float(local['accuracy']) * 100:.2f}%**. Offline Oracle reaches "
            f"**{float(oracle['accuracy']) * 100:.2f}%** but is not deployable.",
            "",
            "This remains exploratory evaluation: Phase-1 outcomes were already inspected during H1 analysis, "
            "and no independent benchmark is used. All learned-policy decisions below are nevertheless strictly "
            "out-of-fold and thresholds were chosen only inside each outer training fold.",
            "",
            "## Data and leakage controls",
            "",
            "Features available before verification include task description, current artifact, preceding artifact "
            "metadata/content, checkpoint/role, static syntax/AST indicators, local heuristic risk, downstream "
            "stage count, and remaining budget. Task ID is retained only as a grouping key and output identifier.",
            "",
            "Explicitly excluded from inputs: `Y_skip`, `Y_verify`, observed Delta, Teacher correction/explanation, "
            "final implementation, future checkpoint artifacts, private tests, and task ID. Every outer and inner "
            "split asserts disjoint task groups. TF-IDF, scaling and SVD are fitted again inside each training fold.",
            "",
            "Models use two action-specific success heads: one estimates `Q_skip`, the other `Q_verify`; "
            "`Delta_pred = Q_verify - Q_skip`.",
            "",
            "## Predictor quality",
            "",
            *predictor_table,
            "",
            "Helpful prevalence is only 19/417, so AUROC can look reasonable even when precision is weak; "
            "Average Precision and realized policy outcomes are more decision-relevant. Correlations measure "
            "ranking alignment but do not by themselves validate a controller.",
            "",
            "## Offline policy evaluation",
            "",
            *table,
            "",
            "`Random Checkpoint` reports its exact 1/3 expectation, hence fractional counts. USD cost is zero "
            "because vLLM was self-hosted; token and latency columns preserve measurable verification usage. "
            "`Highest Risk (offline lookahead)` is diagnostic only because it inspects all three future risk scores.",
            "",
            "![Accuracy and policy budget](policy_comparison.png)",
            "",
            "![Accuracy versus Teacher tokens](accuracy_vs_cost.png)",
            "",
            "## Paired comparisons for learned controllers",
            "",
            f"- Ours Logistic vs Always CP3: **{float(logistic['difference_vs_cp3_pp']):+.2f} pp**, "
            f"95% CI [{float(logistic['difference_vs_cp3_ci95_low_pp']):+.2f}, "
            f"{float(logistic['difference_vs_cp3_ci95_high_pp']):+.2f}].",
            f"- Ours Boosting vs Always CP3: **{float(boosting['difference_vs_cp3_pp']):+.2f} pp**, "
            f"95% CI [{float(boosting['difference_vs_cp3_ci95_low_pp']):+.2f}, "
            f"{float(boosting['difference_vs_cp3_ci95_high_pp']):+.2f}].",
            f"- Ours Logistic vs Local Risk: **{float(logistic['difference_vs_local_risk_pp']):+.2f} pp**, "
            f"95% CI [{float(logistic['difference_vs_local_risk_ci95_low_pp']):+.2f}, "
            f"{float(logistic['difference_vs_local_risk_ci95_high_pp']):+.2f}].",
            f"- Ours Boosting vs Local Risk: **{float(boosting['difference_vs_local_risk_pp']):+.2f} pp**, "
            f"95% CI [{float(boosting['difference_vs_local_risk_ci95_low_pp']):+.2f}, "
            f"{float(boosting['difference_vs_local_risk_ci95_high_pp']):+.2f}].",
            "",
            "Intervals containing zero are not reported as statistically significant. With only 24 non-zero "
            "interventions, uncertainty is expected to be substantial.",
            "",
            "## Inner-validation thresholds",
            "",
            *fold_lines,
            "",
            "At runtime the controller checks CP1, then CP2, then CP3. It verifies the first checkpoint whose "
            "predicted Delta exceeds that fold's threshold, stops immediately after verification, and otherwise "
            "skips all checkpoints. It never ranks a full future trajectory.",
            "",
            "## Failure cases",
            "",
            *failure_case_lines(examples, oof_results),
            "## Research questions",
            "",
            f"**RQ1 — Better benefit prediction than Local Risk?** Logistic Delta Pearson is "
            f"`{float(logistic_predictor['delta_pearson']):.3f}` and Boosting is "
            f"`{float(boosting_predictor['delta_pearson']):.3f}`, versus Local Risk "
            f"`{float(local_predictor['delta_pearson']):.3f}`. Helpful AP is "
            f"`{float(logistic_predictor['helpful_average_precision']):.3f}` / "
            f"`{float(boosting_predictor['helpful_average_precision']):.3f}` versus "
            f"`{float(local_predictor['helpful_average_precision']):.3f}`. Interpret these jointly with the "
            "policy results; a single metric is insufficient.",
            "",
            f"**RQ2 — Better accuracy than Always CP3 and Local Risk?** The paired differences above show the "
            f"observed direction and uncertainty. Best Ours is {float(best_ours['accuracy']) * 100:.2f}%, "
            f"Always CP3 is {float(cp3['accuracy']) * 100:.2f}%, and Local Risk is "
            f"{float(local['accuracy']) * 100:.2f}%. No significance claim is made when a CI includes zero.",
            "",
            f"**RQ3 — Lower Teacher cost while maintaining accuracy?** Best Ours uses "
            f"{float(best_ours['teacher_calls']):.0f} calls and {float(best_ours['teacher_tokens']):,.0f} Teacher "
            f"tokens, versus CP3's {float(cp3['teacher_calls']):.0f} calls and "
            f"{float(cp3['teacher_tokens']):,.0f} tokens. Whether accuracy is maintained is determined by the "
            "paired CI rather than the point estimate alone.",
            "",
            "**RQ4 — Enough signal?** This dataset is sufficient for a feasibility test but not for a stable "
            "general claim: only 19 helpful and 5 harmful pairs exist, all from one benchmark and one model pair. "
            "A larger multi-benchmark, repeated-intervention dataset is required before Phase 3 or deployment.",
            "",
            "## Limitations",
            "",
            "- This is exploratory cross-validation on a dataset already examined in Phase 1, not external validation.",
            "- Each counterfactual pair has one repetition; residual generation nondeterminism is not estimated.",
            "- Two independently fitted probability heads may be poorly calibrated in a small sample.",
            "- Offline paired outcomes support evaluation only at the selected checkpoint under the frozen workflow.",
            "- Serialized joblib checkpoints must only be loaded from this trusted project directory.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "conda activate agent_verify",
            "python results/phase2/train_predictor.py --output-dir results/phase2",
            "```",
            "",
            "See `run_manifest.json` for versions/settings and `fold_selections.json` for every inner-selected "
            "hyperparameter/threshold. OOF metrics never use the final all-data models.",
            "",
        ]
    )


def write_reproduction_files(output_dir: Path, args: argparse.Namespace) -> None:
    wrapper = '''#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent_verify.analysis.phase2 import main

if __name__ == "__main__":
    main()
'''
    (output_dir / "train_predictor.py").write_text(wrapper, encoding="utf-8")
    command = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")/../..\"\n"
        "conda run -n agent_verify python results/phase2/train_predictor.py "
        f"--run-id {args.run_id} --output-dir results/phase2\n"
    )
    reproduce = output_dir / "reproduce.sh"
    reproduce.write_text(command, encoding="utf-8")
    reproduce.chmod(0o755)


def main() -> None:
    args = parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2 or args.bootstrap_reps < 1:
        raise ValueError("Fold counts must be >=2 and bootstrap repetitions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_phase1_examples(
        results_dir=args.results_dir,
        dataset_path=args.dataset,
        run_id=args.run_id,
    )

    oof_results: dict[str, OOFResult] = {}
    for model_offset, model_name in enumerate(
        ["metadata_logistic", "semantic_logistic", "semantic_boosting"]
    ):
        print(f"Training nested OOF predictor: {MODEL_DISPLAY[model_name]}", flush=True)
        oof_results[model_name] = train_nested_oof(
            examples,
            model_name=model_name,
            output_dir=args.output_dir,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            seed=args.seed + model_offset * 1_000_003,
        )

    predictor_rows = predictor_metric_rows(examples, oof_results)
    baseline, policies = build_policies(examples, oof_results)
    bootstrap = bootstrap_policy_metrics(
        baseline,
        policies,
        repetitions=args.bootstrap_reps,
        seed=args.seed + 4_000_003,
    )
    summaries = policy_rows(baseline, policies, bootstrap)
    write_csv(args.output_dir / "predictor_metrics.csv", predictor_rows)
    write_csv(args.output_dir / "policy_comparison.csv", summaries)
    write_oof_predictions(args.output_dir / "out_of_fold_predictions.csv", examples, oof_results)
    plot_accuracy_vs_cost(args.output_dir / "accuracy_vs_cost.png", summaries)
    plot_policy_comparison(args.output_dir / "policy_comparison.png", summaries)

    report = build_report(
        run_id=args.run_id,
        examples=examples,
        predictor_rows=predictor_rows,
        policy_summary=summaries,
        oof_results=oof_results,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )
    (args.output_dir / "phase2_report.md").write_text(report, encoding="utf-8")
    fold_selections = {
        MODEL_DISPLAY[name]: result.selections for name, result in oof_results.items()
    }
    (args.output_dir / "fold_selections.json").write_text(
        json.dumps(fold_selections, indent=2) + "\n", encoding="utf-8"
    )
    write_reproduction_files(args.output_dir, args)

    manifest = {
        "analysis": "phase2_nested_groupkfold_offline_policy_evaluation",
        "run_id": args.run_id,
        "tasks": len(task_order(examples)),
        "pairs": len(examples),
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "bootstrap_repetitions": args.bootstrap_reps,
        "seed": args.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "llm_inference_performed": False,
        "teacher_calls_performed": False,
        "evaluator_calls_performed": False,
        "training_labels": ["y_skip", "y_verify"],
        "forbidden_model_inputs": [
            "task_id",
            "y_skip",
            "y_verify",
            "delta",
            "teacher_corrected_artifact",
            "future_outputs",
            "private_tests",
        ],
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Phase 2 artifacts written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
