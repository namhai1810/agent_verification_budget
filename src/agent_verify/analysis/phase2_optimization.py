"""Exploratory optimization of the Phase-2 boosting controller.

This experiment keeps task-grouped nested CV but expands the boosting search
space and selects a monotone checkpoint-specific threshold profile inside each
outer training fold. It never calls an LLM, Teacher, or evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402

from agent_verify.analysis.phase2_data import PairExample, load_phase1_examples
from agent_verify.analysis.phase2_models import (  # noqa: E402
    TwoHeadPredictor,
    group_folds,
    simulate_sequential_policy,
)

DEFAULT_RUN_ID = "h1_phase1_qwen25_gpu2-09092ebd5e-d03859f54f"
THRESHOLD_LEVELS = (-0.05, 0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)


@dataclass(frozen=True)
class FoldSelection:
    fold: int
    params: dict[str, Any]
    threshold_cp1: float
    threshold_cp2: float
    threshold_cp3: float
    inner_accuracy: float
    inner_calls_per_task: float
    inner_brier: float
    train_tasks: int
    test_tasks: int


@dataclass
class Policy:
    name: str
    outcomes: np.ndarray
    selected_indices: np.ndarray
    definition: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dataset", type=Path, default=Path("data/HumanEval.jsonl.gz"))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--phase2-dir", type=Path, default=Path("results/phase2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2_optimization"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_261_024)
    return parser.parse_args()


def boosting_grid() -> list[dict[str, Any]]:
    return [
        {
            "max_leaf_nodes": leaf_nodes,
            "l2_regularization": regularization,
            "svd_components": components,
        }
        for components in (8, 16, 24, 32, 48)
        for leaf_nodes in (3, 7, 15)
        for regularization in (1.0, 5.0, 10.0)
    ]


def monotone_threshold_profiles() -> list[tuple[float, float, float]]:
    """Earlier checkpoints require at least as much predicted gain as later ones."""

    return [
        values
        for values in product(THRESHOLD_LEVELS, repeat=3)
        if values[0] >= values[1] >= values[2]
    ]


def _inner_predictions(
    examples: list[PairExample],
    params: dict[str, Any],
    *,
    inner_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    q_skip = np.empty(len(examples), dtype=float)
    q_verify = np.empty(len(examples), dtype=float)
    for fold, (train_indices, validation_indices) in enumerate(group_folds(examples, inner_folds)):
        train = [examples[int(index)] for index in train_indices]
        validation = [examples[int(index)] for index in validation_indices]
        predictor = TwoHeadPredictor(
            "semantic_boosting",
            params,
            seed + fold * 101,
        ).fit(train)
        skip_prediction, verify_prediction = predictor.predict(validation)
        q_skip[validation_indices] = skip_prediction
        q_verify[validation_indices] = verify_prediction
    return q_skip, q_verify


def select_configuration(
    examples: list[PairExample],
    *,
    inner_folds: int,
    seed: int,
) -> FoldSelection:
    y_skip = np.asarray([example.y_skip for example in examples], dtype=float)
    y_verify = np.asarray([example.y_verify for example in examples], dtype=float)
    best_key: tuple[Any, ...] | None = None
    best: FoldSelection | None = None
    profiles = monotone_threshold_profiles()
    for candidate_index, params in enumerate(boosting_grid()):
        q_skip, q_verify = _inner_predictions(
            examples,
            params,
            inner_folds=inner_folds,
            seed=seed + candidate_index * 1_003,
        )
        brier = float(
            np.mean(np.concatenate([(q_skip - y_skip) ** 2, (q_verify - y_verify) ** 2]))
        )
        predicted_delta = q_verify - q_skip
        for profile in profiles:
            thresholds = np.asarray([profile[example.checkpoint - 1] for example in examples])
            outcomes, selected, _ = simulate_sequential_policy(
                examples,
                predicted_delta,
                thresholds,
            )
            accuracy = float(outcomes.mean())
            calls = float(np.mean(selected >= 0))
            key = (accuracy, -calls, -brier, profile)
            if best_key is None or key > best_key:
                best_key = key
                best = FoldSelection(
                    fold=-1,
                    params=dict(params),
                    threshold_cp1=profile[0],
                    threshold_cp2=profile[1],
                    threshold_cp3=profile[2],
                    inner_accuracy=accuracy,
                    inner_calls_per_task=calls,
                    inner_brier=brier,
                    train_tasks=0,
                    test_tasks=0,
                )
    assert best is not None
    return best


def nested_oof(
    examples: list[PairExample],
    *,
    output_dir: Path,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[FoldSelection]]:
    q_skip = np.empty(len(examples), dtype=float)
    q_verify = np.empty(len(examples), dtype=float)
    fold_ids = np.empty(len(examples), dtype=int)
    thresholds = np.empty(len(examples), dtype=float)
    selections: list[FoldSelection] = []
    checkpoint_dir = output_dir / "model_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_indices, test_indices) in enumerate(group_folds(examples, outer_folds)):
        train = [examples[int(index)] for index in train_indices]
        test = [examples[int(index)] for index in test_indices]
        selected = select_configuration(
            train,
            inner_folds=inner_folds,
            seed=seed + fold * 10_007,
        )
        selected = FoldSelection(
            fold=fold,
            params=selected.params,
            threshold_cp1=selected.threshold_cp1,
            threshold_cp2=selected.threshold_cp2,
            threshold_cp3=selected.threshold_cp3,
            inner_accuracy=selected.inner_accuracy,
            inner_calls_per_task=selected.inner_calls_per_task,
            inner_brier=selected.inner_brier,
            train_tasks=len({example.task_id for example in train}),
            test_tasks=len({example.task_id for example in test}),
        )
        predictor = TwoHeadPredictor(
            "semantic_boosting",
            selected.params,
            seed + fold * 100_003,
        ).fit(train)
        skip_prediction, verify_prediction = predictor.predict(test)
        q_skip[test_indices] = skip_prediction
        q_verify[test_indices] = verify_prediction
        fold_ids[test_indices] = fold
        profile = (selected.threshold_cp1, selected.threshold_cp2, selected.threshold_cp3)
        for index in test_indices:
            thresholds[index] = profile[examples[int(index)].checkpoint - 1]
        selections.append(selected)
        joblib.dump(predictor, checkpoint_dir / f"fold_{fold}.joblib", compress=3)
        (checkpoint_dir / f"fold_{fold}.json").write_text(
            json.dumps(asdict(selected), indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"Fold {fold}: inner_acc={selected.inner_accuracy:.4f}, "
            f"thresholds={profile}, params={selected.params}",
            flush=True,
        )
    return q_skip, q_verify, fold_ids, thresholds, selections


def task_ids(examples: list[PairExample]) -> list[str]:
    return list(dict.fromkeys(example.task_id for example in examples))


def group_indices(examples: list[PairExample]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(example.task_id, []).append(index)
    return grouped


def policy_from_selected(
    name: str,
    examples: list[PairExample],
    selected_indices: np.ndarray,
    definition: str,
) -> Policy:
    tasks = task_ids(examples)
    grouped = group_indices(examples)
    outcomes = []
    for task_id, selected_index in zip(tasks, selected_indices, strict=True):
        if selected_index < 0:
            outcomes.append(examples[grouped[task_id][0]].y_skip)
        else:
            if examples[int(selected_index)].task_id != task_id:
                raise AssertionError("Selected intervention belongs to another task")
            outcomes.append(examples[int(selected_index)].y_verify)
    return Policy(name, np.asarray(outcomes, dtype=float), selected_indices, definition)


def load_original_selected(
    examples: list[PairExample], phase2_dir: Path, model_name: str
) -> np.ndarray:
    with (phase2_dir / "out_of_fold_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["model"] == model_name]
    selected_by_task = {
        row["task_id"]: int(row["checkpoint"])
        for row in rows
        if int(row["selected_by_controller"]) == 1
    }
    grouped = group_indices(examples)
    return np.asarray(
        [
            next(
                (
                    index
                    for index in grouped[task_id]
                    if examples[index].checkpoint == selected_by_task[task_id]
                ),
                -1,
            )
            if task_id in selected_by_task
            else -1
            for task_id in task_ids(examples)
        ],
        dtype=int,
    )


def build_comparison_policies(
    examples: list[PairExample],
    optimized_selected: np.ndarray,
    phase2_dir: Path,
) -> tuple[np.ndarray, list[Policy]]:
    tasks = task_ids(examples)
    grouped = group_indices(examples)
    baseline = np.asarray([examples[grouped[task_id][0]].y_skip for task_id in tasks], dtype=float)
    policies = [
        policy_from_selected(
            "Never Verify",
            examples,
            np.full(len(tasks), -1, dtype=int),
            "Skip all checkpoints.",
        )
    ]
    local_selected = []
    cp3_selected = []
    oracle_selected = []
    for task_id in tasks:
        indices = sorted(grouped[task_id], key=lambda index: examples[index].checkpoint)
        local_selected.append(next((index for index in indices if examples[index].local_risk >= 0.5), -1))
        cp3_selected.append(next(index for index in indices if examples[index].checkpoint == 3))
        oracle_selected.append(next((index for index in indices if examples[index].delta > 0), -1))
    policies.extend(
        [
            policy_from_selected(
                "Local Risk Controller",
                examples,
                np.asarray(local_selected),
                "Verify first checkpoint with local risk >= 0.50.",
            ),
            policy_from_selected(
                "Original Ours Logistic",
                examples,
                load_original_selected(examples, phase2_dir, "Ours Logistic"),
                "Original Phase-2 semantic logistic OOF controller.",
            ),
            policy_from_selected(
                "Original Ours Boosting",
                examples,
                load_original_selected(examples, phase2_dir, "Ours Boosting"),
                "Original Phase-2 boosting OOF controller.",
            ),
            policy_from_selected(
                "Optimized Boosting",
                examples,
                optimized_selected,
                "Expanded inner search plus monotone CP-specific thresholds.",
            ),
            policy_from_selected(
                "Always Verify CP3",
                examples,
                np.asarray(cp3_selected),
                "Verify CP3 for every task.",
            ),
            policy_from_selected(
                "Offline Oracle",
                examples,
                np.asarray(oracle_selected),
                "First checkpoint with observed positive Delta; analysis upper bound.",
            ),
        ]
    )
    return baseline, policies


def percentile(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def policy_summary_rows(
    examples: list[PairExample],
    baseline: np.ndarray,
    policies: list[Policy],
    *,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(baseline), size=(repetitions, len(baseline)))
    by_name = {policy.name: policy for policy in policies}
    references = {
        "vs_cp3": by_name["Always Verify CP3"].outcomes,
        "vs_original_boosting": by_name["Original Ours Boosting"].outcomes,
        "vs_original_logistic": by_name["Original Ours Logistic"].outcomes,
        "vs_local_risk": by_name["Local Risk Controller"].outcomes,
    }
    rows = []
    for policy in policies:
        outcomes = policy.outcomes
        accuracy_interval = percentile(outcomes[samples].mean(axis=1))
        selected = [int(index) for index in policy.selected_indices if index >= 0]
        deltas = [examples[index].delta for index in selected]
        row: dict[str, Any] = {
            "method": policy.name,
            "accuracy": float(outcomes.mean()),
            "accuracy_ci95_low": accuracy_interval[0],
            "accuracy_ci95_high": accuracy_interval[1],
            "gain_vs_never_pp": float(np.mean(outcomes - baseline) * 100),
            "teacher_calls": len(selected),
            "average_calls_per_task": len(selected) / len(baseline),
            "teacher_tokens": sum(examples[index].teacher_tokens for index in selected),
            "teacher_latency_seconds": sum(examples[index].teacher_latency_seconds for index in selected),
            "total_cost_usd": sum(examples[index].teacher_cost_usd for index in selected),
            "helpful": sum(delta > 0 for delta in deltas),
            "harmful": sum(delta < 0 for delta in deltas),
            "no_change": sum(delta == 0 for delta in deltas),
            "skip_all_rate": 1 - len(selected) / len(baseline),
            "definition": policy.definition,
        }
        for name, reference in references.items():
            differences = outcomes - reference
            interval = percentile(differences[samples].mean(axis=1))
            row[f"difference_{name}_pp"] = float(differences.mean() * 100)
            row[f"difference_{name}_ci95_low_pp"] = interval[0] * 100
            row[f"difference_{name}_ci95_high_pp"] = interval[1] * 100
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_oof(
    path: Path,
    examples: list[PairExample],
    q_skip: np.ndarray,
    q_verify: np.ndarray,
    folds: np.ndarray,
    thresholds: np.ndarray,
    selected_indices: np.ndarray,
) -> None:
    selected_set = {int(index) for index in selected_indices if index >= 0}
    rows = []
    for index, example in enumerate(examples):
        rows.append(
            {
                "task_id": example.task_id,
                "checkpoint": example.checkpoint,
                "role": example.role,
                "outer_fold": int(folds[index]),
                "q_skip_pred": float(q_skip[index]),
                "q_verify_pred": float(q_verify[index]),
                "delta_pred": float(q_verify[index] - q_skip[index]),
                "checkpoint_threshold": float(thresholds[index]),
                "selected_by_controller": int(index in selected_set),
                "y_skip": example.y_skip,
                "y_verify": example.y_verify,
                "delta_observed": example.delta,
                "local_risk": example.local_risk,
            }
        )
    write_csv(path, rows)


def plot_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    names = [str(row["method"]) for row in rows]
    accuracy = np.asarray([float(row["accuracy"]) * 100 for row in rows])
    low = np.asarray([float(row["accuracy_ci95_low"]) * 100 for row in rows])
    high = np.asarray([float(row["accuracy_ci95_high"]) * 100 for row in rows])
    calls = np.asarray([float(row["teacher_calls"]) for row in rows])
    colors = ["#94A3B8", "#F59E0B", "#2563EB", "#0891B2", "#10B981", "#7C3AED", "#EF4444"]
    y = np.arange(len(names))
    fig, (left, right) = plt.subplots(1, 2, figsize=(14.5, 7.2), gridspec_kw={"width_ratios": [1.5, 1]})
    left.barh(y, accuracy, color=colors)
    left.errorbar(
        accuracy,
        y,
        xerr=np.vstack([accuracy - low, high - accuracy]),
        fmt="none",
        ecolor="#0F172A",
        capsize=3,
    )
    left.set_yticks(y, names)
    left.invert_yaxis()
    left.set_xlabel("Final accuracy (%)")
    left.set_title("Nested OOF accuracy with task-bootstrap 95% CI")
    left.grid(axis="x", alpha=0.22)
    right.barh(y, calls, color=colors)
    right.set_yticks(y, [""] * len(names))
    right.invert_yaxis()
    right.set_xlabel("Teacher calls / 139 tasks")
    right.set_title("Realized verification usage")
    right.grid(axis="x", alpha=0.22)
    fig.suptitle("Phase 2 exploratory model optimization", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build_report(
    rows: list[dict[str, Any]],
    selections: list[FoldSelection],
    delta_correlation: float,
    bootstrap_reps: int,
) -> str:
    by_name = {row["method"]: row for row in rows}
    optimized = by_name["Optimized Boosting"]
    cp3 = by_name["Always Verify CP3"]
    table = [
        "| Method | Accuracy | Gain | Calls | Tokens | Helpful / Harmful |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['method']} | {float(row['accuracy']) * 100:.2f}% "
            f"[{float(row['accuracy_ci95_low']) * 100:.2f}, {float(row['accuracy_ci95_high']) * 100:.2f}] | "
            f"{float(row['gain_vs_never_pp']):+.2f} pp | {int(row['teacher_calls'])} | "
            f"{float(row['teacher_tokens']):,.0f} | {int(row['helpful'])} / {int(row['harmful'])} |"
        )
    fold_lines = [
        f"- Fold {selection.fold}: thresholds "
        f"({selection.threshold_cp1:+.3f}, {selection.threshold_cp2:+.3f}, {selection.threshold_cp3:+.3f}); "
        f"SVD={selection.params['svd_components']}, leaves={selection.params['max_leaf_nodes']}, "
        f"L2={selection.params['l2_regularization']}."
        for selection in selections
    ]
    return "\n".join(
        [
            "# Phase 2 — Exploratory Model Optimization",
            "",
            "This experiment uses the same 139 tasks as Phase 2. It is an adaptive exploratory optimization, "
            "not a new independent test set. All candidate/threshold selection remains inside inner "
            "task-grouped folds.",
            "",
            "## What changed",
            "",
            "1. Expanded Boosting search from 4 to 45 configurations: SVD dimensions 8/16/24/32/48, "
            "tree sizes 3/7/15 leaves, and L2 regularization 1/5/10.",
            "2. Replaced one global threshold with `(threshold_CP1, threshold_CP2, threshold_CP3)`.",
            "3. Constrained thresholds to be monotone: CP1 ≥ CP2 ≥ CP3, requiring stronger evidence "
            "to spend verification early and allowing the controller to wait for CP3.",
            "4. Hyperparameters and thresholds are selected only by 4-fold inner GroupKFold within "
            "each outer fold.",
            "",
            "## Result",
            "",
            *table,
            "",
            f"Optimized Boosting reaches **{float(optimized['accuracy']) * 100:.2f}%**, improving "
            "over original "
            f"Boosting by **{float(optimized['difference_vs_original_boosting_pp']):+.2f} pp** and original "
            f"Logistic by **{float(optimized['difference_vs_original_logistic_pp']):+.2f} pp**. It selects "
            f"{int(optimized['teacher_calls'])} calls with {int(optimized['helpful'])} helpful and "
            f"{int(optimized['harmful'])} harmful interventions.",
            "",
            f"Against Always CP3 it remains **{float(optimized['difference_vs_cp3_pp']):+.2f} pp** "
            f"(95% CI [{float(optimized['difference_vs_cp3_ci95_low_pp']):+.2f}, "
            f"{float(optimized['difference_vs_cp3_ci95_high_pp']):+.2f}]). It saves "
            f"{int(cp3['teacher_calls']) - int(optimized['teacher_calls'])} calls and "
            f"{float(cp3['teacher_tokens']) - float(optimized['teacher_tokens']):,.0f} Teacher tokens.",
            "",
            f"Against Local Risk it improves accuracy by "
            f"**{float(optimized['difference_vs_local_risk_pp']):+.2f} pp** "
            f"(95% CI [{float(optimized['difference_vs_local_risk_ci95_low_pp']):+.2f}, "
            f"{float(optimized['difference_vs_local_risk_ci95_high_pp']):+.2f}]).",
            "",
            f"OOF predicted-Delta Pearson correlation is `{delta_correlation:.3f}`: better than "
            "original Boosting but still below Local Risk's `0.223`. The gain comes partly from "
            "stage-aware thresholding, not a "
            "strong general benefit predictor.",
            "",
            "![Optimization comparison](optimization_comparison.png)",
            "",
            "## Inner-selected fold configurations",
            "",
            *fold_lines,
            "",
            "The selected configurations vary substantially across folds. This is evidence of high variance, "
            "consistent with only 19 helpful and 5 harmful pairs.",
            "",
            "## Scientific interpretation",
            "",
            "- **There is a real point-estimate improvement over the original learned models.**",
            "- **There is still no evidence that the optimized model beats Always CP3.**",
            "- Because the same dataset motivated this optimization, these numbers are not "
            "independent validation.",
            "- Freeze this specification and evaluate it on new tasks/benchmarks before further tuning.",
            "",
            f"Bootstrap used {bootstrap_reps:,} task-level paired resamples. No LLM, Teacher or "
            "evaluator call was made.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "conda activate agent_verify",
            "python results/phase2_optimization/train_optimized.py",
            "```",
            "",
        ]
    )


def write_runner(output_dir: Path) -> None:
    runner = '''#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent_verify.analysis.phase2_optimization import main

if __name__ == "__main__":
    main()
'''
    (output_dir / "train_optimized.py").write_text(runner, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_phase1_examples(
        results_dir=args.results_dir,
        dataset_path=args.dataset,
        run_id=args.run_id,
    )
    q_skip, q_verify, folds, thresholds, selections = nested_oof(
        examples,
        output_dir=args.output_dir,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
    )
    predicted_delta = q_verify - q_skip
    _, optimized_selected, _ = simulate_sequential_policy(examples, predicted_delta, thresholds)
    baseline, policies = build_comparison_policies(examples, optimized_selected, args.phase2_dir)
    rows = policy_summary_rows(
        examples,
        baseline,
        policies,
        repetitions=args.bootstrap_reps,
        seed=args.seed + 4_000_003,
    )
    write_csv(args.output_dir / "optimized_policy_comparison.csv", rows)
    write_oof(
        args.output_dir / "optimized_oof_predictions.csv",
        examples,
        q_skip,
        q_verify,
        folds,
        thresholds,
        optimized_selected,
    )
    (args.output_dir / "fold_selections.json").write_text(
        json.dumps([asdict(selection) for selection in selections], indent=2) + "\n",
        encoding="utf-8",
    )
    plot_comparison(args.output_dir / "optimization_comparison.png", rows)
    observed_delta = np.asarray([example.delta for example in examples], dtype=float)
    delta_correlation = float(np.corrcoef(predicted_delta, observed_delta)[0, 1])
    (args.output_dir / "optimization_report.md").write_text(
        build_report(rows, selections, delta_correlation, args.bootstrap_reps),
        encoding="utf-8",
    )
    write_runner(args.output_dir)
    manifest = {
        "experiment": "phase2_exploratory_optimization",
        "run_id": args.run_id,
        "tasks": len(task_ids(examples)),
        "pairs": len(examples),
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "boosting_configurations": len(boosting_grid()),
        "threshold_profiles": len(monotone_threshold_profiles()),
        "threshold_constraint": "CP1 >= CP2 >= CP3",
        "bootstrap_repetitions": args.bootstrap_reps,
        "seed": args.seed,
        "adaptive_exploratory_optimization": True,
        "independent_test_set": False,
        "llm_inference_performed": False,
        "teacher_calls_performed": False,
        "evaluator_calls_performed": False,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Optimized Phase-2 artifacts written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

