"""Train Phase-2 predictors with a frozen benchmark-stratified outer test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from agent_verify.analysis.phase2 import (
    OOFResult,
    bootstrap_policy_metrics,
    build_policies,
    policy_rows,
    predictor_metric_rows,
    write_csv,
    write_oof_predictions,
)
from agent_verify.analysis.phase2_data import (
    CHECKPOINT_ROLES,
    PairExample,
    build_observable_features,
    load_jsonl,
    load_phase1_examples,
)
from agent_verify.analysis.phase2_models import (
    TwoHeadPredictor,
    select_hyperparameters,
    simulate_sequential_policy,
)
from agent_verify.analysis.phase2_optimization import (
    boosting_grid,
    monotone_threshold_profiles,
    select_configuration,
)

MODEL_NAMES = ("metadata_logistic", "semantic_logistic", "semantic_boosting")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["livecodebench", "combined"], default="livecodebench")
    parser.add_argument("--livecodebench-results", type=Path, default=Path("results/livecodebench"))
    parser.add_argument(
        "--livecodebench-extra-results",
        type=Path,
        action="append",
        default=[],
        help="Additional LiveCodeBench result directory; incomplete task triples are excluded.",
    )
    parser.add_argument("--humaneval-results", type=Path, default=Path("results"))
    parser.add_argument("--humaneval-dataset", type=Path, default=Path("data/HumanEval.jsonl.gz"))
    parser.add_argument("--humaneval-run-id")
    parser.add_argument("--output-dir", type=Path, default=Path("results/livecodebench/phase2"))
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _latest_run_id(results_dir: Path) -> str:
    rows = load_jsonl(results_dir / "manifests.jsonl")
    if not rows:
        raise ValueError(f"No run manifest in {results_dir}")
    return str(rows[-1]["run_id"])


def load_livecodebench_examples(
    results_dir: Path,
    run_id: str,
    *,
    drop_incomplete_tasks: bool = False,
) -> tuple[list[PairExample], dict[str, dict], dict[str, int]]:
    selected = json.loads((results_dir / "selected_tasks.json").read_text(encoding="utf-8"))["tasks"]
    tasks = {str(row["task_id"]): row for row in selected}
    pairs = [
        row
        for row in load_jsonl(results_dir / "intervention_pairs.jsonl")
        if row.get("run_id") == run_id and row.get("status") == "completed"
    ]
    baselines = [
        row
        for row in load_jsonl(results_dir / "baseline.jsonl")
        if row.get("run_id") == run_id and row.get("status") == "completed"
    ]
    invocations = {
        str(row["invocation_id"]): row for row in load_jsonl(results_dir / "llm_invocations.jsonl")
    }
    states = {
        (str(baseline["task_id"]), int(state["checkpoint_index"])): state
        for baseline in baselines
        for state in baseline["checkpoint_states"]
    }
    counts: dict[str, int] = {}
    examples: list[PairExample] = []
    names: tuple[str, ...] | None = None
    for pair in pairs:
        task_id = str(pair["task_id"])
        checkpoint = int(pair["checkpoint"])
        if task_id not in tasks or (task_id, checkpoint) not in states:
            raise ValueError(f"Missing public task/state for {task_id} CP{checkpoint}")
        counts[task_id] = counts.get(task_id, 0) + 1
        task = tasks[task_id]
        state = states[(task_id, checkpoint)]
        text, metadata, current_names = build_observable_features(
            prompt=str(task["prompt"]),
            entry_point=str(task.get("entry_point", "")),
            checkpoint=checkpoint,
            role=CHECKPOINT_ROLES[checkpoint],
            current_artifact=dict(pair["original_artifact"]),
            preceding_artifacts=list(state.get("preceding_artifacts", [])),
            local_risk=float(pair["local_risk_score"]),
        )
        if names is None:
            names = current_names
        elif names != current_names:
            raise AssertionError("Inconsistent metadata schema")
        invocation = invocations.get(str(pair.get("teacher_invocation_id")), {})
        examples.append(
            PairExample(
                task_id=task_id,
                checkpoint=checkpoint,
                role=CHECKPOINT_ROLES[checkpoint],
                text=text,
                metadata=metadata,
                metadata_names=current_names,
                y_skip=int(pair["y_skip"]),
                y_verify=int(pair["y_verify"]),
                delta=int(pair["delta"]),
                local_risk=float(pair["local_risk_score"]),
                teacher_changed=bool(pair.get("teacher_changed", False)),
                teacher_tokens=int(invocation.get("input_tokens", 0) or 0)
                + int(invocation.get("output_tokens", 0) or 0),
                teacher_latency_seconds=float(pair.get("teacher_latency_seconds", 0.0) or 0.0),
                teacher_cost_usd=float(pair.get("teacher_cost_usd", 0.0) or 0.0),
                original_artifact=dict(pair["original_artifact"]),
                teacher_explanation=str(pair.get("teacher_explanation", "")),
            )
        )
    incomplete = {task_id: count for task_id, count in counts.items() if count != 3}
    if incomplete and not drop_incomplete_tasks:
        raise ValueError(f"Incomplete checkpoint triples: {incomplete}")
    if incomplete:
        examples = [row for row in examples if row.task_id not in incomplete]
    complete_task_ids = {row.task_id for row in examples}
    active_tasks = {task_id: tasks[task_id] for task_id in complete_task_ids}
    examples.sort(key=lambda row: (row.task_id, row.checkpoint))
    return examples, active_tasks, incomplete


def deterministic_holdout(
    task_metadata: dict[str, dict[str, Any]],
    *,
    fraction: float,
    seed: int,
) -> list[str]:
    """Stratify by benchmark/difficulty using only IDs and pre-treatment metadata."""

    if not 0 < fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    groups: dict[tuple[str, str], list[str]] = {}
    for task_id, row in task_metadata.items():
        key = (str(row.get("benchmark", "unknown")), str(row.get("difficulty", "unknown")))
        groups.setdefault(key, []).append(task_id)
    selected: list[str] = []
    for _key, task_ids in sorted(groups.items()):
        ordered = sorted(
            task_ids,
            key=lambda task_id: hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(),
        )
        count = max(1, round(len(ordered) * fraction)) if len(ordered) > 1 else 0
        selected.extend(ordered[:count])
    return sorted(selected)


def load_or_create_holdout(
    path: Path,
    task_metadata: dict[str, dict[str, Any]],
    *,
    fraction: float,
    seed: int,
) -> list[str]:
    universe_rows = [
        {
            "task_id": task_id,
            "benchmark": str(row.get("benchmark", "unknown")),
            "difficulty": str(row.get("difficulty", "unknown")),
        }
        for task_id, row in sorted(task_metadata.items())
    ]
    universe_sha256 = hashlib.sha256(
        json.dumps(universe_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("task_universe_sha256") != universe_sha256:
            raise ValueError(
                "Frozen test split belongs to a different task universe; "
                "use a separate --output-dir for each training mode/dataset."
            )
        task_ids = list(payload["task_ids"])
        missing = sorted(set(task_ids).difference(task_metadata))
        if missing:
            raise ValueError(f"Frozen test split references missing tasks: {missing[:5]}")
        return task_ids
    task_ids = deterministic_holdout(task_metadata, fraction=fraction, seed=seed)
    payload = {
        "schema_version": 1,
        "seed": seed,
        "test_fraction": fraction,
        "task_universe_sha256": universe_sha256,
        "task_universe_size": len(universe_rows),
        "task_ids": task_ids,
        "selection_rule": "SHA256(seed:task_id), stratified by benchmark and difficulty; no outcomes used",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return task_ids


def _test_result(
    examples: list[PairExample],
    model_name: str,
    predictor: TwoHeadPredictor,
    threshold: float,
) -> OOFResult:
    thresholds = np.full(len(examples), threshold, dtype=float)
    return _test_result_with_thresholds(examples, model_name, predictor, thresholds)


def _test_result_with_thresholds(
    examples: list[PairExample],
    model_name: str,
    predictor: TwoHeadPredictor,
    thresholds: np.ndarray,
) -> OOFResult:
    q_skip, q_verify = predictor.predict(examples)
    _, selected, _ = simulate_sequential_policy(examples, q_verify - q_skip, thresholds)
    selected_mask = np.zeros(len(examples), dtype=int)
    for index in selected:
        if index >= 0:
            selected_mask[index] = 1
    return OOFResult(
        model_name=model_name,
        q_skip=q_skip,
        q_verify=q_verify,
        fold=np.full(len(examples), -1, dtype=int),
        threshold=np.asarray(thresholds, dtype=float),
        selected=selected_mask,
        selections=[],
    )


def _policy_bundle(
    examples: list[PairExample],
    standard_results: dict[str, OOFResult],
    optimized_result: OOFResult,
):
    baseline, policies = build_policies(examples, standard_results)
    optimized_inputs = {**standard_results, "semantic_boosting": optimized_result}
    _, optimized_policies = build_policies(examples, optimized_inputs)
    optimized = next(policy for policy in optimized_policies if policy.name == "Ours Boosting")
    optimized.name = "Optimized Boosting"
    optimized.definition = (
        "Expanded 45-configuration search with monotone checkpoint-specific thresholds; "
        "selected using training-only GroupKFold."
    )
    policies.append(optimized)
    return baseline, policies


def _write_split_artifacts(
    output_dir: Path,
    examples: list[PairExample],
    metadata: dict[str, dict[str, Any]],
    holdout: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_ids = sorted({row.task_id for row in examples})
    counts = {task_id: sum(row.task_id == task_id for row in examples) for task_id in task_ids}
    if set(counts.values()) != {3}:
        raise ValueError(f"Every included task must have CP1/CP2/CP3, got {set(counts.values())}")
    assignments = [
        {
            "task_id": task_id,
            "benchmark": str(metadata[task_id]["benchmark"]),
            "difficulty": str(metadata[task_id].get("difficulty", "unknown")),
            "split": "test" if task_id in holdout else "train",
        }
        for task_id in task_ids
    ]
    write_csv(output_dir / "split_assignments.csv", assignments)

    summary: list[dict[str, Any]] = []
    for split in ("all", "train", "test"):
        split_ids = {
            row["task_id"]
            for row in assignments
            if split == "all" or row["split"] == split
        }
        for benchmark in ("all", "humaneval", "livecodebench"):
            benchmark_ids = {
                task_id
                for task_id in split_ids
                if benchmark == "all" or metadata[task_id]["benchmark"] == benchmark
            }
            rows = [row for row in examples if row.task_id in benchmark_ids]
            if not rows:
                continue
            summary.append(
                {
                    "split": split,
                    "benchmark": benchmark,
                    "tasks": len(benchmark_ids),
                    "checkpoint_rows": len(rows),
                    "helpful": sum(row.delta > 0 for row in rows),
                    "harmful": sum(row.delta < 0 for row in rows),
                    "no_change": sum(row.delta == 0 for row in rows),
                    "baseline_accuracy": sum(
                        next(row.y_skip for row in rows if row.task_id == task_id)
                        for task_id in benchmark_ids
                    )
                    / len(benchmark_ids),
                }
            )
    write_csv(output_dir / "dataset_summary.csv", summary)
    train_ids = {row["task_id"] for row in assignments if row["split"] == "train"}
    test_ids = {row["task_id"] for row in assignments if row["split"] == "test"}
    audit = {
        "schema_version": 1,
        "group_key": "task_id",
        "train_test_task_overlap": sorted(train_ids.intersection(test_ids)),
        "rows_per_task": sorted(set(counts.values())),
        "holdout_selection_uses_outcomes": False,
        "outer_test_used_for_hyperparameter_selection": False,
        "outer_test_used_for_threshold_selection": False,
        "inner_cv": "GroupKFold(task_id)",
        "all_three_checkpoints_co_located": True,
        "task_id_used_as_model_feature": False,
    }
    (output_dir / "leakage_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return summary, audit


def main() -> None:
    args = parse_args()
    training_label = (
        "Combined HumanEval + LiveCodeBench"
        if args.mode == "combined"
        else "LiveCodeBench-only"
    )
    if args.inner_folds < 2 or args.bootstrap_reps < 1:
        raise ValueError("inner-folds must be >=2 and bootstrap-reps positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples: list[PairExample] = []
    metadata: dict[str, dict[str, Any]] = {}
    source_runs: dict[str, str] = {}
    dropped_incomplete: dict[str, dict[str, int]] = {}
    lcb_sources = [args.livecodebench_results, *args.livecodebench_extra_results]
    for source_index, results_dir in enumerate(lcb_sources):
        run_id = _latest_run_id(results_dir)
        lcb_examples, lcb_tasks, dropped = load_livecodebench_examples(
            results_dir,
            run_id,
            drop_incomplete_tasks=True,
        )
        overlap = {row.task_id for row in examples}.intersection(row.task_id for row in lcb_examples)
        if overlap:
            raise ValueError(f"Duplicate tasks across LiveCodeBench sources: {sorted(overlap)[:5]}")
        examples.extend(lcb_examples)
        metadata.update(
            {
                task_id: {**row, "benchmark": "livecodebench"}
                for task_id, row in lcb_tasks.items()
            }
        )
        source_key = f"livecodebench_{source_index}"
        source_runs[source_key] = run_id
        if dropped:
            dropped_incomplete[str(results_dir)] = dropped

    if args.mode == "combined":
        human_run = args.humaneval_run_id or _latest_run_id(args.humaneval_results)
        human = load_phase1_examples(
            results_dir=args.humaneval_results,
            dataset_path=args.humaneval_dataset,
            run_id=human_run,
        )
        examples.extend(human)
        human_ids = {example.task_id for example in human}
        metadata.update(
            {
                task_id: {"benchmark": "humaneval", "difficulty": "unspecified"}
                for task_id in human_ids
            }
        )
        source_runs["humaneval"] = human_run

    examples.sort(key=lambda row: (row.task_id, row.checkpoint))
    example_task_ids = {row.task_id for row in examples}
    if example_task_ids != set(metadata):
        raise AssertionError("Metadata universe differs from the complete task-triple universe")

    holdout_path = args.output_dir / "fixed_test_tasks.json"
    holdout = set(
        load_or_create_holdout(
            holdout_path,
            metadata,
            fraction=args.test_fraction,
            seed=args.seed,
        )
    )
    train = [example for example in examples if example.task_id not in holdout]
    test = [example for example in examples if example.task_id in holdout]
    train_tasks = {row.task_id for row in train}
    test_tasks = {row.task_id for row in test}
    if train_tasks.intersection(test_tasks):
        raise AssertionError("Task leakage between train and frozen test")
    if len(train_tasks) < args.inner_folds:
        raise ValueError("Not enough training tasks for requested GroupKFold")
    if not test:
        raise ValueError("Frozen test split is empty")
    summary, leakage_audit = _write_split_artifacts(
        args.output_dir, examples, metadata, holdout
    )

    models: dict[str, TwoHeadPredictor] = {}
    selections: dict[str, Any] = {}
    for offset, model_name in enumerate(MODEL_NAMES):
        print(f"Selecting compact {model_name} configuration...", flush=True)
        selection = select_hyperparameters(
            train,
            model_name,
            inner_splits=args.inner_folds,
            seed=args.seed + offset * 100_003,
        )
        predictor = TwoHeadPredictor(
            model_name, selection.params, args.seed + offset * 900_001
        ).fit(train)
        models[model_name] = predictor
        selections[model_name] = selection
        joblib.dump(predictor, args.output_dir / f"{model_name}.joblib", compress=3)

    print("Selecting expanded optimized Boosting configuration...", flush=True)
    optimized_selection = select_configuration(
        train,
        inner_folds=args.inner_folds,
        seed=args.seed + 4_000_003,
    )
    optimized_model = TwoHeadPredictor(
        "semantic_boosting",
        optimized_selection.params,
        args.seed + 5_000_003,
    ).fit(train)
    joblib.dump(optimized_model, args.output_dir / "optimized_boosting.joblib", compress=3)

    policy_output: list[dict[str, Any]] = []
    predictor_output: list[dict[str, Any]] = []
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    scopes = ["combined", *sorted({str(metadata[row.task_id]["benchmark"]) for row in test})]
    for scope_index, scope in enumerate(scopes):
        scope_test = [
            row
            for row in test
            if scope == "combined" or metadata[row.task_id]["benchmark"] == scope
        ]
        standard_results = {
            model_name: _test_result(
                scope_test,
                model_name,
                models[model_name],
                float(selections[model_name].threshold),
            )
            for model_name in MODEL_NAMES
        }
        profile = (
            optimized_selection.threshold_cp1,
            optimized_selection.threshold_cp2,
            optimized_selection.threshold_cp3,
        )
        optimized_thresholds = np.asarray(
            [profile[row.checkpoint - 1] for row in scope_test], dtype=float
        )
        optimized_result = _test_result_with_thresholds(
            scope_test,
            "semantic_boosting",
            optimized_model,
            optimized_thresholds,
        )
        baseline, policies = _policy_bundle(scope_test, standard_results, optimized_result)
        bootstrap = bootstrap_policy_metrics(
            baseline,
            policies,
            repetitions=args.bootstrap_reps,
            seed=args.seed + 10_007 * (scope_index + 1),
        )
        policy_output.extend(
            {"benchmark": scope, **row}
            for row in policy_rows(baseline, policies, bootstrap)
        )
        predictor_rows = predictor_metric_rows(scope_test, standard_results)
        for metric_row in predictor_rows:
            if metric_row["model"] != "Local Risk heuristic":
                metric_row["evaluation"] = "frozen outer test; trained and tuned on train only"
        optimized_metrics = predictor_metric_rows(
            scope_test, {"semantic_boosting": optimized_result}
        )[-1]
        optimized_metrics["model"] = "Optimized Boosting"
        optimized_metrics["evaluation"] = "frozen outer test; trained and tuned on train only"
        predictor_output.extend(
            [{"benchmark": scope, **row} for row in predictor_rows]
            + [{"benchmark": scope, **optimized_metrics}]
        )
        write_oof_predictions(
            prediction_dir / f"{scope}_compact.csv", scope_test, standard_results
        )
        write_oof_predictions(
            prediction_dir / f"{scope}_optimized_boosting.csv",
            scope_test,
            {"semantic_boosting": optimized_result},
        )

    write_csv(args.output_dir / "policy_comparison_by_benchmark.csv", policy_output)
    write_csv(args.output_dir / "predictor_metrics_by_benchmark.csv", predictor_output)
    selection_payload = {
        "compact": {name: asdict(value) for name, value in selections.items()},
        "optimized_boosting": asdict(optimized_selection),
    }
    (args.output_dir / "inner_selections.json").write_text(
        json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "experiment": f"{args.mode}_phase2_fixed_holdout_optimization",
        "mode": args.mode,
        "source_runs": source_runs,
        "source_result_dirs": [str(path) for path in lcb_sources],
        "dropped_incomplete_tasks": dropped_incomplete,
        "all_tasks": len(example_task_ids),
        "all_checkpoint_rows": len(examples),
        "train_tasks": len(train_tasks),
        "train_checkpoint_rows": len(train),
        "fixed_test_tasks": len(test_tasks),
        "fixed_test_checkpoint_rows": len(test),
        "test_fraction": args.test_fraction,
        "inner_folds": args.inner_folds,
        "compact_model_configurations": 12,
        "optimized_boosting_configurations": len(boosting_grid()),
        "optimized_threshold_profiles": len(monotone_threshold_profiles()),
        "optimized_threshold_constraint": "CP1 >= CP2 >= CP3",
        "bootstrap_repetitions": args.bootstrap_reps,
        "seed": args.seed,
        "leakage_audit": leakage_audit,
        "llm_inference_performed": False,
        "teacher_calls_performed": False,
        "evaluator_calls_performed": False,
        "python": sys.version,
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    selected_methods = {
        "Never Verify",
        "Local Risk Controller",
        "Ours Logistic",
        "Ours Boosting",
        "Optimized Boosting",
        "Always Verify CP3",
        "Offline Oracle",
    }
    table_rows = [row for row in policy_output if row["method"] in selected_methods]
    report = [
        f"# {training_label} Predictor",
        "",
        f"Complete task triples: **{len(example_task_ids)}** ({len(examples)} checkpoint rows). "
        f"Train: **{len(train_tasks)} tasks**; frozen test: **{len(test_tasks)} tasks**.",
        "",
        "The outer split is deterministic and stratified by benchmark/difficulty using only task IDs "
        "and pre-treatment metadata. CP1/CP2/CP3 from the same task are always co-located. "
        "All hyperparameters and thresholds are selected inside training-only GroupKFold.",
        "",
        f"Dropped incomplete task triples: `{dropped_incomplete}`.",
        "",
        "## Frozen-test policy results",
        "",
        "| Scope | Method | Accuracy | Gain vs Never | Calls | Helpful / Harmful |",
        "|---|---|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['benchmark']} | {row['method']} | "
        f"{100 * float(row['accuracy']):.2f}% | "
        f"{float(row['gain_vs_never_pp']):+.2f} pp | "
        f"{float(row['teacher_calls']):.0f} | "
        f"{float(row['helpful']):.0f} / {float(row['harmful']):.0f} |"
        for row in table_rows
    )
    report.extend(
        [
            "",
            "## Optimization",
            "",
            f"Expanded Boosting searched **{len(boosting_grid())}** configurations and "
            f"**{len(monotone_threshold_profiles())}** monotone checkpoint-threshold profiles, "
            "without inspecting the frozen outer test.",
            "",
            "## Scientific status",
            "",
            "This is a genuine frozen-test evaluation for this training run. The same fixed test must "
            "not be reused for further adaptive tuning after reading these results; a new benchmark "
            "slice is required for another optimization round.",
            "",
            "Offline training only: no Student, Teacher, or evaluator call was made.",
            "",
        ]
    )
    (args.output_dir / "phase2_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"{training_label} training artifacts written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
