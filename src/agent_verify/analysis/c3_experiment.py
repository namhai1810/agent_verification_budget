"""Training and evaluation pipeline for the C3 learned sequential controller."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

from agent_verify.analysis.c3_controller import (
    ControllerObservation,
    FittedSequentialController,
    extract_observations,
    group_observations,
)
from agent_verify.analysis.joint_value_cost_data import (
    FrozenSplit,
    JointExample,
    dataset_fingerprint,
    filter_examples,
    load_config,
    load_frozen_split,
    load_joint_examples,
)
from agent_verify.analysis.joint_value_cost_experiment import (
    MODEL_KINDS,
    TRAINING_DATASETS,
    read_csv,
    write_csv,
)
from agent_verify.analysis.joint_value_cost_policy import (
    PolicyReplay,
    replay_summary,
    token_budgets,
)

CONTROLLER_VARIANTS = {
    "continuation": True,
    "myopic": False,
}


def load_environment(
    c3_config: dict[str, Any],
) -> tuple[dict[str, Any], list[JointExample], FrozenSplit]:
    upstream_config = load_config(
        Path(c3_config["upstream"]["joint_value_cost_config"])
    )
    examples = load_joint_examples(upstream_config)
    split = load_frozen_split(upstream_config, examples)
    return upstream_config, examples, split


def _joint_model_path(c3_config: dict[str, Any], mode: str) -> Path:
    return (
        Path(c3_config["upstream"]["joint_value_cost_results"])
        / "training"
        / mode
        / "joint_model.joblib"
    )


def load_frozen_joint(c3_config: dict[str, Any], mode: str) -> Any:
    path = _joint_model_path(c3_config, mode)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pretrained Joint model {path}; C3 never retrains it automatically"
        )
    return joblib.load(path)


def _budget_levels(
    examples: list[JointExample],
    quantiles: list[float],
) -> list[float]:
    costs = np.asarray([row.teacher_total_tokens for row in examples], dtype=float)
    levels = {0.0}
    levels.update(float(np.quantile(costs, quantile)) for quantile in quantiles)
    return sorted(levels)


def prepare_c3(
    config: dict[str, Any],
    *,
    output_root: Path,
) -> None:
    upstream_config, examples, split = load_environment(config)
    required_models = {
        mode: str(_joint_model_path(config, mode)) for mode in TRAINING_DATASETS
    }
    missing = [path for path in required_models.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing pretrained Joint models: {missing}")
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment": config["experiment"]["name"],
        "dataset_fingerprint": dataset_fingerprint(examples),
        "tasks": len({row.group_key for row in examples}),
        "pairs": len(examples),
        "train_tasks": len(split.train_group_keys),
        "test_tasks": len(split.test_group_keys),
        "train_test_overlap": sorted(
            split.train_group_keys.intersection(split.test_group_keys)
        ),
        "frozen_manifest": str(split.manifest_path),
        "frozen_assignments": str(split.assignments_path),
        "frozen_test_status": "exploratory_previously_observed",
        "pretrained_joint_models": required_models,
        "pretrained_joint_retrained": False,
        "student_teacher_calls": False,
        "single_intervention_only": True,
        "q_target_approximation": (
            "Observed VERIFY correctness for verify-now; fitted downstream Q for wait. "
            "Future states are target-only. Cross-task opportunity cost is approximated "
            "by a budget-dependent shadow price and budget per remaining task."
        ),
        "upstream_dataset_fingerprint": dataset_fingerprint(examples),
        "upstream_config": str(
            config["upstream"]["joint_value_cost_config"]
        ),
        "upstream_output": str(
            config["upstream"]["joint_value_cost_results"]
        ),
    }
    (output_root / "data_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _group_folds(observations: list[ControllerObservation], folds: int):
    groups = np.asarray([row.group_key for row in observations])
    splitter = GroupKFold(n_splits=folds)
    for train_indices, validation_indices in splitter.split(
        np.zeros(len(observations)), groups=groups
    ):
        train_groups = {observations[int(index)].group_key for index in train_indices}
        validation_groups = {
            observations[int(index)].group_key for index in validation_indices
        }
        if train_groups.intersection(validation_groups):
            raise AssertionError("Task leakage in C3 GroupKFold")
        yield train_indices, validation_indices


def _stable_task_order(task_ids: list[str], seed: int) -> list[str]:
    return sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{seed}:{task_id}".encode()
        ).hexdigest(),
    )


def replay_controller(
    observations: list[ControllerObservation],
    controller: FittedSequentialController,
    *,
    policy_name: str,
    token_budget: int,
    seed: int,
    admission_mode: str,
) -> tuple[PolicyReplay, list[dict[str, Any]]]:
    """Sequential replay with predicted online feasibility and optional hard accounting."""

    if admission_mode not in {"expected_cost", "retrospective_hard_cap"}:
        raise ValueError(f"Unknown admission mode: {admission_mode}")
    grouped = group_observations(observations)
    task_ids = _stable_task_order(list(grouped), seed)
    index_lookup = {
        (row.group_key, row.example.checkpoint): index
        for index, row in enumerate(observations)
    }
    predicted_remaining = float(token_budget)
    actual_remaining = int(token_budget)
    actual_input = 0
    actual_output = 0
    predicted_total = 0.0
    outcomes: list[float] = []
    selected_indices: list[int] = []
    selected_checkpoints: list[int] = []
    decision_rows: list[dict[str, Any]] = []

    for order_index, task_id in enumerate(task_ids):
        task_rows = grouped[task_id]
        tasks_remaining = len(task_ids) - order_index
        decision_budget = (
            predicted_remaining
            if admission_mode == "expected_cost"
            else float(actual_remaining)
        )
        decision = controller.decide(
            task_rows,
            remaining_budget=decision_budget,
            initial_budget=float(token_budget),
            tasks_remaining=tasks_remaining,
        )
        selected_observation = (
            task_rows[decision.selected_index]
            if decision.selected_index >= 0
            else None
        )
        admitted = False
        global_index = -1
        if selected_observation is not None:
            predicted_cost = max(decision.predicted_cost, 0.0)
            actual_cost = selected_observation.example.teacher_total_tokens
            if admission_mode == "expected_cost":
                admitted = predicted_cost <= predicted_remaining
            else:
                # Non-deployable accounting check: actual current cost is used here
                # only to construct a strict-cap comparison.
                admitted = (
                    predicted_cost <= actual_remaining
                    and actual_cost <= actual_remaining
                )
            if admitted:
                predicted_remaining -= predicted_cost
                actual_remaining -= actual_cost
                predicted_total += predicted_cost
                actual_input += selected_observation.example.teacher_input_tokens
                actual_output += selected_observation.example.teacher_output_tokens
                global_index = index_lookup[
                    (task_id, selected_observation.example.checkpoint)
                ]
        if admitted and selected_observation is not None:
            outcomes.append(float(selected_observation.example.y_verify))
            selected_indices.append(global_index)
            selected_checkpoints.append(selected_observation.example.checkpoint)
        else:
            outcomes.append(float(task_rows[0].example.y_skip))
            selected_indices.append(-1)
            selected_checkpoints.append(0)

        decision_rows.append(
            {
                "task_group": task_id,
                "task_order": order_index,
                "tasks_remaining": tasks_remaining,
                "budget_before": decision_budget,
                "selected_checkpoint_before_admission": decision.checkpoint,
                "admitted": admitted,
                "selected_checkpoint": (
                    selected_observation.example.checkpoint if admitted else 0
                ),
                "q_verify": decision.q_verify,
                "q_wait": decision.q_wait,
                "predicted_cost": decision.predicted_cost,
                "actual_cost": (
                    selected_observation.example.teacher_total_tokens
                    if selected_observation is not None
                    else 0
                ),
                "trace_json": json.dumps(decision.trace),
            }
        )

    replay = PolicyReplay(
        policy=policy_name,
        admission_mode=admission_mode,
        token_budget=int(token_budget),
        task_ids=tuple(task_ids),
        outcomes=np.asarray(outcomes, dtype=float),
        selected_indices=np.asarray(selected_indices, dtype=int),
        selected_checkpoints=np.asarray(selected_checkpoints, dtype=int),
        actual_input_tokens=actual_input,
        actual_output_tokens=actual_output,
        actual_total_tokens=actual_input + actual_output,
        predicted_total_tokens=predicted_total,
    )
    return replay, decision_rows


def _candidate_score(
    validation_summaries: list[dict[str, float | int | str]],
    *,
    token_weight: float,
    overshoot_weight: float,
) -> float:
    scores = []
    for row in validation_summaries:
        budget = max(int(row["token_budget"]), 1)
        gain = float(row["gain_vs_never_pp"]) / 100.0
        token_fraction = float(row["teacher_total_tokens"]) / budget
        overshoot_fraction = float(row["budget_overshoot_tokens"]) / budget
        scores.append(
            gain
            - float(token_weight) * token_fraction
            - float(overshoot_weight) * overshoot_fraction
        )
    return float(np.mean(scores))


def select_controller_candidate(
    observations: list[ControllerObservation],
    *,
    candidates: list[dict[str, Any]],
    variant: str,
    config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if variant not in CONTROLLER_VARIANTS:
        raise ValueError(f"Unknown C3 variant: {variant}")
    folds = int(config["training"]["group_folds"])
    fractions = [
        float(value) for value in config["training"]["remaining_fraction_grid"]
    ]
    quantiles = [
        float(value) for value in config["training"]["budget_quantiles"]
    ]
    rows: list[dict[str, Any]] = []
    best_key: tuple[float, int] | None = None
    best_candidate: dict[str, Any] | None = None
    candidate_iterator = tqdm(
        enumerate(candidates),
        total=len(candidates),
        desc=f"C3 {variant} candidates",
        unit="candidate",
        dynamic_ncols=True,
    )
    fold_splits = list(_group_folds(observations, folds))
    for candidate_index, candidate in candidate_iterator:
        summaries: list[dict[str, float | int | str]] = []
        fold_iterator = tqdm(
            fold_splits,
            total=folds,
            desc=f"{variant} candidate {candidate_index + 1}/{len(candidates)}",
            unit="fold",
            leave=False,
            dynamic_ncols=True,
        )
        for fold_index, (train_indices, validation_indices) in enumerate(fold_iterator):
            train = [observations[int(index)] for index in train_indices]
            validation = [observations[int(index)] for index in validation_indices]
            levels = _budget_levels(
                [row.example for row in train],
                quantiles,
            )
            controller = FittedSequentialController(
                candidate,
                seed=seed + candidate_index * 10_003 + fold_index,
                use_continuation=CONTROLLER_VARIANTS[variant],
                min_action_margin=float(
                    config["training"].get("min_action_margin", 0.0)
                ),
            ).fit(
                train,
                budget_levels=levels,
                remaining_fractions=fractions,
            )
            validation_tasks = len(group_observations(validation))
            for per_task_budget in levels:
                replay, _decisions = replay_controller(
                    validation,
                    controller,
                    policy_name=f"validation:{variant}",
                    token_budget=int(round(per_task_budget * validation_tasks)),
                    seed=seed + fold_index,
                    admission_mode="expected_cost",
                )
                summaries.append(
                    replay_summary(
                        [row.example for row in validation],
                        replay,
                        bootstrap_repetitions=200,
                        seed=seed + fold_index,
                    )
                )
        score = _candidate_score(
            summaries,
            token_weight=float(
                config["training"].get("selection_token_weight", 0.01)
            ),
            overshoot_weight=float(
                config["training"].get("selection_overshoot_weight", 0.10)
            ),
        )
        mean_gain = float(
            np.mean([float(row["gain_vs_never_pp"]) for row in summaries])
        )
        mean_tokens = float(
            np.mean([float(row["teacher_total_tokens"]) for row in summaries])
        )
        mean_calls = float(
            np.mean([float(row["teacher_calls"]) for row in summaries])
        )
        rows.append(
            {
                "variant": variant,
                "candidate_index": candidate_index,
                "params_json": json.dumps(candidate, sort_keys=True),
                "selection_score": score,
                "mean_gain_vs_never_pp": mean_gain,
                "mean_teacher_tokens": mean_tokens,
                "mean_teacher_calls": mean_calls,
                "folds": folds,
            }
        )
        candidate_iterator.set_postfix(
            score=f"{score:.4f}",
            gain_pp=f"{mean_gain:.2f}",
        )
        key = (score, -candidate_index)
        if best_key is None or key > best_key:
            best_key = key
            best_candidate = dict(candidate)
    if best_candidate is None:
        raise RuntimeError("No C3 controller candidate was selected")
    return best_candidate, rows


def train_controller_mode(
    config: dict[str, Any],
    *,
    mode: str,
    output_root: Path,
    resume: bool,
) -> None:
    if mode not in TRAINING_DATASETS:
        raise ValueError(f"Unknown C3 training mode: {mode}")
    _upstream, examples, split = load_environment(config)
    train_examples = filter_examples(
        examples,
        split,
        partition="train",
        datasets=TRAINING_DATASETS[mode],
    )
    joint_model = load_frozen_joint(config, mode)
    observations = extract_observations(train_examples, joint_model)
    mode_dir = output_root / "training" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        mode_dir / f"controller_{variant}.joblib"
        for variant in CONTROLLER_VARIANTS
    ]
    manifest_path = mode_dir / "training_manifest.json"
    if resume and all(path.exists() for path in expected) and manifest_path.exists():
        tqdm.write(f"[C3 {mode}] already complete; resume skipped training")
        return
    existing = [path for path in [*expected, manifest_path] if path.exists()]
    if existing and not resume:
        raise FileExistsError(
            f"Refusing to overwrite existing C3 artifacts for {mode}: {existing}. "
            "Use a new --output-root or pass --resume."
        )

    seed = int(config["experiment"]["seed"])
    candidates = [
        dict(row) for row in config["training"]["candidates"]
    ]
    selection_path = mode_dir / "candidate_selection.csv"
    selected_path = mode_dir / "selected_controllers.json"
    selection_rows: list[dict[str, Any]] = (
        [dict(row) for row in read_csv(selection_path)]
        if resume and selection_path.exists()
        else []
    )
    selected: dict[str, dict[str, Any]] = (
        json.loads(selected_path.read_text(encoding="utf-8"))
        if resume and selected_path.exists()
        else {}
    )
    levels = _budget_levels(
        train_examples,
        [
            float(value)
            for value in config["training"]["budget_quantiles"]
        ],
    )
    fractions = [
        float(value)
        for value in config["training"]["remaining_fraction_grid"]
    ]

    for variant_index, variant in enumerate(CONTROLLER_VARIANTS):
        model_path = mode_dir / f"controller_{variant}.joblib"
        if resume and model_path.exists():
            controller = joblib.load(model_path)
            selected[variant] = {
                "params": dict(controller.params),
                "use_continuation": CONTROLLER_VARIANTS[variant],
                "budget_levels": levels,
                "remaining_fractions": fractions,
                "resumed_existing_model": True,
            }
            tqdm.write(f"[C3 {mode}/{variant}] preserved existing model")
            continue
        chosen, rows = select_controller_candidate(
            observations,
            candidates=candidates,
            variant=variant,
            config=config,
            seed=seed + variant_index * 1_000_003,
        )
        selection_rows.extend(rows)
        controller = FittedSequentialController(
            chosen,
            seed=seed + variant_index * 2_000_003,
            use_continuation=CONTROLLER_VARIANTS[variant],
            min_action_margin=float(
                config["training"].get("min_action_margin", 0.0)
            ),
        ).fit(
            observations,
            budget_levels=levels,
            remaining_fractions=fractions,
        )
        joblib.dump(controller, model_path, compress=3)
        selected[variant] = {
            "params": chosen,
            "use_continuation": CONTROLLER_VARIANTS[variant],
            "budget_levels": levels,
            "remaining_fractions": fractions,
        }

    write_csv(selection_path, selection_rows)
    selected_path.write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "experiment": config["experiment"]["name"],
        "training_mode": mode,
        "datasets": sorted(TRAINING_DATASETS[mode]),
        "train_tasks": len(group_observations(observations)),
        "train_pairs": len(observations),
        "pretrained_joint_model": str(_joint_model_path(config, mode)),
        "pretrained_joint_retrained": False,
        "controller_grouped_cv_folds": int(
            config["training"]["group_folds"]
        ),
        "frozen_test_used_for_selection": False,
        "future_state_used_as_inference_feature": False,
        "future_state_used_for_continuation_target_only": True,
        "single_intervention_only": True,
        "dataset_fingerprint": dataset_fingerprint(examples),
        "seed": seed,
    }
    (mode_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def evaluate_controllers(
    config: dict[str, Any],
    *,
    output_root: Path,
) -> None:
    _upstream, examples, split = load_environment(config)
    scopes = {
        "humaneval": filter_examples(
            examples, split, partition="test", datasets={"humaneval"}
        ),
        "livecodebench": filter_examples(
            examples, split, partition="test", datasets={"livecodebench"}
        ),
        "combined": filter_examples(examples, split, partition="test"),
    }
    combined_train = filter_examples(examples, split, partition="train")
    quantiles = [
        float(value) for value in config["evaluation"]["budget_quantiles"]
    ]
    seed = int(config["experiment"]["seed"])
    repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    result_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    total = (
        len(TRAINING_DATASETS)
        * len(CONTROLLER_VARIANTS)
        * len(scopes)
        * len({"expected_cost", "retrospective_hard_cap"})
        * (len(quantiles) + 1)
    )
    progress = tqdm(
        total=total,
        desc="C3 sequential evaluation",
        unit="replay",
        dynamic_ncols=True,
    )

    for train_mode in TRAINING_DATASETS:
        joint_model = load_frozen_joint(config, train_mode)
        controllers = {
            variant: joblib.load(
                output_root
                / "training"
                / train_mode
                / f"controller_{variant}.joblib"
            )
            for variant in CONTROLLER_VARIANTS
        }
        for scope, test_examples in scopes.items():
            observations = extract_observations(test_examples, joint_model)
            budgets = token_budgets(
                combined_train,
                len({row.group_key for row in test_examples}),
                quantiles,
            )
            for variant, controller in controllers.items():
                for budget in budgets:
                    for admission_mode in (
                        "expected_cost",
                        "retrospective_hard_cap",
                    ):
                        replay, decisions = replay_controller(
                            observations,
                            controller,
                            policy_name=f"C3:{train_mode}:{variant}",
                            token_budget=budget,
                            seed=seed,
                            admission_mode=admission_mode,
                        )
                        result_rows.append(
                            {
                                "source": "c3",
                                "scope": scope,
                                "train_mode": train_mode,
                                "model": f"c3_{variant}",
                                "continuation": CONTROLLER_VARIANTS[variant],
                                **replay_summary(
                                    test_examples,
                                    replay,
                                    bootstrap_repetitions=repetitions,
                                    seed=seed,
                                ),
                            }
                        )
                        for row in decisions:
                            decision_rows.append(
                                {
                                    "scope": scope,
                                    "train_mode": train_mode,
                                    "model": f"c3_{variant}",
                                    "continuation": CONTROLLER_VARIANTS[variant],
                                    "admission_mode": admission_mode,
                                    "token_budget": budget,
                                    **row,
                                }
                            )
                        progress.set_postfix(
                            train=train_mode,
                            test=scope,
                            variant=variant,
                            budget=budget,
                        )
                        progress.update()
    progress.close()

    for row in result_rows:
        calls = int(row["teacher_calls"])
        cp2_share = float(row["cp2_calls"]) / calls if calls else 0.0
        row["cp2_call_share"] = cp2_share
        row["cp2_bias_vs_uniform_pp"] = 100.0 * (cp2_share - 1.0 / 3.0) if calls else 0.0

    evaluation_dir = output_root / "evaluation"
    write_csv(evaluation_dir / "c3_policy_results.csv", result_rows)
    write_csv(evaluation_dir / "c3_decisions.csv", decision_rows)

    existing = read_csv(
        Path(config["upstream"]["existing_policy_results"])
    )
    baseline_models = {
        "Never Verify",
        "Always CP1",
        "Always CP2",
        "Always CP3",
        "Random Verification",
        "Offline Oracle",
        "Legacy Boosting",
    }
    comparison: list[dict[str, Any]] = []
    for row in existing:
        keep_baseline = (
            row["train_mode"] == "baseline"
            and row["model"] in baseline_models
        )
        keep_threshold = (
            row["train_mode"] == "combined"
            and row["model"] in MODEL_KINDS
        )
        if keep_baseline or keep_threshold:
            comparison.append(
                {
                    "source": (
                        "threshold_or_separate"
                        if keep_threshold
                        else "existing_baseline"
                    ),
                    **row,
                }
            )
    comparison.extend(result_rows)
    for row in comparison:
        if "cp2_call_share" in row:
            continue
        calls = int(row["teacher_calls"])
        cp2_share = float(row["cp2_calls"]) / calls if calls else 0.0
        row["cp2_call_share"] = cp2_share
        row["cp2_bias_vs_uniform_pp"] = (
            100.0 * (cp2_share - 1.0 / 3.0) if calls else 0.0
        )
    for row in comparison:
        peers = [
            other
            for other in comparison
            if other["scope"] == row["scope"]
            and other["admission_mode"] == row["admission_mode"]
        ]
        dominated = any(
            float(other["accuracy"]) >= float(row["accuracy"])
            and int(other["teacher_total_tokens"]) <= int(row["teacher_total_tokens"])
            and (
                float(other["accuracy"]) > float(row["accuracy"])
                or int(other["teacher_total_tokens"]) < int(row["teacher_total_tokens"])
            )
            for other in peers
        )
        row["pareto_efficient_global"] = not dominated
    write_csv(evaluation_dir / "policy_comparison.csv", comparison)
    (evaluation_dir / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "status": "exploratory_previously_observed_frozen_test",
                "frozen_test_reused": True,
                "winner_selection_allowed": False,
                "online_feasibility": "predicted Teacher cost only",
                "hard_cap_status": (
                    "retrospective accounting comparison; not deployable foresight"
                ),
                "max_teacher_calls_per_task": 1,
                "future_features_at_inference": False,
                "multi_verification_outcomes_assumed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
