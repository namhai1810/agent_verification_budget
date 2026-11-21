"""Leakage-safe DARC v0.5 Myopic-i0 retraining on Qwen3.8 Full-8 data."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm

from agent_verify.analysis.cbv_v01_experiment import (
    ReplayResult,
    _passed,
    _stable_order,
)
from agent_verify.analysis.darc_data import encode_or_load
from agent_verify.analysis.darc_qwen38_data import (
    assign_stratified_task_folds,
    load_qwen38_full8_groups,
    load_selection_keys,
    split_audit_rows,
)
from agent_verify.analysis.darc_v04_audit import aggregate_records
from agent_verify.analysis.darc_v04_data import build_states, verify_cost
from agent_verify.analysis.darc_v04_experiment import _write_csv
from agent_verify.analysis.darc_v04_model import DARCv04Ensemble
from agent_verify.analysis.darc_v05_env import CBVLevel2OfflineEnv
from agent_verify.analysis.darc_v05_experiment import (
    _evaluate_v05,
    _precompute_v05,
    _record,
    _tune_v05_policy,
    _writer,
)
from agent_verify.analysis.darc_v05_model import DARCv05Ensemble, V05Prediction
from agent_verify.analysis.darc_v05_training import (
    ConservativePolicy,
    generate_policy_targets,
    policy_grid,
    train_crossfit_iteration,
    validate_target_coverage,
)

ITERATION = 0


def _atomic_save(model: DARCv05Ensemble, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    model.save(temporary)
    temporary.replace(path)


def _reference_budget(groups: dict[str, Any]) -> float:
    values = [int(group.rows["VVV"]["teacher_total_tokens"]) for group in groups.values()]
    if not values or min(values) <= 0:
        raise ValueError("Development VVV costs must be positive")
    return float(np.mean(values))


def _max_incremental_cost(groups: dict[str, Any]) -> int:
    prefixes = ("", "S", "V", "SS", "SV", "VS", "VV")
    return max(verify_cost(group, prefix) for group in groups.values() for prefix in prefixes)


def _copy_prediction(template: V05Prediction) -> dict[str, np.ndarray]:
    return {
        field: np.zeros_like(getattr(template, field))
        for field in V05Prediction.__dataclass_fields__
    }


def summarize_oof_variant(
    records: list[dict[str, Any]],
    *,
    label: str,
    expected_seeds: int,
    reference_budget: float,
    task_count: int,
    token_penalty: float,
) -> dict[str, Any]:
    """Summarize full-budget OOF records using the canonical replay schema."""

    rows = [
        row
        for row in records
        if row["policy"] == label and float(row["fraction"]) == 1.0
    ]
    if len(rows) != expected_seeds:
        raise ValueError(
            f"Expected {expected_seeds} full-budget OOF rows for {label}, "
            f"found {len(rows)}"
        )
    accuracy = float(np.mean([float(row["summary"]["accuracy"]) for row in rows]))
    # _seed_summary names the realized ledger field `teacher_tokens`.
    tokens = float(np.mean([float(row["summary"]["teacher_tokens"]) for row in rows]))
    score = accuracy - token_penalty * (
        tokens / max(reference_budget * task_count, 1.0)
    )
    return {"label": label, "accuracy": accuracy, "tokens": tokens, "score": score}


def oof_predictions(
    model: DARCv05Ensemble,
    encoded: Any,
    states: list[Any],
    allowances: list[float],
    task_folds: dict[str, int],
    *,
    device: str,
) -> list[V05Prediction]:
    """Predict each development task only with members that excluded its fold."""

    assembled: list[dict[str, np.ndarray]] | None = None
    covered = np.zeros(len(states), dtype=np.int64)
    for fold in sorted(set(task_folds.values())):
        fold_predictions = _precompute_v05(
            model.members_for_fold(fold),
            encoded,
            states,
            allowances,
            device=device,
            description=f"OOF prediction fold {fold}",
        )
        if assembled is None:
            assembled = [_copy_prediction(row) for row in fold_predictions]
        mask = np.asarray([task_folds[state.task_key] == fold for state in states], dtype=bool)
        covered += mask.astype(np.int64)
        for allowance_index, prediction in enumerate(fold_predictions):
            assert assembled is not None
            for field in V05Prediction.__dataclass_fields__:
                assembled[allowance_index][field][mask] = getattr(prediction, field)[mask]
    if assembled is None or not np.all(covered == 1):
        raise AssertionError("OOF predictions did not cover every state exactly once")
    return [V05Prediction(**row) for row in assembled]


def _fixed_replay(
    groups: dict[str, Any],
    lookup: dict[tuple[str, str], int],
    *,
    sequence: str,
    token_budget: int,
    budget_fraction: float,
    hard_cap: int,
    seed: int,
    label: str,
) -> tuple[ReplayResult, list[dict[str, Any]]]:
    keys = _stable_order(groups, seed)
    env = CBVLevel2OfflineEnv(groups, lookup, hard_call_token_cap=hard_cap)
    spent = calls = 0
    outcomes: list[float] = []
    baselines: list[float] = []
    masks: list[int] = []
    traces: list[dict[str, Any]] = []
    for key in keys:
        observation = env.reset(key, max(token_budget - spent, 0), per_task_allowance=0.0)
        mask = 0
        while True:
            requested = sequence[observation.checkpoint - 1]
            action = requested if requested == "S" or env.feasible("V") else "S"
            prefix = observation.prefix
            skip_seq = prefix + "S" + "S" * (2 - len(prefix))
            verify_seq = prefix + "V" + "S" * (2 - len(prefix))
            delta = _passed(groups[key].rows[verify_seq]) - _passed(groups[key].rows[skip_seq])
            transition = env.step(action)
            spent += transition.incremental_teacher_tokens
            calls += int(action == "V")
            if action == "V":
                mask |= 1 << (observation.checkpoint - 1)
            traces.append(
                {
                    "policy": label,
                    "seed": seed,
                    "budget_fraction": budget_fraction,
                    "task_key": key,
                    "dataset": groups[key].task.dataset,
                    "checkpoint": observation.checkpoint,
                    "history": prefix or "START",
                    "action": action,
                    "requested_action": requested,
                    "actual_cost_after_action": transition.incremental_teacher_tokens,
                    "helpful_selected": int(action == "V" and delta > 0),
                    "harmful_selected": int(action == "V" and delta < 0),
                    "no_change_selected": int(action == "V" and delta == 0),
                }
            )
            if transition.done:
                outcomes.append(float(transition.terminal_correctness or 0))
                break
            observation = transition.next_observation
            assert observation is not None
        baselines.append(float(_passed(groups[key].rows["SSS"])))
        masks.append(mask)
    if spent > token_budget:
        raise AssertionError("Fixed policy exceeded strict actual-token budget")
    return ReplayResult(
        label,
        "qwen38-heldout-strict",
        "qwen38",
        "combined",
        budget_fraction,
        token_budget,
        tuple(keys),
        np.asarray(outcomes),
        np.asarray(baselines),
        np.asarray(masks),
        calls,
        spent,
        0.0,
    ), traces


def _integrity(
    config: dict[str, Any], *, require_test: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], set[str]]:
    frozen = Path(config["data"]["frozen_root"])
    expected_dev = int(config["evaluation"]["development_tasks"])
    development, metadata = load_qwen38_full8_groups(
        [Path(value) for value in config["data"]["development_roots"]],
        frozen_root=frozen,
        expected_tasks=expected_dev,
    )
    prospective_value = config["data"].get("prospective_test_root")
    expected_test = int(config["evaluation"].get("prospective_test_tasks", 0))
    prospective_root = Path(prospective_value) if prospective_value else None
    selected = load_selection_keys(prospective_root) if prospective_root else set()
    if len(selected) != expected_test:
        raise ValueError(f"Expected {expected_test} prospective task IDs, found {len(selected)}")
    overlap = set(development) & selected
    if overlap:
        raise ValueError(f"Development/prospective leakage: {sorted(overlap)[:3]}")
    prospective: dict[str, Any] = {}
    checkpoint_count = (
        len(list((prospective_root / "task_checkpoints").glob("*.json")))
        if prospective_root
        else 0
    )
    if require_test and checkpoint_count < expected_test:
        raise FileNotFoundError(
            "The sealed prospective 20-task Full-8 collection is not complete. "
            "Run scripts/run_darc_qwen38_full20_extra_gpu2.sh first."
        )
    return development, prospective, metadata, selected


def _figure(output: Path, aggregate: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    policies = sorted({str(row["policy"]) for row in aggregate})
    for policy in policies:
        points = sorted(
            [row for row in aggregate if row["policy"] == policy],
            key=lambda row: float(row["teacher_tokens_mean"]),
        )
        axis.plot(
            [float(row["teacher_tokens_mean"]) / 1000 for row in points],
            [100 * float(row["accuracy_mean"]) for row in points],
            marker="o",
            linewidth=2 if "DARC" in policy else 1.3,
            label=policy,
        )
    axis.set(
        title="Qwen3.8 held-out accuracy–token frontier",
        xlabel="Actual Teacher tokens (thousands)",
        ylabel="Final accuracy (%)",
    )
    axis.grid(color="#E4E8EE", linewidth=0.8)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output / "figures" / "heldout_accuracy_cost.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _report(
    output: Path,
    aggregate: list[dict[str, Any]],
    *,
    reference_budget: float,
    selected_variant: str,
    source_tasks: int,
    development_tasks: int,
    heldout_tasks: int,
    prospective_tasks: int,
    report_filename: str,
) -> Path:
    full = [row for row in aggregate if float(row["budget_fraction"]) == 1.0]
    lines = [
        "# DARC v0.5 Myopic i0 on Qwen3.8 Full-8",
        "",
        f"> The {source_tasks}-task Qwen3.8 cohort is split deterministically into "
        f"{development_tasks} development tasks and {heldout_tasks} retrospective "
        "held-out tasks. Qwen3.8 held-out outcomes are never used for fitting or "
        "policy selection.",
        "",
        "## Protocol",
        "",
        "- Controller: unchanged DARC v0.5 Myopic iteration-0 targets.",
        f"- Development: {development_tasks} Qwen3.8 Full-8 tasks; grouped OOF folds; "
        "three seeds.",
        f"- Evaluation: {heldout_tasks} deterministic task-ID held-out tasks.",
        f"- External prospective tasks configured/accessed: {prospective_tasks}/0.",
        "- Online decisions never observe future artifacts, outcomes, or realized future call cost.",
        f"- Frozen development reference budget: **{reference_budget:,.1f} tokens/task**.",
        f"- OOF-selected model variant before held-out evaluation: **{selected_variant}**.",
        "",
        "## Retrospective held-out full-budget results",
        "",
        "| Policy | Accuracy mean | Actual tokens | Calls | Helpful / Harmful / No-change |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(full, key=lambda item: float(item["accuracy_mean"]), reverse=True):
        lines.append(
            f"| {row['policy']} | {100*float(row['accuracy_mean']):.2f}% | "
            f"{float(row['teacher_tokens_mean']):,.0f} | {float(row['teacher_calls_mean']):.1f} | "
            f"{float(row['helpful_mean']):.1f} / {float(row['harmful_mean']):.1f} / "
            f"{float(row['no_change_mean']):.1f} |"
        )
    lines += [
        "",
        "![Held-out accuracy–cost frontier](../figures/heldout_accuracy_cost.png)",
        "",
        "## Interpretation",
        "",
        f"This {heldout_tasks}-task result is a retrospective internal holdout because "
        f"aggregate outcomes from the {source_tasks}-task cohort were inspected before "
        "this split. It measures leakage-safe pipeline generalization within the new "
        "split, but it is not untouched confirmatory evidence.",
        "",
        "## Artifacts",
        "",
        "- `split_integrity.json` and `tables/task_folds.csv`",
        "- `checkpoints/<variant>/seed_<seed>/iteration_0.pt`",
        "- `tables/oof_results.csv`, `tables/heldout_results.csv`",
        "- `tables/policy_grid.csv`, `tables/decision_trace.csv`",
    ]
    path = output / "report" / report_filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(
    config_path: Path,
    *,
    mode: str,
    resume: bool = False,
) -> Path:
    if mode not in {"dry-run", "smoke", "full"}:
        raise ValueError(mode)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["evaluation"]["frozen_test_access"] != "forbidden":
        raise ValueError("Frozen test access must remain forbidden")
    output = Path(config["output"]["root"])
    for child in ("logs", "checkpoints", "tables", "figures", "report", "status", "tensorboard"):
        (output / child).mkdir(parents=True, exist_ok=True)

    all_groups, _prospective, all_metadata, sealed_test = _integrity(
        config, require_test=False
    )
    outer_folds = assign_stratified_task_folds(
        all_metadata,
        folds=int(config["reproducibility"]["folds"]),
        seed=int(config["reproducibility"]["fold_seed"]),
    )
    heldout_fold = int(config["reproducibility"]["heldout_fold"])
    heldout_keys = {key for key, fold in outer_folds.items() if fold == heldout_fold}
    development = {key: group for key, group in all_groups.items() if key not in heldout_keys}
    heldout = {key: group for key, group in all_groups.items() if key in heldout_keys}
    development_metadata = {key: all_metadata[key] for key in development}
    folds = assign_stratified_task_folds(
        development_metadata,
        folds=int(config["reproducibility"]["inner_folds"]),
        seed=int(config["reproducibility"]["inner_fold_seed"]),
    )
    expected_train = int(config["evaluation"]["retrospective_train_tasks"])
    expected_test = int(config["evaluation"]["retrospective_test_tasks"])
    if (len(development), len(heldout)) != (expected_train, expected_test):
        raise ValueError(
            "Retrospective split count mismatch: "
            f"{len(development)}/{len(heldout)} != {expected_train}/{expected_test}"
        )
    audit_rows = split_audit_rows(all_groups, all_metadata, outer_folds)
    for row in audit_rows:
        key = str(row["task_key"])
        row["partition"] = "heldout" if key in heldout_keys else "development"
        row["inner_fold"] = "" if key in heldout_keys else folds[key]
    _write_csv(output / "tables" / "task_folds.csv", audit_rows)
    reference_budget = _reference_budget(development)
    hard_cap = int(config["offline_rl"]["hard_call_token_cap"])
    max_cost = _max_incremental_cost(development)
    if max_cost > hard_cap:
        raise ValueError(
            f"Observed development call cost {max_cost} exceeds configured hard reserve {hard_cap}"
        )
    integrity = {
        "schema_version": "darc-qwen38-retrain-integrity-v1",
        "source_tasks": len(all_groups),
        "development_tasks": len(development),
        "retrospective_heldout_tasks": len(heldout),
        "prospective_selected_tasks": len(sealed_test),
        "prospective_checkpoint_files": (
            len(
                list(
                    (
                        Path(config["data"]["prospective_test_root"])
                        / "task_checkpoints"
                    ).glob("*.json")
                )
            )
            if config["data"].get("prospective_test_root")
            else 0
        ),
        "development_heldout_overlap": len(set(development) & set(heldout)),
        "source_prospective_overlap": len(set(all_groups) & sealed_test),
        "trajectories_per_task": 8,
        "states_per_task": 7,
        "reference_budget_per_task": reference_budget,
        "max_development_incremental_call_tokens": max_cost,
        "hard_call_token_cap": hard_cap,
        "frozen_test_loaded": False,
    }
    integrity_path = output / "split_integrity.json"
    integrity_path.write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    if mode == "dry-run":
        return integrity_path

    states, lookup = build_states(development)
    fractions = [float(value) for value in config["offline_rl"]["budget_fractions"]]
    allowances = [fraction * reference_budget for fraction in fractions]

    def skip(_state: Any, _allowance: float) -> Any:
        return lambda _observation: "S"

    targets = generate_policy_targets(
        states,
        development,
        lookup,
        allowances=allowances,
        reference_budget=reference_budget,
        hard_call_token_cap=hard_cap,
        continuation_factory=skip,
        target_mode="myopic",
    )
    target_audit = validate_target_coverage(targets, name="qwen38-development-myopic")
    (output / "target_coverage.json").write_text(
        json.dumps(target_audit, indent=2) + "\n", encoding="utf-8"
    )
    if mode == "smoke":
        smoke = output / "smoke_report.json"
        smoke.write_text(
            json.dumps({**integrity, "target_coverage": target_audit, "passed": True}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return smoke

    start = time.time()
    device = str(config["semantic_encoder"]["device"])
    development_cache_split = str(
        config["semantic_encoder"].get(
            "development_cache_split", f"darc_qwen38_development_{len(development)}"
        )
    )
    heldout_cache_split = str(
        config["semantic_encoder"].get(
            "heldout_cache_split", f"darc_qwen38_retrospective_heldout_{len(heldout)}"
        )
    )
    encoded, train_cache = encode_or_load(
        [state.encoder_example for state in states],
        split=development_cache_split,
        config=config["semantic_encoder"],
        cache_dir=Path(config["semantic_encoder"]["cache_dir"]),
    )
    writer = _writer(output / "tensorboard" / "concise", allow_missing=False)
    seeds = [int(value) for value in config["reproducibility"]["seeds"]]
    policy_template = config["policy_template"]
    policy_variant = {
        "conservative": bool(policy_template["conservative"]),
        "dynamic_token_price": bool(policy_template["dynamic_token_price"]),
        "uncertainty_penalty": bool(policy_template["uncertainty_penalty"]),
    }
    policy_candidates = policy_grid(config["policy_grid"], policy_variant)
    oof_records: list[dict[str, Any]] = []
    heldout_records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    selected_policies: dict[tuple[str, int], ConservativePolicy] = {}
    trained_models: dict[tuple[str, int], DARCv05Ensemble] = {}

    for variant in tqdm(config["variants"], desc="Qwen3.8 DARC variants", unit="variant"):
        name = str(variant["name"])
        label = str(variant["display_name"])
        for seed in tqdm(seeds, desc=name, unit="seed", leave=False):
            checkpoint = output / "checkpoints" / name / f"seed_{seed}" / "iteration_0.pt"
            if checkpoint.exists():
                if not resume:
                    raise FileExistsError(f"{checkpoint} exists; use --resume")
                model = DARCv05Ensemble.load(checkpoint)
            else:
                initialization = None
                if variant["initialization"] == "qwen25_v04":
                    initialization = DARCv04Ensemble.load(
                        Path(config["data"]["v04_initialization_root"])
                        / "models" / f"seed_{seed}" / "myopic_targets.pt"
                    )
                model = train_crossfit_iteration(
                    states,
                    encoded,
                    targets,
                    states,
                    encoded,
                    targets,
                    task_folds=folds,
                    folds=int(config["reproducibility"]["inner_folds"]),
                    members_per_fold=int(config["reproducibility"]["members_per_fold"]),
                    iteration=ITERATION,
                    params=config["model"],
                    reference_budget=reference_budget,
                    v04_initialization=initialization,
                    previous=None,
                    seed=seed,
                    deterministic=bool(config["reproducibility"]["deterministic_torch"]),
                    device_name=device,
                    writer=writer,
                    variant=name,
                    progress=config["monitoring"],
                    validation_task_folds=folds,
                )
                _atomic_save(model, checkpoint)
            for member in model.members:
                training_rows.extend(dict(row) for row in member.history)
            oof = oof_predictions(model, encoded, states, allowances, folds, device=device)
            selected, grid = _tune_v05_policy(
                development,
                lookup,
                oof,
                allowances,
                policy_candidates,
                fractions=fractions,
                reference_budget=reference_budget,
                hard_cap=hard_cap,
                seed=seed,
                label=label,
                token_penalty=float(config["selection"]["token_utilization_penalty"]),
                max_harmful_rate=float(config["selection"]["max_harmful_rate"]),
            )
            for row in grid:
                row["model_variant"] = name
                row["selection_source"] = "development_oof_only"
            policy_rows.extend(grid)
            selected_policies[(name, seed)] = selected
            trained_models[(name, seed)] = model
            _evaluate_v05(
                oof_records,
                traces,
                groups=development,
                lookup=lookup,
                predictions=oof,
                allowances=allowances,
                policy=selected,
                fractions=fractions,
                reference_budget=reference_budget,
                hard_cap=hard_cap,
                seed=seed,
                label=f"{label} · OOF",
                iteration=ITERATION,
                writer=writer,
            )
            status = output / "status" / f"{name}_seed_{seed}.json"
            status.write_text(
                json.dumps(
                    {"completed": True, "checkpoint": str(checkpoint), "policy": asdict(selected)},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    # Select between scratch and transfer using only the OOF development records.
    variant_scores: list[dict[str, Any]] = []
    for variant in config["variants"]:
        label = f"{variant['display_name']} · OOF"
        summary = summarize_oof_variant(
            oof_records,
            label=label,
            expected_seeds=len(seeds),
            reference_budget=reference_budget,
            task_count=len(development),
            token_penalty=float(config["selection"]["token_utilization_penalty"]),
        )
        variant_scores.append(
            {
                "variant": variant["name"],
                **summary,
            }
        )
    selected_variant = max(
        variant_scores,
        key=lambda row: (row["score"], row["accuracy"], -row["tokens"]),
    )["variant"]
    for row in variant_scores:
        row["selected_before_heldout"] = int(row["variant"] == selected_variant)

    # Only now, after every model/policy choice is frozen from development OOF,
    # construct features and open outcomes for the 28-task internal holdout.
    heldout_max_cost = _max_incremental_cost(heldout)
    if heldout_max_cost > hard_cap:
        raise ValueError(
            "Held-out incremental call cost exceeds the predeclared hard reserve: "
            f"{heldout_max_cost} > {hard_cap}. Do not tune the cap from outcomes; "
            "rerun with a protocol-level upper bound justified by the serving config."
        )
    test_states, test_lookup = build_states(heldout)
    test_encoded, test_cache = encode_or_load(
        [state.encoder_example for state in test_states],
        split=heldout_cache_split,
        config=config["semantic_encoder"],
        cache_dir=Path(config["semantic_encoder"]["cache_dir"]),
    )
    for variant in config["variants"]:
        name = str(variant["name"])
        label = str(variant["display_name"])
        for seed in seeds:
            test_predictions = _precompute_v05(
                trained_models[(name, seed)],
                test_encoded,
                test_states,
                allowances,
                device=device,
                description=f"Held-out {name} s{seed}",
            )
            _evaluate_v05(
                heldout_records,
                traces,
                groups=heldout,
                lookup=test_lookup,
                predictions=test_predictions,
                allowances=allowances,
                policy=selected_policies[(name, seed)],
                fractions=fractions,
                reference_budget=reference_budget,
                hard_cap=hard_cap,
                seed=seed,
                label=label,
                iteration=ITERATION,
                writer=writer,
            )

    # Strong transparent held-out baselines, replayed under the same ledger.
    for seed in seeds:
        fixed_policies = (
            ("SSS", "Never Verify"),
            ("VSS", "Always CP1"),
            ("SVS", "Always CP2"),
            ("SSV", "Always CP3"),
            ("VVV", "Always VVV"),
        )
        for sequence, label in fixed_policies:
            for fraction in fractions:
                replay, fixed_traces = _fixed_replay(
                    heldout,
                    test_lookup,
                    sequence=sequence,
                    token_budget=int(round(fraction * reference_budget * len(heldout))),
                    budget_fraction=fraction,
                    hard_cap=hard_cap,
                    seed=seed,
                    label=label,
                )
                _record(heldout_records, replay, fixed_traces, seed=seed)
                traces.extend(fixed_traces)

    aggregate = aggregate_records(
        heldout_records,
        bootstrap=int(config["reproducibility"]["bootstrap_repetitions"]),
        seed=seeds[0],
    )
    _write_csv(output / "tables" / "training_history.csv", training_rows)
    _write_csv(output / "tables" / "policy_grid.csv", policy_rows)
    _write_csv(output / "tables" / "variant_selection_oof.csv", variant_scores)
    _write_csv(output / "tables" / "oof_results.csv", [row["summary"] for row in oof_records])
    _write_csv(
        output / "tables" / "heldout_results.csv",
        [row["summary"] for row in heldout_records],
    )
    _write_csv(output / "tables" / "heldout_aggregate.csv", aggregate)
    _write_csv(output / "tables" / "decision_trace.csv", traces)
    _figure(output, aggregate)
    manifest = {
        "schema_version": "darc-qwen38-myopic-retrain-run-v1",
        "source_tasks": len(all_groups),
        "development_tasks": len(development),
        "retrospective_heldout_tasks": len(heldout),
        "prospective_tasks_accessed": 0,
        "prospective_tasks_configured": len(sealed_test),
        "selected_variant_from_oof": selected_variant,
        "reference_budget_per_task": reference_budget,
        "seeds": seeds,
        "outer_folds": int(config["reproducibility"]["folds"]),
        "heldout_fold": heldout_fold,
        "inner_folds": int(config["reproducibility"]["inner_folds"]),
        "frozen_test_loaded": False,
        "embedding_cache": {"development": train_cache, "heldout": test_cache},
        "elapsed_seconds": time.time() - start,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    writer.close()
    return _report(
        output,
        aggregate,
        reference_budget=reference_budget,
        selected_variant=selected_variant,
        source_tasks=len(all_groups),
        development_tasks=len(development),
        heldout_tasks=len(heldout),
        prospective_tasks=len(sealed_test),
        report_filename=str(
            config["output"].get(
                "report_filename", "darc_qwen38_myopic_retrain_report.md"
            )
        ),
    )

