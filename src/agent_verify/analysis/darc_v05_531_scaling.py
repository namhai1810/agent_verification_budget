"""Validation-only DARC v0.5 Myopic-i0 scaling experiment.

This module intentionally reuses the v0.5 model, target builder, optimizer,
policy grid, and strict replay implementation.  The only experimental change
is the Level-2 training cohort: 381 original tasks become 531 tasks after the
outcome-blind BigCodeBench extension.  Validation remains the frozen 98-task
development partition; the frozen test partition is never opened.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm

from agent_verify.analysis.cbv_v01_experiment import MultiHistoryController
from agent_verify.analysis.darc_data import encode_or_load
from agent_verify.analysis.darc_v04_audit import (
    aggregate_records,
    paired_comparison,
    replay_c3_strict,
)
from agent_verify.analysis.darc_v04_data import audit_level2, build_states
from agent_verify.analysis.darc_v04_experiment import _assert_disjoint, _load_groups, _write_csv
from agent_verify.analysis.darc_v04_model import DARCv04Ensemble
from agent_verify.analysis.darc_v05_experiment import (
    _append_target_diagnostics,
    _evaluate_v05,
    _precompute_v05,
    _record,
    _tune_v05_policy,
    _writer,
    replay_v05,
)
from agent_verify.analysis.darc_v05_model import DARCv05Ensemble
from agent_verify.analysis.darc_v05_training import (
    ConservativePolicy,
    assign_task_folds,
    generate_policy_targets,
    policy_grid,
    train_crossfit_iteration,
    validate_target_coverage,
)
from agent_verify.analysis.darc_v052_synergy_audit import load_expanded_train_groups


NEW_LABEL = "DARC v0.5 Myopic i0 (531 train)"
ORIGINAL_LABEL = "DARC v0.5 Myopic i0 (381 train)"
C3_LABEL = "C3 MultiHistory Myopic (381 train)"
VARIANT = "myopic_targets"
ITERATION = 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def selected_original_policy(root: Path, seed: int) -> ConservativePolicy:
    """Recover the predeclared i0 policy chosen in the original 381-task run."""

    expected = "DARC v0.5 · myopic_targets · i0"
    rows = [
        row
        for row in _read_csv(root / "tables" / "policy_grid.csv")
        if int(row["seed"]) == seed
        and row["variant"] == expected
        and int(row["iteration"]) == ITERATION
        and _truthy(row["selected"])
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one selected original policy for seed {seed}, found {len(rows)}")
    row = rows[0]
    fields = ConservativePolicy.__dataclass_fields__
    values: dict[str, Any] = {}
    for key in fields:
        values[key] = _truthy(row[key]) if key in {
            "conservative", "dynamic_token_price", "uncertainty_penalty"
        } else float(row[key])
    return ConservativePolicy(**values)


def _metadata_signature(row: dict[str, Any]) -> dict[str, Any]:
    """Metadata that must match even when collection cache fingerprints differ."""

    return {
        "schema_version": row.get("schema_version"),
        "dataset_version": row.get("dataset_version"),
        "workflow_version": row.get("workflow_version"),
        "model_pair_id": row.get("model_pair_id"),
        "model_pair_fingerprint": row.get("model_pair_fingerprint"),
        "prompt_versions": row.get("prompt_versions"),
    }


def audit_split_and_compatibility(
    original: dict[str, Any],
    extension: dict[str, Any],
    validation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    combined = {**original, **extension}
    overlaps = {
        "original_extension": sorted(set(original) & set(extension)),
        "train_validation": sorted(set(combined) & set(validation)),
    }
    if overlaps["original_extension"] or overlaps["train_validation"]:
        raise ValueError(f"Task leakage detected: {overlaps}")
    if (len(original), len(extension), len(combined), len(validation)) != (381, 150, 531, 98):
        raise ValueError(
            "Expected original/extension/train/validation counts 381/150/531/98, "
            f"found {len(original)}/{len(extension)}/{len(combined)}/{len(validation)}"
        )

    original_bcb = [
        group for group in original.values() if str(group.task.dataset).lower() == "bigcodebench"
    ]
    if not original_bcb:
        raise ValueError("No original BigCodeBench Level-2 tasks available for compatibility audit")
    reference_signatures = {
        json.dumps(_metadata_signature(row), sort_keys=True)
        for group in original_bcb
        for row in group.rows.values()
    }
    extension_signatures = {
        json.dumps(_metadata_signature(row), sort_keys=True)
        for group in extension.values()
        for row in group.rows.values()
    }
    compatible = len(reference_signatures) == 1 and extension_signatures == reference_signatures
    if not compatible:
        raise ValueError("Extension model/prompt/workflow metadata is incompatible with original BigCodeBench")

    rows = audit_level2({"train": combined, "validation": validation})
    payload = {
        "schema_version": "darc-v0.5-531-split-audit-v1",
        "counts": {
            "original_train": len(original),
            "extension_train": len(extension),
            "combined_train": len(combined),
            "validation": len(validation),
        },
        "overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "extension_all_bigcodebench": all(
            str(group.task.dataset).lower() == "bigcodebench" for group in extension.values()
        ),
        "metadata_signature_match": compatible,
        "metadata_signature": json.loads(next(iter(reference_signatures))),
        "frozen_test_loaded": False,
        "all_checks_pass": compatible and not any(overlaps.values()),
    }
    return rows, payload


def pareto_rows(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark globally non-dominated accuracy/cost points."""

    rows: list[dict[str, Any]] = []
    for row in aggregate:
        cost = float(row["teacher_tokens_mean"])
        accuracy = float(row["accuracy_mean"])
        dominated = any(
            float(other["teacher_tokens_mean"]) <= cost
            and float(other["accuracy_mean"]) >= accuracy
            and (
                float(other["teacher_tokens_mean"]) < cost
                or float(other["accuracy_mean"]) > accuracy
            )
            for other in aggregate
        )
        rows.append({**row, "pareto_optimal": int(not dominated)})
    return rows


def _atomic_save(model: DARCv05Ensemble, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    model.save(temporary)
    temporary.replace(path)


def _original_records(
    groups: dict[str, Any],
    states: list[Any],
    lookup: dict[tuple[str, str], int],
    encoded: Any,
    *,
    config: dict[str, Any],
    allowances: list[float],
    reference_budget: float,
    fractions: list[float],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    traces_out: list[dict[str, Any]] = []
    root = Path(config["data"]["original_v05_results"])
    hard_cap = int(config["offline_rl"]["hard_call_token_cap"])
    device = str(config["semantic_encoder"]["device"])
    for seed in tqdm(seeds, desc="Replay original DARC-381", unit="seed"):
        model = DARCv05Ensemble.load(
            root / "checkpoints" / VARIANT / f"seed_{seed}" / "iteration_0.pt"
        )
        if not np.isclose(model.reference_budget, reference_budget):
            raise ValueError(
                f"Original seed {seed} uses reference budget {model.reference_budget}, "
                f"expected {reference_budget}"
            )
        predictions = _precompute_v05(
            model,
            encoded,
            states,
            allowances,
            device=device,
            description=f"Replay DARC-381 s{seed}",
        )
        policy = selected_original_policy(root, seed)
        for fraction in fractions:
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
                label=ORIGINAL_LABEL,
            )
            _record(records, replay, traces, seed=seed)
            traces_out.extend(traces)
    return records, traces_out


def _c3_records(
    groups: dict[str, Any],
    *,
    config: dict[str, Any],
    reference_budget: float,
    fractions: list[float],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    traces_out: list[dict[str, Any]] = []
    controller: MultiHistoryController = joblib.load(
        Path(config["data"]["c3_results"])
        / "models" / "combined" / "multi_myopic.joblib"
    )
    hard_cap = int(config["offline_rl"]["hard_call_token_cap"])
    for seed in tqdm(seeds, desc="Replay C3 Myopic", unit="seed"):
        for fraction in fractions:
            replay, traces = replay_c3_strict(
                groups,
                controller,
                token_budget=int(round(fraction * reference_budget * len(groups))),
                budget_fraction=fraction,
                hard_call_token_cap=hard_cap,
                seed=seed,
                label=C3_LABEL,
            )
            _record(records, replay, traces, seed=seed)
            traces_out.extend(traces)
    return records, traces_out


def _figures(output: Path, aggregate: list[dict[str, Any]], training: list[dict[str, Any]]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {NEW_LABEL: "#1769AA", ORIGINAL_LABEL: "#8A94A6", C3_LABEL: "#E07A3F"}
    fig, axis = plt.subplots(figsize=(8.6, 5.2))
    for policy in (NEW_LABEL, ORIGINAL_LABEL, C3_LABEL):
        points = sorted(
            [row for row in aggregate if row["policy"] == policy],
            key=lambda row: float(row["teacher_tokens_mean"]),
        )
        axis.plot(
            [float(row["teacher_tokens_mean"]) / 1000 for row in points],
            [100 * float(row["accuracy_mean"]) for row in points],
            marker="o",
            linewidth=2.2 if policy == NEW_LABEL else 1.5,
            color=colors[policy],
            label=policy,
        )
    axis.set(
        title="DARC v0.5 scaling: matched-budget validation frontier",
        xlabel="Actual Teacher tokens (thousands)",
        ylabel="Final accuracy (%)",
    )
    axis.grid(color="#E3E7ED", linewidth=0.8)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "accuracy_cost_pareto.png", dpi=240, bbox_inches="tight")
    fig.savefig(figures / "accuracy_cost_pareto.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.6, 5.2))
    for seed in sorted({int(row["seed"]) for row in training}):
        rows = [
            row for row in training
            if int(row["seed"]) == seed and int(row["fold"]) == 0 and int(row["member"]) == 0
        ]
        axis.plot(
            [int(row["epoch"]) for row in rows],
            [float(row["validation_total_loss"]) for row in rows],
            label=str(seed),
        )
    axis.set(title="DARC-531 validation loss", xlabel="Epoch", ylabel="Validation loss")
    axis.grid(color="#E3E7ED", linewidth=0.8)
    axis.legend(title="Seed", frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "training_convergence.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _report(
    output: Path,
    aggregate: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    split_audit: dict[str, Any],
) -> Path:
    full = [row for row in aggregate if float(row["budget_fraction"]) == 1.0]
    lines = [
        "# DARC v0.5 Myopic Iteration 0 — 531-task scaling experiment",
        "",
        "> Validation-only development experiment. Frozen test was not loaded; no Student/Teacher inference was run.",
        "",
        "## Controlled change",
        "",
        "The architecture, v0.4 initialization, Myopic iteration-0 targets, optimizer, policy grid, three seeds, and strict replay are unchanged. Only training Level-2 tasks increase from **381 to 531**. The original **98 validation tasks** and reference budget **4,912 tokens/task** remain fixed.",
        "",
        "## Integrity audit",
        "",
        f"- Train: {split_audit['counts']['original_train']} original + {split_audit['counts']['extension_train']} extension = {split_audit['counts']['combined_train']} tasks.",
        f"- Validation: {split_audit['counts']['validation']} unchanged tasks.",
        f"- Train/validation overlap: {split_audit['overlap_counts']['train_validation']}.",
        f"- Model/prompt/workflow metadata compatible: **{split_audit['metadata_signature_match']}**.",
        "- Frozen test loaded: **False**.",
        "",
        "## Full-budget results",
        "",
        "| Policy | Accuracy mean (95% task CI) | Actual tokens | Calls | Helpful / Harmful / No-change | Seed variance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(full, key=lambda item: float(item["accuracy_mean"]), reverse=True):
        lines.append(
            f"| {row['policy']} | {100*float(row['accuracy_mean']):.2f}% "
            f"[{100*float(row['accuracy_ci95_low']):.2f}, {100*float(row['accuracy_ci95_high']):.2f}] | "
            f"{float(row['teacher_tokens_mean']):,.0f} | {float(row['teacher_calls_mean']):.1f} | "
            f"{float(row['helpful_mean']):.1f} / {float(row['harmful_mean']):.1f} / "
            f"{float(row['no_change_mean']):.1f} | {float(row['accuracy_variance']):.6f} |"
        )
    lines.extend([
        "",
        "## Paired matched-budget comparisons",
        "",
        "| Budget | Candidate − reference | Accuracy Δ pp (95% CI) | p | Token Δ |",
        "|---:|---|---:|---:|---:|",
    ])
    for row in comparisons:
        lines.append(
            f"| {100*float(row['budget_fraction']):.0f}% | {row['candidate']} − {row['reference']} | "
            f"{float(row['accuracy_difference_pp']):+.2f} "
            f"[{float(row['difference_ci95_low_pp']):+.2f}, {float(row['difference_ci95_high_pp']):+.2f}] | "
            f"{float(row['paired_permutation_p']):.4f} | {float(row['token_difference_mean']):+,.0f} |"
        )
    lines.extend([
        "",
        "![Matched-budget Pareto frontier](../figures/accuracy_cost_pareto.png)",
        "",
        "## Interpretation boundary",
        "",
        "This report measures scaling on the unchanged development validation set. It does not constitute frozen-test confirmation. Policy margins are selected under the same legacy validation protocol, so the comparison isolates data scaling but remains development evidence.",
        "",
        "## Artifacts",
        "",
        "- `checkpoints/myopic_targets/seed_<seed>/iteration_0.pt`",
        "- `tables/seed_results.csv`, `aggregate_results.csv`, `pareto_points.csv`",
        "- `tables/decision_trace.csv`, `paired_comparisons.csv`, `training_history.csv`",
        "- `split_integrity.json`, `run_manifest.json`, `figures/`, `tensorboard/`",
    ])
    path = output / "report" / "darc_v05_531_scaling_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(config_path: Path, *, smoke: bool = False, resume: bool = False) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["evaluation"]["partition"] != "validation":
        raise ValueError("Scaling experiment must remain validation-only")
    if config["evaluation"]["frozen_test_access"] != "forbidden":
        raise ValueError("Frozen test access must be forbidden")
    if [int(value) for value in config["reproducibility"]["seeds"]] != [
        20261031, 20261101, 20261117
    ]:
        raise ValueError("Scaling experiment must retain the original three seeds")
    if int(config["offline_rl"]["policy_iterations"]) != 1:
        raise ValueError("Scaling experiment is restricted to Myopic iteration 0")

    output = Path(config["output"]["root"])
    for child in ("logs", "checkpoints", "tables", "figures", "report", "tensorboard", "status"):
        (output / child).mkdir(parents=True, exist_ok=True)
    frozen_root = Path(config["data"]["frozen_root"])
    original, extension = load_expanded_train_groups(
        frozen_root,
        Path(config["data"]["extension_root"]),
        Path(config["data"]["extension_selection_manifest"]),
    )
    validation_groups = _load_groups(frozen_root, "validation")
    audit_rows, split_audit = audit_split_and_compatibility(original, extension, validation_groups)
    (output / "split_integrity.json").write_text(
        json.dumps(split_audit, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output / "tables" / "data_audit.csv", audit_rows)
    if smoke:
        payload = {
            "schema_version": "darc-v0.5-531-smoke-v1",
            "passed": True,
            "checks": [
                "531 train tasks loaded",
                "98 validation tasks loaded",
                "zero task overlap",
                "model/prompt/workflow metadata compatible",
                "frozen test not loaded",
            ],
        }
        path = output / "smoke" / "smoke_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    start = time.time()
    train_groups = {**original, **extension}
    _assert_disjoint(train_groups, validation_groups)
    train_states, train_lookup = build_states(train_groups)
    validation_states, validation_lookup = build_states(validation_groups)
    cache = Path(config["semantic_encoder"]["cache_dir"])
    train_encoded, train_cache = encode_or_load(
        [state.encoder_example for state in train_states],
        split="darc_v05_531_train",
        config=config["semantic_encoder"],
        cache_dir=cache,
    )
    validation_encoded, validation_cache = encode_or_load(
        [state.encoder_example for state in validation_states],
        split="darc_v05_531_validation",
        config=config["semantic_encoder"],
        cache_dir=cache,
    )
    reference_budget = float(config["offline_rl"]["fixed_reference_budget_per_task"])
    if not np.isclose(reference_budget, 4912.0):
        raise ValueError("Matched comparison requires the original 4912-token reference budget")
    fractions = [float(value) for value in config["offline_rl"]["budget_fractions"]]
    allowances = [fraction * reference_budget for fraction in fractions]
    hard_cap = int(config["offline_rl"]["hard_call_token_cap"])
    seeds = [int(value) for value in config["reproducibility"]["seeds"]]
    folds = int(config["offline_rl"]["crossfit_folds"])
    device = str(config["semantic_encoder"]["device"])
    writer = _writer(output / "tensorboard" / "concise", allow_missing=False)
    variant = config["variants"][0]
    if variant["name"] != VARIANT or variant["target_mode"] != "myopic":
        raise ValueError("Scaling experiment must train only unchanged myopic_targets")

    training_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    traces_out: list[dict[str, Any]] = []
    v04_root = Path(config["data"]["darc_v04_audit"])
    folds_by_task = assign_task_folds(
        list(train_groups), folds, int(config["reproducibility"]["crossfit_seed"])
    )
    skip_continuation = lambda _state, _allowance: (lambda _observation: "S")

    for seed in tqdm(seeds, desc="DARC-531 Myopic i0", unit="seed"):
        train_targets = generate_policy_targets(
            train_states,
            train_groups,
            train_lookup,
            allowances=allowances,
            reference_budget=reference_budget,
            hard_call_token_cap=hard_cap,
            continuation_factory=skip_continuation,
            target_mode="myopic",
        )
        validation_targets = generate_policy_targets(
            validation_states,
            validation_groups,
            validation_lookup,
            allowances=allowances,
            reference_budget=reference_budget,
            hard_call_token_cap=hard_cap,
            continuation_factory=skip_continuation,
            target_mode="myopic",
        )
        for split, targets in (("train", train_targets), ("validation", validation_targets)):
            diagnostics = validate_target_coverage(
                targets, name=f"{split}/{VARIANT}/seed_{seed}/iteration_0"
            )
            _append_target_diagnostics(
                target_rows,
                diagnostics,
                split=split,
                variant=VARIANT,
                seed=seed,
                iteration=ITERATION,
            )
        checkpoint = output / "checkpoints" / VARIANT / f"seed_{seed}" / "iteration_0.pt"
        if checkpoint.exists():
            if not resume:
                raise FileExistsError(
                    f"Checkpoint already exists: {checkpoint}. Use --resume to reuse it safely."
                )
            model = DARCv05Ensemble.load(checkpoint)
        else:
            initial = DARCv04Ensemble.load(
                v04_root / "models" / f"seed_{seed}" / "myopic_targets.pt"
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
                iteration=ITERATION,
                params=config["model"],
                reference_budget=reference_budget,
                v04_initialization=initial,
                previous=None,
                seed=seed,
                deterministic=bool(config["reproducibility"]["deterministic_torch"]),
                device_name=device,
                writer=writer,
                variant=VARIANT,
                progress=config["monitoring"],
            )
            _atomic_save(model, checkpoint)
        for member in model.members:
            training_rows.extend(dict(row) for row in member.history)
        predictions = _precompute_v05(
            model,
            validation_encoded,
            validation_states,
            allowances,
            device=device,
            description=f"Predict DARC-531 s{seed}",
        )
        selected_policy, grid = _tune_v05_policy(
            validation_groups,
            validation_lookup,
            predictions,
            allowances,
            policy_grid(config["policy_grid"], variant),
            fractions=fractions,
            reference_budget=reference_budget,
            hard_cap=hard_cap,
            seed=seed,
            label=NEW_LABEL,
            token_penalty=float(config["selection"]["token_utilization_penalty"]),
            max_harmful_rate=float(config["selection"]["max_harmful_rate"]),
        )
        for row in grid:
            row["iteration"] = ITERATION
        policy_rows.extend(grid)
        _evaluate_v05(
            records,
            traces_out,
            groups=validation_groups,
            lookup=validation_lookup,
            predictions=predictions,
            allowances=allowances,
            policy=selected_policy,
            fractions=fractions,
            reference_budget=reference_budget,
            hard_cap=hard_cap,
            seed=seed,
            label=NEW_LABEL,
            iteration=ITERATION,
            writer=writer,
        )
        status = {
            "seed": seed,
            "checkpoint": str(checkpoint),
            "selected_policy": asdict(selected_policy),
            "completed": True,
        }
        (output / "status" / f"seed_{seed}.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )

    original_records, original_traces = _original_records(
        validation_groups,
        validation_states,
        validation_lookup,
        validation_encoded,
        config=config,
        allowances=allowances,
        reference_budget=reference_budget,
        fractions=fractions,
        seeds=seeds,
    )
    c3_records, c3_traces = _c3_records(
        validation_groups,
        config=config,
        reference_budget=reference_budget,
        fractions=fractions,
        seeds=seeds,
    )
    records.extend(original_records)
    records.extend(c3_records)
    traces_out.extend(original_traces)
    traces_out.extend(c3_traces)
    aggregate = aggregate_records(
        records,
        bootstrap=int(config["reproducibility"]["bootstrap_repetitions"]),
        seed=seeds[0],
    )
    comparisons: list[dict[str, Any]] = []
    for fraction in fractions:
        for reference in (ORIGINAL_LABEL, C3_LABEL):
            comparisons.append(
                paired_comparison(
                    records,
                    candidate=NEW_LABEL,
                    reference=reference,
                    fraction=fraction,
                    bootstrap=int(config["reproducibility"]["bootstrap_repetitions"]),
                    permutations=int(config["reproducibility"]["permutation_repetitions"]),
                    seed=seeds[0],
                )
            )

    _write_csv(output / "tables" / "training_history.csv", training_rows)
    _write_csv(output / "tables" / "target_coverage.csv", target_rows)
    _write_csv(output / "tables" / "policy_grid.csv", policy_rows)
    _write_csv(output / "tables" / "seed_results.csv", [row["summary"] for row in records])
    _write_csv(output / "tables" / "aggregate_results.csv", aggregate)
    _write_csv(output / "tables" / "pareto_points.csv", pareto_rows(aggregate))
    _write_csv(output / "tables" / "paired_comparisons.csv", comparisons)
    _write_csv(output / "tables" / "decision_trace.csv", traces_out)
    _figures(output, aggregate, training_rows)
    manifest = {
        "schema_version": "darc-v0.5-531-scaling-run-v1",
        "algorithm_change": False,
        "variant": VARIANT,
        "policy_iteration": ITERATION,
        "train_tasks": len(train_groups),
        "validation_tasks": len(validation_groups),
        "frozen_test_loaded": False,
        "seeds": seeds,
        "budget_fractions": fractions,
        "reference_budget_per_task": reference_budget,
        "hard_call_token_cap": hard_cap,
        "resumed": resume,
        "embedding_cache": {"train": train_cache, "validation": validation_cache},
        "elapsed_seconds": time.time() - start,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    report = _report(output, aggregate, comparisons, split_audit)
    writer.close()
    return report
