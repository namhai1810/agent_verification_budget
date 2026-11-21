"""End-to-end Full-Tree CV, strict replay, reporting, and final fitting."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm

from agent_verify.analysis.darc_data import EncodedDARC, encode_or_load
from agent_verify.analysis.darc_fulltree_data import (
    PREFIX_INDEX,
    SEQUENCES,
    FullTreeDataset,
    assert_disjoint_folds,
    build_fulltree_dataset,
)
from agent_verify.analysis.darc_fulltree_model import (
    FullTreePolicyCheckpoint,
    base_arrays,
    policy_batch,
)
from agent_verify.analysis.darc_fulltree_objective import trajectory_probabilities
from agent_verify.analysis.darc_fulltree_training import train_fulltree_policy
from agent_verify.analysis.darc_model import _torch
from agent_verify.analysis.darc_qwen38_data import (
    assign_stratified_task_folds,
    load_qwen38_full8_groups,
    split_audit_rows,
)
from agent_verify.analysis.darc_v04_data import build_states

POLICY_NAME = {
    "cp3_anchor": "DARC Full-Tree · CP3 anchor",
    "no_anchor": "DARC Full-Tree · no anchor",
}
FIXED = {
    "Never Verify (SSS)": "SSS",
    "Always CP1": "VSS",
    "Always CP2": "SVS",
    "Always CP3": "SSV",
    "SVV": "SVV",
    "VVV": "VVV",
}


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stable_order(keys: tuple[str, ...], seed: int) -> list[str]:
    return sorted(keys, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())


def _load(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[Any], Any, FullTreeDataset]:
    groups, metadata = load_qwen38_full8_groups(
        [Path(value) for value in config["data"]["full8_roots"]],
        frozen_root=Path(config["data"]["frozen_root"]),
        expected_tasks=int(config["data"]["expected_tasks"]),
        catalog_splits=("train",),
    )
    states, lookup = build_states(groups)
    tree = build_fulltree_dataset(groups, states, lookup)
    return groups, metadata, states, lookup, tree


def _reference_budget(tree: FullTreeDataset) -> float:
    return float(tree.trajectory_tokens[:, SEQUENCES.index("VVV")].mean())


def _max_edge_cost(tree: FullTreeDataset) -> int:
    return int(tree.edge_tokens.max())


def _rows_for_keys(tree: FullTreeDataset, keys: set[str]) -> np.ndarray:
    return np.asarray([index for index, key in enumerate(tree.task_keys) if key in keys], dtype=np.int64)


def _probabilities(
    checkpoint: FullTreePolicyCheckpoint,
    encoded: EncodedDARC,
    states: list[Any],
    tree: FullTreeDataset,
    allowance: float,
    device_name: str,
) -> np.ndarray:
    torch = _torch()
    network, device = checkpoint.instantiate(device_name)
    arrays = base_arrays(encoded, states, checkpoint.metadata_mean, checkpoint.metadata_scale)
    remaining = tree.remaining_allowance(allowance)
    batch = policy_batch(
        arrays,
        tree.state_indices.reshape(-1),
        remaining.reshape(-1),
        reference_budget=checkpoint.reference_budget,
        device=device,
    )
    with torch.inference_mode():
        probability = torch.sigmoid(network(batch)).reshape(len(tree.task_keys), 7)
    return probability.cpu().numpy()


def _classify_calls(sequence: str, outcomes: np.ndarray) -> tuple[int, int, int]:
    helpful = harmful = no_change = 0
    actual = int(outcomes[SEQUENCES.index(sequence)])
    for checkpoint, action in enumerate(sequence):
        if action != "V":
            continue
        alternative = sequence[:checkpoint] + "S" + sequence[checkpoint + 1 :]
        delta = actual - int(outcomes[SEQUENCES.index(alternative)])
        helpful += delta > 0
        harmful += delta < 0
        no_change += delta == 0
    return helpful, harmful, no_change


def _task_result(
    *,
    policy: str,
    variant: str,
    seed: int,
    fraction: float,
    task_key: str,
    dataset: str,
    sequence: str,
    tree: FullTreeDataset,
    row: int,
    fold: int | str,
) -> dict[str, Any]:
    outcome = int(tree.outcomes[row, SEQUENCES.index(sequence)])
    tokens = int(tree.trajectory_tokens[row, SEQUENCES.index(sequence)])
    helpful, harmful, no_change = _classify_calls(sequence, tree.outcomes[row])
    return {
        "policy": policy,
        "variant": variant,
        "seed": seed,
        "budget_fraction": fraction,
        "task_key": task_key,
        "dataset": dataset,
        "fold": fold,
        "sequence": sequence,
        "correct": outcome,
        "teacher_tokens": tokens,
        "teacher_calls": sequence.count("V"),
        "cp1_calls": int(sequence[0] == "V"),
        "cp2_calls": int(sequence[1] == "V"),
        "cp3_calls": int(sequence[2] == "V"),
        "helpful_calls": helpful,
        "harmful_calls": harmful,
        "no_change_calls": no_change,
    }


def _expected_rows(
    *,
    checkpoint: FullTreePolicyCheckpoint,
    encoded: EncodedDARC,
    states: list[Any],
    tree: FullTreeDataset,
    fractions: list[float],
    device_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    torch = _torch()
    metrics: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for fraction in fractions:
        allowance = fraction * checkpoint.reference_budget
        verify = _probabilities(checkpoint, encoded, states, tree, allowance, device_name)
        paths = trajectory_probabilities(torch.as_tensor(verify, dtype=torch.float32)).numpy()
        if not np.allclose(paths.sum(axis=1), 1.0, atol=1e-6):
            raise AssertionError("Full-Tree trajectory probabilities do not sum to one")
        expected_correct = (paths * tree.outcomes).sum(axis=1)
        expected_tokens = (paths * tree.trajectory_tokens).sum(axis=1)
        call_count = np.asarray([sequence.count("V") for sequence in SEQUENCES])
        expected_calls = (paths * call_count[None, :]).sum(axis=1)
        for row, task_key in enumerate(tree.task_keys):
            metrics.append(
                {
                    "policy": POLICY_NAME[checkpoint.variant],
                    "variant": checkpoint.variant,
                    "seed": checkpoint.seed,
                    "fold": checkpoint.fold,
                    "budget_fraction": fraction,
                    "task_key": task_key,
                    "dataset": tree.datasets[row],
                    "expected_correctness": float(expected_correct[row]),
                    "expected_teacher_tokens": float(expected_tokens[row]),
                    "expected_teacher_calls": float(expected_calls[row]),
                    "probability_sum": float(paths[row].sum()),
                }
            )
            for sequence_index, sequence in enumerate(SEQUENCES):
                probability_rows.append(
                    {
                        "variant": checkpoint.variant,
                        "seed": checkpoint.seed,
                        "fold": checkpoint.fold,
                        "budget_fraction": fraction,
                        "task_key": task_key,
                        "sequence": sequence,
                        "probability": float(paths[row, sequence_index]),
                    }
                )
    return metrics, probability_rows


def _runtime(
    checkpoint: FullTreePolicyCheckpoint,
    encoded: Any,
    states: list[Any],
    device: str,
) -> dict[str, Any]:
    network, torch_device = checkpoint.instantiate(device)
    return {
        "checkpoint": checkpoint,
        "network": network,
        "device": torch_device,
        "arrays": base_arrays(encoded, states, checkpoint.metadata_mean, checkpoint.metadata_scale),
    }


def _one_probability(runtime: dict[str, Any], state_index: int, allowance: float) -> float:
    torch = _torch()
    checkpoint = runtime["checkpoint"]
    batch = policy_batch(
        runtime["arrays"],
        np.asarray([state_index], dtype=np.int64),
        np.asarray([max(allowance, 0.0)], dtype=np.float32),
        reference_budget=checkpoint.reference_budget,
        device=runtime["device"],
    )
    with torch.inference_mode():
        return float(torch.sigmoid(runtime["network"](batch))[0].cpu())


def _strict_policy_replay(
    *,
    variant: str,
    seed: int,
    fraction: float,
    folds: dict[str, int],
    runtimes: dict[int, dict[str, Any]],
    tree: FullTreeDataset,
    hard_reserve: int,
    reference_budget: float,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total_budget = int(round(fraction * reference_budget * len(tree.task_keys)))
    remaining_global = total_budget
    key_to_row = {key: index for index, key in enumerate(tree.task_keys)}
    results: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    order = _stable_order(tree.task_keys, seed)
    for task_position, key in enumerate(order):
        row = key_to_row[key]
        prefix = ""
        tasks_remaining = len(order) - task_position
        while len(prefix) < 3:
            state_column = PREFIX_INDEX[prefix]
            state_index = int(tree.state_indices[row, state_column])
            allowance = min(remaining_global / max(tasks_remaining, 1), reference_budget)
            probability = _one_probability(runtimes[folds[key]], state_index, allowance)
            action = "V" if probability >= threshold and remaining_global >= hard_reserve else "S"
            before = remaining_global
            cost = 0
            if action == "V":
                cost = int(tree.edge_tokens[row, state_column])
                if cost > hard_reserve or cost > remaining_global:
                    raise AssertionError("Strict admission contract failed")
                remaining_global -= cost
            traces.append(
                {
                    "policy": POLICY_NAME[variant],
                    "variant": variant,
                    "seed": seed,
                    "budget_fraction": fraction,
                    "task_key": key,
                    "fold": folds[key],
                    "checkpoint": len(prefix) + 1,
                    "prefix": prefix or "START",
                    "remaining_global_before": before,
                    "observable_allowance": allowance,
                    "verify_probability": probability,
                    "action": action,
                    "actual_tokens_after_action": cost,
                }
            )
            prefix += action
        result = _task_result(
            policy=POLICY_NAME[variant],
            variant=variant,
            seed=seed,
            fraction=fraction,
            task_key=key,
            dataset=tree.datasets[row],
            sequence=prefix,
            tree=tree,
            row=row,
            fold=folds[key],
        )
        results.append(result)
    if sum(int(row["teacher_tokens"]) for row in results) > total_budget:
        raise AssertionError("Full-Tree deterministic replay exceeded global budget")
    return results, traces


def _fixed_replay(
    *,
    policy: str,
    requested: str,
    seed: int,
    fraction: float,
    tree: FullTreeDataset,
    hard_reserve: int,
    reference_budget: float,
) -> list[dict[str, Any]]:
    budget = int(round(fraction * reference_budget * len(tree.task_keys)))
    remaining = budget
    key_to_row = {key: index for index, key in enumerate(tree.task_keys)}
    results = []
    for key in _stable_order(tree.task_keys, seed):
        row = key_to_row[key]
        prefix = ""
        for wanted in requested:
            action = wanted
            if action == "V" and remaining < hard_reserve:
                action = "S"
            if action == "V":
                cost = int(tree.edge_tokens[row, PREFIX_INDEX[prefix]])
                if cost > hard_reserve or cost > remaining:
                    raise AssertionError("Fixed replay hard reserve is invalid")
                remaining -= cost
            prefix += action
        results.append(
            _task_result(
                policy=policy,
                variant="fixed",
                seed=seed,
                fraction=fraction,
                task_key=key,
                dataset=tree.datasets[row],
                sequence=prefix,
                tree=tree,
                row=row,
                fold="fixed",
            )
        )
    if sum(int(row["teacher_tokens"]) for row in results) > budget:
        raise AssertionError("Fixed replay exceeded global budget")
    return results


def _oracle_rows(
    *, fraction: float, seed: int, tree: FullTreeDataset, reference_budget: float
) -> list[dict[str, Any]]:
    budget = int(round(fraction * reference_budget * len(tree.task_keys)))
    selected = ["SSS"] * len(tree.task_keys)
    candidates: list[tuple[int, int, str]] = []
    for row in range(len(tree.task_keys)):
        if tree.outcomes[row, SEQUENCES.index("SSS")] == 1:
            continue
        successful = [
            (int(tree.trajectory_tokens[row, index]), sequence)
            for index, sequence in enumerate(SEQUENCES)
            if tree.outcomes[row, index] == 1
        ]
        if successful:
            cost, sequence = min(successful, key=lambda value: (value[0], value[1].count("V")))
            candidates.append((cost, row, sequence))
    spent = 0
    for cost, row, sequence in sorted(candidates):
        if spent + cost <= budget:
            selected[row] = sequence
            spent += cost
    return [
        _task_result(
            policy="Diagnostic oracle",
            variant="oracle",
            seed=seed,
            fraction=fraction,
            task_key=key,
            dataset=tree.datasets[row],
            sequence=selected[row],
            tree=tree,
            row=row,
            fold="oracle",
        )
        for row, key in enumerate(tree.task_keys)
    ]


def _historical_myopic_rows(
    root: Path,
    *,
    tree: FullTreeDataset,
    seeds: list[int],
    fractions: list[float],
) -> list[dict[str, Any]]:
    """Compose saved Myopic OOF + held-out traces without retraining it.

    The two partitions used separate strict ledgers in the historical experiment;
    their token budgets add to the same total, which is disclosed in the report.
    """

    path = root / "tables" / "decision_trace.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing existing Myopic trace: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        source = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("policy", "")).startswith("DARC v0.5 Myopic i0 · Qwen3.8-210 scratch")
        ]
    grouped: dict[tuple[int, float, str], list[dict[str, str]]] = defaultdict(list)
    for row in source:
        grouped[(int(row["seed"]), float(row["budget_fraction"]), row["task_key"])].append(row)
    key_to_row = {key: index for index, key in enumerate(tree.task_keys)}
    output: list[dict[str, Any]] = []
    for seed in seeds:
        for fraction in fractions:
            covered = 0
            for task_key in tree.task_keys:
                records = sorted(
                    grouped.get((seed, fraction, task_key), []),
                    key=lambda row: int(row["checkpoint"]),
                )
                if len(records) != 3:
                    raise ValueError(
                        f"Historical Myopic trace missing {seed}/{fraction}/{task_key}: "
                        f"{len(records)} decisions"
                    )
                sequence = "".join(row["action"] for row in records)
                tree_row = key_to_row[task_key]
                output.append(
                    _task_result(
                        policy="Existing DARC Qwen3.8 Myopic Scratch",
                        variant="historical_myopic",
                        seed=seed,
                        fraction=fraction,
                        task_key=task_key,
                        dataset=tree.datasets[tree_row],
                        sequence=sequence,
                        tree=tree,
                        row=tree_row,
                        fold="historical_split_composed",
                    )
                )
                covered += 1
            if covered != len(tree.task_keys):
                raise AssertionError("Historical Myopic trace did not cover all 210 tasks")
    return output


def _bootstrap(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _aggregate(rows: list[dict[str, Any]], repetitions: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), float(row["budget_fraction"]))].append(row)
    output = []
    task_values: dict[tuple[str, float], dict[str, float]] = {}
    for (policy, fraction), selected in sorted(grouped.items()):
        seeds = sorted({int(row["seed"]) for row in selected})
        tasks = sorted({str(row["task_key"]) for row in selected})
        values = {
            key: float(
                np.mean(
                    [float(row["correct"]) for row in selected if row["task_key"] == key]
                )
            )
            for key in tasks
        }
        task_values[(policy, fraction)] = values
        per_task = np.asarray([values[key] for key in tasks])
        low, high = _bootstrap(per_task, repetitions, seed)
        seed_rows = []
        for current_seed in seeds:
            local = [row for row in selected if int(row["seed"]) == current_seed]
            seed_rows.append(
                {
                    "correct": sum(int(row["correct"]) for row in local),
                    "tokens": sum(int(row["teacher_tokens"]) for row in local),
                    "calls": sum(int(row["teacher_calls"]) for row in local),
                    "cp1": sum(int(row["cp1_calls"]) for row in local),
                    "cp2": sum(int(row["cp2_calls"]) for row in local),
                    "cp3": sum(int(row["cp3_calls"]) for row in local),
                    "helpful": sum(int(row["helpful_calls"]) for row in local),
                    "harmful": sum(int(row["harmful_calls"]) for row in local),
                    "no_change": sum(int(row["no_change_calls"]) for row in local),
                }
            )
        output.append(
            {
                "policy": policy,
                "budget_fraction": fraction,
                "seeds": len(seeds),
                "tasks": len(tasks),
                "accuracy_mean": float(per_task.mean()),
                "accuracy_ci95_low": low,
                "accuracy_ci95_high": high,
                "correct_tasks_mean": float(np.mean([row["correct"] for row in seed_rows])),
                "teacher_tokens_mean": float(np.mean([row["tokens"] for row in seed_rows])),
                "teacher_calls_mean": float(np.mean([row["calls"] for row in seed_rows])),
                "cp1_calls_mean": float(np.mean([row["cp1"] for row in seed_rows])),
                "cp2_calls_mean": float(np.mean([row["cp2"] for row in seed_rows])),
                "cp3_calls_mean": float(np.mean([row["cp3"] for row in seed_rows])),
                "helpful_calls_mean": float(np.mean([row["helpful"] for row in seed_rows])),
                "harmful_calls_mean": float(np.mean([row["harmful"] for row in seed_rows])),
                "no_change_calls_mean": float(np.mean([row["no_change"] for row in seed_rows])),
            }
        )
    oracle = {float(row["budget_fraction"]): row for row in output if row["policy"] == "Diagnostic oracle"}
    vvv = {float(row["budget_fraction"]): row for row in output if row["policy"] == "VVV"}
    for row in output:
        fraction = float(row["budget_fraction"])
        row["oracle_gap_pp"] = 100 * (float(oracle[fraction]["accuracy_mean"]) - float(row["accuracy_mean"]))
        row["token_saving_vs_vvv"] = float(vvv[fraction]["teacher_tokens_mean"]) - float(
            row["teacher_tokens_mean"]
        )

        current = task_values[(str(row["policy"]), fraction)]
        never = task_values[("Never Verify (SSS)", fraction)]
        paired_keys = sorted(set(current) & set(never))
        if set(current) != set(never):
            raise AssertionError(
                f"Paired comparison has unequal task support for {row['policy']} at {fraction}"
            )
        differences = np.asarray([current[key] - never[key] for key in paired_keys])
        paired_low, paired_high = _bootstrap(differences, repetitions, seed + 17)
        row["paired_gain_vs_never_pp"] = 100 * float(differences.mean())
        row["paired_gain_vs_never_ci95_low_pp"] = 100 * paired_low
        row["paired_gain_vs_never_ci95_high_pp"] = 100 * paired_high

    deployable = [row for row in output if row["policy"] != "Diagnostic oracle"]
    for row in output:
        row["pareto_nondominated"] = bool(
            row["policy"] != "Diagnostic oracle"
            and not any(
                float(other["teacher_tokens_mean"]) <= float(row["teacher_tokens_mean"])
                and float(other["accuracy_mean"]) >= float(row["accuracy_mean"])
                and (
                    float(other["teacher_tokens_mean"]) < float(row["teacher_tokens_mean"])
                    or float(other["accuracy_mean"]) > float(row["accuracy_mean"])
                )
                for other in deployable
            )
        )
    return output


def _expected_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), float(row["budget_fraction"]))].append(row)
    return [
        {
            "policy": policy,
            "budget_fraction": fraction,
            "task_seed_rows": len(selected),
            "expected_accuracy": float(np.mean([row["expected_correctness"] for row in selected])),
            "expected_teacher_tokens_total": float(
                np.mean(
                    [
                        sum(row["expected_teacher_tokens"] for row in selected if row["seed"] == current_seed)
                        for current_seed in sorted({row["seed"] for row in selected})
                    ]
                )
            ),
            "expected_teacher_calls_total": float(
                np.mean(
                    [
                        sum(row["expected_teacher_calls"] for row in selected if row["seed"] == current_seed)
                        for current_seed in sorted({row["seed"] for row in selected})
                    ]
                )
            ),
        }
        for (policy, fraction), selected in sorted(grouped.items())
    ]


def _plot(output: Path, summary: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(9.2, 5.4))
    for policy in sorted({row["policy"] for row in summary if row["policy"] != "Diagnostic oracle"}):
        points = sorted(
            [row for row in summary if row["policy"] == policy],
            key=lambda row: float(row["teacher_tokens_mean"]),
        )
        axis.plot(
            [float(row["teacher_tokens_mean"]) / 1e6 for row in points],
            [100 * float(row["accuracy_mean"]) for row in points],
            marker="o",
            linewidth=2.4 if "Full-Tree" in policy else 1.25,
            alpha=1.0 if "Full-Tree" in policy else 0.75,
            label=policy,
        )
    frontier = sorted(
        [row for row in summary if bool(row.get("pareto_nondominated"))],
        key=lambda row: float(row["teacher_tokens_mean"]),
    )
    if frontier:
        axis.plot(
            [float(row["teacher_tokens_mean"]) / 1e6 for row in frontier],
            [100 * float(row["accuracy_mean"]) for row in frontier],
            color="#111827",
            linestyle="--",
            marker="D",
            markersize=5,
            linewidth=2.0,
            label="Strict deployable Pareto frontier",
            zorder=10,
        )
    axis.set(
        title="DARC Full-Tree: deterministic cross-fitted accuracy–cost frontier",
        xlabel="Actual Teacher tokens (millions)",
        ylabel="Final accuracy (%)",
    )
    axis.grid(color="#E2E8F0", linewidth=0.8)
    axis.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    path = output / "figures" / "accuracy_token_frontier.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _report(output: Path, summary: list[dict[str, Any]], expected: list[dict[str, Any]]) -> Path:
    full = [row for row in summary if float(row["budget_fraction"]) == 1.0]
    lines = [
        "# DARC Full-Tree Budgeted Policy Optimization",
        "",
        "> Five-fold task-level cross-validation on 210 Qwen3.8 Full-8 training tasks. "
        "No validation, frozen-test, prospective, Student, or Teacher access.",
        "",
        "## Method",
        "",
        "The policy observes only the public task, current branch artifact, observed action/artifact "
        "history, checkpoint, and remaining shadow budget. For each task, probabilities of all eight "
        "terminal paths are products of the three branch-specific policy decisions. Training directly "
        "maximizes exact expected final correctness minus a learned Lagrangian token penalty; no "
        "myopic forced-action label or best-trajectory label is used.",
        "",
        "## Deterministic cross-fitted deployment at full budget",
        "",
        "| Policy | Accuracy (task-bootstrap 95% CI) | Correct | Actual tokens | "
        "Calls CP1/CP2/CP3 | H/Harm/NC | Gain vs SSS (paired 95% CI) | "
        "Save vs VVV | Oracle gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        full,
        key=lambda value: (
            -float(value["accuracy_mean"]),
            float(value["teacher_tokens_mean"]),
        ),
    ):
        lines.append(
            f"| {row['policy']} | {100 * float(row['accuracy_mean']):.2f}% "
            f"[{100 * float(row['accuracy_ci95_low']):.2f}, {100 * float(row['accuracy_ci95_high']):.2f}] | "
            f"{float(row['correct_tasks_mean']):.1f}/210 | {float(row['teacher_tokens_mean']):,.0f} | "
            f"{float(row['cp1_calls_mean']):.1f}/{float(row['cp2_calls_mean']):.1f}/"
            f"{float(row['cp3_calls_mean']):.1f} | {float(row['helpful_calls_mean']):.1f}/"
            f"{float(row['harmful_calls_mean']):.1f}/{float(row['no_change_calls_mean']):.1f} | "
            f"{float(row['paired_gain_vs_never_pp']):+.2f} pp "
            f"[{float(row['paired_gain_vs_never_ci95_low_pp']):+.2f}, "
            f"{float(row['paired_gain_vs_never_ci95_high_pp']):+.2f}] | "
            f"{float(row['token_saving_vs_vvv']):,.0f} | "
            f"{float(row['oracle_gap_pp']):.2f} pp |"
        )
    lines += [
        "",
        "![Accuracy–token frontier](../figures/accuracy_token_frontier.png)",
        "",
        "## Stochastic expected policy",
        "",
        "Expected metrics integrate all eight path probabilities and are reported separately from "
        "thresholded deployment. They enforce the expected-cost objective, not a per-task hard cap.",
        "",
        "| Policy | Budget | Expected accuracy | Expected tokens | Expected calls |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in expected:
        lines.append(
            f"| {row['policy']} | {100 * float(row['budget_fraction']):.0f}% | "
            f"{100 * float(row['expected_accuracy']):.2f}% | "
            f"{float(row['expected_teacher_tokens_total']):,.0f} | "
            f"{float(row['expected_teacher_calls_total']):.1f} |"
        )
    lines += [
        "",
        "## Interpretation safeguards",
        "",
        "- CV folds are task-disjoint; every task is evaluated only by a model that excluded it.",
        "- Deterministic replay uses a strict global actual-token ledger and a "
        "protocol-level hard-call reserve.",
        "- Shadow allowance is a policy feature, not a per-task feasibility mask; "
        "expected cost is controlled by the Lagrangian, while only deterministic "
        "deployment applies hard global admission.",
        "- Actual current/future call costs never enter a pre-action feature; "
        "completed-call cost may update later budget state.",
        "- Diagnostic oracle is globally budget-feasible and branch-consistent, "
        "but is not a deployable policy.",
        "- The historical Myopic baseline is composed from its saved OOF and "
        "held-out traces; its two ledgers sum to the same total budget but task "
        "ordering is not identical to the new single-ledger replay.",
        "- Any final-fit checkpoint is trained on all 210 tasks and is explicitly "
        "not independently evaluated.",
    ]
    path = output / "report" / "darc_fulltree_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _smoke(encoded_size: int = 14) -> dict[str, Any]:
    torch = _torch()
    from agent_verify.analysis.darc_fulltree_model import network_class
    from agent_verify.analysis.darc_fulltree_objective import edge_expected_tokens, fulltree_expectation

    torch.manual_seed(3)
    probability = torch.sigmoid(torch.randn(2, 7, requires_grad=True))
    outcomes = torch.randint(0, 2, (2, 8), dtype=torch.float32)
    edges = torch.randint(1, 20, (2, 7), dtype=torch.float32)
    # Construct terminal costs from edge paths so the identity is exact.
    terminal = []
    for task in range(2):
        row = []
        for sequence in SEQUENCES:
            prefix = ""
            cost = 0.0
            for action in sequence:
                if action == "V":
                    cost += float(edges[task, PREFIX_INDEX[prefix]])
                prefix += action
            row.append(cost)
        terminal.append(row)
    terminal_tensor = torch.tensor(terminal)
    result = fulltree_expectation(
        probability,
        outcomes,
        terminal_tensor,
        target_tokens=20,
        reference_tokens=50,
        lagrange_multiplier=0.1,
    )
    identity = edge_expected_tokens(probability, edges)
    result.loss.backward()
    model = network_class()(8, 3, {"hidden_dim": 8, "attention_heads": 2, "dropout": 0.0})
    return {
        "passed": bool(
            torch.allclose(result.probabilities.sum(1), torch.ones(2), atol=1e-6)
            and torch.allclose(identity, result.expected_tokens, atol=1e-5)
            and probability.grad_fn is not None
            and sum(parameter.numel() for parameter in model.parameters()) > encoded_size
        ),
        "probability_sums": result.probabilities.sum(1).detach().tolist(),
        "cost_identity_max_error": float(
            (identity - result.expected_tokens).abs().max().detach()
        ),
    }


def run(config_path: Path, *, mode: str, resume: bool = False) -> Path:
    if mode not in {"dry-run", "smoke", "crossval", "final-fit"}:
        raise ValueError(mode)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["evaluation"]["frozen_test_access"] != "forbidden":
        raise ValueError("Frozen test access must be forbidden")
    output = Path(config["output"]["root"])
    for child in ("logs", "checkpoints", "tables", "figures", "report", "status", "embedding_cache"):
        (output / child).mkdir(parents=True, exist_ok=True)
    groups, metadata, states, lookup, tree = _load(config)
    folds = assign_stratified_task_folds(
        metadata,
        folds=int(config["reproducibility"]["folds"]),
        seed=int(config["reproducibility"]["fold_seed"]),
    )
    assert_disjoint_folds(tree.task_keys, folds, int(config["reproducibility"]["folds"]))
    reference_budget = _reference_budget(tree)
    hard_reserve = int(config["evaluation"]["hard_call_token_reserve"])
    max_edge = _max_edge_cost(tree)
    if max_edge > hard_reserve:
        raise ValueError(f"Hard-call reserve {hard_reserve} is below observed training edge {max_edge}")
    split_rows = split_audit_rows(groups, metadata, folds)
    _write_csv(output / "tables" / "task_folds.csv", split_rows)
    integrity = {
        "schema_version": "darc-fulltree-integrity-v1",
        "tasks": len(tree.task_keys),
        "trajectories": int(tree.outcomes.size),
        "observable_states": int(tree.state_indices.size),
        "folds": int(config["reproducibility"]["folds"]),
        "task_overlap_across_fold_train_eval": 0,
        "reference_budget_per_task": reference_budget,
        "max_actual_edge_tokens": max_edge,
        "hard_call_token_reserve": hard_reserve,
        "terminal_edge_cost_identity": True,
        "catalog_splits_loaded": ["train"],
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "prospective_accessed": False,
    }
    integrity_path = output / "integrity.json"
    _write_json(integrity_path, integrity)
    if mode == "dry-run":
        return integrity_path
    if mode == "smoke":
        smoke = {**integrity, **_smoke()}
        path = output / "smoke_report.json"
        _write_json(path, smoke)
        return path

    encoded, cache_manifest = encode_or_load(
        [state.encoder_example for state in states],
        split="darc_qwen38_fulltree_210",
        config=config["semantic_encoder"],
        cache_dir=output / "embedding_cache",
    )
    seeds = [int(value) for value in config["reproducibility"]["seeds"]]
    device = str(config["semantic_encoder"]["device"])
    started = time.time()
    if mode == "final-fit":
        variants = [str(config["final_fit"].get("variant", "cp3_anchor"))]
        for variant in variants:
            for seed in tqdm(seeds, desc="FullTree final fit", unit="seed"):
                base = output / "checkpoints" / "final_fit" / variant / f"seed_{seed}"
                final_path = base / "model.pt"
                if final_path.exists():
                    if not resume:
                        raise FileExistsError(f"{final_path} exists; use --resume")
                    continue
                train_fulltree_policy(
                    all_states=states,
                    encoded=encoded,
                    train_tree=tree,
                    params=config["model"],
                    optimization=config["optimization"],
                    anchor=config["anchor"],
                    variant=variant,
                    reference_budget=reference_budget,
                    seed=seed,
                    fold=-1,
                    device_name=device,
                    deterministic=bool(config["reproducibility"]["deterministic_torch"]),
                    checkpoint_path=final_path,
                    last_path=base / "last.pt",
                    resume=resume,
                    progress=config["monitoring"],
                )
        manifest = {
            **integrity,
            "mode": "final-fit",
            "label": "trained_on_all_210_not_independently_evaluated",
            "seeds": seeds,
            "variants": variants,
            "embedding_cache": cache_manifest,
            "elapsed_seconds": time.time() - started,
        }
        path = output / "final_fit_manifest.json"
        _write_json(path, manifest)
        return path

    fractions = [float(value) for value in config["evaluation"]["budget_fractions"]]
    variants = [str(value) for value in config["crossval"]["variants"]]
    fold_count = int(config["reproducibility"]["folds"])
    checkpoints: dict[tuple[str, int, int], FullTreePolicyCheckpoint] = {}
    history_rows: list[dict[str, Any]] = []
    jobs = [(variant, seed, fold) for variant in variants for seed in seeds for fold in range(fold_count)]
    for variant, seed, fold in tqdm(jobs, desc="FullTree CV models", unit="model"):
        train_keys = {key for key in tree.task_keys if folds[key] != fold}
        train_tree = tree.subset(_rows_for_keys(tree, train_keys))
        base = output / "checkpoints" / "crossval" / variant / f"seed_{seed}" / f"fold_{fold}"
        final_path = base / "model.pt"
        if final_path.exists():
            if not resume:
                raise FileExistsError(f"{final_path} exists; use --resume")
            checkpoint = FullTreePolicyCheckpoint.load(final_path)
        else:
            checkpoint = train_fulltree_policy(
                all_states=states,
                encoded=encoded,
                train_tree=train_tree,
                params=config["model"],
                optimization=config["optimization"],
                anchor=config["anchor"],
                variant=variant,
                reference_budget=reference_budget,
                seed=seed,
                fold=fold,
                device_name=device,
                deterministic=bool(config["reproducibility"]["deterministic_torch"]),
                checkpoint_path=final_path,
                last_path=base / "last.pt",
                resume=resume,
                progress=config["monitoring"],
            )
        checkpoints[(variant, seed, fold)] = checkpoint
        history_rows.extend(checkpoint.history)

    expected_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    deterministic_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    threshold = float(config["evaluation"].get("deterministic_threshold", 0.5))
    for variant in variants:
        for seed in tqdm(seeds, desc=f"FullTree evaluate {variant}", unit="seed"):
            runtimes = {
                fold: _runtime(checkpoints[(variant, seed, fold)], encoded, states, device)
                for fold in range(fold_count)
            }
            for fold in range(fold_count):
                heldout_keys = {key for key in tree.task_keys if folds[key] == fold}
                held_tree = tree.subset(_rows_for_keys(tree, heldout_keys))
                metrics, probabilities = _expected_rows(
                    checkpoint=checkpoints[(variant, seed, fold)],
                    encoded=encoded,
                    states=states,
                    tree=held_tree,
                    fractions=fractions,
                    device_name=device,
                )
                expected_rows.extend(metrics)
                probability_rows.extend(probabilities)
            for fraction in fractions:
                rows, traces = _strict_policy_replay(
                    variant=variant,
                    seed=seed,
                    fraction=fraction,
                    folds=folds,
                    runtimes=runtimes,
                    tree=tree,
                    hard_reserve=hard_reserve,
                    reference_budget=reference_budget,
                    threshold=threshold,
                )
                deterministic_rows.extend(rows)
                trace_rows.extend(traces)

    for seed in seeds:
        for fraction in fractions:
            for policy, sequence in FIXED.items():
                deterministic_rows.extend(
                    _fixed_replay(
                        policy=policy,
                        requested=sequence,
                        seed=seed,
                        fraction=fraction,
                        tree=tree,
                        hard_reserve=hard_reserve,
                        reference_budget=reference_budget,
                    )
                )
            deterministic_rows.extend(
                _oracle_rows(
                    fraction=fraction,
                    seed=seed,
                    tree=tree,
                    reference_budget=reference_budget,
                )
            )

    deterministic_rows.extend(
        _historical_myopic_rows(
            Path(config["data"]["historical_myopic_root"]),
            tree=tree,
            seeds=seeds,
            fractions=fractions,
        )
    )

    summary = _aggregate(
        deterministic_rows,
        int(config["reproducibility"]["bootstrap_repetitions"]),
        seeds[0],
    )
    expected_summary = _expected_summary(expected_rows)
    _write_csv(output / "tables" / "training_history.csv", history_rows)
    _write_csv(output / "tables" / "deterministic_task_results.csv", deterministic_rows)
    _write_csv(output / "tables" / "decision_trace.csv", trace_rows)
    _write_csv(output / "tables" / "stochastic_expected_task_metrics.csv", expected_rows)
    _write_csv(output / "tables" / "trajectory_probabilities.csv", probability_rows)
    _write_csv(output / "tables" / "cv_summary.csv", summary)
    _write_csv(output / "tables" / "stochastic_expected_summary.csv", expected_summary)
    _plot(output, summary)
    report = _report(output, summary, expected_summary)
    _write_json(
        output / "run_manifest.json",
        {
            **integrity,
            "mode": "crossval",
            "seeds": seeds,
            "variants": variants,
            "budget_fractions": fractions,
            "embedding_cache": cache_manifest,
            "elapsed_seconds": time.time() - started,
            "report": str(report),
        },
    )
    return report
