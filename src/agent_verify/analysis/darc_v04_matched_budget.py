"""Exact matched-budget replay for DARC v0.4 Myopic and C3 MultiHistory.

No model is trained here.  Every policy is replayed on the same Level-2
validation tasks, task order, absolute Teacher-token cap and strict actual-cost
ledger.  Unknown future cost/outcomes are never used for online admission.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from agent_verify.analysis.darc_data import encode_or_load
from agent_verify.analysis.darc_v04_audit import (
    _seed_summary,
    aggregate_records,
    paired_comparison,
    replay_c3_strict,
)
from agent_verify.analysis.darc_v04_data import build_states
from agent_verify.analysis.darc_v04_experiment import (
    SequentialPolicy,
    _allowances,
    _load_groups,
    _precompute,
    _write_csv,
    replay_v04,
)
from agent_verify.analysis.darc_v04_model import DARCv04Ensemble


MYOPIC_LABEL = "DARC v0.4 · myopic_targets"
C3_LABELS = ("C3 MultiHistory Myopic", "C3 MultiHistory Continuation")


def parse_budgets(value: str) -> list[int]:
    budgets = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not budgets or any(item <= 0 for item in budgets):
        raise ValueError("Matched budgets must be positive integers")
    return budgets


def _selected_policies(path: Path, seeds: list[int]) -> dict[int, SequentialPolicy]:
    rows = list(csv.DictReader(path.open()))
    output: dict[int, SequentialPolicy] = {}
    for seed in seeds:
        matches = [
            row
            for row in rows
            if int(row["seed"]) == seed
            and row["variant"] == MYOPIC_LABEL
            and str(row["selected"]).lower() in {"1", "true", "yes"}
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one selected Myopic policy for seed {seed}, got {len(matches)}")
        row = matches[0]
        output[seed] = SequentialPolicy(
            harmful_weight=float(row["harmful_weight"]),
            uncertainty_beta=float(row["uncertainty_beta"]),
            cost_lambda=float(row["cost_lambda"]),
            decision_margin=float(row["decision_margin"]),
        )
    return output


def _normalized_darc_traces(traces: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "seed": seed,
            "no_change_selected": int(row.get("wasted_selected", 0)),
        }
        for row in traces
    ]


def _plot(output: Path, aggregate: list[dict[str, Any]], budgets: list[int]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        MYOPIC_LABEL: "#D9485F",
        "C3 MultiHistory Myopic": "#4C78A8",
        "C3 MultiHistory Continuation": "#72A0C1",
    }
    fig, axis = plt.subplots(figsize=(9.2, 5.8))
    for policy in (MYOPIC_LABEL, *C3_LABELS):
        points = sorted(
            [row for row in aggregate if row["policy"] == policy],
            key=lambda row: int(row["token_cap"]),
        )
        axis.errorbar(
            [float(row["teacher_tokens_mean"]) / 1000 for row in points],
            [100 * float(row["accuracy_mean"]) for row in points],
            yerr=[
                [100 * (float(row["accuracy_mean"]) - float(row["accuracy_ci95_low"])) for row in points],
                [100 * (float(row["accuracy_ci95_high"]) - float(row["accuracy_mean"])) for row in points],
            ],
            marker="o",
            linewidth=2.6 if policy == MYOPIC_LABEL else 1.8,
            markersize=7,
            capsize=3,
            color=colors[policy],
            label=policy,
        )
        for row in points:
            axis.annotate(
                f"cap {int(row['token_cap'])//1000}K",
                (float(row["teacher_tokens_mean"]) / 1000, 100 * float(row["accuracy_mean"])),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7,
                color=colors[policy],
            )
    axis.set_title("Exact matched-budget validation replay", weight="bold")
    axis.set_xlabel("Actual Teacher tokens used (thousands; mean across seeds)")
    axis.set_ylabel("Downstream accuracy (%)")
    axis.legend(frameon=False)
    axis.set_xticks(sorted({0, *[value // 1000 for value in budgets]}))
    fig.tight_layout()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "matched_budget_frontier.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "matched_budget_frontier.pdf", bbox_inches="tight")
    plt.close(fig)


def _report(
    output: Path,
    aggregate: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    budgets: list[int],
) -> Path:
    lines = [
        "# DARC Myopic vs C3 — Exact Matched-Budget Evaluation",
        "",
        "> Validation-only offline replay. No training and no frozen-test access.",
        "",
        "All methods receive the same absolute Teacher-token cap, Level-2 tasks, per-seed task order, 8,192-token hard-call reservation and post-action actual-cost accounting.",
        "",
        "| Token cap | Policy | Accuracy mean (95% task CI) | Actual tokens | Calls | Helpful / Harmful / No-change |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for budget in budgets:
        rows = sorted(
            [row for row in aggregate if int(row["token_cap"]) == budget],
            key=lambda row: float(row["accuracy_mean"]),
            reverse=True,
        )
        for row in rows:
            lines.append(
                f"| {budget:,} | {row['policy']} | {100*float(row['accuracy_mean']):.2f}% [{100*float(row['accuracy_ci95_low']):.2f}, {100*float(row['accuracy_ci95_high']):.2f}] | {float(row['teacher_tokens_mean']):,.0f} | {float(row['teacher_calls_mean']):.1f} | {float(row['helpful_mean']):.1f} / {float(row['harmful_mean']):.1f} / {float(row['no_change_mean']):.1f} |"
            )
    lines.extend(
        [
            "",
            "## Paired DARC-minus-C3 comparisons",
            "",
            "| Token cap | C3 reference | Accuracy difference | 95% CI | Permutation p | Token difference |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {int(row['token_cap']):,} | {row['reference']} | {float(row['accuracy_difference_pp']):+.2f} pp | [{float(row['difference_ci95_low_pp']):+.2f}, {float(row['difference_ci95_high_pp']):+.2f}] | {float(row['paired_permutation_p']):.4f} | {float(row['token_difference_mean']):+,.0f} |"
        )
    lines.extend(
        [
            "",
            "![Matched-budget frontier](../figures/matched_budget_frontier.png)",
            "",
            "Interpretation: compare policies within each cap. A higher-cost point from another cap is not evidence of superiority at the current cap.",
        ]
    )
    report = output / "report" / "matched_budget_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(config_path: Path, budgets: list[int]) -> Path:
    config = yaml.safe_load(config_path.read_text())
    if config["evaluation"]["partition"] != "validation":
        raise ValueError("Matched-budget evaluation is validation-only")
    if config["evaluation"]["frozen_test_access"] != "forbidden":
        raise ValueError("Frozen test must remain forbidden")
    audit_root = Path(config["output"]["root"])
    manifest = json.loads((audit_root / "manifest.json").read_text())
    if bool(manifest.get("frozen_test_loaded")):
        raise RuntimeError("Refusing a matched-budget run from a manifest that loaded test")
    seeds = [int(value) for value in config["reproducibility"]["seeds"]]
    policies = _selected_policies(audit_root / "tables" / "policy_grid_results.csv", seeds)
    frozen_root = Path(config["data"]["frozen_root"])
    groups = _load_groups(frozen_root, "validation")
    states, lookup = build_states(groups)
    encoded, _cache = encode_or_load(
        [state.encoder_example for state in states],
        split="v04_audit_validation",
        config=config["semantic_encoder"],
        cache_dir=Path(config["semantic_encoder"]["cache_dir"]),
    )
    reference_budget = float(manifest["reference_budget_per_task"])
    allowances = _allowances(
        reference_budget, [float(value) for value in config["targets"]["budget_fractions"]]
    )
    c3_root = Path(config["data"]["c3_results"]) / "models" / "combined"
    c3 = {
        "C3 MultiHistory Myopic": joblib.load(c3_root / "multi_myopic.joblib"),
        "C3 MultiHistory Continuation": joblib.load(c3_root / "multi_continuation.joblib"),
    }
    hard_cap = int(config["evaluation"]["hard_call_token_cap"])
    records: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    fraction_to_cap: dict[float, int] = {}
    for seed in seeds:
        model = DARCv04Ensemble.load(
            audit_root / "models" / f"seed_{seed}" / "myopic_targets.pt"
        )
        predictions = _precompute(
            model,
            encoded,
            states,
            allowances,
            device=str(config["semantic_encoder"]["device"]),
        )
        for budget in budgets:
            fraction = budget / max(reference_budget * len(groups), 1.0)
            fraction_to_cap[fraction] = budget
            replay, traces = replay_v04(
                groups,
                states,
                lookup,
                predictions,
                allowances,
                policies[seed],
                token_budget=budget,
                budget_fraction=fraction,
                hard_call_token_cap=hard_cap,
                seed=seed,
                label=MYOPIC_LABEL,
            )
            normalized = _normalized_darc_traces(traces, seed)
            summary = _seed_summary(replay, normalized, seed=seed)
            summary["token_cap"] = budget
            seed_rows.append(summary)
            records.append(
                {"seed": seed, "policy": replay.policy, "fraction": fraction, "replay": replay, "summary": summary}
            )
            trace_rows.extend({**row, "token_cap": budget} for row in normalized)
            for label, controller in c3.items():
                replay, traces = replay_c3_strict(
                    groups,
                    controller,
                    token_budget=budget,
                    budget_fraction=fraction,
                    hard_call_token_cap=hard_cap,
                    seed=seed,
                    label=label,
                )
                summary = _seed_summary(replay, traces, seed=seed)
                summary["token_cap"] = budget
                seed_rows.append(summary)
                records.append(
                    {"seed": seed, "policy": replay.policy, "fraction": fraction, "replay": replay, "summary": summary}
                )
                trace_rows.extend({**row, "token_cap": budget} for row in traces)

    aggregate = aggregate_records(
        records,
        bootstrap=int(config["reproducibility"]["bootstrap_repetitions"]),
        seed=seeds[0],
    )
    for row in aggregate:
        row["token_cap"] = fraction_to_cap[float(row["budget_fraction"])]
    comparisons: list[dict[str, Any]] = []
    for fraction, budget in sorted(fraction_to_cap.items(), key=lambda item: item[1]):
        for reference in C3_LABELS:
            row = paired_comparison(
                records,
                candidate=MYOPIC_LABEL,
                reference=reference,
                fraction=fraction,
                bootstrap=int(config["reproducibility"]["bootstrap_repetitions"]),
                permutations=int(config["reproducibility"]["permutation_repetitions"]),
                seed=seeds[0] + budget,
            )
            row["token_cap"] = budget
            comparisons.append(row)

    output = audit_root / "matched_budget"
    _write_csv(output / "tables" / "seed_results.csv", seed_rows)
    _write_csv(output / "tables" / "aggregate_results.csv", aggregate)
    _write_csv(output / "tables" / "paired_comparisons.csv", comparisons)
    _write_csv(output / "tables" / "decision_trace.csv", trace_rows)
    _plot(output, aggregate, budgets)
    report = _report(output, aggregate, comparisons, budgets)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "darc-v0.4-matched-budget-v1",
                "source_audit_manifest": str(audit_root / "manifest.json"),
                "partition": "validation",
                "frozen_test_loaded": False,
                "absolute_token_caps": budgets,
                "seeds": seeds,
                "policies": [MYOPIC_LABEL, *C3_LABELS],
                "selected_myopic_policies": {str(seed): asdict(policy) for seed, policy in policies.items()},
                "report": str(report),
            },
            indent=2,
        )
        + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/darc_v04_audit.yaml"))
    parser.add_argument("--budgets", default="100000,150000,200000,250000")
    args = parser.parse_args()
    report = run(args.config, parse_budgets(args.budgets))
    print(f"Matched-budget evaluation completed. Report: {report}")


if __name__ == "__main__":
    main()

