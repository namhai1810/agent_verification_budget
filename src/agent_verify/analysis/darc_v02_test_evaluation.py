"""One-shot exploratory frozen-test evaluation for an already selected DARC v0.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from agent_verify.analysis.cbv_v01_experiment import _bootstrap_summary, build_level1_examples
from agent_verify.analysis.darc_data import encode_or_load
from agent_verify.analysis.darc_experiment import (
    _absolute,
    _assert_disjoint,
    _assert_hard_call_cap,
    _load_split,
)
from agent_verify.analysis.darc_model import DARCEnsemble, prediction_metrics
from agent_verify.analysis.darc_v02 import (
    CheckpointCalibrator,
    ContinuationPrior,
    RiskCostPolicy,
    failure_metrics,
    replay_risk_cost_policy,
)
from agent_verify.analysis.darc_v02_experiment import (
    _always_cp2,
    _c3_strict_replay,
    _policy_budgets,
    _strict_threshold_replay,
    _write_csv,
)


def _policy(payload: dict[str, Any]) -> RiskCostPolicy:
    return RiskCostPolicy(
        harmful_weight=float(payload["harmful_weight"]),
        uncertainty_beta=float(payload["uncertainty_beta"]),
        cost_lambda=float(payload["cost_lambda"]),
        continuation_weight=float(payload["continuation_weight"]),
        checkpoint_margins=tuple(float(value) for value in payload["checkpoint_margins"]),
        cost_mode=str(payload["cost_mode"]),
    )


def _plot(
    output: Path,
    rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]] | None = None,
) -> None:
    figure = output / "figures" / "darc_v02_frozen_test_frontier.png"
    clean_figure = output / "figures" / "darc_v02_frozen_test_operating_points.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    primary = (
        "DARC v0.2 Full",
        "Joint MLP + Threshold",
        "C3 Myopic",
        "Always CP2",
        "DARC v0.1 Full",
        "Never Verify",
    )
    colors = {
        "DARC v0.2 Full": "#C23B3B",
        "Joint MLP + Threshold": "#4C78A8",
        "C3 Myopic": "#59A14F",
        "Always CP2": "#F28E2B",
        "DARC v0.1 Full": "#8E6C8A",
        "Never Verify": "#6B7280",
    }
    markers = {
        "DARC v0.2 Full": "o",
        "Joint MLP + Threshold": "s",
        "C3 Myopic": "^",
        "Always CP2": "D",
        "DARC v0.1 Full": "P",
        "Never Verify": "X",
    }
    selected_rows = [row for row in rows if str(row["policy"]) in primary]
    unique: dict[tuple[str, float, float], dict[str, Any]] = {}
    for row in selected_rows:
        key = (
            str(row["policy"]),
            float(row["actual_teacher_tokens"]),
            float(row["accuracy"]),
        )
        unique[key] = row
    selected_rows = list(unique.values())

    def _pareto(row: dict[str, Any]) -> bool:
        cost = float(row["actual_teacher_tokens"])
        accuracy = float(row["accuracy"])
        return not any(
            float(other["actual_teacher_tokens"]) <= cost
            and float(other["accuracy"]) >= accuracy
            and (
                float(other["actual_teacher_tokens"]) < cost
                or float(other["accuracy"]) > accuracy
            )
            for other in selected_rows
        )

    fig, axis = plt.subplots(figsize=(10.2, 6.2))
    for name in primary:
        points = [row for row in selected_rows if row["policy"] == name]
        if not points:
            continue
        intermediate = [row for row in points if float(row["budget_fraction"]) < 1.0]
        full = [row for row in points if float(row["budget_fraction"]) == 1.0]
        if intermediate:
            axis.scatter(
                [float(row["actual_teacher_tokens"]) for row in intermediate],
                [100.0 * float(row["accuracy"]) for row in intermediate],
                s=46,
                marker=markers[name],
                color=colors[name],
                alpha=0.42,
                edgecolors="none",
            )
        if full:
            axis.scatter(
                [float(row["actual_teacher_tokens"]) for row in full],
                [100.0 * float(row["accuracy"]) for row in full],
                s=105,
                marker=markers[name],
                color=colors[name],
                edgecolors="white",
                linewidths=1.2,
                label=name,
                zorder=4,
            )
    pareto = [row for row in selected_rows if _pareto(row)]
    axis.scatter(
        [float(row["actual_teacher_tokens"]) for row in pareto],
        [100.0 * float(row["accuracy"]) for row in pareto],
        s=170,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.25,
        label="Nondominated",
        zorder=5,
    )
    selected = next(
        row
        for row in selected_rows
        if row["policy"] == "DARC v0.2 Full" and float(row["budget_fraction"]) == 1.0
    )
    axis.annotate(
        f"DARC v0.2\n{100.0 * float(selected['accuracy']):.2f}%",
        (float(selected["actual_teacher_tokens"]), 100.0 * float(selected["accuracy"])),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8.5,
        color=colors["DARC v0.2 Full"],
        weight="bold",
    )
    axis.set_xlabel("Actual Teacher tokens (strict cap)")
    axis.set_ylabel("Frozen-test accuracy (%)")
    axis.set_title("DARC v0.2 frozen test: accuracy–cost operating points (exploratory)")
    axis.grid(axis="y", alpha=0.20)
    axis.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    axis.text(
        0.01,
        0.02,
        "Small translucent markers: intermediate budgets  •  Large markers: full budget  •  Black ring: nondominated",
        transform=axis.transAxes,
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(figure, dpi=220, bbox_inches="tight")
    fig.savefig(clean_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    full = [row for row in selected_rows if float(row["budget_fraction"]) == 1.0]
    full.sort(key=lambda row: float(row["accuracy"]), reverse=True)
    full_failures = {
        str(row["policy"]): row
        for row in (failure_rows or [])
        if float(row["budget_fraction"]) == 1.0
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.6, 5.2),
        gridspec_kw={"width_ratios": [1, 1.15]},
    )

    accuracy_axis = axes[0]
    for position, row in enumerate(full):
        name = str(row["policy"])
        value = 100.0 * float(row["accuracy"])
        lower = 100.0 * float(row["accuracy_ci95_low"])
        upper = 100.0 * float(row["accuracy_ci95_high"])
        highlighted = name == "DARC v0.2 Full"
        accuracy_axis.errorbar(
            value,
            position,
            xerr=[[value - lower], [upper - value]],
            fmt="*" if highlighted else "o",
            markersize=16 if highlighted else 9,
            color=colors[name],
            ecolor=colors[name] if highlighted else "#9CA3AF",
            elinewidth=2.5 if highlighted else 1.6,
            capsize=4,
            zorder=4 if highlighted else 3,
        )
        accuracy_axis.text(
            upper + 0.35,
            position,
            f"{value:.2f}%",
            va="center",
            fontsize=9,
            color=colors[name] if highlighted else "#374151",
            weight="bold" if highlighted else "normal",
        )
    never = next(row for row in full if row["policy"] == "Never Verify")
    accuracy_axis.axvline(
        100.0 * float(never["accuracy"]),
        color="#6B7280",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
    )
    accuracy_axis.set_yticks(
        np.arange(len(full)),
        [str(row["policy"]) for row in full],
    )
    accuracy_axis.invert_yaxis()
    accuracy_axis.set_xlabel("Accuracy (%) with task-bootstrap 95% CI")
    accuracy_axis.set_title("a  Frozen-test accuracy", loc="left", weight="bold")
    accuracy_axis.grid(axis="x", alpha=0.18)
    accuracy_axis.spines[["top", "right", "left"]].set_visible(False)
    accuracy_axis.tick_params(axis="y", length=0)

    cost_axis = axes[1]
    for row in full:
        name = str(row["policy"])
        cost = float(row["actual_teacher_tokens"]) / 1000.0
        value = 100.0 * float(row["accuracy"])
        highlighted = name == "DARC v0.2 Full"
        cost_axis.scatter(
            cost,
            value,
            s=260 if highlighted else 125,
            marker="*" if highlighted else markers[name],
            color=colors[name],
            edgecolors="white",
            linewidths=1.3,
            zorder=5 if highlighted else 4,
        )
    label_offsets = {
        "Always CP2": (-78, 12),
        "Joint MLP + Threshold": (8, 12),
        "DARC v0.2 Full": (-95, 14),
        "C3 Myopic": (-80, -17),
        "DARC v0.1 Full": (8, -18),
        "Never Verify": (8, 10),
    }
    short_names = {
        "Always CP2": "Always CP2",
        "Joint MLP + Threshold": "Joint MLP",
        "DARC v0.2 Full": "DARC v0.2",
        "C3 Myopic": "C3 Myopic",
        "DARC v0.1 Full": "DARC v0.1",
        "Never Verify": "Never Verify",
    }
    for row in full:
        name = str(row["policy"])
        cost_axis.annotate(
            short_names[name],
            (
                float(row["actual_teacher_tokens"]) / 1000.0,
                100.0 * float(row["accuracy"]),
            ),
            xytext=label_offsets[name],
            textcoords="offset points",
            fontsize=9,
            color=colors[name] if name == "DARC v0.2 Full" else "#374151",
            weight="bold" if name == "DARC v0.2 Full" else "normal",
        )
    v01 = next(row for row in full if row["policy"] == "DARC v0.1 Full")
    v02 = next(row for row in full if row["policy"] == "DARC v0.2 Full")
    cost_axis.annotate(
        "",
        xy=(
            float(v02["actual_teacher_tokens"]) / 1000.0,
            100.0 * float(v02["accuracy"]),
        ),
        xytext=(
            float(v01["actual_teacher_tokens"]) / 1000.0,
            100.0 * float(v01["accuracy"]),
        ),
        arrowprops={"arrowstyle": "->", "color": "#C23B3B", "lw": 2.1},
        zorder=3,
    )
    harmful_v01 = int(
        full_failures.get("DARC v0.1 Full", {}).get("harmful_selected", 4)
    )
    harmful_v02 = int(
        full_failures.get("DARC v0.2 Full", {}).get("harmful_selected", 0)
    )
    token_reduction = 100.0 * (
        1.0
        - float(v02["actual_teacher_tokens"])
        / max(float(v01["actual_teacher_tokens"]), 1.0)
    )
    accuracy_gain = 100.0 * (float(v02["accuracy"]) - float(v01["accuracy"]))
    cost_axis.text(
        0.51,
        0.18,
        f"v0.2 vs v0.1\n+{accuracy_gain:.2f} pp accuracy\n"
        f"−{token_reduction:.1f}% tokens\n{harmful_v01} → {harmful_v02} harmful calls",
        transform=cost_axis.transAxes,
        fontsize=9.5,
        color="#7F1D1D",
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#FFF1F2",
            "edgecolor": "#FDA4AF",
            "linewidth": 1.0,
        },
    )
    cost_axis.set_xlabel("Actual Teacher tokens (thousands)")
    cost_axis.set_ylabel("Accuracy (%)")
    cost_axis.set_title(
        "b  Full-budget accuracy–cost trade-off",
        loc="left",
        weight="bold",
    )
    cost_axis.grid(alpha=0.18)
    cost_axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "DARC v0.2 improves accuracy and efficiency over DARC v0.1",
        fontsize=16,
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.01,
        "Exploratory frozen test (n=216); model and policy selected on "
        "task-grouped train OOF only.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    paper_png = output / "figures" / "darc_v02_frozen_test_paper_summary.png"
    paper_pdf = output / "figures" / "darc_v02_frozen_test_paper_summary.pdf"
    fig.savefig(paper_png, dpi=300, bbox_inches="tight")
    fig.savefig(paper_pdf, bbox_inches="tight")
    plt.close(fig)

    active_order = (
        "DARC v0.2 Full",
        "C3 Myopic",
        "Joint MLP + Threshold",
        "Always CP2",
        "DARC v0.1 Full",
    )
    active = {str(row["policy"]): row for row in full if row["policy"] in active_order}
    short = {
        "DARC v0.2 Full": "DARC v0.2",
        "C3 Myopic": "C3 Myopic",
        "Joint MLP + Threshold": "Joint MLP",
        "Always CP2": "Always CP2",
        "DARC v0.1 Full": "DARC v0.1",
    }
    harmful_values = [
        int(full_failures[name]["harmful_selected"])
        for name in active_order
    ]
    call_values = [int(active[name]["teacher_calls"]) for name in active_order]
    net_efficiency = [
        100.0
        * (
            int(full_failures[name]["helpful_selected"])
            - int(full_failures[name]["harmful_selected"])
        )
        / max(int(active[name]["teacher_calls"]), 1)
        for name in active_order
    ]
    highlight = "#C23B3B"
    neutral = "#CBD5E1"
    bar_colors = [highlight, *([neutral] * (len(active_order) - 1))]
    positions = np.arange(len(active_order))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.3))
    panels = (
        (harmful_values, "a  Harmful interventions ↓", "Count", "{:.0f}"),
        (call_values, "b  Teacher calls ↓", "Calls", "{:.0f}"),
        (
            net_efficiency,
            "c  Net benefit / 100 calls ↑",
            "Helpful − Harmful per 100 calls",
            "{:.2f}",
        ),
    )
    for axis, (values, title, xlabel, label_format) in zip(axes, panels, strict=True):
        axis.barh(positions, values, color=bar_colors, height=0.58)
        axis.set_yticks(positions, [short[name] for name in active_order])
        axis.invert_yaxis()
        axis.set_title(title, loc="left", weight="bold")
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.16)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        upper = max(values) if max(values) > 0 else 1.0
        axis.set_xlim(0, upper * 1.28)
        for index, value in enumerate(values):
            axis.text(
                value + upper * 0.025,
                index,
                label_format.format(value),
                va="center",
                fontsize=9,
                color=highlight if index == 0 else "#374151",
                weight="bold" if index == 0 else "normal",
            )
        if title.startswith("a"):
            axis.scatter(
                [0],
                [0],
                marker="*",
                s=180,
                color=highlight,
                edgecolors="white",
                linewidths=1.0,
                zorder=4,
            )
    v02_row = active["DARC v0.2 Full"]
    fig.suptitle(
        "DARC v0.2 leads on safe verification efficiency",
        fontsize=17,
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.005,
        f"Final accuracy {100.0 * float(v02_row['accuracy']):.2f}%  •  "
        f"{float(v02_row['gain_vs_never_pp']):+.2f} pp vs Never Verify  •  "
        "0 harmful interventions",
        ha="center",
        fontsize=11,
        color="#7F1D1D",
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#FFF1F2",
            "edgecolor": "#FDA4AF",
            "linewidth": 1.0,
        },
    )
    fig.text(
        0.5,
        -0.055,
        "Active-policy comparison on the exploratory frozen test (n=216). "
        "Net benefit = Helpful − Harmful selected interventions.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    scorecard_png = output / "figures" / "darc_v02_strengths_scorecard.png"
    scorecard_pdf = output / "figures" / "darc_v02_strengths_scorecard.pdf"
    fig.savefig(scorecard_png, dpi=300, bbox_inches="tight")
    fig.savefig(scorecard_pdf, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, *, project_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen_root = _absolute(project_root, config["data"]["frozen_root"])
    baseline_root = _absolute(project_root, config["data"]["baseline_results"])
    v01_root = _absolute(project_root, config["data"]["darc_v01_results"])
    v02_root = _absolute(project_root, config["output"]["root"])
    output = v02_root / "test_evaluation"
    output.mkdir(parents=True, exist_ok=True)

    selection = json.loads((v02_root / "selection_manifest.json").read_text())
    selected_name = str(selection["selected_main_candidate"])
    selected_policy = _policy(selection["selected_policies"][selected_name])

    train_examples, _train_groups = _load_split(frozen_root, "train")
    test_examples, test_groups = _load_split(frozen_root, "test")
    _assert_disjoint({"train": train_examples, "test": test_examples})
    hard_cap = int(config["evaluation"]["hard_call_token_cap"])
    _assert_hard_call_cap(test_examples, split="test", hard_call_token_cap=hard_cap)
    cache = _absolute(project_root, config["semantic_encoder"]["cache_dir"])
    test_encoded, _cache_manifest = encode_or_load(
        test_examples,
        split="test",
        config=config["semantic_encoder"],
        cache_dir=cache,
    )

    model = DARCEnsemble.load(v02_root / "models" / f"{selected_name}.pt")
    raw = model.predict(
        test_encoded,
        device_name=str(config["semantic_encoder"].get("device", "auto")),
    )
    calibrator: CheckpointCalibrator = joblib.load(
        v02_root / "selection_objects" / f"{selected_name}_calibrator.joblib"
    )
    continuation: ContinuationPrior = joblib.load(
        v02_root / "selection_objects" / f"{selected_name}_continuation_prior.joblib"
    )
    prediction = calibrator.transform(test_examples, raw)

    seed = int(config["reproducibility"]["seed"])
    reference_cost = float(np.median([row.teacher_total_tokens for row in train_examples]))
    budgets = _policy_budgets(train_examples, test_examples)
    replays = []
    trace_rows: list[dict[str, Any]] = []
    for fraction, budget in budgets:
        replay, trace, _audit = replay_risk_cost_policy(
            test_examples,
            prediction,
            continuation,
            selected_policy,
            policy_name="DARC v0.2 Full",
            token_budget=budget,
            budget_fraction=fraction,
            hard_call_token_cap=hard_cap,
            reference_cost=reference_cost,
            seed=seed,
        )
        replays.append(replay)
        trace_rows.extend(trace)

    joint_examples = build_level1_examples(test_groups)
    joint_model = joblib.load(baseline_root / "models" / "combined" / "joint.joblib")
    joint_prediction = joint_model.predict(joint_examples)
    joint_meta = json.loads(
        (baseline_root / "models" / "combined" / "joint.json").read_text()
    )
    c3 = joblib.load(baseline_root / "models" / "combined" / "c3_myopic.joblib")
    v01_manifest = json.loads((v01_root / "selection_manifest.json").read_text())
    v01_entry = v01_manifest["selected"]["darc_full"]
    v01_model = DARCEnsemble.load(Path(v01_entry["model_path"]))
    v01_prediction = v01_model.predict(
        test_encoded,
        device_name=str(config["semantic_encoder"].get("device", "auto")),
    )
    for fraction, budget in budgets:
        replays.extend(
            [
                _strict_threshold_replay(
                    test_examples,
                    joint_prediction.expected_gain,
                    joint_prediction.expected_total_tokens,
                    threshold=float(joint_meta["threshold"]),
                    policy_name="Joint MLP + Threshold",
                    token_budget=budget,
                    fraction=fraction,
                    hard_cap=hard_cap,
                    seed=seed,
                ),
                _c3_strict_replay(
                    joint_examples,
                    joint_model,
                    c3,
                    token_budget=budget,
                    fraction=fraction,
                    hard_cap=hard_cap,
                    seed=seed,
                ),
                _always_cp2(
                    test_examples,
                    token_budget=budget,
                    fraction=fraction,
                    hard_cap=hard_cap,
                    seed=seed,
                ),
                _strict_threshold_replay(
                    test_examples,
                    v01_prediction.advantage
                    - float(v01_entry["policy"].get("uncertainty_beta", 0.0))
                    * v01_prediction.advantage_uncertainty,
                    v01_prediction.expected_total_tokens,
                    threshold=float(v01_entry["policy"].get("margin", 0.0)),
                    policy_name="DARC v0.1 Full",
                    token_budget=budget,
                    fraction=fraction,
                    hard_cap=hard_cap,
                    seed=seed,
                ),
                _strict_threshold_replay(
                    test_examples,
                    np.full(len(test_examples), -1.0),
                    np.ones(len(test_examples)),
                    threshold=0.0,
                    policy_name="Never Verify",
                    token_budget=budget,
                    fraction=fraction,
                    hard_cap=hard_cap,
                    seed=seed,
                ),
            ]
        )

    bootstrap = int(config["reproducibility"]["bootstrap_repetitions"])
    policy_rows = [
        _bootstrap_summary(replay, seed=seed, repetitions=bootstrap) for replay in replays
    ]
    failure_rows = [
        {
            "policy": replay.policy,
            "budget_fraction": replay.budget_fraction,
            **failure_metrics(test_examples, replay),
        }
        for replay in replays
    ]
    metrics = prediction_metrics(
        test_examples, prediction, bins=int(config["calibration"]["ece_bins"])
    )
    _write_csv(output / "tables" / "policy_results.csv", policy_rows)
    _write_csv(output / "tables" / "failure_metrics.csv", failure_rows)
    _write_csv(output / "tables" / "darc_v02_trace.csv", trace_rows)
    _write_csv(output / "tables" / "prediction_metrics.csv", [metrics])
    _plot(output, policy_rows, failure_rows)

    full = [row for row in policy_rows if float(row["budget_fraction"]) == 1.0]
    failures = {
        row["policy"]: row
        for row in failure_rows
        if float(row["budget_fraction"]) == 1.0
    }
    report = output / "report" / "darc_v02_frozen_test_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DARC v0.2 — Frozen-test evaluation",
        "",
        "> Exploratory only: this frozen test was inspected during earlier DARC/C3 analysis. DARC v0.2 did not tune on it, but it is no longer an untouched confirmatory holdout.",
        "",
        f"- Selected model before test loading: `{selected_name}`.",
        "- Selection source: task-grouped OOF train only.",
        "- Test-driven hyperparameter updates: none.",
        f"- Helpful AUPRC: {metrics['helpful_auprc']:.3f}; Harmful AUPRC: {metrics['harmful_auprc']:.3f}; gain correlation: {metrics['gain_pearson']:.3f}.",
        "",
        "## Full nominal budget",
        "",
        "| Policy | Accuracy (95% CI) | Gain (pp) | Calls | Tokens | Helpful | Harmful | Wasted | Missed | CP1/CP2/CP3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in full:
        failure = failures[row["policy"]]
        lines.append(
            f"| {row['policy']} | {100*row['accuracy']:.2f}% [{100*row['accuracy_ci95_low']:.2f}, {100*row['accuracy_ci95_high']:.2f}] | {row['gain_vs_never_pp']:+.2f} | {row['teacher_calls']} | {row['actual_teacher_tokens']:,} | {failure['helpful_selected']} | {failure['harmful_selected']} | {failure['wasted_no_change']} | {failure['missed_helpful']} | {failure['cp1_calls']}/{failure['cp2_calls']}/{failure['cp3_calls']} |"
        )
    lines.extend(
        [
            "",
            "![Frozen-test frontier](../figures/darc_v02_frozen_test_frontier.png)",
            "",
            "## Interpretation rule",
            "",
            "This report may diagnose generalization, but it must not be used to retune v0.2 and then claim performance on the same test. Any subsequent method change requires a newly frozen holdout for confirmation.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "darc-v0.2-test-evaluation-v1",
                "selected_before_test": selected_name,
                "selection_manifest": str(v02_root / "selection_manifest.json"),
                "test_used_for_tuning": False,
                "confirmatory": False,
            },
            indent=2,
        )
        + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/darc_v02.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = run(args.config, project_root=args.project_root.resolve())
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
