"""Visualizations and research report for the C3 sequential controller."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from agent_verify.analysis.joint_value_cost_experiment import read_csv

COLORS = {
    "C3 continuation": "#0F766E",
    "C3 no continuation": "#14B8A6",
    "Joint threshold": "#2563EB",
    "Separate value-cost": "#7C3AED",
    "Legacy Boosting": "#F59E0B",
    "Always CP1": "#94A3B8",
    "Always CP2": "#A78BFA",
    "Always CP3": "#FCA5A5",
    "Random Verification": "#84CC16",
    "Never Verify": "#64748B",
    "Offline Oracle": "#EF4444",
}
PAPER = "#F8FAFC"
GRID = "#E2E8F0"
INK = "#0F172A"


def _label(row: dict[str, str]) -> str | None:
    if row.get("source") == "c3" and row.get("train_mode") == "combined":
        if row["model"] == "c3_continuation":
            return "C3 continuation"
        if row["model"] == "c3_myopic":
            return "C3 no continuation"
    if row.get("source") == "threshold_or_separate":
        if row["model"] == "joint":
            return "Joint threshold"
        if row["model"] == "separate":
            return "Separate value-cost"
    if row.get("source") == "existing_baseline":
        if row["model"] in {
            "Legacy Boosting",
            "Always CP1",
            "Always CP2",
            "Always CP3",
            "Random Verification",
            "Never Verify",
            "Offline Oracle",
        }:
            return row["model"]
    return None


def _style(axis: plt.Axes, title: str) -> None:
    axis.set_facecolor("white")
    axis.set_title(title, loc="left", fontweight="bold", color=INK)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _save(figure: plt.Figure, base: Path) -> None:
    figure.savefig(
        base.with_suffix(".png"),
        dpi=190,
        bbox_inches="tight",
        facecolor=PAPER,
    )
    figure.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor=PAPER,
    )
    plt.close(figure)


def plot_accuracy_cost(output_root: Path) -> None:
    rows = read_csv(output_root / "evaluation" / "policy_comparison.csv")
    rows = [
        row
        for row in rows
        if row["admission_mode"] == "retrospective_hard_cap"
        and _label(row) is not None
    ]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=PAPER)
    figure.suptitle(
        "C3 accuracy–Teacher-token frontier",
        x=0.05,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    for axis, scope in zip(
        axes, ("humaneval", "livecodebench", "combined"), strict=True
    ):
        _style(axis, scope)
        scope_rows = [row for row in rows if row["scope"] == scope]
        for label in COLORS:
            selected = [row for row in scope_rows if _label(row) == label]
            if not selected:
                continue
            selected.sort(key=lambda row: int(row["teacher_total_tokens"]))
            axis.plot(
                [int(row["teacher_total_tokens"]) for row in selected],
                [100.0 * float(row["accuracy"]) for row in selected],
                marker="o",
                linewidth=1.6,
                color=COLORS[label],
                label=label,
            )
            pareto = [
                row
                for row in selected
                if row.get("pareto_efficient_global", "False").lower() == "true"
            ]
            axis.scatter(
                [int(row["teacher_total_tokens"]) for row in pareto],
                [100.0 * float(row["accuracy"]) for row in pareto],
                marker="*",
                s=80,
                color=COLORS[label],
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )
        axis.set_xlabel("Actual Teacher tokens")
        axis.set_ylabel("Final accuracy (%)")
        axis.legend(frameon=False, fontsize=7)
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.88))
    _save(figure, output_root / "figures" / "c3_accuracy_cost_frontier")


def _max_budget_rows(output_root: Path) -> list[dict[str, str]]:
    rows = read_csv(output_root / "evaluation" / "policy_comparison.csv")
    result: list[dict[str, str]] = []
    for scope in ("humaneval", "livecodebench", "combined"):
        scoped = [
            row
            for row in rows
            if row["scope"] == scope
            and row["admission_mode"] == "retrospective_hard_cap"
            and _label(row) is not None
        ]
        maximum = max(int(row["token_budget"]) for row in scoped)
        result.extend(
            row for row in scoped if int(row["token_budget"]) == maximum
        )
    return result


def plot_selection_behavior(output_root: Path) -> None:
    rows = _max_budget_rows(output_root)
    labels = [
        "C3 continuation",
        "C3 no continuation",
        "Joint threshold",
        "Legacy Boosting",
    ]
    combined = [row for row in rows if row["scope"] == "combined"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=PAPER)
    figure.suptitle(
        "Intervention quality and checkpoint allocation — Combined test",
        x=0.06,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=INK,
    )

    _style(axes[0], "Helpful vs harmful selections")
    x = np.arange(len(labels))
    helpful = [
        int(next(row for row in combined if _label(row) == label)["helpful_selected"])
        for label in labels
    ]
    harmful = [
        int(next(row for row in combined if _label(row) == label)["harmful_selected"])
        for label in labels
    ]
    axes[0].bar(x - 0.18, helpful, 0.36, color="#10B981", label="Helpful")
    axes[0].bar(x + 0.18, harmful, 0.36, color="#EF4444", label="Harmful")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("Selected interventions")
    axes[0].legend(frameon=False)

    _style(axes[1], "Checkpoint selection distribution")
    bottoms = np.zeros(len(labels))
    for checkpoint, color in zip(
        (1, 2, 3), ("#2563EB", "#F59E0B", "#7C3AED"), strict=True
    ):
        values = [
            int(next(row for row in combined if _label(row) == label)[f"cp{checkpoint}_calls"])
            for label in labels
        ]
        axes[1].bar(
            x,
            values,
            bottom=bottoms,
            color=color,
            label=f"CP{checkpoint}",
        )
        bottoms += np.asarray(values)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("Teacher calls")
    axes[1].legend(frameon=False)
    figure.tight_layout(rect=(0.03, 0.02, 0.99, 0.86))
    _save(figure, output_root / "figures" / "c3_selection_behavior")


def plot_continuation_ablation(output_root: Path) -> None:
    rows = read_csv(output_root / "evaluation" / "c3_policy_results.csv")
    rows = [
        row
        for row in rows
        if row["train_mode"] == "combined"
        and row["admission_mode"] == "retrospective_hard_cap"
    ]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=PAPER)
    figure.suptitle(
        "Learned continuation-value ablation",
        x=0.05,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=INK,
    )
    for axis, scope in zip(
        axes, ("humaneval", "livecodebench", "combined"), strict=True
    ):
        _style(axis, scope)
        for model, label in (
            ("c3_continuation", "C3 continuation"),
            ("c3_myopic", "C3 no continuation"),
        ):
            selected = [
                row for row in rows if row["scope"] == scope and row["model"] == model
            ]
            selected.sort(key=lambda row: int(row["token_budget"]))
            axis.plot(
                [int(row["token_budget"]) for row in selected],
                [100.0 * float(row["accuracy"]) for row in selected],
                marker="o",
                color=COLORS[label],
                label=label,
            )
        axis.set_xlabel("Token-budget cap")
        axis.set_ylabel("Accuracy (%)")
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.86))
    _save(figure, output_root / "figures" / "c3_continuation_ablation")


def plot_training_scope_transfer(output_root: Path) -> None:
    rows = read_csv(output_root / "evaluation" / "c3_policy_results.csv")
    rows = [
        row
        for row in rows
        if row["model"] == "c3_continuation"
        and row["admission_mode"] == "retrospective_hard_cap"
    ]
    mode_colors = {
        "humaneval": "#2563EB",
        "livecodebench": "#F59E0B",
        "combined": "#0F766E",
    }
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=PAPER)
    figure.suptitle(
        "C3 training-scope transfer",
        x=0.05,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=INK,
    )
    for axis, scope in zip(
        axes, ("humaneval", "livecodebench", "combined"), strict=True
    ):
        _style(axis, f"Test: {scope}")
        for mode, color in mode_colors.items():
            selected = [
                row
                for row in rows
                if row["scope"] == scope and row["train_mode"] == mode
            ]
            selected.sort(key=lambda row: int(row["teacher_total_tokens"]))
            axis.plot(
                [int(row["teacher_total_tokens"]) for row in selected],
                [100.0 * float(row["accuracy"]) for row in selected],
                marker="o",
                color=color,
                label=f"Train: {mode}",
            )
        axis.set_xlabel("Actual Teacher tokens")
        axis.set_ylabel("Final accuracy (%)")
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.86))
    _save(figure, output_root / "figures" / "c3_training_scope_transfer")


def generate_c3_visualizations(output_root: Path) -> None:
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    plot_accuracy_cost(output_root)
    plot_selection_behavior(output_root)
    plot_continuation_ablation(output_root)
    plot_training_scope_transfer(output_root)


def generate_c3_report(config: dict[str, Any], output_root: Path) -> Path:
    manifest = json.loads(
        (output_root / "data_manifest.json").read_text(encoding="utf-8")
    )
    rows = _max_budget_rows(output_root)
    selected = {
        mode: json.loads(
            (
                output_root
                / "training"
                / mode
                / "selected_controllers.json"
            ).read_text(encoding="utf-8")
        )
        for mode in ("humaneval", "livecodebench", "combined")
    }
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = report_dir / f"c3_sequential_controller_report_{timestamp}.md"
    if report.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {report}")
    lines = [
        "# C3 — Learned Sequential Budget Controller",
        "",
        "> Exploratory evaluation on the previously observed frozen test split.",
        "",
        "## Objective and scope",
        "",
        "C3 reuses the frozen Joint Value–Cost MLP and learns checkpoint-specific "
        "Q(Verify Now) and Q(Wait) models. It does not call or retrain the "
        "Student or Teacher and permits at most one Teacher verification per task.",
        "",
        "## Data and leakage controls",
        "",
        f"- Tasks/pairs: {manifest['tasks']}/{manifest['pairs']}.",
        f"- Frozen train/test groups: {manifest['train_tasks']}/{manifest['test_tasks']}.",
        "- CP1/CP2/CP3 remain grouped by (dataset, task_id).",
        "- Joint representations and predictions are frozen upstream inputs.",
        "- Future checkpoint states are used only to construct continuation targets.",
        "- Online feasibility uses predicted cost; actual cost never enters an online feature.",
        "",
        "## Q-target construction",
        "",
        "Q(Verify Now) uses observed Y_verify minus a validation-selected, "
        "budget-dependent Teacher-cost shadow price. Q(Wait) equals Y_skip at "
        "CP3. At CP2 and CP1 it bootstraps the best feasible fitted action value at "
        "the next checkpoint. Training therefore proceeds CP3 → CP2 → CP1.",
        "",
        "The no-continuation ablation replaces every Wait target with Y_skip. "
        "This isolates the effect of learned continuation value.",
        "",
        "## Controlled approximation",
        "",
        "The data contain single-intervention counterfactuals, not trajectories with "
        "multiple Teacher calls. C3 therefore terminates a task after Verify and only "
        "bootstraps along the all-Skip baseline trajectory. Cross-task opportunity cost "
        "is approximated using budget per remaining task and a cost shadow price; it is "
        "not claimed to be an exact global-budget MDP solution.",
        "",
        "## Selected controller settings",
        "",
    ]
    for mode, variants in selected.items():
        lines.extend([f"### {mode}", ""])
        for variant, payload in variants.items():
            lines.append(
                f"- **{variant}:** {json.dumps(payload['params'], sort_keys=True)}"
            )
        lines.append("")

    lines.extend(
        [
            "## Main results at the largest common budget",
            "",
            "All 95% intervals use task-clustered bootstrap resampling.",
            "",
            "| Test | Policy | Accuracy (95% CI) | Gain vs Never (95% CI) | Calls | Actual tokens | Helpful | Harmful | CP1/CP2/CP3 | CP2 share |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    order = [
        "Never Verify",
        "C3 continuation",
        "C3 no continuation",
        "Joint threshold",
        "Separate value-cost",
        "Legacy Boosting",
        "Always CP1",
        "Always CP2",
        "Always CP3",
        "Random Verification",
        "Offline Oracle",
    ]
    for scope in ("humaneval", "livecodebench", "combined"):
        scoped = [row for row in rows if row["scope"] == scope]
        for label in order:
            matches = [row for row in scoped if _label(row) == label]
            if not matches:
                continue
            row = matches[0]
            lines.append(
                f"| {scope} | {label} | {100 * float(row['accuracy']):.2f}% "
                f"[{100 * float(row['accuracy_ci95_low']):.2f}, "
                f"{100 * float(row['accuracy_ci95_high']):.2f}] | "
                f"{float(row['gain_vs_never_pp']):+.2f} pp "
                f"[{float(row['gain_ci95_low_pp']):+.2f}, "
                f"{float(row['gain_ci95_high_pp']):+.2f}] | {row['teacher_calls']} | "
                f"{int(row['teacher_total_tokens']):,} | {row['helpful_selected']} | "
                f"{row['harmful_selected']} | {row['cp1_calls']}/{row['cp2_calls']}/{row['cp3_calls']} | "
                f"{100 * float(row['cp2_call_share']):.1f}% |"
            )

    lines.extend(
        [
            "",
            "![Accuracy cost frontier](../figures/c3_accuracy_cost_frontier.png)",
            "",
            "![Selection behavior](../figures/c3_selection_behavior.png)",
            "",
            "## Continuation-value ablation",
            "",
            "Compare C3 continuation with C3 no continuation at each common budget. "
            "A gain is attributable to sequential learning only when the continuation "
            "variant improves accuracy/cost without test-time retuning.",
            "",
            "![Continuation ablation](../figures/c3_continuation_ablation.png)",
            "",
            "## HumanEval-only, LiveCodeBench-only, and Combined training",
            "",
            "This transfer plot holds the C3 algorithm fixed and changes only the "
            "pretrained Joint/controller training scope. It is reported for diagnosis, "
            "not for selecting a winner on the previously inspected test split.",
            "",
            "![Training-scope transfer](../figures/c3_training_scope_transfer.png)",
            "",
            "## Budget semantics",
            "",
            "- expected_cost: deployable predicted-cost admission; actual tokens may overshoot.",
            "- retrospective_hard_cap: strict accounting using observed current cost; "
            "reported only as a non-deployable comparison.",
            "",
            "## Limitations",
            "",
            "- Helpful/Harmful labels are rare, especially on LiveCodeBench.",
            "- Fitted continuation targets use the observed all-Skip future trajectory.",
            "- The global budget shadow price is an approximation, not Offline RL.",
            "- The pretrained Joint representation is frozen but was fitted on the full frozen "
            "training partition before controller cross-validation; this is not a fully nested "
            "end-to-end validation of the upstream representation.",
            "- The frozen test has been inspected in earlier phases; results are exploratory.",
            "- No multi-verification trajectory is inferred or fabricated.",
            "",
            f"Seed: {config['experiment']['seed']}.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
