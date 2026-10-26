"""Paper-ready figures and Markdown reporting for Joint Value-Cost experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from agent_verify.analysis.joint_value_cost_experiment import read_csv

NAVY = "#0F172A"
BLUE = "#2563EB"
GREEN = "#10B981"
RED = "#EF4444"
AMBER = "#F59E0B"
PURPLE = "#7C3AED"
SLATE = "#64748B"
GRID = "#E2E8F0"
PAPER = "#F8FAFC"


def _style(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor("white")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=NAVY)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".png"), dpi=190, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def plot_dataset(output_root: Path) -> None:
    rows = read_csv(output_root / "data" / "sample_index.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), facecolor=PAPER)
    fig.suptitle(
        "Joint Value–Cost data composition and frozen split",
        x=0.055,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=NAVY,
    )
    combinations = [
        ("humaneval", "train"),
        ("humaneval", "test"),
        ("livecodebench", "train"),
        ("livecodebench", "test"),
    ]
    labels = ["HE train", "HE test", "LCB train", "LCB test"]
    tasks = [
        len({row["task_id"] for row in rows if row["dataset"] == dataset and row["partition"] == split})
        for dataset, split in combinations
    ]
    _style(axes[0], "Task composition")
    bars = axes[0].bar(labels, tasks, color=[BLUE, AMBER, GREEN, RED], zorder=3)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Tasks")
    for bar, value in zip(bars, tasks, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 2, str(value), ha="center")

    _style(axes[1], "Helpful / Harmful / No Change")
    datasets = ["humaneval", "livecodebench"]
    value_labels = ["no_change", "helpful", "harmful"]
    colors = [SLATE, GREEN, RED]
    bottoms = np.zeros(2)
    for value_label, color in zip(value_labels, colors, strict=True):
        values = np.asarray(
            [
                sum(row["dataset"] == dataset and row["value_label"] == value_label for row in rows)
                for dataset in datasets
            ]
        )
        axes[1].bar(datasets, values, bottom=bottoms, color=color, label=value_label, zorder=3)
        bottoms += values
    axes[1].legend(frameon=False)
    axes[1].set_ylabel("Checkpoint pairs")

    _style(axes[2], "Outcome distribution by checkpoint")
    checkpoints = [1, 2, 3]
    width = 0.25
    x = np.arange(3)
    for offset, (value_label, color) in enumerate(zip(value_labels, colors, strict=True)):
        values = [
            sum(int(row["checkpoint"]) == cp and row["value_label"] == value_label for row in rows)
            for cp in checkpoints
        ]
        axes[2].bar(x + (offset - 1) * width, values, width, color=color, label=value_label, zorder=3)
    axes[2].set_xticks(x, ["CP1", "CP2", "CP3"])
    axes[2].set_ylabel("Pairs")
    axes[2].legend(frameon=False)
    fig.tight_layout(rect=(0.03, 0.02, 0.98, 0.88), w_pad=2.2)
    _save(fig, output_root / "figures" / "dataset_composition")


def plot_prediction_quality(output_root: Path) -> None:
    predictions = read_csv(output_root / "evaluation" / "predictions.csv")
    calibration = read_csv(output_root / "evaluation" / "value_calibration.csv")
    selected = [
        row
        for row in predictions
        if row["train_mode"] == "combined" and row["model"] == "joint" and row["scope"] == "combined"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), facecolor=PAPER)
    fig.suptitle(
        "Joint model prediction quality — exploratory frozen test",
        x=0.055,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=NAVY,
    )
    _style(axes[0], "Predicted vs actual Teacher tokens")
    for dataset, color in (("humaneval", PURPLE), ("livecodebench", GREEN)):
        rows = [row for row in selected if row["dataset"] == dataset]
        axes[0].scatter(
            [float(row["teacher_total_tokens"]) for row in rows],
            [float(row["predicted_total_tokens"]) for row in rows],
            s=18,
            alpha=0.65,
            color=color,
            label=dataset,
        )
    maximum = max(
        max(float(row["teacher_total_tokens"]) for row in selected),
        max(float(row["predicted_total_tokens"]) for row in selected),
    )
    axes[0].plot([0, maximum], [0, maximum], color=NAVY, linestyle="--", linewidth=1)
    axes[0].set_xlabel("Actual tokens")
    axes[0].set_ylabel("Predicted tokens")
    axes[0].legend(frameon=False)

    _style(axes[1], "Value calibration")
    rows = [
        row
        for row in calibration
        if row["train_mode"] == "combined" and row["model"] == "joint" and row["scope"] == "combined"
    ]
    for target, color in (("helpful", GREEN), ("harmful", RED)):
        target_rows = [row for row in rows if row["target"] == target]
        axes[1].plot(
            [float(row["mean_predicted"]) for row in target_rows],
            [float(row["observed_rate"]) for row in target_rows],
            marker="o",
            color=color,
            label=target,
        )
    axes[1].plot([0, 1], [0, 1], color=NAVY, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed frequency")
    axes[1].legend(frameon=False)

    _style(axes[2], "Expected gain by observed outcome")
    outcome_order = ["harmful", "no_change", "helpful"]
    values = [
        [float(row["expected_gain"]) for row in selected if row["value_label"] == label]
        for label in outcome_order
    ]
    axes[2].boxplot(values, tick_labels=outcome_order, patch_artist=True)
    axes[2].axhline(0, color=NAVY, linewidth=0.8)
    axes[2].set_ylabel("Predicted expected gain")
    fig.tight_layout(rect=(0.03, 0.02, 0.98, 0.88), w_pad=2.3)
    _save(fig, output_root / "figures" / "value_cost_quality")


def plot_training_comparison(output_root: Path) -> None:
    metrics = read_csv(output_root / "evaluation" / "prediction_metrics.csv")
    rows = [row for row in metrics if row["checkpoint"] == "all"]
    modes = ["humaneval", "livecodebench", "combined"]
    scopes = ["humaneval", "livecodebench"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=PAPER)
    fig.suptitle(
        "Training configuration, Joint vs Separate, and generalization gap",
        x=0.06,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=NAVY,
    )
    width = 0.12
    x = np.arange(len(modes))
    for scope_index, scope in enumerate(scopes):
        for model_index, model in enumerate(("joint", "separate")):
            values = [
                float(
                    next(
                        row["helpful_auprc"]
                        for row in rows
                        if row["train_mode"] == mode
                        and row["model"] == model
                        and row["scope"] == scope
                    )
                )
                for mode in modes
            ]
            offset = ((scope_index * 2 + model_index) - 1.5) * width
            color = [PURPLE, BLUE, GREEN, AMBER][scope_index * 2 + model_index]
            axes[0].bar(
                x + offset,
                values,
                width,
                color=color,
                label=f"{scope}/{model}",
                zorder=3,
            )
    _style(axes[0], "Helpful AUPRC")
    axes[0].set_xticks(x, ["HE-only", "LCB-only", "Combined"])
    axes[0].legend(frameon=False, fontsize=8)

    _style(axes[1], "Cost mean relative error")
    for model_index, model in enumerate(("joint", "separate")):
        values = [
            float(
                next(
                    row["cost_mean_relative_error"]
                    for row in rows
                    if row["train_mode"] == mode
                    and row["model"] == model
                    and row["scope"] == "combined"
                )
            )
            for mode in modes
        ]
        axes[1].bar(
            x + (model_index - 0.5) * 0.3,
            values,
            0.3,
            color=BLUE if model == "joint" else AMBER,
            label=model,
            zorder=3,
        )
    axes[1].set_xticks(x, ["HE-only", "LCB-only", "Combined"])
    axes[1].legend(frameon=False)
    fig.tight_layout(rect=(0.03, 0.02, 0.98, 0.86), w_pad=2.5)
    _save(fig, output_root / "figures" / "training_and_ablation")


def plot_budget_frontier(output_root: Path) -> None:
    rows = read_csv(output_root / "budget_evaluation" / "policy_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), facecolor=PAPER)
    fig.suptitle(
        "Accuracy–Teacher-token Pareto frontiers",
        x=0.05,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=NAVY,
    )
    for ax, scope in zip(axes, ("humaneval", "livecodebench", "combined"), strict=True):
        _style(ax, scope)
        scope_rows = [
            row
            for row in rows
            if row["scope"] == scope
            and row["admission_mode"] == "retrospective_hard_cap"
            and (
                row["train_mode"] == "baseline"
                or row["train_mode"] == "combined"
            )
        ]
        for label, color in (
            ("joint", GREEN),
            ("separate", BLUE),
            ("Legacy Logistic", PURPLE),
            ("Legacy Boosting", AMBER),
            ("Offline Oracle", RED),
        ):
            model_rows = [row for row in scope_rows if row["model"] == label]
            if not model_rows:
                continue
            model_rows.sort(key=lambda row: int(row["teacher_total_tokens"]))
            ax.plot(
                [int(row["teacher_total_tokens"]) for row in model_rows],
                [100 * float(row["accuracy"]) for row in model_rows],
                marker="o",
                linewidth=1.5,
                color=color,
                label=label,
            )
        ax.set_xlabel("Actual Teacher tokens")
        ax.set_ylabel("Final accuracy (%)")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.86), w_pad=2.0)
    _save(fig, output_root / "figures" / "accuracy_cost_pareto")


def generate_visualizations(output_root: Path) -> None:
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    plot_dataset(output_root)
    plot_prediction_quality(output_root)
    plot_training_comparison(output_root)
    plot_budget_frontier(output_root)


def _percent(value: str | float) -> str:
    return f"{100 * float(value):.3f}%"


def generate_report(config: dict[str, Any], output_root: Path) -> Path:
    split_rows = read_csv(output_root / "data" / "split_summary.csv")
    metrics = [
        row
        for row in read_csv(output_root / "evaluation" / "prediction_metrics.csv")
        if row["checkpoint"] == "all"
    ]
    budget = read_csv(output_root / "budget_evaluation" / "policy_results.csv")
    selected = {
        mode: json.loads(
            (output_root / "training" / mode / "selected_models.json").read_text(
                encoding="utf-8"
            )
        )
        for mode in ("humaneval", "livecodebench", "combined")
    }
    report_path = output_root / "report" / "joint_value_cost_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Joint Value–Cost Learning for Budgeted Verification",
        "",
        "> Status: exploratory evaluation on a previously observed frozen test split. "
        "These results are not a final untouched estimate.",
        "",
        "## Experimental settings",
        "",
        "The workflow, Student, Teacher, prompts, checkpoints, and intervention semantics remain frozen. "
        "No new LLM or evaluator call is performed. The experiment learns value and Teacher-only token "
        "cost from the previously collected paired counterfactual records.",
        "",
        "## Data split",
        "",
        "| Partition | Dataset | Tasks | Pairs | Helpful | Harmful | No Change | Teacher tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_rows:
        lines.append(
            f"| {row['partition']} | {row['dataset']} | {row['tasks']} | {row['pairs']} | "
            f"{row['helpful']} | {row['harmful']} | {row['no_change']} | "
            f"{int(float(row['teacher_total_tokens'])):,} |"
        )
    lines.extend(
        [
            "",
            "Split groups are `(dataset, task_id)`; CP1/CP2/CP3 are co-located. The exact Phase-2 "
            "frozen manifest is reused. All encoders, scaling, calibration, model selection, loss weights, "
            "and policy thresholds are fitted inside the training partition only.",
            "",
            "![Dataset composition](../figures/dataset_composition.png)",
            "",
            "## Model architecture",
            "",
            "Joint: TF-IDF → train-fitted SVD + structured metadata → shared tanh layer → "
            "three-class Value head and robust log-token Cost head. The Value head predicts Helpful, "
            "Harmful, and No Change. The Cost head predicts conditional mean and upper quantile. "
            "Expected Gain is `P(Helpful) - P(Harmful)`.",
            "",
            "Separate baseline: identical feature encoder, independent multinomial Logistic value model, "
            "Huber log-cost regressor, and quantile cost regressor.",
            "",
            "## Hyperparameter selection",
            "",
        ]
    )
    for mode, payload in selected.items():
        lines.append(f"### {mode}")
        lines.append("")
        for model, selection in payload.items():
            lines.append(
                f"- **{model}:** `{json.dumps(selection['params'], sort_keys=True)}`; "
                f"temperature={selection['temperature']:.3f}; gain/1k threshold="
                f"{selection['gain_per_1k_threshold']:.4f}."
            )
        lines.append("")
    lines.extend(
        [
            "## Prediction quality and cross-domain results",
            "",
            "| Train | Model | Test | Helpful AUPRC | Harmful AUPRC | Gain r | Cost MAE | Cost rel. error | Q coverage |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['train_mode']} | {row['model']} | {row['scope']} | "
            f"{float(row['helpful_auprc']):.4f} | {float(row['harmful_auprc']):.4f} | "
            f"{float(row['gain_pearson']):.4f} | {float(row['cost_mae']):.1f} | "
            f"{float(row['cost_mean_relative_error']):.4f} | "
            f"{float(row['cost_upper_coverage']):.4f} |"
        )
    lines.extend(
        [
            "",
            "![Value and cost quality](../figures/value_cost_quality.png)",
            "",
            "![Training and ablation](../figures/training_and_ablation.png)",
            "",
            "## Budget-aware offline replay",
            "",
            "The controller visits CP1→CP2→CP3 and selects the first checkpoint whose positive expected "
            "gain per 1,000 predicted tokens exceeds a training-only threshold. At most one Teacher call "
            "is allowed per task.",
            "",
            "`expected_cost` admission is deployable but can overshoot the actual budget. "
            "`retrospective_hard_cap` prevents overshoot using observed cost and is reported only as an "
            "accounting upper bound, not as deployable foresight.",
            "",
            "![Accuracy–cost Pareto](../figures/accuracy_cost_pareto.png)",
            "",
            "## Joint vs Separate ablation",
            "",
            "Joint and Separate use the same input schema and frozen split. Compare their rows above at "
            "the same actual Teacher-token budget; no winner is selected or retuned using frozen-test outcomes.",
            "",
            "## Limitations",
            "",
            "- Helpful and especially Harmful labels are rare; uncertainty is large.",
            "- The frozen test has already been inspected in earlier phases, so all results are exploratory.",
            "- Predicted cost cannot guarantee a hard cap; only retrospective replay can enforce it exactly.",
            "- Counterfactual replay is valid only for the frozen one-intervention workflow and does not model repeated verification.",
            "- Upper-quantile coverage is empirical and not conformal coverage.",
            "- No conclusion here justifies changing Student, Teacher, prompts, or verification semantics.",
            "",
            "## Reproducibility",
            "",
            f"Seed: `{config['experiment']['seed']}`. Full machine-readable outputs are stored next to this report.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
