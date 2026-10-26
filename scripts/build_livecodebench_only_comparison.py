#!/usr/bin/env python3
"""Compare LiveCodeBench-only and combined training on the same frozen LCB tasks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METHODS = ("Ours Logistic", "Ours Boosting", "Optimized Boosting")
DISPLAY_NAMES = {
    "Ours Logistic": "Semantic Logistic",
    "Ours Boosting": "Compact Boosting",
    "Optimized Boosting": "Optimized Boosting",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-dir", type=Path, default=Path("results/combined_phase2_optimized"))
    parser.add_argument(
        "--lcb-only-dir", type=Path, default=Path("results/livecodebench_only_phase2_optimized")
    )
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def policy_outcomes(path: Path, method: str) -> tuple[dict[str, float], int, int]:
    rows = [row for row in read_csv(path) if row["model"] == method]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    outcomes: dict[str, float] = {}
    calls = 0
    tokens = 0
    for task_id, task_rows in grouped.items():
        selected = [row for row in task_rows if int(row["selected_by_controller"]) == 1]
        if len(selected) > 1:
            raise ValueError(f"Multiple checkpoints selected for {task_id}")
        if selected:
            outcomes[task_id] = float(selected[0]["y_verify"])
            calls += 1
        else:
            outcomes[task_id] = float(task_rows[0]["y_skip"])
    return outcomes, calls, tokens


def policy_metric(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    return next(row for row in rows if row["benchmark"] == "livecodebench" and row["method"] == method)


def predictor_metric(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    return next(row for row in rows if row["benchmark"] == "livecodebench" and row["model"] == method)


def main() -> None:
    args = parse_args()
    combined_assignments = read_csv(args.combined_dir / "split_assignments.csv")
    lcb_assignments = read_csv(args.lcb_only_dir / "split_assignments.csv")
    combined_test = {
        row["task_id"]
        for row in combined_assignments
        if row["benchmark"] == "livecodebench" and row["split"] == "test"
    }
    lcb_test = {row["task_id"] for row in lcb_assignments if row["split"] == "test"}
    if combined_test != lcb_test:
        raise ValueError("Combined and LCB-only frozen test task sets differ")

    policy_sources = {
        "Combined": read_csv(args.combined_dir / "policy_comparison_by_benchmark.csv"),
        "LCB-only": read_csv(args.lcb_only_dir / "policy_comparison_by_benchmark.csv"),
    }
    predictor_sources = {
        "Combined": read_csv(args.combined_dir / "predictor_metrics_by_benchmark.csv"),
        "LCB-only": read_csv(args.lcb_only_dir / "predictor_metrics_by_benchmark.csv"),
    }
    prediction_dirs = {
        "Combined": args.combined_dir / "predictions",
        "LCB-only": args.lcb_only_dir / "predictions",
    }
    file_names = {
        "Ours Logistic": "livecodebench_compact.csv",
        "Ours Boosting": "livecodebench_compact.csv",
        "Optimized Boosting": "livecodebench_optimized_boosting.csv",
    }
    prediction_model_names = {
        "Ours Logistic": "Ours Logistic",
        "Ours Boosting": "Ours Boosting",
        # The optimized CSV keeps its underlying estimator-family display name.
        "Optimized Boosting": "Ours Boosting",
    }

    rng = np.random.default_rng(args.seed)
    comparison: list[dict[str, object]] = []
    for method in METHODS:
        prediction_model = prediction_model_names[method]
        combined_outcomes, _, _ = policy_outcomes(
            prediction_dirs["Combined"] / file_names[method], prediction_model
        )
        lcb_outcomes, _, _ = policy_outcomes(
            prediction_dirs["LCB-only"] / file_names[method], prediction_model
        )
        task_ids = sorted(combined_test)
        if set(combined_outcomes) != combined_test or set(lcb_outcomes) != combined_test:
            raise ValueError(f"Prediction universe mismatch for {method}")
        combined_values = np.asarray([combined_outcomes[task_id] for task_id in task_ids])
        lcb_values = np.asarray([lcb_outcomes[task_id] for task_id in task_ids])
        difference = lcb_values - combined_values
        samples = rng.integers(0, len(task_ids), size=(args.bootstrap_reps, len(task_ids)))
        bootstrap = difference[samples].mean(axis=1) * 100
        low, high = np.percentile(bootstrap, [2.5, 97.5])
        combined_policy = policy_metric(policy_sources["Combined"], method)
        lcb_policy = policy_metric(policy_sources["LCB-only"], method)
        combined_predictor = predictor_metric(predictor_sources["Combined"], method)
        lcb_predictor = predictor_metric(predictor_sources["LCB-only"], method)
        comparison.append(
            {
                "model": DISPLAY_NAMES[method],
                "test_tasks": len(task_ids),
                "combined_accuracy": float(combined_values.mean()),
                "lcb_only_accuracy": float(lcb_values.mean()),
                "lcb_only_minus_combined_pp": float(difference.mean() * 100),
                "paired_ci95_low_pp": float(low),
                "paired_ci95_high_pp": float(high),
                "combined_calls": int(float(combined_policy["teacher_calls"])),
                "lcb_only_calls": int(float(lcb_policy["teacher_calls"])),
                "combined_tokens": int(float(combined_policy["teacher_tokens"])),
                "lcb_only_tokens": int(float(lcb_policy["teacher_tokens"])),
                "combined_helpful_auroc": float(combined_predictor["helpful_auroc"]),
                "lcb_only_helpful_auroc": float(lcb_predictor["helpful_auroc"]),
                "combined_helpful_ap": float(combined_predictor["helpful_average_precision"]),
                "lcb_only_helpful_ap": float(lcb_predictor["helpful_average_precision"]),
            }
        )
    write_csv(args.lcb_only_dir / "training_mode_comparison.csv", comparison)

    combined_selection = json.loads(
        (args.combined_dir / "inner_selections.json").read_text(encoding="utf-8")
    )["optimized_boosting"]
    lcb_selection = json.loads((args.lcb_only_dir / "inner_selections.json").read_text(encoding="utf-8"))[
        "optimized_boosting"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), facecolor="#F8FAFC")
    fig.suptitle(
        "LiveCodeBench frozen test — Combined vs LiveCodeBench-only training",
        x=0.055,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.055,
        0.90,
        "Same 35 test tasks • tuning uses train-only GroupKFold • paired task bootstrap (20,000 reps)",
        fontsize=10,
        color="#475569",
    )
    colors = {"Combined": "#2563EB", "LCB-only": "#10B981"}
    names = [row["model"] for row in comparison]
    x = np.arange(len(names))
    width = 0.34

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    combined_accuracy = [100 * float(row["combined_accuracy"]) for row in comparison]
    lcb_accuracy = [100 * float(row["lcb_only_accuracy"]) for row in comparison]
    axes[0].bar(x - width / 2, combined_accuracy, width, color=colors["Combined"], label="Combined", zorder=3)
    axes[0].bar(x + width / 2, lcb_accuracy, width, color=colors["LCB-only"], label="LCB-only", zorder=3)
    axes[0].set_title("Final accuracy", loc="left", fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 28)
    axes[0].set_xticks(x, ["Logistic", "Compact\nBoosting", "Optimized\nBoosting"])
    axes[0].legend(frameon=False)
    for index, values in enumerate(zip(combined_accuracy, lcb_accuracy, strict=True)):
        for offset, value in zip((-width / 2, width / 2), values, strict=True):
            axes[0].text(index + offset, value + 0.6, f"{value:.2f}%", ha="center", fontsize=8)

    combined_calls = [int(row["combined_calls"]) for row in comparison]
    lcb_calls = [int(row["lcb_only_calls"]) for row in comparison]
    axes[1].bar(x - width / 2, combined_calls, width, color=colors["Combined"], zorder=3)
    axes[1].bar(x + width / 2, lcb_calls, width, color=colors["LCB-only"], zorder=3)
    axes[1].set_title("Teacher calls", loc="left", fontweight="bold")
    axes[1].set_ylabel("Calls / 35 tasks")
    axes[1].set_xticks(x, ["Logistic", "Compact\nBoosting", "Optimized\nBoosting"])
    for index, values in enumerate(zip(combined_calls, lcb_calls, strict=True)):
        for offset, value in zip((-width / 2, width / 2), values, strict=True):
            axes[1].text(index + offset, value + 0.7, str(value), ha="center", fontsize=8)

    combined_ap = [float(row["combined_helpful_ap"]) for row in comparison]
    lcb_ap = [float(row["lcb_only_helpful_ap"]) for row in comparison]
    axes[2].bar(x - width / 2, combined_ap, width, color=colors["Combined"], zorder=3)
    axes[2].bar(x + width / 2, lcb_ap, width, color=colors["LCB-only"], zorder=3)
    axes[2].set_title("Helpful average precision", loc="left", fontweight="bold")
    axes[2].set_ylabel("Average precision")
    axes[2].set_xticks(x, ["Logistic", "Compact\nBoosting", "Optimized\nBoosting"])

    fig.text(
        0.055,
        0.025,
        "LCB-only optimized config: "
        f"leaves={lcb_selection['params']['max_leaf_nodes']}, "
        f"L2={lcb_selection['params']['l2_regularization']}, "
        f"SVD={lcb_selection['params']['svd_components']}, thresholds="
        f"({lcb_selection['threshold_cp1']:.2f}, {lcb_selection['threshold_cp2']:.2f}, "
        f"{lcb_selection['threshold_cp3']:.2f})  |  Combined thresholds="
        f"({combined_selection['threshold_cp1']:.2f}, {combined_selection['threshold_cp2']:.2f}, "
        f"{combined_selection['threshold_cp3']:.2f})",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.04, 0.07, 0.98, 0.86), w_pad=2.5)
    output = args.lcb_only_dir / "livecodebench_training_mode_comparison.png"
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Comparison CSV: {args.lcb_only_dir / 'training_mode_comparison.csv'}")
    print(f"Comparison figure: {output}")


if __name__ == "__main__":
    main()
