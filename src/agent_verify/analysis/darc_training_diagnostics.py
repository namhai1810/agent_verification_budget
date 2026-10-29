"""Visualize DARC optimization traces stored inside trained checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from agent_verify.analysis.darc_model import DARCEnsemble

_CANDIDATES = (
    "focal",
    "balanced",
    "balanced_ranking_ablation",
)
_DISPLAY = {
    "focal": "Focal CE (selected)",
    "balanced": "Balanced CE",
    "balanced_ranking_ablation": "Balanced CE + Ranking",
}
_COLORS = {
    "focal": "#C23B3B",
    "balanced": "#4C78A8",
    "balanced_ranking_ablation": "#59A14F",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    histories: list[tuple[dict[str, float | int], ...]],
    key: str,
    *,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = min(len(history) for history in histories)
    values = np.asarray(
        [
            [float(history[index][key]) * scale for index in range(length)]
            for history in histories
        ],
        dtype=float,
    )
    epochs = np.arange(1, length + 1)
    return epochs, values.mean(axis=0), values.min(axis=0), values.max(axis=0)


def write_training_diagnostics(
    result_root: Path,
    *,
    max_epochs: int = 120,
) -> dict[str, Any]:
    """Create optimization plots without retraining any model.

    Final-refit histories monitor the same training data used for fitting because
    the refit epoch count is already fixed by grouped cross-validation. Therefore
    these traces diagnose optimization only; grouped-CV selected epochs carry the
    held-out model-selection signal.
    """

    models: dict[str, DARCEnsemble] = {}
    histories: dict[str, list[tuple[dict[str, float | int], ...]]] = {}
    history_rows: list[dict[str, Any]] = []
    for candidate in _CANDIDATES:
        model = DARCEnsemble.load(result_root / "models" / f"{candidate}.pt")
        models[candidate] = model
        histories[candidate] = [member.history for member in model.members]
        for member_index, member in enumerate(model.members, start=1):
            for row in member.history:
                history_rows.append(
                    {
                        "candidate": candidate,
                        "member": member_index,
                        **row,
                    }
                )
    _write_csv(result_root / "tables" / "training_history.csv", history_rows)

    cv_rows = _read_csv(result_root / "tables" / "grouped_cv.csv")
    cv_epochs: dict[str, list[int]] = defaultdict(list)
    for row in cv_rows:
        cv_epochs[row["candidate"]].append(int(float(row["median_best_epoch"])))

    figure_root = result_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))

    selected_histories = histories["focal"]
    epochs, mean, lower, upper = _aggregate(selected_histories, "total_loss")
    axes[0].fill_between(epochs, lower, upper, color="#FCA5A5", alpha=0.35, linewidth=0)
    axes[0].plot(epochs, mean, color=_COLORS["focal"], linewidth=2.2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training objective (log scale)")
    axes[0].set_title("a  Selected Focal refit", loc="left", weight="bold")
    axes[0].grid(alpha=0.18)
    axes[0].text(
        0.04,
        0.06,
        f"{mean[0]:.3f} → {mean[-1]:.3f}\n{len(epochs)} fixed refit epochs",
        transform=axes[0].transAxes,
        fontsize=9,
        color="#7F1D1D",
        weight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF1F2", "edgecolor": "#FDA4AF"},
    )

    components = (
        ("value_loss", 1.0, "Focal classification", "#C23B3B"),
        ("cost_mean_loss", 0.25, "Expected-cost contribution", "#4C78A8"),
        ("cost_quantile_loss", 0.20, "Upper-cost contribution", "#59A14F"),
    )
    for key, scale, label, color in components:
        component_epochs, component_mean, _low, _high = _aggregate(
            selected_histories,
            key,
            scale=scale,
        )
        axes[1].plot(
            component_epochs,
            np.maximum(component_mean, 1e-8),
            label=label,
            color=color,
            linewidth=2.0,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted loss contribution")
    axes[1].set_title("b  Objective decomposition", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.18)

    candidate_positions = np.arange(len(_CANDIDATES))
    fold_markers = ("o", "s", "^")
    for position, candidate in zip(candidate_positions, _CANDIDATES, strict=True):
        values = cv_epochs[candidate]
        for fold_index, (value, marker) in enumerate(
            zip(values, fold_markers, strict=True),
            start=1,
        ):
            axes[2].scatter(
                [value],
                [position],
                marker=marker,
                s=100 if candidate == "focal" else 78,
                color=_COLORS[candidate],
                edgecolors="white",
                linewidths=1.0,
                zorder=3,
            )
            if candidate == "focal":
                axes[2].annotate(
                    str(value),
                    (value, position),
                    xytext=(0, 9 + 2 * fold_index),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
    axes[2].axvline(
        max_epochs,
        color="#6B7280",
        linestyle="--",
        linewidth=1.3,
        label=f"Configured cap = {max_epochs}",
    )
    axes[2].set_yticks(candidate_positions, [_DISPLAY[name] for name in _CANDIDATES])
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, max_epochs * 1.08)
    axes[2].set_xlabel("Grouped-CV median best epoch")
    axes[2].set_title("c  Epoch selection across folds", loc="left", weight="bold")
    for fold_index, marker in enumerate(fold_markers, start=1):
        axes[2].scatter(
            [],
            [],
            marker=marker,
            s=55,
            color="#6B7280",
            label=f"Fold {fold_index}",
        )
    axes[2].legend(
        frameon=False,
        fontsize=8,
        loc="center right",
        bbox_to_anchor=(1.0, 0.50),
        ncol=2,
    )
    axes[2].grid(axis="x", alpha=0.18)
    axes[2].spines[["top", "right", "left"]].set_visible(False)
    axes[2].tick_params(axis="y", length=0)

    fig.suptitle("DARC v0.2 training diagnostics", fontsize=16, weight="bold", y=1.02)
    fig.text(
        0.5,
        -0.025,
        "Panels a–b are in-sample final-refit optimization traces; panel c is the "
        "grouped-CV epoch-selection evidence.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    png = figure_root / "darc_v02_training_diagnostics.png"
    pdf = figure_root / "darc_v02_training_diagnostics.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    selected_cv = cv_epochs["focal"]
    selected_model = models["focal"]
    final_epochs = int(selected_model.params["epochs"])
    summary = {
        "selected_candidate": "focal",
        "configured_max_epochs": int(max_epochs),
        "patience": int(selected_model.params.get("patience", 0)),
        "final_refit_epochs": final_epochs,
        "grouped_cv_median_best_epochs": selected_cv,
        "near_cap_fold": any(value >= 0.9 * max_epochs for value in selected_cv),
        "assessment": (
            "mixed; optimization is stable, but one grouped-CV fold selected an epoch "
            "near the cap, so full cross-fold convergence is not established"
        ),
        "important_limit": (
            "final-refit monitoring is in-sample and cannot diagnose generalization or "
            "overfitting"
        ),
        "figure_png": str(png),
        "figure_pdf": str(pdf),
    }
    (result_root / "training_diagnostics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/darc_v02_29102026"),
    )
    parser.add_argument("--max-epochs", type=int, default=120)
    args = parser.parse_args()
    summary = write_training_diagnostics(
        args.result_root.resolve(),
        max_epochs=args.max_epochs,
    )
    print(f"Training diagnostics: {summary['figure_png']}")


if __name__ == "__main__":
    main()
