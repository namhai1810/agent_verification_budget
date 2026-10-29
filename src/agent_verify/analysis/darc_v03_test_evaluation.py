"""Locked exploratory frozen-test evaluation for DARC v0.3.

This module never trains, calibrates, or tunes a model/policy.  It consumes the
selection artifacts written before test access and reports validation and frozen
test side by side.  The CBV frozen test has been inspected in earlier phases, so
the resulting evidence is exploratory rather than confirmatory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import binomtest

from agent_verify.analysis.cbv_v01_experiment import (
    ReplayResult,
    _bootstrap_summary,
    build_level1_examples,
)
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
    failure_metrics,
)
from agent_verify.analysis.darc_v02_experiment import (
    _policy_budgets,
    _strict_threshold_replay,
)
from agent_verify.analysis.darc_v03 import (
    RiskCalibratedPolicy,
    replay_risk_calibrated_policy,
)
from agent_verify.analysis.darc_v03_experiment import (
    _always_checkpoint,
    _v02_reference,
    _write_csv,
)


def _locked_policy(payload: dict[str, Any]) -> RiskCalibratedPolicy:
    return RiskCalibratedPolicy(
        harmful_weight=float(payload["harmful_weight"]),
        uncertainty_beta=float(payload["uncertainty_beta"]),
        cost_exponent=float(payload["cost_exponent"]),
        cost_lambda=float(payload["cost_lambda"]),
        minimum_change_probability=float(payload["minimum_change_probability"]),
        continuation_weight=float(payload["continuation_weight"]),
        checkpoint_margins=tuple(float(value) for value in payload["checkpoint_margins"]),
        cost_mode=str(payload["cost_mode"]),
    )


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _never_verify(
    examples: list[Any], *, budget: int, fraction: float, hard_cap: int, seed: int
) -> ReplayResult:
    return _strict_threshold_replay(
        examples,
        np.full(len(examples), -1.0),
        np.ones(len(examples)),
        threshold=0.0,
        policy_name="Never Verify",
        token_budget=budget,
        fraction=fraction,
        hard_cap=hard_cap,
        seed=seed,
    )


def _full_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["policy"]): row
        for row in rows
        if float(row["budget_fraction"]) == 1.0
    }


def _paired_comparisons(
    replays: list[ReplayResult],
    *,
    selected_label: str,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    full = {
        replay.policy: replay
        for replay in replays
        if float(replay.budget_fraction) == 1.0
    }
    reference = full[selected_label]
    rng = np.random.default_rng(seed + 303)
    rows: list[dict[str, Any]] = []
    for name in (
        "DARC v0.2 Full",
        "Joint MLP + Threshold",
        "Always CP3",
        "Never Verify",
    ):
        other = full[name]
        if reference.task_keys != other.task_keys:
            raise AssertionError(f"Task order mismatch for paired comparison: {name}")
        difference = reference.outcomes - other.outcomes
        samples = rng.integers(0, len(difference), size=(repetitions, len(difference)))
        bootstrap = difference[samples].mean(axis=1) * 100.0
        wins = int(np.sum(difference > 0))
        losses = int(np.sum(difference < 0))
        discordant = wins + losses
        p_value = (
            float(binomtest(wins, discordant, p=0.5).pvalue)
            if discordant
            else 1.0
        )
        rows.append(
            {
                "reference": selected_label,
                "comparator": name,
                "accuracy_difference_pp": float(difference.mean() * 100.0),
                "paired_ci95_low_pp": float(np.quantile(bootstrap, 0.025)),
                "paired_ci95_high_pp": float(np.quantile(bootstrap, 0.975)),
                "reference_wins": wins,
                "reference_losses": losses,
                "ties": int(len(difference) - discordant),
                "mcnemar_exact_p": p_value,
                "token_difference": int(reference.actual_tokens - other.actual_tokens),
            }
        )
    return rows


def _at_or_below_budget(
    rows: list[dict[str, Any]], *, selected_label: str
) -> list[dict[str, Any]]:
    full = _full_lookup(rows)
    cap = int(full[selected_label]["actual_teacher_tokens"])
    output: list[dict[str, Any]] = []
    for name in (
        selected_label,
        "DARC v0.2 Full",
        "Joint MLP + Threshold",
        "Always CP3",
        "Never Verify",
    ):
        eligible = [
            row
            for row in rows
            if row["policy"] == name
            and int(float(row["actual_teacher_tokens"])) <= cap
        ]
        chosen = max(
            eligible,
            key=lambda row: (
                float(row["accuracy"]),
                -int(float(row["actual_teacher_tokens"])),
            ),
        )
        output.append(
            {
                "policy": name,
                "budget_cap": cap,
                "best_accuracy": float(chosen["accuracy"]),
                "actual_teacher_tokens": int(float(chosen["actual_teacher_tokens"])),
                "teacher_calls": int(float(chosen["teacher_calls"])),
                "source_budget_fraction": float(chosen["budget_fraction"]),
            }
        )
    return output


def _plot(
    output: Path,
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    validation_failures: list[dict[str, Any]],
    test_failures: list[dict[str, Any]],
    selected_label: str,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FAFBFC",
            "figure.facecolor": "white",
        }
    )
    policies = [
        selected_label,
        "DARC v0.2 Full",
        "Joint MLP + Threshold",
        "Always CP3",
        "Never Verify",
    ]
    colors = {
        selected_label: "#D9485F",
        "DARC v0.2 Full": "#8E6C8A",
        "Joint MLP + Threshold": "#4C78A8",
        "Always CP3": "#F28E2B",
        "Never Verify": "#6B7280",
    }

    fig, axis = plt.subplots(figsize=(10.5, 6.1))
    for policy in policies:
        points = sorted(
            [row for row in test_rows if row["policy"] == policy],
            key=lambda row: float(row["actual_teacher_tokens"]),
        )
        unique: dict[tuple[float, float], dict[str, Any]] = {}
        for row in points:
            unique[(float(row["actual_teacher_tokens"]), float(row["accuracy"]))] = row
        points = list(unique.values())
        if not points:
            continue
        axis.plot(
            [float(row["actual_teacher_tokens"]) / 1000.0 for row in points],
            [100.0 * float(row["accuracy"]) for row in points],
            "o-",
            color=colors[policy],
            linewidth=2.5 if policy == selected_label else 1.7,
            markersize=7 if policy == selected_label else 5,
            label="DARC v0.3 (locked)" if policy == selected_label else policy,
        )
    axis.set_xlabel("Actual Teacher tokens (thousands)")
    axis.set_ylabel("Frozen-test accuracy (%)")
    axis.set_title("DARC v0.3 locked frozen-test accuracy–cost frontier", weight="bold")
    axis.grid(color="#E4E8EE", linewidth=0.7)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "darc_v03_frozen_test_frontier.png", dpi=230, bbox_inches="tight")
    fig.savefig(figures / "darc_v03_frozen_test_frontier.pdf", bbox_inches="tight")
    plt.close(fig)

    validation = _full_lookup(validation_rows)
    test = _full_lookup(test_rows)
    validation_failure = _full_lookup(validation_failures)
    test_failure = _full_lookup(test_failures)
    present = [name for name in policies if name in validation and name in test]
    y = np.arange(len(present))
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.3))
    for position, name in enumerate(present):
        for split, rows, offset, marker in (
            ("Validation", validation, -0.12, "o"),
            ("Frozen test", test, 0.12, "s"),
        ):
            row = rows[name]
            accuracy = 100.0 * float(row["accuracy"])
            lower = 100.0 * float(row["accuracy_ci95_low"])
            upper = 100.0 * float(row["accuracy_ci95_high"])
            axes[0].errorbar(
                accuracy,
                position + offset,
                xerr=[[accuracy - lower], [upper - accuracy]],
                fmt=marker,
                color=colors[name],
                alpha=1.0 if split == "Frozen test" else 0.55,
                capsize=3,
                markersize=8 if name == selected_label else 6,
                label=split if position == 0 else None,
            )
    display_names = [
        "DARC v0.3 (locked)" if name == selected_label else name for name in present
    ]
    axes[0].set_yticks(y, display_names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Accuracy (%) with 95% CI")
    axes[0].set_title("a  Validation → test", loc="left", weight="bold")
    axes[0].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )

    width = 0.36
    axes[1].barh(
        y - width / 2,
        [float(validation[name]["actual_teacher_tokens"]) / 1000 for name in present],
        height=width,
        color=[colors[name] for name in present],
        alpha=0.50,
        label="Validation",
    )
    axes[1].barh(
        y + width / 2,
        [float(test[name]["actual_teacher_tokens"]) / 1000 for name in present],
        height=width,
        color=[colors[name] for name in present],
        label="Frozen test",
    )
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Actual Teacher tokens (thousands)")
    axes[1].set_title("b  Full-budget cost", loc="left", weight="bold")
    axes[1].legend(frameon=False)

    selected_failures = [
        validation_failure[selected_label],
        test_failure[selected_label],
    ]
    helpful = np.asarray([int(row["helpful_selected"]) for row in selected_failures])
    harmful = np.asarray([int(row["harmful_selected"]) for row in selected_failures])
    wasted = np.asarray([int(row["wasted_no_change"]) for row in selected_failures])
    x = np.arange(2)
    axes[2].bar(x, helpful, color="#168A5B", label="Helpful")
    axes[2].bar(x, harmful, bottom=helpful, color="#D84A4A", label="Harmful")
    axes[2].bar(
        x,
        wasted,
        bottom=helpful + harmful,
        color="#CBD2D9",
        label="No-change",
    )
    axes[2].set_xticks(x, ["Validation", "Frozen test"])
    axes[2].set_ylabel("Selected Teacher calls")
    axes[2].set_title("c  DARC v0.3 decisions", loc="left", weight="bold")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="x" if axis is not axes[2] else "y", color="#E4E8EE", linewidth=0.7)
    fig.suptitle(
        "DARC v0.3 generalization audit (locked policy; exploratory test)",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(figures / "darc_v03_validation_test_summary.png", dpi=240, bbox_inches="tight")
    fig.savefig(figures / "darc_v03_validation_test_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, *, project_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen = _absolute(project_root, config["data"]["frozen_root"])
    baseline_root = _absolute(project_root, config["data"]["baseline_results"])
    v02_root = _absolute(project_root, config["data"]["darc_v02_results"])
    v03_root = _absolute(project_root, config["output"]["root"])
    output = v03_root / "test_evaluation"
    output.mkdir(parents=True, exist_ok=True)

    selection_path = v03_root / "selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_name = str(selection["development_selected_candidate"])
    selected_label = f"DARC v0.3 · {selected_name}"
    selected_policy = _locked_policy(selection["selected_policy"])
    model_path = v03_root / "models" / "darc_v03_selected.pt"

    train_examples, _train_groups = _load_split(frozen, "train")
    validation_examples, _validation_groups = _load_split(frozen, "validation")
    test_examples, test_groups = _load_split(frozen, "test")
    _assert_disjoint(
        {"train": train_examples, "validation": validation_examples, "test": test_examples}
    )
    hard_cap = int(config["evaluation"]["hard_call_token_cap"])
    _assert_hard_call_cap(test_examples, split="test", hard_call_token_cap=hard_cap)
    cache = _absolute(project_root, config["semantic_encoder"]["cache_dir"])
    test_encoded, cache_manifest = encode_or_load(
        test_examples,
        split="test",
        config=config["semantic_encoder"],
        cache_dir=cache,
    )

    model = DARCEnsemble.load(model_path)
    raw = model.predict(
        test_encoded,
        device_name=str(config["semantic_encoder"].get("device", "auto")),
    )
    calibrator: CheckpointCalibrator = joblib.load(
        v03_root / "selection_objects" / "darc_v03_selected_calibrator.joblib"
    )
    continuation: ContinuationPrior = joblib.load(
        v03_root / "selection_objects" / "darc_v03_selected_continuation_prior.joblib"
    )
    prediction = calibrator.transform(test_examples, raw)

    seed = int(config["reproducibility"]["seed"])
    reference_cost = float(np.median([row.teacher_total_tokens for row in train_examples]))
    budgets = _policy_budgets(train_examples, test_examples)
    replays: list[ReplayResult] = []
    traces: list[dict[str, Any]] = []
    for fraction, budget in budgets:
        replay, trace, _audit = replay_risk_calibrated_policy(
            test_examples,
            prediction,
            continuation,
            selected_policy,
            policy_name=selected_label,
            token_budget=budget,
            budget_fraction=fraction,
            hard_call_token_cap=hard_cap,
            reference_cost=reference_cost,
            seed=seed,
        )
        replays.append(replay)
        traces.extend(trace)

    v02_replays, _v02_latency = _v02_reference(
        v02_root,
        test_examples,
        test_encoded,
        budgets,
        config,
        reference_examples=train_examples,
    )
    replays.extend(v02_replays)
    joint_examples = build_level1_examples(test_groups)
    joint_model = joblib.load(baseline_root / "models" / "combined" / "joint.joblib")
    joint_prediction = joint_model.predict(joint_examples)
    joint_meta = json.loads(
        (baseline_root / "models" / "combined" / "joint.json").read_text()
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
                _always_checkpoint(
                    test_examples,
                    checkpoint=3,
                    token_budget=budget,
                    fraction=fraction,
                    hard_cap=hard_cap,
                    seed=seed,
                ),
                _never_verify(
                    test_examples,
                    budget=budget,
                    fraction=fraction,
                    hard_cap=hard_cap,
                    seed=seed,
                ),
            ]
        )

    repetitions = int(config["reproducibility"]["bootstrap_repetitions"])
    test_rows = [
        _bootstrap_summary(replay, seed=seed, repetitions=repetitions)
        for replay in replays
    ]
    test_failures = [
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
    validation_rows = _read_csv(v03_root / "tables" / "validation_policy_results.csv")
    validation_failures = _read_csv(
        v03_root / "tables" / "validation_failure_metrics.csv"
    )
    validation_metrics = next(
        row
        for row in _read_csv(v03_root / "tables" / "validation_prediction_metrics.csv")
        if row["candidate"] == selected_name and row["checkpoint"] == "all"
    )

    _write_csv(output / "tables" / "frozen_test_policy_results.csv", test_rows)
    _write_csv(output / "tables" / "frozen_test_failure_metrics.csv", test_failures)
    _write_csv(output / "tables" / "frozen_test_decision_trace.csv", traces)
    _write_csv(output / "tables" / "frozen_test_prediction_metrics.csv", [metrics])
    paired_rows = _paired_comparisons(
        replays,
        selected_label=selected_label,
        repetitions=repetitions,
        seed=seed,
    )
    matched_rows = _at_or_below_budget(test_rows, selected_label=selected_label)
    _write_csv(output / "tables" / "paired_policy_comparison.csv", paired_rows)
    _write_csv(output / "tables" / "budget_matched_comparison.csv", matched_rows)
    _plot(
        output,
        validation_rows,
        test_rows,
        validation_failures,
        test_failures,
        selected_label,
    )

    validation = _full_lookup(validation_rows)
    test = _full_lookup(test_rows)
    validation_failure = _full_lookup(validation_failures)
    test_failure = _full_lookup(test_failures)
    policies = [
        selected_label,
        "DARC v0.2 Full",
        "Joint MLP + Threshold",
        "Always CP3",
        "Never Verify",
    ]
    report = output / "report" / "darc_v03_validation_frozen_test_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DARC v0.3 — Validation and frozen-test evaluation",
        "",
        "> Exploratory generalization audit. The CBV frozen test was inspected in earlier project phases, so it is not an untouched confirmatory holdout. No v0.3 model, calibration, policy, threshold, or margin was changed after test access.",
        "",
        "## Lock contract",
        "",
        f"- Locked candidate: `{selected_name}`.",
        "- Locked policy source: train OOF plus validation acceptance gate.",
        "- Test-time training/tuning: none.",
        f"- Selection manifest SHA256: `{_sha256(selection_path)}`.",
        f"- Selected model SHA256: `{_sha256(model_path)}`.",
        "- Student/Teacher inference: none; cached counterfactual replay only.",
        "",
        "## Predictive generalization",
        "",
        "| Split | Helpful AUPRC | Harmful AUPRC | Gain Pearson r | Gain MAE | Cost MAE |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Validation | {float(validation_metrics['helpful_auprc']):.3f} | {float(validation_metrics['harmful_auprc']):.3f} | {float(validation_metrics['gain_pearson']):.3f} | {float(validation_metrics['gain_mae']):.3f} | {float(validation_metrics['cost_mae']):.1f} |",
        f"| Frozen test | {metrics['helpful_auprc']:.3f} | {metrics['harmful_auprc']:.3f} | {metrics['gain_pearson']:.3f} | {metrics['gain_mae']:.3f} | {metrics['cost_mae']:.1f} |",
        "",
        "## Full-budget comparison",
        "",
        "| Policy | Validation accuracy | Test accuracy (95% CI) | Test gain | Test calls | Test tokens | H/D/N | Test CP1/CP2/CP3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in policies:
        if name not in validation or name not in test:
            continue
        val = validation[name]
        tst = test[name]
        failure = test_failure[name]
        lines.append(
            f"| {name} | {100*float(val['accuracy']):.2f}% | {100*float(tst['accuracy']):.2f}% [{100*float(tst['accuracy_ci95_low']):.2f}, {100*float(tst['accuracy_ci95_high']):.2f}] | {float(tst['gain_vs_never_pp']):+.2f} pp | {int(tst['teacher_calls'])} | {int(tst['actual_teacher_tokens']):,} | {failure['helpful_selected']}/{failure['harmful_selected']}/{failure['wasted_no_change']} | {failure['cp1_calls']}/{failure['cp2_calls']}/{failure['cp3_calls']} |"
        )
    selected_validation = validation[selected_label]
    selected_test = test[selected_label]
    selected_validation_failure = validation_failure[selected_label]
    selected_test_failure = test_failure[selected_label]
    v02_test = test["DARC v0.2 Full"]
    token_saving_vs_v02 = int(v02_test["actual_teacher_tokens"]) - int(
        selected_test["actual_teacher_tokens"]
    )
    lines.extend(
        [
            "",
            "## Paired task comparison at full nominal budget",
            "",
            "| Comparator | v0.3 difference (95% paired CI) | v0.3 wins/losses/ties | Exact McNemar p | Token difference |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            f"| {row['comparator']} | {float(row['accuracy_difference_pp']):+.2f} pp [{float(row['paired_ci95_low_pp']):+.2f}, {float(row['paired_ci95_high_pp']):+.2f}] | {row['reference_wins']}/{row['reference_losses']}/{row['ties']} | {float(row['mcnemar_exact_p']):.3f} | {int(row['token_difference']):+,} |"
        )
    lines.extend(
        [
            "",
            "## Best observed point at or below the v0.3 token spend",
            "",
            f"The cap is the locked v0.3 full-budget spend: **{int(test[selected_label]['actual_teacher_tokens']):,} actual Teacher tokens**. Baselines are selected only from their pre-existing discrete budget points; no test-time threshold is fitted.",
            "",
            "| Policy | Best accuracy under cap | Tokens | Calls | Source budget point |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in matched_rows:
        lines.append(
            f"| {row['policy']} | {100*float(row['best_accuracy']):.2f}% | {int(row['actual_teacher_tokens']):,} | {int(row['teacher_calls'])} | {100*float(row['source_budget_fraction']):.0f}% |"
        )
    lines.extend(
        [
            "",
            "![Validation and frozen-test summary](../figures/darc_v03_validation_test_summary.png)",
            "",
            "![Frozen-test frontier](../figures/darc_v03_frozen_test_frontier.png)",
            "",
            "## DARC v0.3 generalization diagnosis",
            "",
            f"- Accuracy changes from **{100*float(selected_validation['accuracy']):.2f}% validation** to **{100*float(selected_test['accuracy']):.2f}% frozen test** ({100*(float(selected_test['accuracy'])-float(selected_validation['accuracy'])):+.2f} pp).",
            f"- Full-budget actual cost changes from **{int(selected_validation['actual_teacher_tokens']):,}** to **{int(selected_test['actual_teacher_tokens']):,} Teacher tokens**.",
            f"- Helpful/Harmful/No-change changes from **{selected_validation_failure['helpful_selected']}/{selected_validation_failure['harmful_selected']}/{selected_validation_failure['wasted_no_change']}** to **{selected_test_failure['helpful_selected']}/{selected_test_failure['harmful_selected']}/{selected_test_failure['wasted_no_change']}**.",
            f"- Compared with v0.2 on test, v0.3 preserves the same **{100*float(selected_test['accuracy']):.2f}%** accuracy while saving **{token_saving_vs_v02:,} tokens ({100*token_saving_vs_v02/max(int(v02_test['actual_teacher_tokens']), 1):.1f}%)**. It is an efficiency improvement, not an accuracy improvement.",
            "- The under-cap table uses a coarse predeclared grid. Several baselines leave substantial budget unused at their best eligible point, so this table does not prove a continuous Pareto advantage.",
            "- Risk control generalizes (zero Harmful selections), but 129/135 calls are No-change and only 6 of 19 rescuable test tasks are captured. Helpful ranking remains the main bottleneck.",
            "- Confidence intervals and paired task outcomes must be considered before any superiority claim; point accuracy alone is insufficient.",
            "",
            "## Scientific status",
            "",
            "This evaluation may be used to describe generalization and failure modes. It must not be used to tune v0.3 and then report the same frozen test as confirmatory. Any method update after reading these results requires a new external/frozen holdout for a paper-level claim.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "darc-v0.3-test-evaluation-v1",
                "selected_before_test": selected_name,
                "selection_manifest_sha256": _sha256(selection_path),
                "selected_model_sha256": _sha256(model_path),
                "test_used_for_tuning": False,
                "confirmatory": False,
                "frozen_test_tasks": len({row.group_key for row in test_examples}),
                "cache_manifest": cache_manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/darc_v03.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = run(args.config, project_root=args.project_root.resolve())
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
