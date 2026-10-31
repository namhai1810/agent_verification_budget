#!/usr/bin/env python3
"""Rebuild a compact TensorBoard dashboard from completed DARC v0.5 CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from agent_verify.analysis.darc_v05_experiment import TensorBoardRunRouter


def export(results: Path, destination: Path) -> None:
    tables = results / "tables"
    history = pd.read_csv(tables / "training_history.csv")
    policies = pd.read_csv(tables / "seed_results.csv")
    router = TensorBoardRunRouter(destination, SummaryWriter)

    training_groups = ("variant", "seed", "iteration", "fold", "member")
    for keys, rows in history.groupby(list(training_groups), sort=True):
        variant, seed, iteration, fold, member = keys
        writer = router.for_run(
            f"{variant}/seed_{seed}/iteration_{iteration}/fold_{fold}_member_{member}"
        )
        for row in rows.sort_values("epoch").itertuples(index=False):
            step = int(row.epoch)
            writer.add_scalar("loss/train_total", float(row.total_loss), step)
            writer.add_scalar("loss/validation_total", float(row.validation_total_loss), step)
            writer.add_scalar(
                "loss/validation_q_success", float(row.validation_q_success_loss), step
            )
            writer.add_scalar(
                "loss/validation_q_cost", float(row.validation_q_cost_loss), step
            )
            writer.add_scalar(
                "diagnostics/mean_absolute_advantage",
                float(row.mean_absolute_predicted_advantage),
                step,
            )
            writer.add_scalar("diagnostics/gradient_norm", float(row.gradient_norm), step)

    baseline = {
        (int(row.seed), float(row.budget_fraction)): float(row.accuracy)
        for row in policies[policies.policy == "Fixed SSS"].itertuples(index=False)
    }
    for keys, rows in policies.groupby(["policy", "seed"], sort=True):
        policy, seed = keys
        writer = router.for_run(f"{policy}/seed_{seed}/policy_evaluation")
        for row in rows.sort_values("budget_fraction").itertuples(index=False):
            step = int(round(100 * float(row.budget_fraction)))
            writer.add_scalar("policy/accuracy", float(row.accuracy), step)
            writer.add_scalar("policy/actual_teacher_tokens", int(row.teacher_tokens), step)
            writer.add_scalar("policy/teacher_calls", int(row.teacher_calls), step)
            writer.add_scalar(
                "policy/budget_utilization",
                int(row.teacher_tokens) / max(int(row.token_budget), 1),
                step,
            )
            writer.add_scalar("policy/helpful", int(row.helpful), step)
            writer.add_scalar("policy/harmful", int(row.harmful), step)
            writer.add_scalar("policy/no_change", int(row.no_change), step)
            writer.add_scalar(
                "policy/improvement_over_never",
                float(row.accuracy) - baseline[(int(seed), float(row.budget_fraction))],
                step,
            )
    router.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/darc_v05_offline_rl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/darc_v05_offline_rl/tensorboard/concise"),
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.rglob("events.out.tfevents.*")):
        raise SystemExit(
            f"Refusing to mix duplicate events into non-empty output: {args.output}"
        )
    export(args.results, args.output)
    print(f"Compact TensorBoard logs: {args.output.resolve()}")


if __name__ == "__main__":
    main()

