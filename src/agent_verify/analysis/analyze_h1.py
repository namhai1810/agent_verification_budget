from __future__ import annotations

import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from agent_verify.logging import JsonlStore


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    value = float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])
    return None if math.isnan(value) else value


def _fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def analyze_h1(results_dir: Path, run_id: str) -> None:
    pair_rows = [
        row
        for row in JsonlStore(results_dir / "intervention_pairs.jsonl").read_all()
        if row.get("run_id") == run_id and row.get("status") == "completed"
    ]
    baselines = [
        row
        for row in JsonlStore(results_dir / "baseline.jsonl").read_all()
        if row.get("run_id") == run_id and row.get("status") == "completed"
    ]
    manifests = [
        row for row in JsonlStore(results_dir / "manifests.jsonl").read_all() if row.get("run_id") == run_id
    ]
    manifest = manifests[-1] if manifests else {}

    report_dir = results_dir / "reports" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    analysis_csv = report_dir / "h1_analysis.csv"
    _write_analysis_csv(analysis_csv, pair_rows)
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _make_figures(figures_dir, pair_rows)

    n = len(pair_rows)
    helpful = sum(row["delta"] == 1 for row in pair_rows)
    harmful = sum(row["delta"] == -1 for row in pair_rows)
    unchanged = sum(row["delta"] == 0 for row in pair_rows)
    by_checkpoint: dict[int, list[dict]] = defaultdict(list)
    for row in pair_rows:
        by_checkpoint[int(row["checkpoint"])].append(row)

    raw_corr = _correlation(
        [float(row["local_risk_score"]) for row in pair_rows],
        [float(row["delta"]) for row in pair_rows],
    )
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in pair_rows:
        grouped[(row["task_id"], int(row["checkpoint"]))].append(row)
    grouped_risk = [mean(float(x["local_risk_score"]) for x in rows) for rows in grouped.values()]
    grouped_delta = [mean(float(x["delta"]) for x in rows) for rows in grouped.values()]
    mean_corr = _correlation(grouped_risk, grouped_delta)

    task_groups: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for (task_id, checkpoint), rows in grouped.items():
        task_groups[task_id][checkpoint] = (
            mean(float(x["local_risk_score"]) for x in rows),
            mean(float(x["delta"]) for x in rows),
        )
    disagreements = 0
    comparable = 0
    tied = 0
    risk_selected_values: list[float] = []
    oracle_values: list[float] = []
    for checkpoints in task_groups.values():
        if set(checkpoints) != {1, 2, 3}:
            continue
        comparable += 1
        max_risk = max(value[0] for value in checkpoints.values())
        max_delta = max(value[1] for value in checkpoints.values())
        risk_best = {cp for cp, value in checkpoints.items() if value[0] == max_risk}
        oracle_best = {cp for cp, value in checkpoints.items() if value[1] == max_delta}
        if len(risk_best) > 1 or len(oracle_best) > 1:
            tied += 1
        if risk_best.isdisjoint(oracle_best):
            disagreements += 1
        risk_selected_values.append(mean(checkpoints[cp][1] for cp in risk_best))
        oracle_values.append(max_delta)

    mock_only = bool(manifest.get("mock_only", False))
    report = _build_report(
        run_id=run_id,
        mock_only=mock_only,
        n=n,
        baseline_count=len(baselines),
        helpful=helpful,
        harmful=harmful,
        unchanged=unchanged,
        by_checkpoint=by_checkpoint,
        raw_corr=raw_corr,
        mean_corr=mean_corr,
        comparable=comparable,
        disagreements=disagreements,
        tied=tied,
        risk_selected=mean(risk_selected_values) if risk_selected_values else None,
        oracle_selected=mean(oracle_values) if oracle_values else None,
        pair_rows=pair_rows,
    )
    (report_dir / "summary.md").write_text(report, encoding="utf-8")

    # Keep stable latest-run aliases for existing tooling while preserving the
    # canonical report bundle under reports/<run_id>/.
    shutil.copy2(analysis_csv, results_dir / "h1_analysis.csv")
    latest_figures = results_dir / "figures"
    latest_figures.mkdir(parents=True, exist_ok=True)
    for figure in figures_dir.glob("*.png"):
        shutil.copy2(figure, latest_figures / figure.name)
    (results_dir / "summary.md").write_text(report, encoding="utf-8")


def _write_analysis_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "run_id",
        "pair_id",
        "task_id",
        "checkpoint",
        "repetition",
        "local_risk_score",
        "y_skip",
        "y_verify",
        "delta",
        "teacher_verdict",
        "teacher_changed",
        "teacher_cost_usd",
        "teacher_latency_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_figures(figures_dir: Path, rows: list[dict]) -> None:
    checkpoints = [1, 2, 3]
    grouped = {cp: [row for row in rows if int(row["checkpoint"]) == cp] for cp in checkpoints}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(
        [row["local_risk_score"] for row in rows],
        [row["delta"] for row in rows],
        alpha=0.65,
    )
    ax.set(xlabel="Local risk score", ylabel="Observed delta", title="Local risk vs. delta")
    ax.set_yticks([-1, 0, 1])
    fig.tight_layout()
    fig.savefig(figures_dir / "risk_vs_delta.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(
        [str(cp) for cp in checkpoints],
        [mean([row["delta"] for row in grouped[cp]]) if grouped[cp] else 0 for cp in checkpoints],
    )
    ax.set(xlabel="Checkpoint", ylabel="Mean observed delta", title="Verification value")
    fig.tight_layout()
    fig.savefig(figures_dir / "checkpoint_verification_value.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    categories = ["Helpful", "Unchanged", "Harmful"]
    counts = [
        sum(row["delta"] == 1 for row in rows),
        sum(row["delta"] == 0 for row in rows),
        sum(row["delta"] == -1 for row in rows),
    ]
    ax.bar(categories, counts, color=["#2ca02c", "#7f7f7f", "#d62728"])
    ax.set(ylabel="Number of pairs", title="Observed verification outcomes")
    fig.tight_layout()
    fig.savefig(figures_dir / "helpful_harmful_rates.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(
        [str(cp) for cp in checkpoints],
        [mean([row["teacher_cost_usd"] for row in grouped[cp]]) if grouped[cp] else 0 for cp in checkpoints],
    )
    ax.set(xlabel="Checkpoint", ylabel="Mean teacher cost (USD)", title="Teacher cost")
    fig.tight_layout()
    fig.savefig(figures_dir / "cost_by_checkpoint.png", dpi=160)
    plt.close(fig)


def _build_report(**stats) -> str:
    n = stats["n"]

    def pct(count: int) -> float:
        return (100.0 * count / n) if n else 0.0

    lines = [
        "# Phase 1 — H1 Analysis",
        "",
        f"Run ID: `{stats['run_id']}`",
        f"Run type: **{'MOCK VALIDATION (not empirical evidence)' if stats['mock_only'] else 'REAL MODEL EXPERIMENT'}**",
        f"Sample: {stats['baseline_count']} baseline tasks; {n} paired checkpoint interventions.",
        "",
        "## Direct answers",
        "",
    ]
    if stats["mock_only"]:
        lines.extend(
            [
                "- Does verification meaningfully change downstream success? **Not answerable from this mock run.** The mock teacher intentionally preserves artifacts.",
                "- Which checkpoint helps most? **Not answerable from mock data.**",
                "- How well does local risk predict helpful verification? **Not answerable from mock data.**",
                "- Are there risk/value disagreements? Only mechanical mock outcomes are available; they are not scientific evidence.",
                "- Is there enough signal for Phase 2? **No. A credentialed, budget-authorized real-model pilot is required.**",
            ]
        )
    else:
        best_cp = None
        if stats["by_checkpoint"]:
            best_cp = max(
                stats["by_checkpoint"],
                key=lambda cp: mean(row["delta"] for row in stats["by_checkpoint"][cp]),
            )
        lines.extend(
            [
                f"- Verification changed final binary success in {pct(stats['helpful'] + stats['harmful']):.1f}% of pairs ({stats['helpful']} helpful, {stats['harmful']} harmful).",
                f"- Highest observed mean delta was at checkpoint {best_cp if best_cp else 'N/A'}; this is descriptive, not causal generalization.",
                f"- Local-risk/observed-delta correlation: {_fmt(stats['raw_corr'])}.",
                f"- Risk and empirical rankings were disjoint for {stats['disagreements']}/{stats['comparable']} comparable tasks; {stats['tied']} tasks contained ranking ties.",
                "- Phase 2 is justified only if the effect sizes, variability, and case evidence below are sufficiently non-degenerate; this report does not impose a positive conclusion.",
            ]
        )
    lines.extend(
        [
            "",
            "## Verification outcomes",
            "",
            f"- Helpful: {stats['helpful']}/{n} ({pct(stats['helpful']):.1f}%)",
            f"- Harmful: {stats['harmful']}/{n} ({pct(stats['harmful']):.1f}%)",
            f"- Unchanged: {stats['unchanged']}/{n} ({pct(stats['unchanged']):.1f}%)",
            "",
            "| Checkpoint | n | Mean delta | Mean teacher cost | Mean teacher latency (s) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for cp in (1, 2, 3):
        rows = stats["by_checkpoint"].get(cp, [])
        lines.append(
            f"| {cp} | {len(rows)} | {_fmt(mean([r['delta'] for r in rows]) if rows else None)} "
            f"| {_fmt(mean([r['teacher_cost_usd'] for r in rows]) if rows else None, 6)} "
            f"| {_fmt(mean([r['teacher_latency_seconds'] for r in rows]) if rows else None)} |"
        )
    lines.extend(
        [
            "",
            "## Local risk vs. downstream value",
            "",
            f"- Pair-level Pearson correlation: {_fmt(stats['raw_corr'])} (n={n}).",
            f"- Correlation with per-task/checkpoint mean delta: {_fmt(stats['mean_corr'])}.",
            f"- Mean delta when selecting the highest-risk checkpoint: {_fmt(stats['risk_selected'])}.",
            f"- Offline-oracle mean delta at the highest empirical-value checkpoint: {_fmt(stats['oracle_selected'])}.",
            f"- Disjoint risk/value maxima: {stats['disagreements']}/{stats['comparable']}; tasks with ties: {stats['tied']}.",
            "",
            "Correlations are descriptive. Binary deltas are discrete, repeated pairs may not be independent, and undefined correlations are reported as N/A.",
            "",
            "## Observed case studies",
            "",
        ]
    )
    cases = [
        ("Low risk, positive benefit", lambda r: r["local_risk_score"] <= 0.33 and r["delta"] == 1),
        (
            "High risk, negligible benefit",
            lambda r: r["local_risk_score"] >= 0.67 and r["delta"] == 0,
        ),
        ("Harmful teacher verification", lambda r: r["delta"] == -1),
        (
            "Downstream correction without teacher",
            lambda r: r["checkpoint"] < 3 and r["local_risk_score"] >= 0.67 and r["y_skip"] == 1,
        ),
    ]
    for title, predicate in cases:
        matches = [row for row in stats["pair_rows"] if predicate(row)]
        lines.append(f"### {title}")
        lines.append("")
        if not matches:
            lines.append("No observed case matched this definition; none was invented.")
        else:
            for row in matches[:3]:
                lines.append(
                    f"- `{row['task_id']}`, checkpoint {row['checkpoint']}, pair `{row['pair_id']}`: "
                    f"risk={row['local_risk_score']:.3f}, Y_skip={row['y_skip']}, "
                    f"Y_verify={row['y_verify']}, delta={row['delta']}, "
                    f"teacher_changed={row['teacher_changed']}. Artifact and trajectory references are in `intervention_pairs.jsonl`."
                )
        lines.append("")
    lines.extend(
        [
            "## Limitations",
            "",
            "Local-risk scores are proxies with role-specific meanings, not calibrated probabilities. The offline empirical-value ranking uses observed counterfactual outcomes and is not a deployable routing policy. Small samples, ties, mostly-zero outcomes, and provider nondeterminism limit inference.",
            "",
        ]
    )
    return "\n".join(lines)
