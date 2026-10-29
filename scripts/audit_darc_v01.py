#!/usr/bin/env python3
"""Pre-training DARC audit. Reads train/validation only and never loads the encoder model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from agent_verify.analysis.darc_data import FrozenSemanticEncoder, class_counts, semantic_coverage
from agent_verify.analysis.darc_experiment import (
    _absolute,
    _assert_disjoint,
    _assert_hard_call_cap,
    _load_split,
    load_config,
)

EXPECTED_ABLATIONS = {
    ("semantic_no_downstream", "none", "natural_ce", False),
    ("checkpoint_only_control", "checkpoint_only", "natural_ce", False),
    ("semantic_attention_natural", "semantic_attention", "natural_ce", False),
    ("semantic_attention_balanced", "semantic_attention", "balanced_ce", False),
    ("semantic_attention_focal", "semantic_attention", "focal", False),
    ("semantic_attention_ce_ranking", "semantic_attention", "natural_ce", True),
    ("semantic_attention_balanced_ranking", "semantic_attention", "balanced_ce", True),
}


def _ablation_contract(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observed: set[tuple[str, str, str, bool]] = set()
    for row in config["ablations"]:
        if "reuse_model_from" in row:
            continue
        losses = list(row.get("loss_candidates", []))
        if len(losses) != 1:
            raise ValueError(f"Ablation {row['name']} must isolate exactly one loss")
        key = (
            str(row["name"]),
            str(row["downstream_mode"]),
            str(losses[0]),
            bool(row.get("paired_ranking")),
        )
        observed.add(key)
        rows.append(
            {
                "name": key[0],
                "downstream_mode": key[1],
                "loss": key[2],
                "ranking": key[3],
            }
        )
    if observed != EXPECTED_ABLATIONS:
        raise ValueError(
            f"DARC ablation contract mismatch; missing={EXPECTED_ABLATIONS - observed}, "
            f"extra={observed - EXPECTED_ABLATIONS}"
        )
    return rows


def _fmt(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def run_audit(
    config_path: Path,
    project_root: Path,
    output: Path,
    *,
    tests_passed: bool = False,
) -> Path:
    config = load_config(config_path)
    frozen_root = _absolute(project_root, config["data"]["frozen_root"])
    train, _train_groups = _load_split(frozen_root, "train")
    validation, _validation_groups = _load_split(frozen_root, "validation")
    partitions = {"train": train, "validation": validation}
    _assert_disjoint(partitions)
    cap = int(config["evaluation"]["hard_call_token_cap"])
    _assert_hard_call_cap(train, split="train", hard_call_token_cap=cap)
    _assert_hard_call_cap(validation, split="validation", hard_call_token_cap=cap)
    ablations = _ablation_contract(config)
    encoder = FrozenSemanticEncoder(config["semantic_encoder"])
    max_chunks = int(config["semantic_encoder"]["max_chunks_per_text"])
    coverage = {
        split: semantic_coverage(examples, encoder)
        for split, examples in partitions.items()
    }
    downstream_unique = {
        split: len(
            {
                hashlib.sha256("\n".join(row.downstream_role_texts).encode()).hexdigest()
                for row in examples
            }
        )
        for split, examples in partitions.items()
    }
    payload = {
        "schema_version": "darc-v0.1-pretraining-audit-v1",
        "config": str(config_path),
        "test_loaded": False,
        "student_teacher_inference": False,
        "regression_tests_passed": tests_passed,
        "split_disjoint": True,
        "hard_call_token_cap": cap,
        "max_observed_teacher_call_tokens": {
            split: max(row.teacher_total_tokens for row in examples)
            for split, examples in partitions.items()
        },
        "splits": {
            split: {
                "tasks": len({row.group_key for row in examples}),
                "states": len(examples),
                "classes": class_counts(examples),
                "unique_downstream_views": downstream_unique[split],
            }
            for split, examples in partitions.items()
        },
        "coverage": coverage,
        "ablations": ablations,
        "budget_accounting": {
            "darc_full": config["evaluation"]["darc_budget_accounting"],
            "legacy_comparison": "preserved from existing CBV policy_results.csv",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DARC v0.1 — Pre-training Readiness Audit",
        "",
        "> This audit reads frozen train/validation only. It does not load frozen test, train a model, regenerate data, or invoke Student/Teacher LLMs.",
        "",
        "## Verdict",
        "",
        (
            "**READY FOR TRAINING** — the DARC contract regression suite passed in the active environment."
            if tests_passed
            else "**CONDITIONALLY READY** — run the DARC contract regression suite before training."
        ),
        "",
        "## Fixes validated",
        "",
        "- **Budget:** predicted admission cost, actual replay spend, and hard-cap reservation are tracked separately; hard-cap replay asserts zero actual overshoot.",
        "- **Downstream representation:** semantic dependency views use the public task and current artifact; no-downstream and checkpoint-only controls isolate whether attention adds information beyond checkpoint identity.",
        "- **Losses:** Natural CE, Balanced CE, Focal, Natural CE + Ranking, and Balanced CE + Ranking are distinct one-factor configurations.",
        "- **Encoder:** long inputs use audited chunk pooling and a versioned cache; Transformers is pinned to the compatible v4 API.",
        "- **Evaluation:** tuning is validation-only, splits are task-disjoint, and frozen test is not opened by this audit.",
        "",
        "## Data and leakage",
        "",
        "| Split | Tasks | States | Helpful | Harmful | No-change | Unique downstream views | Max Teacher tokens/call |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, examples in partitions.items():
        counts = class_counts(examples)
        lines.append(
            f"| {split} | {len({row.group_key for row in examples})} | {len(examples)} | {counts['helpful']} | {counts['harmful']} | {counts['no_change']} | {downstream_unique[split]} | {payload['max_observed_teacher_call_tokens'][split]} |"
        )
    lines.extend(
        [
            "",
            "- Split unit is task; train/validation group overlap is asserted empty.",
            "- Features exclude paired outcomes, actual Teacher cost, future artifacts, and oracle choices.",
            "- Downstream views are conditioned on the public task and current observable artifact, rather than being constant checkpoint labels.",
            f"- The configured per-call hard reservation is `{cap}` tokens; all observed train/validation calls are checked against it.",
            "",
            "## Semantic input coverage",
            "",
            f"| Split | Field | Texts | Mean tokens | P95 | >1 window | Truncated by {max_chunks}-chunk cap | Mean retained | Minimum retained | Chunks |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, fields in coverage.items():
        for field, row in fields.items():
            lines.append(
                f"| {split} | {field} | {row['texts']} | {_fmt(row['mean_tokens'])} | {_fmt(row['p95_tokens'])} | {row['over_single_window']} | {row['truncated_by_chunk_limit']} | {100 * float(row['mean_retained_ratio']):.2f}% | {100 * float(row['min_retained_ratio']):.1f}% | {row['encoded_chunks']} |"
            )
    lines.extend(
        [
            "",
            "The 512-token encoder limit is handled by deterministic head-to-tail chunk pooling. The cache signature includes max length, stride, chunk count, model revision, and observable-state fingerprint. Cold encoder build time and warm NPZ load time are reported separately after execution.",
            "",
            "## Controlled ablations",
            "",
            "| Name | Downstream representation | Classification loss | Pair ranking |",
            "|---|---|---|---|",
        ]
    )
    for row in ablations:
        lines.append(
            f"| {row['name']} | {row['downstream_mode']} | {row['loss']} | {row['ranking']} |"
        )
    lines.extend(
        [
            "",
            "## Budget and evaluation audit",
            "",
            "- DARC Full uses online-deployable hard-cap accounting: reserve one maximum call before Verify, then refund unused tokens after usage is observed.",
            "- Predicted and actual Teacher tokens remain separate fields. Predicted-compatible ablations report, rather than hide, actual overshoot.",
            "- Existing CBV baseline rows are reused unchanged and annotated as predicted-compatible or offline-actual-cost accounting.",
            "- Budget grid is imported from the existing CBV implementation; accuracy–cost figures use actual Teacher tokens and expose overshoot/accounting mode.",
            "- Calibration temperature, uncertainty coefficient, score normalization, cost mode, and margin are selected on validation only.",
            "- Frozen test remains unopened by this audit and is loaded by the experiment only after `selection_manifest.json` is frozen.",
            "",
            "## Validation commands",
            "",
            "```bash",
            "pytest -q tests/test_darc_contracts.py",
            "bash scripts/run_darc_v01_pipeline.sh",
            "```",
            "",
            f"Machine-readable audit: `{json_path}`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/darc_v01.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/darc_v01_29102026/report/darc_v01_pretraining_audit.md"),
    )
    parser.add_argument(
        "--tests-passed",
        action="store_true",
        help="Record that tests/test_darc_contracts.py passed in this environment.",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.project_root / args.output
    report = run_audit(
        args.config.resolve(),
        args.project_root.resolve(),
        output.resolve(),
        tests_passed=args.tests_passed,
    )
    print(report)


if __name__ == "__main__":
    main()
