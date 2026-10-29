#!/usr/bin/env python3
"""Freeze CBV-Bench v0.1 into grouped, leakage-safe train/validation/test files.

The split uses only task identity and pre-inference metadata.  It never reads
correctness, verification gain, Teacher cost, or oracle outcomes to assign a split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from report_cbv_bench_all import build_specs, load_collection


SPLITS = ("train", "validation", "test")
HE_QUOTAS = {"train": 98, "validation": 32, "test": 33}
LCB_DIFFICULTY_QUOTAS = {
    "easy": {"train": 26, "validation": 9, "test": 8},
    "medium": {"train": 31, "validation": 10, "test": 11},
    "hard": {"train": 48, "validation": 16, "test": 16},
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _stable_order(task_ids: Iterable[str], *, seed: int, stratum: str) -> list[str]:
    def key(task_id: str) -> tuple[str, str]:
        value = f"{seed}|{stratum}|{task_id}".encode("utf-8")
        return hashlib.sha256(value).hexdigest(), task_id

    return sorted(task_ids, key=key)


def _assign_exact(
    task_ids: Iterable[str],
    quotas: dict[str, int],
    *,
    seed: int,
    stratum: str,
) -> dict[str, str]:
    ordered = _stable_order(task_ids, seed=seed, stratum=stratum)
    if sum(quotas.values()) != len(ordered):
        raise ValueError(
            f"Quotas for {stratum} total {sum(quotas.values())}, expected {len(ordered)}"
        )
    assignments: dict[str, str] = {}
    cursor = 0
    for split in SPLITS:
        next_cursor = cursor + quotas[split]
        assignments.update({task_id: split for task_id in ordered[cursor:next_cursor]})
        cursor = next_cursor
    return assignments


def _source_checksums(collections: list[dict[str, Any]], project_root: Path) -> list[dict[str, Any]]:
    paths = {
        (project_root / shard.results_dir / "trajectories.jsonl").resolve()
        for item in collections
        for shard in item["spec"].shards
    }
    return [
        {
            "path": str(path.relative_to(project_root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(paths)
    ]


def _availability(item: dict[str, Any], task_id: str, level: int) -> bool:
    groups = item["level1_complete"] if level == 1 else item["complete"]
    return (task_id, 0) in groups


def _plot_split(
    output: Path,
    task_rows: list[dict[str, Any]],
    level_rows: list[dict[str, Any]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    datasets = ("humaneval", "livecodebench", "bigcodebench")
    labels = ("HumanEval", "LiveCodeBench", "BigCodeBench")
    colors = {"train": "#2563EB", "validation": "#F59E0B", "test": "#16A34A"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(datasets))
    width = 0.24
    for offset, split in zip((-width, 0, width), SPLITS):
        values = [
            sum(row["dataset"] == dataset and row["split"] == split for row in task_rows)
            for dataset in datasets
        ]
        bars = axes[0].bar(x + offset, values, width, label=split.title(), color=colors[split])
        axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set(
        xticks=x, xticklabels=labels, ylabel="Unique task IDs",
        title="Frozen task split by benchmark",
    )
    axes[0].legend()

    lcb_rows = [row for row in task_rows if row["dataset"] == "livecodebench"]
    difficulties = ("easy", "medium", "hard")
    x_diff = np.arange(len(difficulties))
    for offset, split in zip((-width, 0, width), SPLITS):
        values = [
            sum(row["difficulty"] == difficulty and row["split"] == split for row in lcb_rows)
            for difficulty in difficulties
        ]
        bars = axes[1].bar(
            x_diff + offset, values, width, label=split.title(), color=colors[split]
        )
        axes[1].bar_label(bars, padding=3, fontsize=9)
    axes[1].set(
        xticks=x_diff, xticklabels=[value.title() for value in difficulties],
        ylabel="LiveCodeBench tasks", title="Outcome-blind difficulty stratification",
    )
    axes[1].legend()

    x_level = np.arange(len(SPLITS))
    level1 = [next(row["level1_complete"] for row in level_rows if row["split"] == split) for split in SPLITS]
    level2 = [next(row["level2_complete"] for row in level_rows if row["split"] == split) for split in SPLITS]
    bars1 = axes[2].bar(x_level - 0.18, level1, 0.36, label="Level 1", color="#2563EB")
    bars2 = axes[2].bar(x_level + 0.18, level2, 0.36, label="Level 2", color="#8B5CF6")
    axes[2].bar_label(bars1, padding=3)
    axes[2].bar_label(bars2, padding=3)
    axes[2].set(
        xticks=x_level, xticklabels=[value.title() for value in SPLITS],
        ylabel="Complete task groups", title="Usable counterfactual groups",
    )
    axes[2].legend()
    fig.suptitle("CBV-Bench v0.1 frozen split", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "frozen_split_distribution.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "frozen_split_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def freeze(args: argparse.Namespace) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    output = (project_root / args.output_dir).resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args.base_date_suffix, args.scale_date_suffix)
    collections = [load_collection(spec) for spec in specs]
    by_key = {item["spec"].key: item for item in collections}
    he = by_key["humaneval_l2"]
    lcb = by_key["livecodebench_l2"]
    bcb_l1 = by_key["bigcodebench_l1"]
    bcb_l2 = by_key["bigcodebench_l2"]

    assignments = _assign_exact(
        he["selected_ids"], HE_QUOTAS, seed=args.seed, stratum="humaneval"
    )
    for difficulty, quotas in LCB_DIFFICULTY_QUOTAS.items():
        task_ids = [
            task_id for task_id, record in lcb["task_records"].items()
            if str(record.get("difficulty")) == difficulty
        ]
        assignments.update(
            _assign_exact(
                task_ids, quotas, seed=args.seed,
                stratum=f"livecodebench:{difficulty}",
            )
        )
    for task_id, record in bcb_l1["task_records"].items():
        split = str(record.get("split"))
        if split not in SPLITS:
            raise ValueError(f"BigCodeBench task {task_id} has invalid frozen split {split!r}")
        assignments[task_id] = split

    all_unique = set(he["selected_ids"]) | set(lcb["selected_ids"]) | set(bcb_l1["selected_ids"])
    if set(assignments) != all_unique:
        missing = sorted(all_unique - set(assignments))
        extra = sorted(set(assignments) - all_unique)
        raise AssertionError(f"Assignment universe mismatch: missing={missing[:5]}, extra={extra[:5]}")

    for task_id, record in bcb_l2["task_records"].items():
        if assignments[task_id] != record["split"]:
            raise AssertionError(
                f"BigCodeBench L1/L2 split mismatch for {task_id}: "
                f"{assignments[task_id]} != {record['split']}"
            )

    task_rows: list[dict[str, Any]] = []
    full_task_rows: list[dict[str, Any]] = []
    for dataset, item in (
        ("humaneval", he),
        ("livecodebench", lcb),
        ("bigcodebench", bcb_l1),
    ):
        for task_id in sorted(item["selected_ids"]):
            record = item["task_records"][task_id]
            if dataset == "bigcodebench":
                level2_selected = task_id in bcb_l2["selected_ids"]
                level2_complete = level2_selected and _availability(bcb_l2, task_id, 2)
            else:
                level2_selected = True
                level2_complete = _availability(item, task_id, 2)
            summary = {
                "task_id": task_id,
                "dataset": dataset,
                "dataset_version": record.get("dataset_version", ""),
                "split": assignments[task_id],
                "difficulty": str(record.get("difficulty", "unknown")),
                "source_split": str(record.get("split", "")),
                "level1_complete": _availability(item, task_id, 1),
                "level2_selected": level2_selected,
                "level2_complete": level2_complete,
                "public_task_sha256": record.get("public_task_sha256", ""),
            }
            task_rows.append(summary)
            full_task_rows.append(
                {
                    "schema_version": "cbv-bench-frozen-task-v1",
                    **summary,
                    "source_metadata": record.get("source_metadata", {}),
                    "public_task": record.get("public_task", {}),
                }
            )

    trajectory_rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in collections:
        collection_key = item["spec"].key
        for row in item["trajectory_rows"]:
            task_id = str(row["task_id"])
            split = assignments[task_id]
            trajectory_rows_by_split[split].append(
                {**row, "frozen_split": split, "frozen_collection": collection_key}
            )

    source_checksums = _source_checksums(collections, project_root)
    assignment_payload = "\n".join(
        f"{row['dataset']}\t{row['task_id']}\t{row['split']}"
        for row in sorted(task_rows, key=lambda value: (value["dataset"], value["task_id"]))
    )
    fingerprint_payload = {
        "schema_version": "cbv-bench-frozen-v1",
        "split_seed": args.seed,
        "assignments_sha256": hashlib.sha256(assignment_payload.encode()).hexdigest(),
        "source_checksums": source_checksums,
    }
    freeze_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with tempfile.TemporaryDirectory(prefix=".cbv_freeze_", dir=output.parent) as temp_name:
        temp = Path(temp_name)
        split_rows: list[dict[str, Any]] = []
        for split in SPLITS:
            split_tasks = sorted(
                (row for row in full_task_rows if row["split"] == split),
                key=lambda row: (row["dataset"], row["task_id"]),
            )
            trajectories = sorted(
                trajectory_rows_by_split[split],
                key=lambda row: (
                    str(row["dataset"]), str(row["task_id"]),
                    str(row["frozen_collection"]), str(row["action_sequence"]),
                ),
            )
            tasks_count = _write_jsonl(temp / split / "tasks.jsonl", split_tasks)
            trajectories_count = _write_jsonl(
                temp / split / "trajectories.jsonl", trajectories
            )
            (temp / split / "task_ids.txt").write_text(
                "\n".join(row["task_id"] for row in split_tasks) + "\n",
                encoding="utf-8",
            )
            split_rows.append(
                {
                    "split": split,
                    "unique_tasks": tasks_count,
                    "trajectories": trajectories_count,
                    "level1_complete": sum(row["level1_complete"] for row in split_tasks),
                    "level2_complete": sum(row["level2_complete"] for row in split_tasks),
                }
            )

        compact_rows = sorted(task_rows, key=lambda row: (row["dataset"], row["task_id"]))
        _write_csv(temp / "task_split_manifest.csv", compact_rows)
        _write_json(temp / "source_checksums.json", source_checksums)
        _write_csv(temp / "split_summary.csv", split_rows)

        difficulty_rows: list[dict[str, Any]] = []
        for difficulty in ("easy", "medium", "hard"):
            for split in SPLITS:
                difficulty_rows.append(
                    {
                        "dataset": "livecodebench",
                        "difficulty": difficulty,
                        "split": split,
                        "tasks": sum(
                            row["dataset"] == "livecodebench"
                            and row["difficulty"] == difficulty
                            and row["split"] == split
                            for row in task_rows
                        ),
                    }
                )
        _write_csv(temp / "livecodebench_difficulty_split.csv", difficulty_rows)

        manifest = {
            "schema_version": "cbv-bench-frozen-v1",
            "benchmark_version": "CBV-Bench-v0.1",
            "freeze_fingerprint": freeze_fingerprint,
            "split_seed": args.seed,
            "split_policy": {
                "unit": "task_id",
                "outcome_blind": True,
                "humaneval": {"method": "SHA256 deterministic exact 60/20/20", **HE_QUOTAS},
                "livecodebench": {
                    "method": "SHA256 deterministic, difficulty-stratified exact 60/20/20",
                    "difficulty_quotas": LCB_DIFFICULTY_QUOTAS,
                },
                "bigcodebench": "preserve existing train/validation/test assignments",
                "level_rule": "all levels and action sequences for a task share one split",
            },
            "selected_unique_tasks": len(task_rows),
            "split_summary": split_rows,
            "source_checksums": source_checksums,
        }
        _write_json(temp / "manifest.json", manifest)

        split_sets = {
            split: {row["task_id"] for row in task_rows if row["split"] == split}
            for split in SPLITS
        }
        overlaps = {
            f"{left}_vs_{right}": sorted(split_sets[left] & split_sets[right])
            for index, left in enumerate(SPLITS)
            for right in SPLITS[index + 1 :]
        }
        lcb_counts = Counter(
            (row["difficulty"], row["split"])
            for row in task_rows if row["dataset"] == "livecodebench"
        )
        expected_lcb = {
            (difficulty, split): count
            for difficulty, quotas in LCB_DIFFICULTY_QUOTAS.items()
            for split, count in quotas.items()
        }
        audit = {
            "schema_version": "cbv-bench-split-audit-v1",
            "passed": (
                all(not value for value in overlaps.values())
                and len(set().union(*split_sets.values())) == len(task_rows)
                and all(lcb_counts[key] == value for key, value in expected_lcb.items())
                and all(assignments[task_id] == record["split"] for task_id, record in bcb_l2["task_records"].items())
            ),
            "checks": {
                "split_disjointness": overlaps,
                "split_union_tasks": len(set().union(*split_sets.values())),
                "manifest_tasks": len(task_rows),
                "lcb_difficulty_exact": all(
                    lcb_counts[key] == value for key, value in expected_lcb.items()
                ),
                "bigcodebench_l1_l2_consistent": True,
                "outcomes_used_for_assignment": False,
            },
            "incomplete_groups_retained": [
                {
                    "task_id": row["task_id"],
                    "split": row["split"],
                    "level1_complete": row["level1_complete"],
                    "level2_complete": row["level2_complete"],
                }
                for row in task_rows
                if not row["level1_complete"] or (row["level2_selected"] and not row["level2_complete"])
            ],
        }
        if not audit["passed"]:
            raise AssertionError(f"Frozen split audit failed: {audit}")
        _write_json(temp / "leakage_audit.json", audit)
        _plot_split(temp, task_rows, split_rows)

        readme = f"""# CBV-Bench v0.1 — Frozen split

Freeze fingerprint: `{freeze_fingerprint}`

This folder is a materialized, task-grouped split of the collected CBV-Bench data.
No LLM inference or outcome-dependent assignment was used to create it.

## Split

| Split | Unique tasks | Trajectories | L1-complete groups | L2-complete groups |
|---|---:|---:|---:|---:|
"""
        for row in split_rows:
            readme += (
                f"| {row['split']} | {row['unique_tasks']} | {row['trajectories']} | "
                f"{row['level1_complete']} | {row['level2_complete']} |\n"
            )
        readme += """

## Files

- `manifest.json`: immutable split policy, counts, fingerprint and source checksums.
- `task_split_manifest.csv`: one row per unique task ID.
- `{train,validation,test}/tasks.jsonl`: public task records and availability flags.
- `{train,validation,test}/trajectories.jsonl`: fully materialized trajectory records.
- `leakage_audit.json`: disjointness, L1/L2 consistency and missing-group audit.
- `source_checksums.json`: SHA256 of raw trajectory sources.
- `artifact_sha256sums.txt`: SHA256 of every frozen artifact except itself.

## Scientific status

The raw collection was characterized before this split was frozen. Therefore previous
results are exploratory. From this point onward, model and hyperparameter development
must use train/validation only; test is reserved for one-shot confirmatory evaluation.
Previously trained models that saw HumanEval/LiveCodeBench outcomes must not be reused
as final frozen-test models.
"""
        (temp / "README.md").write_text(readme, encoding="utf-8")

        checksum_paths = sorted(
            path for path in temp.rglob("*")
            if path.is_file() and path.name != "artifact_sha256sums.txt"
        )
        (temp / "artifact_sha256sums.txt").write_text(
            "".join(
                f"{_sha256_file(path)}  {path.relative_to(temp)}\n"
                for path in checksum_paths
            ),
            encoding="utf-8",
        )
        os.replace(temp, output)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/cbv_bench_v01_frozen_28102026"),
    )
    parser.add_argument("--base-date-suffix", default="26102026")
    parser.add_argument("--scale-date-suffix", default="28102026")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = freeze(args)
    print(output)


if __name__ == "__main__":
    main()
