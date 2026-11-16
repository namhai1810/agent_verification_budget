#!/usr/bin/env python3
"""Summarize frozen CBV v0.1 metadata plus the train-only L2 extension.

Frozen validation/test trajectory files are intentionally never opened.  Outcome
statistics are computed on training trajectories only.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SEQUENCES = ("SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def effect_label(y_skip: bool, y_verify: bool) -> str:
    if not y_skip and y_verify:
        return "Helpful"
    if y_skip and not y_verify:
        return "Harmful"
    return "NoChange"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project = Path.cwd()
    frozen = project / "data/cbv_bench_v01_frozen_28102026"
    extension = project / "results/cbv_train_l2_extension_12112026"
    output = extension / "combined_summary"
    output.mkdir(parents=True, exist_ok=True)

    task_rows = list(csv.DictReader((frozen / "task_split_manifest.csv").open(encoding="utf-8")))
    selected = json.loads(
        (extension / "selection/bigcodebench_train_l2_extension.json").read_text(encoding="utf-8")
    )
    new_ids = {row["task_id"] for row in selected["task_records"]}
    if len(new_ids) != 150:
        raise RuntimeError(f"Expected 150 extension tasks, found {len(new_ids)}")
    by_id = {row["task_id"]: row for row in task_rows}
    if any(by_id[task_id]["split"] != "train" for task_id in new_ids):
        raise RuntimeError("Extension contains a non-train task")
    if any(by_id[task_id]["level2_complete"] == "True" for task_id in new_ids):
        raise RuntimeError("Extension overlaps old Level-2 tasks")

    inventory_rows = []
    for dataset in ("humaneval", "livecodebench", "bigcodebench"):
        for split in ("train", "validation", "test"):
            scope = [row for row in task_rows if row["dataset"] == dataset and row["split"] == split]
            inventory_rows.append({
                "dataset": dataset,
                "split": split,
                "unique_tasks": len(scope),
                "level1_complete": sum(row["level1_complete"] == "True" for row in scope),
                "old_level2_complete": sum(row["level2_complete"] == "True" for row in scope),
                "new_level2_extension": sum(row["task_id"] in new_ids for row in scope),
                "updated_level2_complete": sum(
                    row["level2_complete"] == "True" or row["task_id"] in new_ids for row in scope
                ),
            })
    write_csv(output / "tasks_by_dataset_split.csv", inventory_rows)

    frozen_train = read_jsonl(frozen / "train/trajectories.jsonl")
    old_l2_task_ids = {
        row["task_id"] for row in task_rows
        if row["split"] == "train" and row["level2_complete"] == "True"
    }
    old_l2 = {
        (row["dataset"], row["task_id"], row["action_sequence"]): row
        for row in frozen_train
        if row["task_id"] in old_l2_task_ids
        and str(row.get("frozen_collection", "")).endswith("_l2")
    }
    manifests = read_jsonl(extension / "collection/manifests.jsonl")
    compatibility = manifests[-1]["compatibility_fingerprint"]
    new_rows = [
        row for row in read_jsonl(extension / "collection/trajectories.jsonl")
        if row.get("compatibility_fingerprint") == compatibility
    ]
    new_l2 = {
        (row["dataset"], row["task_id"], row["action_sequence"]): row
        for row in new_rows
    }
    if set(key[1] for key in old_l2).intersection(key[1] for key in new_l2):
        raise RuntimeError("Old/new Level-2 task overlap detected")
    combined = {**old_l2, **new_l2}
    grouped: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for (dataset, task_id, sequence), row in combined.items():
        grouped[(dataset, task_id)][sequence] = bool(row["passed"])
    incomplete = [key for key, values in grouped.items() if set(values) != set(SEQUENCES)]
    if incomplete:
        raise RuntimeError(f"Incomplete combined Level-2 tasks: {incomplete[:5]}")

    sequence_rows = []
    for dataset in ("humaneval", "livecodebench", "bigcodebench", "ALL"):
        scope = [
            (key, values) for key, values in grouped.items()
            if dataset == "ALL" or key[0] == dataset
        ]
        for sequence in SEQUENCES:
            passed = sum(values[sequence] for _key, values in scope)
            sequence_rows.append({
                "dataset": dataset,
                "sequence": sequence,
                "tasks": len(scope),
                "passed": passed,
                "accuracy": passed / len(scope) if scope else 0.0,
            })
    write_csv(output / "train_level2_sequence_outcomes.csv", sequence_rows)

    conditional = Counter()
    single = Counter()
    for (dataset, _task_id), values in grouped.items():
        single[(dataset, "CP1", effect_label(values["SSS"], values["VSS"]))] += 1
        single[(dataset, "CP2", effect_label(values["SSS"], values["SVS"]))] += 1
        single[(dataset, "CP3", effect_label(values["SSS"], values["SSV"]))] += 1
        for checkpoint in range(3):
            other = [index for index in range(3) if index != checkpoint]
            for bits in range(4):
                actions = ["S", "S", "S"]
                actions[other[0]] = "V" if bits & 1 else "S"
                actions[other[1]] = "V" if bits & 2 else "S"
                skip = actions.copy()
                verify = actions.copy()
                verify[checkpoint] = "V"
                label = effect_label(values["".join(skip)], values["".join(verify)])
                conditional[(dataset, f"CP{checkpoint + 1}", label)] += 1

    def counter_rows(counter: Counter, supervision: str) -> list[dict]:
        rows = []
        for dataset in ("humaneval", "livecodebench", "bigcodebench", "ALL"):
            for checkpoint in ("CP1", "CP2", "CP3"):
                counts = {
                    label: (
                        sum(counter[(source, checkpoint, label)]
                            for source in ("humaneval", "livecodebench", "bigcodebench"))
                        if dataset == "ALL" else counter[(dataset, checkpoint, label)]
                    )
                    for label in ("Helpful", "Harmful", "NoChange")
                }
                total = sum(counts.values())
                rows.append({
                    "supervision": supervision,
                    "dataset": dataset,
                    "checkpoint": checkpoint,
                    **counts,
                    "total": total,
                    "helpful_rate": counts["Helpful"] / total if total else 0.0,
                    "harmful_rate": counts["Harmful"] / total if total else 0.0,
                })
        return rows

    single_rows = counter_rows(single, "single_verify_from_SSS")
    conditional_rows = counter_rows(conditional, "all_shared_prefix_histories")
    write_csv(output / "train_single_verify_labels.csv", single_rows)
    write_csv(output / "train_conditional_effect_labels.csv", conditional_rows)

    frozen_manifest = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
    split_summary = {row["split"]: row for row in frozen_manifest["split_summary"]}
    updated_split = []
    for split in ("train", "validation", "test"):
        old = split_summary[split]
        added = 150 if split == "train" else 0
        old_l2_count = sum(
            row["dataset"] == dataset and row["split"] == split
            and row["level2_complete"] == "True"
            for dataset in ("humaneval", "livecodebench", "bigcodebench")
            for row in task_rows
        )
        updated_split.append({
            "split": split,
            "unique_tasks": int(old["unique_tasks"]),
            "level1_complete": int(old["level1_complete"]),
            "old_level2_complete": old_l2_count,
            "new_level2_tasks": added,
            "updated_level2_complete": old_l2_count + added,
            "old_raw_stored_trajectories": int(old["trajectories"]),
            "updated_raw_stored_trajectories": int(old["trajectories"]) + added * 8,
        })
    write_csv(output / "updated_split_summary.csv", updated_split)

    old_level2_complete_total = sum(
        row["level2_complete"] == "True" for row in task_rows
    )
    totals = {
        "unique_tasks": len(task_rows),
        "level1_complete_tasks": sum(row["level1_complete"] == "True" for row in task_rows),
        "old_level2_complete_tasks": old_level2_complete_total,
        "new_level2_extension_tasks": len(new_ids),
        "updated_level2_complete_tasks": old_level2_complete_total + len(new_ids),
        "old_level2_train_tasks": len({(key[0], key[1]) for key in old_l2}),
        "updated_level2_train_tasks": len(grouped),
        "updated_level2_train_trajectories": len(combined),
        "new_collection_rows": len(new_rows),
        "new_net_additional_task_sequence_rows": len(new_rows) - len(new_ids) * 4,
        "updated_raw_stored_trajectories": sum(row["updated_raw_stored_trajectories"] for row in updated_split),
        "updated_deduplicated_task_sequence_rows": (
            sum((row["updated_level2_complete"] * 8)
                + ((row["level1_complete"] - row["updated_level2_complete"]) * 4)
                for row in updated_split)
        ),
        "frozen_test_outcome_files_accessed": False,
    }
    recovery = json.loads((extension / "integrity/evaluator_recovery.json").read_text())
    validation_candidates = list((extension / "collection/validation").rglob("validation.json"))
    validation = json.loads(max(validation_candidates, key=lambda path: path.stat().st_mtime).read_text())
    totals.update({
        "extension_incremental_teacher_calls": recovery["teacher_calls_after"],
        "extension_incremental_teacher_tokens": recovery["teacher_tokens_after"],
        "extension_full_tree_unique_teacher_calls": validation["teacher_calls"],
        "extension_full_tree_unique_teacher_tokens": validation["teacher_tokens"],
    })
    summary = {
        "schema_version": "cbv-combined-data-summary-v1",
        "totals": totals,
        "task_inventory": inventory_rows,
        "split_summary": updated_split,
        "notes": [
            "The extension upgrades 150 existing BigCodeBench train tasks; it adds no new task IDs.",
            "Raw stored rows retain separate BigCodeBench L1/L2 conditions.",
            "Deduplicated task-sequence rows count each (dataset, task_id, sequence) once.",
            "Outcome/label statistics use train trajectories only; frozen validation/test outcomes are untouched.",
        ],
    }
    (output / "combined_data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    inventory_md = []
    for row in inventory_rows:
        inventory_md.append(
            f"| {row['dataset']} | {row['split']} | {row['unique_tasks']} | "
            f"{row['level1_complete']} | {row['old_level2_complete']} | "
            f"+{row['new_level2_extension']} | {row['updated_level2_complete']} |"
        )
    split_md = [
        f"| {row['split']} | {row['unique_tasks']} | {row['level1_complete']} | "
        f"{row['old_level2_complete']} | +{row['new_level2_tasks']} | "
        f"{row['updated_level2_complete']} |"
        for row in updated_split
    ]
    (output / "CBV_data_inventory.md").write_text("\n".join([
        "# CBV data inventory after BigCodeBench train L2 extension", "",
        "## Main totals", "",
        f"- Unique task IDs: **{totals['unique_tasks']:,}** (unchanged).",
        f"- Level-1 complete tasks: **{totals['level1_complete_tasks']:,}**.",
        f"- Level-2 complete tasks: **{totals['old_level2_complete_tasks']:,} → "
        f"{totals['updated_level2_complete_tasks']:,}** (**+150**).",
        f"- Train Level-2 tasks: **{totals['old_level2_train_tasks']:,} → "
        f"{totals['updated_level2_train_tasks']:,}**.",
        f"- Train Level-2 trajectories: **{totals['updated_level2_train_trajectories']:,}**.",
        f"- Net-new deduplicated task-sequence rows: "
        f"**{totals['new_net_additional_task_sequence_rows']:,}**.",
        f"- Updated deduplicated task-sequence rows across all splits: "
        f"**{totals['updated_deduplicated_task_sequence_rows']:,}**.", "",
        "## By dataset and split", "",
        "| Dataset | Split | Tasks | L1 complete | Old L2 | Added L2 | Updated L2 |",
        "|---|---:|---:|---:|---:|---:|---:|", *inventory_md, "",
        "## Split totals", "",
        "| Split | Tasks | L1 complete | Old L2 | Added L2 | Updated L2 |",
        "|---|---:|---:|---:|---:|---:|", *split_md, "",
        "## Extension accounting", "",
        f"- Complete extension records: **{len(new_rows):,} = 150 × 8**.",
        "- Reused Level-1 records: **600 = 150 × 4**.",
        "- Newly added multi-Verify records: **600 = 150 × 4**.",
        f"- Incremental Teacher usage: **{totals['extension_incremental_teacher_calls']:,} calls / "
        f"{totals['extension_incremental_teacher_tokens']:,} tokens**.",
        f"- Full extension tree accounting including imported L1: "
        f"**{totals['extension_full_tree_unique_teacher_calls']:,} calls / "
        f"{totals['extension_full_tree_unique_teacher_tokens']:,} tokens**.", "",
        "## Leakage boundary", "",
        "Outcome and label tables were computed from **train trajectories only**. "
        "Frozen validation/test trajectory files were not opened.", "",
    ]) + "\n", encoding="utf-8")
    print(output / "CBV_data_inventory.md")


if __name__ == "__main__":
    main()
