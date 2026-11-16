"""Outcome-blind planning and integrity checks for a train-only Level-2 extension.

This module deliberately reads only split/completeness metadata and frozen selection
manifests while choosing tasks.  It never opens frozen train/validation/test outcome
tables.  Collection outcomes are read only by :func:`validate_extension_output`, after
the new train-only run has completed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from agent_verify.budget import BudgetManager
from agent_verify.cbv_bench.cache import sha256_json
from agent_verify.cbv_bench.config import CBVConfig
from agent_verify.cbv_bench.sequences import LEVEL2_SEQUENCES
from agent_verify.schemas import utc_now


class ExtensionPlanningError(RuntimeError):
    """Raised when a leakage/completeness invariant is violated."""


@dataclass(frozen=True)
class ExtensionPaths:
    output_root: Path
    selection_manifest: Path
    audit_json: Path
    estimate_json: Path
    report_md: Path

    @classmethod
    def under(cls, output_root: Path) -> "ExtensionPaths":
        selection_dir = output_root / "selection"
        return cls(
            output_root=output_root,
            selection_manifest=selection_dir / "bigcodebench_train_l2_extension.json",
            audit_json=selection_dir / "selection_audit.json",
            estimate_json=output_root / "estimates" / "incremental_level2_estimate.json",
            report_md=output_root / "extension_plan.md",
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtensionPlanningError(f"Cannot read JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExtensionPlanningError(f"Expected a JSON object in {path}")
    return value


def _manifest_task_ids(path: Path) -> set[str]:
    """Read IDs from a selection/task-id manifest without inspecting outcomes."""
    if not path.exists():
        return set()
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        return {str(row["task_id"]) for row in rows if isinstance(row, dict) and row.get("task_id")}
    payload = _read_json(path)
    if isinstance(payload.get("task_records"), list):
        return {str(row["task_id"]) for row in payload["task_records"] if row.get("task_id")}
    values = payload.get("task_ids", [])
    if not isinstance(values, list):
        raise ExtensionPlanningError(f"Unsupported exploratory manifest schema: {path}")
    return {str(value) for value in values}


def _strict_bool(value: str, *, field: str, task_id: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ExtensionPlanningError(f"Invalid {field}={value!r} for {task_id}")


def _load_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "task_id", "dataset", "dataset_version", "split", "source_split",
        "level1_complete", "level2_complete", "public_task_sha256",
    }
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ExtensionPlanningError(f"Split manifest is missing columns: {sorted(missing)}")
    ids = [row["task_id"] for row in rows]
    duplicates = [task_id for task_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ExtensionPlanningError(f"Duplicate task IDs in split manifest: {duplicates[:5]}")
    return rows


def _load_universe(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = _read_json(path)
    records = payload.get("task_records")
    if not isinstance(records, list):
        raise ExtensionPlanningError(f"Missing task_records in {path}")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = str(record.get("task_id", ""))
        if not task_id:
            raise ExtensionPlanningError(f"Task record without task_id in {path}")
        if task_id in by_id:
            raise ExtensionPlanningError(f"Duplicate universe task_id: {task_id}")
        by_id[task_id] = record
    return by_id, payload


def _rank_key(seed: int, task_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"cbv-train-l2-extension-v1:{seed}:{task_id}".encode("utf-8")
    ).hexdigest()
    return digest, task_id


def select_training_tasks(
    *,
    split_manifest: Path,
    universe_manifest: Path,
    existing_level2_manifests: Iterable[Path],
    exploratory_manifests: Iterable[Path],
    max_tasks: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select an outcome-blind, train-only subset with complete L1 and missing L2."""
    if not 1 <= max_tasks <= 150:
        raise ExtensionPlanningError("max_tasks must be in [1, 150]")
    rows = _load_split_rows(split_manifest)
    universe, universe_payload = _load_universe(universe_manifest)
    existing_l2_ids = set().union(*(
        _manifest_task_ids(path) for path in existing_level2_manifests
    )) if existing_level2_manifests else set()
    exploratory_ids = set().union(*(
        _manifest_task_ids(path) for path in exploratory_manifests
    )) if exploratory_manifests else set()
    exploratory_ids.update(
        task_id for task_id, record in universe.items()
        if str(record.get("exploratory_reason", "")).strip()
        or record.get("split") == "exploratory"
    )

    counts = Counter()
    candidates: list[tuple[dict[str, str], dict[str, Any]]] = []
    validation_ids: set[str] = set()
    test_ids: set[str] = set()
    for row in rows:
        task_id = row["task_id"]
        if row["dataset"] != "bigcodebench":
            counts["non_bigcodebench"] += 1
            continue
        if row["split"] == "validation":
            validation_ids.add(task_id)
        elif row["split"] == "test":
            test_ids.add(task_id)
        if row["split"] != "train" or row["source_split"] != "train":
            counts[f"excluded_split_{row['split']}"] += 1
            continue
        if not _strict_bool(row["level1_complete"], field="level1_complete", task_id=task_id):
            counts["excluded_incomplete_level1"] += 1
            continue
        if _strict_bool(row["level2_complete"], field="level2_complete", task_id=task_id):
            counts["excluded_complete_level2"] += 1
            continue
        if task_id in existing_l2_ids:
            counts["excluded_existing_level2_manifest"] += 1
            continue
        if task_id in exploratory_ids:
            counts["excluded_exploratory"] += 1
            continue
        record = universe.get(task_id)
        if record is None:
            raise ExtensionPlanningError(f"Eligible task absent from universe manifest: {task_id}")
        if record.get("dataset") != "bigcodebench" or record.get("split") != "train":
            raise ExtensionPlanningError(
                f"Universe/split disagreement for {task_id}: {record.get('split')}"
            )
        if record.get("public_task_sha256") != row["public_task_sha256"]:
            raise ExtensionPlanningError(f"Public-task hash disagreement for {task_id}")
        candidates.append((row, record))

    candidates.sort(key=lambda item: _rank_key(seed, item[0]["task_id"]))
    selected_pairs = candidates[:max_tasks]
    if len(selected_pairs) < max_tasks:
        raise ExtensionPlanningError(
            f"Requested {max_tasks} tasks but only {len(selected_pairs)} are eligible"
        )
    selected_records: list[dict[str, Any]] = []
    for rank, (_row, source_record) in enumerate(selected_pairs):
        record = dict(source_record)
        record.update(
            selection_seed=seed,
            selection_rank=rank,
            split="train",
            exploratory_reason="",
        )
        selected_records.append(record)

    selected_ids = {record["task_id"] for record in selected_records}
    forbidden = selected_ids.intersection(validation_ids | test_ids | exploratory_ids | existing_l2_ids)
    if forbidden:
        raise ExtensionPlanningError(f"Leakage/duplicate exclusion failed: {sorted(forbidden)[:5]}")
    audit = {
        "schema_version": "cbv-train-l2-extension-audit-v1",
        "selection_protocol": (
            "SHA-256 ordering over task_id and fixed seed; uses split/completeness metadata only; "
            "no correctness, gain, trajectory, or evaluator outcome is read"
        ),
        "seed": seed,
        "requested_tasks": max_tasks,
        "eligible_before_cap": len(candidates),
        "selected_tasks": len(selected_records),
        "selected_task_ids": [record["task_id"] for record in selected_records],
        "exclusion_counts": dict(sorted(counts.items())),
        "known_existing_level2_ids": len(existing_l2_ids),
        "known_exploratory_ids": len(exploratory_ids),
        "validation_ids_selected": 0,
        "test_ids_selected": 0,
        "exploratory_ids_selected": 0,
        "duplicate_level2_ids_selected": 0,
        "frozen_test_outcomes_accessed": False,
        "outcome_files_accessed": [],
        "universe_selection_sha256": universe_payload.get("selection_sha256", ""),
    }
    return selected_records, audit


def estimate_incremental_collection(config: CBVConfig, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return cold-tree and L1-cache-reuse upper bounds without invoking a model."""
    models = config.models
    retry_factor = config.provider.max_retries + 1
    safety = config.collection.teacher_prompt_token_safety_factor
    cold_calls_by_cp = {1: 1, 2: 2, 3: 4}
    # L1 already contains VSS, SVS and SSV.  Level 2 adds VV, VSV, SVV and VVV edges.
    incremental_calls_by_cp = {1: 0, 2: 1, 3: 3}

    def token_bound(calls_by_cp: dict[int, int]) -> int:
        total = 0
        for record in records:
            prompt = str(record.get("public_task", {}).get("prompt", ""))
            public_tokens = BudgetManager.estimate_tokens(prompt)
            input_upper = {
                1: public_tokens + models.planner.max_output_tokens + 900,
                2: public_tokens + models.planner.max_output_tokens
                + models.coder.max_output_tokens + 900,
                3: public_tokens + models.planner.max_output_tokens
                + models.coder.max_output_tokens + models.reviewer.max_output_tokens + 900,
            }
            total += sum(
                calls_by_cp[checkpoint]
                * (input_upper[checkpoint] + models.teacher.max_output_tokens)
                for checkpoint in (1, 2, 3)
            )
        return int(total * safety * retry_factor)

    task_count = len(records)
    return {
        "schema_version": "cbv-train-l2-extension-estimate-v1",
        "tasks": task_count,
        "level2_trajectories": task_count * len(LEVEL2_SEQUENCES),
        "new_multi_verify_trajectories": task_count * 4,
        "cold_level2_teacher_calls": task_count * sum(cold_calls_by_cp.values()),
        "cold_level2_teacher_calls_by_checkpoint_per_task": cold_calls_by_cp,
        "cold_level2_teacher_token_upper_bound": token_bound(cold_calls_by_cp),
        "expected_incremental_teacher_calls_after_level1_reuse": (
            task_count * sum(incremental_calls_by_cp.values())
        ),
        "incremental_teacher_calls_by_checkpoint_per_task": incremental_calls_by_cp,
        "expected_incremental_teacher_token_upper_bound_after_level1_reuse": (
            token_bound(incremental_calls_by_cp)
        ),
        "assumptions": [
            "Every selected task imports all four complete Level-1 trajectories.",
            "Only the four missing shared-prefix Level-2 branches are generated.",
            "Token bounds use configured maximum outputs, prompt envelopes, safety factor, and retries.",
            "Actual calls/tokens can be lower after compatible partial-run cache hits.",
        ],
    }


def prepare_extension(
    *,
    config: CBVConfig,
    output_root: Path,
    split_manifest: Path,
    universe_manifest: Path,
    existing_level2_manifests: list[Path],
    exploratory_manifests: list[Path],
    max_tasks: int,
    seed: int,
) -> dict[str, Any]:
    paths = ExtensionPaths.under(output_root)
    records, audit = select_training_tasks(
        split_manifest=split_manifest,
        universe_manifest=universe_manifest,
        existing_level2_manifests=existing_level2_manifests,
        exploratory_manifests=exploratory_manifests,
        max_tasks=max_tasks,
        seed=seed,
    )
    paths.selection_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.estimate_json.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "cbv-selection-v2",
        "created_at_utc": utc_now(),
        "dataset": "bigcodebench",
        "dataset_version": "v0.1.4",
        "mode": "train-level2-extension",
        "selection_seed": seed,
        "selection_protocol": audit["selection_protocol"],
        "source_metadata": {
            "split_manifest": str(split_manifest),
            "split_manifest_sha256": _sha256_file(split_manifest),
            "universe_manifest": str(universe_manifest),
            "universe_manifest_sha256": _sha256_file(universe_manifest),
            "existing_level2_manifests": [str(path) for path in existing_level2_manifests],
            "exploratory_manifests": [str(path) for path in exploratory_manifests],
        },
        "task_records": records,
        "extension": {
            "target_level": 2,
            "requires_complete_level1": True,
            "split": "train",
            "outcome_blind": True,
        },
    }
    payload["selection_sha256"] = sha256_json(payload)
    if paths.selection_manifest.exists():
        existing = _read_json(paths.selection_manifest)
        existing_ids = [row["task_id"] for row in existing.get("task_records", [])]
        new_ids = [row["task_id"] for row in records]
        if existing_ids != new_ids or existing.get("selection_seed") != seed:
            raise ExtensionPlanningError(
                f"Existing selection differs; choose a new output directory: {paths.selection_manifest}"
            )
        payload = existing
    else:
        paths.selection_manifest.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    estimate = estimate_incremental_collection(config, records)
    audit.update({
        "created_at_utc": utc_now(),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": _sha256_file(split_manifest),
        "universe_manifest": str(universe_manifest),
        "universe_manifest_sha256": _sha256_file(universe_manifest),
        "selection_manifest": str(paths.selection_manifest),
        "selection_sha256": payload["selection_sha256"],
    })
    paths.audit_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    paths.estimate_json.write_text(json.dumps(estimate, indent=2) + "\n", encoding="utf-8")
    ids_preview = "\n".join(f"- `{task_id}`" for task_id in audit["selected_task_ids"][:20])
    paths.report_md.write_text(
        "\n".join([
            "# CBV-Bench BigCodeBench train Level-2 extension plan", "",
            "## Selection", "",
            f"- Selected: **{audit['selected_tasks']}** / {audit['eligible_before_cap']} eligible tasks.",
            "- Required state: complete Level-1 and missing Level-2.",
            "- Allowed split: **train/source train only**.",
            "- Validation, frozen-test, exploratory and prior Level-2 IDs selected: **0**.",
            f"- Deterministic seed: `{seed}`.",
            "- Selection uses identifiers and completeness metadata only; no outcome is used.", "",
            "## Estimated collection", "",
            f"- Final Level-2 trajectories: **{estimate['level2_trajectories']:,}** (8/task).",
            f"- Newly required multi-Verify trajectories: **{estimate['new_multi_verify_trajectories']:,}**.",
            f"- Expected new Teacher calls after Level-1 reuse: "
            f"**{estimate['expected_incremental_teacher_calls_after_level1_reuse']:,}** (4/task).",
            f"- Incremental Teacher-token conservative upper bound: "
            f"**{estimate['expected_incremental_teacher_token_upper_bound_after_level1_reuse']:,}**.",
            f"- Cold Level-2 upper bound (if reuse is rejected): "
            f"**{estimate['cold_level2_teacher_calls']:,} calls / "
            f"{estimate['cold_level2_teacher_token_upper_bound']:,} tokens**.", "",
            "## Output paths", "",
            f"- Selection: `{paths.selection_manifest}`",
            f"- Selection audit: `{paths.audit_json}`",
            f"- Estimate: `{paths.estimate_json}`",
            f"- Collection directory: `{output_root / 'collection'}`",
            f"- Run logs: `{output_root / 'logs'}`", "",
            "## First 20 selected task IDs", "", ids_preview, "",
        ]) + "\n",
        encoding="utf-8",
    )
    return {"paths": paths, "audit": audit, "estimate": estimate, "manifest": payload}


def validate_extension_output(
    *, output_root: Path, split_manifest: Path, exploratory_manifests: list[Path]
) -> dict[str, Any]:
    """Check the newly collected train-only Level-2 scope after standard CBV validation."""
    paths = ExtensionPaths.under(output_root)
    selection = _read_json(paths.selection_manifest)
    selected = {str(row["task_id"]): row for row in selection.get("task_records", [])}
    split_rows = {row["task_id"]: row for row in _load_split_rows(split_manifest)}
    exploratory_ids = set().union(*(
        _manifest_task_ids(path) for path in exploratory_manifests
    )) if exploratory_manifests else set()
    errors: list[str] = []
    for task_id, record in selected.items():
        frozen = split_rows.get(task_id)
        if frozen is None:
            errors.append(f"Selected task absent from split manifest: {task_id}")
            continue
        if frozen["split"] != "train" or frozen["source_split"] != "train":
            errors.append(f"Non-train task selected: {task_id}")
        if record.get("split") != "train":
            errors.append(f"Selection record is not train: {task_id}")
        if task_id in exploratory_ids:
            errors.append(f"Exploratory task selected: {task_id}")

    collection = output_root / "collection"
    manifests_path = collection / "manifests.jsonl"
    manifests = [json.loads(line) for line in manifests_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()] if manifests_path.exists() else []
    matching = [row for row in manifests if row.get("requested_level") == 2]
    if not matching:
        errors.append("No Level-2 collection manifest found")
        compatibility = ""
    else:
        latest = matching[-1]
        compatibility = str(latest.get("compatibility_fingerprint", ""))
        if latest.get("status") != "complete":
            errors.append(f"Latest Level-2 run status is {latest.get('status')!r}, not complete")

    trajectory_path = collection / "trajectories.jsonl"
    rows = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if trajectory_path.exists() else []
    rows = [row for row in rows if row.get("compatibility_fingerprint") == compatibility]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if task_id not in selected:
            errors.append(f"Out-of-scope trajectory task: {task_id}")
            continue
        if row.get("split") != "train":
            errors.append(f"Non-train trajectory: {task_id}/{row.get('action_sequence')}")
        if int(row.get("rollout_index", -1)) != 0:
            errors.append(f"Unexpected rollout index for {task_id}: {row.get('rollout_index')}")
        grouped[task_id].append(row)

    required = set(LEVEL2_SEQUENCES)
    for task_id in selected:
        sequences = [str(row.get("action_sequence", "")) for row in grouped.get(task_id, [])]
        counts = Counter(sequences)
        missing = required.difference(counts)
        duplicates = sorted(sequence for sequence, count in counts.items() if count > 1)
        unexpected = set(counts).difference(required)
        if missing:
            errors.append(f"{task_id}: missing sequences {sorted(missing)}")
        if duplicates:
            errors.append(f"{task_id}: duplicate sequences {duplicates}")
        if unexpected:
            errors.append(f"{task_id}: unexpected sequences {sorted(unexpected)}")

    report = {
        "schema_version": "cbv-train-l2-extension-integrity-v1",
        "valid": not errors,
        "selected_tasks": len(selected),
        "complete_level2_tasks": sum(
            {str(row.get("action_sequence", "")) for row in task_rows} == required
            for task_rows in grouped.values()
        ),
        "trajectory_count": len(rows),
        "expected_trajectory_count": len(selected) * len(required),
        "compatibility_fingerprint": compatibility,
        "frozen_test_outcomes_accessed": False,
        "errors": errors,
    }
    validation_dir = output_root / "integrity"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "extension_integrity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (validation_dir / "extension_integrity.md").write_text(
        "\n".join([
            "# Train Level-2 extension integrity", "",
            f"- Valid: **{report['valid']}**",
            f"- Complete tasks: **{report['complete_level2_tasks']}/{report['selected_tasks']}**",
            f"- Trajectories: **{report['trajectory_count']}/{report['expected_trajectory_count']}**",
            "- Frozen-test outcomes accessed: **No**", "", "## Errors", "",
            *(f"- {error}" for error in errors),
            *( ["- None"] if not errors else [] ), "",
        ]) + "\n",
        encoding="utf-8",
    )
    return report
