"""Qwen3.8 full eight-trajectory checkpoint-verification pilot.

The experiment is intentionally train-only and reuses the cached Qwen2.5 Student
``SSS`` root/prefix where it is exactly compatible.  It then constructs the full
shared-prefix tree causally.  A verified artifact always flows through the real
downstream Student agent before a later checkpoint is reached.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from tqdm.auto import tqdm

from agent_verify.agents.fixed import CoderAgent, ReviewerAgent
from agent_verify.analysis.qwen38_vvv_pilot import (
    DATASETS,
    RuntimeConfig,
    SmokeConfig,
    StudentConfig,
    TeacherConfig,
    _atomic_csv,
    _atomic_json,
    _evaluator,
    _invocation,
    _load_benchmark_selections,
    _load_groups,
    _model_from_cached,
    _provider,
    _reasoning_tokens,
    _resolve,
    _student_recovery,
    _task_records,
    _teacher_model,
)
from agent_verify.analysis.qwen38_vvv_pilot import (
    check_apis as _check_apis,
)
from agent_verify.budget import BudgetManager
from agent_verify.cbv_bench.cache import sha256_json
from agent_verify.cbv_bench.sequences import LEVEL2_SEQUENCES, required_prefixes
from agent_verify.cbv_bench.tree import PrivacyGuard, deterministic_seed
from agent_verify.config import BudgetConfig, StrictModel
from agent_verify.llm import LLMClient
from agent_verify.logging import JsonlStore, content_id, hash_file
from agent_verify.schemas import (
    CheckpointState,
    CodeArtifact,
    InvocationContext,
    PlanArtifact,
    PublicTask,
    ReviewArtifact,
    utc_now,
)
from agent_verify.verification.teacher import TeacherVerifier, artifact_schema

SEQUENCES = tuple(LEVEL2_SEQUENCES)
COHORT_UNSOLVED = "old_oracle_unsolvable"
COHORT_SINGLE = "single_verify_rescued"
COHORT_SSS = "sss_correct"
COHORT_MULTI = "multi_verify_required"
COHORTS = (COHORT_UNSOLVED, COHORT_SINGLE, COHORT_SSS, COHORT_MULTI)
SINGLE_VERIFY = ("VSS", "SVS", "SSV")
MULTI_VERIFY = ("VVS", "VSV", "SVV", "VVV")


class DataConfig(StrictModel):
    frozen_root: Path
    extension_root: Path
    extension_selection_manifest: Path
    outcome_table: Path
    excluded_experiment_manifest: Path | None = None
    additional_excluded_experiment_manifests: list[Path] = Field(default_factory=list)
    # Optional exact task population and compatible VVV cache.  This mode is
    # used to complete the original VVV40 cohort without selecting new tasks or
    # repeating its three already-collected Teacher edges.
    reference_selection_manifest: Path | None = None
    preseed_vvv_checkpoint_root: Path | None = None
    # Optional locked subset that must be retained while selecting additional
    # disjoint tasks. Unlike exact reference mode, this does not imply cached
    # Qwen3.8 edges and is intended for prospective cohort extension.
    base_selection_manifest: Path | None = None
    split: Literal["train"] = "train"


class SelectionConfig(StrictModel):
    seed: int = 43
    old_oracle_unsolvable: int = Field(default=8, ge=0)
    single_verify_rescued: int = Field(default=6, ge=0)
    sss_correct: int = Field(default=4, ge=0)
    multi_verify_required: int = Field(default=2, ge=0)

    @property
    def total(self) -> int:
        return (
            self.old_oracle_unsolvable
            + self.single_verify_rescued
            + self.sss_correct
            + self.multi_verify_required
        )


class FullPilotConfig(StrictModel):
    schema_version: Literal["darc-qwen38-full20-v1", "darc-qwen38-full-v2"]
    data: DataConfig
    selection: SelectionConfig
    runtime: RuntimeConfig
    student: StudentConfig
    teacher: TeacherConfig
    smoke: SmokeConfig = Field(default_factory=SmokeConfig)

    @model_validator(mode="after")
    def locked_protocol(self) -> FullPilotConfig:
        if self.schema_version == "darc-qwen38-full20-v1":
            if self.selection.seed != 43 or self.selection.total != 20:
                raise ValueError("Full20 locks seed=43 and exactly 20 tasks")
            if self.data.additional_excluded_experiment_manifests:
                raise ValueError("Legacy Full20 does not accept additional exclusions")
            if self.data.reference_selection_manifest is not None:
                raise ValueError("Legacy Full20 does not accept a reference selection")
        elif self.selection.total < 1:
            raise ValueError("Full trajectory pilot requires at least one task")
        if self.data.reference_selection_manifest is not None:
            if self.data.base_selection_manifest is not None:
                raise ValueError("Exact reference and base-extension modes are exclusive")
            if self.data.preseed_vvv_checkpoint_root is None:
                raise ValueError("Reference selection requires preseed_vvv_checkpoint_root")
        elif self.data.excluded_experiment_manifest is None:
            raise ValueError("Seeded selection requires excluded_experiment_manifest")
        if (
            self.schema_version == "darc-qwen38-full20-v1"
            and self.data.base_selection_manifest is not None
        ):
            raise ValueError("Legacy Full20 does not accept a base selection")
        if self.runtime.output_root == Path("results/darc_qwen38_vvv40"):
            raise ValueError("Full trajectory pilot must not overwrite Experiment 1")
        return self

    @property
    def experiment_label(self) -> str:
        return f"Full{self.selection.total}"


def load_config(path: Path) -> FullPilotConfig:
    import yaml

    return FullPilotConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def check_apis(config: FullPilotConfig) -> dict[str, Any]:
    """Use the shared readiness check with this experiment's configured ports."""

    return _check_apis(config)


def classify_outcome_row(row: dict[str, str]) -> str:
    outcomes = {sequence: int(row[f"outcome_{sequence}"]) for sequence in SEQUENCES}
    if not any(outcomes.values()):
        return COHORT_UNSOLVED
    if outcomes["SSS"]:
        return COHORT_SSS
    if any(outcomes[sequence] for sequence in SINGLE_VERIFY):
        return COHORT_SINGLE
    if any(outcomes[sequence] for sequence in MULTI_VERIFY):
        return COHORT_MULTI
    raise ValueError(f"Unclassifiable outcome pattern for {row.get('task_key')}")


def _stable_rank(namespace: str, seed: int, cohort: str, task_key: str) -> str:
    return hashlib.sha256(f"{namespace}:{seed}:{cohort}:{task_key}".encode()).hexdigest()


def select_tasks(
    rows: list[dict[str, str]],
    selection: SelectionConfig,
    excluded_task_keys: set[str],
    *,
    namespace: str = "qwen38-full20",
) -> list[dict[str, Any]]:
    """Select four disjoint strata after applying outcome-blind task exclusions."""

    if len(rows) != 531 or len({row["task_key"] for row in rows}) != 531:
        raise ValueError("Outcome table must contain exactly 531 unique task rows")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Outcome table contains a non-training task")
    pools: dict[str, list[dict[str, str]]] = {cohort: [] for cohort in COHORTS}
    for row in rows:
        if row["task_key"] not in excluded_task_keys:
            pools[classify_outcome_row(row)].append(row)
    requested = {
        COHORT_UNSOLVED: selection.old_oracle_unsolvable,
        COHORT_SINGLE: selection.single_verify_rescued,
        COHORT_SSS: selection.sss_correct,
        COHORT_MULTI: selection.multi_verify_required,
    }
    selected: list[dict[str, Any]] = []
    for cohort in COHORTS:
        ranked = sorted(
            pools[cohort],
            key=lambda row: _stable_rank(
                namespace, selection.seed, cohort, row["task_key"]
            ),
        )
        if len(ranked) < requested[cohort]:
            raise ValueError(f"Cohort {cohort} has only {len(ranked)} eligible tasks")
        for rank, row in enumerate(ranked[: requested[cohort]]):
            selected.append(
                {
                    **row,
                    "selection_cohort": cohort,
                    "selection_seed": selection.seed,
                    "cohort_rank": rank,
                }
            )
    keys = {row["task_key"] for row in selected}
    if len(selected) != selection.total or len(keys) != selection.total:
        raise AssertionError("Full selection is not distinct or has the wrong size")
    if keys & excluded_task_keys:
        raise AssertionError("Full selection overlaps an excluded experiment")
    return selected


def _dataset_selection_payload(
    dataset: str,
    selected: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    seed: int,
    *,
    mode: str = "qwen38-full20-seeded-disjoint-strata",
) -> dict[str, Any]:
    task_records = []
    for rank, selected_row in enumerate(selected):
        source = records[selected_row["task_key"]]
        task_records.append(
            {
                "schema_version": "cbv-bench-v2",
                "record_type": "cbv_task",
                "dataset": dataset,
                "dataset_version": source["dataset_version"],
                "task_id": source["task_id"],
                "difficulty": source.get("difficulty", "unknown"),
                "release_date": source.get("source_metadata", {}).get("release_date", ""),
                "source_metadata": source.get("source_metadata", {}),
                "public_task": source["public_task"],
                "public_task_sha256": source["public_task_sha256"],
                "selection_seed": seed,
                "selection_rank": rank,
                "split": "train",
                "exploratory_reason": "",
            }
        )
    payload = {
        "schema_version": "cbv-selection-v2",
        "created_at_utc": utc_now(),
        "dataset": dataset,
        "dataset_version": task_records[0]["dataset_version"] if task_records else "",
        "mode": mode,
        "selection_seed": seed,
        "selection_protocol": (
            "Exact training-only task population reused from the locked VVV40 manifest"
            if mode == "qwen38-full-exact-reference-manifest"
            else (
                (
                    "Locked training-only base selection plus deterministic SHA-256-ranked "
                    "extension in the same old-Qwen2.5 outcome strata"
                )
                if mode == "qwen38-full-base-extension-manifest"
                else (
                    "Training-only seeded SHA-256 ordering in four old-Qwen2.5 outcome "
                    "strata; "
                    + (
                        "all Experiment-1 tasks excluded before ranking"
                        if mode == "qwen38-full20-seeded-disjoint-strata"
                        else "all prior experiment tasks excluded before ranking"
                    )
                )
            )
        ),
        "source_metadata": {"cohorts": dict(Counter(row["selection_cohort"] for row in selected))},
        "task_records": task_records,
    }
    payload["selection_sha256"] = sha256_json(payload)
    return payload


def prepare_selection(config: FullPilotConfig, root: Path) -> tuple[list[dict[str, Any]], Path]:
    outcome_path = _resolve(root, config.data.outcome_table)
    with outcome_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    outcome_by_key = {row["task_key"]: row for row in rows}
    if config.data.reference_selection_manifest is not None:
        reference_path = _resolve(root, config.data.reference_selection_manifest)
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference selection is missing: {reference_path}")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_tasks = reference.get("tasks", [])
        reference_keys = [str(row["task_key"]) for row in reference_tasks]
        if len(reference_keys) != config.selection.total or len(set(reference_keys)) != len(
            reference_keys
        ):
            raise ValueError("Reference selection has the wrong size or duplicate tasks")
        missing = sorted(set(reference_keys) - set(outcome_by_key))
        if missing:
            raise ValueError(f"Reference selection contains unknown training tasks: {missing[:3]}")
        selected = []
        cohort_ranks: Counter[str] = Counter()
        for key in reference_keys:
            row = outcome_by_key[key]
            cohort = classify_outcome_row(row)
            selected.append(
                {
                    **row,
                    "selection_cohort": cohort,
                    "selection_seed": config.selection.seed,
                    "cohort_rank": cohort_ranks[cohort],
                }
            )
            cohort_ranks[cohort] += 1
        expected_counts = {
            COHORT_UNSOLVED: config.selection.old_oracle_unsolvable,
            COHORT_SINGLE: config.selection.single_verify_rescued,
            COHORT_SSS: config.selection.sss_correct,
            COHORT_MULTI: config.selection.multi_verify_required,
        }
        if dict(Counter(row["selection_cohort"] for row in selected)) != {
            key: value for key, value in expected_counts.items() if value
        }:
            raise ValueError("Reference cohort counts do not match the locked configuration")
        excluded: set[str] = set()
        excluded_manifests: list[dict[str, Any]] = []
    else:
        reference_path = None
        configured_exclusions = [
            config.data.excluded_experiment_manifest,
            *config.data.additional_excluded_experiment_manifests,
        ]
        if any(path is None for path in configured_exclusions):
            raise ValueError("Seeded selection has an empty exclusion manifest")
        configured_exclusions = [path for path in configured_exclusions if path is not None]
        excluded_paths = [_resolve(root, path) for path in configured_exclusions]
        excluded = set()
        excluded_manifests = []
        for configured_path, excluded_path in zip(
            configured_exclusions, excluded_paths, strict=True
        ):
            if not excluded_path.exists():
                raise FileNotFoundError(
                    f"Selection manifest is required for disjointness: {excluded_path}"
                )
            excluded_payload = json.loads(excluded_path.read_text(encoding="utf-8"))
            task_keys = {str(row["task_key"]) for row in excluded_payload.get("tasks", [])}
            if not task_keys:
                raise ValueError(f"Exclusion manifest contains no tasks: {excluded_path}")
            overlap = excluded & task_keys
            if overlap:
                raise ValueError(
                    f"Exclusion manifests overlap on {len(overlap)} tasks: {excluded_path}"
                )
            excluded.update(task_keys)
            excluded_manifests.append(
                {
                    "path": str(configured_path),
                    "sha256": hash_file(excluded_path),
                    "task_count": len(task_keys),
                }
            )
        if config.schema_version == "darc-qwen38-full20-v1" and len(excluded) != 40:
            raise ValueError(f"Expected 40 Experiment-1 exclusions, found {len(excluded)}")
        namespace = (
            "qwen38-full20"
            if config.schema_version == "darc-qwen38-full20-v1"
            else f"qwen38-full{config.selection.total}"
        )
        base_path = None
        base_selected: list[dict[str, Any]] = []
        base_counts: Counter[str] = Counter()
        if config.data.base_selection_manifest is not None:
            base_path = _resolve(root, config.data.base_selection_manifest)
            if not base_path.exists():
                raise FileNotFoundError(f"Base selection is missing: {base_path}")
            base_payload = json.loads(base_path.read_text(encoding="utf-8"))
            base_tasks = list(base_payload.get("tasks", []))
            base_keys = [str(row["task_key"]) for row in base_tasks]
            if not base_keys or len(base_keys) != len(set(base_keys)):
                raise ValueError("Base selection is empty or contains duplicate tasks")
            unknown = sorted(set(base_keys) - set(outcome_by_key))
            if unknown:
                raise ValueError(f"Base selection contains unknown tasks: {unknown[:3]}")
            overlap = set(base_keys) & excluded
            if overlap:
                raise ValueError(
                    f"Base selection overlaps prior experiment exclusions: {sorted(overlap)[:3]}"
                )
            for key in base_keys:
                row = outcome_by_key[key]
                cohort = classify_outcome_row(row)
                base_selected.append(
                    {
                        **row,
                        "selection_cohort": cohort,
                        "selection_seed": config.selection.seed,
                        "selection_source": "locked_base_manifest",
                    }
                )
                base_counts[cohort] += 1
            requested = {
                COHORT_UNSOLVED: config.selection.old_oracle_unsolvable,
                COHORT_SINGLE: config.selection.single_verify_rescued,
                COHORT_SSS: config.selection.sss_correct,
                COHORT_MULTI: config.selection.multi_verify_required,
            }
            if any(base_counts[cohort] > requested[cohort] for cohort in COHORTS):
                raise ValueError("Base cohort counts exceed the requested final selection")
            residual = SelectionConfig(
                old_oracle_unsolvable=requested[COHORT_UNSOLVED]
                - base_counts[COHORT_UNSOLVED],
                single_verify_rescued=requested[COHORT_SINGLE] - base_counts[COHORT_SINGLE],
                sss_correct=requested[COHORT_SSS] - base_counts[COHORT_SSS],
                multi_verify_required=requested[COHORT_MULTI] - base_counts[COHORT_MULTI],
                seed=config.selection.seed,
            )
            extension = select_tasks(
                rows,
                residual,
                excluded | set(base_keys),
                namespace=f"{namespace}-extension",
            )
            extension = [
                {**row, "selection_source": "deterministic_extension"}
                for row in extension
            ]
            selected = [*base_selected, *extension]
            cohort_ranks: Counter[str] = Counter()
            for row in selected:
                cohort = str(row["selection_cohort"])
                row["cohort_rank"] = cohort_ranks[cohort]
                cohort_ranks[cohort] += 1
            if len(selected) != config.selection.total:
                raise AssertionError("Base extension produced the wrong task count")
        else:
            selected = select_tasks(
                rows,
                config.selection,
                excluded,
                namespace=namespace,
            )

    output = _resolve(root, config.runtime.output_root)
    selection_dir = output / "selection"
    master_path = selection_dir / "selected_tasks.json"
    manifest = {
        "schema_version": (
            "darc-qwen38-full20-selection-v1"
            if config.schema_version == "darc-qwen38-full20-v1"
            else "darc-qwen38-full-selection-v2"
        ),
        "selection_seed": config.selection.seed,
        "source_outcome_table": str(config.data.outcome_table),
        "source_outcome_sha256": hash_file(outcome_path),
        "excluded_task_count": len(excluded),
        "tasks": selected,
        "counts": dict(Counter(row["selection_cohort"] for row in selected)),
        "dataset_counts": dict(Counter(row["dataset"] for row in selected)),
    }
    if config.schema_version == "darc-qwen38-full20-v1":
        manifest.update(
            {
                "excluded_experiment_manifest": str(
                    config.data.excluded_experiment_manifest
                ),
                "excluded_experiment_sha256": excluded_manifests[0]["sha256"],
            }
        )
    else:
        manifest["excluded_experiment_manifests"] = excluded_manifests
        if config.data.base_selection_manifest is not None:
            assert base_path is not None
            manifest.update(
                {
                    "base_selection_manifest": str(config.data.base_selection_manifest),
                    "base_selection_sha256": hash_file(base_path),
                    "base_task_count": len(base_selected),
                    "extension_task_count": len(selected) - len(base_selected),
                    "selection_mode": "locked_base_plus_deterministic_extension",
                }
            )
    if reference_path is not None:
        manifest.update(
            {
                "reference_selection_manifest": str(config.data.reference_selection_manifest),
                "reference_selection_sha256": hash_file(reference_path),
                "selection_mode": "exact_reference_manifest",
            }
        )
    manifest["selection_sha256"] = sha256_json(manifest)
    if master_path.exists():
        current = json.loads(master_path.read_text(encoding="utf-8"))
        if current != manifest:
            raise ValueError(
                f"Locked {config.experiment_label} selection differs from deterministic regeneration"
            )
    else:
        _atomic_json(master_path, manifest)
    _atomic_csv(selection_dir / "selected_tasks.csv", selected)

    records = _task_records(root, config.data.frozen_root)
    for dataset in DATASETS:
        dataset_rows = [row for row in selected if row["dataset"] == dataset]
        if not dataset_rows:
            continue
        payload = _dataset_selection_payload(
            dataset,
            dataset_rows,
            records,
            config.selection.seed,
            mode=(
                "qwen38-full20-seeded-disjoint-strata"
                if config.schema_version == "darc-qwen38-full20-v1"
                else (
                    "qwen38-full-exact-reference-manifest"
                    if reference_path is not None
                    else (
                        "qwen38-full-base-extension-manifest"
                        if config.data.base_selection_manifest is not None
                        else f"qwen38-full{config.selection.total}-seeded-disjoint-strata"
                    )
                )
            ),
        )
        path = selection_dir / f"{dataset}.json"
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            payload["created_at_utc"] = current.get("created_at_utc", payload["created_at_utc"])
            payload["selection_sha256"] = sha256_json(
                {key: value for key, value in payload.items() if key != "selection_sha256"}
            )
            if current != payload:
                raise ValueError(f"Locked dataset selection changed: {path}")
        else:
            _atomic_json(path, payload)
    return selected, master_path


def _state(
    *,
    run_id: str,
    task: PublicTask,
    checkpoint: int,
    prefix: str,
    rollout_seed: int,
    preceding: list[dict[str, Any]],
    current: dict[str, Any],
    model: Any,
    invocation_id: str,
    parent_state_id: str | None,
) -> CheckpointState:
    identity = {
        "run_id": run_id,
        "task_id": task.task_id,
        "checkpoint": checkpoint,
        "prefix": prefix,
        "preceding": preceding,
        "current": current,
        "parent": parent_state_id,
    }
    return CheckpointState(
        state_id=content_id("q38_full_state", identity),
        run_id=run_id,
        task_id=task.task_id,
        checkpoint_index=checkpoint,
        role={1: "planner", 2: "coder", 3: "reviewer"}[checkpoint],
        branch=f"qwen38-full:{prefix or 'root'}",
        rollout_seed=rollout_seed,
        parent_state_id=parent_state_id,
        preceding_artifacts=preceding,
        current_artifact=current,
        messages=[],
        llm_config=model,
        invocation_id=invocation_id,
    )


def _teacher_metrics(output: Any, store: JsonlStore) -> dict[str, Any]:
    record = _invocation(store, output.invocation_id)
    return {
        "verdict": output.result.verdict,
        "changed": bool(output.result.changed),
        "explanation": output.result.explanation,
        "corrected_artifact": output.corrected_artifact.model_dump(mode="json"),
        "invocation_id": output.invocation_id,
        "input_tokens": int(record["input_tokens"]),
        "output_tokens": int(record["output_tokens"]),
        "actual_total_tokens": int(record["input_tokens"]) + int(record["output_tokens"]),
        "reasoning_tokens": _reasoning_tokens(record),
        "latency_seconds": float(record["latency_seconds"]),
    }


def _student_metrics(output: Any) -> dict[str, Any]:
    record = output.llm_result.record
    return {
        "artifact": output.artifact.model_dump(mode="json"),
        "invocation_id": record.invocation_id,
        "input_tokens": int(record.input_tokens),
        "output_tokens": int(record.output_tokens),
        "latency_seconds": float(record.latency_seconds),
        "source": "generated",
    }


def _cached_downstream(artifact: dict[str, Any], invocation_id: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "invocation_id": invocation_id,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_seconds": 0.0,
        "source": "old_qwen25_sss_cache",
    }


def _task_file(output: Path, task_key: str) -> Path:
    digest = hashlib.sha256(task_key.encode()).hexdigest()[:16]
    return output / "task_checkpoints" / f"{digest}.json"


def _save(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at_utc"] = utc_now()
    _atomic_json(path, payload)


def _preseed_vvv_progress(
    progress: dict[str, Any],
    *,
    vvv_checkpoint: dict[str, Any],
    root_plan: PlanArtifact,
) -> None:
    """Import the compatible V/VV/VVV shared-prefix path from the VVV40 pilot."""

    if vvv_checkpoint.get("status") != "completed":
        raise ValueError("VVV cache checkpoint is not complete")
    if vvv_checkpoint.get("task_key") != progress.get("task_key"):
        raise ValueError("VVV cache task does not match the Full task")
    checkpoints = vvv_checkpoint.get("checkpoints", {})
    if set(checkpoints) != {"CP1", "CP2", "CP3"}:
        raise ValueError("VVV cache must contain CP1, CP2, and CP3")

    cp1 = dict(checkpoints["CP1"])
    cp2 = dict(checkpoints["CP2"])
    cp3 = dict(checkpoints["CP3"])
    for teacher in (cp1, cp2, cp3):
        teacher["source"] = "qwen38_vvv40_cache"

    coder = dict(vvv_checkpoint["coder_after_CP1"])
    coder["source"] = "qwen38_vvv40_cache"
    reviewer = dict(vvv_checkpoint["reviewer_after_CP2"])
    reviewer["source"] = "qwen38_vvv40_cache"
    plan_after = cp1["corrected_artifact"]
    code_before_cp2 = coder["artifact"]
    code_after_cp2 = cp2["corrected_artifact"]
    review_before_cp3 = reviewer["artifact"]
    review_after_cp3 = cp3["corrected_artifact"]

    cached_edges = {
        "V": {
            "checkpoint": 1,
            "prefix": "V",
            "parent_prefix": "",
            "action": "V",
            "input_artifact": root_plan.model_dump(mode="json"),
            "teacher": cp1,
            "effective_artifact": plan_after,
            "downstream": coder,
            "output_artifact": code_before_cp2,
            "complete": True,
            "source": "qwen38_vvv40_cache",
        },
        "VV": {
            "checkpoint": 2,
            "prefix": "VV",
            "parent_prefix": "V",
            "action": "V",
            "input_artifact": code_before_cp2,
            "teacher": cp2,
            "effective_artifact": code_after_cp2,
            "downstream": reviewer,
            "output_artifact": review_before_cp3,
            "complete": True,
            "source": "qwen38_vvv40_cache",
        },
        "VVV": {
            "checkpoint": 3,
            "prefix": "VVV",
            "parent_prefix": "VV",
            "action": "V",
            "input_artifact": review_before_cp3,
            "teacher": cp3,
            "effective_artifact": review_after_cp3,
            "output_artifact": review_after_cp3,
            "complete": True,
            "source": "qwen38_vvv40_cache",
        },
    }
    for prefix, edge in cached_edges.items():
        progress["edges"].setdefault(prefix, edge)

    evaluation = dict(vvv_checkpoint["evaluation"])
    progress["trajectories"].setdefault(
        "VVV",
        {
            "sequence": "VVV",
            "passed": bool(evaluation["passed"]),
            "evaluation": evaluation,
            "evaluation_source": "qwen38_vvv40_cache",
            "teacher_calls": 3,
            "teacher_input_tokens": sum(int(cp["input_tokens"]) for cp in (cp1, cp2, cp3)),
            "teacher_output_tokens": sum(
                int(cp["output_tokens"]) for cp in (cp1, cp2, cp3)
            ),
            "teacher_reasoning_tokens": sum(
                int(cp["reasoning_tokens"]) for cp in (cp1, cp2, cp3)
            ),
            "teacher_total_tokens": sum(
                int(cp["actual_total_tokens"]) for cp in (cp1, cp2, cp3)
            ),
            "teacher_latency_seconds": sum(
                float(cp["latency_seconds"]) for cp in (cp1, cp2, cp3)
            ),
        },
    )
    progress["preseed"] = {
        "source": "qwen38_vvv40_cache",
        "reused_teacher_edges": ["V", "VV", "VVV"],
        "reused_trajectories": ["VVV"],
    }


def _run_task(
    *,
    selected_row: dict[str, Any],
    group: Any,
    task: PublicTask,
    private_case: Any,
    evaluator: Any,
    config: FullPilotConfig,
    client: LLMClient,
    invocations: JsonlStore,
    events: JsonlStore,
    root: Path,
    output: Path,
    run_id: str,
) -> dict[str, Any]:
    path = _task_file(output, selected_row["task_key"])
    if path.exists():
        progress = json.loads(path.read_text(encoding="utf-8"))
        if progress.get("status") == "completed":
            return progress
    else:
        progress = {
            "schema_version": f"darc-qwen38-{config.experiment_label.lower()}-task-v1",
            "status": "running",
            "run_id": run_id,
            "task_key": selected_row["task_key"],
            "dataset": selected_row["dataset"],
            "task_id": selected_row["task_id"],
            "selection_cohort": selected_row["selection_cohort"],
            "old_outcomes": {
                sequence: bool(int(selected_row[f"outcome_{sequence}"])) for sequence in SEQUENCES
            },
            "edges": {},
            "trajectories": {},
        }
        _save(path, progress)

    cached_sss = group.rows["SSS"]
    rollout_seed = int(cached_sss["rollout_seed"])
    planner_model = _model_from_cached(cached_sss["student_models"]["planner"], config)
    coder_model = _model_from_cached(cached_sss["student_models"]["coder"], config)
    reviewer_model = _model_from_cached(cached_sss["student_models"]["reviewer"], config)
    teacher_model = _teacher_model(config)
    teacher = TeacherVerifier(
        client,
        teacher_model,
        events,
        request_overrides={
            "thinking_token_budget": config.teacher.thinking_token_budget,
            "include_reasoning": config.teacher.include_reasoning,
        },
    )
    # Keep the frozen Student generation protocol unchanged on the first attempt.
    # A narrowly scoped anti-repetition override is enabled only after the
    # invocation log proves that the same task/role previously hit the output
    # limit.  This makes interrupted shared-tree collection resumable without
    # silently changing the protocol for successful tasks.
    coder_recovery = _student_recovery(invocations, task_id=task.task_id, role="coder")
    reviewer_recovery = _student_recovery(invocations, task_id=task.task_id, role="reviewer")
    coder = CoderAgent(
        client,
        coder_model,
        request_overrides=(
            {"repetition_penalty": coder_recovery["repetition_penalty"]}
            if coder_recovery
            else None
        ),
    )
    reviewer = ReviewerAgent(
        client,
        reviewer_model,
        events,
        request_overrides=(
            {"repetition_penalty": reviewer_recovery["repetition_penalty"]}
            if reviewer_recovery
            else None
        ),
        line_safe_output=bool(
            reviewer_recovery and reviewer_recovery.get("line_safe_output")
        ),
    )
    if coder_recovery or reviewer_recovery:
        progress["student_recovery"] = {
            "coder": coder_recovery,
            "reviewer": reviewer_recovery,
        }
        _save(path, progress)

    plan = PlanArtifact.model_validate(cached_sss["artifact_snapshots"][0])
    if config.data.preseed_vvv_checkpoint_root is not None:
        vvv_path = _resolve(root, config.data.preseed_vvv_checkpoint_root) / path.name
        if not vvv_path.exists():
            raise FileNotFoundError(f"Missing compatible VVV checkpoint: {vvv_path}")
        _preseed_vvv_progress(
            progress,
            vvv_checkpoint=json.loads(vvv_path.read_text(encoding="utf-8")),
            root_plan=plan,
        )
        _save(path, progress)
    root_state = _state(
        run_id=run_id,
        task=task,
        checkpoint=1,
        prefix="",
        rollout_seed=rollout_seed,
        preceding=[],
        current=plan.model_dump(mode="json"),
        model=planner_model,
        invocation_id=f"cached:{cached_sss['checkpoint_state_ids'][0]}",
        parent_state_id=None,
    )
    states: dict[str, CheckpointState] = {"": root_state}
    terminals: dict[str, ReviewArtifact] = {}

    for prefix in required_prefixes(2):
        parent = prefix[:-1]
        input_state = states[parent]
        checkpoint = len(prefix)
        action = prefix[-1]
        schema = artifact_schema(checkpoint)
        original = schema.model_validate(input_state.current_artifact)
        edge = progress["edges"].setdefault(
            prefix,
            {
                "checkpoint": checkpoint,
                "prefix": prefix,
                "parent_prefix": parent,
                "action": action,
                "input_artifact": original.model_dump(mode="json"),
            },
        )
        effective = original
        if action == "V":
            if "teacher" not in edge:
                context = InvocationContext(
                    task_id=task.task_id,
                    checkpoint=checkpoint,
                    role="teacher",
                    branch=f"qwen38-full:{prefix}",
                    run_id=run_id,
                    seed=deterministic_seed(
                        rollout_seed, checkpoint, teacher_model.seed, prefix
                    ),
                )
                verified = teacher.verify(task, input_state, original, context)
                record = _invocation(invocations, verified.invocation_id)
                PrivacyGuard.assert_messages_public(record["messages"], private_case)
                edge["teacher"] = _teacher_metrics(verified, invocations)
                edge["effective_artifact"] = edge["teacher"]["corrected_artifact"]
                _save(path, progress)
            effective = schema.model_validate(edge["teacher"]["corrected_artifact"])
        else:
            edge["effective_artifact"] = original.model_dump(mode="json")

        if checkpoint == 1:
            effective_plan = PlanArtifact.model_validate(effective.model_dump())
            if "downstream" not in edge:
                if prefix == "S":
                    edge["downstream"] = _cached_downstream(
                        cached_sss["artifact_snapshots"][1],
                        f"cached:{cached_sss['checkpoint_state_ids'][1]}",
                    )
                else:
                    context = InvocationContext(
                        task_id=task.task_id,
                        checkpoint=2,
                        role="coder",
                        branch=f"qwen38-full:{prefix}",
                        run_id=run_id,
                        seed=deterministic_seed(rollout_seed, 2, coder_model.seed, prefix),
                    )
                    generated = coder.run(task, effective_plan, context)
                    PrivacyGuard.assert_messages_public(generated.messages, private_case)
                    edge["downstream"] = _student_metrics(generated)
                    if coder_recovery:
                        edge["student_recovery"] = coder_recovery
                _save(path, progress)
            code = CodeArtifact.model_validate(edge["downstream"]["artifact"])
            states[prefix] = _state(
                run_id=run_id,
                task=task,
                checkpoint=2,
                prefix=prefix,
                rollout_seed=rollout_seed,
                preceding=[effective_plan.model_dump(mode="json")],
                current=code.model_dump(mode="json"),
                model=coder_model,
                invocation_id=edge["downstream"]["invocation_id"],
                parent_state_id=input_state.state_id,
            )
            edge["output_artifact"] = code.model_dump(mode="json")
        elif checkpoint == 2:
            preceding_plan = PlanArtifact.model_validate(input_state.preceding_artifacts[0])
            effective_code = CodeArtifact.model_validate(effective.model_dump())
            if "downstream" not in edge:
                if prefix == "SS":
                    edge["downstream"] = _cached_downstream(
                        cached_sss["artifact_snapshots"][2],
                        f"cached:{cached_sss['checkpoint_state_ids'][2]}",
                    )
                else:
                    context = InvocationContext(
                        task_id=task.task_id,
                        checkpoint=3,
                        role="reviewer",
                        branch=f"qwen38-full:{prefix}",
                        run_id=run_id,
                        seed=deterministic_seed(rollout_seed, 3, reviewer_model.seed, prefix),
                    )
                    generated = reviewer.run(task, preceding_plan, effective_code, context)
                    PrivacyGuard.assert_messages_public(generated.messages, private_case)
                    edge["downstream"] = _student_metrics(generated)
                    if reviewer_recovery:
                        edge["student_recovery"] = reviewer_recovery
                _save(path, progress)
            review = ReviewArtifact.model_validate(edge["downstream"]["artifact"])
            states[prefix] = _state(
                run_id=run_id,
                task=task,
                checkpoint=3,
                prefix=prefix,
                rollout_seed=rollout_seed,
                preceding=[
                    preceding_plan.model_dump(mode="json"),
                    effective_code.model_dump(mode="json"),
                ],
                current=review.model_dump(mode="json"),
                model=reviewer_model,
                invocation_id=edge["downstream"]["invocation_id"],
                parent_state_id=input_state.state_id,
            )
            edge["output_artifact"] = review.model_dump(mode="json")
        else:
            terminal = ReviewArtifact.model_validate(effective.model_dump())
            terminals[prefix] = terminal
            edge["output_artifact"] = terminal.model_dump(mode="json")
        edge["complete"] = True
        _save(path, progress)

    for sequence in SEQUENCES:
        if sequence in progress["trajectories"]:
            continue
        prefixes = [sequence[:depth] for depth in (1, 2, 3)]
        teacher_edges = [progress["edges"][prefix] for prefix in prefixes if prefix.endswith("V")]
        if sequence == "SSS":
            evaluation = cached_sss["evaluation"]
            evaluation_source = "old_qwen25_sss_cache"
        else:
            result = evaluator.evaluate(
                run_id=run_id,
                case=private_case,
                code=terminals[sequence].code,
                branch=f"qwen38-full:{sequence}",
            )
            evaluation = result.model_dump(mode="json")
            evaluation_source = "generated"
        progress["trajectories"][sequence] = {
            "sequence": sequence,
            "passed": bool(evaluation["passed"]),
            "evaluation": evaluation,
            "evaluation_source": evaluation_source,
            "teacher_calls": len(teacher_edges),
            "teacher_input_tokens": sum(
                int(edge["teacher"]["input_tokens"]) for edge in teacher_edges
            ),
            "teacher_output_tokens": sum(
                int(edge["teacher"]["output_tokens"]) for edge in teacher_edges
            ),
            "teacher_reasoning_tokens": sum(
                int(edge["teacher"]["reasoning_tokens"]) for edge in teacher_edges
            ),
            "teacher_total_tokens": sum(
                int(edge["teacher"]["actual_total_tokens"]) for edge in teacher_edges
            ),
            "teacher_latency_seconds": sum(
                float(edge["teacher"]["latency_seconds"]) for edge in teacher_edges
            ),
        }
        _save(path, progress)

    old_rows = group.rows
    progress["old_trajectories"] = {
        sequence: {
            "passed": bool(old_rows[sequence]["passed"]),
            "teacher_calls": int(old_rows[sequence]["teacher_calls"]),
            "teacher_total_tokens": int(old_rows[sequence]["teacher_total_tokens"]),
            "total_latency_seconds": float(old_rows[sequence]["total_latency_seconds"]),
        }
        for sequence in SEQUENCES
    }
    progress["new_oracle_passed"] = any(
        bool(progress["trajectories"][sequence]["passed"]) for sequence in SEQUENCES
    )
    progress["old_oracle_passed"] = any(
        bool(progress["old_trajectories"][sequence]["passed"]) for sequence in SEQUENCES
    )
    progress["unique_teacher_calls"] = sum(
        int(prefix.endswith("V")) for prefix in progress["edges"]
    )
    progress["unique_teacher_tokens"] = sum(
        int(edge["teacher"]["actual_total_tokens"])
        for prefix, edge in progress["edges"].items()
        if prefix.endswith("V")
    )
    progress["unique_teacher_latency_seconds"] = sum(
        float(edge["teacher"]["latency_seconds"])
        for prefix, edge in progress["edges"].items()
        if prefix.endswith("V")
    )
    progress["reused_teacher_calls"] = sum(
        int(edge.get("teacher", {}).get("source") == "qwen38_vvv40_cache")
        for prefix, edge in progress["edges"].items()
        if prefix.endswith("V")
    )
    progress["incremental_teacher_calls"] = (
        progress["unique_teacher_calls"] - progress["reused_teacher_calls"]
    )
    progress["incremental_teacher_tokens"] = sum(
        int(edge["teacher"]["actual_total_tokens"])
        for prefix, edge in progress["edges"].items()
        if prefix.endswith("V")
        and edge.get("teacher", {}).get("source") != "qwen38_vvv40_cache"
    )
    progress["status"] = "completed"
    _save(path, progress)
    return progress


def _trajectory_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for sequence in SEQUENCES:
            new = result["trajectories"][sequence]
            old = result["old_trajectories"][sequence]
            rows.append(
                {
                    "task_key": result["task_key"],
                    "dataset": result["dataset"],
                    "selection_cohort": result["selection_cohort"],
                    "sequence": sequence,
                    "verify_calls": sequence.count("V"),
                    "old_qwen25_passed": int(old["passed"]),
                    "new_qwen38_passed": int(new["passed"]),
                    "old_teacher_tokens": old["teacher_total_tokens"],
                    "new_teacher_tokens": new["teacher_total_tokens"],
                    "new_reasoning_tokens": new["teacher_reasoning_tokens"],
                    "new_teacher_latency_seconds": new["teacher_latency_seconds"],
                }
            )
    return rows


def conditional_effect_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        trajectories = result["trajectories"]
        for checkpoint_index in range(3):
            checkpoint = checkpoint_index + 1
            histories = [""] if checkpoint_index == 0 else [
                "".join(bits)
                for bits in __import__("itertools").product("SV", repeat=checkpoint_index)
            ]
            suffix_length = 2 - checkpoint_index
            suffixes = [
                "".join(bits)
                for bits in __import__("itertools").product("SV", repeat=suffix_length)
            ]
            for history in histories:
                for suffix in suffixes:
                    skip_sequence = history + "S" + suffix
                    verify_sequence = history + "V" + suffix
                    y_skip = int(trajectories[skip_sequence]["passed"])
                    y_verify = int(trajectories[verify_sequence]["passed"])
                    gain = y_verify - y_skip
                    effect = "Helpful" if gain > 0 else "Harmful" if gain < 0 else "NoChange"
                    verify_prefix = verify_sequence[:checkpoint]
                    teacher = result["edges"][verify_prefix]["teacher"]
                    rows.append(
                        {
                            "task_key": result["task_key"],
                            "dataset": result["dataset"],
                            "selection_cohort": result["selection_cohort"],
                            "checkpoint": checkpoint,
                            "history": history,
                            "future_suffix": suffix,
                            "skip_sequence": skip_sequence,
                            "verify_sequence": verify_sequence,
                            "y_skip": y_skip,
                            "y_verify": y_verify,
                            "gain": gain,
                            "effect": effect,
                            "verified_edge_changed": int(teacher["changed"]),
                            "verified_edge_tokens": teacher["actual_total_tokens"],
                            "trajectory_token_delta": (
                                int(trajectories[verify_sequence]["teacher_total_tokens"])
                                - int(trajectories[skip_sequence]["teacher_total_tokens"])
                            ),
                        }
                    )
    return rows


def minimum_correct(trajectories: dict[str, dict[str, Any]]) -> tuple[str, int] | tuple[None, None]:
    successful = [sequence for sequence in SEQUENCES if trajectories[sequence]["passed"]]
    if not successful:
        return None, None
    sequence = min(
        successful,
        key=lambda value: (
            int(trajectories[value]["teacher_total_tokens"]),
            value.count("V"),
            value,
        ),
    )
    return sequence, int(trajectories[sequence]["teacher_total_tokens"])


def _task_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        new_sequence, new_tokens = minimum_correct(result["trajectories"])
        old_sequence, old_tokens = minimum_correct(result["old_trajectories"])
        rows.append(
            {
                "task_key": result["task_key"],
                "dataset": result["dataset"],
                "task_id": result["task_id"],
                "selection_cohort": result["selection_cohort"],
                "old_oracle_passed": int(result["old_oracle_passed"]),
                "new_oracle_passed": int(result["new_oracle_passed"]),
                "newly_solvable": int(result["new_oracle_passed"] and not result["old_oracle_passed"]),
                "old_minimum_correct_sequence": old_sequence or "",
                "old_minimum_correct_tokens": "" if old_tokens is None else old_tokens,
                "new_minimum_correct_sequence": new_sequence or "",
                "new_minimum_correct_tokens": "" if new_tokens is None else new_tokens,
                "unique_teacher_calls": result["unique_teacher_calls"],
                "unique_teacher_tokens": result["unique_teacher_tokens"],
                "unique_teacher_latency_seconds": result["unique_teacher_latency_seconds"],
                "reused_teacher_calls": result.get("reused_teacher_calls", 0),
                "incremental_teacher_calls": result.get(
                    "incremental_teacher_calls", result["unique_teacher_calls"]
                ),
                "incremental_teacher_tokens": result.get(
                    "incremental_teacher_tokens", result["unique_teacher_tokens"]
                ),
            }
        )
    return rows


def _summary(results: list[dict[str, Any]], effects: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = len(results)
    old_correct = sum(bool(row["old_oracle_passed"]) for row in results)
    new_correct = sum(bool(row["new_oracle_passed"]) for row in results)
    by_checkpoint = {}
    for checkpoint in (1, 2, 3):
        subset = [row for row in effects if row["checkpoint"] == checkpoint]
        counts = Counter(row["effect"] for row in subset)
        by_checkpoint[f"CP{checkpoint}"] = {
            "comparisons": len(subset),
            "helpful": counts["Helpful"],
            "harmful": counts["Harmful"],
            "no_change": counts["NoChange"],
        }
    sequence_summary = {}
    trajectories = _trajectory_rows(results)
    for sequence in SEQUENCES:
        subset = [row for row in trajectories if row["sequence"] == sequence]
        sequence_summary[sequence] = {
            "old_qwen25_correct": sum(row["old_qwen25_passed"] for row in subset),
            "new_qwen38_correct": sum(row["new_qwen38_passed"] for row in subset),
            "new_teacher_tokens": sum(row["new_teacher_tokens"] for row in subset),
            "new_reasoning_tokens": sum(row["new_reasoning_tokens"] for row in subset),
            "new_teacher_latency_seconds": sum(
                float(row["new_teacher_latency_seconds"]) for row in subset
            ),
        }
    return {
        "tasks": tasks,
        "old_oracle_correct": old_correct,
        "old_oracle_accuracy": old_correct / tasks if tasks else 0.0,
        "new_oracle_correct": new_correct,
        "new_oracle_accuracy": new_correct / tasks if tasks else 0.0,
        "newly_solvable_tasks": [
            row["task_key"]
            for row in results
            if row["new_oracle_passed"] and not row["old_oracle_passed"]
        ],
        "cohorts": dict(Counter(row["selection_cohort"] for row in results)),
        "datasets": dict(Counter(row["dataset"] for row in results)),
        "unique_collection_teacher_calls": sum(row["unique_teacher_calls"] for row in results),
        "unique_collection_teacher_tokens": sum(row["unique_teacher_tokens"] for row in results),
        "unique_collection_teacher_latency_seconds": sum(
            row["unique_teacher_latency_seconds"] for row in results
        ),
        "reused_collection_teacher_calls": sum(
            row.get("reused_teacher_calls", 0) for row in results
        ),
        "incremental_collection_teacher_calls": sum(
            row.get("incremental_teacher_calls", row["unique_teacher_calls"])
            for row in results
        ),
        "incremental_collection_teacher_tokens": sum(
            row.get("incremental_teacher_tokens", row["unique_teacher_tokens"])
            for row in results
        ),
        "checkpoint_effects": by_checkpoint,
        "trajectory_summary": sequence_summary,
    }


def _write_report(
    output: Path,
    summary: dict[str, Any],
    mode: str,
    *,
    experiment_label: str = "Full20",
) -> Path:
    lines = [
        "# Qwen3.8 Full 8-Trajectory Verification Pilot",
        "",
        f"Mode: `{mode}`. All tasks are from the frozen Level-2 training split.",
        "",
        "## Oracle comparison",
        "",
        "| Teacher | Oracle solved | Oracle accuracy |",
        "|---|---:|---:|",
        (
            f"| Cached Qwen2.5 | {summary['old_oracle_correct']}/{summary['tasks']} | "
            f"{summary['old_oracle_accuracy']:.1%} |"
        ),
        (
            f"| Qwen3.8 | {summary['new_oracle_correct']}/{summary['tasks']} | "
            f"{summary['new_oracle_accuracy']:.1%} |"
        ),
        "",
        f"Newly solvable tasks: **{len(summary['newly_solvable_tasks'])}**.",
        "",
        "## Conditional checkpoint effects",
        "",
        "These are paired final-outcome effects under matched history and future action suffix.",
        "",
        "| Checkpoint | Comparisons | Helpful | Harmful | No-change |",
        "|---|---:|---:|---:|---:|",
    ]
    for checkpoint in ("CP1", "CP2", "CP3"):
        row = summary["checkpoint_effects"][checkpoint]
        lines.append(
            f"| {checkpoint} | {row['comparisons']} | {row['helpful']} | "
            f"{row['harmful']} | {row['no_change']} |"
        )
    lines.extend(
        [
            "",
            "## Collection cost",
            "",
            f"Unique Teacher calls: **{summary['unique_collection_teacher_calls']}**; "
            f"actual tokens: **{summary['unique_collection_teacher_tokens']:,}**; "
            f"summed latency: **{summary['unique_collection_teacher_latency_seconds'] / 60:.1f} min**.",
            (
                f"Reused Teacher calls: **{summary['reused_collection_teacher_calls']}**; "
                f"new calls made by this completion run: "
                f"**{summary['incremental_collection_teacher_calls']}** "
                f"({summary['incremental_collection_teacher_tokens']:,} new tokens)."
            ),
            "",
            "The trajectory table attributes shared edge costs to every path containing that edge. "
            "The collection totals above count each shared Teacher call exactly once.",
            "",
            "## Scientific boundary",
            "",
            "Selection uses only cached old-Qwen2.5 training outcomes. Hidden evaluator labels and "
            "feedback are never included in model messages. Oracle and minimum-correct paths are "
            "offline diagnostics, not deployable policies.",
        ]
    )
    report = output / "report" / f"darc_qwen38_{experiment_label.lower()}_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def dry_run(config: FullPilotConfig, root: Path) -> dict[str, Any]:
    selected, manifest = prepare_selection(config, root)
    reused_vvv = config.data.preseed_vvv_checkpoint_root is not None
    payload = {
        "mode": "dry-run",
        "tasks": len(selected),
        "trajectories": len(selected) * len(SEQUENCES),
        "cohorts": dict(Counter(row["selection_cohort"] for row in selected)),
        "datasets": dict(Counter(row["dataset"] for row in selected)),
        "unique_teacher_calls": len(selected) * 7,
        "reused_teacher_calls": len(selected) * (3 if reused_vvv else 0),
        "incremental_teacher_calls": len(selected) * (4 if reused_vvv else 7),
        "new_student_calls": len(selected) * (2 if reused_vvv else 4),
        "reused_student_prefixes": len(selected) * (4 if reused_vvv else 2),
        "reused_sss_evaluations": len(selected),
        "reused_vvv_evaluations": len(selected) if reused_vvv else 0,
        "new_trajectory_evaluations": len(selected) * (6 if reused_vvv else 7),
        "selection_manifest": str(manifest),
        "output_root": str(config.runtime.output_root),
    }
    _atomic_json(_resolve(root, config.runtime.output_root) / "dry_run.json", payload)
    return payload


def run(
    config: FullPilotConfig,
    root: Path,
    *,
    mode: Literal["smoke", "full"],
    resume: bool,
) -> dict[str, Any]:
    preflight_started = time.perf_counter()
    print(
        f"[preflight] Preparing {config.experiment_label} selection and cached Level-2 data...",
        flush=True,
    )
    selected, _ = prepare_selection(config, root)
    if mode == "smoke":
        selected = selected[: config.smoke.total_tasks]
    output = _resolve(root, config.runtime.output_root) / ("smoke" if mode == "smoke" else "")
    output.mkdir(parents=True, exist_ok=True)
    existing = (
        list((output / "task_checkpoints").glob("*.json"))
        if (output / "task_checkpoints").exists()
        else []
    )
    if existing and not resume:
        raise FileExistsError(f"Existing incremental results found under {output}; use --resume")

    groups = _load_groups(config, root)
    benchmark_selections = _load_benchmark_selections(config, root, selected)
    public: dict[str, PublicTask] = {}
    private: dict[str, Any] = {}
    for dataset, selection in benchmark_selections.items():
        for task in selection.public_tasks:
            key = f"{dataset}::{task.task_id}"
            public[key] = task
            private[key] = selection.private_cases[task.task_id]

    invocations = JsonlStore(output / "llm_invocations.jsonl")
    events = JsonlStore(output / "events.jsonl")
    evaluations = JsonlStore(output / "evaluations.jsonl")
    errors = JsonlStore(output / "errors.jsonl")
    provider = _provider(config)
    client = LLMClient(
        provider,
        BudgetManager(BudgetConfig(max_spend_usd=0.0, allow_paid_calls=True), provider),
        invocations,
    )
    print(
        f"[preflight] Loaded {len(selected)} tasks; checking sandbox, official evaluators, "
        "and dependencies...",
        flush=True,
    )
    evaluators = {
        dataset: _evaluator(dataset, selection, config, evaluations, root)
        for dataset, selection in benchmark_selections.items()
    }
    for dataset, evaluator in evaluators.items():
        evaluator.check_sandbox()
        if hasattr(evaluator, "check_official_evaluator"):
            evaluator.check_official_evaluator()
        if hasattr(evaluator, "check_dependencies"):
            evaluator.check_dependencies(benchmark_selections[dataset].public_tasks)
    print(
        f"[preflight] Completed in {time.perf_counter() - preflight_started:.1f}s; "
        "starting task progress.",
        flush=True,
    )

    # Preload completed checkpoints so resume reflects real progress immediately
    # and does not make cached tasks look like a fresh run.
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in selected:
        checkpoint_path = _task_file(output, row["task_key"])
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("status") == "completed":
                results.append(checkpoint)
                continue
        pending.append(row)
    failures = 0
    started = time.perf_counter()
    bar = tqdm(
        pending,
        initial=len(results),
        total=len(selected),
        desc=f"Qwen3.8 {config.experiment_label} ({mode})",
        unit="task",
        dynamic_ncols=True,
    )
    for row in bar:
        try:
            result = _run_task(
                selected_row=row,
                group=groups[row["task_key"]],
                task=public[row["task_key"]],
                private_case=private[row["task_key"]],
                evaluator=evaluators[row["dataset"]],
                config=config,
                client=client,
                invocations=invocations,
                events=events,
                root=root,
                output=output,
                run_id=f"{config.runtime.run_id}-{mode}",
            )
            results.append(result)
        except Exception as exc:
            failures += 1
            errors.append(
                {
                    "record_type": f"qwen38_{config.experiment_label.lower()}_error",
                    "timestamp_utc": utc_now(),
                    "mode": mode,
                    "task_key": row["task_key"],
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        bar.set_postfix(
            completed=len(results),
            failures=failures,
            oracle=sum(bool(item.get("new_oracle_passed")) for item in results),
            teacher_tokens=sum(int(item.get("unique_teacher_tokens", 0)) for item in results),
        )
    bar.close()

    trajectories = _trajectory_rows(results)
    effects = conditional_effect_rows(results)
    tasks = _task_rows(results)
    _atomic_csv(output / "tables" / "trajectory_comparison.csv", trajectories)
    _atomic_csv(output / "tables" / "checkpoint_effects.csv", effects)
    _atomic_csv(output / "tables" / "task_oracle_comparison.csv", tasks)
    summary = _summary(results, effects)
    summary.update(
        {
            "schema_version": (
                "darc-qwen38-full20-summary-v1"
                if config.schema_version == "darc-qwen38-full20-v1"
                else "darc-qwen38-full-summary-v2"
            ),
            "mode": mode,
            "requested_tasks": len(selected),
            "failed_tasks": failures,
            "elapsed_seconds": time.perf_counter() - started,
            "selection_seed": config.selection.seed,
            "thinking_token_budget": config.teacher.thinking_token_budget,
            "max_output_tokens": config.teacher.max_output_tokens,
        }
    )
    _atomic_json(output / "summary.json", summary)
    report = _write_report(
        output,
        summary,
        mode,
        experiment_label=config.experiment_label,
    )
    manifest = {
        "schema_version": (
            "darc-qwen38-full20-run-v1"
            if config.schema_version == "darc-qwen38-full20-v1"
            else "darc-qwen38-full-run-v2"
        ),
        "status": "complete" if len(results) == len(selected) and failures == 0 else "incomplete",
        "mode": mode,
        "tasks": len(selected),
        "completed": len(results),
        "failures": failures,
        "student_model": config.student.model_id,
        "teacher_model": config.teacher.model_id,
        "student_endpoint": config.runtime.student_base_url,
        "teacher_endpoint": config.runtime.teacher_base_url,
        "sequences": list(SEQUENCES),
        "hidden_feedback_exposed": False,
        "report": str(report),
    }
    _atomic_json(output / "run_manifest.json", manifest)
    if manifest["status"] != "complete":
        raise RuntimeError(
            f"{config.experiment_label} {mode} incomplete: "
            f"{len(results)}/{len(selected)} tasks; resume after inspection"
        )
    return {"summary": summary, "report": str(report), "output": str(output)}

