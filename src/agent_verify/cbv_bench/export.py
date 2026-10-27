"""Versioned JSONL/Parquet exports for downstream research code."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from agent_verify.cbv_bench.config import CBVConfig
from agent_verify.cbv_bench.datasets import load_selection, project_path
from agent_verify.cbv_bench.labels import build_effect_records
from agent_verify.cbv_bench.model_metadata import build_model_pair_snapshot
from agent_verify.cbv_bench.validation import validate_cbv_data


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def export_cbv_dataset(config: CBVConfig, project_root: Path | None = None,
                       *, force: bool = False) -> Path:
    root = project_root or Path(__file__).resolve().parents[3]
    results = project_path(config.collection.results_dir, root)
    validation = validate_cbv_data(config, root)
    if not validation["valid"]:
        raise RuntimeError("Refusing full export: validation reports missing/invalid branches")
    fingerprint = str(validation["compatibility_fingerprint"])
    model_pair = build_model_pair_snapshot(config)
    safe_pair_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_pair.model_pair_id).strip("-")
    output = results / "exports" / (
        f"{safe_pair_id}_{fingerprint[:16]}_{config.fingerprint[:16]}_"
        f"level{config.collection.level}"
    )
    if output.exists() and not force:
        raise FileExistsError(f"Export already exists: {output}; pass --force only after review")
    output.mkdir(parents=True, exist_ok=True)
    allowed = {tuple(value.split("::", 1)) for value in validation["scope_task_keys"]}
    trajectories = [row for row in _read(results / "trajectories.jsonl")
                    if row.get("compatibility_fingerprint") == fingerprint
                    and (row.get("dataset"), row.get("task_id")) in allowed
                    and int(row.get("rollout_index", -1)) < config.collection.repeated_rollouts]
    transitions = [row for row in _read(results / "cache" / "transitions.jsonl")
                   if row.get("compatibility_fingerprint") == fingerprint
                   and (row.get("dataset"), row.get("task_id")) in allowed
                   and int(row.get("rollout_index", -1)) < config.collection.repeated_rollouts]
    states = _read(results / "checkpoint_states.jsonl")
    referenced_states = {value for row in trajectories for value in row.get("checkpoint_state_ids", [])}
    states = [row for row in states if row.get("state_id") in referenced_states]
    effects_path = build_effect_records(results, fingerprint, config.collection.level)
    effects = [
        row for row in _read(effects_path)
        if (row.get("dataset"), row.get("task_id")) in allowed
        and int(row.get("rollout_index", -1)) < config.collection.repeated_rollouts
    ]
    tasks = []
    for spec in config.datasets:
        if spec.enabled:
            selection = load_selection(config, spec, root)
            tasks.extend(record.model_dump(mode="json") for record in selection.task_records.values())
    tables = {
        "model_pairs": [model_pair.model_dump(mode="json")],
        "tasks": tasks,
        "trajectories": trajectories,
        "transitions": transitions,
        "states": states,
        "effects": effects,
    }
    for name, rows in tables.items():
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    if config.collection.export_parquet:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet export requested but pyarrow is unavailable; install the locked environment "
                "or set collection.export_parquet=false"
            ) from exc
        for name, rows in tables.items():
            if rows:
                pq.write_table(pa.Table.from_pylist(rows), output / f"{name}.parquet", compression="zstd")
    metadata = {
        "schema_version": "cbv-export-v2", "compatibility_fingerprint": fingerprint,
        "model_pair_id": model_pair.model_pair_id,
        "model_pair_fingerprint": model_pair.model_pair_fingerprint,
        "student_model": model_pair.student_model.model_dump(mode="json"),
        "teacher_model": model_pair.teacher_model.model_dump(mode="json"),
        "prompt_version": model_pair.prompt_version,
        "inference_config": model_pair.inference_config,
        "model_pair": model_pair.model_dump(mode="json"),
        "level": config.collection.level, "counts": {name: len(rows) for name, rows in tables.items()},
        "validation": validation, "parquet": config.collection.export_parquet,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    shutil.copy2(
        results / "validation" / fingerprint / config.fingerprint[:16] / "validation.md",
        output / "VALIDATION.md",
    )
    return output
