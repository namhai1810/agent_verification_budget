#!/usr/bin/env python3
"""Export validated CBV data while encoding empty mappings safely in Parquet.

JSONL remains the canonical lossless representation. PyArrow cannot serialize a
struct with zero child fields (for example ``provenance: {}``), so only the
Parquet projection maps empty dictionaries to null. Row counts are verified after
writing every Parquet table.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from agent_verify.cbv_bench.cli import _override, parser as cbv_parser
from agent_verify.cbv_bench.config import load_cbv_config
from agent_verify.cbv_bench.datasets import load_selection, project_path
from agent_verify.cbv_bench.labels import build_effect_records
from agent_verify.cbv_bench.model_metadata import build_model_pair_snapshot
from agent_verify.cbv_bench.validation import validate_cbv_data


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parquet_safe(value: Any) -> Any:
    if isinstance(value, dict):
        if not value:
            return None
        return {key: _parquet_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_parquet_safe(item) for item in value]
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def safe_export(config_path: str, cli_args: list[str], *, force: bool) -> Path:
    parsed = cbv_parser().parse_args(["--config", config_path, "export", *cli_args])
    config = _override(load_cbv_config(config_path), parsed)
    root = Path.cwd()
    results = project_path(config.collection.results_dir, root)
    validation = validate_cbv_data(config, root)
    if not validation["valid"]:
        raise RuntimeError("Refusing export: validation reports missing/invalid branches")

    fingerprint = str(validation["compatibility_fingerprint"])
    model_pair = build_model_pair_snapshot(config)
    safe_pair_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_pair.model_pair_id).strip("-")
    output = results / "exports" / (
        f"{safe_pair_id}_{fingerprint[:16]}_{config.fingerprint[:16]}_"
        f"level{config.collection.level}"
    )
    if output.exists() and not force:
        raise FileExistsError(f"Export already exists: {output}; pass --force after review")
    output.mkdir(parents=True, exist_ok=True)

    allowed = {tuple(value.split("::", 1)) for value in validation["scope_task_keys"]}
    trajectories = [
        row
        for row in _read(results / "trajectories.jsonl")
        if row.get("compatibility_fingerprint") == fingerprint
        and (row.get("dataset"), row.get("task_id")) in allowed
        and int(row.get("rollout_index", -1)) < config.collection.repeated_rollouts
    ]
    transitions = [
        row
        for row in _read(results / "cache" / "transitions.jsonl")
        if row.get("compatibility_fingerprint") == fingerprint
        and (row.get("dataset"), row.get("task_id")) in allowed
        and int(row.get("rollout_index", -1)) < config.collection.repeated_rollouts
    ]
    states = _read(results / "checkpoint_states.jsonl")
    referenced_states = {
        value for row in trajectories for value in row.get("checkpoint_state_ids", [])
    }
    states = [row for row in states if row.get("state_id") in referenced_states]
    effects_path = build_effect_records(results, fingerprint, config.collection.level)
    effects = [
        row
        for row in _read(effects_path)
        if (row.get("dataset"), row.get("task_id")) in allowed
        and int(row.get("rollout_index", -1)) < config.collection.repeated_rollouts
    ]
    tasks: list[dict[str, Any]] = []
    for spec in config.datasets:
        if spec.enabled:
            selection = load_selection(config, spec, root)
            tasks.extend(
                record.model_dump(mode="json")
                for record in selection.task_records.values()
            )
    tables = {
        "model_pairs": [model_pair.model_dump(mode="json")],
        "tasks": tasks,
        "trajectories": trajectories,
        "transitions": transitions,
        "states": states,
        "effects": effects,
    }

    parquet_counts: dict[str, int] = {}
    for name, rows in tables.items():
        _write_jsonl(output / f"{name}.jsonl", rows)
        if not rows:
            parquet_counts[name] = 0
            continue
        safe_rows = [_parquet_safe(row) for row in rows]
        parquet_path = output / f"{name}.parquet"
        pq.write_table(
            pa.Table.from_pylist(safe_rows), parquet_path, compression="zstd"
        )
        written = pq.read_metadata(parquet_path).num_rows
        if written != len(rows):
            raise RuntimeError(
                f"Parquet row-count mismatch for {name}: expected {len(rows)}, got {written}"
            )
        parquet_counts[name] = written

    metadata = {
        "schema_version": "cbv-export-v2",
        "compatibility_fingerprint": fingerprint,
        "model_pair_id": model_pair.model_pair_id,
        "model_pair_fingerprint": model_pair.model_pair_fingerprint,
        "student_model": model_pair.student_model.model_dump(mode="json"),
        "teacher_model": model_pair.teacher_model.model_dump(mode="json"),
        "prompt_version": model_pair.prompt_version,
        "inference_config": model_pair.inference_config,
        "model_pair": model_pair.model_dump(mode="json"),
        "level": config.collection.level,
        "counts": {name: len(rows) for name, rows in tables.items()},
        "parquet": True,
        "parquet_counts_verified": parquet_counts,
        "parquet_empty_mapping_policy": (
            "JSONL preserves {}; Parquet encodes empty mappings as null because Arrow "
            "does not support zero-child structs"
        ),
        "validation": validation,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    shutil.copy2(
        results
        / "validation"
        / fingerprint
        / config.fingerprint[:16]
        / "validation.md",
        output / "VALIDATION.md",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--level", type=int, choices=(1, 2), required=True)
    parser.add_argument("--repeated-rollouts", type=int, default=1)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cli_args = [
        "--dataset",
        args.dataset,
        "--selection-manifest",
        args.selection_manifest,
        "--level",
        str(args.level),
        "--repeated-rollouts",
        str(args.repeated_rollouts),
        "--results-dir",
        args.results_dir,
    ]
    print(safe_export(args.config, cli_args, force=args.force))


if __name__ == "__main__":
    main()
