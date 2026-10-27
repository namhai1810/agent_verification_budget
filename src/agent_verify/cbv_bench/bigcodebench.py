"""BigCodeBench adapter kept separate from existing benchmark adapters/evaluators."""

from __future__ import annotations

import ast
from typing import Any

from agent_verify.cbv_bench.cache import sha256_json
from agent_verify.cbv_bench.config import DatasetSourceConfig
from agent_verify.schemas import BigCodeBenchPrivateCase, BigCodeBenchPublicTask


def load_bigcodebench_candidates(spec: DatasetSourceConfig) -> tuple[list, dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("BigCodeBench adapter requires the pinned datasets dependency") from exc
    kwargs: dict[str, Any] = {"split": spec.split}
    if spec.revision:
        kwargs["revision"] = spec.revision
    if spec.config_name:
        dataset = load_dataset("bigcode/bigcodebench", spec.config_name, **kwargs)
    else:
        dataset = load_dataset("bigcode/bigcodebench", **kwargs)
    required = {"task_id", "instruct_prompt", "test", "entry_point"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Pinned BigCodeBench schema is missing {sorted(missing)}")
    candidates = []
    public_rows = []
    for source in dataset:
        task_id = str(source["task_id"])
        raw_libraries = source.get("libs") or []
        if isinstance(raw_libraries, str):
            try:
                raw_libraries = ast.literal_eval(raw_libraries)
            except (SyntaxError, ValueError):
                raw_libraries = [raw_libraries]
        libraries = [str(item) for item in raw_libraries]
        public = BigCodeBenchPublicTask(
            task_id=task_id,
            prompt=str(source["instruct_prompt"]),
            entry_point=str(source["entry_point"]),
            libraries=libraries,
        )
        # canonical_solution is intentionally never copied into either public records or cache keys.
        private = BigCodeBenchPrivateCase(
            task_id=task_id,
            test=str(source["test"]),
            entry_point=str(source["entry_point"]),
            code_prompt=str(source.get("code_prompt") or ""),
            official_schema_version=spec.version,
        )
        metadata = {"difficulty": "unknown", "release_date": "", "libraries": libraries}
        candidates.append((public, private, metadata))
        public_rows.append(public.model_dump(mode="json"))
    return candidates, {
        "dataset_id": "bigcode/bigcodebench",
        "revision": spec.revision,
        "split": spec.split,
        "source_rows": len(candidates),
        "public_content_sha256": sha256_json(public_rows),
        "canonical_solution_loaded_into_public_data": False,
    }
