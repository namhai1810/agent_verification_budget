"""Leakage-safe decision-time examples and frozen semantic embedding cache for DARC."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from agent_verify.analysis.cbv_v01_experiment import (
    ROLES,
    SINGLE_SEQUENCES,
    TaskTrajectories,
    _artifact,
    _domain_metadata,
    _passed,
)
from agent_verify.analysis.joint_value_cost_data import VALUE_TO_INDEX, value_label
from agent_verify.analysis.phase2_data import build_observable_features

ROLE_DESCRIPTIONS = {
    1: "Planner: produces an algorithm, assumptions, edge cases, and complexity guidance for the coder.",
    2: "Coder: converts the task and plan into executable source code consumed by review and evaluation.",
    3: "Reviewer: inspects the implementation and may return corrected final code for execution.",
}

DOWNSTREAM_DEPENDENCY_TEMPLATES = {
    1: (
        "Coder depends directly on the current plan to choose the algorithm and implementation details.",
        "Reviewer depends indirectly on the plan through the code produced by the coder.",
        "Final evaluator depends indirectly on whether the plan leads to correct executable code.",
    ),
    2: (
        "Reviewer depends directly on the current code to identify defects and produce reviewed code.",
        "Final evaluator executes code descended from the current implementation.",
    ),
    3: (
        "Final evaluator executes the reviewed code produced at this checkpoint.",
    ),
}

FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "y_skip",
        "y_verify",
        "delta",
        "value_label",
        "passed",
        "teacher_total_tokens",
        "teacher_input_tokens",
        "teacher_output_tokens",
        "actual_cost",
        "future_artifact",
        "future_outcome",
        "oracle",
    }
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DARCExample:
    """One single-intervention state; labels are deliberately separate from inputs."""

    group_key: str
    task_id: str
    dataset: str
    difficulty: str
    checkpoint: int
    task_text: str
    artifact_text: str
    history_text: str
    current_role_text: str
    downstream_role_texts: tuple[str, ...]
    metadata: tuple[float, ...]
    metadata_names: tuple[str, ...]
    y_skip: int
    y_verify: int
    delta: int
    value_label: int
    teacher_total_tokens: int

    @property
    def state_key(self) -> str:
        return f"{self.group_key}::cp{self.checkpoint}"

    @property
    def state_fingerprint(self) -> str:
        observable = {
            "task": self.task_text,
            "artifact": self.artifact_text,
            "history": self.history_text,
            "role": self.current_role_text,
            "downstream": self.downstream_role_texts,
            "metadata": self.metadata,
            "metadata_names": self.metadata_names,
        }
        return _sha256_text(_stable_json(observable))


@dataclass(frozen=True)
class EncodedDARC:
    state_keys: tuple[str, ...]
    task: np.ndarray
    artifact: np.ndarray
    history: np.ndarray
    current_role: np.ndarray
    downstream: np.ndarray
    downstream_mask: np.ndarray
    metadata: np.ndarray
    checkpoint: np.ndarray

    def subset(self, indices: np.ndarray) -> EncodedDARC:
        return EncodedDARC(
            tuple(self.state_keys[int(i)] for i in indices),
            self.task[indices],
            self.artifact[indices],
            self.history[indices],
            self.current_role[indices],
            self.downstream[indices],
            self.downstream_mask[indices],
            self.metadata[indices],
            self.checkpoint[indices],
        )


def downstream_dependency_texts(
    *,
    checkpoint: int,
    task_prompt: str,
    entry_point: str,
    current_artifact: dict[str, Any],
) -> tuple[str, ...]:
    """Build observable, task-conditioned dependency views for semantic attention.

    The relation template supplies workflow structure, while the public task and
    current artifact make keys differ across tasks that share a checkpoint. No
    downstream artifact or outcome is included.
    """

    task_interface = "\n".join(
        (
            f"Entry point: {entry_point}",
            f"Public task requirements:\n{task_prompt}",
            f"Observable current artifact:\n{_stable_json(current_artifact)}",
        )
    )
    return tuple(
        "\n".join(
            (
                f"Dependency relation: {relation}",
                "Question: how can defects in this current artifact propagate to this downstream consumer?",
                task_interface,
            )
        )
        for relation in DOWNSTREAM_DEPENDENCY_TEMPLATES[checkpoint]
    )


def build_darc_examples(groups: dict[str, TaskTrajectories]) -> list[DARCExample]:
    """Materialize only information observable on the SSS prefix at CP1/CP2/CP3."""

    examples: list[DARCExample] = []
    schema: tuple[str, ...] | None = None
    for key in sorted(groups):
        group = groups[key]
        baseline = group.rows["SSS"]
        y_skip = _passed(baseline)
        for checkpoint, sequence in SINGLE_SEQUENCES.items():
            verified = group.rows[sequence]
            current = _artifact(baseline, checkpoint)
            preceding = [_artifact(baseline, prior) for prior in range(1, checkpoint)]
            # Reuse the established pre-treatment metadata contract. The returned text
            # is not used because DARC keeps semantic fields separately for ablations.
            _text, metadata, names = build_observable_features(
                prompt=group.task.prompt,
                entry_point=group.task.entry_point,
                checkpoint=checkpoint,
                role=ROLES[checkpoint],
                current_artifact=current,
                preceding_artifacts=preceding,
                local_risk=0.0,
            )
            domain_values, domain_names = _domain_metadata(group.task)
            names = (*names, *domain_names)
            metadata = (*metadata, *domain_values)
            if schema is None:
                schema = names
            elif schema != names:
                raise ValueError("DARC metadata schema changed across frozen examples")
            forbidden = FORBIDDEN_INPUT_FIELDS.intersection(names)
            if forbidden:
                raise ValueError(f"Post-treatment DARC metadata: {sorted(forbidden)}")
            task_text = "\n".join(
                (
                    f"Dataset: {group.task.dataset}",
                    f"Difficulty: {group.task.difficulty}",
                    f"Entry point: {group.task.entry_point}",
                    f"Task:\n{group.task.prompt}",
                )
            )
            artifact_text = f"Current {ROLES[checkpoint]} artifact:\n{_stable_json(current)}"
            history_text = (
                "No preceding artifacts are available at the first checkpoint."
                if not preceding
                else "Observed preceding artifacts:\n" + _stable_json(preceding)
            )
            input_tokens = int(verified.get("teacher_input_tokens", 0) or 0)
            output_tokens = int(verified.get("teacher_output_tokens", 0) or 0)
            total_tokens = int(
                verified.get("teacher_total_tokens", input_tokens + output_tokens) or 0
            )
            if total_tokens != input_tokens + output_tokens:
                raise ValueError(f"Teacher token mismatch for {key} {sequence}")
            y_verify = _passed(verified)
            examples.append(
                DARCExample(
                    group_key=key,
                    task_id=group.task.task_id,
                    dataset=group.task.dataset,
                    difficulty=group.task.difficulty,
                    checkpoint=checkpoint,
                    task_text=task_text,
                    artifact_text=artifact_text,
                    history_text=history_text,
                    current_role_text=ROLE_DESCRIPTIONS[checkpoint],
                    downstream_role_texts=downstream_dependency_texts(
                        checkpoint=checkpoint,
                        task_prompt=group.task.prompt,
                        entry_point=group.task.entry_point,
                        current_artifact=current,
                    ),
                    metadata=metadata,
                    metadata_names=names,
                    y_skip=y_skip,
                    y_verify=y_verify,
                    delta=y_verify - y_skip,
                    value_label=value_label(y_skip, y_verify),
                    teacher_total_tokens=total_tokens,
                )
            )
    validate_darc_examples(examples)
    return examples


def validate_darc_examples(examples: list[DARCExample]) -> None:
    if not examples:
        raise ValueError("No DARC examples")
    seen: set[str] = set()
    grouped: dict[str, list[int]] = {}
    schema = examples[0].metadata_names
    for row in examples:
        if row.state_key in seen:
            raise ValueError(f"Duplicate DARC state: {row.state_key}")
        seen.add(row.state_key)
        grouped.setdefault(row.group_key, []).append(row.checkpoint)
        if row.metadata_names != schema:
            raise ValueError("Inconsistent DARC metadata schema")
        if FORBIDDEN_INPUT_FIELDS.intersection(row.metadata_names):
            raise ValueError(f"Forbidden field in {row.state_key}")
        if row.delta != row.y_verify - row.y_skip:
            raise ValueError(f"Invalid paired outcome in {row.state_key}")
        if row.value_label != value_label(row.y_skip, row.y_verify):
            raise ValueError(f"Invalid value class in {row.state_key}")
        if not row.downstream_role_texts:
            raise ValueError(f"Missing downstream dependency in {row.state_key}")
    incomplete = {key: cps for key, cps in grouped.items() if sorted(cps) != [1, 2, 3]}
    if incomplete:
        raise ValueError(f"Incomplete DARC task groups: {list(incomplete)[:5]}")


def dataset_fingerprint(examples: Iterable[DARCExample]) -> str:
    rows = [
        (row.state_key, row.state_fingerprint, row.y_skip, row.y_verify, row.teacher_total_tokens)
        for row in examples
    ]
    return _sha256_text(_stable_json(sorted(rows)))


def feature_fingerprint(examples: Iterable[DARCExample]) -> str:
    """Hash observable states only; labels/costs cannot affect cache identity."""

    rows = [(row.state_key, row.state_fingerprint) for row in examples]
    return _sha256_text(_stable_json(sorted(rows)))


def chunk_spans(
    token_count: int,
    *,
    capacity: int,
    stride: int,
    max_chunks: int,
) -> list[tuple[int, int]]:
    """Deterministic head-to-tail chunk plan used by the frozen encoder."""

    if token_count < 0 or capacity <= 0 or stride <= 0 or max_chunks <= 0:
        raise ValueError("Invalid semantic chunking parameters")
    if token_count <= capacity:
        return [(0, token_count)]
    last_start = token_count - capacity
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    if len(starts) > max_chunks:
        selected = np.linspace(0, len(starts) - 1, num=max_chunks, dtype=int)
        starts = sorted({starts[int(index)] for index in selected})
    return [(start, min(start + capacity, token_count)) for start in starts]


def _covered_tokens(spans: list[tuple[int, int]]) -> int:
    total = 0
    end = 0
    for start, stop in sorted(spans):
        if stop <= end:
            continue
        total += stop - max(start, end)
        end = stop
    return total


class FrozenSemanticEncoder:
    """A frozen Hugging Face encoder with deterministic masked mean pooling."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.model_id = str(config["model_id"])
        self.revision = str(config.get("revision", "main"))
        self.device_name = str(config.get("device", "auto"))
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: Any = None
        self._token_cache: dict[str, list[int]] = {}

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError(
                "DARC needs torch and transformers. Install with: pip install -e '.[darc]'"
            ) from error
        selected = self.device_name
        if selected == "auto":
            selected = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(selected)
        self._load_tokenizer()
        kwargs = {
            "revision": self.revision,
            "local_files_only": bool(self.config.get("local_files_only", False)),
            "trust_remote_code": False,
        }
        self.model = AutoModel.from_pretrained(self.model_id, **kwargs).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _load_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "DARC needs transformers. Install with: pip install -e '.[darc]'"
            ) from error
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            local_files_only=bool(self.config.get("local_files_only", False)),
            trust_remote_code=False,
        )

    def _token_ids(self, text: str) -> list[int]:
        self._load_tokenizer()
        if text not in self._token_cache:
            self._token_cache[text] = list(
                self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            )
        return self._token_cache[text]

    def _plan(self, text: str) -> tuple[list[list[int]], int, int]:
        ids = self._token_ids(text)
        max_length = int(self.config.get("max_length", 512))
        special = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        capacity = max_length - special
        spans = chunk_spans(
            len(ids),
            capacity=capacity,
            stride=int(self.config.get("chunk_stride", capacity)),
            max_chunks=int(self.config.get("max_chunks_per_text", 4)),
        )
        chunks = [self.tokenizer.build_inputs_with_special_tokens(ids[start:stop]) for start, stop in spans]
        return chunks, len(ids), _covered_tokens(spans)

    def coverage(self, texts: list[str]) -> dict[str, float | int]:
        lengths: list[int] = []
        retained: list[int] = []
        chunks: list[int] = []
        for text in texts:
            plan, length, covered = self._plan(text)
            lengths.append(length)
            retained.append(covered)
            chunks.append(len(plan))
        ratios = [covered / max(length, 1) for length, covered in zip(lengths, retained, strict=True)]
        max_length = int(self.config.get("max_length", 512))
        capacity = max_length - int(self.tokenizer.num_special_tokens_to_add(pair=False))
        return {
            "texts": len(texts),
            "mean_tokens": float(np.mean(lengths)) if lengths else 0.0,
            "p95_tokens": float(np.percentile(lengths, 95)) if lengths else 0.0,
            "max_tokens": int(max(lengths, default=0)),
            "over_single_window": int(sum(length > capacity for length in lengths)),
            "over_single_window_rate": float(np.mean(np.asarray(lengths) > capacity)) if lengths else 0.0,
            "mean_retained_ratio": float(np.mean(ratios)) if ratios else 1.0,
            "min_retained_ratio": float(min(ratios, default=1.0)),
            "truncated_by_chunk_limit": int(
                sum(covered < length for length, covered in zip(lengths, retained, strict=True))
            ),
            "encoded_chunks": int(sum(chunks)),
        }

    def encode(self, texts: list[str], *, description: str) -> tuple[np.ndarray, dict[str, Any]]:
        self._load()
        import torch

        chunk_ids: list[list[int]] = []
        owners: list[int] = []
        lengths: list[int] = []
        retained: list[int] = []
        for owner, text in enumerate(texts):
            chunks, length, covered = self._plan(text)
            chunk_ids.extend(chunks)
            owners.extend([owner] * len(chunks))
            lengths.append(length)
            retained.append(covered)
        chunk_outputs: list[np.ndarray] = []
        batch_size = int(self.config.get("batch_size", 32))
        for start in tqdm(
            range(0, len(chunk_ids), batch_size),
            desc=description,
            unit="batch",
            dynamic_ncols=True,
        ):
            encoded = self.tokenizer.pad(
                {"input_ids": chunk_ids[start : start + batch_size]},
                padding=True,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            chunk_outputs.append(pooled.float().cpu().numpy())
        chunk_matrix = np.vstack(chunk_outputs).astype(np.float32)
        output = np.zeros((len(texts), chunk_matrix.shape[1]), dtype=np.float32)
        counts = np.zeros(len(texts), dtype=np.float32)
        for row, owner in zip(chunk_matrix, owners, strict=True):
            output[owner] += row
            counts[owner] += 1.0
        output /= np.maximum(counts[:, None], 1.0)
        if bool(self.config.get("normalize", True)):
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            output /= np.maximum(norms, 1e-12)
        ratios = [covered / max(length, 1) for length, covered in zip(lengths, retained, strict=True)]
        return output, {
            "unique_texts": len(texts),
            "encoded_chunks": len(chunk_ids),
            "mean_retained_ratio": float(np.mean(ratios)) if ratios else 1.0,
            "min_retained_ratio": float(min(ratios, default=1.0)),
        }


def semantic_coverage(
    examples: list[DARCExample], encoder: FrozenSemanticEncoder
) -> dict[str, dict[str, float | int]]:
    field_texts = {
        "task": [row.task_text for row in examples],
        "artifact": [row.artifact_text for row in examples],
        "history": [row.history_text for row in examples],
        "current_role": [row.current_role_text for row in examples],
        "downstream": [text for row in examples for text in row.downstream_role_texts],
    }
    return {field: encoder.coverage(texts) for field, texts in field_texts.items()}


def encode_or_load(
    examples: list[DARCExample],
    *,
    split: str,
    config: dict[str, Any],
    cache_dir: Path,
) -> tuple[EncodedDARC, dict[str, Any]]:
    """Encode unique observable strings and persist an auditable split-local cache."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = feature_fingerprint(examples)
    encoder_signature = _sha256_text(
        _stable_json(
            {
                "model_id": config["model_id"],
                "revision": config.get("revision", "main"),
                "max_length": config.get("max_length", 512),
                "chunk_stride": config.get("chunk_stride", 480),
                "max_chunks_per_text": config.get("max_chunks_per_text", 4),
                "normalize": config.get("normalize", True),
            }
        )
    )[:16]
    stem = f"{split}_{fingerprint[:16]}_{encoder_signature}"
    array_path = cache_dir / f"{stem}.npz"
    manifest_path = cache_dir / f"{stem}.json"
    expected_keys = tuple(row.state_key for row in examples)
    if array_path.exists() and manifest_path.exists():
        load_started = time.perf_counter()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "darc-embedding-cache-v2":
            raise ValueError(f"Obsolete DARC embedding cache: {manifest_path}")
        with np.load(array_path, allow_pickle=False) as payload:
            state_keys = tuple(str(value) for value in payload["state_keys"].tolist())
            if state_keys != expected_keys or manifest.get("feature_fingerprint") != fingerprint:
                raise ValueError(f"Embedding cache identity mismatch: {array_path}")
            encoded = EncodedDARC(
                state_keys,
                payload["task"],
                payload["artifact"],
                payload["history"],
                payload["current_role"],
                payload["downstream"],
                payload["downstream_mask"].astype(bool),
                payload["metadata"],
                payload["checkpoint"].astype(np.int64),
            )
        manifest["cache_hit"] = True
        manifest["cache_load_seconds"] = time.perf_counter() - load_started
        return encoded, manifest

    started = time.perf_counter()
    unique_texts = sorted(
        {
            text
            for row in examples
            for text in (
                row.task_text,
                row.artifact_text,
                row.history_text,
                row.current_role_text,
                *row.downstream_role_texts,
            )
        }
    )
    semantic = FrozenSemanticEncoder(config)
    coverage = semantic_coverage(examples, semantic)
    matrix, encoding_stats = semantic.encode(
        unique_texts, description=f"DARC {split} frozen embeddings"
    )
    lookup = {text: matrix[index] for index, text in enumerate(unique_texts)}
    max_downstream = max(len(row.downstream_role_texts) for row in examples)
    dimension = matrix.shape[1]
    downstream = np.zeros((len(examples), max_downstream, dimension), dtype=np.float32)
    downstream_mask = np.zeros((len(examples), max_downstream), dtype=bool)
    for index, row in enumerate(examples):
        for slot, text in enumerate(row.downstream_role_texts):
            downstream[index, slot] = lookup[text]
            downstream_mask[index, slot] = True
    encoded = EncodedDARC(
        expected_keys,
        np.vstack([lookup[row.task_text] for row in examples]),
        np.vstack([lookup[row.artifact_text] for row in examples]),
        np.vstack([lookup[row.history_text] for row in examples]),
        np.vstack([lookup[row.current_role_text] for row in examples]),
        downstream,
        downstream_mask,
        np.asarray([row.metadata for row in examples], dtype=np.float32),
        np.asarray([row.checkpoint - 1 for row in examples], dtype=np.int64),
    )
    np.savez_compressed(
        array_path,
        state_keys=np.asarray(expected_keys),
        task=encoded.task,
        artifact=encoded.artifact,
        history=encoded.history,
        current_role=encoded.current_role,
        downstream=encoded.downstream,
        downstream_mask=encoded.downstream_mask,
        metadata=encoded.metadata,
        checkpoint=encoded.checkpoint,
    )
    elapsed = time.perf_counter() - started
    manifest = {
        "schema_version": "darc-embedding-cache-v2",
        "split": split,
        "feature_fingerprint": fingerprint,
        "encoder_signature": encoder_signature,
        "model_id": config["model_id"],
        "revision": config.get("revision", "main"),
        "examples": len(examples),
        "unique_texts": len(unique_texts),
        "embedding_dimension": dimension,
        "build_seconds": elapsed,
        "cache_load_seconds": 0.0,
        "cache_hit": False,
        "chunking": {
            "max_length": int(config.get("max_length", 512)),
            "stride": int(config.get("chunk_stride", 480)),
            "max_chunks_per_text": int(config.get("max_chunks_per_text", 4)),
            **encoding_stats,
        },
        "coverage_by_field": coverage,
        "input_contract": "task/current artifact/preceding artifacts/task-conditioned role dependencies only",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return encoded, manifest


def class_counts(examples: list[DARCExample]) -> dict[str, int]:
    return {
        name: sum(row.value_label == index for row in examples)
        for name, index in VALUE_TO_INDEX.items()
    }
