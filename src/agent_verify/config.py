from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictModel):
    name: Literal["openai_humaneval", "livecodebench_code_generation_lite"] = "openai_humaneval"
    source: Literal["bundled", "huggingface"] = "bundled"
    path: Path = Path("data/HumanEval.jsonl.gz")
    upstream_commit: str = ""
    sha256: str = ""
    config_name: str = ""
    split: str = "test"
    revision: str = ""
    minimum_contest_date: str = ""
    selection_seed: int = 42
    selected_tasks_path: Path = Path("results/livecodebench/selected_tasks.json")
    dataset_manifest_path: Path = Path("results/livecodebench/dataset_manifest.json")
    official_evaluator_path: Path = Path(".vendor/LiveCodeBench")
    official_evaluator_commit: str = ""


class ExperimentConfig(StrictModel):
    name: str = "h1_phase1"
    seed: int = 20261024
    task_count: int = Field(default=4, ge=1, le=1000)
    task_offset: int = Field(default=0, ge=0, le=999)
    task_selection: Literal["ordered", "shuffled"] = "shuffled"
    repeated_pairs: int = Field(default=1, ge=1)
    results_dir: Path = Path("results")
    resume: bool = True
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    @model_validator(mode="after")
    def valid_slice(self) -> ExperimentConfig:
        # The pinned LiveCodeBench v6/date filter currently contains 175 eligible
        # tasks. Keep the bound explicit so an offset-100 extension can collect
        # the remaining 75 tasks without silently changing the frozen dataset.
        upper = 164 if self.dataset.name == "openai_humaneval" else 175
        if self.task_offset + self.task_count > upper:
            raise ValueError(f"task_offset + task_count must be <= {upper} for {self.dataset.name}")
        return self


class ModelConfig(StrictModel):
    provider: Literal["mock", "heuristic", "openai_compatible"]
    model_id: str = Field(min_length=1)
    # Optional per-model routing. This lets child and teacher models use
    # different OpenAI-compatible servers while preserving the global default.
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    max_output_tokens: int = Field(gt=0)
    seed: int
    prompt_version: str = Field(min_length=1)
    input_cost_per_million: float = Field(default=0.0, ge=0.0)
    output_cost_per_million: float = Field(default=0.0, ge=0.0)


class ModelsConfig(StrictModel):
    planner: ModelConfig
    coder: ModelConfig
    reviewer: ModelConfig
    teacher: ModelConfig
    critic: ModelConfig


class ProviderConfig(StrictModel):
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    response_format: Literal["json_schema", "json_object"] = "json_schema"


class BudgetConfig(StrictModel):
    max_spend_usd: float = Field(default=0.0, ge=0.0)
    allow_paid_calls: bool = False


class EvaluationConfig(StrictModel):
    sandbox: Literal["landlock_seccomp"] = "landlock_seccomp"
    timeout_seconds: int = Field(default=8, ge=1)
    memory_mb: int = Field(default=512, ge=64)
    max_processes: int = Field(default=64, ge=1)


class AppConfig(StrictModel):
    experiment: ExperimentConfig
    models: ModelsConfig
    provider: ProviderConfig
    budget: BudgetConfig
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def valid_role_providers(self) -> AppConfig:
        for role in ("planner", "coder", "reviewer", "teacher"):
            if getattr(self.models, role).provider == "heuristic":
                raise ValueError(f"models.{role}.provider cannot be heuristic")
        return self

    @property
    def fingerprint(self) -> str:
        payload_dict = self.model_dump(mode="json")
        dataset = self.experiment.dataset
        if dataset.name == "openai_humaneval":
            # Keep historical HumanEval run IDs stable after adding Phase-3-only fields.
            payload_dict["experiment"]["dataset"] = {
                "name": dataset.name,
                "source": dataset.source,
                "path": str(dataset.path),
                "upstream_commit": dataset.upstream_commit,
                "sha256": dataset.sha256,
            }
        payload = json.dumps(payload_dict, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(config_path: Path, seen: set[Path] | None = None) -> dict:
    seen = set() if seen is None else seen
    config_path = config_path.resolve()
    if config_path in seen:
        raise ValueError(f"Config inheritance cycle at {config_path}")
    seen.add(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_name = raw.pop("_base_", None)
    if base_name:
        base_raw = _load_raw_config((config_path.parent / base_name).resolve(), seen)
        raw = _deep_merge(base_raw, raw)
    seen.remove(config_path)
    return raw


def load_config(path: str | Path) -> AppConfig:
    return AppConfig.model_validate(_load_raw_config(Path(path)))
