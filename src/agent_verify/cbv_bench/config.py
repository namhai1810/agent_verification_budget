"""Configuration isolated from the historical experiment configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from agent_verify.config import (
    BudgetConfig,
    EvaluationConfig,
    ModelsConfig,
    ProviderConfig,
    StrictModel,
    _load_raw_config,
)


class DatasetSourceConfig(StrictModel):
    name: Literal["humaneval", "livecodebench", "bigcodebench"]
    enabled: bool = True
    version: str
    revision: str = ""
    split: str = "test"
    config_name: str = ""
    data_path: Path | None = None
    sha256: str = ""
    selection_manifest: Path
    existing_results_dir: Path | None = None
    existing_results_dirs: list[Path] = Field(default_factory=list)
    task_count: int = Field(default=10, ge=1)
    task_offset: int = Field(default=0, ge=0)
    minimum_release_date: str = ""
    maximum_release_date: str = ""
    difficulties: list[str] = Field(default_factory=list)
    difficulty_weights: dict[str, float] = Field(default_factory=dict)
    official_evaluator_path: Path | None = None
    official_evaluator_commit: str = ""

    @property
    def legacy_result_dirs(self) -> list[Path]:
        values = list(self.existing_results_dirs)
        if self.existing_results_dir is not None:
            values.insert(0, self.existing_results_dir)
        return list(dict.fromkeys(values))


class SelectionConfig(StrictModel):
    seed: int = 42
    split_seed: int = 20261026
    train_fraction: float = Field(default=0.70, ge=0.0, le=1.0)
    validation_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    test_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    exploratory_task_files: list[Path] = Field(default_factory=list)

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> "SelectionConfig":
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1.0) > 1e-9:
            raise ValueError("train/validation/test fractions must sum to 1.0")
        return self


class CollectionConfig(StrictModel):
    name: str = "cbv_bench_v2"
    level: Literal[1, 2] = 2
    results_dir: Path = Path("results/cbv_bench_v2")
    rollout_seed: int = 20261026
    repeated_rollouts: int = Field(default=1, ge=1)
    batch_size: int = Field(default=1, ge=1)
    resume: bool = True
    retry_failed: bool = False
    reuse_legacy_level1: bool = True
    max_teacher_calls: int | None = Field(default=None, ge=1)
    max_teacher_tokens: int | None = Field(default=None, ge=1)
    teacher_prompt_token_safety_factor: float = Field(default=1.25, ge=1.0)
    export_parquet: bool = True
    progress: bool = True


class ModelIdentityConfig(StrictModel):
    """Stable model identity, intentionally separate from inference settings."""

    model_id: str = Field(min_length=1)
    family: str = ""
    revision: str = ""
    quantization: str = ""
    parameter_count_b: float | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelPairConfig(StrictModel):
    """Human-readable identity for one Student--Teacher condition."""

    pair_id: str = Field(min_length=1)
    student_model: ModelIdentityConfig
    teacher_model: ModelIdentityConfig
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CBVConfig(StrictModel):
    schema_version: Literal["cbv-bench-config-v2"] = "cbv-bench-config-v2"
    collection: CollectionConfig
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    datasets: list[DatasetSourceConfig]
    model_pair: ModelPairConfig
    models: ModelsConfig
    provider: ProviderConfig
    budget: BudgetConfig
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def unique_enabled_datasets(self) -> "CBVConfig":
        names = [item.name for item in self.datasets if item.enabled]
        if not names:
            raise ValueError("At least one dataset must be enabled")
        if len(names) != len(set(names)):
            raise ValueError("Enabled dataset names must be unique")
        for role in ("planner", "coder", "reviewer", "teacher"):
            if getattr(self.models, role).provider == "heuristic":
                raise ValueError(f"models.{role}.provider cannot be heuristic")
        student_ids = {
            getattr(self.models, role).model_id for role in ("planner", "coder", "reviewer")
        }
        if student_ids != {self.model_pair.student_model.model_id}:
            raise ValueError(
                "models.planner/coder/reviewer must match model_pair.student_model.model_id"
            )
        if self.models.teacher.model_id != self.model_pair.teacher_model.model_id:
            raise ValueError("models.teacher must match model_pair.teacher_model.model_id")
        return self

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_cbv_config(path: str | Path) -> CBVConfig:
    return CBVConfig.model_validate(_load_raw_config(Path(path)))
