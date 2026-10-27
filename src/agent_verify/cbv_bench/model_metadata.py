"""Resolve model-pair identity and inference provenance into dataset metadata."""

from __future__ import annotations

import hashlib

from agent_verify.cbv_bench.config import CBVConfig
from agent_verify.cbv_bench.schemas import ModelPairSnapshot
from agent_verify.logging import canonical_json


_ROLES = ("planner", "coder", "reviewer", "teacher")


def build_model_pair_snapshot(config: CBVConfig) -> ModelPairSnapshot:
    """Create the exact, secret-free model condition stored with every trajectory.

    ``api_key_env`` records only the environment-variable name. The secret value is never read or
    serialized. Keeping identity separate from inference settings lets research code group either
    by model pair or by the exact prompt/sampling condition.
    """
    prompt_version: dict[str, str] = {}
    inference_config: dict[str, dict] = {}
    role_model_ids: dict[str, str] = {}
    for role in _ROLES:
        model = getattr(config.models, role)
        role_model_ids[role] = model.model_id
        prompt_version[role] = model.prompt_version
        inference_config[role] = {
            "provider": model.provider,
            "model_id": model.model_id,
            "base_url": model.base_url or config.provider.base_url,
            "api_key_env": model.api_key_env or config.provider.api_key_env,
            "temperature": model.temperature,
            "top_p": model.top_p,
            "max_output_tokens": model.max_output_tokens,
            "seed": model.seed,
            "input_cost_per_million": model.input_cost_per_million,
            "output_cost_per_million": model.output_cost_per_million,
        }
    payload = {
        "schema_version": "cbv-model-pair-v1",
        "model_pair_id": config.model_pair.pair_id,
        "student_model": config.model_pair.student_model.model_dump(mode="json"),
        "teacher_model": config.model_pair.teacher_model.model_dump(mode="json"),
        "student_role_model_ids": {
            role: role_model_ids[role] for role in ("planner", "coder", "reviewer")
        },
        "prompt_version": prompt_version,
        "inference_config": inference_config,
        "runtime_config": {
            "response_format": config.provider.response_format,
            "timeout_seconds": config.provider.timeout_seconds,
            "max_retries": config.provider.max_retries,
        },
        "tags": config.model_pair.tags,
        "metadata": config.model_pair.metadata,
    }
    fingerprint = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return ModelPairSnapshot.model_validate({**payload, "model_pair_fingerprint": fingerprint})
