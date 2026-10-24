from __future__ import annotations

import pytest

from agent_verify.budget import BudgetError, BudgetManager
from agent_verify.config import load_config
from agent_verify.llm import LLMClient
from agent_verify.logging import JsonlStore
from agent_verify.schemas import InvocationContext, Message, PlanArtifact


def test_default_and_inherited_config_are_valid(default_config, monkeypatch, tmp_path):
    assert default_config.models.teacher.model_id == "mock-parent-teacher-v1"
    assert default_config.experiment.task_count == 4

    real = load_config("configs/real_example.yaml")
    assert real.models.planner.provider == "openai_compatible"
    assert real.models.teacher.model_id == "REPLACE_STRONG_PARENT_MODEL_ID"
    assert real.evaluation.sandbox == "landlock_seccomp"

    local = load_config("configs/vllm_gpu2.yaml")
    assert local.models.planner.model_id == "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
    assert local.models.planner.base_url == "http://127.0.0.1:8011/v1"
    assert local.models.teacher.model_id == "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
    assert local.models.teacher.base_url == "http://127.0.0.1:8012/v1"
    assert local.models.critic.provider == "heuristic"


def test_per_model_credential_override_is_used(default_config, monkeypatch):
    model = default_config.models.planner.model_copy(
        update={
            "provider": "openai_compatible",
            "api_key_env": "CHILD_VLLM_API_KEY",
        }
    )
    budget = default_config.budget.model_copy(update={"allow_paid_calls": True})
    manager = BudgetManager(budget, default_config.provider)
    monkeypatch.setenv("AGENT_VERIFY_ALLOW_SPEND", "YES")
    monkeypatch.setenv("CHILD_VLLM_API_KEY", "local-test-key")
    monkeypatch.delenv(default_config.provider.api_key_env, raising=False)

    assert manager.authorize(model, "prompt") == 0.0


def test_paid_call_requires_both_explicit_authorizations(default_config, monkeypatch):
    model = default_config.models.planner.model_copy(update={"provider": "openai_compatible"})
    manager = BudgetManager(default_config.budget, default_config.provider)
    monkeypatch.delenv("AGENT_VERIFY_ALLOW_SPEND", raising=False)
    with pytest.raises(BudgetError, match="blocked"):
        manager.authorize(model, "prompt")


def test_budget_pre_call_upper_bound(default_config, monkeypatch):
    model = default_config.models.planner.model_copy(
        update={
            "provider": "openai_compatible",
            "input_cost_per_million": 10.0,
            "output_cost_per_million": 10.0,
        }
    )
    budget = default_config.budget.model_copy(update={"allow_paid_calls": True, "max_spend_usd": 0.000001})
    manager = BudgetManager(budget, default_config.provider)
    monkeypatch.setenv("AGENT_VERIFY_ALLOW_SPEND", "YES")
    monkeypatch.setenv(default_config.provider.api_key_env, "test-only")
    with pytest.raises(BudgetError, match="exceeds"):
        manager.authorize(model, "large prompt" * 100)


def test_each_retry_is_costed_and_failed_attempt_is_preserved(default_config, monkeypatch, tmp_path):
    model = default_config.models.planner.model_copy(
        update={
            "provider": "openai_compatible",
            "input_cost_per_million": 1.0,
            "output_cost_per_million": 1.0,
        }
    )
    budget_config = default_config.budget.model_copy(update={"allow_paid_calls": True, "max_spend_usd": 1.0})
    manager = BudgetManager(budget_config, default_config.provider)
    client = LLMClient(
        default_config.provider,
        manager,
        JsonlStore(tmp_path / "invocations.jsonl"),
    )
    responses = iter(
        [
            ("not-json", 10, 5),
            (
                PlanArtifact(
                    summary="Valid plan",
                    algorithm=["Do the work"],
                    edge_cases=[],
                    complexity="O(1)",
                ).model_dump_json(),
                20,
                10,
            ),
        ]
    )
    monkeypatch.setattr(client, "_call_api", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setenv("AGENT_VERIFY_ALLOW_SPEND", "YES")
    monkeypatch.setenv(default_config.provider.api_key_env, "test-only")

    result = client.invoke(
        model=model,
        messages=[Message(role="user", content="test")],
        output_schema=PlanArtifact,
        context=InvocationContext(
            task_id="Synthetic/0",
            checkpoint=1,
            role="planner",
            branch="test",
            run_id="retry-test",
            seed=1,
        ),
        mock_factory=lambda: PlanArtifact(
            summary="unused",
            algorithm=["unused"],
            edge_cases=[],
            complexity="unused",
        ),
    )
    assert result.record.attempt_count == 2
    assert [attempt["status"] for attempt in result.record.attempts] == ["failed", "success"]
    assert result.record.input_tokens == 30
    assert result.record.output_tokens == 15
    assert result.record.cost_usd == pytest.approx(45 / 1_000_000)
    assert manager.spent_usd == pytest.approx(45 / 1_000_000)
