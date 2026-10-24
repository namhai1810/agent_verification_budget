from __future__ import annotations

import json

from agent_verify.agents.fixed import CoderAgent, PlannerAgent, ReviewerAgent
from agent_verify.budget import BudgetManager
from agent_verify.dataset import load_humaneval, read_raw_humaneval
from agent_verify.llm import LLMClient
from agent_verify.logging import JsonlStore
from agent_verify.workflow import FixedWorkflow


def _workflow(default_config, tmp_path):
    invocation_log = JsonlStore(tmp_path / "invocations.jsonl")
    client = LLMClient(
        default_config.provider,
        BudgetManager(default_config.budget, default_config.provider),
        invocation_log,
    )
    workflow = FixedWorkflow(
        PlannerAgent(client, default_config.models.planner),
        CoderAgent(client, default_config.models.coder),
        ReviewerAgent(client, default_config.models.reviewer),
        default_config.models,
        JsonlStore(tmp_path / "checkpoints.jsonl"),
    )
    return workflow, invocation_log


def test_selection_is_deterministic_and_gold_is_separated(default_config):
    first = load_humaneval(default_config.experiment)
    second = load_humaneval(default_config.experiment)
    assert [task.task_id for task in first.public_tasks] == [task.task_id for task in second.public_tasks]
    assert len(first.public_tasks) == 4
    assert set(first.public_tasks[0].model_dump()) == {"task_id", "prompt", "entry_point"}
    assert set(first.private_cases[first.public_tasks[0].task_id].model_dump()) == {
        "task_id",
        "test",
        "entry_point",
    }
    assert "canonical_solution" not in json.dumps(first.metadata)


def test_gold_tests_and_solutions_never_enter_agent_messages(default_config, tmp_path):
    selection = load_humaneval(default_config.experiment)
    task = selection.public_tasks[0]
    workflow, invocation_log = _workflow(default_config, tmp_path)
    workflow.run_baseline(task, "leakage-test", 7)

    official = read_raw_humaneval(default_config.experiment.dataset.path)[task.task_id]
    serialized_messages = json.dumps(
        [row["messages"] for row in invocation_log.read_all()], ensure_ascii=False
    )
    assert official["test"] not in serialized_messages
    assert official["canonical_solution"] not in serialized_messages
