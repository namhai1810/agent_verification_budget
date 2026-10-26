from __future__ import annotations

import json
import signal

from agent_verify.agents.fixed import CoderAgent, PlannerAgent, ReviewerAgent
from agent_verify.analysis.phase3_training import deterministic_holdout, load_or_create_holdout
from agent_verify.budget import BudgetManager
from agent_verify.evaluation.livecodebench_runner import (
    EvaluatorInfrastructureError,
    LiveCodeBenchRunner,
)
from agent_verify.evaluation.worker import _livecodebench_result
from agent_verify.experiments.phase1 import Phase1Experiment
from agent_verify.livecodebench_cli import main as livecodebench_main
from agent_verify.livecodebench_dataset import (
    EXPECTED_COLUMNS,
    row_to_private_case,
    row_to_public_task,
    select_rows,
)
from agent_verify.llm import LLMClient
from agent_verify.logging import JsonlStore
from agent_verify.schemas import EvaluationResult, LiveCodeBenchPublicTask
from agent_verify.workflow import FixedWorkflow


def _row(index: int, difficulty: str, platform: str = "leetcode") -> dict:
    metadata = {"func_name": "solve" if platform == "leetcode" else None}
    public = [{"input": "1\n", "output": "1\n", "testtype": "functional"}]
    private = [{"input": "PRIVATE_SENTINEL", "output": "2\n", "testtype": "functional"}]
    return {
        "question_title": f"Task {index}",
        "question_content": "Solve the stated problem.",
        "platform": platform,
        "question_id": f"q{index:03d}",
        "contest_id": "contest",
        "contest_date": "2025-02-01T00:00:00",
        "starter_code": "class Solution:\n    def solve(self, x):\n",
        "difficulty": difficulty,
        "public_test_cases": json.dumps(public),
        "private_test_cases": json.dumps(private),
        "metadata": json.dumps(metadata),
    }


def test_schema_fixture_matches_expected_columns():
    assert set(_row(0, "easy")) == EXPECTED_COLUMNS


def test_selection_is_deterministic_and_prefers_easy_medium():
    rows = [_row(index, "easy", "leetcode") for index in range(55)]
    rows += [_row(index + 55, "medium", "atcoder") for index in range(50)]
    rows += [_row(index + 105, "hard", "atcoder") for index in range(30)]
    first, stats = select_rows(rows, task_count=100, seed=42)
    second, _ = select_rows(rows, task_count=100, seed=42)
    assert [row["question_id"] for row in first] == [row["question_id"] for row in second]
    assert stats["selected_difficulty"] == {"easy": 50, "medium": 50}
    assert stats["hard_supplemented"] == 0


def test_private_tests_never_enter_public_task():
    row = _row(1, "medium")
    public = row_to_public_task(row)
    private = row_to_private_case(row)
    assert public.benchmark == "livecodebench"
    assert "PRIVATE_SENTINEL" not in public.model_dump_json()
    assert "PRIVATE_SENTINEL" in private.input_output


def test_official_result_mapping():
    passed = _livecodebench_result(
        lambda *_args, **_kwargs: ([True], {}), {"input_output": "{}", "code": "", "per_test_timeout": 1}
    )
    timeout = _livecodebench_result(
        lambda *_args, **_kwargs: ([-3], {"error_code": -3}),
        {"input_output": "{}", "code": "", "per_test_timeout": 1},
    )
    runtime = _livecodebench_result(
        lambda *_args, **_kwargs: ([-4], {"error_code": -4}),
        {"input_output": "{}", "code": "", "per_test_timeout": 1},
    )
    assert passed["status"] == "PASS" and passed["passed"] is True
    assert timeout["status"] == "TIMEOUT"
    assert runtime["status"] == "RUNTIME_ERROR"


def test_evaluator_infrastructure_error_is_not_a_model_failure():
    result = EvaluationResult(
        evaluation_id="eval_infra",
        run_id="run",
        task_id="LiveCodeBench/example",
        branch="baseline",
        passed=False,
        status="EVALUATOR_ERROR:ImportError",
        latency_seconds=0.1,
        sandbox="test",
    )
    try:
        LiveCodeBenchRunner._require_model_outcome(result)
    except EvaluatorInfrastructureError:
        pass
    else:
        raise AssertionError("Infrastructure failure must abort the record, not become Y=0")


def test_worker_resource_limit_exit_is_a_model_timeout():
    assert LiveCodeBenchRunner._worker_exit_status(-signal.SIGXCPU) == "TIMEOUT"
    assert LiveCodeBenchRunner._worker_exit_status(-signal.SIGKILL) == "TIMEOUT"


def test_unknown_worker_exit_remains_an_infrastructure_error():
    status = LiveCodeBenchRunner._worker_exit_status(1)
    assert status == "EVALUATOR_ERROR:worker_exit_1"


def test_livecodebench_prompts_are_io_aware_and_private_free(default_config, tmp_path):
    task = LiveCodeBenchPublicTask(
        task_id="LiveCodeBench/example",
        prompt="Read an integer and print it.",
        entry_point="",
        benchmark="livecodebench",
        title="Echo",
        difficulty="easy",
        platform="atcoder",
        contest_date="2025-01-02",
        input_mode="stdin",
    )
    invocations = JsonlStore(tmp_path / "invocations.jsonl")
    client = LLMClient(
        default_config.provider,
        BudgetManager(default_config.budget, default_config.provider),
        invocations,
    )
    workflow = FixedWorkflow(
        PlannerAgent(client, default_config.models.planner),
        CoderAgent(client, default_config.models.coder),
        ReviewerAgent(client, default_config.models.reviewer),
        default_config.models,
        JsonlStore(tmp_path / "states.jsonl"),
    )
    workflow.run_baseline(task, "lcb-prompt-test", 42)
    records = invocations.read_all()
    coder_system = records[1]["messages"][0]["content"]
    coder_user = records[1]["messages"][1]["content"]
    assert "standard input" in coder_system
    assert "standard output" in coder_system
    assert "BENCHMARK: livecodebench" in coder_user
    assert "PRIVATE_SENTINEL" not in json.dumps(records)


def test_fixed_test_split_is_deterministic_and_grouped():
    metadata = {
        f"LiveCodeBench/{index}": {
            "benchmark": "livecodebench",
            "difficulty": "easy" if index < 10 else "medium",
        }
        for index in range(20)
    }
    first = deterministic_holdout(metadata, fraction=0.2, seed=42)
    second = deterministic_holdout(metadata, fraction=0.2, seed=42)
    assert first == second
    assert len(first) == 4


def test_fixed_test_split_rejects_a_different_task_universe(tmp_path):
    split_path = tmp_path / "fixed_test_tasks.json"
    original = {
        f"LiveCodeBench/{index}": {"benchmark": "livecodebench", "difficulty": "easy"} for index in range(10)
    }
    load_or_create_holdout(split_path, original, fraction=0.2, seed=42)
    changed = {**original, "LiveCodeBench/new": {"benchmark": "livecodebench", "difficulty": "easy"}}
    try:
        load_or_create_holdout(split_path, changed, fraction=0.2, seed=42)
    except ValueError as exc:
        assert "different task universe" in str(exc)
    else:
        raise AssertionError("A frozen split must not be reused with another task universe")


def test_all_livecodebench_cli_help_commands(capsys):
    for command in ("prepare", "run", "analyze"):
        try:
            livecodebench_main([command, "--help"])
        except SystemExit as exc:
            assert exc.code == 0
    assert "usage:" in capsys.readouterr().out


def test_explicit_recovery_run_requires_matching_manifest(default_config, tmp_path):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    results = tmp_path / "results"
    config = default_config.model_copy(
        update={
            "experiment": default_config.experiment.model_copy(
                update={"results_dir": results, "resume": True}
            )
        }
    )
    JsonlStore(results / "manifests.jsonl").append(
        {
            "record_type": "manifest",
            "run_id": "historic-run",
            "config_fingerprint": config.fingerprint,
            "software_sha256": "old-software",
        }
    )
    experiment = Phase1Experiment(config, project_root=project, run_id_override="historic-run")
    assert experiment.run_id == "historic-run"
    assert experiment.resumed_manifest is not None


def test_175_task_extension_preserves_100_task_prefix():
    rows = [_row(index, "easy", "leetcode") for index in range(43)]
    rows += [_row(index + 43, "medium", "atcoder") for index in range(52)]
    rows += [
        _row(index + 95, "hard", "atcoder" if index % 2 == 0 else "leetcode")
        for index in range(80)
    ]
    base, _ = select_rows(rows, task_count=100, seed=42)
    extended, stats = select_rows(rows, task_count=175, seed=42)

    assert [row["question_id"] for row in extended[:100]] == [
        row["question_id"] for row in base
    ]
    assert {row["difficulty"] for row in extended[100:]} == {"hard"}
    assert stats["selected_difficulty"] == {"easy": 43, "hard": 80, "medium": 52}
