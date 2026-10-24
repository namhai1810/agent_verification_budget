from __future__ import annotations

from agent_verify.evaluation.humaneval_runner import HumanEvalRunner
from agent_verify.logging import JsonlStore
from agent_verify.schemas import PrivateEvaluationCase


def _runner(default_config, tmp_path):
    return HumanEvalRunner(default_config.evaluation, JsonlStore(tmp_path / "evaluations.jsonl"))


def test_sandbox_executes_correct_code(default_config, tmp_path):
    runner = _runner(default_config, tmp_path)
    runner.check_sandbox()
    result = runner.evaluate(
        run_id="sandbox-test",
        case=PrivateEvaluationCase(
            task_id="Synthetic/0",
            test="def check(candidate):\n    assert candidate(2) == 3\n",
            entry_point="increment",
        ),
        code="def increment(value):\n    return value + 1\n",
        branch="test",
    )
    assert result.passed is True


def test_sandbox_denies_host_writes(default_config, tmp_path):
    marker = tmp_path / "must-not-exist"
    runner = _runner(default_config, tmp_path)
    code = f"def attempt_write():\n    open({str(marker)!r}, 'w').write('escape')\n    return True\n"
    result = runner.evaluate(
        run_id="sandbox-test",
        case=PrivateEvaluationCase(
            task_id="Synthetic/1",
            test="def check(candidate):\n    assert candidate()\n",
            entry_point="attempt_write",
        ),
        code=code,
        branch="test",
    )
    assert result.passed is False
    assert result.status == "failed:PermissionError"
    assert not marker.exists()


def test_sandbox_denies_network(default_config, tmp_path):
    runner = _runner(default_config, tmp_path)
    result = runner.evaluate(
        run_id="sandbox-test",
        case=PrivateEvaluationCase(
            task_id="Synthetic/2",
            test="def check(candidate):\n    candidate()\n",
            entry_point="attempt_network",
        ),
        code=(
            "def attempt_network():\n"
            "    import socket\n"
            "    socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        ),
        branch="test",
    )
    assert result.passed is False
    assert result.status == "failed:PermissionError"


def test_sandbox_caps_generated_output(default_config, tmp_path):
    runner = _runner(default_config, tmp_path)
    result = runner.evaluate(
        run_id="sandbox-test",
        case=PrivateEvaluationCase(
            task_id="Synthetic/3",
            test="def check(candidate):\n    candidate()\n",
            entry_point="flood_output",
        ),
        code="def flood_output():\n    print('x' * 2_000_000)\n",
        branch="test",
    )
    assert result.passed is False
    assert result.status.startswith("sandbox_exit:")
