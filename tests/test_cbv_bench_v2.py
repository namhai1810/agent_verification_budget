from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_verify.cbv_bench.cache import ExactCache, cache_key
from agent_verify.cbv_bench.config import load_cbv_config
from agent_verify.cbv_bench.labels import build_effect_records
from agent_verify.cbv_bench.model_metadata import build_model_pair_snapshot
from agent_verify.cbv_bench.resource_budget import CollectionBudget, CollectionBudgetExhausted
from agent_verify.cbv_bench.sequences import (
    LEVEL1_SEQUENCES,
    LEVEL2_SEQUENCES,
    required_prefixes,
    teacher_call_count,
)
from agent_verify.cbv_bench.tree import PrivacyGuard, deterministic_seed
from agent_verify.evaluation.bigcodebench_runner import BigCodeBenchRunner
from agent_verify.logging import JsonlStore
from agent_verify.schemas import BigCodeBenchPrivateCase, Message, PrivateEvaluationCase


def test_model_pair_snapshot_is_explicit_and_secret_free() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_cbv_config(root / "configs" / "cbv_bench_bigcodebench_v2.yaml")
    snapshot = build_model_pair_snapshot(config)
    assert snapshot.model_pair_id == "qwen25-coder-7b-awq__qwen25-coder-32b-awq"
    assert snapshot.student_model.model_id == "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
    assert snapshot.teacher_model.model_id == "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
    assert set(snapshot.prompt_version) == {"planner", "coder", "reviewer", "teacher"}
    assert snapshot.inference_config["teacher"]["max_output_tokens"] == 3000
    serialized = snapshot.model_dump_json()
    assert "api_key_env" in serialized
    assert "VLLM_API_KEY=" not in serialized


def test_level_definitions_and_shared_prefix_teacher_cost() -> None:
    assert LEVEL1_SEQUENCES == ("SSS", "VSS", "SVS", "SSV")
    assert set(LEVEL2_SEQUENCES) == {"SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV"}
    assert teacher_call_count(1) == 3
    assert teacher_call_count(2) == 7
    assert len([value for value in required_prefixes(2) if len(value) == 3]) == 8


def test_seed_depends_on_actual_history() -> None:
    assert deterministic_seed(7, 2, 102, "SV") == deterministic_seed(7, 2, 102, "SV")
    assert deterministic_seed(7, 2, 102, "SV") != deterministic_seed(7, 2, 102, "VV")


def test_collection_budget_reserves_before_call() -> None:
    budget = CollectionBudget(max_calls=1, max_tokens=100)
    reservation = budget.reserve(80)
    with pytest.raises(CollectionBudgetExhausted):
        budget.reserve(30)
    budget.settle(reservation, 60)
    assert budget.used_calls == 1
    with pytest.raises(CollectionBudgetExhausted):
        budget.reserve(1)


def test_privacy_guard_rejects_hidden_test() -> None:
    private = PrivateEvaluationCase(task_id="x", test="assert candidate(12345) == 67890",
                                    entry_point="candidate")
    with pytest.raises(RuntimeError, match="Privacy violation"):
        PrivacyGuard.assert_messages_public(
            [Message(role="user", content="assert candidate(12345) == 67890")], private
        )


def test_bigcodebench_scratch_is_writable_without_filesystem_escape(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_cbv_config(root / "configs" / "cbv_bench_bigcodebench_v2.yaml")
    runner = BigCodeBenchRunner(
        config.evaluation,
        JsonlStore(tmp_path / "evaluations.jsonl"),
    )
    code = """
import matplotlib.pyplot as plt

def task_func():
    with open("artifact.txt", "w", encoding="utf-8") as handle:
        handle.write("ok")
    with open("artifact.txt", encoding="utf-8") as handle:
        local_write_ok = handle.read() == "ok"
    plt.plot([0, 1], [0, 1])
    plt.savefig("plot.png")
    plt.close()
    plot_write_ok = __import__("os").path.getsize("plot.png") > 0
    try:
        with open("/etc/passwd", encoding="utf-8") as handle:
            handle.read(1)
    except PermissionError:
        escape_blocked = True
    else:
        escape_blocked = False
    return local_write_ok and plot_write_ok and escape_blocked
"""
    tests = """
import unittest

class TestSandbox(unittest.TestCase):
    def test_private_scratch_only(self):
        self.assertTrue(task_func())
"""
    result = runner.evaluate(
        run_id="sandbox-regression",
        case=BigCodeBenchPrivateCase(
            task_id="BigCodeBench/sandbox-regression",
            test=tests,
            entry_point="task_func",
        ),
        code=code,
        branch="cbv:SSS:r0",
    )
    assert result.passed is True
    assert result.status == "PASS"


def test_cache_rejects_incompatible_or_conflicting_rows(tmp_path: Path) -> None:
    cache = ExactCache(tmp_path / "cache.jsonl", "fp-a")
    key = cache_key("x", "fp-a", {"task": "t"})
    row = {"cache_key": key, "compatibility_fingerprint": "fp-a", "status": "completed", "value": 1}
    cache.put(row)
    assert cache.get(key) == row
    with pytest.raises(ValueError, match="Conflicting"):
        cache.put({**row, "value": 2})
    with pytest.raises(ValueError, match="different compatibility"):
        cache.put({**row, "cache_key": "other", "compatibility_fingerprint": "fp-b"})


def test_level2_effects_use_matched_history_siblings(tmp_path: Path) -> None:
    passed = {
        "SSS": False,
        "VSS": False,
        "SVS": False,
        "SSV": False,
        "VVS": True,
        "VSV": False,
        "SVV": False,
        "VVV": True,
    }
    rows = [
        {
            "status": "completed",
            "compatibility_fingerprint": "fp",
            "run_id": "run",
            "model_pair_id": "student__teacher",
            "model_pair_fingerprint": "pair-fp",
            "dataset": "synthetic-test-only",
            "task_id": "task",
            "rollout_index": 0,
            "action_sequence": sequence,
            "passed": outcome,
            "teacher_total_tokens": sequence.count("V") * 10,
        }
        for sequence, outcome in passed.items()
    ]
    (tmp_path / "trajectories.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = build_effect_records(tmp_path, "fp", 2)
    effects = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    conditional = [row for row in effects if row["effect_type"] == "conditional"]
    assert len(conditional) == 12
    cp2_after_v = next(
        row
        for row in conditional
        if row["checkpoint"] == 2
        and row["preceding_history"] == "V"
        and row["fixed_other_actions"] == "S"
    )
    assert (cp2_after_v["skip_sequence"], cp2_after_v["verify_sequence"]) == ("VSS", "VVS")
    assert cp2_after_v["label"] == "Helpful"
