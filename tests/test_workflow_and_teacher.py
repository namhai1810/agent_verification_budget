from __future__ import annotations

from types import SimpleNamespace

from agent_verify.agents.fixed import CoderAgent, PlannerAgent, ReviewerAgent
from agent_verify.budget import BudgetManager
from agent_verify.dataset import load_humaneval
from agent_verify.llm import LLMClient
from agent_verify.logging import JsonlStore
from agent_verify.schemas import (
    CodeArtifact,
    InvocationContext,
    PlanArtifact,
    PublicTask,
    ReviewArtifact,
    VerificationResult,
)
from agent_verify.verification.teacher import (
    TeacherVerifier,
    artifact_schema,
    checkpoint_guidance,
)
from agent_verify.workflow import FixedWorkflow


def _components(default_config, tmp_path):
    invocations = JsonlStore(tmp_path / "invocations.jsonl")
    events = JsonlStore(tmp_path / "events.jsonl")
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
        JsonlStore(tmp_path / "checkpoints.jsonl"),
    )
    teacher = TeacherVerifier(client, default_config.models.teacher, events)
    return workflow, teacher


def test_checkpoint_restore_is_immutable_and_branches_are_isolated(default_config, tmp_path):
    task = load_humaneval(default_config.experiment).public_tasks[0]
    workflow, _ = _components(default_config, tmp_path)
    baseline = workflow.run_baseline(task, "restore-test", 11)
    state = baseline.checkpoints[0]
    snapshot = state.model_dump_json()
    original = artifact_schema(1).model_validate(state.current_artifact)

    skip = workflow.continue_from(
        task=task,
        restored_state=state,
        artifact=original,
        run_id="restore-test",
        rollout_seed=22,
        branch="skip",
    )
    verify = workflow.continue_from(
        task=task,
        restored_state=state,
        artifact=original.model_copy(deep=True),
        run_id="restore-test",
        rollout_seed=22,
        branch="verify",
    )
    assert state.model_dump_json() == snapshot
    assert skip.final_code == verify.final_code
    assert skip.checkpoints[0].branch == "skip"
    assert verify.checkpoints[0].branch == "verify"
    assert skip.checkpoints[0].state_id != verify.checkpoints[0].state_id


def test_checkpoint_three_artifact_directly_controls_submission(default_config, tmp_path):
    task = load_humaneval(default_config.experiment).public_tasks[0]
    workflow, _ = _components(default_config, tmp_path)
    baseline = workflow.run_baseline(task, "cp3-test", 13)
    state = baseline.checkpoints[2]
    corrected = ReviewArtifact(
        code=f"def {task.entry_point}(*args, **kwargs):\n    return 123\n",
        review_summary="Synthetic test correction.",
        issues_found=["placeholder"],
        changed=True,
    )
    result = workflow.continue_from(
        task=task,
        restored_state=state,
        artifact=corrected,
        run_id="cp3-test",
        rollout_seed=23,
        branch="verify",
    )
    assert result.final_code == corrected.code
    assert result.checkpoints == []


def test_mock_teacher_preserves_original_and_logs(default_config, tmp_path):
    task = load_humaneval(default_config.experiment).public_tasks[0]
    workflow, teacher = _components(default_config, tmp_path)
    baseline = workflow.run_baseline(task, "teacher-test", 17)
    state = baseline.checkpoints[1]
    original = artifact_schema(2).model_validate(state.current_artifact)
    output = teacher.verify(
        task,
        state,
        original,
        InvocationContext(
            task_id=task.task_id,
            checkpoint=2,
            role="teacher",
            branch="verify",
            run_id="teacher-test",
            seed=42,
        ),
    )
    assert output.result.changed is False
    assert output.corrected_artifact == original


def test_teacher_prompt_is_checkpoint_specific_and_downstream_aware(default_config, tmp_path):
    task = load_humaneval(default_config.experiment).public_tasks[0]
    workflow, teacher = _components(default_config, tmp_path)
    baseline = workflow.run_baseline(task, "teacher-prompt-test", 18)

    expected_phrases = {
        1: ("Planner artifact", "Coder and then the Reviewer will run", "planner"),
        2: ("Coder artifact", "Reviewer will run", "coder"),
        3: ("final artifact", "no downstream agent available to repair it", "reviewer"),
    }
    for checkpoint_index, state in enumerate(baseline.checkpoints, start=1):
        original = artifact_schema(checkpoint_index).model_validate(state.current_artifact)
        teacher.verify(
            task,
            state,
            original,
            InvocationContext(
                task_id=task.task_id,
                checkpoint=checkpoint_index,
                role="teacher",
                branch=f"cp{checkpoint_index}_verify",
                run_id="teacher-prompt-test",
                seed=40 + checkpoint_index,
            ),
        )

    teacher_records = [
        record
        for record in JsonlStore(tmp_path / "invocations.jsonl").read_all()
        if record["context"]["role"] == "teacher"
    ]
    assert len(teacher_records) == 3
    for checkpoint_index, record in enumerate(teacher_records, start=1):
        system_prompt = record["messages"][0]["content"]
        user_prompt = record["messages"][1]["content"]
        focus, downstream, role = expected_phrases[checkpoint_index]
        assert focus in system_prompt
        assert downstream in system_prompt
        assert "derive the task contract independently" in system_prompt
        assert "prefer no change over a speculative" in system_prompt
        assert f"CHECKPOINT: {checkpoint_index}" in user_prompt
        assert f"AGENT_ROLE: {role}" in user_prompt
        assert f"ENTRY_POINT: {task.entry_point}" in user_prompt
        assert "PROBLEM:" in user_prompt
        assert "CURRENT_ARTIFACT:" in user_prompt
        assert "ARTIFACT_SCHEMA:" in user_prompt


def test_checkpoint_guidance_rejects_unknown_checkpoint():
    try:
        checkpoint_guidance(4)
    except ValueError as exc:
        assert "Unsupported checkpoint index: 4" in str(exc)
    else:
        raise AssertionError("Expected unsupported checkpoint to raise ValueError")


def test_reviewer_derives_changed_and_logs_model_mismatch(default_config, tmp_path):
    task = PublicTask(
        task_id="Synthetic/reviewer",
        prompt="def answer():\n    pass\n",
        entry_point="answer",
    )
    plan = PlanArtifact(
        summary="Return one.",
        algorithm=["Return one"],
        edge_cases=[],
        complexity="O(1)",
    )
    original = CodeArtifact(code="def answer():\n    return 0\n")
    declared = ReviewArtifact(
        code="def answer():\n    return 1\n",
        review_summary="Corrected return value.",
        issues_found=["wrong value"],
        changed=False,
    )
    llm_result = SimpleNamespace(
        value=declared,
        record=SimpleNamespace(invocation_id="inv_reviewer_mismatch"),
    )
    client = SimpleNamespace(invoke=lambda **_kwargs: llm_result)
    events = JsonlStore(tmp_path / "reviewer-events.jsonl")
    reviewer = ReviewerAgent(client, default_config.models.reviewer, events)

    output = reviewer.run(
        task,
        plan,
        original,
        InvocationContext(
            task_id=task.task_id,
            checkpoint=3,
            role="reviewer",
            branch="baseline",
            run_id="consistency-test",
            seed=1,
        ),
    )

    assert output.artifact.changed is True
    assert output.llm_result.value.changed is False
    warning = events.read_all()[0]
    assert warning["record_type"] == "consistency_warning"
    assert warning["declared_value"] is False
    assert warning["derived_value"] is True


def test_teacher_derives_changed_and_logs_model_mismatch(default_config, tmp_path):
    task = load_humaneval(default_config.experiment).public_tasks[0]
    workflow, _ = _components(default_config, tmp_path)
    baseline = workflow.run_baseline(task, "teacher-consistency-test", 19)
    state = baseline.checkpoints[1]
    original = artifact_schema(2).model_validate(state.current_artifact)
    corrected = original.model_copy(update={"implementation_notes": "Teacher clarified the notes."})
    declared = VerificationResult(
        verdict="correct",
        corrected_artifact=corrected,
        explanation="No code change was required.",
        changed=False,
    )
    llm_result = SimpleNamespace(
        value=declared,
        record=SimpleNamespace(
            invocation_id="inv_teacher_mismatch",
            cost_usd=0.0,
            latency_seconds=0.1,
        ),
    )
    client = SimpleNamespace(invoke=lambda **_kwargs: llm_result)
    events = JsonlStore(tmp_path / "teacher-consistency-events.jsonl")
    teacher = TeacherVerifier(client, default_config.models.teacher, events)

    output = teacher.verify(
        task,
        state,
        original,
        InvocationContext(
            task_id=task.task_id,
            checkpoint=2,
            role="teacher",
            branch="verify",
            run_id="consistency-test",
            seed=2,
        ),
    )

    assert output.result.changed is True
    assert output.corrected_artifact == corrected
    assert llm_result.value.changed is False
    warning = events.read_all()[0]
    assert warning["record_type"] == "consistency_warning"
    assert warning["declared_value"] is False
    assert warning["derived_value"] is True
