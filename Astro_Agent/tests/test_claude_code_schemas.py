from claude_code_toolbox.schemas import (
    ClaudeCodeResult,
    ClaudeCodeStatus,
    ClaudeCodeTask,
    ClaudeCodeTaskType,
)


def test_task_roundtrip():
    t = ClaudeCodeTask(
        task_id="t1",
        type=ClaudeCodeTaskType.PAPER_REFINE,
        title="x",
        prompt="hi",
    )
    payload = t.model_dump(mode="json")
    t2 = ClaudeCodeTask.model_validate(payload)
    assert t2.type is ClaudeCodeTaskType.PAPER_REFINE
    assert t2.human_review_required is True


def test_result_defaults():
    r = ClaudeCodeResult(
        task_id="t1",
        type=ClaudeCodeTaskType.CODE_REVIEW,
        status=ClaudeCodeStatus.OK,
    )
    assert r.changed_files == []
    assert r.human_review_required is True
    assert r.status is ClaudeCodeStatus.OK


def test_all_task_types_addressable():
    # Ensures the enum is exhaustive with the 6 types in the spec.
    expected = {"tool_write", "code_review", "bug_fix", "paper_refine", "latex_check", "experiment_audit"}
    assert {t.value for t in ClaudeCodeTaskType} == expected
