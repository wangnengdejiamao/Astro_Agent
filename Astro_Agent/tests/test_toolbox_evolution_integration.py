"""Sanity test for the toolbox-evolution → tool_write integration path.

We do NOT exercise the real Claude binary — we only assert that when a
``toolbox_gap`` shows up in state, the router emits a tool_write task with the
right shape and that the task carries the gap metadata into its inputs.
"""

from analysis_agent.nodes.claude_code_router import decide_claude_code_tasks
from claude_code_toolbox.schemas import ClaudeCodeTaskType


def test_toolbox_gap_emits_tool_write():
    gap = {
        "name": "missing_xshooter_loader",
        "rationale": "no astro_toolbox module ingests XSHOOTER spectra",
        "suggested_module": "astro_toolbox/xshooter.py",
    }
    state = {"enable_claude_code": True, "toolbox_gap": gap}
    tasks = decide_claude_code_tasks(state)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.type is ClaudeCodeTaskType.TOOL_WRITE
    assert "astro_toolbox/" in t.allowed_write_dirs
    # the gap should be retained in inputs so the prompt template can render it
    assert t.inputs["gap"] == gap
    # patches are never applied automatically
    assert t.human_review_required is True


def test_no_gap_no_tool_write():
    state = {"enable_claude_code": True}
    types = {t.type for t in decide_claude_code_tasks(state)}
    assert ClaudeCodeTaskType.TOOL_WRITE not in types
