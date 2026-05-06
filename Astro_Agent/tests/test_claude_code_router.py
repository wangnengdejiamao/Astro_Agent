from analysis_agent.nodes.claude_code_router import (
    claude_code_router_node,
    decide_claude_code_tasks,
)
from claude_code_toolbox.schemas import ClaudeCodeTaskType


def test_disabled_returns_empty():
    state = {"errors": ["boom"], "enable_claude_code": False}
    assert decide_claude_code_tasks(state) == []


def test_errors_route_to_bug_fix():
    state = {"enable_claude_code": True, "errors": ["NameError: x"]}
    tasks = decide_claude_code_tasks(state)
    assert len(tasks) == 1
    assert tasks[0].type is ClaudeCodeTaskType.BUG_FIX


def test_paper_refine_requires_flag():
    state = {"enable_claude_code": True, "paper_draft": "Some text."}
    assert decide_claude_code_tasks(state) == []  # no need_paper_refinement
    state["need_paper_refinement"] = True
    tasks = decide_claude_code_tasks(state)
    assert any(t.type is ClaudeCodeTaskType.PAPER_REFINE for t in tasks)


def test_router_writes_state_keys():
    state = {
        "enable_claude_code": True,
        "toolbox_gap": {"name": "missing_xshooter_loader"},
    }
    out = claude_code_router_node(state)
    assert "claude_code_tasks" in out
    assert len(out["claude_code_tasks"]) == 1
    assert out["claude_code_tasks"][0]["type"] == "tool_write"
    assert out["claude_code_results"] == []


def test_multiple_signals_produce_multiple_tasks():
    state = {
        "enable_claude_code": True,
        "errors": ["x"],
        "latex_compile_error": "Undefined ref",
        "paper_draft": "draft",
        "need_paper_refinement": True,
    }
    tasks = decide_claude_code_tasks(state)
    types = {t.type for t in tasks}
    assert ClaudeCodeTaskType.BUG_FIX in types
    assert ClaudeCodeTaskType.LATEX_CHECK in types
    assert ClaudeCodeTaskType.PAPER_REFINE in types
