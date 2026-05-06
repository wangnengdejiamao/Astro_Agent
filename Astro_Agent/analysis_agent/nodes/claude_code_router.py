"""Router node: decides which Claude Code tasks (if any) to enqueue.

Routing rules (mirrors spec section 5):
- ``state.errors``                       -> bug_fix
- ``state.toolbox_gap``                  -> tool_write
- ``state.paper_draft`` && need_paper_refinement -> paper_refine
- ``state.latex_compile_error``          -> latex_check
- always (before final paper)            -> experiment_audit

If none apply, returns state unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from claude_code_toolbox import build_task
from claude_code_toolbox.schemas import ClaudeCodeTask, ClaudeCodeTaskType


def decide_claude_code_tasks(state: Mapping[str, Any]) -> List[ClaudeCodeTask]:
    """Pure function: state -> list[ClaudeCodeTask]. Easy to unit test."""
    if not state.get("enable_claude_code", False):
        return []

    tasks: list[ClaudeCodeTask] = []

    errors = state.get("errors") or []
    if errors:
        tasks.append(
            build_task(
                ClaudeCodeTaskType.BUG_FIX,
                {"error_log": "\n".join(str(e) for e in errors[-20:])},
            )
        )

    gap = state.get("toolbox_gap")
    if gap:
        tasks.append(build_task(ClaudeCodeTaskType.TOOL_WRITE, {"gap": gap}))

    if state.get("paper_draft") and state.get("need_paper_refinement"):
        tasks.append(
            build_task(
                ClaudeCodeTaskType.PAPER_REFINE,
                {"draft": state["paper_draft"], "target_files": state.get("paper_files", [])},
            )
        )

    if state.get("latex_compile_error"):
        tasks.append(
            build_task(
                ClaudeCodeTaskType.LATEX_CHECK,
                {
                    "tex_path": state.get("paper_tex_path", ""),
                    "error_log": state["latex_compile_error"],
                },
            )
        )

    if state.get("need_code_review") and state.get("code_review_targets"):
        tasks.append(
            build_task(
                ClaudeCodeTaskType.CODE_REVIEW,
                {
                    "target_files": state["code_review_targets"],
                    "context": state.get("code_review_context", ""),
                },
            )
        )

    if state.get("run_experiment_audit"):
        tasks.append(
            build_task(
                ClaudeCodeTaskType.EXPERIMENT_AUDIT,
                {"run_dir": state.get("run_dir", "")},
            )
        )

    return tasks


def claude_code_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    tasks = decide_claude_code_tasks(state)
    state["claude_code_tasks"] = [t.model_dump(mode="json") for t in tasks]
    state.setdefault("claude_code_results", [])
    return state


__all__ = ["claude_code_router_node", "decide_claude_code_tasks"]
