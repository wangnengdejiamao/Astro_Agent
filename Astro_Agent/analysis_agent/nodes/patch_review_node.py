"""Patch review gate.

Sits after the execute node. Any task that produced ``changed_files`` and is
flagged ``human_review_required`` forces ``state['needs_human_review'] = True``
so downstream orchestration can pause before merging anything. We never
auto-apply patches, never auto-commit, never auto-push.
"""

from __future__ import annotations

from typing import Any, Dict


def patch_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
    results = state.get("claude_code_results") or []
    pending: list[dict] = []
    for r in results:
        if r.get("changed_files") and r.get("human_review_required", True):
            pending.append(
                {
                    "task_id": r.get("task_id"),
                    "type": r.get("type"),
                    "changed_files": r.get("changed_files", []),
                    "risks": r.get("risks", []),
                    "raw_output_path": r.get("raw_output_path"),
                }
            )

    if pending:
        state["needs_human_review"] = True
        state["claude_code_pending_review"] = pending
    else:
        state.setdefault("needs_human_review", False)
        state.setdefault("claude_code_pending_review", [])
    return state


__all__ = ["patch_review_node"]
