"""Execution node: runs every queued ClaudeCodeTask via the client.

The node is defensive — if the binary is missing, or pydantic deserialization
fails, it records a SKIPPED/ERROR result instead of raising, so the rest of
the LangGraph pipeline keeps moving.
"""

from __future__ import annotations

from typing import Any, Dict

from analysis_agent import method_learning
from claude_code_toolbox import ClaudeCodeTask
from claude_code_toolbox.client import make_default_client
from claude_code_toolbox.safety.output_guard import (
    PaperRefinementGuardError,
    guard_paper_refinement,
)
from claude_code_toolbox.schemas import ClaudeCodeStatus, ClaudeCodeTaskType


def claude_code_execute_node(state: Dict[str, Any]) -> Dict[str, Any]:
    raw_tasks = state.get("claude_code_tasks") or []
    if not raw_tasks:
        state.setdefault("claude_code_results", [])
        return state

    run_id = state.get("run_id") or state.get("astrotool_run") or None
    client = make_default_client(run_id=run_id)

    results: list[dict] = list(state.get("claude_code_results") or [])
    reports: dict[str, Any] = {}

    for raw in raw_tasks:
        task = ClaudeCodeTask.model_validate(raw)
        result = client.execute(task)

        # paper_refine guard: never let unsupported new claims through
        if (
            task.type is ClaudeCodeTaskType.PAPER_REFINE
            and result.status is ClaudeCodeStatus.OK
        ):
            original = str(task.inputs.get("draft", ""))
            try:
                guard_paper_refinement(original, result.summary)
            except PaperRefinementGuardError as exc:
                result.status = ClaudeCodeStatus.BLOCKED
                result.risks.append(str(exc))
                result.human_review_required = True

        if task.type is ClaudeCodeTaskType.PAPER_REFINE:
            reports["paper_refinement_report"] = {
                "status": result.status.value,
                "summary": result.summary,
                "risks": result.risks,
                "raw_output_path": result.raw_output_path,
            }
        elif task.type is ClaudeCodeTaskType.CODE_REVIEW:
            reports["code_quality_report"] = {
                "status": result.status.value,
                "findings": result.findings,
                "summary": result.summary,
                "raw_output_path": result.raw_output_path,
            }
        elif task.type is ClaudeCodeTaskType.TOOL_WRITE:
            registration = method_learning.register_dynamic_skill_if_valid(
                gap=task.inputs.get("gap"),
                claude_result=result.model_dump(mode="json"),
            )
            reports["toolbox_gap_report"] = {
                "status": result.status.value,
                "patch_summary": result.patch_summary or result.summary,
                "changed_files": result.changed_files,
                "tests": result.tests,
                "docs": result.docs,
                "risks": result.risks,
                "human_review_required": result.human_review_required,
                "dynamic_skill_registration": registration,
            }
            state["dynamic_skill_registration"] = registration

        results.append(result.model_dump(mode="json"))

    state["claude_code_results"] = results
    for k, v in reports.items():
        state[k] = v
    return state


__all__ = ["claude_code_execute_node"]
