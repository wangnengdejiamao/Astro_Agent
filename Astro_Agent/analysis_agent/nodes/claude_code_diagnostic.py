"""Claude Code diagnostic hook.

Runs *only* when QA Gate is not pass. Queues two read-only Claude Code tasks:

1. ``experiment_audit`` — looks at run artifacts and reports blocking issues,
   reproducibility score and recommended next steps.
2. ``code_review``     — looks at workflow.py / tools.py / qa_gate node and
   reports suspected workflow bugs with suggested patches.

Neither task is allowed to modify production code. All results land under
``runs/<run_id>/claude_code/`` and are flagged ``human_review_required=True``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from claude_code_toolbox import build_task
from claude_code_toolbox.client import make_default_client
from claude_code_toolbox.schemas import ClaudeCodeStatus, ClaudeCodeTaskType


_DEFAULT_REVIEW_TARGETS = [
    "analysis_agent/workflow.py",
    "analysis_agent/tools.py",
    "analysis_agent/nodes/qa_gate.py",
    "analysis_agent/nodes/simbad_prior.py",
]


def _qa_is_failed(state: Dict[str, Any]) -> bool:
    qa = state.get("qa") or {}
    if qa.get("apj_gate") and qa.get("apj_gate") != "pass":
        return True
    return bool(
        qa.get("infrastructure_reasons")
        or qa.get("science_reasons")
        or qa.get("blocking_reasons")
    )


def _audit_inputs(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": state.get("run_id") or state.get("astrotool_run") or "",
        "run_dir": state.get("output_root", ""),
        "target": state.get("target", ""),
        "resolved": state.get("resolved", {}),
        "data_fetch": state.get("data_fetch", {}),
        "simbad_prior": state.get("simbad_prior", {}),
        "qa": state.get("qa", {}),
        "iterations": state.get("iterations", []),
        "rag_results_count": len(state.get("rag_results") or []),
        "kg_results_count": len(state.get("kg_results") or []),
    }


def _review_inputs(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_files": _DEFAULT_REVIEW_TARGETS,
        "context": json.dumps(
            {
                "qa": state.get("qa", {}),
                "simbad_prior": state.get("simbad_prior", {}),
                "modeling_skipped": state.get("modeling_skipped", False),
                "modeling_skip_reason": state.get("modeling_skip_reason"),
            },
            ensure_ascii=False,
        ),
    }


def claude_code_diagnostic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run experiment_audit + code_review *only* when QA failed."""
    state.setdefault("claude_code_results", [])
    state.setdefault("claude_code_tasks", [])

    if not _qa_is_failed(state):
        state["claude_code_diagnostic"] = {
            "status": "skipped",
            "reason": "QA gate did not fail; diagnostic not needed.",
        }
        return state

    if not state.get("enable_claude_code", False):
        state["claude_code_diagnostic"] = {
            "status": "skipped",
            "reason": "enable_claude_code is false; diagnostic queued but not executed.",
            "queued_task_types": ["experiment_audit", "code_review"],
        }
        return state

    audit_task = build_task(ClaudeCodeTaskType.EXPERIMENT_AUDIT, _audit_inputs(state))
    review_task = build_task(ClaudeCodeTaskType.CODE_REVIEW, _review_inputs(state))

    # Diagnostic tasks must never write production code, only report.
    audit_task.allowed_write_dirs = ["runs/"]
    review_task.allowed_write_dirs = ["runs/"]
    audit_task.human_review_required = True
    review_task.human_review_required = True

    run_id = state.get("run_id") or state.get("astrotool_run") or None
    client = make_default_client(run_id=run_id)

    queued = [audit_task, review_task]
    state["claude_code_tasks"].extend(t.model_dump(mode="json") for t in queued)

    results: List[Dict[str, Any]] = []
    for task in queued:
        result = client.execute(task)
        # Diagnostics never auto-merge.
        result.human_review_required = True
        if result.status is ClaudeCodeStatus.OK:
            result.status = ClaudeCodeStatus.NEEDS_REVIEW
        results.append(result.model_dump(mode="json"))

    state["claude_code_results"].extend(results)

    # Also write a small index under the run dir so the abnormal_report can
    # link to the diagnostic outputs without scanning runs/.
    output_root = state.get("output_root")
    if output_root:
        try:
            d = Path(output_root) / "claude_code"
            d.mkdir(parents=True, exist_ok=True)
            (d / "claude_code_tasks.json").write_text(
                json.dumps(state["claude_code_tasks"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (d / "claude_code_results.json").write_text(
                json.dumps(state["claude_code_results"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            state.setdefault("artifacts", []).append(str(d / "claude_code_results.json"))
        except Exception:
            pass

    state["claude_code_diagnostic"] = {
        "status": "executed",
        "task_ids": [t.task_id for t in queued],
        "results_count": len(results),
    }
    return state


__all__ = ["claude_code_diagnostic_node"]
