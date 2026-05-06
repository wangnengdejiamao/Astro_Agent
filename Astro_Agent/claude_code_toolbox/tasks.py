"""Builders that turn a raw signal from AnalysisState into a ClaudeCodeTask.

MVP scope: ``paper_refine`` and ``code_review`` are fully populated; the four
remaining types are stubs returning a minimal task that the client will execute
the same way (so tests can exercise the routing path even before the prompts
are finalized).
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, Mapping

from .prompt_loader import load_agent_prompt, render
from .schemas import ClaudeCodeTask, ClaudeCodeTaskType


def _new_task_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# MVP builders
# ---------------------------------------------------------------------------

_PAPER_REFINE_DEFAULT = """\
You are astro-paper-refiner. Improve clarity, flow and logical structure of the
draft below. Do NOT introduce any new scientific claim, number, or citation
that is not already supported by the existing draft. Preserve all units,
provenance markers, and figure/table references exactly.

DRAFT:
{{draft}}
"""

_CODE_REVIEW_DEFAULT = """\
You are astro-code-reviewer. Review the following code changes for correctness,
security, unit-system consistency (CGS vs SI vs astropy.units) and provenance
(every numerical value should trace back to a source artifact). Output JSON
with keys: findings (list), risks (list), suggested_followups (list).

TARGET FILES: {{target_files}}

CONTEXT:
{{context}}
"""


def build_paper_refine_task(inputs: Mapping[str, Any]) -> ClaudeCodeTask:
    draft = str(inputs.get("draft", ""))
    template = load_agent_prompt("astro-paper-refiner") or _PAPER_REFINE_DEFAULT
    prompt = render(template, {"draft": draft})
    return ClaudeCodeTask(
        task_id=_new_task_id("paper-refine"),
        type=ClaudeCodeTaskType.PAPER_REFINE,
        title="Refine paper draft",
        prompt=prompt,
        inputs=dict(inputs),
        target_files=list(inputs.get("target_files", [])),
        allowed_write_dirs=["papers/drafts/", "runs/"],
        permission_mode="plan",
        human_review_required=True,
    )


def build_code_review_task(inputs: Mapping[str, Any]) -> ClaudeCodeTask:
    target_files = list(inputs.get("target_files", []))
    context = str(inputs.get("context", ""))
    template = load_agent_prompt("astro-code-reviewer") or _CODE_REVIEW_DEFAULT
    prompt = render(
        template,
        {"target_files": ", ".join(target_files), "context": context},
    )
    return ClaudeCodeTask(
        task_id=_new_task_id("code-review"),
        type=ClaudeCodeTaskType.CODE_REVIEW,
        title="Review code changes",
        prompt=prompt,
        inputs=dict(inputs),
        target_files=target_files,
        allowed_write_dirs=["runs/"],  # review writes only its report
        permission_mode="plan",
        human_review_required=True,
    )


# ---------------------------------------------------------------------------
# Stub builders for the four phase-2 task types (still safe to execute)
# ---------------------------------------------------------------------------

def _stub_builder(
    type_: ClaudeCodeTaskType,
    title: str,
    agent_name: str,
    default_prompt: str,
    write_dirs: list[str],
) -> Callable[[Mapping[str, Any]], ClaudeCodeTask]:
    def _build(inputs: Mapping[str, Any]) -> ClaudeCodeTask:
        template = load_agent_prompt(agent_name) or default_prompt
        prompt = render(template, {k: str(v) for k, v in inputs.items()})
        return ClaudeCodeTask(
            task_id=_new_task_id(type_.value.replace("_", "-")),
            type=type_,
            title=title,
            prompt=prompt,
            inputs=dict(inputs),
            target_files=list(inputs.get("target_files", [])),
            allowed_write_dirs=write_dirs,
            permission_mode="plan",
            human_review_required=True,
        )

    return _build


build_tool_write_task = _stub_builder(
    ClaudeCodeTaskType.TOOL_WRITE,
    "Add new astro_toolbox tool",
    "astro-tool-writer",
    "You are astro-tool-writer. Implement a new tool, tests, and docs for the gap: {{gap}}",
    ["astro_toolbox/", "tests/", "docs/"],
)

build_bug_fix_task = _stub_builder(
    ClaudeCodeTaskType.BUG_FIX,
    "Propose patch from error log",
    "astro-bug-fixer",
    "You are astro-bug-fixer. Propose a minimal patch for: {{error_log}}",
    ["astro_toolbox/", "analysis_agent/", "tests/"],
)

build_latex_check_task = _stub_builder(
    ClaudeCodeTaskType.LATEX_CHECK,
    "Check ApJ/aastex631 LaTeX",
    "astro-latex-checker",
    "You are astro-latex-checker. Validate aastex631 conformance: {{tex_path}}",
    ["papers/drafts/", "runs/"],
)

build_experiment_audit_task = _stub_builder(
    ClaudeCodeTaskType.EXPERIMENT_AUDIT,
    "Audit experiment reproducibility",
    "astro-experiment-auditor",
    "You are astro-experiment-auditor. Audit reproducibility and QA gate completeness: {{run_dir}}",
    ["runs/"],
)


TASK_REGISTRY: Dict[ClaudeCodeTaskType, Callable[[Mapping[str, Any]], ClaudeCodeTask]] = {
    ClaudeCodeTaskType.PAPER_REFINE: build_paper_refine_task,
    ClaudeCodeTaskType.CODE_REVIEW: build_code_review_task,
    ClaudeCodeTaskType.TOOL_WRITE: build_tool_write_task,
    ClaudeCodeTaskType.BUG_FIX: build_bug_fix_task,
    ClaudeCodeTaskType.LATEX_CHECK: build_latex_check_task,
    ClaudeCodeTaskType.EXPERIMENT_AUDIT: build_experiment_audit_task,
}


def build_task(type_: ClaudeCodeTaskType, inputs: Mapping[str, Any]) -> ClaudeCodeTask:
    builder = TASK_REGISTRY[type_]
    return builder(inputs)


__all__ = [
    "build_task",
    "build_paper_refine_task",
    "build_code_review_task",
    "build_tool_write_task",
    "build_bug_fix_task",
    "build_latex_check_task",
    "build_experiment_audit_task",
    "TASK_REGISTRY",
]
