"""Builders that turn a raw signal from AnalysisState into a ClaudeCodeTask.

MVP scope: ``paper_refine`` and ``code_review`` are fully populated; the four
remaining types are stubs returning a minimal task that the client will execute
the same way (so tests can exercise the routing path even before the prompts
are finalized).
"""

from __future__ import annotations

import json
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
# Phase-2 task builders
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


_TOOL_WRITE_DEFAULT = """\
You are astro-tool-writer inside Astro_Agent. Implement a literature-derived
analysis capability only from the bounded toolbox_gap JSON below.

TOOLBOX_GAP_JSON:
{{gap_json}}

Rules:
- Treat the RAG/KG references as provenance, not as permission to invent missing
  equations or thresholds.
- Inspect similar modules listed in the gap and follow their local style.
- Add or patch only bounded files under astro_toolbox/, tests/, docs/, or the
  run-local artifacts needed for validation.
- The primary function must return a JSON-serializable dict with status,
  products, warnings, provenance, and validation.
- Add a smoke test or fixture that exercises success and a controlled failure.
- Do not claim final science results when required inputs are missing.
- Do not register a skill yourself unless the repository already has that
  convention; the host will register after validation.

Required final response: strict JSON with keys:
validation_passed (bool), changed_files (list), tests (list), docs (list),
risks (list), implementation_summary (string), skill_registration_notes (string).
"""


def build_tool_write_task(inputs: Mapping[str, Any]) -> ClaudeCodeTask:
    gap = inputs.get("gap", {})
    gap_json = json.dumps(gap, ensure_ascii=False, indent=2, default=str)
    template = load_agent_prompt("astro-tool-writer") or _TOOL_WRITE_DEFAULT
    prompt = render(template, {"gap": gap_json, "gap_json": gap_json})
    target_module = ""
    if isinstance(gap, Mapping):
        target_module = str(gap.get("target_module") or "")
    target_files = list(inputs.get("target_files", []))
    if target_module:
        target_files.append(f"astro_toolbox/{target_module}.py")
    return ClaudeCodeTask(
        task_id=_new_task_id("tool-write"),
        type=ClaudeCodeTaskType.TOOL_WRITE,
        title="Add validated astro_toolbox method",
        prompt=prompt,
        inputs=dict(inputs),
        target_files=list(dict.fromkeys(target_files)),
        allowed_write_dirs=["astro_toolbox/", "tests/", "docs/", "runs/"],
        permission_mode=str(inputs.get("permission_mode", "plan")),
        timeout_sec=int(inputs.get("timeout_sec", 900) or 900),
        human_review_required=True,
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
