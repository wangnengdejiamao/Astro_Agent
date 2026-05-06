"""Pydantic schemas for Claude Code toolbox tasks and results.

Kept intentionally minimal so the router and the existing AnalysisState can
absorb them without touching unrelated agents (analysis_agent core, RAG, KG,
PaperOrchestra, QA Gate).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - pydantic is a runtime dep
    raise


class ClaudeCodeTaskType(str, Enum):
    TOOL_WRITE = "tool_write"
    CODE_REVIEW = "code_review"
    BUG_FIX = "bug_fix"
    PAPER_REFINE = "paper_refine"
    LATEX_CHECK = "latex_check"
    EXPERIMENT_AUDIT = "experiment_audit"


class ClaudeCodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    NEEDS_REVIEW = "needs_review"


class ClaudeCodeTask(BaseModel):
    task_id: str
    type: ClaudeCodeTaskType
    title: str
    prompt: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    target_files: List[str] = Field(default_factory=list)
    allowed_write_dirs: List[str] = Field(default_factory=list)
    permission_mode: str = "plan"
    timeout_sec: int = 600
    human_review_required: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ClaudeCodeResult(BaseModel):
    task_id: str
    type: ClaudeCodeTaskType
    status: ClaudeCodeStatus
    summary: str = ""
    patch_summary: Optional[str] = None
    changed_files: List[str] = Field(default_factory=list)
    tests: List[str] = Field(default_factory=list)
    docs: List[str] = Field(default_factory=list)
    test_commands: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    raw_output_path: Optional[str] = None
    raw_stdout: str = ""
    raw_stderr: str = ""
    human_review_required: bool = True
    finished_at: datetime = Field(default_factory=datetime.utcnow)


__all__ = [
    "ClaudeCodeTask",
    "ClaudeCodeResult",
    "ClaudeCodeTaskType",
    "ClaudeCodeStatus",
]
