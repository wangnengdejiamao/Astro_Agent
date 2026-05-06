"""Claude Code toolbox: scheduled engineering / writing / review augmentation.

Sits alongside analysis_agent, astro_toolbox, RAG, KG, PaperOrchestra and the
QA Gate. Never replaces them — only adds optional task types that the router
can dispatch when a gap, error, or refinement opportunity is detected.
"""

from .schemas import (
    ClaudeCodeTask,
    ClaudeCodeResult,
    ClaudeCodeTaskType,
    ClaudeCodeStatus,
)
from .client import ClaudeCodeClient, HeadlessClaudeBackend, SDKClaudeBackend
from .tasks import build_task, TASK_REGISTRY

__all__ = [
    "ClaudeCodeTask",
    "ClaudeCodeResult",
    "ClaudeCodeTaskType",
    "ClaudeCodeStatus",
    "ClaudeCodeClient",
    "HeadlessClaudeBackend",
    "SDKClaudeBackend",
    "build_task",
    "TASK_REGISTRY",
]
