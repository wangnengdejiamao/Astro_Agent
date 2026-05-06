"""Prompt template loader for Claude Code subagents.

Templates live next to this module under ``prompts/`` and inside
``.claude/agents/``. We support a tiny ``{{var}}`` interpolation so we don't
add a templating dependency.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_DIR = _PROJECT_ROOT / ".claude" / "agents"

_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render(template: str, variables: Mapping[str, str]) -> str:
    def _sub(m: re.Match[str]) -> str:
        return str(variables.get(m.group(1), ""))

    return _VAR_RE.sub(_sub, template)


def load_agent_prompt(name: str) -> str:
    """Load a subagent markdown spec by name (e.g. ``astro-paper-refiner``).

    Returns an empty string if the file is missing — callers fall back to an
    inline default prompt so the system stays runnable in stripped checkouts.
    """
    path = _AGENTS_DIR / f"{name}.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


__all__ = ["render", "load_agent_prompt"]
