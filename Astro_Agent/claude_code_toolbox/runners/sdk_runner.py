"""Placeholder for the Claude Agent SDK backend.

Phase 1 uses the headless CLI. Once the Claude Agent SDK is wired up, replace
the body of ``run_sdk`` with the actual SDK call. The signature must stay
compatible with :func:`headless_runner.run_headless` so the client can swap
backends without touching call sites.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_sdk(
    prompt: str,
    *,
    cwd: Optional[Path] = None,
    timeout: int = 600,
    permission_mode: str = "plan",
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Stubbed SDK runner. Returns a ``not_implemented`` status until wired up."""
    return {
        "status": "not_implemented",
        "reason": "SDKClaudeBackend is reserved for future use; "
                  "set backend='headless' for now.",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "stdout": "",
        "stderr": "",
        "cmd": ["<sdk>"],
    }


__all__ = ["run_sdk"]
