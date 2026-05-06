"""Headless Claude Code CLI runner.

Thin wrapper around ``subprocess`` that calls ``claude --print`` (the headless
mode) with a JSON output format. We deliberately do NOT reuse
``analysis_agent.codex_tool.claude_code_exec`` directly so the toolbox can be
imported in test environments where ``analysis_agent`` is unavailable.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_TIMEOUT = int(os.getenv("ASTRO_AGENT_SUBPROC_TIMEOUT", "600"))


def _resolve_cmd() -> List[str]:
    raw = os.getenv("ASTRO_AGENT_CLAUDE_BIN", "claude")
    return shlex.split(raw)


def run_headless(
    prompt: str,
    *,
    cwd: Optional[Path] = None,
    timeout: int = DEFAULT_TIMEOUT,
    permission_mode: str = "plan",
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Invoke ``claude --print`` and capture the result as a dict."""
    started = datetime.utcnow().isoformat() + "Z"
    cmd = _resolve_cmd()
    cmd = [
        *cmd,
        "--print",
        "--permission-mode",
        permission_mode,
        "--output-format",
        "json",
    ]
    if extra_args:
        cmd.extend(extra_args)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ},
        )
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "stdout": proc.stdout[-40000:],
            "stderr": proc.stderr[-8000:],
            "cmd": cmd,
            "started_at": started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "returncode": -1,
            "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "") if isinstance(exc.stderr, str) else "",
            "cmd": cmd,
            "started_at": started,
            "timeout_sec": timeout,
        }
    except FileNotFoundError as exc:
        return {
            "status": "missing_binary",
            "error": str(exc),
            "cmd": cmd,
            "started_at": started,
        }


def parse_json_output(stdout: str) -> Dict[str, Any]:
    try:
        return json.loads(stdout)
    except Exception:
        return {}


__all__ = ["run_headless", "parse_json_output", "DEFAULT_TIMEOUT"]
