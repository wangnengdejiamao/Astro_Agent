"""External CLI adapters: Codex CLI and Claude Code CLI as agent tools.

These let the LangGraph workflow delegate code-execution / code-repair / general
engineering tasks to the bundled `vendor/codex-main` and the local Claude Code
CLI without leaving the audit trail. Output is captured as JSON-friendly dicts
that drop straight into the SharedContext.

All commands run with strict timeouts. Secrets are pulled from env (loaded by
`llm_client.load_default_env`). No keys are logged.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_client import load_default_env


DEFAULT_TIMEOUT = int(os.getenv("ASTRO_AGENT_SUBPROC_TIMEOUT", "600"))


def _resolve_cmd(env_var: str, fallback: Optional[str] = None) -> Optional[List[str]]:
    load_default_env()
    raw = os.getenv(env_var) or fallback
    if not raw:
        return None
    return shlex.split(raw)


def _run(cmd: List[str], cwd: Optional[Path], timeout: int, stdin_text: Optional[str]) -> Dict[str, Any]:
    started = datetime.utcnow().isoformat() + "Z"
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ},
        )
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "stdout": proc.stdout[-20000:],
            "stderr": proc.stderr[-8000:],
            "cmd": cmd,
            "started_at": started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "returncode": -1,
            "stdout": (exc.stdout or "")[-20000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            "cmd": cmd,
            "started_at": started,
            "timeout_sec": timeout,
        }
    except FileNotFoundError as exc:
        return {"status": "missing_binary", "error": str(exc), "cmd": cmd, "started_at": started}


def codex_exec(
    prompt: str,
    cwd: Optional[Path] = None,
    timeout: int = DEFAULT_TIMEOUT,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run a one-shot Codex CLI task. Sandboxed, non-interactive."""
    cmd = _resolve_cmd("ASTRO_AGENT_CODEX_BIN")
    if cmd is None:
        return {"status": "disabled", "reason": "ASTRO_AGENT_CODEX_BIN not set"}
    cmd = [*cmd, "exec", "--skip-git-repo-check", "--sandbox", "workspace-write"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(prompt)
    return _run(cmd, cwd=cwd, timeout=timeout, stdin_text=None)


def claude_code_exec(
    prompt: str,
    cwd: Optional[Path] = None,
    timeout: int = DEFAULT_TIMEOUT,
    permission_mode: str = "plan",
) -> Dict[str, Any]:
    """Run a one-shot Claude Code task in headless mode."""
    cmd = _resolve_cmd("ASTRO_AGENT_CLAUDE_BIN", fallback="claude")
    if cmd is None:
        return {"status": "disabled", "reason": "ASTRO_AGENT_CLAUDE_BIN not set"}
    cmd = [*cmd, "--print", "--permission-mode", permission_mode, "--output-format", "json"]
    return _run(cmd, cwd=cwd, timeout=timeout, stdin_text=prompt)


def parse_claude_json(result: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort parser for `claude --output-format json` stdout."""
    out = result.get("stdout") or ""
    try:
        return {**result, "parsed": json.loads(out)}
    except Exception:
        return result


__all__ = ["codex_exec", "claude_code_exec", "parse_claude_json"]
