"""Backend-agnostic client for Claude Code tasks.

The client is responsible for:
- enforcing :mod:`claude_code_toolbox.safety.permissions`
- writing every raw run to ``runs/<run_id>/claude_code/<task_id>/``
- normalizing backend output into a :class:`ClaudeCodeResult`
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from .runners import headless_runner, sdk_runner
from .safety.output_guard import extract_changed_files
from .safety.permissions import PermissionPolicy, default_policy
from .schemas import (
    ClaudeCodeResult,
    ClaudeCodeStatus,
    ClaudeCodeTask,
    ClaudeCodeTaskType,
)


class ClaudeBackend(Protocol):
    name: str

    def run(self, task: ClaudeCodeTask) -> Dict[str, Any]: ...


class HeadlessClaudeBackend:
    name = "headless"

    def __init__(self, cwd: Optional[Path] = None) -> None:
        self.cwd = cwd

    def run(self, task: ClaudeCodeTask) -> Dict[str, Any]:
        return headless_runner.run_headless(
            task.prompt,
            cwd=self.cwd,
            timeout=task.timeout_sec,
            permission_mode=task.permission_mode,
        )


class SDKClaudeBackend:
    name = "sdk"

    def __init__(self, cwd: Optional[Path] = None) -> None:
        self.cwd = cwd

    def run(self, task: ClaudeCodeTask) -> Dict[str, Any]:
        return sdk_runner.run_sdk(
            task.prompt,
            cwd=self.cwd,
            timeout=task.timeout_sec,
            permission_mode=task.permission_mode,
        )


_BACKEND_STATUS_TO_RESULT = {
    "ok": ClaudeCodeStatus.OK,
    "error": ClaudeCodeStatus.ERROR,
    "timeout": ClaudeCodeStatus.ERROR,
    "missing_binary": ClaudeCodeStatus.SKIPPED,
    "disabled": ClaudeCodeStatus.SKIPPED,
    "not_implemented": ClaudeCodeStatus.SKIPPED,
}


class ClaudeCodeClient:
    """High-level facade used by the LangGraph node."""

    def __init__(
        self,
        backend: Optional[ClaudeBackend] = None,
        *,
        runs_root: Optional[Path] = None,
        run_id: Optional[str] = None,
        policy: Optional[PermissionPolicy] = None,
    ) -> None:
        self.backend: ClaudeBackend = backend or HeadlessClaudeBackend()
        self.runs_root = Path(runs_root) if runs_root else Path("runs")
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.policy = policy or default_policy()

    # ------------------------------------------------------------------
    def _output_dir(self, task: ClaudeCodeTask) -> Path:
        d = self.runs_root / self.run_id / "claude_code" / task.task_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _check_targets(self, task: ClaudeCodeTask) -> Optional[ClaudeCodeResult]:
        for path in task.target_files:
            if self.policy.is_denied(path):
                return ClaudeCodeResult(
                    task_id=task.task_id,
                    type=task.type,
                    status=ClaudeCodeStatus.BLOCKED,
                    summary=f"target file is on denylist: {path}",
                    risks=[f"denylist hit: {path}"],
                    human_review_required=True,
                )
        return None

    # ------------------------------------------------------------------
    def execute(self, task: ClaudeCodeTask) -> ClaudeCodeResult:
        blocked = self._check_targets(task)
        if blocked is not None:
            return blocked

        out_dir = self._output_dir(task)
        raw = self.backend.run(task)

        raw_path = out_dir / "raw.json"
        try:
            raw_path.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

        backend_status = raw.get("status", "error")
        status = _BACKEND_STATUS_TO_RESULT.get(backend_status, ClaudeCodeStatus.ERROR)

        stdout = raw.get("stdout", "") or ""
        stderr = raw.get("stderr", "") or ""

        changed = extract_changed_files(stdout)
        # filter writes the policy would block
        risky = [p for p in changed if self.policy.is_denied(p)]
        changed_clean = [p for p in changed if not self.policy.is_denied(p)]

        risks: list[str] = []
        if risky:
            risks.append(f"backend reported writes on denylisted paths: {risky}")
            status = ClaudeCodeStatus.NEEDS_REVIEW
        if task.human_review_required and status == ClaudeCodeStatus.OK:
            status = ClaudeCodeStatus.NEEDS_REVIEW

        summary = self._extract_summary(stdout) or stderr[-400:]

        return ClaudeCodeResult(
            task_id=task.task_id,
            type=task.type,
            status=status,
            summary=summary,
            changed_files=changed_clean,
            risks=risks,
            raw_output_path=str(raw_path),
            raw_stdout=stdout[-4000:],
            raw_stderr=stderr[-2000:],
            human_review_required=task.human_review_required,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_summary(stdout: str) -> str:
        if not stdout:
            return ""
        try:
            data = json.loads(stdout)
            if isinstance(data, dict):
                for key in ("result", "summary", "text", "message"):
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()[:1200]
        except Exception:
            pass
        return stdout.strip()[:1200]


def make_default_client(run_id: Optional[str] = None) -> ClaudeCodeClient:
    backend_name = os.getenv("ASTRO_CLAUDE_BACKEND", "headless").lower()
    backend: ClaudeBackend = (
        SDKClaudeBackend() if backend_name == "sdk" else HeadlessClaudeBackend()
    )
    return ClaudeCodeClient(backend=backend, run_id=run_id)


__all__ = [
    "ClaudeCodeClient",
    "ClaudeBackend",
    "HeadlessClaudeBackend",
    "SDKClaudeBackend",
    "make_default_client",
]
