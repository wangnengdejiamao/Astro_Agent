"""Path-based read/write permission policy for Claude Code tasks.

This is *advisory* — it does not stop the underlying CLI from accessing the
filesystem. Instead the router and node call ``check_*`` before launching a
task, and the test suite asserts the policy stays in sync with the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Iterable, List, Sequence


# Hard denylist: never read or write these (paths or path prefixes).
DEFAULT_DENY: tuple[str, ...] = (
    ".env",
    "secrets/",
    "data/raw/",
    "configs/production.yaml",
)

# Allowlist for writes. Reads outside this list are still permitted.
DEFAULT_WRITE_ALLOW: tuple[str, ...] = (
    "astro_toolbox/",
    "analysis_agent/",
    "tests/",
    "docs/",
    "papers/drafts/",
    "runs/",
)


class PathDeniedError(PermissionError):
    """Raised when a path is on the denylist or outside the write allowlist."""


def _norm(p: str) -> str:
    return str(PurePosixPath(p.replace("\\", "/")))


def _matches(path: str, prefixes: Sequence[str]) -> bool:
    norm = _norm(path)
    for raw in prefixes:
        pref = _norm(raw)
        if raw.endswith("/"):
            if norm == pref.rstrip("/") or norm.startswith(pref.rstrip("/") + "/"):
                return True
        else:
            if norm == pref or norm.endswith("/" + pref):
                return True
    return False


@dataclass
class PermissionPolicy:
    deny: List[str] = field(default_factory=lambda: list(DEFAULT_DENY))
    write_allow: List[str] = field(default_factory=lambda: list(DEFAULT_WRITE_ALLOW))

    def is_denied(self, path: str) -> bool:
        return _matches(path, self.deny)

    def can_read(self, path: str) -> bool:
        return not self.is_denied(path)

    def can_write(self, path: str) -> bool:
        if self.is_denied(path):
            return False
        return _matches(path, self.write_allow)

    def check_read(self, path: str) -> None:
        if not self.can_read(path):
            raise PathDeniedError(f"read denied: {path}")

    def check_write(self, path: str) -> None:
        if not self.can_write(path):
            raise PathDeniedError(f"write denied: {path}")

    def filter_writable(self, paths: Iterable[str]) -> List[str]:
        return [p for p in paths if self.can_write(p)]


def default_policy() -> PermissionPolicy:
    return PermissionPolicy()


__all__ = [
    "PermissionPolicy",
    "PathDeniedError",
    "default_policy",
    "DEFAULT_DENY",
    "DEFAULT_WRITE_ALLOW",
]
