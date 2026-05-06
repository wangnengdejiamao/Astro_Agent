"""Launch the local graph frontend as a detached Windows process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    creationflags = 0
    if sys.platform.startswith("win"):
        create_breakaway_from_job = 0x01000000
        creationflags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
            | create_breakaway_from_job
        )

    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "run_frontend_server.py")],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=creationflags,
    )
    print(proc.pid)


if __name__ == "__main__":
    main()
