"""Start the Prompt2Graph visualization server with file logging.

This helper is mainly for Windows/pythonw launches, where console output is
otherwise discarded and startup errors become hard to diagnose.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "output" / "white_dwarf_kg"
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.stdout = (LOG_DIR / "frontend_5011.out.log").open("a", encoding="utf-8", buffering=1)
sys.stderr = (LOG_DIR / "frontend_5011.err.log").open("a", encoding="utf-8", buffering=1)

os.chdir(ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception as exc:  # pragma: no cover - best-effort startup logging
    print(f"Warning: failed to load .env: {exc}", file=sys.stderr)

import uvicorn


def main() -> None:
    port = int(os.getenv("FRONTEND_PORT", "5011"))
    uvicorn.run("vis_graph_v1:app", host="127.0.0.1", port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
