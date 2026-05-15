"""Prompt experiment log — DSPy / MIPROv2 foundation.

For every LLM call made by the drafter we record:
  (specialist, section, system_prompt_hash, user_prompt_hash,
   output_hash, output_chars, paper_qc_score_before, paper_qc_score_after,
   reflexion_retry_index, timestamp)

The log accumulates in SQLite under output/analysis_agent/_prompt_experiments.sqlite.

Once we have ≥ 50 rows we can feed (prompt, output, qc_score) into DSPy
MIPROv2 as a labeled dataset to discover better instructions / few-shot
examples automatically (Khattab+ 2024; documented for production use in
the DSPy 2.5 docs).  Until then, the log is a passive recorder.

This module is intentionally cheap: it never gates the workflow, never
raises, and stores only hashes + short metadata.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_runs (
    timestamp           TEXT,
    source_id           TEXT,
    source_class        TEXT,
    specialist          TEXT,
    section             TEXT,
    reflexion_retry_idx INTEGER,
    system_prompt_hash  TEXT,
    user_prompt_hash    TEXT,
    output_hash         TEXT,
    output_chars        INTEGER,
    paper_qc_pass       INTEGER,
    paper_qc_warn       INTEGER,
    paper_qc_fail       INTEGER,
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_prompt_specialist
    ON prompt_runs(specialist, section);
"""


def _h(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def _default_path() -> Path:
    pkg = Path(__file__).resolve().parent.parent
    return pkg / "output" / "analysis_agent" / "_prompt_experiments.sqlite"


def record_call(
    *,
    source_id: Optional[str],
    source_class: Optional[str],
    specialist: str,
    section: str,
    reflexion_retry_idx: int,
    system_prompt: str,
    user_prompt: str,
    output: str,
    paper_qc: Optional[Mapping[str, Any]] = None,
    notes: str = "",
    sqlite_path: Optional[Path] = None,
) -> Optional[str]:
    """Append one row to the prompt-experiment log. Never raises."""
    sqlite_path = sqlite_path or _default_path()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(sqlite_path))
        try:
            conn.executescript(_SCHEMA)
            row = (
                datetime.utcnow().isoformat(timespec="seconds"),
                str(source_id or ""),
                str(source_class or ""),
                specialist,
                section,
                int(reflexion_retry_idx or 0),
                _h(system_prompt),
                _h(user_prompt),
                _h(output),
                len(output or ""),
                int((paper_qc or {}).get("n_pass") or 0),
                int((paper_qc or {}).get("n_warn") or 0),
                int((paper_qc or {}).get("n_fail") or 0),
                notes,
            )
            conn.execute(
                "INSERT INTO prompt_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )
            conn.commit()
            return str(sqlite_path)
        finally:
            conn.close()
    except Exception:
        return None


def summarize(sqlite_path: Optional[Path] = None) -> dict:
    sqlite_path = sqlite_path or _default_path()
    if not Path(sqlite_path).exists():
        return {"status": "empty"}
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT specialist, section, "
            "COUNT(*) AS n, AVG(paper_qc_pass) AS avg_pass, "
            "AVG(paper_qc_fail) AS avg_fail "
            "FROM prompt_runs GROUP BY specialist, section ORDER BY n DESC"
        ).fetchall()
        return {
            "status": "ok",
            "by_specialist_section": [dict(r) for r in rows],
            "total_calls": sum(int(r["n"]) for r in rows),
        }
    finally:
        conn.close()


__all__ = ["record_call", "summarize"]
