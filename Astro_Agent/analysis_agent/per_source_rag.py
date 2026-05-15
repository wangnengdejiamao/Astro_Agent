"""Per-source RAG.

Build a small full-text search index over the references SIMBAD returned for
THIS target.  Each row is one abstract + bibcode + title + year, optionally
augmented by source_research_pipeline's mentions extractor.  The drafter
queries this index instead of (or in addition to) the global RAG, so its
citations are about the SOURCE, not the broad domain.

This is the minimum-viable Plan A2: no PDF download yet, but every
\citep{<bibcode>} the drafter writes has a chance to actually be a paper
that studies this source.

The index uses SQLite with FTS5 if available; otherwise it falls back to
a simple LIKE-based search.  Both code paths return the same dict shape.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_refs (
    bibcode TEXT PRIMARY KEY,
    year    INTEGER,
    title   TEXT,
    journal TEXT,
    abstract TEXT,
    mentions_target INTEGER,        -- 1 = abstract mentions a target alias
    n_param_extractions INTEGER     -- how many published_params rows came from this paper
);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS source_refs_fts USING fts5(
    bibcode UNINDEXED,
    title,
    abstract,
    tokenize = 'unicode61'
);
"""


def _has_fts5(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE __probe__ USING fts5(x);").fetchall()
        conn.execute("DROP TABLE __probe__;")
        return True
    except sqlite3.OperationalError:
        return False


def _read_simbad_references_txt(path: Path) -> List[Dict[str, Any]]:
    """Reuse the parser from published_params (kept self-contained to avoid a
    circular import between modules during early node execution)."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    entries: List[Dict[str, Any]] = []
    blocks = re.split(r"\n=+\n", text)
    for block in blocks:
        head = re.search(r"^\s*\[(\d+)\]\s+(\S+)\s*$", block, flags=re.MULTILINE)
        if not head:
            continue
        bibcode = head.group(2)
        title_match = re.search(r"Title:\s*(.+?)(?=\n\s*Authors:|\Z)", block, flags=re.DOTALL)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
        journal_match = re.search(r"Journal:\s*(.+?)(?=\n\s*ADS:|\n\s*Abstract:|\Z)", block, flags=re.DOTALL)
        journal = re.sub(r"\s+", " ", journal_match.group(1)).strip() if journal_match else ""
        abstract_match = re.search(r"Abstract:\s*(.+?)(?=\n=+|\Z)", block, flags=re.DOTALL)
        abstract = re.sub(r"\s+", " ", abstract_match.group(1)).strip() if abstract_match else ""
        ymatch = re.match(r"^(\d{4})", bibcode or "")
        entries.append({
            "bibcode": bibcode,
            "year": int(ymatch.group(1)) if ymatch else None,
            "title": title,
            "journal": journal,
            "abstract": abstract,
        })
    return entries


def _build_aliases(target: Optional[str]) -> List[str]:
    if not target:
        return []
    raw = str(target).strip()
    aliases = {raw, raw.replace(" ", "")}
    m = re.search(r"J(\d{4,8})([+\-]\d{4,8})", raw.replace(" ", ""))
    if m:
        ra_part = m.group(1)
        dec_part = m.group(2)
        aliases.update({
            "J" + ra_part[:4] + dec_part[:5],
            "J" + ra_part[:4],
            ra_part[:4] + dec_part[:5],
            ra_part[:4],
        })
    return [a for a in aliases if len(a) >= 4]


def build_source_rag(
    *,
    astrotool_root: Path,
    target: Optional[str],
    sqlite_path: Path,
) -> Dict[str, Any]:
    """Build the per-source SQLite RAG from SIMBAD references."""
    refs = _read_simbad_references_txt(astrotool_root / "simbad_references.txt")
    aliases = _build_aliases(target)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()
    conn = sqlite3.connect(str(sqlite_path))
    try:
        conn.executescript(_SCHEMA)
        fts_ok = _has_fts5(conn)
        if fts_ok:
            conn.executescript(_FTS_SCHEMA)
        n_with_target = 0
        for ref in refs:
            mentions = any(a.lower() in (ref.get("abstract") or "").lower() for a in aliases)
            if mentions:
                n_with_target += 1
            conn.execute(
                "INSERT OR REPLACE INTO source_refs VALUES (?,?,?,?,?,?,0)",
                (
                    ref.get("bibcode"),
                    ref.get("year"),
                    ref.get("title"),
                    ref.get("journal"),
                    ref.get("abstract"),
                    1 if mentions else 0,
                ),
            )
            if fts_ok:
                conn.execute(
                    "INSERT INTO source_refs_fts (bibcode, title, abstract) VALUES (?,?,?)",
                    (ref.get("bibcode"), ref.get("title"), ref.get("abstract")),
                )
        conn.commit()
    finally:
        conn.close()
    return {
        "sqlite_path": str(sqlite_path),
        "n_refs": len(refs),
        "n_refs_mentioning_target": n_with_target,
        "target_aliases": aliases,
        "fts5_enabled": fts_ok if refs else False,
    }


def search_source_rag(
    sqlite_path: Path,
    queries: Iterable[str],
    *,
    limit_per_query: int = 4,
    prefer_target_matches: bool = True,
) -> List[Dict[str, Any]]:
    """Run keyword queries against the per-source RAG."""
    if not Path(sqlite_path).exists():
        return []
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    out: List[Dict[str, Any]] = []
    try:
        # Detect FTS availability
        try:
            conn.execute("SELECT 1 FROM source_refs_fts LIMIT 1").fetchone()
            fts_ok = True
        except sqlite3.OperationalError:
            fts_ok = False
        seen: set = set()
        for q in queries:
            q = (q or "").strip()
            if not q:
                continue
            rows: List[sqlite3.Row] = []
            if fts_ok:
                try:
                    rows = conn.execute(
                        "SELECT sr.* FROM source_refs sr "
                        "JOIN source_refs_fts fts ON sr.bibcode = fts.bibcode "
                        "WHERE source_refs_fts MATCH ? "
                        f"ORDER BY sr.mentions_target DESC, sr.year DESC LIMIT {int(limit_per_query)}",
                        (q,),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            if not rows:
                like = "%" + q + "%"
                rows = conn.execute(
                    "SELECT * FROM source_refs WHERE title LIKE ? OR abstract LIKE ? "
                    "ORDER BY mentions_target DESC, year DESC LIMIT ?",
                    (like, like, int(limit_per_query)),
                ).fetchall()
            for row in rows:
                bib = row["bibcode"]
                if bib in seen:
                    continue
                seen.add(bib)
                out.append({
                    "bibcode": bib,
                    "year": row["year"],
                    "title": row["title"],
                    "journal": row["journal"],
                    "abstract": row["abstract"],
                    "mentions_target": bool(row["mentions_target"]),
                    "matched_query": q,
                })
    finally:
        conn.close()
    if prefer_target_matches:
        out.sort(key=lambda r: (0 if r.get("mentions_target") else 1, r.get("year") or 0), )
        # Restore year-desc within each bucket
        out.sort(key=lambda r: (0 if r.get("mentions_target") else 1, -(r.get("year") or 0)))
    return out


__all__ = ["build_source_rag", "search_source_rag"]
