"""ADS live literature query (graceful fallback when API key absent).

Reads ADS_DEV_KEY (or ADS_API_KEY) from the environment and queries the
NASA ADS Bumblebee API.  Supports two query modes:

  * positional cone search (target ra/dec)
  * keyword search ("ZTF J213056.71+442046.5")

Returns a list of {bibcode, title, year, authors, abstract} records that
can be merged into the per-source RAG sqlite for the drafter to cite.

If no API key is set, returns status="no_api_key" without raising — the
workflow continues with the SIMBAD-cached references it already has.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional


_ADS_BASE = "https://api.adsabs.harvard.edu/v1/search/query"


def _get_key() -> Optional[str]:
    return os.getenv("ADS_DEV_KEY") or os.getenv("ADS_API_KEY")


def query_ads(
    *,
    target: Optional[str] = None,
    ra_deg: Optional[float] = None,
    dec_deg: Optional[float] = None,
    search_radius_deg: float = 0.005,
    year_min: int = 2018,
    rows: int = 30,
    timeout_sec: int = 30,
) -> Dict[str, Any]:
    """Run an ADS query.  Returns dict {status, n_papers, papers: [...]}.

    Strategy:
      1. If `ra_deg` and `dec_deg` are supplied, do a positional search
         `pos(ra, dec, radius)` joined with `database:astronomy`.
      2. Otherwise do a keyword search on `target`.
      3. Filter to `year >= year_min` to keep responses focused on recent work.
    """
    key = _get_key()
    if not key:
        return {
            "status": "no_api_key",
            "note": "Set ADS_DEV_KEY in environment to enable live ADS queries.",
        }
    try:
        import requests  # type: ignore
    except ImportError:
        return {"status": "requests_not_installed"}
    q_parts: List[str] = []
    if ra_deg is not None and dec_deg is not None:
        q_parts.append(f"pos(circle {ra_deg} {dec_deg} {search_radius_deg})")
    elif target:
        q_parts.append(f'full:"{target}"')
    else:
        return {"status": "no_query_inputs"}
    q_parts.append(f"year:[{year_min} TO 9999]")
    q_parts.append("database:astronomy")
    q = " AND ".join(q_parts)

    params = {
        "q": q,
        "fl": "bibcode,title,author,year,abstract,doi",
        "rows": rows,
        "sort": "date desc",
    }
    headers = {"Authorization": f"Bearer {key}"}
    try:
        resp = requests.get(_ADS_BASE, params=params, headers=headers,
                            timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    docs = (data.get("response") or {}).get("docs") or []
    papers: List[Dict[str, Any]] = []
    for doc in docs:
        papers.append({
            "bibcode": doc.get("bibcode"),
            "title": (doc.get("title") or [""])[0],
            "authors": doc.get("author") or [],
            "year": int(doc.get("year") or 0) or None,
            "abstract": doc.get("abstract") or "",
            "doi": (doc.get("doi") or [""])[0] if doc.get("doi") else "",
        })
    return {
        "status": "ok",
        "query": q,
        "n_papers": len(papers),
        "papers": papers,
    }


def merge_into_source_rag(
    *,
    sqlite_path: str,
    papers: List[Mapping[str, Any]],
    target_aliases: List[str],
) -> Dict[str, Any]:
    """Insert ADS-fetched papers into the per-source RAG sqlite.  Uses the
    same schema as per_source_rag.build_source_rag."""
    import sqlite3
    if not papers:
        return {"status": "no_papers"}
    conn = sqlite3.connect(sqlite_path)
    n_added = 0
    n_mentions = 0
    try:
        for p in papers:
            bib = p.get("bibcode")
            if not bib:
                continue
            mentions = any(a.lower() in (p.get("abstract") or "").lower()
                           for a in target_aliases)
            if mentions:
                n_mentions += 1
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO source_refs "
                    "VALUES (?, ?, ?, ?, ?, ?, 0)",
                    (
                        bib,
                        p.get("year"),
                        p.get("title"),
                        "",  # journal not in our query
                        p.get("abstract"),
                        1 if mentions else 0,
                    ),
                )
                if conn.total_changes:
                    n_added += 1
            except sqlite3.OperationalError:
                continue
        conn.commit()
    finally:
        conn.close()
    return {"status": "ok", "n_papers_added": n_added, "n_mentioning_target": n_mentions}


__all__ = ["query_ads", "merge_into_source_rag"]
