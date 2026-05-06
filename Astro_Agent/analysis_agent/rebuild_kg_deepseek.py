"""Prepare and run DeepSeek stage1/2/3 KG rebuild.

This wrapper keeps API keys in environment variables only.  It creates a compact
corpus from the local RAG database, then runs prompt2graph staged extraction and
Leiden community clustering with the DeepSeek-compatible LLM adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
P2G_ROOT = REPO_ROOT / "graph_for_astronomy"
RAG_DB = REPO_ROOT / "rag_pipeline" / "index" / "white_dwarf_rag.sqlite"
CORPUS_DIR = P2G_ROOT / "input" / "white_dwarf_kg_deepseek"
CORPUS_PATH = CORPUS_DIR / "corpus_cleaned.json"
CONFIG = P2G_ROOT / "configs" / "white_dwarf_kg_deepseek_stage123.yml"


def require_env() -> None:
    if not os.getenv("LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "DeepSeek API key missing. Set LLM_API_KEY or OPENAI_API_KEY in the environment; do not write keys into repo files."
        )
    os.environ.setdefault("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    os.environ.setdefault("LLM_MODEL", os.getenv("OPENAI_MODEL", "deepseek-v4-pro"))
    os.environ.setdefault("LLM_TEMPERATURE", "0.2")
    os.environ.setdefault("LLM_MAX_TOKENS", "8192")


def build_corpus(limit_papers: int | None = None, method_only: bool = True) -> dict[str, int | str]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(RAG_DB)
    con.row_factory = sqlite3.Row
    where = "WHERE c.source = 'metadata' OR c.method_priority = 1" if method_only else ""
    limit_sql = f"LIMIT {int(limit_papers)}" if limit_papers else ""
    rows = con.execute(
        f"""
        SELECT p.bibcode, p.title, p.year, p.journal, p.abstract,
               group_concat(c.text, '\n\n') AS chunk_text
        FROM papers p
        JOIN chunks c ON c.paper_id = p.id
        {where}
        GROUP BY p.id
        ORDER BY p.year DESC, p.bibcode
        {limit_sql}
        """
    ).fetchall()
    docs = []
    for row in rows:
        text = "\n\n".join(
            part
            for part in [
                f"Bibcode: {row['bibcode']}",
                f"Title: {row['title']}",
                f"Year: {row['year']}",
                f"Journal: {row['journal']}",
                f"Abstract: {row['abstract']}",
                row["chunk_text"] or "",
            ]
            if part
        )
        docs.append({"id": row["bibcode"], "title": row["title"], "text": text[:60000]})
    con.close()
    CORPUS_PATH.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"corpus_path": str(CORPUS_PATH), "documents": len(docs)}


def run_pipeline(output_dir: Path, limit_chunks: int | None = None) -> int:
    cmd = [
        sys.executable,
        "run_end2end_pipeline.py",
        str(CONFIG),
    ]
    # run_end2end_pipeline creates its own timestamped directory from config;
    # run_pipeline_by_stage is better for an explicit existing chunks directory,
    # but the end-to-end script performs chunking and clustering in one call.
    env = os.environ.copy()
    env["PROJECT_DIR"] = str(P2G_ROOT)
    proc = subprocess.run(cmd, cwd=str(P2G_ROOT), env=env, text=True)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild white dwarf KG with DeepSeek stage1/2/3.")
    parser.add_argument("--limit-papers", type=int, default=None, help="Limit corpus size for smoke tests.")
    parser.add_argument("--prepare-only", action="store_true", help="Only create corpus/config inputs.")
    args = parser.parse_args()
    require_env()
    summary = build_corpus(limit_papers=args.limit_papers)
    run_root = P2G_ROOT / "output" / "white_dwarf_kg" / f"deepseek_stage123_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    summary["planned_output_root"] = str(run_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.prepare_only:
        raise SystemExit(run_pipeline(run_root))


if __name__ == "__main__":
    main()
