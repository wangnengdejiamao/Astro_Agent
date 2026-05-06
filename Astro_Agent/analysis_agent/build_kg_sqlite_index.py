"""Build a SQLite/FTS index for the local white-dwarf knowledge graph.

The current JSON export is large enough that loading and regex-scanning it for
every source query is slow.  This index gives the agent source-level KG lookup
without repeatedly parsing the 100+ MB JSON file.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KG_JSON = (
    REPO_ROOT
    / "prompt2graph_for_astronomy"
    / "output"
    / "white_dwarf_kg"
    / "production_full"
    / "multi_stage_deduplicated.json"
)
DEFAULT_DB = REPO_ROOT / "prompt2graph_for_astronomy" / "output" / "white_dwarf_kg" / "kg_index.sqlite"


def node_name(node: dict[str, Any] | None) -> str:
    if not node:
        return ""
    props = node.get("properties") or {}
    name = props.get("name") or node.get("name") or ""
    if isinstance(name, list):
        return str(name[0]) if name else ""
    return str(name)


def node_type(node: dict[str, Any] | None) -> str:
    if not node:
        return ""
    props = node.get("properties") or {}
    return str(props.get("schema_type") or node.get("label") or "")


def title_from_edge(edge: dict[str, Any]) -> str:
    for key in ("title", "paper_title"):
        value = edge.get(key)
        if value:
            return str(value)
    for node_key in ("start_node", "end_node"):
        props = (edge.get(node_key) or {}).get("properties") or {}
        if props.get("title"):
            return str(props.get("title"))
    return ""


def build_index(kg_json: Path, db_path: Path) -> dict[str, int | str]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    con.executescript(
        """
        PRAGMA journal_mode = WAL;
        CREATE TABLE kg_edges (
            id INTEGER PRIMARY KEY,
            subject TEXT,
            subject_type TEXT,
            relation TEXT,
            object TEXT,
            object_type TEXT,
            source TEXT,
            evidence TEXT,
            chunk_id TEXT,
            title TEXT,
            raw_json TEXT
        );
        CREATE VIRTUAL TABLE kg_edges_fts USING fts5(
            subject, relation, object, source, evidence, title,
            content='kg_edges', content_rowid='id'
        );
        CREATE INDEX idx_kg_subject ON kg_edges(subject);
        CREATE INDEX idx_kg_object ON kg_edges(object);
        CREATE INDEX idx_kg_relation ON kg_edges(relation);
        """
    )

    data = json.loads(kg_json.read_text(encoding="utf-8"))
    rows = []
    for edge in data:
        if not isinstance(edge, dict):
            continue
        subj = node_name(edge.get("start_node"))
        obj = node_name(edge.get("end_node"))
        rel = str(edge.get("relation") or "")
        if not (subj or obj or rel):
            continue
        rows.append(
            (
                subj,
                node_type(edge.get("start_node")),
                rel,
                obj,
                node_type(edge.get("end_node")),
                str(edge.get("source") or ""),
                str(edge.get("evidence") or ""),
                str(edge.get("chunk_id") or ""),
                title_from_edge(edge),
                json.dumps(edge, ensure_ascii=False),
            )
        )

    con.executemany(
        """
        INSERT INTO kg_edges
        (subject, subject_type, relation, object, object_type, source, evidence, chunk_id, title, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.execute(
        """
        INSERT INTO kg_edges_fts(rowid, subject, relation, object, source, evidence, title)
        SELECT id, subject, relation, object, source, evidence, title FROM kg_edges
        """
    )
    con.commit()
    con.close()
    return {"db": str(db_path), "rows": len(rows), "source_json": str(kg_json)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLite index for KG JSON.")
    parser.add_argument("--kg-json", default=str(DEFAULT_KG_JSON))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()
    result = build_index(Path(args.kg_json), Path(args.db))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
