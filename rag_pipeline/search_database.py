from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "rag_pipeline" / "index" / "white_dwarf_rag.sqlite"


def make_match_query(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")
    if any(operator in query.upper().split() for operator in ("AND", "OR", "NOT", "NEAR")):
        return query
    tokens = re.findall(r"[\w.+-]+", query, flags=re.UNICODE)
    if not tokens:
        return '"' + query.replace('"', '""') + '"'
    return " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def search(args: argparse.Namespace) -> list[dict[str, object]]:
    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    where = ["1 = 1"]
    params: list[object] = [make_match_query(args.query)]

    if args.category:
        where.append(
            """
            EXISTS (
                SELECT 1
                FROM paper_categories pc2
                JOIN categories c2 ON c2.id = pc2.category_id
                WHERE pc2.paper_id = p.id AND c2.name LIKE ?
            )
            """
        )
        params.append(f"%{args.category}%")
    if args.section:
        where.append("LOWER(c.section) LIKE ?")
        params.append(f"%{args.section.lower()}%")
    if args.method_only:
        where.append("c.method_priority = 1")
    if args.year_from is not None:
        where.append("p.year >= ?")
        params.append(args.year_from)
    if args.year_to is not None:
        where.append("p.year <= ?")
        params.append(args.year_to)

    params.append(args.limit)
    sql = f"""
        WITH matches AS (
            SELECT
                rowid,
                bm25(chunks_fts) AS score,
                snippet(chunks_fts, 2, '[', ']', ' ... ', 28) AS snippet
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
        )
        SELECT
            c.id AS chunk_id,
            p.bibcode,
            p.title,
            p.year,
            p.journal,
            c.section,
            c.page_start,
            c.page_end,
            c.method_priority,
            c.instruments_json,
            c.object_ids_json,
            c.methods_json,
            COALESCE((
                SELECT group_concat(name, ' ; ')
                FROM (
                    SELECT DISTINCT c3.name AS name
                    FROM paper_categories pc3
                    JOIN categories c3 ON c3.id = pc3.category_id
                    WHERE pc3.paper_id = p.id
                    ORDER BY c3.name
                )
            ), '') AS categories,
            matches.score AS score,
            matches.snippet AS snippet
        FROM matches
        JOIN chunks c ON c.id = matches.rowid
        JOIN papers p ON p.id = c.paper_id
        WHERE {' AND '.join(where)}
        ORDER BY matches.score
        LIMIT ?
    """

    rows = con.execute(sql, params).fetchall()
    con.close()
    return [dict(row) for row in rows]


def print_text(rows: list[dict[str, object]]) -> None:
    for index, row in enumerate(rows, start=1):
        instruments = json.loads(str(row.get("instruments_json") or "[]"))
        methods = json.loads(str(row.get("methods_json") or "[]"))
        page = ""
        if row.get("page_start"):
            page = f" p.{row['page_start']}"
            if row.get("page_end") and row["page_end"] != row["page_start"]:
                page += f"-{row['page_end']}"
        print(f"{index}. {row['bibcode']} ({row['year']}) [{row['section']}{page}] score={row['score']:.4f}")
        print(f"   {row['title']}")
        print(f"   categories: {row['categories']}")
        if instruments:
            print(f"   instruments: {', '.join(instruments[:10])}")
        if methods:
            print(f"   methods: {', '.join(methods[:10])}")
        print(f"   {row['snippet']}")
        print()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the white-dwarf RAG SQLite database.")
    parser.add_argument("query", help="FTS query, e.g. 'Gaia parallax effective temperature'")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path")
    parser.add_argument("--category", help="Filter by category name substring")
    parser.add_argument("--section", help="Filter by section substring, e.g. Methods or Data Reduction")
    parser.add_argument("--method-only", action="store_true", help="Only return method/data/analysis chunks")
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--year-to", type=int)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    results = search(args)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print_text(results)
