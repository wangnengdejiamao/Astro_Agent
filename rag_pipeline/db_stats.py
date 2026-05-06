from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "rag_pipeline" / "index" / "white_dwarf_rag.sqlite"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print summary statistics for the white-dwarf RAG database.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path")
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    queries = [
        ("papers", "SELECT count(1) FROM papers"),
        ("categories", "SELECT count(1) FROM categories"),
        ("paper_categories", "SELECT count(1) FROM paper_categories"),
        ("chunks_total", "SELECT count(1) FROM chunks"),
        ("pdf_chunks", "SELECT count(1) FROM chunks WHERE source = 'pdf'"),
        ("abstract_chunks", "SELECT count(1) FROM chunks WHERE source = 'metadata'"),
        ("method_priority_chunks", "SELECT count(1) FROM chunks WHERE method_priority = 1"),
    ]

    for label, sql in queries:
        print(f"{label}: {con.execute(sql).fetchone()[0]}")

    print("\npdf_status:")
    for status, count in con.execute("SELECT pdf_status, count(1) FROM papers GROUP BY pdf_status ORDER BY count(1) DESC"):
        print(f"  {status}: {count}")

    print("\ntop_sections:")
    for section, count in con.execute("SELECT section, count(1) FROM chunks GROUP BY section ORDER BY count(1) DESC LIMIT 12"):
        print(f"  {section}: {count}")

    print("\ncategories:")
    for name, count in con.execute(
        """
        SELECT c.name, count(pc.paper_id) AS n
        FROM categories c
        LEFT JOIN paper_categories pc ON pc.category_id = c.id
        GROUP BY c.id
        ORDER BY n DESC, c.name
        """
    ):
        print(f"  {name}: {count}")

    con.close()


if __name__ == "__main__":
    main()
