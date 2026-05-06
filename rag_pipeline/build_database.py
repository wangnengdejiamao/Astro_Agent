from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "literature" / "ads_complete_20260416" / "COMPLETE_DATASET.json"
DEFAULT_PDF_DIR = ROOT / "literature" / "pdfs"
DEFAULT_DB = ROOT / "rag_pipeline" / "index" / "white_dwarf_rag.sqlite"

SKIP_CATEGORY_FILES = {
    "ALL_PAPERS",
    "all_papers",
    "SUPPLEMENT_PAPERS",
    "COMPLETE_DATASET",
}

SECTION_ALIASES = [
    ("Abstract", re.compile(r"^(abstract)\b", re.I)),
    ("Introduction", re.compile(r"^(?:\d+(?:\.\d+)*\.?\s*)?(introduction|background)\b", re.I)),
    (
        "Observations",
        re.compile(
            r"^(?:\d+(?:\.\d+)*\.?\s*)?(observations?|observational data|data|sample selection)\b",
            re.I,
        ),
    ),
    (
        "Data Reduction",
        re.compile(r"^(?:\d+(?:\.\d+)*\.?\s*)?(data reduction|reduction|calibration)\b", re.I),
    ),
    (
        "Methods",
        re.compile(
            r"^(?:\d+(?:\.\d+)*\.?\s*)?(methods?|methodology|analysis|model(?:ling|ing)?|models?|"
            r"spectral analysis|photometric analysis|asteroseismic analysis|fitting procedure)\b",
            re.I,
        ),
    ),
    ("Results", re.compile(r"^(?:\d+(?:\.\d+)*\.?\s*)?(results?|measurements?)\b", re.I)),
    ("Discussion", re.compile(r"^(?:\d+(?:\.\d+)*\.?\s*)?(discussion)\b", re.I)),
    (
        "Conclusion",
        re.compile(r"^(?:\d+(?:\.\d+)*\.?\s*)?(conclusions?|summary|summary and conclusions)\b", re.I),
    ),
    ("Acknowledgments", re.compile(r"^(acknowledg(e)?ments?)\b", re.I)),
    ("References", re.compile(r"^(references|bibliography)\b", re.I)),
]

INSTRUMENT_PATTERNS = {
    "2MASS": r"\b2MASS\b",
    "APOGEE": r"\bAPOGEE\b",
    "ASAS-SN": r"\bASAS[-\s]?SN\b",
    "CFHT": r"\bCFHT\b",
    "Chandra": r"\bChandra\b",
    "CRTS": r"\bCRTS\b",
    "DES": r"\bDES\b",
    "DESI": r"\bDESI\b",
    "FUSE": r"\bFUSE\b",
    "GALEX": r"\bGALEX\b",
    "Gaia": r"\bGaia(?:\s+(?:DR1|DR2|DR3|EDR3))?\b",
    "Gemini": r"\bGemini\b",
    "GTC": r"\bGTC\b|\bGran Telescopio Canarias\b",
    "HARPS": r"\bHARPS\b",
    "HET": r"\bHET\b|\bHobby[-\s]Eberly\b",
    "HST": r"\bHST\b|\bHubble Space Telescope\b",
    "JWST": r"\bJWST\b|\bJames Webb Space Telescope\b",
    "K2": r"\bK2\b",
    "Kepler": r"\bKepler\b",
    "LAMOST": r"\bLAMOST\b|\bGuoshoujing\b",
    "LCO": r"\bLCO\b|\bLas Cumbres\b",
    "LSST": r"\bLSST\b|\bRubin Observatory\b",
    "NOIRLab": r"\bNOIRLab\b",
    "NTT": r"\bNTT\b|\bNew Technology Telescope\b",
    "Pan-STARRS": r"\bPan[-\s]?STARRS\b|\bPS1\b",
    "RAVE": r"\bRAVE\b",
    "ROSAT": r"\bROSAT\b",
    "SDSS": r"\bSDSS\b|\bSloan Digital Sky Survey\b",
    "SOAR": r"\bSOAR\b",
    "Spitzer": r"\bSpitzer\b",
    "Subaru": r"\bSubaru\b",
    "Swift": r"\bSwift\b",
    "TESS": r"\bTESS\b",
    "UKIDSS": r"\bUKIDSS\b",
    "VISTA": r"\bVISTA\b",
    "VLT": r"\bVLT\b|\bVery Large Telescope\b",
    "WEAVE": r"\bWEAVE\b",
    "WISE": r"\bWISE\b|\bAllWISE\b",
    "WHT": r"\bWHT\b|\bWilliam Herschel Telescope\b",
    "XMM-Newton": r"\bXMM[-\s]?Newton\b",
    "X-Shooter": r"\bX[-\s]?Shooter\b",
    "ZTF": r"\bZTF\b|\bZwicky Transient Facility\b",
}

METHOD_PATTERNS = {
    "MCMC": r"\bMCMC\b|\bMarkov Chain Monte Carlo\b",
    "Bayesian inference": r"\bBayesian\b",
    "Koester atmosphere models": r"\bKoester\b",
    "TLUSTY": r"\bTLUSTY\b",
    "MESA": r"\bMESA\b",
    "LPCODE": r"\bLPCODE\b",
    "Montreal cooling models": r"\bMontreal\b",
    "cooling sequence": r"\bcooling sequence\b|\bcooling track",
    "spectral fitting": r"\bspectral fit(?:ting)?\b|\bspectrum fit(?:ting)?\b",
    "SED fitting": r"\bSED fit(?:ting)?\b|\bspectral energy distribution\b",
    "photometry": r"\bphotometr(?:y|ic)\b",
    "spectroscopy": r"\bspectroscop(?:y|ic)\b",
    "astrometry": r"\bastrometr(?:y|ic)\b",
    "parallax": r"\bparallax\b",
    "radial velocity": r"\bradial velocit(?:y|ies)\b",
    "light-curve modelling": r"\blight[-\s]?curve\b",
    "periodogram": r"\bperiodogram\b|\bLomb[-\s]?Scargle\b",
    "asteroseismology": r"\basteroseismolog(?:y|ical)\b",
    "population synthesis": r"\bpopulation synthesis\b",
    "common envelope": r"\bcommon envelope\b",
    "cyclotron spectral fitting": r"\bcyclotron\b",
}

OBJECT_PATTERNS = [
    re.compile(r"\bSDSS\s?J\d{4,6}[+-]\d{4,6}(?:\.\d+)?\b", re.I),
    re.compile(r"\bZTF\s?J\d{6}(?:\.\d+)?[+-]\d{6}(?:\.\d+)?\b", re.I),
    re.compile(r"\bGaia\s+(?:DR1|DR2|DR3|EDR3)\s+\d{8,}\b", re.I),
    re.compile(r"\bWD\s?J?\d{4}[+-]\d{3,4}\b", re.I),
    re.compile(r"\bJ\d{4,6}[+-]\d{4,6}(?:\.\d+)?\b"),
]

REFERENCE_SECTIONS = {"References", "Acknowledgments"}
METHOD_SECTIONS = {"Methods", "Data Reduction", "Observations"}


@dataclass
class ParsedPdf:
    bibcode: str
    status: str
    chunks: list[dict[str, Any]]
    page_count: int = 0
    error: str | None = None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_year(value: Any) -> int | None:
    if value in (None, ""):
        return None
    match = re.search(r"\d{4}", str(value))
    return int(match.group(0)) if match else None


def json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False, sort_keys=True)


def split_categories(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    else:
        items = re.split(r"[,;|]", str(value or ""))
    return [str(item).strip() for item in items if str(item).strip() and str(item).strip().lower() != "none"]


def paper_key(paper: dict[str, Any]) -> str:
    bibcode = str(paper.get("bibcode") or "").strip()
    if bibcode:
        return bibcode
    raw = json.dumps(paper, ensure_ascii=False, sort_keys=True)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def merge_paper(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if key == "category":
            continue
        if merged.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
            merged[key] = value
    return merged


def load_papers_and_categories(metadata_path: Path, literature_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, dict[str, Any]]]:
    metadata = read_json(metadata_path)
    papers_blob = metadata.get("papers", metadata if isinstance(metadata, list) else [])
    papers: dict[str, dict[str, Any]] = {}
    paper_categories: dict[str, set[str]] = {}
    categories: dict[str, dict[str, Any]] = {}

    for group_name, source in (("main_categories", "main"), ("supplement_categories", "supplement")):
        for name, expected_count in (metadata.get(group_name) or {}).items():
            categories.setdefault(name, {"name": name, "source": source, "expected_count": expected_count, "file_path": None})

    for paper in papers_blob:
        key = paper_key(paper)
        papers[key] = merge_paper(papers.get(key, {}), paper)
        for category in split_categories(paper.get("category")):
            paper_categories.setdefault(key, set()).add(category)
            categories.setdefault(category, {"name": category, "source": "metadata", "expected_count": None, "file_path": None})

    for path in literature_dir.rglob("*.json"):
        if path.stem in SKIP_CATEGORY_FILES:
            continue
        # The English files are an older duplicate export. Prefer the Chinese
        # category files that match the user's current taxonomy.
        if "ads_papers_20260415_150007" in str(path):
            continue
        category = path.stem
        categories.setdefault(category, {"name": category, "source": "category_file", "expected_count": None, "file_path": str(path)})
        categories[category]["file_path"] = str(path)
        try:
            data = read_json(path)
        except Exception as exc:
            print(f"warning: cannot read category file {path}: {exc}", file=sys.stderr)
            continue
        entries = data.get("papers", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        for paper in entries:
            key = paper_key(paper)
            papers[key] = merge_paper(papers.get(key, {}), paper)
            paper_categories.setdefault(key, set()).add(category)

    return papers, paper_categories, categories


def extract_arxiv_ids(paper: dict[str, Any]) -> set[str]:
    raw_parts = []
    raw_parts.extend(str(item) for item in as_list(paper.get("doi")))
    raw_parts.extend(str(item) for item in as_list(paper.get("identifier")))
    raw_parts.append(str(paper.get("bibcode") or ""))
    raw = " ".join(raw_parts)
    ids = set()
    for match in re.finditer(r"arXiv[.:/ ]+(\d{4}\.\d{4,5})(?:v\d+)?", raw, re.I):
        ids.add(match.group(1))
    for match in re.finditer(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b", raw):
        ids.add(match.group(1))
    return ids


def map_pdfs(pdf_dir: Path, papers: dict[str, dict[str, Any]]) -> dict[str, Path]:
    pdfs = {path.stem: path for path in pdf_dir.glob("*.pdf")}
    by_arxiv = {stem: path for stem, path in pdfs.items() if re.match(r"^\d{4}\.\d{4,5}$", stem)}
    result: dict[str, Path] = {}

    for key, paper in papers.items():
        bibcode = str(paper.get("bibcode") or key)
        if bibcode in pdfs:
            result[key] = pdfs[bibcode]
            continue
        for arxiv_id in extract_arxiv_ids(paper):
            if arxiv_id in by_arxiv:
                result[key] = by_arxiv[arxiv_id]
                break

    return result


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    return con


def create_schema(con: sqlite3.Connection, rebuild: bool) -> None:
    if rebuild:
        con.executescript(
            """
            DROP TABLE IF EXISTS chunks_fts;
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS paper_categories;
            DROP TABLE IF EXISTS categories;
            DROP TABLE IF EXISTS papers;
            DROP TABLE IF EXISTS build_meta;
            """
        )

    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY,
            bibcode TEXT UNIQUE NOT NULL,
            title TEXT,
            year INTEGER,
            journal TEXT,
            abstract TEXT,
            authors_json TEXT NOT NULL,
            keywords_json TEXT NOT NULL,
            doi_json TEXT NOT NULL,
            citations INTEGER,
            arxiv_ids_json TEXT NOT NULL,
            pdf_path TEXT,
            pdf_status TEXT DEFAULT 'pending',
            pdf_pages INTEGER,
            pdf_error TEXT,
            source_json TEXT
        );

        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            source TEXT,
            expected_count INTEGER,
            file_path TEXT
        );

        CREATE TABLE IF NOT EXISTS paper_categories (
            paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
            PRIMARY KEY (paper_id, category_id)
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            section TEXT,
            chunk_type TEXT NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            text TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            token_estimate INTEGER NOT NULL,
            instruments_json TEXT NOT NULL,
            object_ids_json TEXT NOT NULL,
            methods_json TEXT NOT NULL,
            method_priority INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL,
            UNIQUE (paper_id, chunk_index, source)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            title,
            abstract,
            body,
            category,
            authors,
            instruments,
            object_ids,
            methods,
            section,
            tokenize = 'unicode61 remove_diacritics 2'
        );

        CREATE TABLE IF NOT EXISTS build_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
        CREATE INDEX IF NOT EXISTS idx_papers_bibcode ON papers(bibcode);
        CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);
        CREATE INDEX IF NOT EXISTS idx_chunks_method ON chunks(method_priority);
        CREATE INDEX IF NOT EXISTS idx_paper_categories_category ON paper_categories(category_id);
        """
    )


def insert_metadata(
    con: sqlite3.Connection,
    papers: dict[str, dict[str, Any]],
    paper_categories: dict[str, set[str]],
    categories: dict[str, dict[str, Any]],
    pdf_map: dict[str, Path],
    metadata_path: Path,
) -> dict[str, int]:
    for category in sorted(categories.values(), key=lambda item: item["name"]):
        con.execute(
            """
            INSERT INTO categories(name, source, expected_count, file_path)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                source = excluded.source,
                expected_count = COALESCE(excluded.expected_count, categories.expected_count),
                file_path = COALESCE(excluded.file_path, categories.file_path)
            """,
            (category["name"], category.get("source"), category.get("expected_count"), category.get("file_path")),
        )

    category_ids = {row[1]: row[0] for row in con.execute("SELECT id, name FROM categories")}
    paper_ids: dict[str, int] = {}

    for key, paper in papers.items():
        bibcode = str(paper.get("bibcode") or key)
        authors = as_list(paper.get("authors"))
        keywords = as_list(paper.get("keywords"))
        doi = as_list(paper.get("doi"))
        arxiv_ids = sorted(extract_arxiv_ids(paper))
        pdf_path = pdf_map.get(key)
        con.execute(
            """
            INSERT INTO papers(
                bibcode, title, year, journal, abstract, authors_json, keywords_json, doi_json,
                citations, arxiv_ids_json, pdf_path, pdf_status, source_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bibcode) DO UPDATE SET
                title = excluded.title,
                year = excluded.year,
                journal = excluded.journal,
                abstract = excluded.abstract,
                authors_json = excluded.authors_json,
                keywords_json = excluded.keywords_json,
                doi_json = excluded.doi_json,
                citations = excluded.citations,
                arxiv_ids_json = excluded.arxiv_ids_json,
                pdf_path = excluded.pdf_path,
                pdf_status = excluded.pdf_status,
                source_json = excluded.source_json
            """,
            (
                bibcode,
                paper.get("title"),
                normalize_year(paper.get("year")),
                paper.get("journal"),
                paper.get("abstract"),
                json_dumps(authors),
                json_dumps(keywords),
                json_dumps(doi),
                paper.get("citations") or 0,
                json_dumps(arxiv_ids),
                str(pdf_path) if pdf_path else None,
                "pending" if pdf_path else "missing_pdf",
                str(metadata_path),
            ),
        )
        paper_id = con.execute("SELECT id FROM papers WHERE bibcode = ?", (bibcode,)).fetchone()[0]
        paper_ids[key] = paper_id
        for category in sorted(paper_categories.get(key) or split_categories(paper.get("category"))):
            category_id = category_ids.get(category)
            if category_id is None:
                continue
            con.execute(
                "INSERT OR IGNORE INTO paper_categories(paper_id, category_id) VALUES (?, ?)",
                (paper_id, category_id),
            )

    con.commit()
    return paper_ids


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def canonical_section(line: str) -> str | None:
    candidate = re.sub(r"\s+", " ", line.strip())
    candidate = candidate.strip(".:-")
    if not candidate or len(candidate) > 120:
        return None
    for name, pattern in SECTION_ALIASES:
        if pattern.search(candidate):
            return name
    return None


def split_into_sections(pages: list[str]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current = {"section": "Full Text", "page_start": 1, "page_end": 1, "lines": []}

    for page_number, page_text in enumerate(pages, start=1):
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                current["lines"].append("")
                continue
            section = canonical_section(line)
            if section and len(current["lines"]) > 8:
                current["page_end"] = page_number
                sections.append(current)
                current = {"section": section, "page_start": page_number, "page_end": page_number, "lines": []}
            else:
                current["lines"].append(line)
                current["page_end"] = page_number

    if current["lines"]:
        sections.append(current)

    compact: list[dict[str, Any]] = []
    for section in sections:
        text = clean_text("\n".join(section["lines"]))
        if text:
            compact.append(
                {
                    "section": section["section"],
                    "page_start": section["page_start"],
                    "page_end": section["page_end"],
                    "text": text,
                }
            )
    return compact


def window_text(text: str, max_chars: int, overlap_chars: int) -> Iterable[str]:
    text = text.strip()
    if len(text) <= max_chars:
        yield text
        return

    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            paragraph_break = text.rfind("\n\n", start + max_chars // 2, end)
            sentence_break = max(text.rfind(". ", start + max_chars // 2, end), text.rfind("; ", start + max_chars // 2, end))
            cut = paragraph_break if paragraph_break != -1 else sentence_break
            if cut != -1:
                end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= length:
            break
        start = max(0, end - overlap_chars)


def find_terms(text: str, patterns: dict[str, str]) -> list[str]:
    hits = []
    for name, pattern in patterns.items():
        if re.search(pattern, text, re.I):
            hits.append(name)
    return hits


def find_objects(text: str, limit: int = 80) -> list[str]:
    hits: list[str] = []
    seen = set()
    for pattern in OBJECT_PATTERNS:
        for match in pattern.finditer(text):
            value = re.sub(r"\s+", " ", match.group(0).strip())
            key = value.lower()
            if key not in seen:
                hits.append(value)
                seen.add(key)
            if len(hits) >= limit:
                return hits
    return hits


def make_chunk(
    text: str,
    section: str,
    chunk_type: str,
    page_start: int | None,
    page_end: int | None,
    source: str,
) -> dict[str, Any]:
    instruments = find_terms(text, INSTRUMENT_PATTERNS)
    methods = find_terms(text, METHOD_PATTERNS)
    method_priority = int(section in METHOD_SECTIONS or bool(methods))
    return {
        "section": section,
        "chunk_type": chunk_type,
        "page_start": page_start,
        "page_end": page_end,
        "text": text,
        "char_count": len(text),
        "token_estimate": max(1, len(text) // 4),
        "instruments": instruments,
        "object_ids": find_objects(text),
        "methods": methods,
        "method_priority": method_priority,
        "source": source,
    }


def parse_pdf_task(args: tuple[str, str, int, bool]) -> ParsedPdf:
    bibcode, pdf_path, max_chars, include_references = args
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover
        return ParsedPdf(bibcode=bibcode, status="no_parser", chunks=[], error=str(exc))

    try:
        doc = fitz.open(pdf_path)
        pages = [clean_text(page.get_text("text")) for page in doc]
        page_count = len(pages)
        doc.close()
    except Exception as exc:
        return ParsedPdf(bibcode=bibcode, status="pdf_error", chunks=[], error=str(exc))

    sections = split_into_sections(pages)
    chunks: list[dict[str, Any]] = []
    for section in sections:
        section_name = section["section"]
        if section_name in REFERENCE_SECTIONS and not include_references:
            continue
        for piece in window_text(section["text"], max_chars=max_chars, overlap_chars=500):
            if len(piece) < 120:
                continue
            chunks.append(
                make_chunk(
                    piece,
                    section=section_name,
                    chunk_type="pdf_section",
                    page_start=section["page_start"],
                    page_end=section["page_end"],
                    source="pdf",
                )
            )

    return ParsedPdf(bibcode=bibcode, status="parsed", chunks=chunks, page_count=page_count)


def insert_chunk(
    con: sqlite3.Connection,
    paper_id: int,
    chunk_index: int,
    chunk: dict[str, Any],
    title: str,
    abstract: str,
    authors: list[str],
    categories: list[str],
) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO chunks(
            paper_id, chunk_index, section, chunk_type, page_start, page_end, text,
            char_count, token_estimate, instruments_json, object_ids_json, methods_json,
            method_priority, source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_id,
            chunk_index,
            chunk["section"],
            chunk["chunk_type"],
            chunk.get("page_start"),
            chunk.get("page_end"),
            chunk["text"],
            chunk["char_count"],
            chunk["token_estimate"],
            json_dumps(chunk["instruments"]),
            json_dumps(chunk["object_ids"]),
            json_dumps(chunk["methods"]),
            chunk["method_priority"],
            chunk["source"],
        ),
    )
    chunk_id = con.execute(
        "SELECT id FROM chunks WHERE paper_id = ? AND chunk_index = ? AND source = ?",
        (paper_id, chunk_index, chunk["source"]),
    ).fetchone()[0]
    con.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
    con.execute(
        """
        INSERT INTO chunks_fts(
            rowid, title, abstract, body, category, authors, instruments, object_ids, methods, section
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            title or "",
            abstract or "",
            chunk["text"],
            " ; ".join(categories),
            " ; ".join(authors),
            " ; ".join(chunk["instruments"]),
            " ; ".join(chunk["object_ids"]),
            " ; ".join(chunk["methods"]),
            chunk["section"] or "",
        ),
    )


def insert_abstract_chunks(con: sqlite3.Connection) -> None:
    rows = list(
        con.execute(
            """
            SELECT p.id, p.title, p.abstract, p.authors_json,
                   COALESCE(group_concat(c.name, ' ; '), '') AS categories
            FROM papers p
            LEFT JOIN paper_categories pc ON pc.paper_id = p.id
            LEFT JOIN categories c ON c.id = pc.category_id
            GROUP BY p.id
            ORDER BY p.id
            """
        )
    )
    for paper_id, title, abstract, authors_json, category_blob in rows:
        if not abstract:
            continue
        authors = json.loads(authors_json or "[]")
        categories = [item.strip() for item in (category_blob or "").split(";") if item.strip()]
        chunk = make_chunk(
            clean_text(str(abstract)),
            section="Abstract",
            chunk_type="metadata_abstract",
            page_start=None,
            page_end=None,
            source="metadata",
        )
        insert_chunk(con, paper_id, 0, chunk, title or "", abstract or "", authors, categories)
    con.commit()


def papers_to_parse(con: sqlite3.Connection, limit: int | None, resume: bool) -> list[tuple[str, str, int, list[str], str, str, list[str]]]:
    where = "WHERE p.pdf_path IS NOT NULL"
    if resume:
        where += " AND (p.pdf_status IS NULL OR p.pdf_status NOT IN ('parsed', 'pdf_error', 'no_parser'))"
    query = f"""
        SELECT p.bibcode, p.pdf_path, p.id, p.authors_json, p.title, p.abstract,
               COALESCE(group_concat(c.name, ' ; '), '') AS categories
        FROM papers p
        LEFT JOIN paper_categories pc ON pc.paper_id = p.id
        LEFT JOIN categories c ON c.id = pc.category_id
        {where}
        GROUP BY p.id
        ORDER BY p.year, p.bibcode
    """
    rows = []
    for row in con.execute(query):
        bibcode, pdf_path, paper_id, authors_json, title, abstract, category_blob = row
        rows.append((bibcode, pdf_path, paper_id, json.loads(authors_json or "[]"), title or "", abstract or "", [item.strip() for item in (category_blob or "").split(";") if item.strip()]))
        if limit and len(rows) >= limit:
            break
    return rows


def build_database(args: argparse.Namespace) -> None:
    metadata_path = Path(args.metadata).resolve()
    pdf_dir = Path(args.pdf_dir).resolve()
    db_path = Path(args.out).resolve()

    literature_dir = metadata_path.parents[1]
    print(f"metadata: {metadata_path}")
    print(f"pdf_dir : {pdf_dir}")
    print(f"db      : {db_path}")

    papers, paper_categories, categories = load_papers_and_categories(metadata_path, literature_dir)
    pdf_map = map_pdfs(pdf_dir, papers)
    print(f"loaded {len(papers):,} unique papers, {len(categories):,} categories, matched {len(pdf_map):,} PDFs")

    con = connect(db_path)
    create_schema(con, rebuild=args.rebuild)
    insert_metadata(con, papers, paper_categories, categories, pdf_map, metadata_path)
    insert_abstract_chunks(con)

    con.execute("INSERT OR REPLACE INTO build_meta(key, value) VALUES (?, ?)", ("metadata_path", str(metadata_path)))
    con.execute("INSERT OR REPLACE INTO build_meta(key, value) VALUES (?, ?)", ("pdf_dir", str(pdf_dir)))
    con.execute("INSERT OR REPLACE INTO build_meta(key, value) VALUES (?, ?)", ("built_at_epoch", str(int(time.time()))))
    con.commit()

    if args.metadata_only:
        con.close()
        print("metadata-only build complete")
        return

    rows = papers_to_parse(con, limit=args.limit, resume=args.resume)
    print(f"PDF parse queue: {len(rows):,} papers")
    if not rows:
        con.close()
        return

    by_bibcode = {row[0]: row for row in rows}
    parse_args = [(row[0], row[1], args.max_chars, args.include_references) for row in rows]
    started = time.time()
    done = 0
    total_chunks = 0

    if args.workers <= 1 or args.executor == "sequential":
        result_iter = (parse_pdf_task(item) for item in parse_args)
    else:
        executor_cls = concurrent.futures.ProcessPoolExecutor if args.executor == "process" else concurrent.futures.ThreadPoolExecutor
        executor = executor_cls(max_workers=args.workers)
        futures = [executor.submit(parse_pdf_task, item) for item in parse_args]
        result_iter = (future.result() for future in concurrent.futures.as_completed(futures))

    try:
        for result in result_iter:
            row = by_bibcode[result.bibcode]
            _, _, paper_id, authors, title, abstract, categories_for_paper = row

            con.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE paper_id = ? AND source = 'pdf')", (paper_id,))
            con.execute("DELETE FROM chunks WHERE paper_id = ? AND source = 'pdf'", (paper_id,))

            chunk_index = 1
            for chunk in result.chunks:
                insert_chunk(
                    con,
                    paper_id=paper_id,
                    chunk_index=chunk_index,
                    chunk=chunk,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    categories=categories_for_paper,
                )
                chunk_index += 1

            con.execute(
                "UPDATE papers SET pdf_status = ?, pdf_pages = ?, pdf_error = ? WHERE id = ?",
                (result.status, result.page_count, result.error, paper_id),
            )
            done += 1
            total_chunks += len(result.chunks)
            if done % args.commit_every == 0:
                con.commit()
            if done == 1 or done % args.progress_every == 0 or done == len(rows):
                elapsed = max(1, time.time() - started)
                rate = done / elapsed
                print(
                    f"parsed {done:,}/{len(rows):,} PDFs | {total_chunks:,} chunks | "
                    f"{rate:.2f} pdf/s | last={result.bibcode} status={result.status}",
                    flush=True,
                )
    finally:
        if "executor" in locals():
            executor.shutdown(wait=True)

    con.commit()
    con.execute("INSERT OR REPLACE INTO build_meta(key, value) VALUES (?, ?)", ("pdfs_parsed_last_run", str(done)))
    con.execute("INSERT OR REPLACE INTO build_meta(key, value) VALUES (?, ?)", ("pdf_chunks_last_run", str(total_chunks)))
    con.commit()
    con.close()
    print("build complete")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a white-dwarf literature RAG database with metadata, PDF chunks, and FTS5.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Path to COMPLETE_DATASET.json")
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR), help="Directory containing PDFs")
    parser.add_argument("--out", default=str(DEFAULT_DB), help="Output SQLite database")
    parser.add_argument("--rebuild", action="store_true", help="Drop and rebuild all tables")
    parser.add_argument("--resume", action="store_true", help="Skip PDFs already parsed in the output database")
    parser.add_argument("--metadata-only", action="store_true", help="Only ingest metadata and abstracts")
    parser.add_argument("--include-references", action="store_true", help="Keep references and acknowledgments sections")
    parser.add_argument("--limit", type=int, default=None, help="Parse at most N PDFs")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)), help="PDF parser workers")
    parser.add_argument("--executor", choices=("thread", "process", "sequential"), default="thread", help="PDF parser executor")
    parser.add_argument("--max-chars", type=int, default=5500, help="Approximate max characters per chunk")
    parser.add_argument("--commit-every", type=int, default=50, help="Commit after this many parsed PDFs")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress after this many parsed PDFs")
    return parser.parse_args(argv)


if __name__ == "__main__":
    build_database(parse_args(sys.argv[1:]))
