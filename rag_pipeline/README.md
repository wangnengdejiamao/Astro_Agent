# White Dwarf RAG Pipeline

This folder contains a local, reproducible RAG database builder for the white-dwarf literature corpus.

## What It Builds

- `papers`: ADS metadata for each unique paper.
- `categories`: the existing Chinese white-dwarf taxonomy from the ADS JSON exports.
- `paper_categories`: many-to-many category assignments, preserving cross-topic papers.
- `chunks`: metadata abstract chunks plus PDF full-text chunks.
- `chunks_fts`: SQLite FTS5 index for BM25-style keyword retrieval.

The default output is:

```powershell
rag_pipeline\index\white_dwarf_rag.sqlite
```

## Build

```powershell
python rag_pipeline\build_database.py --rebuild
```

For a quick metadata-only build:

```powershell
python rag_pipeline\build_database.py --rebuild --metadata-only
```

To resume PDF parsing:

```powershell
python rag_pipeline\build_database.py --resume
```

## Search

```powershell
python rag_pipeline\search_database.py "Gaia parallax effective temperature" --method-only --limit 5
python rag_pipeline\search_database.py "cyclotron spectral fitting" --category "磁白矮星" --method-only
python rag_pipeline\search_database.py "SDSS J" --category "激变变星与吸积" --limit 10
```

## Notes

This first local version uses PyMuPDF for PDF extraction because Marker/Nougat are not installed in the current environment. The schema is designed so embedding vectors or an external vector store can be added later while keeping the same paper/category/chunk identifiers.
