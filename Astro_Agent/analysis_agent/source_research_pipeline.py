"""End-to-end source research package builder.

This script is the stricter per-source workflow requested for astronomy papers:

1. Run or ingest a fresh astro_toolbox output directory.
2. Query SIMBAD references for the source and download every referenced PDF
   that can be resolved through ADS/arXiv.
3. Read local RAG/metadata and KG relations for the source.
4. Re-analyze spectra, HST, SED, and variability products.
5. Produce structured JSON artifacts for the paper drafter.

It intentionally writes evidence artifacts first; the paper drafter should only
write claims traceable to these files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


REPO_ROOT = Path(__file__).resolve().parents[2]
ASTRO_AGENT_DIR = REPO_ROOT / "Astro_Agent"
ASTRO_TOOLBOX_DIR = ASTRO_AGENT_DIR / "astro_toolbox"
RAG_DB = REPO_ROOT / "rag_pipeline" / "index" / "white_dwarf_rag.sqlite"
KG_WORKSPACE = Path(os.getenv("ASTRO_AGENT_KG_WORKSPACE", str(REPO_ROOT / ".local_kg")))
KG_INDEX = KG_WORKSPACE / "output" / "white_dwarf_kg" / "kg_index.sqlite"
METADATA_JSON = REPO_ROOT / "literature" / "ads_complete_20260416" / "COMPLETE_DATASET.json"
PDF_DIR = REPO_ROOT / "literature" / "pdfs"


ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> str:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def write_text(path: Path, text: str) -> str:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_astrotool(target: str, ra: float, dec: float, output_root: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ASTRO_TOOLBOX_DIR / "run_single_target_all_tools.py"),
        "--target",
        target,
        "--ra",
        f"{ra:.10f}",
        "--dec",
        f"{dec:.10f}",
        "--output-root",
        str(output_root),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=7200)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-12000:],
        "stderr_tail": proc.stderr[-6000:],
        "output_root": str(output_root),
        "module_status": read_csv(output_root / "module_status.csv"),
        "run_summary": json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
        if (output_root / "run_summary.json").exists()
        else {},
    }


def query_simbad_all_refs(ra: float, dec: float, max_refs: int = 10000) -> dict[str, Any]:
    sys.path.insert(0, str(ASTRO_AGENT_DIR))
    from astro_toolbox import utils

    try:
        refs = utils.query_simbad_references(ra, dec, max_refs=max_refs)
        return refs or {"status": "empty", "references": [], "n_refs": 0}
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "references": [],
            "n_refs": 0,
        }


def load_metadata_by_bibcode() -> dict[str, dict[str, Any]]:
    if not METADATA_JSON.exists():
        return {}
    data = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("papers", [])
    result = {}
    for row in rows:
        bib = str(row.get("bibcode") or "").strip()
        if bib:
            result[bib] = row
    return result


def enrich_refs_from_local_metadata(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata = load_metadata_by_bibcode()
    enriched = []
    for ref in refs:
        bib = str(ref.get("bibcode") or "").strip()
        merged = dict(metadata.get(bib, {}))
        merged.update({k: v for k, v in ref.items() if v not in ("", None, [], {})})
        if bib:
            merged["bibcode"] = bib
        enriched.append(merged)
    return enriched


def read_local_simbad_references(run_roots: Path | list[Path]) -> dict[str, Any]:
    """Fallback to astro_toolbox-exported SIMBAD references when online query is empty."""
    csv_path = locate_file(run_roots, ["simbad_references.csv"])
    txt_path = locate_file(run_roots, ["simbad_references.txt"])
    rows = read_csv(csv_path) if csv_path else []
    refs: list[dict[str, Any]] = []
    for row in rows:
        bibcode = str(row.get("bibcode") or row.get("Bibcode") or "").strip()
        if not bibcode:
            continue
        refs.append(
            {
                "bibcode": bibcode,
                "year": row.get("year") or row.get("Year") or bibcode[:4],
                "title": row.get("title") or row.get("Title") or "",
                "authors": row.get("authors") or row.get("author") or row.get("Authors") or "",
                "journal": row.get("journal") or row.get("pub") or row.get("Journal") or "",
                "url": row.get("url") or row.get("URL") or "",
                "has_abstract": row.get("has_abstract") or "",
            }
        )
    return {
        "status": "local_fallback" if refs else "empty",
        "source_csv": str(csv_path) if csv_path else "",
        "source_txt": str(txt_path) if txt_path else "",
        "n_refs": len(refs),
        "references": refs,
    }


def _safe_pdf_name(bibcode: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", bibcode.strip())


def _pdf_path_for_bibcode(bibcode: str) -> Path:
    return PDF_DIR / f"{_safe_pdf_name(bibcode)}.pdf"


def _normalize_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _arxiv_id_from_bibcode(bibcode: str) -> str:
    match = re.search(r"arXiv(\d{2})(\d{2})(\d{5})", bibcode)
    if match:
        return f"{match.group(1)}{match.group(2)}.{match.group(3)}"
    return ""


def _arxiv_id_from_ref(ref: dict[str, Any]) -> str:
    for key in ("arxiv", "eprint", "identifier", "url", "abstract", "title"):
        text = str(ref.get(key) or "")
        match = re.search(r"arXiv[:\s]*([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", text, re.I)
        if match:
            return match.group(1)
    return _arxiv_id_from_bibcode(str(ref.get("bibcode") or ""))


def _download_binary(url: str, output_path: Path, timeout: int = 90) -> tuple[bool, str]:
    ensure_dir(output_path.parent)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Astro_Agent source research PDF resolver"},
    )
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
        data_head = tmp_path.read_bytes()[:8]
        if not data_head.startswith(b"%PDF"):
            tmp_path.unlink(missing_ok=True)
            return False, "downloaded file is not a PDF"
        tmp_path.replace(output_path)
        return True, f"downloaded {url}"
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"


def _query_arxiv_by_title(title: str, timeout: int = 45) -> dict[str, str]:
    if not title.strip():
        return {}
    query = urllib.parse.urlencode(
        {
            "search_query": f'ti:"{title.strip()}"',
            "start": "0",
            "max_results": "3",
        }
    )
    url = f"https://export.arxiv.org/api/query?{query}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            xml_text = resp.read().decode("utf-8", errors="replace")
        root = ET.fromstring(xml_text)
        target_norm = _normalize_title(title)
        for entry in root.findall("atom:entry", ARXIV_NS):
            entry_title = " ".join((entry.findtext("atom:title", default="", namespaces=ARXIV_NS) or "").split())
            entry_norm = _normalize_title(entry_title)
            if not entry_norm:
                continue
            if target_norm and (target_norm in entry_norm or entry_norm in target_norm):
                entry_id = entry.findtext("atom:id", default="", namespaces=ARXIV_NS) or ""
                arxiv_id = entry_id.rstrip("/").split("/")[-1]
                return {"arxiv_id": arxiv_id, "title": entry_title, "api_url": url}
    except Exception:
        return {}
    return {}


def _download_ref_pdf_fallback(ref: dict[str, Any]) -> dict[str, Any]:
    bibcode = str(ref.get("bibcode") or "").strip()
    pdf_path = _pdf_path_for_bibcode(bibcode)
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return {
            "bibcode": bibcode,
            "downloaded": True,
            "info": "already exists",
            "pdf_path": str(pdf_path),
            "resolver": "local_cache",
        }

    arxiv_id = _arxiv_id_from_ref(ref)
    resolver = "arxiv_id"
    if not arxiv_id:
        # arXiv asks clients to avoid rapid-fire API calls; this path is only
        # used when the local ADS/arXiv downloader is unavailable.
        time.sleep(3.0)
        hit = _query_arxiv_by_title(str(ref.get("title") or ""))
        arxiv_id = hit.get("arxiv_id", "")
        resolver = "arxiv_title" if arxiv_id else "unresolved"

    if not arxiv_id:
        return {
            "bibcode": bibcode,
            "downloaded": False,
            "info": "no arXiv PDF resolved from bibcode/title",
            "pdf_path": "",
            "resolver": resolver,
        }

    ok, info = _download_binary(f"https://arxiv.org/pdf/{arxiv_id}", pdf_path)
    return {
        "bibcode": bibcode,
        "downloaded": bool(ok),
        "info": info,
        "pdf_path": str(pdf_path) if pdf_path.exists() else "",
        "resolver": resolver,
        "arxiv_id": arxiv_id,
    }


def download_simbad_pdfs(refs: list[dict[str, Any]], max_workers: int = 4) -> dict[str, Any]:
    """Download all SIMBAD bibcode PDFs using the local ADS/arXiv downloader."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import retry_failed
    except Exception as exc:
        retry_failed = None
        import_error = f"{type(exc).__name__}: {exc}"
    else:
        import_error = ""

    ref_by_bibcode = {
        str(r.get("bibcode") or "").strip(): r
        for r in refs
        if str(r.get("bibcode") or "").strip()
    }
    bibcodes = sorted(ref_by_bibcode, reverse=True)
    rows = []
    if not bibcodes:
        return {"status": "empty", "rows": rows}

    def worker(bib: str) -> dict[str, Any]:
        if retry_failed is None:
            return _download_ref_pdf_fallback(ref_by_bibcode[bib])
        ok, info = retry_failed.download_pdf(bib)
        return {
            "bibcode": bib,
            "downloaded": bool(ok),
            "info": info,
            "pdf_path": str(_pdf_path_for_bibcode(bib)) if _pdf_path_for_bibcode(bib).exists() else "",
            "resolver": "retry_failed",
        }

    if retry_failed is None:
        for bib in bibcodes:
            try:
                rows.append(worker(bib))
            except Exception as exc:
                rows.append({"bibcode": bib, "downloaded": False, "info": f"{type(exc).__name__}: {exc}", "pdf_path": ""})
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(worker, bib): bib for bib in bibcodes}
            for fut in as_completed(futures):
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    rows.append({"bibcode": futures[fut], "downloaded": False, "info": f"{type(exc).__name__}: {exc}", "pdf_path": ""})
    rows.sort(key=lambda r: r["bibcode"], reverse=True)
    return {
        "status": "ok" if retry_failed is not None else "ok_fallback_arxiv",
        "fallback_reason": import_error,
        "n_total": len(bibcodes),
        "n_available_pdf": sum(1 for r in rows if r.get("pdf_path")),
        "rows": rows,
    }


def pdf_text(path: Path, max_pages: int = 8) -> str:
    try:
        import fitz

        doc = fitz.open(path)
        pages = []
        for idx in range(min(len(doc), max_pages)):
            pages.append(doc[idx].get_text("text"))
        return "\n".join(pages)
    except Exception:
        try:
            proc = subprocess.run(
                ["pdftotext", "-l", str(max_pages), str(path), "-"],
                text=True,
                capture_output=True,
                timeout=60,
            )
            return proc.stdout
        except Exception:
            return ""


def source_mentions(refs: list[dict[str, Any]], identifiers: list[str]) -> list[dict[str, Any]]:
    patterns = [re.compile(re.escape(x), re.I) for x in identifiers if x]
    rows = []
    for ref in refs:
        bib = str(ref.get("bibcode") or "")
        chunks = []
        text_fields = "\n".join(str(ref.get(k) or "") for k in ("title", "abstract", "summary"))
        for pat in patterns:
            if pat.search(text_fields):
                chunks.append(text_fields[:1600])
                break
        pdf_path = _pdf_path_for_bibcode(bib)
        if pdf_path.exists():
            text = pdf_text(pdf_path)
            for pat in patterns:
                m = pat.search(text)
                if m:
                    lo = max(0, m.start() - 700)
                    hi = min(len(text), m.end() + 1200)
                    chunks.append(re.sub(r"\s+", " ", text[lo:hi]).strip())
                    break
        rows.append(
            {
                "bibcode": bib,
                "title": ref.get("title") or ref.get("Title") or "",
                "year": ref.get("year") or ref.get("Year") or bib[:4],
                "journal": ref.get("journal") or ref.get("pub") or "",
                "authors": ref.get("authors") or ref.get("author") or "",
                "doi": ref.get("doi") or ref.get("doi_json") or "",
                "pdf_available": pdf_path.exists(),
                "source_mentions": chunks[:3],
                "mentions_source": bool(chunks),
            }
        )
    return rows


def _identifier_variants(name: str) -> list[str]:
    clean = " ".join(str(name or "").strip().split())
    if not clean:
        return []
    variants = {clean, clean.replace(" ", "")}
    if clean.upper().startswith("ZTF J"):
        variants.add("ZTFJ" + clean[5:])
    if clean.upper().startswith("ZTFJ"):
        variants.add("ZTF J" + clean[4:])

    match = re.search(r"J(\d{2})(\d{2})(\d{2}(?:\.\d+)?)([+-])(\d{2})(\d{2})(\d{2}(?:\.\d+)?)", clean)
    if match:
        hh, mm, _ss, sign, dd, dm, _ds = match.groups()
        short = f"J{hh}{mm}{sign}{dd}{dm}"
        variants.update({short, f"ZTF {short}", f"ZTF{short}"})
        long_id = clean[match.start():match.end()]
        variants.update({long_id, f"ZTF {long_id}", f"ZTF{long_id}"})
    return sorted(variants, key=lambda x: (len(x), x), reverse=True)


def query_simbad_object_ids(main_id: str) -> list[str]:
    if not main_id:
        return []
    try:
        from astroquery.simbad import Simbad

        table = Simbad.query_objectids(main_id)
        if table is None:
            return []
        col = "ID" if "ID" in table.colnames else table.colnames[0]
        return [str(row[col]).strip() for row in table if str(row[col]).strip()]
    except Exception:
        return []


def build_source_identifiers(target: str, simbad: dict[str, Any], ra: float, dec: float) -> list[str]:
    identifiers: list[str] = []

    def add(value: str) -> None:
        for variant in _identifier_variants(value):
            if variant and variant not in identifiers:
                identifiers.append(variant)

    add(target)
    add(str(simbad.get("main_id") or ""))
    for object_id in query_simbad_object_ids(str(simbad.get("main_id") or target)):
        add(object_id)

    coord_tag = f"RA={ra:.6f}, Dec={dec:.6f}"
    if coord_tag not in identifiers:
        identifiers.append(coord_tag)
    return identifiers[:160]


def rag_exact_bibcodes(bibcodes: Iterable[str]) -> list[dict[str, Any]]:
    bibcodes = [b for b in bibcodes if b]
    if not bibcodes or not RAG_DB.exists():
        return []
    con = sqlite3.connect(RAG_DB)
    con.row_factory = sqlite3.Row
    rows = []
    for bib in bibcodes:
        paper = con.execute("SELECT * FROM papers WHERE bibcode = ?", (bib,)).fetchone()
        if not paper:
            continue
        chunks = con.execute(
            """
            SELECT section, page_start, page_end, text, instruments_json, object_ids_json, methods_json, method_priority
            FROM chunks
            WHERE paper_id = ?
            ORDER BY method_priority DESC, id
            LIMIT 12
            """,
            (paper["id"],),
        ).fetchall()
        rows.append(
            {
                "bibcode": bib,
                "title": paper["title"],
                "year": paper["year"],
                "journal": paper["journal"],
                "pdf_status": paper["pdf_status"],
                "chunks": [dict(c) for c in chunks],
            }
        )
    con.close()
    return rows


def kg_source_relations(identifiers: list[str], limit: int = 200) -> list[dict[str, Any]]:
    if not KG_INDEX.exists():
        return []
    con = sqlite3.connect(KG_INDEX)
    con.row_factory = sqlite3.Row
    seen = set()
    rows = []
    for ident in identifiers:
        if not ident:
            continue
        like = f"%{ident}%"
        query_rows = con.execute(
            """
            SELECT subject, subject_type, relation, object, object_type, title, source, evidence, chunk_id
            FROM kg_edges
            WHERE subject LIKE ? OR object LIKE ? OR source LIKE ? OR evidence LIKE ?
            LIMIT ?
            """,
            (like, like, like, like, limit),
        ).fetchall()
        for row in query_rows:
            key = (row["subject"], row["relation"], row["object"], row["chunk_id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(row))
            if len(rows) >= limit:
                con.close()
                return rows
    con.close()
    return rows


def normalize_roots(run_roots: Path | list[Path]) -> list[Path]:
    if isinstance(run_roots, list):
        return [root for root in run_roots if root and root.exists()]
    return [run_roots] if run_roots.exists() else []


def locate_file(run_roots: Path | list[Path], names: list[str]) -> Optional[Path]:
    for root in normalize_roots(run_roots):
        for name in names:
            hits = list(root.rglob(name))
            if hits:
                return hits[0]
    return None


def load_spectrum_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv(path)
    if not rows:
        return np.array([]), np.array([]), np.array([])
    wave_col = "wavelength_A" if "wavelength_A" in rows[0] else "wavelength"
    flux_col = "flux"
    err_col = "error" if "error" in rows[0] else "flux_err"
    wave = np.array([float(r[wave_col]) for r in rows if r.get(wave_col)], dtype=float)
    flux = np.array([float(r[flux_col]) for r in rows if r.get(wave_col)], dtype=float)
    err = np.array([float(r.get(err_col) or "nan") for r in rows if r.get(wave_col)], dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux)
    if len(err) == len(wave):
        good &= np.isfinite(err) | (err == 0)
    return wave[good], flux[good], err[good] if len(err) == len(wave) else np.full(np.count_nonzero(good), np.nan)


def gaussian(x: np.ndarray, amp: float, center: float, sigma: float, offset: float) -> np.ndarray:
    return offset + amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _line_fit_uncertainties(pcov: np.ndarray, amp: float, sigma: float, rest: float) -> dict[str, float]:
    if pcov is None or np.shape(pcov) != (4, 4) or not np.all(np.isfinite(pcov)):
        return {
            "amplitude_err_norm": float("nan"),
            "center_err_A": float("nan"),
            "sigma_err_A": float("nan"),
            "equivalent_width_err_A": float("nan"),
            "velocity_err_kms": float("nan"),
        }
    perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
    amp_err, center_err, sigma_err, _offset_err = [float(v) for v in perr]
    root2pi = math.sqrt(2 * math.pi)
    # EW = -amp * sigma * sqrt(2pi). Keep covariance between amp and sigma.
    grad = np.array([-sigma * root2pi, 0.0, -amp * root2pi, 0.0], dtype=float)
    ew_var = float(grad @ pcov @ grad.T)
    ew_err = math.sqrt(max(ew_var, 0.0)) if np.isfinite(ew_var) else float("nan")
    return {
        "amplitude_err_norm": amp_err,
        "center_err_A": center_err,
        "sigma_err_A": sigma_err,
        "equivalent_width_err_A": ew_err,
        "velocity_err_kms": center_err / rest * 299792.458 if rest else float("nan"),
    }


def _rejection_reason(
    kind: str,
    amp: float,
    amp_err: float,
    velocity: float,
    cen: float,
    rest: float,
    boundary_fit: bool,
) -> str:
    if kind != "emission":
        return "absorption_or_no_emission"
    if boundary_fit:
        return "fit_parameter_at_bound"
    if abs(velocity) > 500.0 or abs(cen - rest) > 10.0:
        return "line_center_inconsistent_with_rest_wavelength"
    if np.isfinite(amp_err) and amp_err > 0 and abs(amp / amp_err) < 3.0:
        return "emission_amplitude_below_3sigma"
    if abs(amp) < 0.05:
        return "emission_amplitude_below_floor"
    return ""


def fit_lines_for_spectrum(path: Path, out_dir: Path, label: str) -> dict[str, Any]:
    wave, flux, err = load_spectrum_csv(path)
    line_defs = [
        ("Halpha", 6562.8),
        ("Hbeta", 4861.33),
        ("Hgamma", 4340.47),
        ("Hdelta", 4101.74),
        ("H-epsilon/CaIIH", 3968.47),
        ("HeI4471", 4471.5),
        ("HeII4686", 4685.7),
        ("HeI5876", 5875.6),
        ("HeI6678", 6678.2),
    ]
    rows = []
    if len(wave) < 20:
        return {"status": "no_data", "rows": rows}
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for ax, (name, rest) in zip(axes.ravel(), line_defs):
        mask = (wave > rest - 70) & (wave < rest + 70)
        if np.count_nonzero(mask) < 10:
            ax.set_visible(False)
            continue
        x = wave[mask]
        y = flux[mask]
        side = (np.abs(x - rest) > 30) & (np.abs(x - rest) < 70)
        if np.count_nonzero(side) < 5:
            cont = np.nanmedian(y)
        else:
            cont = np.nanmedian(y[side])
        if not np.isfinite(cont) or cont == 0:
            cont = 1.0
        yn = y / cont
        if np.count_nonzero(side) < 5:
            cont_rms = float(np.nanstd(yn - np.nanmedian(yn)))
        else:
            side_norm = y[side] / cont
            cont_rms = float(np.nanstd(side_norm - np.nanmedian(side_norm)))
        sigma_y = None
        if len(err) == len(wave):
            errn = err[mask] / abs(cont)
            good_err = np.isfinite(errn) & (errn > 0)
            if np.count_nonzero(good_err) >= max(5, int(0.5 * len(errn))):
                sigma_y = np.where(good_err, errn, np.nanmedian(errn[good_err]))
        amp0 = float(np.nanmin(yn) - 1.0)
        try:
            popt, pcov = curve_fit(
                gaussian,
                x,
                yn,
                p0=[amp0, rest, 8.0, 1.0],
                bounds=([-2.0, rest - 20, 0.8, 0.4], [2.0, rest + 20, 40.0, 1.6]),
                sigma=sigma_y,
                absolute_sigma=bool(sigma_y is not None),
                maxfev=20000,
            )
            model = gaussian(x, *popt)
            amp, cen, sigma, offset = [float(v) for v in popt]
            ew = -amp * abs(sigma) * math.sqrt(2 * math.pi)
            velocity = (cen / rest - 1.0) * 299792.458
            resid = yn - model
            dof = max(1, len(x) - len(popt))
            if sigma_y is not None:
                fit_reduced_chi2 = float(np.nansum((resid / sigma_y) ** 2) / dof)
            else:
                fit_reduced_chi2 = float(np.nansum(resid**2) / dof)
            unc = _line_fit_uncertainties(pcov, amp, abs(sigma), rest)
            kind = "emission" if amp > 0 else "absorption"
            boundary_fit = (
                abs(cen - (rest - 20.0)) < 0.2
                or abs(cen - (rest + 20.0)) < 0.2
                or abs(abs(sigma) - 40.0) < 0.2
                or abs(abs(sigma) - 0.8) < 0.05
            )
            amp_err = unc["amplitude_err_norm"]
            rejection_reason = _rejection_reason(kind, amp, amp_err, velocity, cen, rest, boundary_fit)
            robust_emission = bool(
                kind == "emission"
                and amp >= 0.05
                and (not np.isfinite(amp_err) or amp_err <= 0 or amp / amp_err >= 3.0)
                and not boundary_fit
                and abs(velocity) <= 500.0
                and abs(cen - rest) <= 10.0
            )
            rows.append(
                {
                    "survey": label,
                    "line": name,
                    "rest_A": rest,
                    "kind": kind,
                    "robust_emission": robust_emission,
                    "boundary_fit": boundary_fit,
                    "amplitude_norm": amp,
                    "amplitude_err_norm": unc["amplitude_err_norm"],
                    "center_A": cen,
                    "center_err_A": unc["center_err_A"],
                    "sigma_A": abs(sigma),
                    "sigma_err_A": unc["sigma_err_A"],
                    "equivalent_width_A_positive_absorption": ew,
                    "equivalent_width_err_A": unc["equivalent_width_err_A"],
                    "velocity_kms": velocity,
                    "velocity_err_kms": unc["velocity_err_kms"],
                    "continuum_norm": cont,
                    "continuum_rms_norm": cont_rms,
                    "n_points": int(len(x)),
                    "fit_reduced_chi2": fit_reduced_chi2,
                    "emission_snr": amp / amp_err if np.isfinite(amp_err) and amp_err > 0 else float("nan"),
                    "rejection_reason": "" if robust_emission else rejection_reason,
                }
            )
            ax.plot(x, yn, "k-", lw=0.8)
            ax.plot(x, model, "r-", lw=1.0)
            ax.axvline(rest, color="0.4", ls=":", lw=0.8)
            ax.set_title(f"{label} {name}: {kind}, EW={ew:.2f} A", fontsize=9)
        except Exception as exc:
            rows.append({"survey": label, "line": name, "rest_A": rest, "status": f"fit_failed: {exc}"})
            ax.plot(x, yn, "k-", lw=0.8)
            ax.axvline(rest, color="0.4", ls=":", lw=0.8)
            ax.set_title(f"{label} {name}: fit failed", fontsize=9)
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Normalized flux")
    path_out = out_dir / f"{label.lower()}_line_fits.png"
    fig.savefig(path_out, dpi=180)
    plt.close(fig)
    table_path = out_dir / f"{label.lower()}_line_fits.csv"
    write_csv(table_path, rows)
    emission_candidates = [
        row for row in rows
        if row.get("kind") == "emission" and abs(float(row.get("amplitude_norm") or 0.0)) >= 0.05
    ]
    robust_emission = [row for row in rows if row.get("robust_emission")]
    return {
        "status": "ok",
        "input": str(path),
        "plot": str(path_out),
        "table_csv": str(table_path),
        "rows": rows,
        "emission_candidate_count": len(emission_candidates),
        "emission_candidates": emission_candidates,
        "robust_emission_count": len(robust_emission),
        "robust_emission": robust_emission,
    }


def analyze_hst(run_roots: Path | list[Path], out_dir: Path) -> dict[str, Any]:
    path = locate_file(run_roots, ["hst_spectrum.csv"])
    if not path:
        return {"status": "missing"}
    wave, flux, err = load_spectrum_csv(path)
    okerr = np.isfinite(err) & (err > 0)
    snr = np.abs(flux[okerr] / err[okerr]) if np.any(okerr) else np.array([])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wave, flux, "k-", lw=0.5)
    ax.set_xlabel("Wavelength (A)")
    ax.set_ylabel("Flux")
    ax.set_title("HST spectrum with QA statistics")
    ax.text(
        0.02,
        0.95,
        f"N={len(wave)}\\nmedian S/N={np.nanmedian(snr) if snr.size else 0:.2f}\\nzero/negative={np.mean(flux <= 0):.2%}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    plot = out_dir / "hst_spectrum_qa.png"
    fig.savefig(plot, dpi=180)
    plt.close(fig)
    return {
        "status": "ok",
        "csv": str(path),
        "provenance": "fresh_or_first_available_analysis_root",
        "plot": str(plot),
        "n_points": int(len(wave)),
        "wavelength_min_A": float(np.nanmin(wave)) if len(wave) else None,
        "wavelength_max_A": float(np.nanmax(wave)) if len(wave) else None,
        "median_snr": float(np.nanmedian(snr)) if snr.size else 0.0,
        "zero_or_negative_fraction": float(np.mean(flux <= 0)) if len(flux) else None,
    }


def analyze_sed(run_roots: Path | list[Path], out_dir: Path) -> dict[str, Any]:
    sed_csv = locate_file(run_roots, ["sed_photometry.csv"])
    if not sed_csv:
        return {"status": "missing"}
    rows = read_csv(sed_csv)
    if not rows:
        return {"status": "empty", "csv": str(sed_csv)}
    phot = {}
    provenance_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            band = row["band"]
            mag = float(row["mag"])
            mag_err = float(row["mag_err"])
            wave_a = float(row["wave_A"])
            is_spherex = band.startswith("SPHEREx")
            is_wise = band.startswith("WISE")
            fit_inclusion = "primary_fit"
            provenance = "public_optical_photometry"
            if is_spherex:
                provenance = "local_spherex_label_unvalidated"
                fit_inclusion = "context_only_provisional"
            elif is_wise:
                provenance = "public_wise_ir_photometry"
                fit_inclusion = "ir_context_only"
            elif wave_a > 12000:
                fit_inclusion = "context_only_outside_optical_da_fit"
            phot[band] = (mag, mag_err, wave_a)
            provenance_rows.append(
                {
                    "band": band,
                    "wavelength_A": wave_a,
                    "mag": mag,
                    "mag_err": mag_err,
                    "provenance_class": provenance,
                    "fit_inclusion": fit_inclusion,
                    "included_in_primary_fit": fit_inclusion == "primary_fit",
                }
            )
        except Exception:
            continue
    sys.path.insert(0, str(ASTRO_AGENT_DIR))
    from astro_toolbox import wd_fitting

    primary_phot = {
        band: values
        for band, values in phot.items()
        if any(row["band"] == band and row["included_in_primary_fit"] for row in provenance_rows)
    }
    all_band_fit = wd_fitting.fit_sed(phot, parallax_mas=11.47)
    primary_fit = wd_fitting.fit_sed(primary_phot, parallax_mas=11.47) if len(primary_phot) >= 3 else None
    sed_fit = primary_fit or all_band_fit
    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
        constrained_layout=True,
    )
    waves, fluxes, ferrs, labels, included = [], [], [], [], []
    from astro_toolbox import config, utils

    for band, (mag, err, wave_a) in phot.items():
        zero = config.BAND_INFO.get(band, {}).get("zero_Jy", 3631.0)
        f, fe = utils.mag_to_flux_cgs(mag, wave_a, err, zero)
        waves.append(wave_a)
        fluxes.append(f)
        ferrs.append(fe)
        labels.append(band)
        included.append(band in primary_phot)
    inc = np.asarray(included, dtype=bool)
    waves_arr = np.asarray(waves, dtype=float)
    fluxes_arr = np.asarray(fluxes, dtype=float)
    ferrs_arr = np.asarray(ferrs, dtype=float)
    if np.any(inc):
        ax.errorbar(waves_arr[inc], fluxes_arr[inc], yerr=ferrs_arr[inc], fmt="o", color="k", capsize=3, label="primary fit bands")
    if np.any(~inc):
        ax.errorbar(
            waves_arr[~inc],
            fluxes_arr[~inc],
            yerr=ferrs_arr[~inc],
            fmt="s",
            mfc="white",
            mec="0.45",
            ecolor="0.55",
            color="0.45",
            capsize=3,
            label="context/excluded bands",
        )
    for x, y, lab in zip(waves, fluxes, labels):
        ax.annotate(lab.replace("_", " "), (x, y), fontsize=7, xytext=(3, 4), textcoords="offset points")
    residual_rows: list[dict[str, Any]] = []
    if sed_fit:
        templates = wd_fitting._load_koester2()
        tmpl = templates.get((sed_fit["teff_sed"], sed_fit["logg_sed"]))
        if tmpl:
            model_wave = np.asarray(tmpl["wavelength"], dtype=float)
            model_flux = np.asarray(tmpl["flux"], dtype=float) * float(sed_fit["scale"])
            ax.plot(model_wave, model_flux, "C1-", lw=0.9, label="Koester DA scaled")
            interp_model = np.interp(np.asarray(waves, dtype=float), model_wave, model_flux, left=np.nan, right=np.nan)
            provenance_by_band = {row["band"]: row for row in provenance_rows}
            for band, wave_a, obs, obs_err, mod in zip(labels, waves, fluxes, ferrs, interp_model):
                resid_sigma = (obs - mod) / obs_err if obs_err and np.isfinite(mod) else np.nan
                prov = provenance_by_band.get(band, {})
                residual_rows.append(
                    {
                        "band": band,
                        "wavelength_A": wave_a,
                        "flux_cgs_A": obs,
                        "flux_err_cgs_A": obs_err,
                        "model_flux_cgs_A": mod,
                        "residual_sigma": resid_sigma,
                        "provenance_class": prov.get("provenance_class", ""),
                        "fit_inclusion": prov.get("fit_inclusion", ""),
                        "included_in_primary_fit": prov.get("included_in_primary_fit", False),
                    }
                )
            rax.axhline(0.0, color="0.3", lw=0.8)
            residual_values = np.array([r["residual_sigma"] for r in residual_rows], dtype=float)
            if np.any(inc):
                rax.errorbar(waves_arr[inc], residual_values[inc], yerr=np.ones(np.count_nonzero(inc)), fmt="o", color="C3", capsize=3)
            if np.any(~inc):
                rax.errorbar(
                    waves_arr[~inc],
                    residual_values[~inc],
                    yerr=np.ones(np.count_nonzero(~inc)),
                    fmt="s",
                    mfc="white",
                    mec="0.45",
                    ecolor="0.55",
                    color="0.45",
                    capsize=3,
                )
        else:
            rax.text(0.02, 0.5, "Best-fit template missing", transform=rax.transAxes)
    else:
        rax.text(0.02, 0.5, "No Koester SED fit returned", transform=rax.transAxes)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$f_\lambda$ (erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)")
    rax.set_xscale("log")
    rax.set_xlabel("Wavelength (A)")
    rax.set_ylabel("Residual (sigma)")
    title = "SED fit"
    if sed_fit:
        title += f": primary Teff={sed_fit['teff_sed']} K logg={sed_fit['logg_sed']:.2f} chi2={sed_fit['chi2_sed']:.2f}"
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both")
    rax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    plot = out_dir / "sed_koester_fit.png"
    fig.savefig(plot, dpi=180)
    plt.close(fig)
    residual_csv = out_dir / "sed_koester_residuals.csv"
    write_csv(residual_csv, residual_rows)
    provenance_csv = out_dir / "sed_band_provenance.csv"
    write_csv(provenance_csv, provenance_rows)
    fit_status = "ok"
    qa_flags = []
    if not sed_fit:
        fit_status = "no_fit"
        qa_flags.append("Koester grid fit did not return a result")
    elif float(sed_fit.get("chi2_sed", 0.0)) > 10.0:
        fit_status = "poor_fit"
        qa_flags.append("Reduced chi-square exceeds 10; do not quote final atmosphere parameters without systematic review")
    if any(row["provenance_class"] == "local_spherex_label_unvalidated" for row in provenance_rows):
        qa_flags.append("SPHEREx-labeled points are treated as context only until their public-survey provenance is independently validated")
    if any(row["fit_inclusion"] == "ir_context_only" for row in provenance_rows):
        qa_flags.append("WISE infrared points are not used in the primary single-temperature DA atmosphere fit")
    return {
        "status": fit_status,
        "csv": str(sed_csv),
        "plot": str(plot),
        "residual_csv": str(residual_csv),
        "band_provenance_csv": str(provenance_csv),
        "fit": sed_fit,
        "primary_fit": primary_fit,
        "all_band_fit": all_band_fit,
        "primary_fit_bands": sorted(primary_phot),
        "context_only_bands": sorted(set(phot) - set(primary_phot)),
        "n_bands": len(phot),
        "n_primary_fit_bands": len(primary_phot),
        "qa_flags": qa_flags,
    }


def analyze_spectra(run_root: Path | list[Path], out_dir: Path) -> dict[str, Any]:
    outputs = {}
    for label, filename in (("SDSS", "sdss_spectrum.csv"), ("DESI", "desi_spectrum.csv")):
        path = locate_file(run_root, [filename])
        if path:
            outputs[label] = fit_lines_for_spectrum(path, out_dir, label)
    return outputs


def copy_core_figures(run_roots: Path | list[Path], out_dir: Path) -> dict[str, str]:
    fig_dir = ensure_dir(out_dir / "figures_from_toolbox")
    result = {}
    for root in normalize_roots(run_roots):
        for path in root.rglob("*.png"):
            if path.stat().st_size <= 0:
                continue
            rel_key = "_".join(path.relative_to(root).parts).replace(".png", "")
            dst = fig_dir / f"{root.name}_{path.parent.name}_{path.name}"
            if dst.exists():
                continue
            shutil.copy2(path, dst)
            result[rel_key] = str(dst)
    return result


def build_figure_manifest(analysis: dict[str, Any]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}

    def add(key: str, path_value: Any, note: str = "") -> None:
        if not path_value:
            return
        path = Path(str(path_value))
        manifest[key] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "note": note,
        }

    add("hst_spectrum_qa", analysis.get("hst", {}).get("plot"), "HST spectrum and QA statistics")
    add("sed_plot_path", analysis.get("sed", {}).get("plot"), "Koester SED fit plus residuals")
    for label, spec in analysis.get("line_fits", {}).items():
        add(f"{label.lower()}_line_fit_plot_path", spec.get("plot"), f"{label} optical line fits")
        add(f"{label.lower()}_line_fit_table_path", spec.get("table_csv"), f"{label} line fit CSV")
    for key, path in analysis.get("copied_toolbox_figures", {}).items():
        add(f"toolbox_{key}", path, "Copied astro_toolbox figure")
    return manifest


def build_markdown_report(package: dict[str, Any]) -> str:
    lines = [
        "# Source Research Package",
        "",
        f"Target: {package['target']}",
        f"Coordinates: RA={package['ra_deg']}, Dec={package['dec_deg']} deg",
        "",
        "## Source Identifiers",
        ", ".join(package.get("source_identifiers", [])[:24]) or "No identifiers resolved.",
        "",
        "## SIMBAD References",
        f"SIMBAD returned n_refs={package.get('simbad', {}).get('n_refs', 0)}.",
        f"PDF downloader status={package.get('downloads', {}).get('status', 'unknown')}; available={package.get('downloads', {}).get('n_available_pdf', 0)}.",
    ]
    for ref in package.get("source_mentions", []):
        mark = "mentions source" if ref.get("mentions_source") else "no direct mention in extracted text"
        lines.append(f"- {ref.get('bibcode')} ({ref.get('year')}): {ref.get('title')} [{mark}]")
    lines.extend(["", "## KG Relations"])
    for row in package.get("kg_relations", [])[:80]:
        lines.append(f"- {row.get('subject')} --{row.get('relation')}--> {row.get('object')}")
    lines.extend(["", "## Data QA"])
    lines.append(json.dumps(package.get("analysis", {}), ensure_ascii=False, indent=2)[:8000])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a strict per-source research package.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--astrotool-run", default="")
    parser.add_argument("--cached-astrotool-root", default="", help="Optional older astro_toolbox root used only when fresh products are missing.")
    parser.add_argument("--run-astrotool", action="store_true")
    parser.add_argument("--download-simbad-pdfs", action="store_true")
    args = parser.parse_args()

    out_root = ensure_dir(Path(args.output_root).resolve())
    astrotool_run = Path(args.astrotool_run).resolve() if args.astrotool_run else out_root / "astrotool_run"
    if args.run_astrotool:
        astrotool = run_astrotool(args.target, args.ra, args.dec, astrotool_run)
    else:
        astrotool = {
            "output_root": str(astrotool_run),
            "module_status": read_csv(astrotool_run / "module_status.csv"),
            "run_summary": json.loads((astrotool_run / "run_summary.json").read_text(encoding="utf-8"))
            if (astrotool_run / "run_summary.json").exists()
            else {},
        }
    write_json(out_root / "astrotool_run_summary.json", astrotool)
    analysis_roots = [astrotool_run]
    cached_root = Path(args.cached_astrotool_root).resolve() if args.cached_astrotool_root else None
    if cached_root and cached_root.exists():
        analysis_roots.append(cached_root)

    simbad = query_simbad_all_refs(args.ra, args.dec)
    if not simbad.get("references"):
        fallback = read_local_simbad_references(analysis_roots)
        if fallback.get("references"):
            fallback["online_simbad_status"] = simbad.get("status")
            fallback["online_simbad_error"] = simbad.get("error", "")
            simbad = fallback
    refs = enrich_refs_from_local_metadata(simbad.get("references", []))
    simbad["references"] = refs
    write_json(out_root / "simbad_all_references.json", simbad)

    downloads = {"status": "skipped"}
    if args.download_simbad_pdfs:
        downloads = download_simbad_pdfs(refs)
    write_json(out_root / "simbad_pdf_downloads.json", downloads)

    identifiers = build_source_identifiers(args.target, simbad, args.ra, args.dec)
    write_json(out_root / "source_identifiers.json", identifiers)
    mentions = source_mentions(refs, identifiers)
    write_json(out_root / "simbad_source_mentions.json", mentions)
    rag_rows = rag_exact_bibcodes([r.get("bibcode", "") for r in refs])
    write_json(out_root / "rag_exact_simbad_papers.json", rag_rows)
    kg_rows = kg_source_relations(identifiers)
    write_json(out_root / "kg_source_relations.json", kg_rows)

    analysis_dir = ensure_dir(out_root / "analysis_products")
    analysis = {
        "analysis_roots_priority": [str(root) for root in analysis_roots],
        "hst": analyze_hst(analysis_roots, analysis_dir),
        "sed": analyze_sed(analysis_roots, analysis_dir),
        "line_fits": analyze_spectra(analysis_roots, analysis_dir),
        "copied_toolbox_figures": copy_core_figures(analysis_roots, analysis_dir),
    }
    try:
        sys.path.insert(0, str(ASTRO_AGENT_DIR))
        from astro_toolbox.compact_binary_report import build_report, write_report

        compact = build_report(analysis_roots[0], target=args.target, ra=args.ra, dec=args.dec)
        analysis["compact_binary_report"] = write_report(compact, analysis_dir / "compact_binary_report")
    except Exception as exc:
        analysis["compact_binary_report"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    analysis["figure_manifest"] = build_figure_manifest(analysis)
    write_json(out_root / "source_analysis_products.json", analysis)

    package = {
        "target": args.target,
        "ra_deg": args.ra,
        "dec_deg": args.dec,
        "astrotool": astrotool,
        "simbad": simbad,
        "source_identifiers": identifiers,
        "downloads": downloads,
        "source_mentions": mentions,
        "rag_exact_papers": rag_rows,
        "kg_relations": kg_rows,
        "analysis": analysis,
    }
    write_json(out_root / "source_research_package.json", package)
    write_text(out_root / "source_research_report.md", build_markdown_report(package))
    print(json.dumps({"output_root": str(out_root), "simbad_n_refs": simbad.get("n_refs"), "kg_relations": len(kg_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
