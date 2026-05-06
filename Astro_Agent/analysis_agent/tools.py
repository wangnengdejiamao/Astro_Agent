"""Local tool adapters used by the Chief Investigator workflow."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PACKAGE_DIR = Path(__file__).resolve().parent
ASTRO_AGENT_DIR = PACKAGE_DIR.parent
REPO_ROOT = ASTRO_AGENT_DIR.parent
ASTRO_TOOLBOX_DIR = ASTRO_AGENT_DIR / "astro_toolbox"
RAG_DB = REPO_ROOT / "rag_pipeline" / "index" / "white_dwarf_rag.sqlite"
KG_JSON = (
    REPO_ROOT
    / "prompt2graph_for_astronomy"
    / "output"
    / "white_dwarf_kg"
    / "production_full"
    / "multi_stage_deduplicated.json"
)
KG_INDEX = (
    REPO_ROOT
    / "prompt2graph_for_astronomy"
    / "output"
    / "white_dwarf_kg"
    / "kg_index.sqlite"
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(text: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._+-]+", "_", str(text).strip())
    return safe.strip("_") or "target"


def json_dump(path: Path, payload: Any) -> str:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def validate_coordinates(ra_deg: float, dec_deg: float) -> Tuple[float, float]:
    ra = float(ra_deg)
    dec = float(dec_deg)
    if not (0.0 <= ra < 360.0):
        raise ValueError(f"RA must be in [0, 360) deg, got {ra}")
    if not (-90.0 <= dec <= 90.0):
        raise ValueError(f"Dec must be in [-90, 90] deg, got {dec}")
    return ra, dec


def resolve_target(target: str, ra_deg: Optional[float], dec_deg: Optional[float]) -> Dict[str, Any]:
    """Resolve a target name or validate supplied coordinates."""
    if ra_deg is not None and dec_deg is not None:
        ra, dec = validate_coordinates(ra_deg, dec_deg)
        return {
            "status": "ok",
            "target": target,
            "ra_deg": ra,
            "dec_deg": dec,
            "resolver": "user_supplied_coordinates",
            "frame": "ICRS",
            "unit_checks": ["RA/Dec validated as decimal degrees"],
        }

    errors: List[str] = []
    try:
        from astropy.coordinates import SkyCoord

        coord = SkyCoord.from_name(target)
        ra, dec = validate_coordinates(coord.ra.deg, coord.dec.deg)
        return {
            "status": "ok",
            "target": target,
            "ra_deg": ra,
            "dec_deg": dec,
            "resolver": "astropy.SkyCoord.from_name",
            "frame": "ICRS",
            "unit_checks": ["Name resolved to ICRS decimal degrees"],
        }
    except Exception as exc:
        errors.append(f"SkyCoord.from_name failed: {type(exc).__name__}: {exc}")

    try:
        from astroquery.simbad import Simbad

        table = Simbad.query_object(target)
        if table is not None and len(table) > 0:
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            row = table[0]
            coord = SkyCoord(str(row["RA"]), str(row["DEC"]), unit=(u.hourangle, u.deg))
            ra, dec = validate_coordinates(coord.ra.deg, coord.dec.deg)
            return {
                "status": "ok",
                "target": target,
                "ra_deg": ra,
                "dec_deg": dec,
                "resolver": "astroquery.simbad.Simbad.query_object",
                "frame": "ICRS",
                "unit_checks": ["SIMBAD sexagesimal coordinates converted to degrees"],
            }
    except Exception as exc:
        errors.append(f"SIMBAD name resolution failed: {type(exc).__name__}: {exc}")

    return {
        "status": "needs_human",
        "target": target,
        "errors": errors,
        "message": "Could not resolve target name. Provide --ra and --dec in decimal degrees.",
    }


def query_simbad_crossmatch(ra_deg: float, dec_deg: float) -> Dict[str, Any]:
    try:
        sys.path.insert(0, str(ASTRO_AGENT_DIR))
        from astro_toolbox import utils

        table = utils.query_simbad(ra_deg, dec_deg)
        if table is None or len(table) == 0:
            return {"status": "empty", "matches": 0}
        rows = []
        for row in table[:5]:
            rows.append({name: str(row[name]) for name in table.colnames})
        return {"status": "ok", "matches": len(table), "rows": rows}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def run_astrotool(
    target: str,
    ra_deg: float,
    dec_deg: float,
    output_root: Path,
    dry_run: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """Run or plan the existing astro_toolbox single-target workflow."""
    ensure_dir(output_root)
    command = [
        sys.executable,
        str(ASTRO_TOOLBOX_DIR / "run_single_target_all_tools.py"),
        "--target",
        target,
        "--ra",
        f"{ra_deg:.10f}",
        "--dec",
        f"{dec_deg:.10f}",
        "--output-root",
        str(output_root),
    ]
    if dry_run and not force:
        return {
            "status": "planned",
            "command": command,
            "output_root": str(output_root),
            "note": "dry-run mode; use --execute to fetch survey data and run physical modules",
            "existing_outputs": summarize_output_root(output_root),
        }

    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        timeout=3600,
    )
    result = {
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "command": command,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "output_root": str(output_root),
        "existing_outputs": summarize_output_root(output_root),
    }
    return result


def summarize_output_root(output_root: Path) -> Dict[str, Any]:
    status_path = output_root / "module_status.csv"
    run_summary = read_json(output_root / "run_summary.json") or {}
    target_info = read_json(output_root / "target_info.json") or {}
    status_rows: List[Dict[str, str]] = []
    if status_path.exists():
        with status_path.open(newline="", encoding="utf-8") as fh:
            status_rows = list(csv.DictReader(fh))
    counts: Dict[str, int] = {}
    for row in status_rows:
        counts[row.get("status", "")] = counts.get(row.get("status", ""), 0) + 1
    files = [str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file()]
    inferred = infer_legacy_module_rows(files)
    if not status_rows and inferred:
        status_rows = inferred
        counts = {}
        for row in status_rows:
            counts[row.get("status", "")] = counts.get(row.get("status", ""), 0) + 1
    if not run_summary and files:
        file_set = set(files)
        run_summary = {
            "legacy_inferred": True,
            "spectra_available": any(name in file_set for name in ["sdss_spectrum.csv", "desi_spectrum.csv", "hst_spectrum.csv", "spherex_spectrum.csv"]),
            "hst_spectrum_available": "hst_spectrum.csv" in file_set,
            "sed_available": any(name in file_set for name in ["sed.png", "sed_photometry.csv"]),
            "rv_report_available": any(name in file_set for name in ["rv_analysis.txt", "rv_analysis.csv"]),
            "period_products_available": any("period" in name.lower() or "fold" in name.lower() for name in file_set),
            "wd_fitting_available": any(name in file_set for name in ["wd_fitting.json", "wd_fit_results.csv", "wd_model_fit.csv"]),
        }
    return {
        "status_file": str(status_path) if status_path.exists() else None,
        "module_status_counts": counts,
        "module_rows": status_rows,
        "run_summary": run_summary,
        "target_info": target_info,
        "n_files": len(files),
        "sample_files": sorted(files)[:30],
    }


def infer_legacy_module_rows(files: Sequence[str]) -> List[Dict[str, str]]:
    """Infer module status from older flat astrotool output directories."""
    file_set = set(files)
    checks = [
        ("SDSS_spectrum", ["sdss_spectrum.csv", "sdss_spectrum.png"]),
        ("DESI_spectrum", ["desi_spectrum.csv", "desi_spectrum.png"]),
        ("HST_spectrum", ["hst_spectrum.csv", "hst_spectrum.png"]),
        ("SPHEREx_spectrum", ["spherex_spectrum.csv", "spherex_spectrum.png"]),
        ("ZTF_lightcurve", ["ztf_lightcurve.csv", "ztf_lightcurve.png"]),
        ("WISE_lightcurve", ["wise_lightcurve.csv", "wise_lightcurve.png"]),
        ("TESS_lightcurve", ["tess_lightcurve.csv", "tess_lightcurve.png"]),
        ("SED", ["sed.png", "sed_photometry.csv"]),
        ("HR_diagram", ["hr_diagram.png"]),
        ("RV_analysis", ["rv_analysis.txt", "rv_analysis.csv"]),
        ("Orbit_traceback", ["orbit_traceback.txt", "orbit_traceback_candidates.csv"]),
        ("SIMBAD_refs", ["simbad_references.csv", "simbad_references.txt"]),
    ]
    rows: List[Dict[str, str]] = []
    for module, expected in checks:
        present = [name for name in expected if name in file_set]
        if present:
            rows.append(
                {
                    "module": module,
                    "status": "ok",
                    "note": "legacy output inferred from files: " + ", ".join(present),
                }
            )
    return rows


def search_rag(
    query: str,
    category: Optional[str] = None,
    method_only: bool = False,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    if not RAG_DB.exists():
        return [{"status": "missing", "message": f"RAG database not found: {RAG_DB}"}]
    sys.path.insert(0, str(REPO_ROOT))
    from rag_pipeline.search_database import search

    args = Namespace(
        query=query,
        db=str(RAG_DB),
        category=category,
        section=None,
        method_only=method_only,
        year_from=None,
        year_to=None,
        limit=limit,
        json=True,
    )
    try:
        return search(args)
    except sqlite3.OperationalError as exc:
        return [{"status": "error", "query": query, "message": str(exc)}]


def flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(flatten_text(v) for v in value)
    return str(value)


def _node_name(node: Dict[str, Any]) -> str:
    props = node.get("properties") or {}
    return str(props.get("name") or node.get("name") or "")


def _shorten(text: Any, limit: int = 500) -> str:
    value = str(text or "").replace("\n", " ")
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def search_kg_sqlite(queries: Sequence[str], limit: int = 12) -> List[Dict[str, Any]]:
    """Fast local graph search over the SQLite KG index."""
    if not KG_INDEX.exists():
        return []
    terms = []
    for query in queries:
        terms.extend(term for term in re.findall(r"[\w.+-]+", query) if len(term) > 2)
    if not terms:
        return []

    con = sqlite3.connect(KG_INDEX)
    con.row_factory = sqlite3.Row
    scored: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    try:
        for term in terms:
            like = f"%{term}%"
            rows = con.execute(
                """
                SELECT subject, subject_type, relation, object, object_type, title, source, evidence, chunk_id
                FROM kg_edges
                WHERE subject LIKE ? OR object LIKE ? OR relation LIKE ? OR title LIKE ? OR source LIKE ? OR evidence LIKE ?
                LIMIT ?
                """,
                (like, like, like, like, like, like, max(limit * 6, 50)),
            ).fetchall()
            for row in rows:
                key = (row["subject"], row["relation"], row["object"], row["chunk_id"])
                item = scored.setdefault(
                    key,
                    {
                        "score": 0,
                        "subject": row["subject"],
                        "subject_type": row["subject_type"],
                        "relation": row["relation"],
                        "object": row["object"],
                        "object_type": row["object_type"],
                        "title": row["title"],
                        "source": _shorten(row["source"]),
                        "evidence": _shorten(row["evidence"], limit=300),
                        "chunk_id": row["chunk_id"],
                        "index": str(KG_INDEX),
                    },
                )
                item["score"] += 1
    finally:
        con.close()
    return sorted(scored.values(), key=lambda item: item["score"], reverse=True)[:limit]


def search_kg(queries: Sequence[str], limit: int = 12) -> List[Dict[str, Any]]:
    """Local graph search. Prefer SQLite index; fall back to JSON only if needed."""
    indexed = search_kg_sqlite(queries, limit=limit)
    if indexed:
        return indexed
    if not KG_JSON.exists():
        return [{"status": "missing", "message": f"KG index/export not found: {KG_INDEX} / {KG_JSON}"}]
    terms = [term.lower() for query in queries for term in re.findall(r"[\w.+-]+", query)]
    terms = [term for term in terms if len(term) > 2]
    if not terms:
        return []

    data = json.loads(KG_JSON.read_text(encoding="utf-8"))
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for triple in data:
        text = flatten_text(triple).lower()
        score = sum(1 for term in terms if term in text)
        if score <= 0:
            continue
        start = triple.get("start_node", {})
        end = triple.get("end_node", {})
        scored.append(
            (
                score,
                {
                    "score": score,
                    "subject": _node_name(start),
                    "relation": triple.get("relation", ""),
                    "object": _node_name(end),
                    "source": _shorten(triple.get("source", "")),
                    "evidence": _shorten(triple.get("evidence", ""), limit=300),
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:limit]]


def extract_physics_checks(output_summary: Dict[str, Any]) -> Dict[str, Any]:
    rows = output_summary.get("module_rows", [])
    failed = [row for row in rows if row.get("status") in {"error", "empty"}]
    warnings: List[str] = []
    if failed:
        warnings.append(f"{len(failed)} astro_toolbox modules returned error/empty")

    run_summary = output_summary.get("run_summary", {})
    if run_summary and not run_summary.get("wd_fitting_available"):
        warnings.append("WD fitting is unavailable; no final stellar parameters should be claimed")
    if run_summary and not run_summary.get("rv_report_available"):
        warnings.append("RV report is unavailable; orbit traceback and kinematics remain provisional")

    return {
        "module_failures": failed[:20],
        "warnings": warnings,
        "run_summary": run_summary,
    }


def detect_anomalies(data_fetch: Dict[str, Any], iterations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    reasons: List[str] = []
    if data_fetch.get("status") == "error":
        reasons.append("astro_toolbox execution failed")
    if data_fetch.get("status") == "planned":
        reasons.append("workflow is in dry-run mode; survey data have not been fetched")

    output = data_fetch.get("existing_outputs", {})
    rows = output.get("module_rows", [])
    error_rows = [row for row in rows if row.get("status") == "error"]
    if error_rows:
        reasons.append(f"{len(error_rows)} module errors require inspection")

    for item in iterations:
        for warning in item.get("warnings", []):
            low = warning.lower()
            if "unavailable" in low or "cannot" in low or "failed" in low:
                reasons.append(warning)

    nonconverged = any(item.get("status") == "nonconverged" for item in iterations)
    if nonconverged:
        reasons.append("three required modeling iterations did not converge")

    return {
        "human_review_required": bool(reasons),
        "reasons": sorted(set(reasons)),
    }


def write_text(path: Path, text: str) -> str:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return str(path)


def format_rag_bullets(rows: Iterable[Dict[str, Any]], max_items: int = 8) -> str:
    lines = []
    for idx, row in enumerate(rows, start=1):
        if idx > max_items:
            break
        title = str(row.get("title") or "").replace("\n", " ")
        bibcode = row.get("bibcode", "")
        year = row.get("year", "")
        methods = row.get("methods_json", "[]")
        lines.append(f"- {bibcode} ({year}): {title}; methods={methods}")
    return "\n".join(lines) or "- No local RAG hits."


def format_kg_bullets(rows: Iterable[Dict[str, Any]], max_items: int = 8) -> str:
    lines = []
    for idx, row in enumerate(rows, start=1):
        if idx > max_items:
            break
        lines.append(
            f"- {row.get('subject', '')} --{row.get('relation', '')}--> "
            f"{row.get('object', '')}; evidence={row.get('evidence', '')}"
        )
    return "\n".join(lines) or "- No local KG hits."


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False
