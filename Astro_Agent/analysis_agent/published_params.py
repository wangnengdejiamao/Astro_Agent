"""Published-parameter extractor.

When SIMBAD already lists peer-reviewed work on a target, we can pull the
quoted parameter values straight out of the local references CSV + a few
known measurement files (period_analysis.csv, rv_analysis.txt, etc.) and
build a structured table the drafter can cite as `(value, error, unit,
bibcode, source_kind)`.

This is the minimal-viable replacement for "per-source RAG" (Plan A2) — it
does NOT download PDFs, it only mines what SIMBAD + astro_toolbox already
emitted.  It is enough to lift the drafter from "withheld pending QA" to
"here is what the literature reports + here is what this run independently
measures".
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# Regex hits for quantities mentioned in SIMBAD abstract snippets.
# Each pattern captures: number, optional uncertainty, unit, "label" (what
# the number is). We keep this conservative — we only extract canonical
# unit forms and let the drafter quote them with explicit bibcode.
_NUM = r"([0-9]+(?:\.[0-9]+)?)"
_PM = r"(?:\s*(?:\\pm|\xb1|\+/-|±)\s*([0-9]+(?:\.[0-9]+)?))?"

_PARAMETER_PATTERNS = [
    # Orbital periods: "P_orb = 39.34 min" / "orbital period of 39 minutes"
    (re.compile(
        rf"(?:P[_ ]?orb|orbital period(?: of)?)\s*[=:]?\s*{_NUM}{_PM}\s*(?:min|minutes|m\b)",
        re.IGNORECASE,
    ), "orbital_period_min", "min"),
    # Effective temperature
    (re.compile(
        rf"T[_ ]?eff\s*[=:~]?\s*{_NUM}{_PM}\s*K", re.IGNORECASE,
    ), "Teff_K", "K"),
    # Surface gravity
    (re.compile(
        rf"log\s*g\s*[=:~]?\s*{_NUM}{_PM}", re.IGNORECASE,
    ), "logg", "dex"),
    # Component masses
    (re.compile(
        rf"M[_ ]?(?:sdOB|sdB|donor|1)\s*[=:~]?\s*{_NUM}{_PM}\s*(?:M[_]?(?:sun|⊙)|Msun|\\msun)",
        re.IGNORECASE,
    ), "M_donor_Msun", "Msun"),
    (re.compile(
        rf"M[_ ]?(?:WD|2|accretor)\s*[=:~]?\s*{_NUM}{_PM}\s*(?:M[_]?(?:sun|⊙)|Msun|\\msun)",
        re.IGNORECASE,
    ), "M_accretor_Msun", "Msun"),
    # Orbital inclination
    (re.compile(
        rf"\b(?:i|inclination)\s*[=:~]?\s*{_NUM}{_PM}\s*(?:deg|°)", re.IGNORECASE,
    ), "inclination_deg", "deg"),
    # Merger / decay timescale
    (re.compile(
        rf"merger\s*(?:in|time)?\s*[=:~]?\s*{_NUM}{_PM}\s*Myr", re.IGNORECASE,
    ), "tau_merger_Myr", "Myr"),
    # Orbital period derivative (LISA verification binaries)
    (re.compile(
        rf"(?:P[_ ]?dot|orbital period decay)\s*[=:~]?\s*\(?\s*-?{_NUM}{_PM}\s*\)?\s*(?:×|x|\\times)?\s*10\^?[-]?12\s*s/?s",
        re.IGNORECASE,
    ), "Pdot_s_per_s_x1e-12", "s/s × 1e-12"),
]


def _read_simbad_references_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.append(row)
    return out


def _read_simbad_references_txt(path: Path) -> List[Dict[str, Any]]:
    """The simbad_references.txt produced by astro_toolbox has [N] bibcode + title + abstract."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    entries: List[Dict[str, Any]] = []
    # Each entry starts with "[N] bibcode" on its own line.
    blocks = re.split(r"\n=+\n", text)
    for block in blocks:
        head_match = re.search(r"^\s*\[(\d+)\]\s+(\S+)\s*$", block, flags=re.MULTILINE)
        if not head_match:
            continue
        bibcode = head_match.group(2)
        title_match = re.search(r"Title:\s*(.+?)(?=\n\s*Authors:|\Z)", block, flags=re.DOTALL)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
        abstract_match = re.search(r"Abstract:\s*(.+?)(?=\n=+|\Z)", block, flags=re.DOTALL)
        abstract = re.sub(r"\s+", " ", abstract_match.group(1)).strip() if abstract_match else ""
        year_match = re.match(r"^(\d{4})", bibcode or "")
        entries.append({
            "bibcode": bibcode,
            "year": int(year_match.group(1)) if year_match else None,
            "title": title,
            "abstract": abstract,
        })
    return entries


def _target_aliases(target: Optional[str]) -> List[str]:
    """Build a set of name variants that should plausibly appear near a parameter
    quoted as belonging to *this* target rather than another source named in
    the same abstract. Conservative: we want false-negative > false-positive."""
    if not target:
        return []
    raw = str(target).strip()
    aliases = {raw}
    aliases.add(raw.replace(" ", ""))
    # ZTF J213056.71+442046.5 -> J2130 / J213056 / 2130+4420 / 213056+442046
    import re as _re
    m = _re.search(r"J(\d{4,8})([+\-]\d{4,8})", raw.replace(" ", ""))
    if m:
        ra_part = m.group(1)
        dec_part = m.group(2)
        aliases.add("J" + ra_part[:4] + dec_part[:5])
        aliases.add("J" + ra_part[:4])
        aliases.add(ra_part[:4] + dec_part[:5])
        aliases.add(ra_part[:4])
    # Strip catalog prefix
    prefix_match = _re.match(r"^([A-Z]+)[\s_]?(.+)$", raw)
    if prefix_match:
        aliases.add(prefix_match.group(2))
    return [a for a in aliases if len(a) >= 4]


def _text_mentions_target(text: str, aliases: Iterable[str]) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(a.lower() in low for a in aliases if a)


def _extract_from_text(
    text: str,
    bibcode: str,
    *,
    target_aliases: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Mine (value, error, unit) tuples from a SIMBAD abstract.

    target_aliases: if non-empty, every extracted row gets a `target_match`
        boolean indicating whether the abstract actually mentions THIS source.
        Downstream consumers should drop rows with target_match=False before
        quoting them as "literature parameters for the current target".
    """
    out: List[Dict[str, Any]] = []
    mentions_target = (
        _text_mentions_target(text, target_aliases) if target_aliases else None
    )
    for pattern, label, unit in _PARAMETER_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1)
            error = match.group(2) if match.lastindex and match.lastindex >= 2 else None
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            ferr = None
            if error:
                try:
                    ferr = float(error)
                except (TypeError, ValueError):
                    ferr = None
            # Tighten the mention check: require the alias to appear in the
            # same 250-character window as the parameter mention, so we don't
            # confuse abstracts that name multiple sources (e.g. Yang+2025
            # quotes both J1710 and references to other systems).
            window_start = max(0, match.start() - 250)
            window_end = match.end() + 250
            local_text = text[window_start:window_end]
            local_match = (
                _text_mentions_target(local_text, target_aliases)
                if target_aliases else None
            )
            out.append({
                "parameter": label,
                "value": fval,
                "error": ferr,
                "unit": unit,
                "bibcode": bibcode,
                "source_kind": "simbad_abstract",
                "snippet": text[max(0, match.start() - 40): match.end() + 40].strip(),
                "target_match_abstract": mentions_target,
                "target_match_window": local_match,
            })
    return out


def _measurements_from_run(astrotool_root: Path) -> List[Dict[str, Any]]:
    """Read deterministic measurement files produced by astro_toolbox."""
    out: List[Dict[str, Any]] = []
    # Period analysis: take rows where quality==good AND fap is small
    period_csv = astrotool_root / "period_analysis.csv"
    if period_csv.exists():
        with period_csv.open("r", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                quality = (row.get("quality") or "").lower()
                detected = (row.get("detected") or "").lower()
                if detected != "true" or quality not in {"good"}:
                    continue
                try:
                    p_hour = float(row.get("best_period_hour") or "")
                    p_min = float(row.get("best_period_min") or "")
                except ValueError:
                    continue
                out.append({
                    "parameter": "photometric_period_min",
                    "value": p_min,
                    "error": None,
                    "unit": "min",
                    "bibcode": None,
                    "source_kind": f"this_work_period_analysis::{row.get('curve')}",
                    "snippet": f"curve={row.get('curve')} method={row.get('method')} fap={row.get('fap')}",
                })
    # RV analysis
    rv_csv = astrotool_root / "rv_analysis.csv"
    if rv_csv.exists():
        with rv_csv.open("r", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                if (row.get("method") or "").lower() == "best":
                    continue  # avoid duplicate
                try:
                    rv = float(row.get("rv_kms") or "")
                    rv_err = float(row.get("rv_err_kms") or "")
                except ValueError:
                    continue
                out.append({
                    "parameter": "radial_velocity_km_s",
                    "value": rv,
                    "error": rv_err,
                    "unit": "km/s",
                    "bibcode": None,
                    "source_kind": f"this_work_rv::{row.get('source')}",
                    "snippet": f"method={row.get('method')} template_dependent=true",
                })
    # Orbit traceback summary (best match cluster, plausibility)
    tb_txt = astrotool_root / "orbit_traceback.txt"
    if tb_txt.exists():
        text = tb_txt.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"Best Match\s*\n\s*Cluster:\s*(\S+)", text)
        if m:
            cluster = m.group(1).replace("_", " ")
            ms = re.search(r"Min separation:\s*([\d\.]+)\s*pc", text)
            mt = re.search(r"Time of closest approach:\s*([\d\.]+)\s*Myr", text)
            plaus = "physically plausible" in text.lower() or "plausible escape host" in text.lower()
            plaus_str = "yes" if plaus else "no (cluster age < traceback time OR outside tidal radius)"
            sep_str = f"{ms.group(1)} pc" if ms else "unknown"
            time_str = f"{mt.group(1)} Myr ago" if mt else "unknown"
            out.append({
                "parameter": "kinematic_traceback_best_match",
                "value": None,
                "error": None,
                "unit": None,
                "bibcode": None,
                "source_kind": "this_work_orbit_traceback",
                "snippet": (
                    f"closest candidate cluster {cluster} at {sep_str} "
                    f"(approach {time_str}); physically plausible host: {plaus_str}"
                ),
            })
    return out


def build_published_params_table(
    astrotool_root: Optional[Path],
    *,
    target: Optional[str] = None,
    max_refs: int = 50,
    require_target_match: bool = True,
) -> Dict[str, Any]:
    """Build a structured table of published values + this-work measurements.

    target: if provided, literature rows are tagged with target_match flags
        and (when require_target_match=True) only rows whose enclosing 250-char
        window mentions a known alias of `target` are kept.  This rules out
        values quoted in an abstract that actually refers to a different
        source named alongside ours.
    """
    rows: List[Dict[str, Any]] = []
    n_refs = 0
    aliases = _target_aliases(target)
    if astrotool_root and astrotool_root.exists():
        # 1) deterministic this-work measurements from astrotool products
        rows.extend(_measurements_from_run(astrotool_root))
        # 2) literature values mined from SIMBAD abstracts
        refs = _read_simbad_references_txt(astrotool_root / "simbad_references.txt")
        n_refs = len(refs)
        for ref in refs[:max_refs]:
            rows.extend(_extract_from_text(
                ref.get("abstract", ""),
                ref.get("bibcode") or "?",
                target_aliases=aliases,
            ))
    if require_target_match and aliases:
        before = len([r for r in rows if r.get("source_kind") == "simbad_abstract"])
        rows = [
            r for r in rows
            if r.get("source_kind") != "simbad_abstract"
            or r.get("target_match_window") is True
        ]
        after = len([r for r in rows if r.get("source_kind") == "simbad_abstract"])
        filtered = before - after
    else:
        filtered = 0
    # Light de-duplication: same parameter+bibcode+value
    seen = set()
    dedup: List[Dict[str, Any]] = []
    for r in rows:
        key = (r.get("parameter"), r.get("bibcode"), r.get("value"), r.get("source_kind"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    by_param: Dict[str, int] = {}
    for r in dedup:
        by_param[r["parameter"]] = by_param.get(r["parameter"], 0) + 1
    return {
        "n_refs_scanned": n_refs,
        "n_rows": len(dedup),
        "n_this_work": sum(1 for r in dedup if str(r.get("source_kind", "")).startswith("this_work")),
        "n_from_literature": sum(1 for r in dedup if r.get("source_kind") == "simbad_abstract"),
        "by_parameter": by_param,
        "target": target,
        "target_aliases": aliases,
        "n_filtered_by_target_match": filtered,
        "rows": dedup,
    }


def render_markdown(table: Dict[str, Any], *, max_rows_per_param: int = 4) -> str:
    """Pretty-print a published-params table for human/LLM consumption."""
    lines = [
        "## Published parameters and this-work measurements",
        "",
        f"- references scanned: {table.get('n_refs_scanned')}",
        f"- rows: {table.get('n_rows')} (this work: {table.get('n_this_work')}, literature: {table.get('n_from_literature')})",
        "",
        "| parameter | value ± error | unit | source | snippet |",
        "|---|---|---|---|---|",
    ]
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in table.get("rows", []):
        grouped.setdefault(row["parameter"], []).append(row)
    for param, rows in grouped.items():
        for r in rows[:max_rows_per_param]:
            val = r.get("value")
            err = r.get("error")
            if val is None:
                val_str = "—"
            elif err is None:
                val_str = f"{val}"
            else:
                val_str = f"{val} ± {err}"
            src = r.get("bibcode") or r.get("source_kind") or "?"
            snippet = (r.get("snippet") or "").replace("|", "/")[:120]
            lines.append(f"| {param} | {val_str} | {r.get('unit') or ''} | `{src}` | {snippet} |")
    return "\n".join(lines) + "\n"


__all__ = ["build_published_params_table", "render_markdown"]
