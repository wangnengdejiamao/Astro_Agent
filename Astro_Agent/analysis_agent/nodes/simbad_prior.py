"""Simbad prior → fitting-strategy node.

Sits between ``data_fetcher`` and the modeling iterations. Reads the Simbad
crossmatch already collected by ``data_fetcher_node`` and emits a structured
``simbad_prior`` block that downstream nodes (iteration_1/2/3, qa_gate,
abnormal_report) MUST consult before applying a standard white-dwarf fit.

This closes the bug where the workflow happily ran a WD SED fit on objects
classified by Simbad as QSO/AGN/galaxy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


# ---------------------------------------------------------------------------
# OTYPE classification rules. Conservative — when unsure we degrade, never
# upgrade, the fitting strategy.
# ---------------------------------------------------------------------------

# Each rule: tag → (substrings to match against OTYPE/MAIN_ID/SP_TYPE upper-cased)
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Hard non-stellar
    ("non_stellar_contaminant", (
        "QSO", "AGN", "BLLAC", "BL LAC", "BLAZAR", "GALAXY", "SEYFERT",
        "RADIOSOURCE", "GINCL", "GROUP", "CLUSTER OF G", "GLENS",
    )),
    # Cataclysmic / accreting / emission-line systems
    ("binary_or_accretion_candidate", (
        "CATACLYS", "CV*", "DWARF NOVA", "NOVA", "X", "XB*", "LXB", "HXB",
        "SYMBIOTIC", "EMISSION", "EM*", "ELL*", "ECLIPSING", "EB*",
        "SPECTROSCOPIC BINARY", "SB*", "RS CVN",
    )),
    # Standard white dwarf
    ("standard_white_dwarf", (
        "WHITE DWARF", "WD*", "WD ", "DA", "DB", "DC", "DO", "DZ", "DQ",
    )),
    # Generic stellar / unknown
    ("ambiguous_stellar_source", (
        "STAR", "*", "SOURCE OF",
    )),
)


_FITTING_STRATEGY_TO_FLAGS: Dict[str, Dict[str, Any]] = {
    "standard_white_dwarf": {
        "allow_standard_wd_fit": True,
        "severity": "ok",
    },
    "binary_or_accretion_candidate": {
        "allow_standard_wd_fit": "cautious",
        "severity": "warning",
    },
    "ambiguous_stellar_source": {
        "allow_standard_wd_fit": "cautious",
        "severity": "warning",
    },
    "non_stellar_contaminant": {
        "allow_standard_wd_fit": False,
        "severity": "blocking",
    },
    "simbad_match_uncertain": {
        "allow_standard_wd_fit": "cautious",
        "severity": "warning",
    },
    "no_simbad_match": {
        "allow_standard_wd_fit": "cautious",
        "severity": "warning",
    },
    "simbad_skipped": {
        "allow_standard_wd_fit": "cautious",
        "severity": "warning",
    },
}


def _haystack(row: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("OTYPE", "OTYPES", "OTYPE_S", "MAIN_ID", "SP_TYPE"):
        val = row.get(key)
        if val:
            parts.append(str(val))
    return " ".join(parts).upper()


def _classify_row(row: Mapping[str, Any]) -> str:
    hay = _haystack(row)
    if not hay:
        return "ambiguous_stellar_source"
    for tag, needles in _RULES:
        for needle in needles:
            if needle in hay:
                return tag
    return "ambiguous_stellar_source"


def _angular_distance(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("ang_dist", "DISTANCE_RESULT", "DISTANCE", "sep_arcsec"):
        val = row.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _redshift(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("Z_VALUE", "REDSHIFT", "RVZ_RADVEL"):
        val = row.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def build_simbad_prior(simbad_crossmatch: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pure function: simbad payload -> simbad_prior dict."""
    sxm = simbad_crossmatch or {}
    status = sxm.get("status")

    if status == "skipped":
        return {
            "status": "skipped",
            "main_id": None,
            "otype": None,
            "sp_type": None,
            "classification": "simbad_skipped",
            "fitting_strategy": "simbad_skipped",
            "allow_standard_wd_fit": "cautious",
            "severity": "warning",
            "science_flags": ["simbad query was skipped — type prior unavailable"],
            "recommended_next_steps": [
                "rerun with simbad enabled before claiming a stellar classification",
            ],
        }

    rows: List[Mapping[str, Any]] = list(sxm.get("rows") or [])
    if not rows:
        return {
            "status": "no_match",
            "main_id": None,
            "otype": None,
            "sp_type": None,
            "classification": "no_simbad_match",
            "fitting_strategy": "no_simbad_match",
            "allow_standard_wd_fit": "cautious",
            "severity": "warning",
            "science_flags": ["no Simbad source within search radius"],
            "recommended_next_steps": [
                "verify coordinates and search radius",
                "treat any subsequent WD fit as provisional until type is established",
            ],
        }

    row = rows[0]
    classification = _classify_row(row)
    flags = _FITTING_STRATEGY_TO_FLAGS.get(classification, {})
    ang_dist = _angular_distance(row)
    redshift = _redshift(row)

    science_flags: list[str] = []
    if classification == "non_stellar_contaminant":
        science_flags.append(
            f"Simbad classifies {row.get('MAIN_ID')!r} as a non-stellar object — "
            "standard white-dwarf fitting is not appropriate."
        )
    if classification == "binary_or_accretion_candidate":
        science_flags.append(
            f"Simbad type for {row.get('MAIN_ID')!r} suggests binary / accreting / emission "
            "system — single-DA interpretation must be cross-checked."
        )
    if ang_dist is not None and ang_dist > 5.0:
        science_flags.append(
            f"Simbad match offset {ang_dist:.2f} arcsec is large; classification may belong "
            "to a different physical source."
        )
        # downgrade
        classification = "simbad_match_uncertain"
        flags = _FITTING_STRATEGY_TO_FLAGS[classification]
    if redshift is not None and redshift > 0.01:
        science_flags.append(
            f"Simbad reports redshift z={redshift:.3f}; this is incompatible with a Galactic WD."
        )
        classification = "non_stellar_contaminant"
        flags = _FITTING_STRATEGY_TO_FLAGS[classification]

    next_steps: list[str] = []
    if classification == "non_stellar_contaminant":
        next_steps.append("skip WD modeling iterations and route to abnormal_report")
        next_steps.append("trigger Claude Code experiment_audit for type-conflict diagnosis")
    elif classification in {"binary_or_accretion_candidate", "ambiguous_stellar_source"}:
        next_steps.append("run baseline fit but require alternative-model comparison")
    elif classification == "standard_white_dwarf":
        next_steps.append("proceed with standard WD pipeline")

    return {
        "status": "ok",
        "main_id": row.get("MAIN_ID"),
        "otype": row.get("OTYPE") or row.get("OTYPES") or row.get("OTYPE_S"),
        "sp_type": row.get("SP_TYPE"),
        "angular_distance_arcsec": ang_dist,
        "redshift": redshift,
        "classification": classification,
        "fitting_strategy": classification,
        "allow_standard_wd_fit": flags.get("allow_standard_wd_fit", "cautious"),
        "severity": flags.get("severity", "warning"),
        "science_flags": science_flags,
        "recommended_next_steps": next_steps,
        "raw_match": {k: row.get(k) for k in ("MAIN_ID", "OTYPE", "SP_TYPE", "RA", "DEC")},
    }


def simbad_prior_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node wrapper. Writes ``03_simbad_prior.json`` artifact."""
    sxm = (state.get("data_fetch") or {}).get("simbad_crossmatch")
    prior = build_simbad_prior(sxm)
    state["simbad_prior"] = prior

    output_root = state.get("output_root")
    if output_root:
        try:
            d = Path(output_root)
            d.mkdir(parents=True, exist_ok=True)
            path = d / "03_simbad_prior.json"
            path.write_text(json.dumps(prior, indent=2, ensure_ascii=False), encoding="utf-8")
            state.setdefault("artifacts", []).append(str(path))
        except Exception:
            # filesystem failures must not break the pipeline
            pass

    return state


__all__ = ["simbad_prior_node", "build_simbad_prior"]
