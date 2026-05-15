"""Auto-generated literature comparison table (UPK 13-c2 style Table 2).

Maintains a small in-code database of well-studied benchmark systems per
source class.  When the target's source_class matches one of these classes,
`build_comparison_table()` produces a 4-row LaTeX `deluxetable*` comparing
the target against 2–4 known systems on the agreed-upon parameters.

Adding a new class:
  1. Append to `KNOWN_SYSTEMS`.
  2. Extend `_PARAMETERS_BY_CLASS` to enumerate the parameter rows for that class.

The values are hardcoded (not LLM-extracted) so the table is reproducible.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional


# ---------------- known systems library -------------------------------------

# Format: list of {name, P_min, e, q_mass_ratio, eclipse_depth_mag,
#                  eclipse_fraction, morphology, age_myr, binary_type, disk_type,
#                  mir_excess, spec_confirmed, bibcodes}
KNOWN_SYSTEMS: Dict[str, List[Dict[str, Any]]] = {
    "sdob_binary_or_single": [
        {
            "name": "ZTF J2130+4420",
            "P_min": 39.34, "e": 0.0, "q_mass_ratio": 0.62,
            "M_sdOB_Msun": 0.337, "M_WD_Msun": 0.545,
            "Teff_sdOB_K": 42400, "logg_sdOB": 5.77,
            "inclination_deg": 86.0,
            "tau_merge_Myr": 17,
            "Roche_lobe_filling": True,
            "discovery_year": 2020,
            "bibcodes": ["2020ApJ...891...45K", "2022ApJ...931...13M"],
        },
        {
            "name": "ZTF J2055+4651",
            "P_min": 56.35, "M_sdOB_Msun": 0.41, "M_WD_Msun": 0.68,
            "Teff_sdOB_K": 26800, "logg_sdOB": 5.50,
            "inclination_deg": 81.0, "Roche_lobe_filling": True,
            "discovery_year": 2020,
            "bibcodes": ["2020ApJ...898L..25K"],
        },
        {
            "name": "LAMOST J1710+5326",
            "P_min": 109.20, "M_sdOB_Msun": 0.44, "M_WD_Msun": 0.54,
            "Teff_sdOB_K": 25164, "inclination_deg": 75.0,
            "Roche_lobe_filling": False, "discovery_year": 2025,
            "bibcodes": ["2025A&A...693A.322Y"],
        },
    ],
    "ultracompact_double_degenerate": [
        {
            "name": "HM Cnc", "P_min": 5.4, "e": 0.0,
            "M_donor_Msun": 0.27, "M_accretor_Msun": 0.55,
            "discovery_year": 1999,
            "bibcodes": ["1999A&A...349L..77I", "2010ApJ...711L.138R"],
        },
        {
            "name": "ES Cet", "P_min": 10.3, "e": 0.0,
            "bibcodes": ["2001ApJ...552L.121W"],
        },
        {
            "name": "ZTF J1539+5027", "P_min": 6.91, "e": 0.0,
            "M_donor_Msun": 0.21, "M_accretor_Msun": 0.61,
            "inclination_deg": 84.0, "discovery_year": 2019,
            "bibcodes": ["2019Natur.571..528B"],
        },
    ],
    "disk_eclipsing_binary": [
        {
            "name": "KH 15D", "P_days": 48.37, "e": 0.68,
            "M_A_Msun": 0.6, "M_B_Msun": 0.5,
            "eclipse_depth_mag": 3.5, "eclipse_fraction": 0.45,
            "age_Myr": 3, "morphology": "flat_bottomed",
            "binary_type": "T Tau + T Tau", "disk_type": "circumbinary",
            "mir_excess": True, "spec_confirmed": True,
            "bibcodes": ["2001ApJ...554L.201H", "2004ApJ...616.1148W",
                         "2006ApJ...644..510W", "2021ApJ...920..145P"],
        },
        {
            "name": "Bernhard-2", "P_days": 63.36, "e": 0.69,
            "M_A_Msun": 1.1, "M_B_Msun": 0.9,
            "eclipse_depth_mag": 1.5, "eclipse_fraction": 0.50,
            "age_Myr": 20, "morphology": "flat_bottomed",
            "binary_type": "MS + MS", "disk_type": "circumbinary",
            "mir_excess": True, "spec_confirmed": True,
            "bibcodes": ["2022ApJ...933L..30Z"],
        },
    ],
    "cataclysmic_variable": [
        {
            "name": "AE Aqr", "P_min": 593.0, "M_WD_Msun": 0.63,
            "M_donor_Msun": 0.57, "discovery_year": 1933,
            "bibcodes": ["1991ApJ...378..674E"],
        },
        {
            "name": "SS Cyg", "P_min": 396.0, "M_WD_Msun": 0.81,
            "M_donor_Msun": 0.55, "discovery_year": 1896,
            "bibcodes": ["2008MNRAS.391.1559P"],
        },
    ],
}


# Parameter rows to emit per source class (label, key, formatter, unit).
_PARAMETERS_BY_CLASS: Dict[str, List[tuple]] = {
    "sdob_binary_or_single": [
        ("Period (min)",          "P_min",          "{:.2f}",  "min"),
        ("M(sdOB) (M$_\\odot$)",  "M_sdOB_Msun",    "{:.3f}",  "M_sun"),
        ("M(WD) (M$_\\odot$)",    "M_WD_Msun",      "{:.3f}",  "M_sun"),
        ("T$_\\mathrm{eff}$ (K)", "Teff_sdOB_K",    "{:.0f}",  "K"),
        ("log\\,$g$",             "logg_sdOB",      "{:.2f}",  "dex"),
        ("Inclination (deg)",     "inclination_deg","{:.0f}",  "deg"),
        ("Roche-lobe filling?",   "Roche_lobe_filling", str,    None),
        ("$\\tau_\\mathrm{merge}$ (Myr)", "tau_merge_Myr", "{:.0f}", "Myr"),
    ],
    "ultracompact_double_degenerate": [
        ("Period (min)",     "P_min",        "{:.2f}", "min"),
        ("M(donor) (M$_\\odot$)", "M_donor_Msun", "{:.3f}", "M_sun"),
        ("M(accretor) (M$_\\odot$)", "M_accretor_Msun", "{:.3f}", "M_sun"),
        ("Inclination (deg)","inclination_deg","{:.0f}", "deg"),
    ],
    "disk_eclipsing_binary": [
        ("Period (days)",     "P_days",          "{:.2f}", "days"),
        ("Eccentricity",      "e",               "{:.2f}", ""),
        ("M$_1$ (M$_\\odot$)","M_A_Msun",        "{:.2f}", "M_sun"),
        ("M$_2$ (M$_\\odot$)","M_B_Msun",        "{:.2f}", "M_sun"),
        ("Eclipse depth (mag)","eclipse_depth_mag","{:.2f}", "mag"),
        ("Eclipse fraction",  "eclipse_fraction","{:.2f}", ""),
        ("Morphology",        "morphology",      str,      None),
        ("Age (Myr)",         "age_Myr",         "{:.0f}", "Myr"),
        ("Binary type",       "binary_type",     str,      None),
        ("Disk type",         "disk_type",       str,      None),
        ("MIR excess",        "mir_excess",      lambda v: "Yes" if v else "No", None),
        ("Spec. confirmed",   "spec_confirmed",  lambda v: "Yes" if v else "Pending", None),
    ],
    "cataclysmic_variable": [
        ("Period (min)",       "P_min",        "{:.1f}", "min"),
        ("M(WD) (M$_\\odot$)", "M_WD_Msun",    "{:.2f}", "M_sun"),
        ("M(donor) (M$_\\odot$)","M_donor_Msun","{:.2f}", "M_sun"),
    ],
}


def _format_value(value: Any, fmt) -> str:
    if value is None:
        return r"\nodata"
    if isinstance(fmt, str):
        try:
            return fmt.format(value)
        except (TypeError, ValueError):
            return r"\nodata"
    if callable(fmt):
        try:
            return fmt(value)
        except Exception:
            return r"\nodata"
    return str(value)


def _target_row_from_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the target's parameter row from per-run artifacts."""
    row: Dict[str, Any] = {"name": state.get("target") or "Target"}
    pp = state.get("published_params") or {}
    for r in pp.get("rows") or []:
        param = r.get("parameter")
        val = r.get("value")
        if val is None:
            continue
        # Map our internal parameter names to comparison-table keys
        if param == "orbital_period_min":
            row.setdefault("P_min", val)
        elif param == "photometric_period_min":
            row.setdefault("P_min", val)
        elif param == "inclination_deg":
            row.setdefault("inclination_deg", val)
        elif param == "logg":
            row.setdefault("logg_sdOB", val)
        elif param == "Teff_K":
            row.setdefault("Teff_sdOB_K", val)
        elif param == "M_donor_Msun":
            row.setdefault("M_sdOB_Msun", val)
            row.setdefault("M_donor_Msun", val)
        elif param == "M_accretor_Msun":
            row.setdefault("M_WD_Msun", val)
            row.setdefault("M_accretor_Msun", val)
        elif param == "tau_merger_Myr":
            row.setdefault("tau_merge_Myr", val)
    # Eccentricity from MCMC posterior
    mcmc = state.get("eclipse_mcmc") or {}
    if mcmc.get("e_pct") and not row.get("e"):
        row["e"] = mcmc["e_pct"][1]
    # Light-curve geometry-derived
    geom = state.get("light_curve_geometry") or {}
    if geom.get("morphology") and not row.get("morphology"):
        row["morphology"] = geom["morphology"]
    if not row.get("P_days") and geom.get("period_min_used"):
        row["P_days"] = geom["period_min_used"] / 1440.0
    return row


def build_comparison_table(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a LaTeX deluxetable string and metadata for the target's class."""
    plan = state.get("analysis_plan") or {}
    source_class = plan.get("source_class") or "unknown"
    benchmarks = KNOWN_SYSTEMS.get(source_class) or []
    params = _PARAMETERS_BY_CLASS.get(source_class) or []
    if not benchmarks or not params:
        # Try disk_eclipsing_binary as fallback when unknown
        if source_class == "unknown":
            benchmarks = KNOWN_SYSTEMS.get("disk_eclipsing_binary") or []
            params = _PARAMETERS_BY_CLASS.get("disk_eclipsing_binary") or []
            if not benchmarks or not params:
                return {"status": "no_benchmarks", "source_class": source_class}
        else:
            return {"status": "no_benchmarks", "source_class": source_class}

    target_row = _target_row_from_state(state)
    all_rows = [target_row, *benchmarks]
    # Build LaTeX
    n_cols = len(all_rows)
    col_spec = "l" + "c" * n_cols
    headers = " & ".join([r"\colhead{Property}"] +
                          [r"\colhead{" + str(r.get("name", "?")).replace("&", r"\&") + "}"
                           for r in all_rows])
    lines: List[str] = [
        r"\begin{deluxetable*}{" + col_spec + "}",
        r"\tablecaption{Comparison with known " + source_class.replace("_", " ") + r" systems"
        + r" \label{tab:comparison}}",
        r"\tablewidth{0pt}",
        r"\tablehead{" + headers + "}",
        r"\startdata",
    ]
    bibcode_set: set = set()
    for label, key, fmt, _unit in params:
        cells = []
        for r in all_rows:
            cells.append(_format_value(r.get(key), fmt))
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\enddata")
    # Build comments + collect bibcodes
    comments = [r"\tablecomments{Target is this work; comparisons drawn from the listed references."]
    for r in all_rows[1:]:
        bibs = r.get("bibcodes") or []
        if not bibs:
            continue
        bibcode_set.update(bibs)
        bib_cite = "\\citep{" + ",".join(bibs) + "}"
        comments.append(f"  {r.get('name')}: {bib_cite}.")
    comments.append("}")
    lines.append("\n".join(comments))
    lines.append(r"\end{deluxetable*}")
    return {
        "status": "ok",
        "source_class": source_class,
        "n_benchmarks": len(benchmarks),
        "n_parameters": len(params),
        "bibcodes": sorted(bibcode_set),
        "latex": "\n".join(lines),
        "target_row": target_row,
    }


__all__ = ["KNOWN_SYSTEMS", "build_comparison_table"]
