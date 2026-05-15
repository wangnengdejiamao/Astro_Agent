"""LangGraph orchestration for the astronomy Chief Investigator agent."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from .state import AnalysisState
from . import (
    ads_live, codex_style, codex_tool, comparison_table, figure_synthesizer,
    hypothesis_scaffold, kg_writeback, latex_compile, memory_advisor,
    method_learning, novelty_detector, paper_agents, paper_orchestra,
    per_source_rag, physics_checks, published_params, reflexion, tools,
)
from .graph_visualization_agent import run_graph_agent
from .llm_client import LLMClient, load_model_config
from .prompts import retrieval as retrieval_prompts


def _append_artifact(state: AnalysisState, path: str) -> None:
    state.setdefault("artifacts", [])
    if path not in state["artifacts"]:
        state["artifacts"].append(path)


def _run_dir(state: AnalysisState) -> Path:
    root = Path(state["output_root"]).resolve()
    return tools.ensure_dir(root)


def resolve_node(state: AnalysisState) -> AnalysisState:
    resolved = tools.resolve_target(
        state["target"],
        state.get("ra_deg"),
        state.get("dec_deg"),
    )
    state["resolved"] = resolved
    if resolved.get("status") == "ok":
        state["ra_deg"] = resolved["ra_deg"]
        state["dec_deg"] = resolved["dec_deg"]
    else:
        state.setdefault("errors", []).extend(resolved.get("errors", []))
    _append_artifact(state, tools.json_dump(_run_dir(state) / "01_resolved_target.json", resolved))
    return state


def data_fetcher_node(state: AnalysisState) -> AnalysisState:
    resolved = state.get("resolved", {})
    if resolved.get("status") != "ok":
        # D2 — propagate target identity failures (mismatch / needs_human)
        # so downstream nodes treat this as a hard fail-closed.
        skipped = {
            "status": "skipped",
            "reason": "target not resolved",
            "resolved_errors": resolved.get("errors", []),
            "target_identity_unresolved": True,
        }
        if resolved.get("mismatch"):
            skipped["mismatch"] = resolved["mismatch"]
        state["target_identity_unresolved"] = True
        state["data_fetch"] = skipped
        # Always persist the artifact so subsequent debugging sees what was
        # skipped and why. Round 2 bug: missing 02_data_fetch.json on resolve failure.
        _append_artifact(state, tools.json_dump(_run_dir(state) / "02_data_fetch.json", skipped))
        # Also write empty published_params + source_rag artifacts so the
        # frontend trace doesn't show them as `pending`.
        state["published_params"] = {"rows": [], "n_rows": 0, "skipped_reason": "resolve_failed"}
        _append_artifact(state, tools.json_dump(_run_dir(state) / "02c_published_params.json", state["published_params"]))
        state["source_rag"] = {"sqlite_path": None, "n_refs": 0, "skipped_reason": "resolve_failed"}
        _append_artifact(state, tools.json_dump(_run_dir(state) / "02d_source_rag.json", state["source_rag"]))
        return state

    ra = float(state["ra_deg"])
    dec = float(state["dec_deg"])
    if state.get("skip_simbad", False):
        simbad = {"status": "skipped", "reason": "skip_simbad enabled"}
    else:
        simbad = tools.query_simbad_crossmatch(ra, dec)
    existing_run = state.get("astrotool_run")
    if existing_run:
        output_root = Path(existing_run).resolve()
        astrotool = {
            "status": "existing",
            "output_root": str(output_root),
            "note": "using caller-supplied astro_toolbox output directory",
            "existing_outputs": tools.summarize_output_root(output_root),
        }
    else:
        output_root = _run_dir(state) / "astrotool_run"
        astrotool = tools.run_astrotool(
            target=state["target"],
            ra_deg=ra,
            dec_deg=dec,
            output_root=output_root,
            dry_run=bool(state.get("dry_run", True)),
            force=bool(state.get("force", False)),
        )
    pack = {
        "status": astrotool.get("status"),
        "simbad_crossmatch": simbad,
        "astrotool": astrotool,
        "existing_outputs": astrotool.get("existing_outputs", {}),
    }
    state["data_fetch"] = pack
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02_data_fetch.json", pack))

    # Build the published-parameters table: literature values mined from
    # SIMBAD abstracts + deterministic this-work measurements pulled out of
    # the astro_toolbox products (period_analysis.csv, rv_analysis.csv,
    # orbit_traceback.txt). This is what lifts the drafter out of
    # "withheld pending QA clearance" boilerplate.
    try:
        pp_root = Path(str(output_root)) if existing_run or astrotool.get("output_root") else None
        if astrotool.get("output_root"):
            pp_root = Path(astrotool["output_root"])
        if existing_run:
            pp_root = Path(existing_run)
        pp_table = (
            published_params.build_published_params_table(
                pp_root,
                target=state.get("target"),
                require_target_match=True,
            )
            if pp_root else {"rows": [], "n_rows": 0, "by_parameter": {}}
        )
    except Exception as exc:
        pp_table = {"rows": [], "n_rows": 0, "by_parameter": {}, "error": f"{type(exc).__name__}: {exc}"}

    # Fallback: if no astrotool root, still record Gaia DR3 parallax/pm as
    # this-work measurements so the drafter has at least the astrometric
    # parameters to quote.
    if pp_table.get("n_rows", 0) == 0 and state.get("ra_deg") is not None:
        try:
            from astro_toolbox.orbit_traceback import get_gaia_astrometry as _get_gaia
            live = _get_gaia(float(state["ra_deg"]), float(state["dec_deg"]), radius_arcsec=5.0)
            if live and live.get("source_id"):
                new_rows = []
                if live.get("Plx") is not None:
                    new_rows.append({
                        "parameter": "parallax_mas",
                        "value": float(live["Plx"]),
                        "error": float(live.get("e_Plx") or 0.0) or None,
                        "unit": "mas",
                        "bibcode": None,
                        "source_kind": "this_work_gaia_dr3",
                        "snippet": f"Gaia DR3 source {live.get('source_id')}",
                    })
                if live.get("pmRA") is not None:
                    new_rows.append({
                        "parameter": "pmRA_mas_per_yr",
                        "value": float(live["pmRA"]),
                        "error": float(live.get("e_pmRA") or 0.0) or None,
                        "unit": "mas/yr",
                        "bibcode": None,
                        "source_kind": "this_work_gaia_dr3",
                        "snippet": f"Gaia DR3 source {live.get('source_id')}",
                    })
                if live.get("pmDE") is not None:
                    new_rows.append({
                        "parameter": "pmDE_mas_per_yr",
                        "value": float(live["pmDE"]),
                        "error": float(live.get("e_pmDE") or 0.0) or None,
                        "unit": "mas/yr",
                        "bibcode": None,
                        "source_kind": "this_work_gaia_dr3",
                        "snippet": f"Gaia DR3 source {live.get('source_id')}",
                    })
                pp_table["rows"] = (pp_table.get("rows") or []) + new_rows
                pp_table["n_rows"] = len(pp_table["rows"])
                pp_table["n_this_work"] = sum(1 for r in pp_table["rows"] if str(r.get("source_kind","")).startswith("this_work"))
                by_param = pp_table.get("by_parameter", {}) or {}
                for r in new_rows:
                    by_param[r["parameter"]] = by_param.get(r["parameter"], 0) + 1
                pp_table["by_parameter"] = by_param
                pp_table["gaia_fallback_used"] = True
        except Exception as exc:
            pp_table["gaia_fallback_error"] = f"{type(exc).__name__}: {exc}"
    state["published_params"] = pp_table
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02c_published_params.json", pp_table))

    # Build a per-source RAG (one SQLite per run) from SIMBAD references.
    # The drafter will query this in addition to the global white-dwarf RAG so
    # that \citep keys point at papers that actually study this source.
    try:
        if pp_root and pp_root.exists():
            sr_db = _run_dir(state) / "source_rag.sqlite"
            source_rag_info = per_source_rag.build_source_rag(
                astrotool_root=pp_root,
                target=state.get("target"),
                sqlite_path=sr_db,
            )
        else:
            source_rag_info = {"sqlite_path": None, "n_refs": 0, "n_refs_mentioning_target": 0}
    except Exception as exc:
        source_rag_info = {"error": f"{type(exc).__name__}: {exc}", "n_refs": 0}
    state["source_rag"] = source_rag_info
    _append_artifact(
        state,
        tools.json_dump(_run_dir(state) / "02d_source_rag.json", source_rag_info),
    )
    return state


_SOURCE_CLASS_BY_SIMBAD = {
    # SIMBAD main_type or other_types token  ->  internal physics route
    # Keep the keys lowercase; classifier lowercases the SIMBAD string before lookup.
    "wd": "single_white_dwarf",
    "wd*": "single_white_dwarf",
    "da*": "single_white_dwarf",
    "db*": "single_white_dwarf",
    "dz*": "single_white_dwarf",
    "whitedwarf": "single_white_dwarf",
    "white dwarf": "single_white_dwarf",
    "hotsubdwarf": "sdob_binary_or_single",
    "sdb": "sdob_binary_or_single",
    "sdob": "sdob_binary_or_single",
    "sd*": "sdob_binary_or_single",
    "cataclyv*": "cataclysmic_variable",
    "cv*": "cataclysmic_variable",
    "novae": "cataclysmic_variable",
    "amher": "polar",
    "polar": "polar",
    "amcvn": "ultracompact_double_degenerate",
    "eclbin": "eclipsing_binary",
    "binary": "eclipsing_binary",
    "**": "wide_binary",
}


def _classify_source(state: AnalysisState) -> Dict[str, Any]:
    """Map SIMBAD types to a physics route. Returns {class, source_type, evidence}."""
    simbad = (state.get("data_fetch", {}) or {}).get("simbad_crossmatch", {}) or {}
    raw_types: List[str] = []

    # 1) direct top-level keys (some upstream paths populate these flat)
    for key in ("main_type", "object_type", "otype"):
        value = simbad.get(key)
        if value:
            raw_types.append(str(value))

    # 2) rows[0] from the astroquery Simbad table (the structure that
    # query_simbad_crossmatch actually emits)
    rows = simbad.get("rows") if isinstance(simbad, dict) else None
    if isinstance(rows, list) and rows:
        first = rows[0] or {}
        for key in (
            "main_type", "OTYPE", "OBJECT_TYPE", "OTYPES",
            "otype", "otypes", "OType", "OTypes",
        ):
            value = first.get(key)
            if value:
                raw_types.append(str(value))

    others = simbad.get("other_types") or simbad.get("otypes") or []
    if isinstance(others, str):
        others = [token.strip() for token in others.split(",") if token.strip()]
    raw_types.extend(str(token) for token in others)

    # 3) Fallback: read simbad_references.txt produced by astro_toolbox.
    # Its header carries `# Object type: HotSubdwarf` which is enough.
    astrotool = (state.get("data_fetch", {}) or {}).get("astrotool", {}) or {}
    out_root = astrotool.get("output_root") or state.get("astrotool_run")
    if out_root and not raw_types:
        try:
            from pathlib import Path as _P
            txt = _P(out_root) / "simbad_references.txt"
            if txt.exists():
                head = txt.read_text(encoding="utf-8", errors="replace")[:1000]
                m = re.search(r"Object type:\s*(\S+)", head)
                if m:
                    raw_types.append(m.group(1))
        except Exception:
            pass

    tokens = [t.lower().replace(" ", "") for t in raw_types]
    matched_class = None
    matched_token = None
    for token in tokens:
        if token in _SOURCE_CLASS_BY_SIMBAD:
            matched_class = _SOURCE_CLASS_BY_SIMBAD[token]
            matched_token = token
            break
    if matched_class is None:
        # Substring fallback (e.g. "HotSubdwarfStar" -> hotsubdwarf)
        joined = "|".join(tokens)
        for key, route in _SOURCE_CLASS_BY_SIMBAD.items():
            if key in joined:
                matched_class = route
                matched_token = key
                break
    return {
        "source_class": matched_class or "unknown",
        "matched_token": matched_token,
        "raw_simbad_types": raw_types,
    }


def memory_advisor_node(state: AnalysisState) -> AnalysisState:
    """CORAL-style: query the cross-run learning ledger BEFORE the structure
    planner picks a method, so historical success/failure rates inform the
    plan."""
    # The ledger lives at output/analysis_agent/_learning_ledger.sqlite
    pkg_root = Path(__file__).resolve().parent.parent
    ledger = pkg_root / "output" / "analysis_agent" / "_learning_ledger.sqlite"

    # We don't know the source_class yet (planner runs next). Run a soft
    # "no-class" preview query that returns *all* method success rates so
    # the planner can use it as a tie-breaker between competing routes.
    try:
        advice_all = memory_advisor.query_advice(
            source_class="*",
            source_id=state.get("target"),
            ledger_path=ledger,
        )
        # And one with the resolved-only metadata (no source class yet).
        advice = advice_all
    except Exception as exc:
        advice = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    state["memory_advice"] = advice
    _append_artifact(
        state,
        tools.json_dump(_run_dir(state) / "02a_memory_advice.json", advice),
    )
    return state


def structure_planner_node(state: AnalysisState) -> AnalysisState:
    """Choose the science route from available data before model fitting."""
    output = state.get("data_fetch", {}).get("existing_outputs", {})
    run_summary = output.get("run_summary", {}) or {}
    module_rows = output.get("module_rows", [])
    successful_statuses = {"ok", "completed", "written", "existing", "available"}
    failed_statuses = {"error", "empty", "failed", "missing", "unavailable"}
    successful_module_names = {
        str(row.get("module", "")).lower()
        for row in module_rows
        if str(row.get("status", "")).lower() in successful_statuses
    }
    failed_module_names = {
        str(row.get("module", "")).lower()
        for row in module_rows
        if str(row.get("status", "")).lower() in failed_statuses
    }
    files = {str(name).lower() for name in output.get("sample_files", [])}

    # ------------------------------------------------------------------
    # Step 0: SIMBAD-type based dispatch.  This decides WHICH physics model
    # the modeling iterations will call.  Without this step, the planner
    # would silently treat a hot subdwarf binary as a single DA white dwarf
    # and the Koester grid fit will fail forever.
    # ------------------------------------------------------------------
    source_class_info = _classify_source(state)
    source_class = source_class_info["source_class"]

    # Now that we know source_class, refine the memory_advisor query so
    # success-rate stats are filtered to this class.  CORAL pattern: the
    # planner consults persistent ledger of past runs before picking a method.
    try:
        pkg_root = Path(__file__).resolve().parent.parent
        ledger = pkg_root / "output" / "analysis_agent" / "_learning_ledger.sqlite"
        refined_advice = memory_advisor.query_advice(
            source_class=source_class,
            source_id=state.get("target"),
            ledger_path=ledger,
        )
        state["memory_advice"] = refined_advice
    except Exception:
        refined_advice = state.get("memory_advice", {})

    spectral_modules = {"sdss", "desi", "lamost", "hst", "galah", "koa"}
    spectral_files = {
        "sdss_spectrum.csv",
        "desi_spectrum.csv",
        "lamost_spectrum.csv",
        "hst_spectrum.csv",
        "koa_spectrum.csv",
    }
    has_spectrum = (
        bool(run_summary.get("spectra_available"))
        or any("spectrum" in name or name in spectral_modules for name in successful_module_names)
        or any(name in files for name in spectral_files)
    )
    has_hst = bool(run_summary.get("hst_spectrum_available")) or any("hst" in name for name in successful_module_names | files)
    has_sed = bool(run_summary.get("sed_available")) or any(name == "sed" for name in successful_module_names) or "sed.png" in files
    has_hrd = any("hr" in name for name in successful_module_names) or "hr_diagram.png" in files
    has_rv = bool(run_summary.get("rv_report_available")) or any("rv" in name for name in successful_module_names)
    has_period = bool(run_summary.get("period_products_available")) or any("period" in name or "fold" in name for name in files)

    if has_spectrum:
        route = "spectroscopy_plus_sed"
        parameter_strategy = [
            "Use optical/UV spectra for classification, line masks, Teff/logg/composition constraints, and emission-line rejection.",
            "Use SED and HR diagram as consistency checks against spectroscopic parameters.",
        ]
    elif has_sed and has_hrd:
        route = "photometric_hrd_sed_fallback"
        parameter_strategy = [
            "No spectrum is available; infer provisional Teff/radius/luminosity from SED plus Gaia/HR-diagram position.",
            "Do not claim spectral type, line detections, radial velocities, atmosphere composition, or logg precision from photometry alone.",
            "Use wider priors and require human review before final WD mass/cooling-age claims.",
        ]
    elif has_sed:
        route = "sed_only_fallback"
        parameter_strategy = [
            "Use SED-only fitting for rough temperature/radius constraints if distance/extinction priors exist.",
            "Mark logg/mass/cooling age as weakly constrained and block final ApJ parameter tables.",
        ]
    else:
        route = "insufficient_data"
        parameter_strategy = [
            "Fetch or repair photometry/spectroscopy before physical-model fitting.",
        ]

    # Append class-aware extra guidance so the modeling stage knows it should NOT
    # apply a single-WD model when SIMBAD already marked the source as sdOB / CV /
    # polar / AMCVn.
    class_guidance = {
        "single_white_dwarf": [
            "Apply DA/DB white-dwarf atmosphere grid (Koester) for primary SED fitting.",
        ],
        "sdob_binary_or_single": [
            "Do NOT apply a single DA-WD Koester fit; this is a hot subdwarf source.",
            "Use a hot subdwarf (TLUSTY-BSTAR or Nemeth) atmosphere with optional WD secondary.",
            "Treat any RV measurement using a DA-WD template as a consistency check only.",
        ],
        "cataclysmic_variable": [
            "Decompose into WD primary + accretion-disc + donor; do not fit single-star SED.",
        ],
        "polar": [
            "Look for Zeeman patterns and cyclotron humps; conventional SED fitting is inadequate.",
        ],
        "ultracompact_double_degenerate": [
            "Use double-WD or sdOB+WD decomposition; expect short orbital period from photometry.",
        ],
        "eclipsing_binary": [
            "Use two-component SED with light-curve modeling (PHOEBE-like) before single-star claims.",
        ],
        "wide_binary": [
            "Resolve Gaia DR3 companions; analyze each component separately.",
        ],
        "unknown": [
            "SIMBAD did not return a recognizable type token; fall back to data-driven route.",
        ],
    }.get(source_class, [])

    # Check whether the chosen pipeline module actually exists on disk so that
    # the QA gate can avoid an infinite retry loop into a not-yet-implemented
    # module.  An sdOB / CV / polar pipeline that has not been built will set
    # `pipeline_implemented=False` and the model-mismatch retry path will skip
    # straight to drafter rather than replanning indefinitely.
    pipeline_module = {
        "single_white_dwarf": "astro_toolbox.wd_fitting",
        "sdob_binary_or_single": "astro_toolbox.sdob_fitting",       # to be implemented (A1 follow-up)
        "cataclysmic_variable": "astro_toolbox.cv_fitting",          # to be implemented
        "polar": "astro_toolbox.polar_fitting",                      # to be implemented
        "ultracompact_double_degenerate": "astro_toolbox.dwd_fitting",
        "eclipsing_binary": "astro_toolbox.binary_lc_fitting",
        "wide_binary": "astro_toolbox.wd_fitting",
        "unknown": "astro_toolbox.wd_fitting",
    }.get(source_class, "astro_toolbox.wd_fitting")
    try:
        module_file = Path(__file__).resolve().parent.parent / Path(*pipeline_module.split("."))
        pipeline_implemented = (module_file.with_suffix(".py")).exists()
    except Exception:
        pipeline_implemented = False

    plan = {
        "status": "written",
        "route": route,
        "source_class": source_class,
        "source_class_info": source_class_info,
        "structure_planner_retry_count": int(state.get("structure_planner_retry_count", 0) or 0),
        "pipeline_implemented": pipeline_implemented,
        "fitting_pipeline_module": pipeline_module,
        "memory_advisor_recommended": (refined_advice or {}).get("recommended_method"),
        "memory_advisor_seen_n_class_runs": sum(
            int(m.get("n_total") or 0) for m in (refined_advice or {}).get("method_success_rate") or []
        ),
        "available_evidence": {
            "spectrum": has_spectrum,
            "hst_spectrum": has_hst,
            "sed": has_sed,
            "hr_diagram": has_hrd,
            "rv": has_rv,
            "period_products": has_period,
            "successful_modules_used_as_evidence": sorted(successful_module_names),
            "failed_modules_not_used_as_evidence": sorted(failed_module_names),
        },
        "parameter_strategy": parameter_strategy + class_guidance,
        "fitting_priority": [
            "1. identity/crossmatch and unit verification",
            "2. spectral classification and emission-line checks when spectra exist",
            "3. SED fit with residual diagnostics",
            "4. HRD/Gaia consistency and distance/extinction systematics",
            "5. RV/period/binary checks when time-domain or spectra exist",
            "6. final parameters only after Supervisor and QA clearance",
        ],
    }
    state["analysis_plan"] = plan
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02b_analysis_plan.json", plan))

    # Build the hypothesis-test plan from the source class + available evidence.
    # This declares which physical hypotheses the agent SHOULD test (even if the
    # fitting modules are not yet implemented), so the drafter can write an
    # honest Methods section about which competing interpretations were
    # considered and which were ruled out by data or by missing tools.
    # Augment available_evidence with cross-cutting flags the data_fetcher
    # observable list doesn't carry directly (Gaia astrometry, MIR excess, etc.).
    evidence_aug = dict(plan["available_evidence"])
    # Gaia astrometry: present if cluster_membership ran with gaia data OR
    # published_params has parallax/pmRA/pmDE rows.
    pp = state.get("published_params") or {}
    has_gaia = bool(any(
        r.get("source_kind") == "this_work_gaia_dr3"
        for r in (pp.get("rows") or [])
    ))
    if not has_gaia:
        cm = state.get("cluster_membership") or {}
        has_gaia = bool((cm.get("gaia_astrometry") or {}).get("pmra") is not None)
    evidence_aug["Gaia_astrometry"] = has_gaia
    # MIR excess: a simple heuristic — module success on WISE_lightcurve OR W3/W4 phot.
    successful_set = set(plan["available_evidence"].get("successful_modules_used_as_evidence") or [])
    evidence_aug["MIR_excess"] = any(m in successful_set for m in ("wise_lightcurve", "wise_photometry", "wise"))
    evidence_aug["UV_spectrum"] = bool(plan["available_evidence"].get("hst_spectrum"))
    try:
        hp = hypothesis_scaffold.hypothesis_plan_for(
            source_class,
            available_evidence=evidence_aug,
        )
    except Exception as exc:
        hp = {"error": f"{type(exc).__name__}: {exc}", "source_class": source_class}
    state["hypothesis_plan"] = hp
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02f_hypothesis_plan.json", hp))
    return state


def cluster_membership_node(state: AnalysisState) -> AnalysisState:
    """Compute χ²_spat / χ²_kin / RV-σ against Hunt+2023 open clusters.

    Astrometry source priority:
      1. astro_toolbox orbit_traceback.txt header (if a prior run exists)
      2. live Gaia DR3 ADQL query via astro_toolbox.orbit_traceback.get_gaia_astrometry
    Then evaluates either a user-named cluster (state['target_cluster']) or
    the top nearest candidates.  This makes the node useful even for fresh
    sources that have no astrotool_run yet.
    """
    from pathlib import Path as _P
    out_root = (state.get("data_fetch", {}).get("astrotool", {}) or {}).get("output_root") \
        or state.get("astrotool_run")

    gaia: Dict[str, Any] = {}

    # Try astrotool orbit_traceback.txt first (cached + offline).
    if out_root:
        tb = _P(out_root) / "orbit_traceback.txt"
        if tb.exists():
            text = tb.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"Parallax:\s*([\d\.\-]+)\s*[±\+\-]+\s*([\d\.]+)\s*mas", text)
            if m:
                gaia["parallax"] = float(m.group(1))
                gaia["e_parallax"] = float(m.group(2))
            m = re.search(r"pmRA:\s*([\d\.\-]+)\s*[±\+\-]+\s*([\d\.]+)", text)
            if m:
                gaia["pmra"] = float(m.group(1))
                gaia["e_pmra"] = float(m.group(2))
            m = re.search(r"pmDE:\s*([\d\.\-]+)\s*[±\+\-]+\s*([\d\.]+)", text)
            if m:
                gaia["pmde"] = float(m.group(1))
                gaia["e_pmde"] = float(m.group(2))
            m = re.search(r"^\#?\s*RA\s*=\s*([\d\.\-]+).*?DEC\s*=\s*([\d\.\-]+)", text, flags=re.MULTILINE)
            if m:
                gaia.setdefault("ra_deg", float(m.group(1)))
                gaia.setdefault("dec_deg", float(m.group(2)))

    # Fallback to live Gaia DR3 ADQL when key astrometry is missing.
    # This makes the cluster-membership node work for fresh sources that
    # have no astrotool_run cached locally.
    needs_pm = ("pmra" not in gaia) or ("pmde" not in gaia)
    if needs_pm and state.get("ra_deg") is not None and state.get("dec_deg") is not None:
        try:
            from astro_toolbox.orbit_traceback import get_gaia_astrometry as _get_gaia
            ra = float(state["ra_deg"])
            dec = float(state["dec_deg"])
            live = _get_gaia(ra, dec, radius_arcsec=5.0)
            if live and live.get("source_id"):
                gaia.setdefault("parallax", float(live.get("Plx") or live.get("parallax") or "nan"))
                gaia.setdefault("e_parallax", float(live.get("e_Plx") or live.get("e_parallax") or "nan"))
                gaia.setdefault("pmra", float(live.get("pmRA") or live.get("pmra") or "nan"))
                gaia.setdefault("e_pmra", float(live.get("e_pmRA") or live.get("e_pmra") or "nan"))
                gaia.setdefault("pmde", float(live.get("pmDE") or live.get("pmde") or "nan"))
                gaia.setdefault("e_pmde", float(live.get("e_pmDE") or live.get("e_pmde") or "nan"))
                gaia["source"] = "gaia_dr3_live"
                gaia["gaia_source_id"] = str(live.get("source_id"))
        except Exception as exc:
            gaia.setdefault("live_query_error", f"{type(exc).__name__}: {exc}")

    ra = float(state.get("ra_deg") or gaia.get("ra_deg") or float("nan"))
    dec = float(state.get("dec_deg") or gaia.get("dec_deg") or float("nan"))
    if not (math.isfinite(ra) and math.isfinite(dec)):
        state["cluster_membership"] = {"status": "skipped", "reason": "no coordinates"}
        _append_artifact(state, tools.json_dump(_run_dir(state) / "02e_cluster_membership.json", state["cluster_membership"]))
        return state

    rv = None
    e_rv = None
    pp = state.get("published_params") or {}
    for row in pp.get("rows", []) or []:
        if row.get("parameter") == "radial_velocity_km_s" and row.get("value") is not None:
            rv = float(row["value"])
            if row.get("error") is not None:
                e_rv = float(row["error"])
            break

    try:
        from astro_toolbox.cluster_membership import membership as _cm
        result = _cm(
            ra_deg=ra,
            dec_deg=dec,
            pmra=gaia.get("pmra"),
            e_pmra=gaia.get("e_pmra"),
            pmde=gaia.get("pmde"),
            e_pmde=gaia.get("e_pmde"),
            parallax=gaia.get("parallax"),
            e_parallax=gaia.get("e_parallax"),
            rv=rv,
            e_rv=e_rv,
            cluster_name=state.get("target_cluster"),
        )
        result["gaia_astrometry"] = gaia  # so paper draft / debug can see what was used
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "gaia_astrometry": gaia}

    # D4 — strict 4-criteria joint verdict override. The upstream toolbox
    # returns spatial+kin only; per L2 domain prior 9, membership requires
    # ALL of: chi^2_spat <= 9, chi^2_kin <= 9, |rv_offset_sigma| <= 3,
    # AND traceback_time_myr <= cluster_age_myr. Anything else → insufficient.
    _SPAT_KIN_THRESH = 9.0
    _RV_SIGMA_THRESH = 3.0
    for cand in (result.get("candidates") or []):
        spat = cand.get("chi2_spat")
        kin = cand.get("chi2_kin")
        rv_sig = cand.get("rv_offset_sigma")
        tb = cand.get("traceback_time_myr")
        age = cand.get("cluster_age_myr")
        missing: List[str] = []
        if spat is None or (isinstance(spat, float) and math.isnan(spat)):
            missing.append("chi2_spat")
        elif spat > _SPAT_KIN_THRESH:
            missing.append(f"chi2_spat>{_SPAT_KIN_THRESH}")
        if kin is None or (isinstance(kin, float) and math.isnan(kin)):
            missing.append("chi2_kin")
        elif kin > _SPAT_KIN_THRESH:
            missing.append(f"chi2_kin>{_SPAT_KIN_THRESH}")
        if rv_sig is None or (isinstance(rv_sig, float) and math.isnan(rv_sig)):
            missing.append("rv_offset_sigma")
        elif abs(rv_sig) > _RV_SIGMA_THRESH:
            missing.append(f"|rv_offset_sigma|>{_RV_SIGMA_THRESH}")
        if tb is None or (isinstance(tb, float) and math.isnan(tb)):
            missing.append("traceback_time_myr")
        elif age is not None and not math.isnan(age) and tb > age:
            missing.append("traceback_time>cluster_age")
        cand["joint_criteria"] = {
            "spat_thresh": _SPAT_KIN_THRESH,
            "kin_thresh": _SPAT_KIN_THRESH,
            "rv_thresh_sigma": _RV_SIGMA_THRESH,
            "missing_or_failing": missing,
        }
        if not missing:
            cand["verdict_strict"] = ["member"]
        else:
            cand["verdict_strict"] = ["insufficient"]
            cand["joint_criteria"]["reason"] = "; ".join(missing)
    # Top-level result also gets a joint summary so downstream consumers
    # (paper_qc cluster_joint_criteria check, evidence_manifest) see it.
    if result.get("candidates"):
        any_member = any(c.get("verdict_strict") == ["member"]
                         for c in result["candidates"])
        result["strict_verdict"] = "member" if any_member else "insufficient"

    state["cluster_membership"] = result
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02e_cluster_membership.json", result))
    return state


def extinction_node(state: AnalysisState) -> AnalysisState:
    """Look up A_V at the source line of sight (Bayestar2019 / SFD98 / Green19).

    D3 — fail-closed: when query_av returns a fallback / unsupported
    provenance, this node now emits status=unavailable and OMITS A_V/E_B_V
    so the downstream drafter (via the evidence_manifest) treats it as
    withheld_unsupported_prov rather than silently quoting a fallback.
    """
    from .prompts.wd_domain import ACCEPTED_EXTINCTION_PROVENANCES
    try:
        from astro_toolbox.extinction import query_av
        ra = state.get("ra_deg")
        dec = state.get("dec_deg")
        if ra is None or dec is None:
            result = {"status": "skipped", "reason": "no coordinates"}
        else:
            # Distance from Gaia parallax in published_params if available.
            distance_pc = None
            pp = state.get("published_params") or {}
            for row in pp.get("rows", []) or []:
                if row.get("parameter") == "parallax_mas" and row.get("value"):
                    plx = float(row["value"])
                    if plx > 0:
                        distance_pc = 1000.0 / plx
                    break
            av_info = query_av(float(ra), float(dec), distance_pc=distance_pc)
            prov = str(av_info.get("provenance") or "").lower()
            accepted = any(p in prov for p in ACCEPTED_EXTINCTION_PROVENANCES)
            if not accepted:
                # Soft fail-closed: keep the artifact for audit but mark
                # status=unavailable and DROP A_V / E_B_V so the drafter
                # cannot quote them.
                result = {
                    "status": "unavailable",
                    "reason": (
                        f"A_V provenance `{prov or 'unknown'}` is not in the "
                        "accepted set {SFD98, Planck13, Green19, Bayestar2019, "
                        "Lallement, 3D-dust}; refusing to publish a fallback value."
                    ),
                    "rejected_provenance": prov or "unknown",
                    "raw_av_info": av_info,  # retained for audit, not for drafter
                }
            else:
                result = {"status": "ok", **av_info}
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    state["extinction"] = result
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02g_extinction.json", result))
    return state


def sed_decoupled_node(state: AnalysisState) -> AnalysisState:
    """Run the SED 3-step decoupled fit (Lin+ 2025 UPK 13-c2 procedure).

    Needs at least an SED CSV from astrotool.  When state['flux_high'] +
    state['flux_low'] are supplied (eclipse mode), runs F_high/F_low/F_diff;
    otherwise falls back to single-state F_high fit on whatever SED is available.
    """
    try:
        from astro_toolbox import sed_decoupled as _sed
        a_v = float((state.get("extinction") or {}).get("A_V") or 0.0)
        # Try to read SED photometry from astrotool's sed_photometry.csv
        out_root = (state.get("data_fetch", {}).get("astrotool", {}) or {}).get("output_root") \
            or state.get("astrotool_run")
        flux_high = state.get("flux_high")
        flux_low = state.get("flux_low")
        if not flux_high and out_root:
            csv_path = Path(out_root) / "sed_photometry.csv"
            if csv_path.exists():
                import csv as _csv
                flux_high = []
                with csv_path.open() as fh:
                    for row in _csv.DictReader(fh):
                        try:
                            band = row.get("band") or row.get("Band") or ""
                            wl = float(row.get("wave_A") or row.get("wavelength_A") or 0)
                            f_cgs = float(row.get("flux_cgs") or row.get("F_nu_cgs") or 0)
                            sig = float(row.get("flux_err_cgs") or row.get("sigma_F_nu") or 0)
                            if wl > 0 and f_cgs > 0:
                                flux_high.append({
                                    "band": band, "wave_A": wl,
                                    "F_nu_obs_cgs": f_cgs, "sigma_F_nu": sig,
                                })
                        except (TypeError, ValueError):
                            continue
        if not flux_high:
            result = {"status": "skipped", "reason": "no SED photometry available"}
        else:
            result = _sed.run_three_step(
                flux_high=flux_high, flux_low=flux_low or [], a_v=a_v,
            )
            result.setdefault("status", "ok")
    except Exception as exc:
        import traceback
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()[-2000:]}
    state["sed_decoupled"] = result
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02h_sed_decoupled.json", result))
    return state


def light_curve_geometry_node(state: AnalysisState) -> AnalysisState:
    """Measure ingress duration, eclipse depth, eclipse fraction τ/P, and
    morphology (flat-bottomed vs U-shaped) from any phase-folded light-curve
    CSV produced by astro_toolbox.  Also derive Keplerian a, v_orb, Roche
    radii from the source-class default mass and any orbital period in the
    published_params table.
    """
    from pathlib import Path as _P
    try:
        from astro_toolbox import binary_orbit, ingress_measurement
    except Exception as exc:
        result = {"status": "error", "error": f"import: {type(exc).__name__}: {exc}"}
        state["light_curve_geometry"] = result
        _append_artifact(state, tools.json_dump(_run_dir(state) / "02j_light_curve_geometry.json", result))
        return state

    out_root = (state.get("data_fetch", {}).get("astrotool", {}) or {}).get("output_root") \
        or state.get("astrotool_run")

    # Find orbital period (prefer literature value, else this-work photometric)
    period_min = None
    period_source = None
    pp = state.get("published_params") or {}
    for row in pp.get("rows", []) or []:
        param = str(row.get("parameter", ""))
        if "orbital_period_min" in param and row.get("value"):
            period_min = float(row["value"])
            period_source = f"literature::{row.get('bibcode') or 'unknown'}"
            break
    if period_min is None:
        # Among photometric_period rows, pick the longest (likely true orbital)
        photo = [float(r["value"]) for r in (pp.get("rows") or [])
                 if "photometric_period_min" in str(r.get("parameter", ""))
                 and r.get("value")]
        if photo:
            period_min = max(photo)
            period_source = "this_work_photometric (longest of band-by-band)"

    # Mass total from source class defaults (or external override via state)
    plan = state.get("analysis_plan") or {}
    source_class = plan.get("source_class") or "unknown"
    M_tot_default = {
        "single_white_dwarf": 0.6,
        "sdob_binary_or_single": 0.88,
        "cataclysmic_variable": 1.0,
        "ultracompact_double_degenerate": 0.8,
        "eclipsing_binary": 1.4,
        "polar": 0.9,
        "unknown": 1.0,
    }.get(source_class, 1.0)

    result: Dict[str, Any] = {
        "status": "ok",
        "period_min_used": period_min,
        "period_source": period_source,
        "M_tot_Msun_assumed": M_tot_default,
        "source_class": source_class,
    }

    # Keplerian orbit
    if period_min is not None and period_min > 0:
        P_days = period_min / 1440.0
        orbit = binary_orbit.summarize_orbit(P_days=P_days, M_tot_Msun=M_tot_default)
        result["orbit"] = orbit

    # Trapezoidal ingress measurement from any ZTF / TESS folded CSV
    ingress_results = {}
    if out_root and period_min:
        for csv_name in ("ztf_lightcurve.csv", "tess_lightcurve.csv", "wise_lightcurve.csv"):
            csv = _P(out_root) / csv_name
            if not csv.exists():
                continue
            # ZTF CSV has band rows mixed; pick one band (the first with enough data).
            # For simplicity we just feed the whole file; the fit handles mixed bands
            # as long as `mag` column exists.
            try:
                fit = ingress_measurement.measure_from_band_csv(
                    csv_path=csv,
                    period_days=period_min / 1440.0,
                )
                ingress_results[csv_name] = fit
            except Exception as exc:
                ingress_results[csv_name] = {"status": "error", "error": str(exc)}
    result["ingress_per_band"] = ingress_results

    # Aggregate: take the best-fitting (lowest chi2/dof) flat-bottomed measurement
    best = None
    for name, fit in ingress_results.items():
        if fit.get("status") != "ok":
            continue
        chi2 = fit.get("chi2", math.inf)
        dof = max(1, fit.get("dof", 1))
        score = chi2 / dof
        if best is None or score < best["score"]:
            best = {"score": score, "name": name, **fit}
    if best:
        result["best_ingress"] = best
        result["t_ingress_days"] = best.get("ingress_days")
        result["eclipse_duration_days"] = best.get("eclipse_duration_days")
        result["eclipse_depth_over_F_high"] = best.get("depth_over_F_high")
        result["morphology"] = best.get("morphology")
        result["tau_over_P"] = best.get("tau_over_P")

    state["light_curve_geometry"] = result
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02j_light_curve_geometry.json", result))
    return state


def eclipse_mcmc_node(state: AnalysisState) -> AnalysisState:
    """Run the 3-parameter disk-eclipse MCMC if we have a measured τ/P and the
    source class is plausibly disk-eclipsing.  Skips gracefully otherwise."""
    try:
        from astro_toolbox import disk_eclipse_mcmc
    except Exception as exc:
        result = {"status": "error", "error": f"import: {type(exc).__name__}: {exc}"}
        state["eclipse_mcmc"] = result
        _append_artifact(state, tools.json_dump(_run_dir(state) / "02k_eclipse_mcmc.json", result))
        return state

    plan = state.get("analysis_plan") or {}
    source_class = plan.get("source_class") or "unknown"
    geom = state.get("light_curve_geometry") or {}
    tau = geom.get("tau_over_P")
    morphology = geom.get("morphology")

    # Decide whether MCMC fires
    eligible_classes = {"unknown", "eclipsing_binary", "wide_binary",
                        "cataclysmic_variable"}
    eligible_morph = {"flat_bottomed"}
    fire = (
        tau is not None and tau > 0.05 and tau < 0.6
        and (source_class in eligible_classes
             or morphology in eligible_morph)
    )
    if not fire:
        result = {
            "status": "skipped",
            "reason": (
                f"not eligible: source_class={source_class}, morphology={morphology}, "
                f"tau/P={tau}"
            ),
        }
        state["eclipse_mcmc"] = result
        _append_artifact(state, tools.json_dump(_run_dir(state) / "02k_eclipse_mcmc.json", result))
        return state

    # Translate ingress/P into a fraction; estimate R_*/a from source class
    ingress = None
    eclipse_dur = geom.get("eclipse_duration_days")
    t_ing = geom.get("t_ingress_days")
    if eclipse_dur and t_ing and eclipse_dur > 0:
        ingress = t_ing / (eclipse_dur / max(tau, 1e-6))  # ingress/P
    orbit = geom.get("orbit") or {}
    a_Rsun = orbit.get("a_Rsun") or 0.0
    R_star_default = {
        "single_white_dwarf": 0.013,
        "sdob_binary_or_single": 0.15,
        "cataclysmic_variable": 0.5,
        "ultracompact_double_degenerate": 0.015,
        "eclipsing_binary": 0.8,
        "unknown": 0.7,
    }.get(source_class, 0.7)
    R_star_over_a = (R_star_default / a_Rsun) if a_Rsun > 0 else 0.05
    try:
        posterior = disk_eclipse_mcmc.fit_disk_eclipse_posterior(
            tau_obs=float(tau), tau_sigma=max(0.02, float(tau) * 0.05),
            ingress_obs=float(ingress) if ingress and 0 < ingress < 0.5 else None,
            ingress_sigma=0.01 if ingress else None,
            R_star_over_a=float(R_star_over_a),
        )
        posterior["inputs"] = {
            "tau_over_P": tau, "ingress_over_P": ingress,
            "R_star_over_a": R_star_over_a,
            "source_class": source_class,
            "morphology": morphology,
        }
        posterior["latex"] = disk_eclipse_mcmc.render_latex(posterior)
    except Exception as exc:
        posterior = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    state["eclipse_mcmc"] = posterior
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02k_eclipse_mcmc.json", posterior))
    return state


def physics_checks_node(state: AnalysisState) -> AnalysisState:
    """Assemble physics-driven argument paragraphs.  Now uses:
      - measured t_ingress from light_curve_geometry (Stage 2)
      - measured eccentricity median from eclipse_mcmc (Stage 2)
      - Keplerian a + v_orb from binary_orbit (Stage 2)
      - SED-decoupled chi^2 ranking (Stage 1)
    """
    try:
        plan = state.get("analysis_plan") or {}
        source_class = plan.get("source_class") or "unknown"
        geom = state.get("light_curve_geometry") or {}
        mcmc = state.get("eclipse_mcmc") or {}

        # Pull period + masses from light_curve_geometry (which already centralised
        # the bookkeeping).  Fallback to defaults if geometry skipped.
        period_min = geom.get("period_min_used")
        M_tot_default = geom.get("M_tot_Msun_assumed") or {
            "single_white_dwarf": 0.6,
            "sdob_binary_or_single": 0.88,
            "cataclysmic_variable": 1.0,
            "ultracompact_double_degenerate": 0.8,
            "eclipsing_binary": 1.4,
            "polar": 0.9,
            "unknown": 1.0,
        }.get(source_class, 1.0)
        R_companion = {
            "single_white_dwarf": 0.013,
            "sdob_binary_or_single": 0.15,
            "cataclysmic_variable": 0.5,
            "unknown": 0.7,
        }.get(source_class, 0.7)

        # Plug in measured t_ingress (days) if available.
        t_ingress_days = geom.get("t_ingress_days")
        # Plug in eccentricity from MCMC posterior if available.
        e_med = 0.0
        if mcmc.get("status") == "ok":
            e_pct = mcmc.get("e_pct")
            if e_pct and len(e_pct) >= 2:
                e_med = float(e_pct[1])

        report = physics_checks.assemble_physics_argument(
            sed_decoupled=state.get("sed_decoupled") or {},
            period_min=period_min,
            M_tot_Msun=M_tot_default,
            R_companion_Rsun=R_companion,
            t_ingress_days=t_ingress_days,
            eccentricity=e_med,
        )
        # Append eclipse-MCMC LaTeX paragraph if available
        if mcmc.get("latex"):
            report.setdefault("sections", []).append(
                {"id": "eclipse_mcmc_posterior"}
            )
            report["latex"] = (report.get("latex") or "") + "\n\n" + mcmc["latex"]

        report["status"] = "ok" if report.get("sections") else "no_inputs"
        report["assumptions"] = {
            "M_tot_default_Msun": M_tot_default,
            "R_companion_default_Rsun": R_companion,
            "period_min_used": period_min,
            "source_class": source_class,
            "t_ingress_days_measured": t_ingress_days,
            "eccentricity_used": e_med,
        }
    except Exception as exc:
        report = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    state["physics_checks"] = report
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02i_physics_checks.json", report))
    return state


def ads_live_node(state: AnalysisState) -> AnalysisState:
    """Live ADS query: pull recent papers within 0.005 deg of the target and
    merge them into the per-source RAG sqlite.  Graceful no-op when
    ADS_DEV_KEY is not set."""
    try:
        ra = state.get("ra_deg")
        dec = state.get("dec_deg")
        result = ads_live.query_ads(
            target=state.get("target"),
            ra_deg=float(ra) if ra is not None else None,
            dec_deg=float(dec) if dec is not None else None,
            year_min=2018,
            rows=30,
        )
        # If we got papers and have a per-source RAG sqlite, merge them in
        sr_info = state.get("source_rag") or {}
        sr_path = sr_info.get("sqlite_path")
        if result.get("status") == "ok" and sr_path:
            try:
                aliases = sr_info.get("target_aliases") or []
                merge_res = ads_live.merge_into_source_rag(
                    sqlite_path=sr_path,
                    papers=result.get("papers") or [],
                    target_aliases=aliases,
                )
                result["merge_into_source_rag"] = merge_res
            except Exception as exc:
                result["merge_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    state["ads_live"] = result
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02l_ads_live.json", result))
    return state


def novelty_detector_node(state: AnalysisState) -> AnalysisState:
    """Compute the this-work vs literature differential and store it as
    state['novelty'] for the drafter to insert in Discussion."""
    try:
        novelty = novelty_detector.compute_novelty(state.get("published_params") or {})
        novelty["latex"] = novelty_detector.render_latex(novelty)
        novelty["status"] = "ok" if novelty.get("items") else "no_items"
    except Exception as exc:
        novelty = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    state["novelty"] = novelty
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02m_novelty.json", novelty))
    return state


def comparison_table_node(state: AnalysisState) -> AnalysisState:
    """Build a literature comparison `deluxetable*` against benchmark systems."""
    try:
        table = comparison_table.build_comparison_table(state)
    except Exception as exc:
        table = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    state["comparison_table"] = table
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02n_comparison_table.json", table))
    return state


def figure_synthesizer_node(state: AnalysisState) -> AnalysisState:
    """Generate 4 publication-ready PNG figures into paper_orchestra/figures/."""
    try:
        workspace = (state.get("paper_orchestra") or {}).get("workspace") \
            or str(_run_dir(state) / "paper_orchestra")
        result = figure_synthesizer.synthesize_all(state, Path(workspace))
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    state["figures"] = result
    _append_artifact(state, tools.json_dump(_run_dir(state) / "11_figures.json", result))
    return state


def latex_compile_node(state: AnalysisState) -> AnalysisState:
    """Try to compile paper_orchestra/final/paper.tex via latexmk → PDF.

    Graceful no-op when neither latexmk nor pdflatex are installed."""
    try:
        workspace = (state.get("paper_orchestra") or {}).get("workspace") \
            or str(_run_dir(state) / "paper_orchestra")
        result = latex_compile.compile_paper(workspace=Path(workspace))
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    state["latex_compile"] = result
    _append_artifact(state, tools.json_dump(_run_dir(state) / "12_latex_compile.json", result))
    return state


def rag_navigator_node(state: AnalysisState) -> AnalysisState:
    base_queries = [
        "white dwarf SED fitting Gaia parallax effective temperature",
        "white dwarf radial velocity orbit traceback six dimensional phase space",
        "white dwarf Bayesian inference systematics uncertainty error propagation",
        "emission line white dwarf binary accretion spectrum residual fitting",
    ]
    source_class = (state.get("analysis_plan") or {}).get("source_class")
    expanded = retrieval_prompts.expand_queries(
        section="Methods",
        source_class=source_class,
        base_queries=base_queries,
        use_llm=bool(state.get("retrieval_llm", False)),
        provider=state.get("retrieval_provider") or state.get("llm_provider"),
    )
    results: List[Dict[str, Any]] = []
    for query in expanded["queries"]:
        rows = tools.search_rag(query, method_only=True, limit=5)
        rows = retrieval_prompts.rerank_hits(rows, source_class)
        results.append({"query": query, "rows": rows})
    state["rag_results"] = results
    state["rag_query_plan"] = {
        "source_class": source_class,
        "rerank_keys": expanded.get("rerank_keys", []),
        "llm_used": expanded.get("llm_used", False),
    }
    _append_artifact(state, tools.json_dump(_run_dir(state) / "03_rag_results.json", results))
    return state


def kg_navigator_node(state: AnalysisState) -> AnalysisState:
    base_queries = [
        "SED fitting Bayesian inference parallax cooling age",
        "6D phase space orbit traceback radial velocity binary white dwarf",
        "residual systematics uncertainty model prior",
    ]
    source_class = (state.get("analysis_plan") or {}).get("source_class")
    expanded = retrieval_prompts.expand_queries(
        section="Methods",
        source_class=source_class,
        base_queries=base_queries,
        use_llm=bool(state.get("retrieval_llm", False)),
        provider=state.get("retrieval_provider") or state.get("llm_provider"),
    )
    rows = tools.search_kg(expanded["queries"], limit=20)
    rows = retrieval_prompts.rerank_hits(rows, source_class, body_key="object", title_key="subject")
    state["kg_results"] = rows
    _append_artifact(state, tools.json_dump(_run_dir(state) / "04_kg_results.json", rows))
    return state


def _strip_json_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def method_scout_node(state: AnalysisState) -> AnalysisState:
    """Investigate reusable/newer methods before deciding what Coder/Claude should change."""
    plan = state.get("analysis_plan", {})
    route = plan.get("route", "unknown")
    base_queries = [
        f"{route} white dwarf parameter inference method",
        "Gaia HR diagram SED fitting white dwarf no spectroscopy",
        "white dwarf SED residual systematic error Bayesian photometric fit",
        "spectral line fitting emission line white dwarf binary classification",
    ]
    if route in {"photometric_hrd_sed_fallback", "sed_only_fallback"}:
        base_queries.extend(
            [
                "photometric white dwarf effective temperature radius Gaia parallax extinction prior",
                "white dwarf mass cooling age constraints without spectroscopy limitations",
            ]
        )
    source_class = plan.get("source_class")
    expanded = retrieval_prompts.expand_queries(
        section="Methods",
        source_class=source_class,
        base_queries=base_queries,
        use_llm=bool(state.get("retrieval_llm", False)),
        provider=state.get("retrieval_provider") or state.get("llm_provider"),
    )
    queries_for_search = expanded["queries"][:8]
    rag_hits = []
    for query in queries_for_search[:6]:
        rows = tools.search_rag(query, method_only=False, limit=4)
        rows = retrieval_prompts.rerank_hits(rows, source_class)
        rag_hits.append({"query": query, "rows": rows})
    kg_hits = tools.search_kg(queries_for_search, limit=20)
    kg_hits = retrieval_prompts.rerank_hits(kg_hits, source_class, body_key="object", title_key="subject")

    recommendations = [
        "Before trusting a model result, compare it against at least one independent observable: spectra if available, otherwise HR-diagram position and SED residuals.",
        "If spectra are absent, only publish photometric parameters as provisional and keep atmosphere composition/logg/mass/cooling age under stronger caveats.",
        "Any fit pinned to a model-grid boundary must be treated as nonconverged and handed back to Coder/Claude for prior/model repair.",
    ]
    scout: Dict[str, Any] = {
        "status": "written",
        "route": route,
        "queries": queries_for_search,
        "queries_expanded_from": base_queries,
        "rerank_keys": expanded.get("rerank_keys", []),
        "rag_hits": rag_hits,
        "kg_hits": kg_hits,
        "recommendations": recommendations,
        "llm_used": False,
    }
    evidence_pack = method_learning.collect_method_evidence(state=state, scout=scout)
    scout["method_evidence"] = evidence_pack

    if state.get("method_scout_llm", False):
        provider = state.get("method_scout_provider") or state.get("llm_provider") or "kimi"
        try:
            cfg = load_model_config(provider)
            client = LLMClient(cfg)
            prompt_payload = {
                "analysis_plan": plan,
                **evidence_pack,
            }
            system, user = method_learning.build_algorithm_extraction_prompt(prompt_payload)
            text = client.complete(system=system, user=user, temperature=0.1, max_output_tokens=5000)
            scout["llm_provider"] = cfg.provider
            scout["llm_model"] = cfg.model
            scout["llm_used"] = True
            scout["llm_recommendations"] = json.loads(_strip_json_fence(text))
        except Exception as exc:
            scout["llm_error"] = f"{type(exc).__name__}: {exc}"

    algorithm_spec = method_learning.build_algorithm_spec(state, scout)
    toolbox_gap = method_learning.build_toolbox_gap(state, algorithm_spec)
    scout["algorithm_spec"] = algorithm_spec
    scout["toolbox_gap"] = toolbox_gap
    if state.get("method_scout_llm", False) and toolbox_gap.get("status") == "ready_for_tool_write":
        state["toolbox_gap"] = toolbox_gap
        _append_artifact(state, tools.json_dump(_run_dir(state) / "04e_toolbox_gap.json", toolbox_gap))

    state["method_scout"] = scout
    _append_artifact(state, tools.json_dump(_run_dir(state) / "04c_method_scout.json", scout))
    return state


def kg_graph_report_node(state: AnalysisState) -> AnalysisState:
    if not state.get("kg_report", False):
        state["kg_graph_report"] = {"status": "skipped", "reason": "kg_report is false"}
        _append_artifact(state, tools.json_dump(_run_dir(state) / "04b_kg_graph_report.json", state["kg_graph_report"]))
        return state

    out_dir = _run_dir(state) / "kg_graph_report"
    try:
        result = run_graph_agent(
            output_root=out_dir,
            use_llm=bool(state.get("kg_report_llm", False)),
            provider=state.get("kg_report_provider") or state.get("llm_provider") or "deepseek",
        )
        result["status"] = "written"
        state["kg_graph_report"] = result
        for path in result.get("artifacts", {}).values():
            if isinstance(path, str) and path:
                _append_artifact(state, path)
    except Exception as exc:
        state["kg_graph_report"] = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "output_root": str(out_dir),
        }
        state.setdefault("warnings", []).append(f"KG graph report failed: {type(exc).__name__}: {exc}")

    _append_artifact(state, tools.json_dump(_run_dir(state) / "04b_kg_graph_report.json", state["kg_graph_report"]))
    return state


def source_research_package_node(state: AnalysisState) -> AnalysisState:
    """Build the strict per-source SIMBAD/RAG/KG/data evidence package."""
    if not state.get("source_research_package", False):
        state["source_research"] = {"status": "skipped", "reason": "source_research_package is false"}
        _append_artifact(state, tools.json_dump(_run_dir(state) / "04d_source_research_package.json", state["source_research"]))
        return state
    if state.get("ra_deg") is None or state.get("dec_deg") is None:
        result = {"status": "skipped", "reason": "coordinates unavailable"}
        state["source_research"] = result
        _append_artifact(state, tools.json_dump(_run_dir(state) / "04d_source_research_package.json", result))
        return state

    astrotool_root = state.get("data_fetch", {}).get("astrotool", {}).get("output_root") or state.get("astrotool_run")
    if not astrotool_root:
        result = {"status": "skipped", "reason": "astrotool output root unavailable"}
        state["source_research"] = result
        _append_artifact(state, tools.json_dump(_run_dir(state) / "04d_source_research_package.json", result))
        return state

    out_root = _run_dir(state) / "source_research_package"
    cmd = [
        sys.executable,
        "-m",
        "Astro_Agent.analysis_agent.source_research_pipeline",
        "--target",
        state["target"],
        "--ra",
        str(state["ra_deg"]),
        "--dec",
        str(state["dec_deg"]),
        "--output-root",
        str(out_root),
        "--astrotool-run",
        str(astrotool_root),
    ]
    if state.get("download_simbad_pdfs", False):
        cmd.append("--download-simbad-pdfs")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            text=True,
            capture_output=True,
            timeout=7200,
        )
        result: Dict[str, Any] = {
            "status": "written" if proc.returncode == 0 else "error",
            "output_root": str(out_root),
            "command": cmd,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
        for name in (
            "source_research_package.json",
            "source_analysis_products.json",
            "source_research_report.md",
            "source_identifiers.json",
            "simbad_all_references.json",
            "simbad_pdf_downloads.json",
            "simbad_source_mentions.json",
            "rag_exact_simbad_papers.json",
            "kg_source_relations.json",
        ):
            path = out_root / name
            if path.exists():
                result[name.replace(".", "_")] = str(path)
                _append_artifact(state, str(path))
        simbad_path = out_root / "simbad_all_references.json"
        if simbad_path.exists():
            simbad = json.loads(simbad_path.read_text(encoding="utf-8"))
            result["simbad_n_refs"] = simbad.get("n_refs")
        downloads_path = out_root / "simbad_pdf_downloads.json"
        if downloads_path.exists():
            downloads = json.loads(downloads_path.read_text(encoding="utf-8"))
            result["simbad_pdf_count"] = downloads.get("n_available_pdf")
        analysis_path = out_root / "source_analysis_products.json"
        if analysis_path.exists():
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            result["sed_status"] = analysis.get("sed", {}).get("status")
            result["robust_emission_counts"] = {
                survey: payload.get("robust_emission_count")
                for survey, payload in analysis.get("line_fits", {}).items()
            }
        if proc.returncode != 0:
            state.setdefault("warnings", []).append("Source research package failed; see 04d_source_research_package.json")
    except Exception as exc:
        result = {
            "status": "error",
            "output_root": str(out_root),
            "error": f"{type(exc).__name__}: {exc}",
        }
        state.setdefault("warnings", []).append(f"Source research package failed: {type(exc).__name__}: {exc}")

    state["source_research"] = result
    _append_artifact(state, tools.json_dump(_run_dir(state) / "04d_source_research_package.json", result))
    return state


def baseline_iteration_node(state: AnalysisState) -> AnalysisState:
    output = state.get("data_fetch", {}).get("existing_outputs", {})
    checks = tools.extract_physics_checks(output)
    using_existing = state.get("data_fetch", {}).get("status") == "existing"
    status = "completed" if using_existing else ("planned" if state.get("dry_run", True) else "completed")
    if not output.get("run_summary") and state.get("dry_run", True) and not using_existing:
        warnings = ["Baseline fit not executed in dry-run mode"]
    else:
        warnings = checks["warnings"]
    iteration = {
        "iteration": 1,
        "name": "Baseline",
        "status": status,
        "objective": "Run standard astro_toolbox spectra, photometry, SED, HRD, WD fitting, period and RV modules.",
        "inputs": {
            "astrotool_output_root": state.get("data_fetch", {}).get("astrotool", {}).get("output_root"),
            "rag_queries": [item["query"] for item in state.get("rag_results", [])],
        },
        "warnings": warnings,
        "checks": checks,
    }
    state.setdefault("iterations", []).append(iteration)
    _append_artifact(state, tools.json_dump(_run_dir(state) / "05_iteration_1_baseline.json", iteration))
    return state


def residual_physics_iteration_node(state: AnalysisState) -> AnalysisState:
    warnings: List[str] = []
    output = state.get("data_fetch", {}).get("existing_outputs", {})
    run_summary = output.get("run_summary", {})
    if not run_summary.get("wd_fitting_available"):
        warnings.append("Cannot inspect WD residuals because WD fitting output is unavailable")
    if not output.get("status_file"):
        warnings.append("No module_status.csv was found for residual inspection")

    iteration = {
        "iteration": 2,
        "name": "Residuals & Physics",
        "status": "nonconverged" if warnings else "completed",
        "objective": "Inspect residuals and reject physically impossible parameter regions.",
        "physics_guards": [
            "cooling_age_gyr must not exceed 13.8 without explicit explanation",
            "logg should remain inside the atmosphere grid and a plausible WD range",
            "dust/IR excess models require temperatures and luminosities consistent with the SED",
            "period-folded light curves must be checked for aliases and survey cadence artifacts",
            "emission lines trigger binary/accretion alternatives before single-DA interpretation",
        ],
        "method_transfer_from_kg": state.get("kg_results", [])[:8],
        "warnings": warnings,
    }
    state.setdefault("iterations", []).append(iteration)
    _append_artifact(state, tools.json_dump(_run_dir(state) / "06_iteration_2_residuals.json", iteration))
    return state


def errors_systematics_iteration_node(state: AnalysisState) -> AnalysisState:
    warnings: List[str] = []
    output = state.get("data_fetch", {}).get("existing_outputs", {})
    counts = output.get("module_status_counts", {})
    if counts.get("error", 0):
        warnings.append(f"{counts['error']} modules errored; confidence intervals must include data-missing systematics")
    if state.get("dry_run", True) and state.get("data_fetch", {}).get("status") != "existing":
        warnings.append("No numerical confidence interval can be certified in dry-run mode")

    iteration = {
        "iteration": 3,
        "name": "Errors & Systematics",
        "status": "nonconverged" if warnings else "completed",
        "objective": "Quantify statistical and systematic uncertainties before final parameters are accepted.",
        "required_error_budget": [
            "photometric zero points and extinction correction",
            "Gaia parallax uncertainty and possible zero-point offset",
            "spectral flux calibration and line-mask choices",
            "model-grid spacing and atmosphere-composition prior",
            "period aliasing and RV template mismatch where applicable",
        ],
        "warnings": warnings,
    }
    state.setdefault("iterations", []).append(iteration)
    _append_artifact(state, tools.json_dump(_run_dir(state) / "07_iteration_3_systematics.json", iteration))
    return state


def model_supervisor_node(state: AnalysisState) -> AnalysisState:
    """Continuously supervise model results and produce concrete repair tasks."""
    output = state.get("data_fetch", {}).get("existing_outputs", {})
    run_summary = output.get("run_summary", {}) or {}
    plan = state.get("analysis_plan", {})
    source_research = state.get("source_research", {})
    route = plan.get("route", "unknown")
    method_scout = state.get("method_scout", {})
    max_rounds = max(1, int(state.get("max_supervision_rounds", 2) or 2))

    rounds: List[Dict[str, Any]] = []
    open_actions: List[Dict[str, Any]] = []
    human_triggers: List[str] = []
    for round_index in range(1, max_rounds + 1):
        checks: List[str] = []
        actions: List[Dict[str, Any]] = []

        if route in {"photometric_hrd_sed_fallback", "sed_only_fallback"}:
            checks.append("spectra_absent_or_not_required_for_route")
            if not run_summary.get("wd_fitting_available"):
                actions.append(
                    {
                        "owner": "Coder/Claude",
                        "priority": "high",
                        "task": "Implement or run photometric fallback fitting: Gaia/HRD + SED + parallax/extinction priors for provisional Teff/radius/luminosity.",
                        "acceptance": "Output JSON with units, chi2/residuals, posterior intervals, and a caveat that logg/mass/cooling age are weak without spectra.",
                    }
                )
            else:
                checks.append("photometric fallback already has WD fitting output")
        elif run_summary.get("spectra_available"):
            checks.append("spectra_available_for_classification_and_line_fitting")
            if not run_summary.get("rv_report_available") and not run_summary.get("wd_fitting_available"):
                actions.append(
                    {
                        "owner": "Coder/Claude",
                        "priority": "medium",
                        "task": "Verify spectral-line fitting products and reject boundary/false emission-line detections.",
                        "acceptance": "Line table includes wavelength, flux/EW, uncertainty, S/N, continuum window, and rejection reason for non-robust candidates.",
                    }
                )
            else:
                checks.append("spectral WD/RV products are already available")

        if run_summary.get("sed_available") and not run_summary.get("wd_fitting_available"):
            actions.append(
                {
                    "owner": "Coder/Claude",
                    "priority": "high",
                    "task": "Repair or add WD/SED fitting result export for the existing SED products.",
                    "acceptance": "Create wd_fitting.json or wd_fit_results.csv with astropy-compatible units, priors, goodness-of-fit, residual summary, and model-grid boundary flags.",
                }
            )
            human_triggers.append("SED exists but WD fitting output is unavailable; final parameters remain blocked.")
        if source_research.get("sed_status") in {"poor_fit", "no_fit"}:
            actions.append(
                {
                    "owner": "Coder/Claude",
                    "priority": "high",
                    "task": "Improve the SED model using the source research package: keep SDSS/Gaia optical bands as the primary DA fit and treat WISE/SPHEREx as context unless provenance and model coverage are validated.",
                    "acceptance": "SED JSON reports primary_fit_bands, context_only_bands, residuals, chi2, and explicitly blocks final Teff/logg/mass/cooling-age claims when chi2 remains high.",
                }
            )
            human_triggers.append("Primary SED fit is poor or absent in the source research package.")
        robust_counts = source_research.get("robust_emission_counts") or {}
        if robust_counts:
            checks.append(f"source_research_robust_emission_counts={robust_counts}")

        if run_summary.get("hst_spectrum_available"):
            checks.append("hst_uv_context_available")
            actions.append(
                {
                    "owner": "Reviewer",
                    "priority": "medium",
                    "task": "Audit whether HST UV data are used as UV context or direct optical-classification evidence.",
                    "acceptance": "Paper text separates UV coverage/SNR limitations from optical spectral classification.",
                }
            )

        for recommendation in method_scout.get("recommendations", [])[:3]:
            checks.append(f"method_scout: {recommendation}")

        round_status = "needs_repair" if actions else "clear"
        rounds.append(
            {
                "round": round_index,
                "status": round_status,
                "checks": checks,
                "actions": actions,
            }
        )
        open_actions.extend(actions)
        if not actions:
            break

    deduped_actions: List[Dict[str, Any]] = []
    seen_action_keys = set()
    for action in open_actions:
        key = (action.get("owner"), action.get("priority"), action.get("task"))
        if key in seen_action_keys:
            continue
        seen_action_keys.add(key)
        deduped_actions.append(action)

    supervision = {
        "status": "needs_repair" if deduped_actions else "clear",
        "route": route,
        "rounds": rounds,
        "open_actions": deduped_actions,
        "human_review_triggers": sorted(set(human_triggers)),
        "policy": "Model outputs are not certified until Supervisor actions are resolved and QA gate clears.",
    }
    state["model_supervision"] = supervision
    _append_artifact(state, tools.json_dump(_run_dir(state) / "07b_model_supervision.json", supervision))
    return state


def claude_code_delegate_node(state: AnalysisState) -> AnalysisState:
    """Delegate concrete code/repair tasks to Claude Code when explicitly enabled."""
    supervision = state.get("model_supervision", {})
    actions = [
        action
        for action in supervision.get("open_actions", [])
        if str(action.get("owner", "")).lower().startswith("coder")
    ]
    toolbox_gap = state.get("toolbox_gap")
    if toolbox_gap and toolbox_gap.get("status") == "ready_for_tool_write":
        actions.insert(
            0,
            {
                "owner": "Coder/Claude",
                "priority": "high",
                "task": "Implement and validate a literature-derived astro_toolbox capability.",
                "acceptance": "Patch must report validation_passed=true or tests_passed=true before dynamic skill registration is allowed.",
                "toolbox_gap": toolbox_gap,
            },
        )
    if not state.get("enable_claude_code", False):
        result = {"status": "skipped", "reason": "enable_claude_code is false", "queued_actions": actions}
        state["claude_code"] = result
        _append_artifact(state, tools.json_dump(_run_dir(state) / "07c_claude_code.json", result))
        return state
    if not actions:
        result = {"status": "skipped", "reason": "no coder actions from model supervisor"}
        state["claude_code"] = result
        _append_artifact(state, tools.json_dump(_run_dir(state) / "07c_claude_code.json", result))
        return state

    prompt = (
        "You are Claude Code acting as the Coder node inside an astronomy research agent. "
        "Do not invent science results. Inspect the repository and propose/implement only bounded code changes needed for these tasks. "
        "If you edit files, list changed paths and required tests. Preserve existing user data.\n\n"
        f"Target: {state.get('target')}\n"
        f"Astrotool output: {state.get('data_fetch', {}).get('astrotool', {}).get('output_root')}\n"
        f"Analysis plan: {json.dumps(state.get('analysis_plan', {}), ensure_ascii=False)}\n"
        f"Literature-derived toolbox gap: {method_learning.render_toolbox_gap_prompt(toolbox_gap or {})}\n"
        f"Actions: {json.dumps(actions, ensure_ascii=False, indent=2)}\n"
        "If you implement a toolbox method, finish with strict JSON containing: "
        "validation_passed, changed_files, tests, docs, risks, and skill_registration_notes.\n"
    )
    result = codex_tool.parse_claude_json(
        codex_tool.claude_code_exec(
            prompt,
            cwd=tools.ASTRO_AGENT_DIR,
            timeout=int(state.get("claude_timeout", 900) or 900),
            permission_mode=str(state.get("claude_permission_mode", "plan")),
        )
    )
    result["queued_actions"] = actions
    if toolbox_gap:
        registration = method_learning.register_dynamic_skill_if_valid(
            gap=toolbox_gap,
            claude_result=result,
            project_root=tools.ASTRO_AGENT_DIR,
        )
        result["dynamic_skill_registration"] = registration
        state["dynamic_skill_registration"] = registration
    state["claude_code"] = result
    _append_artifact(state, tools.json_dump(_run_dir(state) / "07c_claude_code.json", result))
    return state


_MODEL_MISMATCH_NEEDLES = (
    "wd fitting output is unavailable",
    "wd fitting is unavailable",
    "model fit did not return",
    "grid boundary",
    "convergence failure",
    "did not converge",
)


def qa_gate_node(state: AnalysisState) -> AnalysisState:
    qa = tools.detect_anomalies(state.get("data_fetch", {}), state.get("iterations", []))
    supervision = state.get("model_supervision", {})
    if supervision.get("status") == "needs_repair":
        qa.setdefault("reasons", [])
        qa["reasons"].extend(supervision.get("human_review_triggers", []))
        qa["reasons"].append("Model Supervisor has unresolved repair actions.")
        qa["human_review_required"] = True

    # Detect "model mismatch": the underlying scientific data exists but
    # the chosen fitting module is wrong for this source class. We surface
    # it as an infrastructure-class reason so the workflow can re-plan
    # instead of pretending the source itself is anomalous.
    plan = state.get("analysis_plan", {}) or {}
    source_class = plan.get("source_class") or "unknown"
    pipeline_module = plan.get("fitting_pipeline_module") or ""
    reasons = qa.get("reasons", []) or []
    mismatch_hits = [
        r for r in reasons
        if any(needle in str(r).lower() for needle in _MODEL_MISMATCH_NEEDLES)
    ]
    is_non_wd = source_class not in {"single_white_dwarf", "unknown"}
    # Note: the iteration nodes currently always invoke wd_fitting regardless
    # of `fitting_pipeline_module`. So model_mismatch should fire whenever a
    # non-WD source produced wd-fit-unavailable reasons, even before a real
    # sdob_fitting / cv_fitting module exists.
    qa["model_mismatch"] = bool(mismatch_hits) and is_non_wd
    qa["source_class"] = source_class
    qa["fitting_pipeline_module"] = pipeline_module
    qa["model_mismatch_reasons"] = mismatch_hits

    qa["apj_gate"] = "hold_for_human" if qa["human_review_required"] else "clear_for_draft"
    qa["checked_at_utc"] = datetime.utcnow().isoformat(timespec="seconds")
    state["qa"] = qa
    retry_count = int(state.get("structure_planner_retry_count", 0) or 0)
    pipeline_implemented = bool(plan.get("pipeline_implemented", True))
    # Replan only if (a) we detected a real mismatch, (b) we have retries left,
    # and (c) the dispatcher knows of a different pipeline that ACTUALLY exists
    # on disk. Otherwise replanning would just spin into the same dead end.
    if qa["model_mismatch"] and retry_count < 2 and pipeline_implemented:
        state["next_step"] = "replan"
        state["structure_planner_retry_count"] = retry_count + 1
        qa["replan_request"] = {
            "reason": "model_mismatch_detected",
            "retry_count_after_increment": retry_count + 1,
            "source_class": source_class,
            "current_pipeline_module": pipeline_module,
        }
    else:
        if qa["model_mismatch"] and not pipeline_implemented:
            qa["replan_blocked"] = {
                "reason": "no_implementation_for_source_class",
                "source_class": source_class,
                "expected_module": pipeline_module,
                "follow_up": "implement the module or fall back to single_white_dwarf",
            }
        state["next_step"] = "abnormal" if qa["human_review_required"] else "paper"
    _append_artifact(state, tools.json_dump(_run_dir(state) / "08_qa_gate.json", qa))

    # D1 — build per-section evidence-availability manifest right after QA so
    # that drafter and structural_lint can consult it. Storing it on state
    # avoids a second disk read in pack_section_evidence.
    try:
        from .evidence_manifest import build_manifest
        manifest = build_manifest(_run_dir(state), state)
        state["evidence_manifest"] = manifest
        _append_artifact(state, tools.json_dump(_run_dir(state) / "02o_evidence_manifest.json", manifest))
    except Exception as exc:
        state.setdefault("warnings", []).append(
            f"evidence_manifest build failed: {type(exc).__name__}: {exc}"
        )
    return state


def route_after_qa(state: AnalysisState) -> str:
    next_step = state.get("next_step")
    if next_step == "replan":
        return "replan"
    if next_step == "abnormal" and state.get("draft_on_hold", False):
        return "paper"
    return next_step or "abnormal"


def abnormal_report_node(state: AnalysisState) -> AnalysisState:
    run_dir = _run_dir(state)
    qa = state.get("qa", {})
    text = [
        "# 异常分析报告",
        "",
        f"- Target: {state.get('target')}",
        f"- RA/Dec deg: {state.get('ra_deg')}, {state.get('dec_deg')}",
        f"- Gate: {qa.get('apj_gate')}",
        "",
        "## 暂停原因",
    ]
    for reason in qa.get("reasons", []):
        text.append(f"- {reason}")
    text.extend(
        [
            "",
            "## 已完成的三次迭代",
        ]
    )
    for iteration in state.get("iterations", []):
        text.append(
            f"- Iteration {iteration.get('iteration')}: {iteration.get('name')} "
            f"status={iteration.get('status')}"
        )
    text.extend(
        [
            "",
            "## 人类审核建议",
            "- 确认坐标、单位和交叉匹配对象是否正确。",
            "- 检查失败的巡天模块是否由网络、权限、无覆盖或真实非探测造成。",
            "- 若存在强发射线、非收敛拟合或异常冷却年龄，不要写入最终论文参数表。",
        ]
    )
    path = tools.write_text(run_dir / "abnormal_analysis_report.md", "\n".join(text) + "\n")
    guidance = codex_style.write_guidance(run_dir)
    manifest = tools.json_dump(run_dir / "agents_manifest.json", paper_agents.five_agent_manifest())
    state["abnormal_report"] = {"path": path, "status": "written"}
    _append_artifact(state, path)
    _append_artifact(state, guidance)
    _append_artifact(state, manifest)
    return state


def drafter_node(state: AnalysisState) -> AnalysisState:
    run_dir = _run_dir(state)
    orchestra = paper_orchestra.run_astro_paper_orchestra(
        run_dir,
        state,
        use_llm=bool(state.get("use_llm", False)),
        provider=state.get("llm_provider"),
        sectionwise=True,
        target_score=int(state.get("paper_target_score", 80)),
        max_refine_iters=int(state.get("paper_max_iters", 3)),
    )
    state["paper_orchestra"] = orchestra
    rag_rows = []
    for item in state.get("rag_results", []):
        rag_rows.extend(item.get("rows", []))
    context_path = tools.write_text(
        run_dir / "paper" / "method_context.md",
        "# Local RAG Evidence\n\n"
        + tools.format_rag_bullets(rag_rows)
        + "\n\n# KG Method Transfer\n\n"
        + tools.format_kg_bullets(state.get("kg_results", []))
        + "\n",
    )
    state["paper"] = {
        "status": "drafted",
        "tex": orchestra.get("final_tex"),
        "context": context_path,
        "paper_orchestra_workspace": orchestra.get("workspace"),
        "paper_orchestra_final_tex": orchestra.get("final_tex"),
        "final_review": orchestra.get("final_review", {}),
    }
    _append_artifact(state, context_path)
    for key in ("codex_style_guidance", "agents_manifest", "outline", "draft", "final_tex", "worklog", "provenance"):
        value = orchestra.get(key)
        if isinstance(value, str):
            _append_artifact(state, value)
    return state


def reflexion_node(state: AnalysisState) -> AnalysisState:
    """Reflexion-style verbal critique node.

    Reads the latest paper_qc verdict, produces a structured + verbal
    reflection that targets sections by failing check id, and appends to
    reflexion_history.  The conditional edge after this node decides whether
    to send the run back to the drafter for a targeted rewrite (bounded to
    max_reflexion_retries) or to proceed to peer_reviewer.

    Important: LangGraph does not reliably persist state mutations performed
    inside routing functions.  All mutation must happen in the node body, so
    we compute "will we retry?" here, increment the counter here, and store a
    routing decision token the conditional edge can read.
    """
    qc = state.get("paper_qc") or {}
    reflection = reflexion.build_reflection(qc)
    state["reflexion_history"] = reflexion.append_to_history(state, reflection)
    retry_count = int(state.get("reflexion_retry_count") or 0)
    max_retries = int(state.get("max_reflexion_retries") or 2)
    verdict = qc.get("verdict")
    n_fail = int(qc.get("n_fail") or 0)
    n_warn = int(qc.get("n_warn") or 0)
    has_actionable = bool(reflection.get("action_items"))
    needs_rewrite = (verdict in ("fail", "warn") or n_fail > 0 or n_warn >= 1)
    if needs_rewrite and has_actionable and retry_count < max_retries:
        state["reflexion_retry_count"] = retry_count + 1
        state["reflexion_decision"] = "rewrite"
        reflection["routing_decision"] = "rewrite"
        reflection["retry_count_after_increment"] = retry_count + 1
    else:
        state["reflexion_decision"] = "accept"
        reflection["routing_decision"] = "accept"
        if not has_actionable:
            reflection["routing_reason"] = "no_action_items"
        elif not needs_rewrite:
            reflection["routing_reason"] = "qc_clean_enough"
        else:
            reflection["routing_reason"] = "max_retries_reached"
    _append_artifact(
        state,
        tools.json_dump(_run_dir(state) / "09b_reflexion.json", reflection),
    )
    return state


def route_after_reflexion(state: AnalysisState) -> str:
    # Pure reader — node body did the real work.
    return state.get("reflexion_decision") or "accept"


def paper_qc_node(state: AnalysisState) -> AnalysisState:
    """Run the ApJ paper QC checklist on the drafted manuscript."""
    from .nodes.paper_qc import run_paper_qc

    paper = state.get("paper") or {}
    orchestra = state.get("paper_orchestra") or {}
    final_tex = paper.get("paper_orchestra_final_tex") or orchestra.get("final_tex") or paper.get("tex")
    workspace = orchestra.get("workspace")
    qc = run_paper_qc(
        final_tex_path=final_tex,
        workspace_root=workspace,
        published_params_table=state.get("published_params"),
        hypothesis_plan=state.get("hypothesis_plan"),
        cluster_membership=state.get("cluster_membership"),
        resolved_target=state.get("resolved"),
        gold_path=state.get("gold_path"),
        extinction=state.get("extinction"),
        physics_checks=state.get("physics_checks"),
    )
    state["paper_qc"] = qc
    _append_artifact(
        state,
        tools.json_dump(_run_dir(state) / "09_paper_qc.json", qc),
    )
    return state


def peer_reviewer_node(state: AnalysisState) -> AnalysisState:
    questions = [
        "所选大气模型网格是否覆盖目标的 Teff/logg/成分空间，网格边界附近的解是否被错误当成收敛解？",
        "光变曲线周期折叠是否被 ZTF/WISE/TESS 的采样窗口函数或日别名污染？是否有独立波段验证？",
        "Gaia 视差、消光、光谱通量定标和测光零点误差是否进入最终置信区间，而不是只报告统计拟合误差？",
        "若存在强发射线或 X-ray/IR excess，单白矮星模型是否仍是物理上最优解释？",
    ]
    qc = state.get("paper_qc") or {}
    text_lines = [
        "# Peer Reviewer Report",
        "",
        "## ApJ Paper QC verdict",
        f"- summary: {qc.get('summary', 'not_run')}",
    ]
    for check in qc.get("checks", []) or []:
        text_lines.append(f"- [{check.get('verdict')}] {check.get('id')}: {check.get('reason')}")
    text_lines.append("")
    text_lines.append("## Scientific questions")
    for q in questions:
        text_lines.append(f"- {q}")
    text = "\n".join(text_lines) + "\n"
    path = tools.write_text(_run_dir(state) / "paper" / "peer_review.md", text)
    state["peer_review"] = {
        "status": "written",
        "questions": questions,
        "paper_qc_summary": qc.get("summary"),
        "paper_qc_verdict": qc.get("verdict"),
        "path": path,
    }
    _append_artifact(state, path)
    return state


def kg_writeback_node(state: AnalysisState) -> AnalysisState:
    """Append the run's per-source provenance and method-run history to the
    learning ledger."""
    try:
        report = kg_writeback.write_run(
            source_id=str(state.get("target") or "unknown"),
            state=state,
            run_dir=str(_run_dir(state)),
        )
    except Exception as exc:
        report = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    state["kg_writeback"] = report
    _append_artifact(state, tools.json_dump(_run_dir(state) / "10_kg_writeback.json", report))
    return state


def toolbox_evolution_node(state: AnalysisState) -> AnalysisState:
    data = state.get("data_fetch", {})
    output = data.get("existing_outputs", {})
    missing = []
    for row in output.get("module_rows", []):
        if row.get("status") == "error":
            missing.append(
                {
                    "module": row.get("module"),
                    "note": row.get("note"),
                    "action": "Coder should reproduce the failure, add/patch the data adapter, add a smoke test, then update astro_toolbox/README.md.",
                }
            )
    if data.get("status") == "planned":
        missing.append(
            {
                "module": "all survey adapters",
                "note": "dry-run mode",
                "action": "Run with --execute before deciding whether new data-fetch scripts are required.",
            }
        )
    doc = {
        "status": "written",
        "policy": "When a missing capability is confirmed, the Coder must add the script, validate it on the target, and update astro_toolbox documentation in the same run.",
        "candidate_updates": missing,
        "literature_method_learning": {
            "toolbox_gap": state.get("toolbox_gap", {}),
            "dynamic_skill_registration": state.get("dynamic_skill_registration", {}),
            "gate": "A learned method is registered only after Claude Code reports validation_passed=true or tests_passed=true.",
        },
    }
    path = tools.json_dump(_run_dir(state) / "toolbox_evolution_plan.json", doc)
    state["toolbox_evolution"] = {**doc, "path": path}
    _append_artifact(state, path)
    return state


def build_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("resolve", resolve_node)
    graph.add_node("data_fetcher", data_fetcher_node)
    graph.add_node("memory_advisor", memory_advisor_node)
    graph.add_node("structure_planner", structure_planner_node)
    graph.add_node("cluster_membership", cluster_membership_node)
    graph.add_node("extinction", extinction_node)
    graph.add_node("sed_decoupled", sed_decoupled_node)
    graph.add_node("light_curve_geometry", light_curve_geometry_node)
    graph.add_node("eclipse_mcmc", eclipse_mcmc_node)
    graph.add_node("physics_checks", physics_checks_node)
    graph.add_node("ads_live", ads_live_node)
    graph.add_node("novelty_detector", novelty_detector_node)
    graph.add_node("comparison_table", comparison_table_node)
    graph.add_node("rag_navigator", rag_navigator_node)
    graph.add_node("kg_navigator", kg_navigator_node)
    graph.add_node("method_scout", method_scout_node)
    graph.add_node("kg_graph_report", kg_graph_report_node)
    graph.add_node("source_research_package", source_research_package_node)
    graph.add_node("iteration_1_baseline", baseline_iteration_node)
    graph.add_node("iteration_2_residuals", residual_physics_iteration_node)
    graph.add_node("iteration_3_systematics", errors_systematics_iteration_node)
    graph.add_node("model_supervisor", model_supervisor_node)
    graph.add_node("claude_code_delegate", claude_code_delegate_node)
    graph.add_node("qa_gate", qa_gate_node)
    graph.add_node("abnormal_report", abnormal_report_node)
    graph.add_node("drafter", drafter_node)
    graph.add_node("figure_synthesizer", figure_synthesizer_node)
    graph.add_node("latex_compile", latex_compile_node)
    graph.add_node("paper_qc", paper_qc_node)
    graph.add_node("reflexion", reflexion_node)
    graph.add_node("peer_reviewer", peer_reviewer_node)
    graph.add_node("kg_writeback", kg_writeback_node)
    graph.add_node("toolbox_evolution", toolbox_evolution_node)

    graph.add_edge(START, "resolve")
    graph.add_edge("resolve", "data_fetcher")
    graph.add_edge("data_fetcher", "memory_advisor")
    graph.add_edge("memory_advisor", "structure_planner")
    graph.add_edge("structure_planner", "cluster_membership")
    graph.add_edge("cluster_membership", "extinction")
    graph.add_edge("extinction", "sed_decoupled")
    graph.add_edge("sed_decoupled", "light_curve_geometry")
    graph.add_edge("light_curve_geometry", "eclipse_mcmc")
    graph.add_edge("eclipse_mcmc", "physics_checks")
    graph.add_edge("physics_checks", "ads_live")
    graph.add_edge("ads_live", "novelty_detector")
    graph.add_edge("novelty_detector", "comparison_table")
    graph.add_edge("comparison_table", "rag_navigator")
    graph.add_edge("rag_navigator", "kg_navigator")
    graph.add_edge("kg_navigator", "method_scout")
    graph.add_edge("method_scout", "kg_graph_report")
    graph.add_edge("kg_graph_report", "source_research_package")
    graph.add_edge("source_research_package", "iteration_1_baseline")
    graph.add_edge("iteration_1_baseline", "iteration_2_residuals")
    graph.add_edge("iteration_2_residuals", "iteration_3_systematics")
    graph.add_edge("iteration_3_systematics", "model_supervisor")
    graph.add_edge("model_supervisor", "claude_code_delegate")
    graph.add_edge("claude_code_delegate", "qa_gate")
    graph.add_conditional_edges(
        "qa_gate",
        route_after_qa,
        {
            "abnormal": "abnormal_report",
            "paper": "drafter",
            # Self-heal: model mismatch routes back to structure_planner so a
            # different physics pipeline can be selected. Bounded by
            # state["structure_planner_retry_count"] (max 2).
            "replan": "structure_planner",
        },
    )
    graph.add_edge("abnormal_report", "kg_writeback")
    # drafter → figures → latex compile → paper_qc → reflexion
    graph.add_edge("drafter", "figure_synthesizer")
    graph.add_edge("figure_synthesizer", "latex_compile")
    graph.add_edge("latex_compile", "paper_qc")
    # Paper QC -> Reflexion -> [back to drafter for targeted rewrite] or peer_reviewer
    graph.add_edge("paper_qc", "reflexion")
    graph.add_conditional_edges(
        "reflexion",
        route_after_reflexion,
        {
            "rewrite": "drafter",
            "accept": "peer_reviewer",
        },
    )
    graph.add_edge("peer_reviewer", "kg_writeback")
    graph.add_edge("kg_writeback", "toolbox_evolution")
    graph.add_edge("toolbox_evolution", END)
    return graph.compile()


def run_workflow(
    target: str,
    output_root: str,
    ra_deg: float = None,
    dec_deg: float = None,
    dry_run: bool = True,
    force: bool = False,
    use_llm: bool = False,
    llm_provider: str = None,
    astrotool_run: str = None,
    kg_report: bool = False,
    kg_report_llm: bool = False,
    kg_report_provider: str = "deepseek",
    skip_simbad: bool = False,
    draft_on_hold: bool = False,
    method_scout_llm: bool = False,
    method_scout_provider: str = None,
    source_research_package: bool = False,
    download_simbad_pdfs: bool = False,
    enable_claude_code: bool = False,
    claude_timeout: int = 300,
    claude_permission_mode: str = "plan",
    max_supervision_rounds: int = 2,
    target_cluster: str = None,
    max_reflexion_retries: int = 2,
    gold_path: str = None,
) -> AnalysisState:
    app = build_graph()
    initial: AnalysisState = {
        "target": target,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "output_root": output_root,
        "dry_run": dry_run,
        "force": force,
        "use_llm": use_llm,
        "llm_provider": llm_provider,
        "astrotool_run": astrotool_run,
        "kg_report": kg_report,
        "kg_report_llm": kg_report_llm,
        "kg_report_provider": kg_report_provider,
        "skip_simbad": skip_simbad,
        "draft_on_hold": draft_on_hold,
        "method_scout_llm": method_scout_llm,
        "method_scout_provider": method_scout_provider,
        "source_research_package": source_research_package,
        "download_simbad_pdfs": download_simbad_pdfs,
        "enable_claude_code": enable_claude_code,
        "claude_timeout": claude_timeout,
        "claude_permission_mode": claude_permission_mode,
        "max_supervision_rounds": max_supervision_rounds,
        "target_cluster": target_cluster,
        "max_reflexion_retries": max_reflexion_retries,
        "gold_path": gold_path,
        "artifacts": [],
        "warnings": [],
        "errors": [],
        "iterations": [],
    }
    # Recursion-limit budget: 25 (langgraph default) is too low once Reflexion
    # adds a drafter→paper_qc→reflexion cycle. Each rewrite iteration costs
    # ~4 steps, so allow ~10 retries' worth of headroom.
    return app.invoke(initial, config={"recursion_limit": 60})
